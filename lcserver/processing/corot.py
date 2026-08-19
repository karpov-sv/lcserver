"""CoRoT lightcurve acquisition module.

CoRoT watched two small fields near the Galactic plane, one towards the centre
and one away from it, from 2007 to 2012, and it did so from space: half-minute
sampling for months on end, at a precision no ground survey reaches. It sits
between Kepler, which stared at one field far from the plane, and TESS, which
sweeps the whole sky in a month at a time - and its fields are in the plane,
where the disk surveys here work.

It observed in two modes at once. The bright stars of the asteroseismology
field give one flux; the fainter exoplanet field targets were dispersed by a
prism, and the brighter of those have their light split into red, green and
blue as well as summed into white. The white flux is what is published here,
being the one every target has.

The observation log comes from Vizier, which carries it with a path to each
light curve, and the files themselves from the CDS archive.
"""

import os
import io

import requests
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

from astroquery.vizier import Vizier

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    break_at_gaps, quality_field, quality_level,
                    QUALITY_STANDARD, QUALITY_PUBLISHED)


# The observation log, one row per target per run it was observed in
COROT_CATALOGUE = 'B/corot'

# Where the light curves themselves live, under the path the log gives
COROT_FILES = 'https://cdsarc.cds.unistra.fr/ftp/B/corot/files/{}'

COROT_TIMEOUT = 300

# Matching radius, in arcsec. The log carries the catalogue position of the
# target rather than anything about the instrument, whose photometric mask is
# a good deal larger than this.
COROT_SR = 5.0

# A target in both fields and several runs is a file and a plot per run
COROT_MAX_RUNS = 6

# The most points to keep in one run. A bright-star target sampled every
# second for five months arrives with millions, which is more than a browser
# will draw and more than the shape of the curve needs; anything longer is
# binned down by averaging whole blocks, which keeps the light and lowers the
# noise as it should. Reported wherever it happens.
COROT_MAX_POINTS = 30000

# Where to look for the light curve inside the file, in order of preference.
# BAR is corrected to the solar system barycentre, which RAW is not; BARREG is
# the same resampled onto a regular grid, which fills gaps with interpolation
# and so is the second choice rather than the first.
COROT_HDUS = ['BAR', 'BARREG', 'RAW']

# The flux to publish, in order of preference. WHITEFLUX is the summed light
# of the prism-dispersed exoplanet targets and is what every one of them has;
# the asteroseismology field, which is not dispersed, names its column for the
# processing stage instead.
COROT_FLUX_COLUMNS = ['WHITEFLUX', 'FLUXBAR', 'FLUXBARREG', 'RAWFLUX', 'FLUX']

# What to ask Vizier for
COROT_COLUMNS = ['CoRoT', 'Run', 'CCD', 'RAJ2000', 'DEJ2000', 'Vmag', 'SpT',
                 'Teff', 'logg', '[Fe/H]', 'date1', 'date2', 'LCmean',
                 'LCrms', 'FileName']


def _number(row, key):
    """One field of a row as a float, or None where it is absent or masked.

    Vizier returns only the columns that hold something for the rows asked
    for, so an exoplanet-field target - which has no spectral type and no
    temperature - comes back with those columns missing outright rather than
    empty.
    """
    if key not in row.colnames or row[key] is np.ma.masked:
        return None

    try:
        value = float(row[key])
    except (TypeError, ValueError):
        return None

    return value if np.isfinite(value) else None


def _text(row, key):
    """One field of a row as a word, or None where it says nothing."""
    if key not in row.colnames or row[key] is np.ma.masked:
        return None

    return str(row[key]).strip() or None


def _pick(names, wanted):
    """The first of `wanted` that the file actually has."""
    upper = {n.upper(): n for n in names}

    return next((upper[w] for w in wanted if w in upper), None)


def _cadence(mjd):
    """The spacing of a light curve, in whole seconds and at least one."""
    steps = np.diff(np.asarray(mjd, dtype=float))
    steps = steps[steps > 0]

    if not len(steps):
        return 1

    return max(int(round(float(np.median(steps)) * 86400.0)), 1)


def _bin(table, maxpoints=COROT_MAX_POINTS):
    """A light curve binned down to a length worth keeping, and by how much.

    Whole blocks of consecutive cadences are averaged, times and all. The
    quality flags of a block are combined by OR, so that a block holding one
    bad cadence is marked as such rather than having it averaged away.
    """
    n = len(table)

    if n <= maxpoints:
        return table, 1

    factor = int(np.ceil(n / maxpoints))
    kept = (n // factor) * factor

    def blocks(name):
        return np.asarray(table[name], dtype=float)[:kept].reshape(-1, factor)

    binned = Table({
        'mjd': np.mean(blocks('mjd'), axis=1),
        'flux': np.mean(blocks('flux'), axis=1),
    })

    if 'flux_err' in table.colnames:
        binned['flux_err'] = (np.sqrt(np.sum(blocks('flux_err') ** 2, axis=1))
                              / factor)

    if 'quality' in table.colnames:
        flags = np.asarray(table['quality'], dtype=np.int64)[:kept]
        binned['quality'] = np.bitwise_or.reduce(
            flags.reshape(-1, factor), axis=1)

    return binned, factor


def _download(filename):
    """One run's light curve, as the archive stores it.

    Returned with the columns everything in flux mode here uses: an MJD, a
    flux divided by its own median, its error on the same scale, and the
    pipeline's status flags.
    """
    try:
        res = requests.get(COROT_FILES.format(filename), timeout=COROT_TIMEOUT)
        res.raise_for_status()
    except requests.RequestException as e:
        raise SourceError(f"could not fetch {os.path.basename(filename)} - "
                          f"{type(e).__name__}: {e}")

    with fits.open(io.BytesIO(res.content)) as hdus:
        available = {h.name.upper(): h for h in hdus
                     if getattr(h, 'columns', None) is not None}

        chosen = next((available[n] for n in COROT_HDUS if n in available), None)

        if chosen is None:
            return None

        names = [c.name for c in chosen.columns]
        data = chosen.data

        # The time is whichever DATE column holds numbers - the exoplanet
        # files carry a written-out date alongside it, as a string
        time_name = next((n for n in names
                          if n.upper().startswith('DATE')
                          and np.issubdtype(np.asarray(data[n]).dtype, np.number)),
                         None)

        flux_name = _pick(names, COROT_FLUX_COLUMNS)

        if time_name is None or flux_name is None:
            return None

        error_name = _pick(names, [flux_name.upper().replace('FLUX', 'FLUXDEV'),
                                   flux_name.upper() + 'DEV',
                                   'FLUXDEVBAR', 'FLUXDEVBARREG', 'RAWFLUXDEV'])
        status_name = _pick(names, ['STATUS', 'STATUSBAR', 'STATUSBARREG',
                                    'RAWSTATUS'])

        # The files date their cadences in MJD, whatever the column's own unit
        # card says - the values check out against the run dates in the log
        mjd = np.asarray(data[time_name], dtype=float)
        flux = np.asarray(data[flux_name], dtype=float)

        table = Table({'mjd': mjd, 'flux': flux})

        if error_name:
            # A deviation of -1 is the pipeline saying it has none
            error = np.asarray(data[error_name], dtype=float)
            table['flux_err'] = np.where(error >= 0, error, np.nan)

        if status_name:
            table['quality'] = np.asarray(data[status_name], dtype=np.int64)

        # An instrumental flux in electrons per second means nothing beside
        # another star's, so it is put on its own median, as every other flux
        # source here is
        good = np.isfinite(table['mjd']) & np.isfinite(table['flux'])
        table = table[good]

        if not len(table):
            return None

        table.sort('mjd')

        median = float(np.median(table['flux']))

        if median and np.isfinite(median):
            table['flux'] = table['flux'] / median

            if 'flux_err' in table.colnames:
                table['flux_err'] = table['flux_err'] / median

    return table


@survey_source(
    name='CoRoT',
    short_name='CoRoT',
    state_acquiring='acquiring CoRoT lightcurves',
    state_acquired='CoRoT lightcurves acquired',
    log_file='corot.log',
    output_files=['corot.log', 'corot_lc_*.vot', 'corot_lc_*.txt',
                  'corot_lc_*.png'],
    button_text='Get CoRoT lightcurves',
    form_fields={
        'corot_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': COROT_SR,
            'required': False,
        },
        'corot_quality': quality_field({
            QUALITY_STANDARD: 'Drop the cadences the pipeline flagged',
            QUALITY_PUBLISHED: 'None - every cadence as published',
        }),
    },
    help_text='CoRoT, two fields on the Galactic plane from space, 2007-2012',
    order=32,
    # Lightcurve metadata
    votable_file='corot_lc_*.vot',
    lc_flux_column='flux',
    lc_err_column='flux_err',
    lc_quality_column='quality',
    lc_color='#8e44ad',
    lc_mode='flux',
    lc_short=False,
    # The mission observed in runs - initial, long and short, towards the
    # Galactic centre or away from it - each of which is one continuous stare
    lc_segment_name='Run',
    lc_color_palette=['#8e44ad', '#9b59b6', '#6c3483', '#a569bd', '#c39bd3'],
    # Template metadata. No cutout: its photometric mask is tens of arcseconds
    # across and nothing else resolves what went into it.
    template_layout='complex',
    additional_plots=['corot_lc_*.png'],
)
def target_corot(config, basepath=None, verbose=True, show=False):
    """Acquire CoRoT lightcurves."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('corot'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = float(config.get('corot_sr', COROT_SR))

    log(f"within {sr:.1f} arcsec")

    cache_name = f"corot_{ra:.4f}_{dec:.4f}_{sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'CoRoT observations',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            try:
                res = Vizier(columns=COROT_COLUMNS, row_limit=-1).query_region(
                    SkyCoord(ra, dec, unit='deg'), radius=sr * u.arcsec,
                    catalog=COROT_CATALOGUE)
            except Exception as e:
                raise SourceError("could not query Vizier for CoRoT - "
                                  f"{type(e).__name__}: {e}")

            found = res[0] if res and len(res) else None

            if found is None or not len(found):
                cache.save_empty()
                log("\nWarning: No CoRoT observations at this position - the "
                    "mission watched two small fields near the Galactic plane "
                    "and nothing else")
                return

            cache.save(found)

        found = cache.data

    # Nothing here, and cached as nothing - the helper has said so already
    if found is None:
        return

    log(f"\n{len(found)} CoRoT observation(s)")

    log("\n---- What CoRoT observed ----\n")

    for row in found:
        vmag, teff = _number(row, 'Vmag'), _number(row, 'Teff')

        log(f"CoRoT {_text(row, 'CoRoT')}  run {_text(row, 'Run')}"
            f"  CCD {_text(row, 'CCD')}"
            + (f"  V = {vmag:.2f}" if vmag is not None else '')
            + (f"  {_text(row, 'SpT')}" if _text(row, 'SpT') else ''))

        log(f"    {_text(row, 'date1')} to {_text(row, 'date2')}"
            + (f", Teff = {teff:.0f} K" if teff else ''))

    # Oldest first, so that the numbering the viewer sorts on runs with time
    order = (np.argsort(np.asarray([str(_) for _ in found['date1']]), kind='stable')
             if 'date1' in found.colnames else np.arange(len(found)))
    wanted = [found[int(i)] for i in order]

    if len(wanted) > COROT_MAX_RUNS:
        log(f"\n{len(wanted)} runs, of which the first {COROT_MAX_RUNS} are fetched")
        wanted = wanted[:COROT_MAX_RUNS]
    else:
        log(f"\nFetching {len(wanted)} run(s)")

    for index, row in enumerate(wanted, start=1):
        filename = _text(row, 'FileName')
        run = _text(row, 'Run') or 'run'

        if not filename:
            continue

        stem = f"corot_lc_{index:02d}_{run}"

        with cached_votable_query(stem + '_raw.vot', basepath, log,
                                  f'CoRoT {run} lightcurve',
                                  refresh=refresh_cache) as cache:
            if not cache.hit:
                try:
                    lc = _download(filename)

                    # Inside the try, so that a fetch which failed is not
                    # remembered as a run that holds nothing
                    if lc is not None and len(lc):
                        cache.save(lc)
                    else:
                        cache.save_empty()
                        lc = None
                except SourceError as e:
                    log(f"  {run}: {e}")
                    lc = None
                except Exception as e:
                    log(f"  {run}: {type(e).__name__}: {e}")
                    lc = None
            else:
                lc = cache.data

        if lc is None or not len(lc):
            continue

        # The pipeline marks the cadences it could not measure properly -
        # the South Atlantic Anomaly, an Earth eclipse, a cosmic ray - and
        # leaves them in the file with their flux as measured. A third of an
        # exoplanet run is flagged, and unflagged those cadences swamp the
        # star: the run below spans 1.5 percent in flux with them dropped and
        # a factor of 1.5 with them kept. So they go by default.
        #
        # Only the pipeline's own verdict is acted on, not the individual
        # bits: their meanings are not in the description CDS holds, and a
        # mask invented here would be a guess dressed as a filter. What the
        # flags said is reported instead.
        if 'quality' in lc.colnames:
            flags = np.asarray(lc['quality'], dtype=np.int64)
            good = flags == 0

            log(f"  {run}: {int(np.sum(~good))} of {len(lc)} cadences carry a"
                " status flag"
                + (f" ({', '.join(str(int(_)) for _ in sorted(set(flags[~good].tolist()))[:8])}"
                   + (', ...' if len(set(flags[~good].tolist())) > 8 else '') + ')'
                   if np.any(~good) else ''))

            if quality_level(config, 'corot') != QUALITY_PUBLISHED and np.any(~good):
                lc = lc[good]
                log(f"    {len(lc)} cadences left")

                if not len(lc):
                    log("    nothing left at this level of filtering")
                    continue

        lc, factor = _bin(lc)

        if factor > 1:
            log(f"  {run}: binned by {factor} to {len(lc)} points")

        mjd = np.asarray(lc['mjd'], dtype=float)

        # Read off the times rather than carried through the cache: the file
        # does not state its cadence, and after binning what matters is the
        # spacing of what was written rather than of what was downloaded
        cadence = _cadence(mjd)

        # What every flux source here writes, and what the viewer parses the
        # run and the cadence back out of
        name = f"{stem}_{cadence:d}"
        flux = np.asarray(lc['flux'], dtype=float)

        tmin, tmax = Time(mjd.min(), format='mjd'), Time(mjd.max(), format='mjd')

        log(f"  {run}: {len(lc)} points, {cadence} s cadence,"
            f" {tmin.datetime.strftime('%Y-%m-%d')}"
            f" to {tmax.datetime.strftime('%Y-%m-%d')}")

        with plots.figure_saver(os.path.join(basepath, name + '.png'),
                                figsize=(8, 4), show=show) as fig:
            ax = fig.add_subplot(1, 1, 1)

            ax.axhline(1, ls='--', color='gray', alpha=0.3)
            ax.plot(*break_at_gaps(mjd, flux), drawstyle='steps', lw=1,
                    color='#8e44ad')

            ax.grid(alpha=0.2)
            ax.set_ylabel('Normalized white flux')
            ax.set_xlabel('MJD')
            ax.set_title(f"{config['target_name']} - CoRoT {run}"
                         f" - {cadence} s")

        written = lc.copy()
        written['mjd'].unit = 'd'

        written.write(os.path.join(basepath, name + '.vot'),
                      format='votable', overwrite=True)
        written.write(os.path.join(basepath, name + '.txt'),
                      format='ascii.commented_header', overwrite=True)

        log(f"    Run lightcurve plotted in file:{name}.png")
        log(f"    Run lightcurve written to file:{name}.vot")
        log(f"    Run lightcurve written to file:{name}.txt")
