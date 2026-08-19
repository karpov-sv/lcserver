"""ESO archive spectra acquisition module.

Not a light curve: this is everything ESO has ever published as a reduced
one-dimensional spectrum at a position - UVES, X-shooter, FEROS, HARPS,
GIRAFFE, ESPRESSO and the rest - found in one cone search of the archive's
table service and fetched from its data portal.

What it is for is bright stars. The surveys here reach faint and observe
everything in their way at a resolution of a few thousand; ESO's spectrographs
were pointed at particular objects, one at a time, and HARPS resolves a line
at a hundred and fifteen thousand. A star with a hundred HARPS spectra has a
record of its line profiles over a decade, which no survey offers.

Three quarters of what the archive holds is not flux calibrated: HARPS and
GIRAFFE publish detector counts, having been built to measure where a line
sits rather than how much light arrived. Those are still fetched, divided by
their own median so that the number means something, and written without a
flux unit rather than with one that would be untrue.
"""

import os
import io

import requests
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy import units as u

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    break_at_gaps, write_spectrum,
                    SPECTRUM_FLUX_UNIT, SPECTRUM_WAVELENGTH_UNIT)


ESO_TAP = 'https://archive.eso.org/tap_obs/sync'

# One file per call, named by the archive's own identifier for it
ESO_FILE = 'https://dataportal.eso.org/dataPortal/file/{}'

ESO_TIMEOUT = 300

# Matching radius, in arcsec. The archive records a spectrum at the position
# of the target it was pointed at, so this is how far the pointing may be from
# where the target really is rather than how large anything on the sky is.
ESO_SR = 5.0

# How many to fetch, in total and from any one instrument. The archive holds
# 148 HARPS spectra of tau Ceti alone, at five megabytes each, so a limit is
# not optional. Spread across instruments rather than taken in one block: a
# star with UVES, FEROS and HARPS spectra is better served by some of each
# than by the nine best of whichever happened to observe it most.
ESO_MAX_SPECTRA = 9
ESO_MAX_PER_INSTRUMENT = 3

# How many attempts at one file. The portal hangs up part way through a large
# transfer often enough to be worth a second try and rarely enough that a
# third would only be waiting.
ESO_ATTEMPTS = 2

# The largest a spectrum is kept at. A HARPS spectrum arrives with 313131
# points, which is ten megabytes of text apiece, a hundred megabytes of cache
# for one target, and - since the viewer draws what the file holds - a couple
# of million points for a browser to plot.
#
# So anything longer is binned down to this by averaging whole blocks of
# neighbouring pixels, rather than by dropping every second one: averaging
# keeps the light, and the noise falls as it should. What is left is still
# three times the longest spectrum any other source here produces, and around
# R = 40000 at 5000 A, which resolves any stellar line worth looking at. The
# binning is reported wherever it happens, and it is done before the cache so
# that what is stored is what is used.
ESO_MAX_POINTS = 24000

# What to ask the archive for
ESO_COLUMNS = [
    'dp_id', 'obs_collection', 'instrument_name', 'facility_name',
    'em_min', 'em_max', 'em_res_power', 'snr', 't_min', 't_exptime',
    'o_calib_status', 'obs_title', 'access_estsize',
]

# The columns of a Phase 3 one-dimensional spectrum, which the ESO science
# data product standard fixes. The flux may or may not carry a physical unit;
# the wavelength always does, but not always the same one - X-shooter reports
# nanometres where HARPS reports Angstrom - so it is read rather than assumed.
ESO_WAVELENGTH = 'WAVE'

# The flux column, in the order to look for it. A product with no flux scale
# carries its counts under FLUX_REDUCED instead of FLUX - that is how the
# standard says to publish one - and a calibrated product carries both, the
# reduced counts alongside the calibrated flux. So FLUX is preferred where it
# is there, and GIRAFFE, which is the largest single holding in the archive
# and calibrates none of it, is reached through the second name.
ESO_FLUX_COLUMNS = [('FLUX', 'ERR'), ('FLUX_REDUCED', 'ERR_REDUCED')]


def _query(ra, dec, sr, released_before):
    """Every published spectrum at a position, as the archive lists them.

    Only what has come out of its proprietary period: the table lists
    embargoed products too, and the portal answers a request for one with a
    refusal rather than a file.
    """
    query = (f"SELECT {', '.join(ESO_COLUMNS)} FROM ivoa.ObsCore"
             " WHERE dataproduct_type='spectrum'"
             f" AND obs_release_date < '{released_before}'"
             " AND CONTAINS(POINT('ICRS', s_ra, s_dec),"
             f" CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {sr / 3600.0:.9f}))=1")

    try:
        res = requests.post(ESO_TAP, timeout=ESO_TIMEOUT, data={
            'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'votable',
            'QUERY': query})
        res.raise_for_status()
    except requests.RequestException as e:
        raise SourceError("could not query the ESO archive - "
                          f"{type(e).__name__}: {e}")

    if b'QUERY_STATUS" value="ERROR"' in res.content:
        raise SourceError("the ESO archive refused the query: "
                          + res.text[:300])

    try:
        table = Table.read(io.BytesIO(res.content), format='votable')
    except Exception as e:
        raise SourceError("could not read the ESO answer - "
                          f"{type(e).__name__}: {e}")

    return table if len(table) else None


def _number(value):
    """A field of a row as a float, or None where it is masked or absent."""
    if value is np.ma.masked:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if np.isfinite(value) else None


def _convert(values, unit, target):
    """A column in the unit everything here is written in, or None.

    None where the file declares a unit that cannot become the target one -
    counts into erg/s/cm2/A, which is the uncalibrated case and not an error.
    """
    if values is None:
        return None

    values = np.asarray(values, dtype=float)

    if not unit:
        return None

    try:
        return (values * u.Unit(unit)).to_value(u.Unit(target))
    except Exception:
        return None


def _bin(table, maxpoints=ESO_MAX_POINTS):
    """A spectrum binned down to a length worth keeping, and by how much.

    Whole blocks of neighbouring pixels are averaged - the wavelengths with
    them, so the grid stays the grid of what was measured - and the errors
    are combined as independent, which they are between pixels of an extracted
    spectrum. A trailing part-block is dropped rather than averaged over fewer
    points than the rest, being at most a few pixels at one end.
    """
    n = len(table)

    if n <= maxpoints:
        return table, 1

    factor = int(np.ceil(n / maxpoints))
    kept = (n // factor) * factor

    def blocks(values):
        return np.asarray(values, dtype=float)[:kept].reshape(-1, factor)

    binned = Table({
        'wavelength': np.mean(blocks(table['wavelength']), axis=1),
        'flux': np.mean(blocks(table['flux']), axis=1),
    })

    if 'flux_error' in table.colnames:
        # Independent between pixels, so they add in quadrature and the mean
        # of the block is the more precise for it
        binned['flux_error'] = (np.sqrt(np.sum(blocks(table['flux_error']) ** 2,
                                               axis=1)) / factor)

    binned['flux'].unit = table['flux'].unit

    return binned, factor


def _fetch(dp_id):
    """One spectrum, as wavelength in Angstrom and flux as the file has it.

    Returns the table together with whether the flux is on a physical scale.
    The archive says which it should be, in o_calib_status, but the file's own
    unit is what is acted on: it is the thing that would be wrong.
    """
    for attempt in range(ESO_ATTEMPTS):
        try:
            res = requests.get(ESO_FILE.format(requests.utils.quote(dp_id, safe='')),
                               timeout=ESO_TIMEOUT)
            res.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt + 1 == ESO_ATTEMPTS:
                raise SourceError(f"could not fetch {dp_id} - "
                                  f"{type(e).__name__}: {e}")

    with fits.open(io.BytesIO(res.content)) as hdus:
        if len(hdus) < 2 or not hasattr(hdus[1], 'columns'):
            return None, False, 1

        columns = {c.name.upper(): c for c in hdus[1].columns}

        flux_name, error_name = next(
            ((f, e) for f, e in ESO_FLUX_COLUMNS if f in columns), (None, None))

        if ESO_WAVELENGTH not in columns or flux_name is None:
            return None, False, 1

        data = hdus[1].data

        # Each column holds the whole array in a single row
        def column(name):
            return (np.atleast_1d(np.asarray(data[name], dtype=float).ravel())
                    if name in columns else None)

        wavelength = _convert(column(ESO_WAVELENGTH),
                              columns[ESO_WAVELENGTH].unit,
                              SPECTRUM_WAVELENGTH_UNIT)

        if wavelength is None:
            return None, False, 1

        flux = column(flux_name)
        error = column(error_name)

        physical = _convert(flux, columns[flux_name].unit, SPECTRUM_FLUX_UNIT)

        if physical is not None:
            flux = physical
            error = _convert(error, columns[error_name].unit
                             if error_name in columns else None,
                             SPECTRUM_FLUX_UNIT)

        table = Table({'wavelength': wavelength, 'flux': flux})

        if error is not None and len(error) == len(flux):
            table['flux_error'] = error

        # Whether the flux is on a physical scale is carried as the unit of
        # the column itself, rather than beside the table: a VOTable keeps a
        # unit through a round trip and does not keep arbitrary metadata, and
        # the cache is read back on every run after the first
        table['flux'].unit = SPECTRUM_FLUX_UNIT if physical is not None else None

        # The pixels between a spectrograph's two detectors, and beyond the
        # ends of its orders, are filled with exact zeros rather than left
        # out. A real measurement is never exactly zero - not a sky-subtracted
        # one either, which lands a little to one side of it - so these are
        # absences, and dropping them both keeps them out of the median and
        # lets the plot break the line where nothing was observed instead of
        # drawing a notch to the axis.
        keep = np.isfinite(table['wavelength']) & (np.asarray(table['flux']) != 0)

        table = table[keep]

        table, factor = _bin(table)

    return table, physical is not None, factor


def _chosen(found, log):
    """Which of the archive's spectra to fetch, and in what order.

    The best signal to noise from each instrument first, so that a cut at the
    limit leaves a spread of instruments rather than the whole of one.
    """
    snr = np.asarray([_ if np.isfinite(_) else -1.0
                      for _ in np.asarray(found['snr'], dtype=float)])

    order = np.argsort(-snr, kind='stable')

    taken, seen = [], {}

    for i in order:
        instrument = str(found['instrument_name'][i])

        if seen.get(instrument, 0) >= ESO_MAX_PER_INSTRUMENT:
            continue

        seen[instrument] = seen.get(instrument, 0) + 1
        taken.append(i)

        if len(taken) >= ESO_MAX_SPECTRA:
            break

    if len(taken) < len(found):
        kept = ', '.join(f"{n} of {name}" for name, n in sorted(seen.items()))
        log(f"\n{len(found)} spectra, of which {len(taken)} are fetched"
            f" ({kept}) - the best signal to noise of each instrument")
    else:
        log(f"\nFetching {len(taken)} spectr{'um' if len(taken) == 1 else 'a'}")

    return [found[int(i)] for i in taken]


@survey_source(
    name='ESO archive',
    short_name='ESO',
    state_acquiring='acquiring ESO spectra',
    state_acquired='ESO spectra acquired',
    log_file='eso.log',
    output_files=['eso.log', 'eso_*.png', 'eso_*.vot', 'eso_*.txt'],
    button_text='Get ESO spectra',
    form_fields={
        'eso_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': ESO_SR,
            'required': False,
        },
    },
    help_text='Reduced spectra from the ESO archive - UVES, X-shooter, FEROS, '
              'HARPS, GIRAFFE and the rest',
    order=85,
    spectrum_files='eso_*.txt',
    spectrum_palette=['#1a5276', '#1f618d', '#2471a3', '#2e86c1', '#5499c7',
                      '#7fb3d5', '#a9cce3'],
    template_layout='complex',
    additional_plots=['eso_*.png'],
)
def target_eso(config, basepath=None, verbose=True, show=False):
    """Acquire spectra from the ESO archive."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('eso'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = float(config.get('eso_sr', ESO_SR))

    log(f"within {sr:.1f} arcsec")

    cache_name = f"eso_{ra:.4f}_{dec:.4f}_{sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'ESO archive',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            # Today rather than a fixed date, so that a target queried again
            # next year picks up what came out of embargo in between
            found = _query(ra, dec, sr, Time.now().isot)

            if found is None:
                cache.save_empty()
                log("\nWarning: No published ESO spectra at this position - "
                    "its telescopes were pointed at particular objects, and "
                    "cover no more of the sky than they were asked to")
                return

            cache.save(found)

        found = cache.data

    # Nothing here, and cached as nothing - the helper has said so already
    if found is None:
        return

    log(f"\n{len(found)} published ESO spectr{'um' if len(found) == 1 else 'a'}")

    log("\n---- What the archive holds ----\n")

    for name in sorted(set(str(_) for _ in found['instrument_name'])):
        rows = found[np.asarray(found['instrument_name'], dtype=str) == name]

        times = np.asarray(rows['t_min'], dtype=float)
        times = times[np.isfinite(times)]

        power = np.asarray(rows['em_res_power'], dtype=float)
        power = power[np.isfinite(power)]

        # Metres in the archive, nanometres to read
        low = np.nanmin(np.asarray(rows['em_min'], dtype=float)) * 1e9
        high = np.nanmax(np.asarray(rows['em_max'], dtype=float)) * 1e9

        log(f"{name:<10s} {len(rows):4d} spectra  {low:.0f}-{high:.0f} nm"
            + (f"  R ~ {np.nanmedian(power):.0f}" if len(power) else '')
            + (f"  {Time(times.min(), format='mjd').iso[:10]}"
               f" to {Time(times.max(), format='mjd').iso[:10]}"
               if len(times) else '')
            + f"  [{str(rows['o_calib_status'][0])}]")

    for row in _chosen(found, log):
        dp_id = str(row['dp_id'])
        instrument = str(row['instrument_name'])

        # The identifier is a timestamp with colons and dots in it, which do
        # not belong in a filename
        stem = 'eso_' + instrument + '_' + dp_id.replace(':', '').replace('.', '_')

        with cached_votable_query(stem + '.vot', basepath, log,
                                  f'ESO spectrum {dp_id}',
                                  refresh=refresh_cache) as cache:
            if not cache.hit:
                try:
                    spectrum, calibrated, factor = _fetch(dp_id)

                    if factor > 1:
                        log(f"  {dp_id}: binned by {factor} to"
                            f" {len(spectrum)} points")

                    # Inside the try, so that a fetch which failed is not
                    # remembered as a spectrum that does not exist
                    if spectrum is not None and len(spectrum):
                        cache.save(spectrum)
                    else:
                        cache.save_empty()
                        spectrum = None
                except SourceError as e:
                    log(f"  {dp_id}: {e}")
                    spectrum = None
                except Exception as e:
                    log(f"  {dp_id}: {type(e).__name__}: {e}")
                    spectrum = None
            else:
                spectrum = cache.data

        if spectrum is None or not len(spectrum):
            continue

        # Set where the file gave a flux that could become erg/s/cm2/A, and
        # left off where it gave counts - see _fetch
        calibrated = spectrum['flux'].unit is not None

        wavelength = np.asarray(spectrum['wavelength'], dtype=float)
        flux = np.asarray(spectrum['flux'], dtype=float)
        error = (np.asarray(spectrum['flux_error'], dtype=float)
                 if 'flux_error' in spectrum.colnames else None)

        good = np.isfinite(wavelength) & np.isfinite(flux)

        if not np.any(good):
            log(f"  {dp_id}: nothing measurable in it")
            continue

        wavelength, flux = wavelength[good], flux[good]
        error = error[good] if error is not None else None

        # An uncalibrated spectrum is divided by its own median, so that the
        # number it carries means one definite thing - the continuum is one -
        # rather than being whatever the detector counted that night
        if not calibrated:
            median = float(np.median(flux[np.isfinite(flux)]))

            if median and np.isfinite(median):
                flux = flux / median
                error = error / median if error is not None else None

        snr = _number(row['snr'])

        log(f"  {dp_id} ({instrument}): {len(wavelength)} points from"
            f" {wavelength.min():.0f} to {wavelength.max():.0f} A"
            + (f", S/N = {snr:.0f}" if snr is not None else '')
            + ('' if calibrated else ', flux uncalibrated and put on its median'))

        with plots.figure_saver(os.path.join(basepath, stem + '.png'),
                                figsize=(10, 4), show=show) as fig:
            ax = fig.add_subplot(1, 1, 1)

            # An echelle spectrum merged onto one grid can leave gaps between
            # its orders, which are not to be drawn across
            ax.plot(*break_at_gaps(wavelength, flux), '-', lw=0.5,
                    color='#1f618d')

            ax.grid(alpha=0.2)
            ax.set_xlabel('Wavelength, A')
            ax.set_ylabel(r'Flux, erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$'
                          if calibrated else 'Flux, relative to the median')

            when = _number(row['t_min'])
            ax.set_title(f"{config['target_name']} - ESO {instrument}"
                         + (f", {Time(when, format='mjd').iso[:10]}"
                            if when is not None else ''))

        table = Table({'wavelength': wavelength, 'flux': flux})

        if error is not None:
            table['flux_error'] = error

        write_spectrum(table, basepath, stem, calibrated=calibrated)

        log(f"    Spectrum plotted in file:{stem}.png")
        log(f"    Spectrum written to file:{stem}.vot")
        log(f"    Spectrum written to file:{stem}.txt")
