"""NGTS lightcurve acquisition module.

The Next Generation Transit Survey is twelve 20cm telescopes at Paranal, each
following one 2.8 degree field through the night at a 13 second cadence,
through one broad filter from 520 to 890nm. Its second data release holds 72
fields observed between 2015 and 2018 - some 560 square degrees, between -65
and +8 degrees of declination - and every star in them down to I = 16.

ESO publishes the light curves as files of 2 to 6 gigabytes, one per fifth of
a field on a side, which is no way to get one star. Its catalogue service has
the same data as a table indexed by source, though, and that is what is read
here: the source catalogue by position, then the light curve by identifier.

A star has 64 000 to 250 000 measurements, which is more than any viewer here
can draw. The full cadence is kept in the cache, and what is published is the
same light curve in bins of a few minutes - short enough still to resolve a
transit or an eclipse.
"""

import io
import os
import re
import warnings

import requests
import numpy as np

from astropy.table import Table, vstack
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io.votable.exceptions import VOWarning

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    log_bands, log_conversion, plot_with_errors,
                    quality_field, quality_level,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


NGTS_TAP = 'https://archive.eso.org/tap_cat/sync'
NGTS_SOURCES = 'NGTS_SOURCE_CAT_V2'
NGTS_LIGHTCURVES = 'NGTS_LC_V2'

NGTS_TIMEOUT = 600

# The service answers with this many rows at most unless asked for more. A
# light curve is a quarter of a million at the longest.
NGTS_MAXREC = 1000000

# Matching radius, in arcsec. An NGTS pixel is five arcseconds and its
# photometric aperture three pixels across, so two catalogue entries this close
# are one star - the release removed any source detected twice within the
# survey's resolution - and several within it are the same star catalogued in
# overlapping fields.
NGTS_SR = 5.0

# No more than this many catalogue entries are fetched. More than two would be
# a star in three overlapping fields, which DR2 does not have.
NGTS_MAX_SOURCES = 3

# The bin width of the published light curve, in minutes
NGTS_BIN = 10.0

# The fewest measurements a bin may be made of. At a 13 second cadence a ten
# minute bin holds forty-six; one holding two is the edge of a night.
NGTS_BIN_MIN = 3

# The fewest simultaneous bins two catalogue entries of one star need before
# one is put on the scale of the other - see _align
NGTS_ALIGN_MIN = 20

# The pipeline's flags, from Table 3 of the DR2 release description
NGTS_FLAGS = [
    (0x01, 'SATURATION', 'saturated in the aperture'),
    (0x02, 'COSMIC', 'cosmic ray in the aperture'),
    (0x04, 'CROSSING', 'a laser, aircraft or satellite crossed the aperture'),
    (0x08, 'OUTLIER', 'a 7-sigma outlier from the light curve'),
    (0x10, 'SPIKE', 'a blooming spike from a bright neighbour'),
]

# What each level drops. The standard is what the survey itself calls a good
# point; the relaxed level drops only saturation, where the star is not being
# measured at all - a cosmic ray, a crossing or a spike spoils a frame here and
# there, where saturation spoils every frame it is on the same way.
#
# The survey's 7-sigma outliers are not among the choices. They are published
# with neither a flux nor an error, so no level can have them back - which
# means a flare or eclipse deep enough to trip the clip is missing from the
# release itself, not from what this code chose to keep.
NGTS_BAD_FLAGS = {
    QUALITY_STANDARD: 0x1f,
    QUALITY_RELAXED: 0x01,
    QUALITY_PUBLISHED: 0,
}

# The survey times its measurements in HJD (UTC)
NGTS_HJD_TO_MJD = 2400000.5

NGTS_COLOR = '#e6550d'


def _tap(query, what):
    """One synchronous query of ESO's catalogue service, as a table."""
    try:
        res = requests.post(NGTS_TAP, timeout=NGTS_TIMEOUT, data={
            'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'votable',
            'MAXREC': NGTS_MAXREC, 'QUERY': query})
    except requests.RequestException as e:
        raise SourceError(f"could not query the ESO catalogue service for "
                          f"{what} - {type(e).__name__}: {e}")

    # A refused query comes back as a 400 with the reason inside a VOTable,
    # which says a good deal more than the status does, so it is looked for
    # first
    reason = re.search(rb'QUERY_STATUS" value="ERROR">(.*?)</INFO>',
                       res.content, re.S)

    if reason:
        raise SourceError("the ESO catalogue service refused the query: "
                          + ' '.join(reason.group(1).decode(errors='replace')
                                     .split())[:300])

    try:
        res.raise_for_status()
    except requests.RequestException as e:
        raise SourceError(f"could not query the ESO catalogue service for "
                          f"{what} - {type(e).__name__}: {e}")

    # Cut at the row limit rather than finished - a light curve with its end
    # missing, which would pass for a whole one
    if b'value="OVERFLOW"' in res.content:
        raise SourceError(f"the ESO catalogue service returned {what} cut "
                          f"short at {NGTS_MAXREC} rows")

    try:
        # The service writes its fluxes in adu/s, which astropy does not know
        # as a unit and says so for every column that has it
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', VOWarning)
            table = Table.read(io.BytesIO(res.content), format='votable')
    except Exception as e:
        raise SourceError(f"could not read {what} from the ESO catalogue "
                          f"service - {type(e).__name__}: {e}")

    # ... and would say it again on every write of the table to the cache and
    # every read back, so the unit it cannot read is dropped here
    for column in table.itercols():
        if isinstance(column.unit, u.UnrecognizedUnit):
            column.unit = None

    return table


def _sources(ra, dec, sr):
    """The catalogue entries within sr arcsec, nearest first."""
    # Ordered here rather than by the service, which has no DISTANCE
    query = (
        "SELECT SOURCE_ID, RA_NGTS, DEC_NGTS, NGTS_MAG, FLUX_MEAN, NPTS,"
        " NPTS_TOTAL, GAIA_DR2_ID"
        f" FROM {NGTS_SOURCES}"
        " WHERE CONTAINS(POINT('ICRS', RA_NGTS, DEC_NGTS),"
        f" CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {sr / 3600.0:.9f}))=1")

    table = _tap(query, 'the source catalogue')

    if not len(table):
        return None

    # Names as the rest of this reads them, whatever case the service used
    for name in table.colnames:
        table.rename_column(name, name.lower())

    table['dist'] = SkyCoord(ra, dec, unit='deg').separation(
        SkyCoord(np.asarray(table['ra_ngts'], dtype=float),
                 np.asarray(table['dec_ngts'], dtype=float), unit='deg')).arcsec
    table.sort('dist')

    return table


def _lightcurve(source_id):
    """Every measurement of one source, at full cadence."""
    table = _tap(f"SELECT HJD, SYSFLUX, FLUX_ERR, FLAG FROM {NGTS_LIGHTCURVES}"
                 f" WHERE SOURCE_ID = '{source_id}'",
                 f"the light curve of {source_id}")

    return Table({
        'mjd': np.asarray(table['HJD'], dtype=float) - NGTS_HJD_TO_MJD,
        'flux': np.ma.filled(np.ma.asarray(table['SYSFLUX'], dtype=float), np.nan),
        'flux_err': np.ma.filled(np.ma.asarray(table['FLUX_ERR'], dtype=float), np.nan),
        'flag': np.ma.filled(np.ma.asarray(table['FLAG'], dtype=int), 0),
    })


def _log_flags(log, flags, bad):
    """How many points each flag was raised on, and which are dropped."""
    lines = []

    for bit, name, meaning in NGTS_FLAGS:
        count = int(np.sum((flags & bit) > 0))

        if count:
            lines.append(f"    {name:11s} {count:7d}  {meaning}"
                         + ('' if bit & bad else '  [kept]'))

    if lines:
        log("  what the pipeline flagged:")

        for line in lines:
            log(line)


def _bin(table, width):
    """The light curve in bins of width minutes, each source on its own.

    A bin is the error-weighted mean of what fell in it. Its error is the
    larger of the one the weights give and the scatter of its points about
    their mean: the pipeline's errors are formal ones, and a bin should not
    claim more than the points in it actually agree to.
    """
    width = width / 1440.0

    sources = np.asarray(table['source'], dtype=str)
    mjd = np.asarray(table['mjd'], dtype=float)
    flux = np.asarray(table['flux'], dtype=float)
    err = np.asarray(table['flux_err'], dtype=float)

    key = np.char.add(np.char.add(sources, ':'),
                      np.floor(mjd / width).astype(np.int64).astype(str))
    keys, inverse, npts = np.unique(key, return_inverse=True, return_counts=True)

    weight = 1.0 / err**2
    sum_w = np.bincount(inverse, weight)
    mean = np.bincount(inverse, weight * flux) / sum_w
    formal = 1.0 / np.sqrt(sum_w)

    scatter = np.sqrt(np.bincount(inverse, (flux - mean[inverse])**2)
                      / np.maximum(npts - 1, 1) / npts)

    first = np.zeros(len(keys), dtype=int)
    first[inverse[::-1]] = np.arange(len(table))[::-1]

    binned = Table({
        'mjd': np.bincount(inverse, mjd) / npts,
        'flux': mean,
        'flux_err': np.maximum(formal, scatter),
        'npts': npts,
        'source': sources[first],
    })

    return binned[binned['npts'] >= NGTS_BIN_MIN]


def _align(ngts, order, width, log):
    """Put every catalogue entry of one star on the scale of the first.

    A star in the strip where two fields overlap is catalogued once for each,
    and the two catalogue magnitudes differ by a tenth of a magnitude or so -
    each field is tied to APASS on its own. Neighbouring fields were mostly
    observed over the same months by different cameras, so where two entries
    share enough bins in time the second is shifted by the median difference
    between them. Where they do not, nothing says what the shift should be,
    and each is left on its own catalogue's scale.
    """
    names = np.asarray(ngts['source'], dtype=str)
    present = [n for n in order if np.any(names == n)]

    if len(present) < 2:
        return

    reference = names == present[0]
    ref_bins = {int(round(t*1440.0/width)): m for t, m in
                zip(ngts['mjd'][reference], ngts['mag'][reference])}

    for name in present[1:]:
        idx = names == name
        diff = [m - ref_bins[b] for t, m in zip(ngts['mjd'][idx], ngts['mag'][idx])
                for b in [int(round(t*1440.0/width))] if b in ref_bins]

        if len(diff) < NGTS_ALIGN_MIN:
            log(f"  {name}: only {len(diff)} bins at the same time as "
                f"{present[0]}, so it is left on its own catalogue scale")
            continue

        shift = float(np.median(diff))
        ngts['mag'][idx] -= shift

        log(f"  {name}: shifted by {-shift:+.3f} mag onto the scale of "
            f"{present[0]}, from {len(diff)} bins observed by both")


@survey_source(
    name='Next Generation Transit Survey',
    short_name='NGTS',
    state_acquiring='acquiring NGTS lightcurve',
    state_acquired='NGTS lightcurve acquired',
    log_file='ngts.log',
    output_files=['ngts.log', 'ngts_lc.png', 'ngts.vot', 'ngts.txt'],
    button_text='Get NGTS lightcurve',
    form_fields={
        'ngts_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': NGTS_SR,
            'required': False,
        },
        'ngts_bin': {
            'type': 'float',
            'label': 'Bin width, minutes',
            'initial': NGTS_BIN,
            'required': False,
        },
        'ngts_quality': quality_field({
            QUALITY_STANDARD: 'Drop every point the pipeline flagged',
            QUALITY_RELAXED: 'Drop only saturated measurements',
            QUALITY_PUBLISHED: 'None - every measurement as published',
        }),
    },
    help_text='Next Generation Transit Survey DR2, broad 520-890nm band, '
              '13 s cadence, 72 southern fields, 2015-2018',
    order=29,
    about=(
        "The Next Generation Transit Survey - twelve 20cm telescopes at "
        "Paranal following one field each through the night at a 13 second "
        "cadence. Its second data release has 72 fields observed between "
        "2015 and 2018, every star in them down to I = 16."),
    about_links=[
        ('NGTS', 'https://ngtransits.org/'),
        ('NGTS DR2 at ESO',
         'https://www.eso.org/rm/api/v1/public/releaseDescriptions/154'),
        ('Wheatley et al. 2018', 'https://arxiv.org/abs/1710.11100'),
    ],
    acknowledgement=(
        "Based on data collected under the NGTS project at the ESO La Silla "
        "Paranal Observatory. The release description asks that work using "
        "the data cite Wheatley et al. (2018), MNRAS 475, 4476."),
    # Lightcurve metadata
    votable_file='ngts.vot',
    lc_bands=[
        surveys.band('binned', 'mag', 'magerr', surveys.BAND_NATIVE,
                     color=NGTS_COLOR,
                     note='the broad NGTS band, 520-890nm, on the scale of '
                          'the catalogue magnitude, binned'),
    ],
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_color=NGTS_COLOR,
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata
    template_layout='simple',
    requires_coordinates=True,
)
def target_ngts(config, basepath=None, verbose=True, show=False):
    """Acquire NGTS lightcurve."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('ngts'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = float(config.get('ngts_sr') or NGTS_SR)
    width = float(config.get('ngts_bin') or NGTS_BIN)

    # The catalogue match and the light curves are cached apart: the first is
    # a few rows, and says what the second was made of once the second has
    # been read back from the cache
    with cached_votable_query(f"ngts_sources_{ra:.4f}_{dec:.4f}_{sr:.1f}.vot",
                              basepath, log, 'NGTS source catalogue',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            log(f"within {sr:.1f} arcsec")

            found = _sources(ra, dec, sr)

            if found is None:
                cache.save_empty()
                log("Warning: No NGTS source at this position - DR2 covers 72 "
                    "fields of 2.8 degrees, some 560 square degrees of the "
                    "southern sky, and only stars brighter than I = 16")
                return

            cache.save(found)

        sources = cache.data

    if sources is None:
        return

    for row in sources:
        log(f"  {row['source_id']}  {row['dist']:.2f} arcsec away, "
            f"NGTS mag {row['ngts_mag']:.2f}, {row['npts_total']} measurements")

    if len(sources) > 1:
        log("  more than one entry this close is one star catalogued in "
            "overlapping fields - each is fetched")

    if len(sources) > NGTS_MAX_SOURCES:
        sources = sources[:NGTS_MAX_SOURCES]

    # One cache per catalogue entry, named by it, so that the rows need not
    # each say which entry they belong to - at a quarter of a million rows a
    # light curve, that column alone would be a third of the file
    parts = []

    for row in sources:
        source_id = str(row['source_id'])

        with cached_votable_query(f"ngts_lc_{source_id}.vot", basepath, log,
                                  'NGTS full-cadence light curve',
                                  refresh=refresh_cache) as cache:
            if not cache.hit:
                lc = _lightcurve(source_id)
                log(f"  {source_id}: {len(lc)} measurements")

                if not len(lc):
                    cache.save_empty()
                    continue

                cache.save(lc)

            lc = cache.data

        if lc is not None and len(lc):
            lc['source'] = source_id
            parts.append(lc)

    if not parts:
        log("Warning: The NGTS catalogue lists a source here, but its light "
            "curve is empty")
        return

    full = vstack(parts)

    log(f"\n{len(full)} measurements at full cadence")

    flux = np.asarray(full['flux'], dtype=float)
    err = np.asarray(full['flux_err'], dtype=float)
    flags = np.asarray(full['flag'], dtype=int)

    # Unusable whatever was asked for: the pipeline marks a failed measurement
    # with a NaN or a zero flux, and a point without an error cannot be
    # weighted into a bin
    good = np.isfinite(flux) & (flux > 0) & np.isfinite(err) & (err > 0)

    log(f"{int(np.sum(~good))} measurements without a usable flux or error")

    quality = quality_level(config, 'ngts')
    bad = NGTS_BAD_FLAGS[quality]

    _log_flags(log, flags[good], bad)

    good &= (flags & bad) == 0
    full = full[good]

    log(f"{len(full)} measurements left ({quality} filtering)")

    if not len(full):
        log("Warning: No NGTS measurements left at this level of filtering")
        return

    ngts = _bin(full, width)

    log(f"Binned to {width:g} minutes: {len(ngts)} points of at least "
        f"{NGTS_BIN_MIN} measurements each")

    if not len(ngts):
        log("Warning: No bin of the NGTS light curve holds enough points")
        return

    # The fluxes are relative - the pipeline does no absolute calibration - so
    # each source is put on the scale of its own catalogue magnitude, which
    # the release ties to APASS I, through its own mean flux
    zero = {str(row['source_id']): float(row['ngts_mag'])
            + 2.5*np.log10(float(row['flux_mean'])) for row in sources}
    names = np.asarray(ngts['source'], dtype=str)

    ngts['mag'] = np.array([zero[n] for n in names]) - 2.5*np.log10(ngts['flux'])
    ngts['magerr'] = 2.5/np.log(10) * ngts['flux_err'] / ngts['flux']

    _align(ngts, [str(_) for _ in sources['source_id']], width, log)

    ngts.sort('mjd')

    log_conversion(
        log, 'NGTS',
        'mag = NGTS_MAG - 2.5*log10(flux / FLUX_MEAN)',
        {'NGTS_MAG': ('from the source catalogue', 'tied to APASS I by the '
                      'release'),
         'FLUX_MEAN': ('from the source catalogue', 'the mean of the light '
                       'curve')},
        npoints=len(ngts),
        note='not put on the common g scale: the band is 520-890nm wide, and '
             'no relation from it to g has been published',
    )

    log_bands(log, 'NGTS', [
        {'label': 'binned', 'kind': 'native', 'npoints': len(ngts),
         'note': f'binned to {width:g} minutes'},
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'ngts_lc.png'),
                            figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        time = Time(np.asarray(ngts['mjd'], dtype=float), format='mjd').datetime

        for name in sorted(set(names.tolist())):
            idx = np.asarray(ngts['source'], dtype=str) == name
            plot_with_errors(ax, time[idx], ngts['mag'][idx],
                             ngts['magerr'][idx], label=name)

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        if len(set(names.tolist())) > 1:
            ax.legend()

        ax.set_ylabel('NGTS mag')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - NGTS, "
                     f"{width:g} minute bins")

    log("NGTS lightcurve plot saved to file:ngts_lc.png")

    ngts.write(os.path.join(basepath, 'ngts.vot'), format='votable', overwrite=True)
    ngts.write(os.path.join(basepath, 'ngts.txt'),
               format='ascii.commented_header', overwrite=True)
    log("NGTS data written to file:ngts.vot")
    log("NGTS data written to file:ngts.txt")
