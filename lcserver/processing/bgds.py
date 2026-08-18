"""Bochum Galactic Disk Survey lightcurve acquisition module.

Acquires BGDS DR2 lightcurves from the GAVO data centre, finding what was
observed through TAP and fetching each light curve through datalink.
"""

import os
import io

import requests
import numpy as np

from astropy.table import Table
from astropy.time import Time

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    clip_noisy_points, quality_field, quality_level,
                    log_bands, log_conversion, plot_with_errors,
                    assumed_color, r_to_g, R_TO_G_FORMULA, CLIP_RATIO_BY_LEVEL,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


BGDS_TAP = 'https://dc.g-vo.org/tap/sync'

# The union of every band's light curves. The per-band tables (bgds2.lc_r and
# its nine siblings) hold exactly the same rows, so asking the union once is
# one query where asking them separately would be ten.
BGDS_TABLE = 'bgds2.lc_all'

# Where a whole light curve is served from, one object in one band in one
# field per call, keyed by the obs_id the table above gives.
#
# The table itself carries the light curves too, in three array-valued
# columns, and they would come back in the one query that found the object -
# but the service declares those arrays a fixed 154 elements long and cuts
# every longer one down to it, silently and without saying it has. That loses
# half of what the survey recorded for the stars it watched most, which are
# the ones worth having. So the table is asked only what it was observed in,
# and the measurements are fetched from the service that serves them whole.
BGDS_DATALINK = 'https://dc.g-vo.org/bgds/l2/tsdl/dlget'

# Per request to either service. Both answer in well under a second when they
# answer at all.
BGDS_TIMEOUT = 120

# The photometric aperture is adaptive, but averages about five arcseconds
# across, so a match much wider than this is a different star rather than a
# looser match to the same one
BGDS_SR = 3.0

# A guard against a runaway in a crowded field, not a real limit: one star
# brings back at most one row per band per field it was imaged in, which is
# ten bands over the four or five overlapping fields at the very most
BGDS_MAX_ROWS = 200

# What the survey calls each band, and what to call it here. In order of
# wavelength, U at 346nm through z' at 896nm, with the four narrow bands
# interleaved where they fall.
BGDS_BANDS = [
    # (band_name as published, label, colour)
    ('Johnson U', 'U', '#7b4173'),
    ('Johnson B', 'B', '#3182bd'),
    ('Astrodon OIII', 'OIII', '#5254a3'),
    ('Johnson V', 'V', '#31a354'),
    ('SDSS r\'', 'r', '#e6550d'),
    ('Astrodon NB', 'NB', '#8ca252'),
    ('Astrodon Halpha', 'Ha', '#d6616b'),
    ('Astrodon SII', 'SII', '#ce6dbd'),
    ('SDSS i\'', 'i', '#a63603'),
    ('SDSS z\'', 'z', '#636363'),
]

BGDS_LABELS = {name: label for name, label, _ in BGDS_BANDS}


def _read_votable(content, what):
    """An astropy table from a VOTable, or None where it holds no rows."""
    try:
        table = Table.read(io.BytesIO(content), format='votable')
    except Exception as e:
        raise SourceError(f"could not read the BGDS {what} - "
                          f"{type(e).__name__}: {e}")

    return table if len(table) else None


def _query_tap(session, query):
    """What the survey has at a position, or None where it has nothing.

    One row per object per band per field, saying what it was observed in and
    how often, but not the measurements themselves - those are fetched one
    light curve at a time, for the reason given at BGDS_DATALINK.
    """
    try:
        res = session.post(BGDS_TAP, timeout=BGDS_TIMEOUT, data={
            'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'votable',
            'QUERY': query})
        res.raise_for_status()
    except requests.RequestException as e:
        raise SourceError("could not query the BGDS TAP service - "
                          f"{type(e).__name__}: {e}")

    # A refused query is a VOTable too, and one astropy will happily read as a
    # table of no rows, so the refusal has to be caught before it looks like
    # an answer of nothing
    if b'QUERY_STATUS" value="ERROR"' in res.content:
        raise SourceError("the BGDS TAP service refused the query: "
                          + res.text[:300])

    return _read_votable(res.content, 'answer')


def _lightcurve(session, obs_id):
    """One light curve whole, as datalink serves it.

    Three columns: a barycentric MJD, the magnitude in the band the obs_id
    names, and its error. A fourth links to a cutout of the frame the point
    was measured on, which is not kept - there is one per point, and they are
    of no use to a light curve.
    """
    try:
        res = session.get(BGDS_DATALINK, timeout=BGDS_TIMEOUT,
                          params={'ID': obs_id})
        res.raise_for_status()
    except requests.RequestException as e:
        raise SourceError(f"could not fetch {obs_id} - "
                          f"{type(e).__name__}: {e}")

    return _read_votable(res.content, f"light curve of {obs_id}")


def _acquire(session, table, log):
    """Every light curve the survey has at the position, as one table.

    Each row of `table` is one object seen in one band in one field, so a star
    imaged where two fields overlap comes back several times over in the same
    band. They are kept and merged rather than the best of them chosen: the
    survey did not match its own detections across fields, and the medians of
    the fields agree to a few hundredths of a magnitude, so taking one field
    alone would throw away most of the coverage for no gain in consistency.
    Which field a point came from is kept alongside it.
    """
    mjd, mag, magerr, band, field, obs_id = [], [], [], [], [], []
    failed = []

    for row in table:
        name = str(row['band_name'])
        label = BGDS_LABELS.get(name, name)
        this_id = str(row['obs_id'])

        try:
            lc = _lightcurve(session, this_id)
        except SourceError as e:
            # One curve of many, and the rest are still worth having
            log(f"Warning: {e}")
            failed.append(this_id)
            continue

        if lc is None:
            continue

        t = np.asarray(lc['obs_time'], dtype=float)
        m = np.asarray(lc['phot'], dtype=float)
        e = np.asarray(lc['mag_error'], dtype=float)

        idx = np.isfinite(t) & np.isfinite(m)

        if not np.any(idx):
            continue

        n = int(np.sum(idx))

        mjd.append(t[idx])
        mag.append(m[idx])
        magerr.append(e[idx])
        band.append(np.full(n, label))
        field.append(np.full(n, str(row['field'])))
        obs_id.append(np.full(n, this_id))

        log(f"  {label:<4s} {n:5d} points  {row['field']}"
            + (f"  [of {row['nobs']} the table counts]" if n != row['nobs'] else '')
            + ("  [the survey calls it variable]" if row['var'] == 1 else ''))

    # Every one of them refused, which is the service being down rather than
    # the survey having nothing here
    if failed and not mjd:
        raise SourceError(f"none of the {len(failed)} BGDS light curves at "
                          "this position could be fetched")

    if not mjd:
        return None

    return Table([np.concatenate(mjd), np.concatenate(mag),
                  np.concatenate(magerr), np.concatenate(band),
                  np.concatenate(field), np.concatenate(obs_id)],
                 names=['mjd', 'mag', 'magerr', 'filter', 'field', 'obs_id'])


@survey_source(
    name='Bochum Galactic Disk Survey',
    short_name='BGDS',
    state_acquiring='acquiring BGDS lightcurve',
    state_acquired='BGDS lightcurve acquired',
    log_file='bgds.log',
    output_files=['bgds.log', 'bgds_lc.png', 'bgds.vot', 'bgds.txt'],
    button_text='Get BGDS lightcurve',
    form_fields={
        'bgds_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': BGDS_SR,
            'required': False,
        },
        'bgds_quality': quality_field({
            QUALITY_STANDARD: 'Drop the frames a band and field measured worst',
            QUALITY_RELAXED: 'Drop only the very worst frames',
            QUALITY_PUBLISHED: 'None - every measurement as published',
        }),
    },
    help_text="Bochum Galactic Disk Survey DR2, r' and i' along the southern "
              "Galactic plane, 2010-2019",
    order=12,
    # Lightcurve metadata
    votable_file='bgds.vot',
    lc_bands=[
        surveys.band(label, 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value=label,
                     color=color, note=f'as reported by BGDS, in {name}')
        for name, label, color in BGDS_BANDS
    ] + [
        surveys.band('g (from r)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     filter_column='filter', filter_value='r',
                     color='#fdae6b',
                     note="r' taken as Pan-STARRS r and moved onto g using an "
                          "assumed g - r",
                     combined=True),
    ],
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color='#e6550d',
    lc_mode='magnitude',
    lc_short=True,
    # The survey is a six degree wide stripe on the southern Galactic plane,
    # so most of the sky it does not reach is not ruled out by declination at
    # all. These are the extremes of what it did observe, and only spare the
    # northern half of the sky a query that could not have found anything.
    declination_min=-70.0,
    declination_max=2.0,
    # Template metadata
    template_layout='simple',
)
def target_bgds(config, basepath=None, verbose=True, show=False):
    """Acquire Bochum Galactic Disk Survey lightcurve."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('bgds'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    bgds_sr = config.get('bgds_sr', BGDS_SR)

    cache_name = f"bgds_{ra:.4f}_{dec:.4f}_{bgds_sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log,
                              'Bochum Galactic Disk Survey',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            log(f"within {bgds_sr:.1f} arcsec")

            # One session for the discovery query and every light curve after
            # it, so that the connection is opened once rather than per curve
            session = requests.Session()

            table = _query_tap(session,
                f"SELECT TOP {BGDS_MAX_ROWS}"
                " obs_id, band_name, field, nobs, var"
                f" FROM {BGDS_TABLE}"
                " WHERE 1=CONTAINS(POINT('ICRS', ra, dec),"
                f" CIRCLE('ICRS', {float(ra):.7f}, {float(dec):.7f},"
                f" {float(bgds_sr) / 3600.0:.9f}))")

            if table is None:
                cache.save_empty()
                log("Warning: No BGDS object at this position - the survey "
                    "covers a six degree stripe along the Galactic plane, "
                    "not the whole southern sky")
                return

            log(f"{len(table)} BGDS light curves found:")

            bgds = _acquire(session, table, log)

            if bgds is None:
                cache.save_empty()
                log("Warning: The BGDS light curves at this position hold no "
                    "usable measurements")
                return

            # What is cached is the finished light curve rather than the
            # separate curves it was assembled from
            cache.save(bgds)

        bgds = cache.data

    # Nothing here, and cached as nothing - the helper has said so already
    if bgds is None:
        return

    log(f"{len(bgds)} data points")

    # Filter out bad data
    bgds = bgds[np.isfinite(bgds['mag'])]
    bgds = bgds[np.isfinite(bgds['magerr'])]
    bgds = bgds[bgds['magerr'] > 0]
    bgds = bgds[bgds['magerr'] < 1.0]

    log(f"{len(bgds)} data points after filtering")
    log("Times are barycentric MJD, which is how the survey serves them")

    if not len(bgds):
        log("Warning: No valid BGDS data points after filtering")
        return

    # Judged within each band and field at once: the fields were observed to
    # different depths and through different amounts of the plane's dust, so
    # what counts as a bad frame in one is an ordinary one in another.
    quality = quality_level(config, 'bgds')
    groups = np.char.add(np.char.add(np.asarray(bgds['filter'], dtype=str), ' '),
                         np.asarray(bgds['field'], dtype=str))
    clip = (clip_noisy_points(bgds['mag'], bgds['magerr'], groups,
                              log=log, group_name='band and field',
                              ratio=CLIP_RATIO_BY_LEVEL[quality])
            if quality != QUALITY_PUBLISHED
            else np.zeros(len(bgds), dtype=bool))

    if np.any(clip):
        bgds = bgds[~clip]
        log(f"{len(bgds)} data points left")

        if not len(bgds):
            log("Warning: No BGDS measurements left at this level of filtering")
            return

    bgds.sort('mjd')

    bands = [str(_) for _ in np.unique(bgds['filter'])]

    log_conversion(
        log, 'BGDS',
        'no conversion applied - each band is published as measured',
        {'colour term': ('none', 'calibrated against Landolt standards'),
         'bands present': ', '.join(bands)},
        npoints=len(bgds),
    )

    # Onto the common g scale. Only r' makes the trip: it and i' are the two
    # bands the survey observed throughout, and of the two r' is the one that
    # can be read as an r at all. Taking the SDSS r' the survey calibrated for
    # a Pan-STARRS r is an approximation on top of the assumed colour, though
    # a much smaller one than the colour itself.
    g_minus_r, g_minus_r_origin = assumed_color(config, 'g_minus_r')

    idx_r = bgds['filter'] == 'r'
    bgds['mag_g'] = np.nan
    bgds['mag_g'][idx_r] = r_to_g(np.asarray(bgds['mag'][idx_r], dtype=float),
                                  g_minus_r)

    n_r = int(np.sum(idx_r))

    if n_r:
        log_conversion(
            log, 'BGDS',
            R_TO_G_FORMULA,
            {'(g - r)': (g_minus_r, g_minus_r_origin),
             "r'": ('taken as Pan-STARRS r', "BGDS observes in SDSS r'")},
            npoints=n_r,
            note='only so that the r points reach the combined light curve; '
                 'the native bands are untouched',
        )

    log_bands(log, 'BGDS', [
        {'label': band, 'kind': 'native',
         'npoints': int(np.sum(bgds['filter'] == band)),
         'note': 'as reported by BGDS'}
        for band in bands
    ] + [
        {'label': 'g (from r)', 'kind': 'derived', 'npoints': n_r,
         'note': "r' on the common g scale"},
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'bgds_lc.png'),
                            figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        time = Time(np.asarray(bgds['mjd'], dtype=float), format='mjd').datetime

        # In the order the bands were declared in, so that the legend runs
        # from blue to red rather than alphabetically
        for _, label, color in BGDS_BANDS:
            idx = bgds['filter'] == label

            if np.any(idx):
                plot_with_errors(ax, time[idx], bgds['mag'][idx],
                                 bgds['magerr'][idx], color=color, label=label)

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        if len(bands) > 1:
            ax.legend()

        ax.set_ylabel('Magnitude')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - Bochum Galactic Disk Survey")

    log("BGDS lightcurve plot saved to file:bgds_lc.png")

    bgds.write(os.path.join(basepath, 'bgds.vot'), format='votable', overwrite=True)
    bgds.write(os.path.join(basepath, 'bgds.txt'), format='ascii.commented_header',
               overwrite=True)
    log("BGDS data written to file:bgds.vot")
    log("BGDS data written to file:bgds.txt")
