"""FRAM telescopes lightcurve acquisition module.

Acquires photometry from the FRAM archive at fram.fzu.cz, which holds the
robotic telescopes watching the sky over the Pierre Auger Observatory in
Argentina and over the CTAO sites at La Palma and Paranal.
"""

import os

import numpy as np
import requests

from astropy.table import Table
from astropy.time import Time

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    clip_noisy_points, quality_field, quality_level,
                    log_bands, log_conversion, plot_with_errors,
                    assumed_color, v_to_g, b_to_g, rc_to_g,
                    V_TO_G_FORMULA, B_TO_G_FORMULA, RC_TO_G_FORMULA,
                    CLIP_RATIO_BY_LEVEL,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


# The archive's own light curve endpoint. It returns the measurements already
# grouped by band and already passing the archive's quality cuts, which are
# made per camera and per band and use columns - the scatter of the frame's
# zero point, the fraction of calibration stars surviving it - that the JSON
# itself does not carry, so they cannot be reproduced here from what arrives.
FRAM_JSON_URL = 'http://fram.fzu.cz/archive/photometry/json'

# A wide cone over a decade of a partitioned table is tens of seconds of work
# for the archive on a cold cache, and the answer is then ours to keep
FRAM_TIMEOUT = 300

# The bands the pipeline extracts photometry in. B, V, R and I are Johnson-
# Cousins, calibrated against the Gaia DR3 synthetic photometry of the field
# stars; z sits on the La Palma telescope alone; N is the unfiltered channel,
# which is calibrated against catalogue R and so carries a colour term of its
# own on top of it.
FRAM_BANDS = ['B', 'V', 'R', 'I', 'z', 'N']

# How each of them is drawn, and what it is - said here once, and repeated to
# the light curve viewer by the registry entry below
FRAM_BAND_COLORS = {'B': '#1f77b4', 'V': '#2ca02c', 'R': '#d62728',
                    'I': '#8c564b', 'z': '#9467bd', 'N': '#7f7f7f'}

FRAM_BAND_NOTES = {
    'B': 'Johnson B, as reported by FRAM',
    'V': 'Johnson V, as reported by FRAM',
    'R': 'Cousins R, as reported by FRAM',
    'I': 'Cousins I, as reported by FRAM',
    'z': 'as reported by FRAM',
    'N': 'the unfiltered channel, calibrated against R',
}

# What a pixel of each camera covers, in arcsec, from the telescope
# descriptions the archive publishes on its front page. Keyed by the site and
# by the letters leading the camera name: 'NF' and 'C' are the narrow fields
# of the two telescopes, 'WF' the wide-field lenses standing beside them.
FRAM_PIXEL_SCALE = {
    ('auger', 'NF'): 0.92,    # 30 cm f/6.8, 60'x60'
    ('auger', 'WF'): 6.0,     # Nikkor 300/2.8, 7x7 deg
    ('auger2', 'WF'): 6.0,    # Canon 300/2.8, 7x7 deg
    ('cta-n', 'C'): 1.52,     # 25 cm f/6.3, 26'x26'
    ('cta-n', 'WF'): 14.1,    # Zeiss 135/2.0, 15x15 deg
    ('cta-s0', 'WF'): 14.1,   # the same lens, at Paranal
    ('cta-s1', 'WF'): 14.1,
}

# What to assume for a camera the table does not name - the coarsest of them,
# so that an unfamiliar one is never cut harder than the widest field here is.
FRAM_PIXEL_SCALE_DEFAULT = 14.1

# How far from the target a detection may sit and still be it, in pixels of
# the camera that made it. The photometry comes from a free detection rather
# than from a forced measurement at the catalogue position, so a measurement
# lands where the star was found; on the wide fields that wanders by a pixel
# or two, and on a bright star more than that. Two and a half pixels is where
# the offsets of an isolated star run out on every camera of the archive - up
# to half an arcminute on the 14 arcsec pixels of the CTA lenses, and three
# arcsec on the narrow field beside them.
FRAM_MATCH_PIXELS = 2.5

# The cone actually asked for, in arcsec. It has to hold the whole of the
# above for the coarsest camera; each camera is then held to its own radius,
# so a wide cone costs the narrow fields nothing.
FRAM_SR = 40.0


def camera_pixel_scale(site, ccd):
    """How many arcsec a pixel of this camera covers."""
    prefix = ''.join(_ for _ in str(ccd) if _.isalpha())

    return FRAM_PIXEL_SCALE.get((str(site), prefix), FRAM_PIXEL_SCALE_DEFAULT)


def angular_distance(ra, dec, ra0, dec0):
    """Separation from the target, in arcsec, on the tangent plane.

    The offsets here are arcseconds on cones of at most a minute, so the
    small-angle form is exact to well below what it is compared against.
    """
    dra = (np.asarray(ra, dtype=float) - ra0)
    dra = (dra + 180.0) % 360.0 - 180.0

    return np.hypot(dra*np.cos(np.deg2rad(dec0)),
                    np.asarray(dec, dtype=float) - dec0)*3600.0


def nearest_per_frame(image_ids, dist):
    """One measurement per frame - the detection nearest the target.

    A cone wide enough for the wide-field cameras can hold a neighbour as well
    as the star, and then both are measured on the same frame. The nearer of
    the two is the star; the other belongs in somebody else's light curve.
    Measurements whose frame has since left the archive have no id to be
    grouped by, and are left alone.
    """
    image_ids = np.asarray(image_ids)
    dist = np.asarray(dist, dtype=float)
    keep = np.ones(len(image_ids), dtype=bool)

    known = np.where(image_ids >= 0)[0]
    if not len(known):
        return keep

    # By frame, and within a frame by how far the detection sits from the
    # target, so that the first row of each frame is the one to keep
    order = known[np.lexsort((dist[known], image_ids[known]))]

    seen = set()
    for i in order:
        if image_ids[i] in seen:
            keep[i] = False
        else:
            seen.add(image_ids[i])

    return keep


def query_fram(ra, dec, sr, nofiltering=False, color_aware=False, bv=None):
    """Ask the archive for every measurement within the cone.

    `sr` is in degrees, as the endpoint wants it. `nofiltering` turns the
    archive's own quality cuts off; `color_aware` asks for the magnitudes
    calibrated alongside a colour term, with `bv` the colour to apply it with
    where one is known - the archive measures its own from the star's B and V
    pairs when it is not given one.
    """
    params = {'ra': ra, 'dec': dec, 'sr': sr}

    if nofiltering:
        params['nofiltering'] = 1

    if color_aware:
        params['color_aware'] = 'pairs'

        if bv is not None:
            params['bv'] = bv

    try:
        response = requests.get(FRAM_JSON_URL, params=params,
                                timeout=FRAM_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException as e:
        raise SourceError('could not download the FRAM lightcurve - '
                          f'{type(e).__name__}: {e}')
    except ValueError as e:
        raise SourceError('could not parse the FRAM answer - '
                          f'{type(e).__name__}: {e}')

    return payload


# The columns of the archive's JSON that are worth keeping, as the names they
# are given here. The rest of what it sends - the tangent-plane offsets, the
# ISO timestamps - either follows from these or duplicates them.
FRAM_COLUMNS = [
    ('mjds', 'mjd', float),
    ('mags', 'mag', float),
    ('magerrs', 'magerr', float),
    ('ras', 'ra', float),
    ('decs', 'dec', float),
    ('sites', 'site', str),
    ('ccds', 'ccd', str),
    ('nights', 'night', str),
    ('flags', 'flags', int),
    ('fwhms', 'fwhm', float),
    ('stds', 'std', float),
    ('nstars', 'nstars', int),
    ('color_term', 'color_term', float),
    ('image_ids', 'image_id', int),
]


def fram_table(payload):
    """The archive's per-band JSON as one table of measurements.

    A column may be missing for a measurement whose frame is gone from the
    archive, which is what the fill values below are for: an unknown frame is
    -1 rather than a hole, so that the table writes as a VOTable of plain
    numbers.

    The colour the archive applied each frame's colour term with travels in a
    column of its own, rather than being remembered as a property of the
    request: the archive decides per camera and band whether the correction is
    worth making at all, and the answer is cached and read back long after the
    request that produced it. A NaN there means the magnitude beside it is the
    colour-independent one.
    """
    columns = {name: [] for _, name, _t in FRAM_COLUMNS}
    columns['filter'] = []
    columns['bv'] = []

    # What the archive made of the colour term, and where it made it
    bv = payload.get('bv') if payload.get('color_aware') else None
    color_groups = set(payload.get('color_groups') or [])

    for lc in payload.get('lcs') or []:
        fname = str(lc.get('filter'))

        if fname not in FRAM_BANDS:
            continue

        npoints = len(lc.get('mjds') or [])
        columns['filter'].extend([fname]*npoints)

        if bv is None:
            columns['bv'].extend([np.nan]*npoints)
        else:
            columns['bv'].extend(
                [float(bv) if f'{s}/{c}/{fname}' in color_groups else np.nan
                 for s, c in zip(lc.get('sites') or [], lc.get('ccds') or [])])

        for key, name, kind in FRAM_COLUMNS:
            values = lc.get(key) or [None]*npoints

            if kind is str:
                columns[name].extend(['' if _ is None else str(_) for _ in values])
            elif kind is int:
                columns[name].extend([-1 if _ is None else int(_) for _ in values])
            else:
                columns[name].extend([np.nan if _ is None else float(_)
                                      for _ in values])

    if not len(columns['filter']):
        return None

    table = Table(columns)
    table.sort('mjd')

    return table


@survey_source(
    name='FRAM telescopes',
    short_name='FRAM',
    state_acquiring='acquiring FRAM lightcurve',
    state_acquired='FRAM lightcurve acquired',
    log_file='fram.log',
    output_files=['fram.log', 'fram_lc.png', 'fram.vot', 'fram.txt'],
    button_text='Get FRAM lightcurve',
    form_fields={
        'fram_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': FRAM_SR,
            'required': False,
        },
        'fram_color_aware': {
            'type': 'choice',
            'label': 'Colour term',
            'choices': [
                ('auto', "Apply each frame's colour term"),
                ('off', 'Leave the magnitudes as the archive calibrates them'),
            ],
            'initial': 'auto',
            'required': False,
        },
        'fram_quality': quality_field({
            QUALITY_STANDARD: "The archive's cuts, and one match per frame",
            QUALITY_RELAXED: "The archive's cuts over the whole search radius",
            QUALITY_PUBLISHED: 'None - every measurement as extracted',
        }),
    },
    help_text='Robotic telescopes of the Pierre Auger Observatory and CTAO',
    order=61,
    # Lightcurve metadata
    votable_file='fram.vot',
    lc_bands=[
        surveys.band('B', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='B',
                     color='#1f77b4', note='Johnson B, as reported by FRAM'),
        surveys.band('V', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='V',
                     color='#2ca02c', note='Johnson V, as reported by FRAM'),
        surveys.band('R', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='R',
                     color='#d62728', note='Cousins R, as reported by FRAM'),
        surveys.band('I', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='I',
                     color='#8c564b', note='Cousins I, as reported by FRAM'),
        surveys.band('z', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='z',
                     color='#9467bd', note='as reported by FRAM'),
        surveys.band('N', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='N',
                     color='#7f7f7f',
                     note='the unfiltered channel, calibrated against R'),
        surveys.band('g (conv.)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     color='#aec7e8',
                     note='B, V and R put on the common g scale using an '
                          'assumed g - r',
                     combined=True),
    ],
    lc_mag_column='mag_g',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color='#1f77b4',
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata
    template_layout='simple',
    requires_coordinates=True,
)
def target_fram(config, basepath=None, verbose=True, show=False):
    """
    Get FRAM lightcurve.

    Parameters
    ----------
    config : dict
        Configuration dictionary with target coordinates
    basepath : str, optional
        Base path for output files
    verbose : bool or callable, optional
        Verbose logging mode or log function
    show : bool, optional
        Show plots interactively
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('fram'), basepath=basepath)

    ra = config.get('target_ra')
    dec = config.get('target_dec')

    if ra is None or dec is None:
        raise RuntimeError("Cannot operate without target coordinates")

    try:
        fram_sr = float(config.get('fram_sr') or FRAM_SR)
    except (TypeError, ValueError):
        fram_sr = FRAM_SR

    if not np.isfinite(fram_sr) or fram_sr <= 0:
        fram_sr = FRAM_SR

    quality = quality_level(config, 'fram')

    # The magnitudes calibrated alongside a colour term, with that term then
    # applied with the colour of the star. The zero point fitted without one
    # is a compromise over the calibration stars, so a star whose colour is
    # not theirs picks up a different offset on every camera and every season
    # - which is the largest disagreement between the cameras there is here.
    color_aware = str(config.get('fram_color_aware') or 'auto').lower() != 'off'

    B_minus_V, B_minus_V_origin = assumed_color(config, 'B_minus_V')
    bv = B_minus_V if B_minus_V_origin == 'from config' else None

    # Everything that changes what the archive sends back belongs in the cache
    # name, since none of it can be applied to a reply that was made without it
    variant = ''
    if quality == QUALITY_PUBLISHED:
        variant += '_all'
    if color_aware:
        variant += '_bv%.2f' % bv if bv is not None else '_bvauto'

    cache_name = f"fram_{ra:.4f}_{dec:.4f}_{fram_sr:.1f}{variant}.vot"

    with cached_votable_query(cache_name, basepath, log, 'FRAM',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            log(f"for {config['target_name']} within {fram_sr:.1f} arcsec")
            log(f"Querying FRAM at RA={ra:.4f}, Dec={dec:.4f}")

            if color_aware:
                if bv is not None:
                    log(f"Asking for the colour term to be applied with "
                        f"(B - V) = {bv:.3f}")
                else:
                    log("Asking for the colour term to be applied with the "
                        "colour the archive measures itself")

            payload = query_fram(ra, dec, fram_sr/3600.0,
                                 nofiltering=(quality == QUALITY_PUBLISHED),
                                 color_aware=color_aware, bv=bv)

            fram = fram_table(payload)

            if fram is None:
                cache.save_empty()
                log("Warning: No FRAM data found")
                return

            if color_aware and not payload.get('color_aware'):
                # Said here rather than below, where the cached answer no
                # longer knows what was asked of it. The archive refuses the
                # correction where no colour could be found, and where no
                # camera measures its own colour term better than that term
                # varies - which on the narrow fields it does not.
                log("The archive did not apply the colour term - no colour "
                    "was available, or no camera measures its term well "
                    "enough to be worth it")

            log(f"Downloaded {len(fram)} data points from FRAM")

            cache.save(fram)

        fram = cache.data

    # Nothing here, and cached as nothing - the helper has said so already
    if fram is None:
        return

    # A VOTable column carrying a NaN reads back masked, and one that is
    # nothing but NaNs - the colour of a light curve the correction was
    # refused on - reads back as nothing but its mask, which every comparison
    # below would return `masked` from rather than a number. Filled once here,
    # so that the rest of this is arithmetic on plain arrays.
    for name in fram.colnames:
        if fram[name].dtype.kind == 'f':
            fram[name] = np.ma.filled(np.ma.asarray(fram[name], dtype=float),
                                      np.nan)

    log(f"{len(fram)} original data points")

    # How far each detection landed from the position asked for, which is what
    # the matching below is decided on
    fram['dist'] = angular_distance(fram['ra'], fram['dec'], ra, dec)

    # Positional matching, per camera. A single radius cannot serve both ends
    # of this archive: three arcsec is most of a wide-field star's detections
    # thrown away, and half an arcminute on the narrow field is a light curve
    # of whatever else is in the field.
    if quality == QUALITY_STANDARD:
        scales = np.array([camera_pixel_scale(s, c)
                           for s, c in zip(fram['site'], fram['ccd'])])
        radius = np.minimum(FRAM_MATCH_PIXELS*scales, fram_sr)

        idx = fram['dist'] <= radius

        log(f"{int(np.sum(idx))} data points within "
            f"{FRAM_MATCH_PIXELS:.1f} pixels of their own camera")

        fram = fram[idx]

        if len(fram):
            keep = nearest_per_frame(fram['image_id'], fram['dist'])

            if not np.all(keep):
                log(f"Dropping {int(np.sum(~keep))} further detections on "
                    "frames that already have a nearer one")
                fram = fram[keep]

    if not len(fram):
        log("Warning: No FRAM measurements left after matching")
        return

    # Whatever the archive could not measure. Read for what it says rather
    # than judged, so this much is done at every level of filtering.
    idx = np.isfinite(fram['mag']) & np.isfinite(fram['magerr'])
    idx &= fram['magerr'] > 0

    if not np.all(idx):
        log(f"Dropping {int(np.sum(~idx))} points with no usable magnitude")
        fram = fram[idx]

    if not len(fram):
        log("Warning: No usable FRAM measurements")
        return

    # The single ruined frame the archive's own cuts let through, judged
    # against what its camera and band manage at the same brightness
    groups = np.array([f"{s}/{c} {f}" for s, c, f
                       in zip(fram['site'], fram['ccd'], fram['filter'])])

    clip = (clip_noisy_points(fram['mag'], fram['magerr'], groups,
                              log=log, group_name='camera and band',
                              ratio=CLIP_RATIO_BY_LEVEL[quality])
            if quality != QUALITY_PUBLISHED
            else np.zeros(len(fram), dtype=bool))

    if np.any(clip):
        fram = fram[~clip]

    if not len(fram):
        log("Warning: No FRAM measurements left at this level of filtering")
        return

    log(f"{len(fram)} data points after filtering")

    bands = [_ for _ in FRAM_BANDS if np.any(fram['filter'] == _)]

    cameras = np.array([f"{s}/{c}" for s, c in zip(fram['site'], fram['ccd'])])

    log("Cameras: " + ', '.join(f"{name} ({int(np.sum(cameras == name))})"
                                for name in sorted(set(cameras))))

    # What the archive did with each frame's colour term, read off the
    # magnitudes themselves rather than off the request that fetched them
    n_color = int(np.sum(np.isfinite(fram['bv'])))
    bv_applied = (float(np.nanmedian(fram['bv'])) if n_color else None)

    if n_color:
        colour_term = (f"applied per frame to {n_color} of {len(fram)} points",
                       f"with (B - V) = {bv_applied:.3f}")
    elif color_aware:
        colour_term = ('not applied',
                       'asked for, and the archive found no camera whose '
                       'colour term it measures better than that term varies')
    else:
        colour_term = ('not applied', 'not asked for')

    log_conversion(
        log, 'FRAM',
        'no band conversion applied - each is published as measured',
        {'calibration': ('Gaia DR3 synthetic photometry',
                         'of the field stars of every frame'),
         'colour term': colour_term},
        npoints=len(fram),
    )

    # The common g scale. B, V and R each have a relation of their own; I, z
    # and N have none worth using - the first two would need a second assumed
    # colour, and the unfiltered channel is only calibrated against R rather
    # than measured in it, so its own colour term would ride on top of the
    # assumed colour with nothing to bound it.
    g_minus_r, g_minus_r_origin = assumed_color(config, 'g_minus_r')

    fram['mag_g'] = np.nan

    for fname, convert, formula in [('V', v_to_g, V_TO_G_FORMULA),
                                    ('B', b_to_g, B_TO_G_FORMULA),
                                    ('R', rc_to_g, RC_TO_G_FORMULA)]:
        idx = fram['filter'] == fname

        if not np.any(idx):
            continue

        fram['mag_g'][idx] = convert(np.asarray(fram['mag'][idx], dtype=float),
                                     g_minus_r)

        log_conversion(
            log, 'FRAM', formula,
            {'(g - r)': (g_minus_r, g_minus_r_origin)},
            npoints=int(np.sum(idx)),
            note='the colour is assumed constant over the whole light curve',
        )

    log_bands(log, 'FRAM', [
        {'label': fname, 'kind': 'native',
         'npoints': int(np.sum(fram['filter'] == fname)),
         'note': FRAM_BAND_NOTES.get(fname, 'as reported by FRAM')}
        for fname in bands
    ] + [
        {'label': 'g (conv.)', 'kind': 'derived',
         'npoints': int(np.sum(np.isfinite(fram['mag_g']))),
         'note': 'B, V and R on the common g scale'},
    ])

    fram['time'] = Time(fram['mjd'], format='mjd')

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'fram_lc.png'),
                            figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        for fname in bands:
            idx = fram['filter'] == fname

            plot_with_errors(ax, fram['time'][idx].datetime,
                             fram['mag'][idx], fram['magerr'][idx],
                             color=FRAM_BAND_COLORS.get(fname), label=fname,
                             ms=2)

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        if len(bands) > 1:
            ax.legend()

        ax.set_ylabel('Magnitude')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - FRAM")

    log("FRAM lightcurve plot saved to file:fram_lc.png")

    # Time cannot be serialized to VOTable
    fram_save = fram[[_ for _ in fram.columns if _ != 'time']]

    fram_save.write(os.path.join(basepath, 'fram.vot'),
                    format='votable', overwrite=True)
    fram_save.write(os.path.join(basepath, 'fram.txt'),
                    format='ascii.commented_header', overwrite=True)
    log("FRAM data written to file:fram.vot")
    log("FRAM data written to file:fram.txt")

