"""SkyMapper Southern Survey lightcurve acquisition module.

SkyMapper DR4 (Onken et al. 2024) publishes every detection behind its
catalogue of the southern sky: uvgriz from the 1.3 m telescope at Siding
Spring, 2014 to 2021, a few tens of epochs per star, the Shallow Survey's
short exposures reaching stars several magnitudes brighter than DECam can.
Read here from the copy at the Astro Data Lab, which serves the catalogue and
its detections but not the table of images.

That matters for the time of an epoch, which the detections do not carry
except as the identifier of their image: the UT at which the exposure began,
written YYYYMMDDhhmmss - checked against SkyMapper's own images table, whose
start MJDs it gives to the second. The exposures are 5 to 100 s long, so an
epoch is timed up to 50 s early.

g and r are put on the Pan-STARRS g scale of the combined curve through the
Pancino et al. (2022) relations stdpipe uses for the catalogue, with the
colours of the star itself - its mean SkyMapper magnitudes - rather than an
assumed one. The rest are published as measured.
"""

import os

import numpy as np

from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (cleanup_paths, cached_votable_query, datalab_query,
                    datalab_cone, log_bands, log_conversion, plot_with_errors,
                    clip_noisy_points, quality_field, quality_level,
                    skymapper_to_ps1, SKYMAPPER_G_TO_PS1_G, SKYMAPPER_R_TO_PS1_R,
                    SKYMAPPER_TO_PS1_FORMULA, CLIP_RATIO_BY_LEVEL,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


# Matching radius, in arcsec. SkyMapper's pixels are 0.5 arcsec and its seeing
# two or three, so the nearest object within this is taken.
SKYMAPPER_SR = 2.0

# The bands, blue to red
SKYMAPPER_BANDS = ('u', 'v', 'g', 'r', 'i', 'z')
SKYMAPPER_COLORS = {'u': '#9467bd', 'v': '#1f77b4', 'g': '#2ca02c',
                    'r': '#d62728', 'i': '#ff7f0e', 'z': '#8c564b'}

# What SkyMapper itself averages into the catalogue: Source Extractor flags
# below 4 - so nothing saturated, truncated or incomplete, and none of the
# survey's own bits above those - and fewer than five pixels masked as bad,
# saturated or crosstalk. The relaxed level drops only detections flagged
# saturated or truncated at the image edge.
SKYMAPPER_FLAGS_STANDARD = 4
SKYMAPPER_NIMAFLAGS_STANDARD = 5
SKYMAPPER_FLAGS_RELAXED = 4 | 8


def _column(table, name, dtype, fill):
    return np.ma.filled(np.ma.asarray(table[name], dtype=dtype), fill)


def _image_mjd(image_id):
    """The MJD at which each exposure began, from its image identifier."""
    ids = np.char.zfill(np.asarray(image_id).astype(np.int64).astype(str), 14)
    isot = [f"{_[0:4]}-{_[4:6]}-{_[6:8]}T{_[8:10]}:{_[10:12]}:{_[12:14]}"
            for _ in ids]

    return Time(isot, format='isot', scale='utc').mjd


def _fetch(ra, dec, sr, log):
    """The detections of the nearest object, with its mean magnitudes, or None."""
    found = datalab_query(
        "SELECT object_id, raj2000, dej2000, g_psf, r_psf, i_psf "
        "FROM skymapper_dr4.master WHERE "
        + datalab_cone('raj2000', 'dej2000', ra, dec, sr),
        'the SkyMapper catalogue')

    if not len(found):
        return None

    dist = SkyCoord(ra, dec, unit='deg').separation(
        SkyCoord(np.asarray(found['raj2000'], dtype=float),
                 np.asarray(found['dej2000'], dtype=float), unit='deg')).arcsec
    i = int(np.argmin(dist))
    row = found[i]
    oid = int(row['object_id'])

    log(f"  SkyMapper: object {oid}, {dist[i]:.2f} arcsec away"
        + (f", the nearest of {len(found)}" if len(found) > 1 else ""))

    lc = datalab_query(f"SELECT image_id, filter, mag_psf, e_mag_psf, flags,"
                       f" nimaflags, img_qual FROM skymapper_dr4.photometry"
                       f" WHERE object_id = {oid}",
                       'the SkyMapper detections')

    if not len(lc):
        return None

    table = Table({
        'source': np.full(len(lc), str(oid)),
        'image_id': _column(lc, 'image_id', np.int64, 0),
        'mjd': _image_mjd(_column(lc, 'image_id', np.int64, 0)),
        'filter': np.char.strip(np.asarray(lc['filter'], dtype=str)),
        'mag': _column(lc, 'mag_psf', float, np.nan),
        'magerr': _column(lc, 'e_mag_psf', float, np.nan),
        'flags': _column(lc, 'flags', int, -1),
        'nimaflags': _column(lc, 'nimaflags', int, -1),
        'img_qual': _column(lc, 'img_qual', int, -1),
    })

    # The mean magnitudes the conversion takes its colours from, as columns so
    # that a cached result carries them too
    for band in ('g', 'r', 'i'):
        value = row[f'{band}_psf']
        table[f'mean_{band}'] = (float(value) if not np.ma.is_masked(value)
                                 else np.nan)

    return table


def _judge(table, quality, log):
    """The mask of detections to keep, with every removal logged."""
    mag = np.asarray(table['mag'], dtype=float)
    err = np.asarray(table['magerr'], dtype=float)

    keep = (np.isfinite(mag) & np.isfinite(err) & (mag > 0)
            & (err > 0) & (err < 1.0))

    if np.any(~keep):
        log(f"    {int(np.sum(~keep)):5d} removed - no usable magnitude or error")

    if quality == QUALITY_PUBLISHED:
        return keep

    flags = np.asarray(table['flags'], dtype=int)
    nimaflags = np.asarray(table['nimaflags'], dtype=int)

    if quality == QUALITY_STANDARD:
        criteria = [
            ((flags >= 0) & (flags < SKYMAPPER_FLAGS_STANDARD),
             f'Source Extractor flags {SKYMAPPER_FLAGS_STANDARD} or above'),
            ((nimaflags >= 0) & (nimaflags < SKYMAPPER_NIMAFLAGS_STANDARD),
             f'{SKYMAPPER_NIMAFLAGS_STANDARD} or more bad, saturated or '
             f'crosstalk pixels'),
        ]
    else:
        criteria = [
            ((flags >= 0) & ((flags & SKYMAPPER_FLAGS_RELAXED) == 0),
             'saturated or truncated at the image edge'),
        ]

    for passed, meaning in criteria:
        removed = int(np.sum(keep & ~passed))

        if removed:
            log(f"    {removed:5d} removed - {meaning}")

        keep &= passed

    return keep


@survey_source(
    name='SkyMapper Southern Survey DR4',
    short_name='SkyMapper',
    state_acquiring='acquiring SkyMapper lightcurve',
    state_acquired='SkyMapper lightcurve acquired',
    log_file='skymapper.log',
    output_files=['skymapper.log', 'skymapper_lc.png', 'skymapper.vot',
                  'skymapper.txt'],
    button_text='Get SkyMapper lightcurve',
    form_fields={
        'skymapper_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': SKYMAPPER_SR,
            'required': False,
        },
        'skymapper_quality': quality_field({
            QUALITY_STANDARD: 'What the survey averages into its catalogue',
            QUALITY_RELAXED: 'Drop only saturated or truncated detections',
            QUALITY_PUBLISHED: 'None - every detection as published',
        }),
    },
    help_text='SkyMapper DR4 detections, uvgriz, southern sky, 2014-2021',
    order=34,
    about=(
        "SkyMapper DR4, the southern sky in uvgriz from Siding Spring, 2014 "
        "to 2021: a few tens of epochs per star, every detection behind the "
        "catalogue, served by the Astro Data Lab."),
    about_links=[
        ('SkyMapper DR4, Onken et al. 2024', 'https://arxiv.org/abs/2402.02015'),
        ('SkyMapper DR4 at Astro Data Lab', 'https://datalab.noirlab.edu/data/skymapper'),
    ],
    acknowledgement=(
        "SkyMapper is owned and operated by The Australian National "
        "University's Research School of Astronomy and Astrophysics. The "
        "SkyMapper Southern Survey has been funded in part through ARC LIEF "
        "grant LE130100104 from the Australian Research Council, awarded to "
        "the University of Sydney, the Australian National University, "
        "Swinburne University of Technology, the University of Queensland, "
        "the University of Western Australia, the University of Melbourne, "
        "Curtin University of Technology, Monash University and the "
        "Australian Astronomical Observatory. The SkyMapper Southern Survey "
        "dataset has been produced with the support of the National "
        "Computational Infrastructure (NCI) in Canberra, Australia. This "
        "research uses services or data provided by the Astro Data Lab, which "
        "is part of the Community Science and Data Center (CSDC) Program of "
        "NSF NOIRLab. NOIRLab is operated by the Association of Universities "
        "for Research in Astronomy (AURA), Inc. under a cooperative agreement "
        "with the U.S. National Science Foundation. Cite Onken et al. (2024) "
        "for DR4."),
    # Lightcurve metadata
    votable_file='skymapper.vot',
    lc_bands=[
        surveys.band(band, 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value=band,
                     color=SKYMAPPER_COLORS[band],
                     note='as measured by SkyMapper, AB')
        for band in SKYMAPPER_BANDS
    ] + [
        surveys.band('g (from g)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     filter_column='filter', filter_value='g',
                     color='#98df8a',
                     note="g put on the Pan-STARRS scale with the star's own "
                          "mean colours",
                     combined=True),
        surveys.band('g (from r)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     filter_column='filter', filter_value='r',
                     color='#ff9896',
                     note="r put on the Pan-STARRS g scale with the star's own "
                          "mean colours",
                     combined=True),
    ],
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color=SKYMAPPER_COLORS['g'],
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata
    template_layout='with_cutout',
    show_cutout=True,
    cutout_hips='CDS/P/Skymapper/DR4/color',
    requires_coordinates=True,
)
def target_skymapper(config, basepath=None, verbose=True, show=False):
    """Acquire SkyMapper DR4 lightcurve."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('skymapper'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = float(config.get('skymapper_sr') or SKYMAPPER_SR)

    # Not skymapper_: that prefix is the info step's catalogue query
    with cached_votable_query(f"skymapper_lc_{ra:.5f}_{dec:.5f}_{sr:.1f}.vot",
                              basepath, log, 'SkyMapper DR4',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            table = _fetch(ra, dec, sr, log)

            if table is None or not len(table):
                cache.save_empty()
            else:
                cache.save(table)
        else:
            table = cache.data

    if table is None or not len(table):
        log(f"Warning: No SkyMapper detections within {sr:.1f} arcsec - "
            f"SkyMapper covers the southern sky, to about 20 mag")
        return

    log(f"\nSkyMapper DR4: {len(table)} detections")

    quality = quality_level(config, 'skymapper')
    table = table[_judge(table, quality, log)]

    if quality != QUALITY_PUBLISHED and len(table):
        clip = clip_noisy_points(table['mag'], table['magerr'],
                                 np.asarray(table['filter'], dtype=str),
                                 log=log, group_name='band',
                                 ratio=CLIP_RATIO_BY_LEVEL[quality])
        table = table[~clip]

    log(f"  {len(table)} detections left ({quality} filtering)")

    if not len(table):
        log("Warning: No SkyMapper detections left at this level of filtering")
        return

    table.sort('mjd')
    bands = np.asarray(table['filter'], dtype=str)

    # Onto the common g scale, through the star's own colours
    mean = {band: float(table[f'mean_{band}'][0]) for band in ('g', 'r', 'i')}
    table['mag_g'] = np.nan
    idx_g = bands == 'g'
    idx_r = bands == 'r'

    if all(np.isfinite(_) for _ in mean.values()):
        g_minus_r = mean['g'] - mean['r']
        r_minus_i = mean['r'] - mean['i']
        dg = float(skymapper_to_ps1(SKYMAPPER_G_TO_PS1_G, g_minus_r, r_minus_i))
        dr = float(skymapper_to_ps1(SKYMAPPER_R_TO_PS1_R, g_minus_r, r_minus_i))

        # r reaches g by its PS1 counterpart plus the PS1 colour, which the
        # same two relations give: g_PS1 - r_PS1 = (g - r) + dg - dr
        table['mag_g'][idx_g] = table['mag'][idx_g] + dg
        table['mag_g'][idx_r] = table['mag'][idx_r] + dr + (g_minus_r + dg - dr)

        log_conversion(
            log, 'SkyMapper', SKYMAPPER_TO_PS1_FORMULA,
            {'(g - r)': (g_minus_r, "the star's mean SkyMapper colour"),
             '(r - i)': (r_minus_i, "the star's mean SkyMapper colour"),
             'g_PS1 - g_SM': dg,
             'r_PS1 - r_SM': dr,
             '(g - r)_PS1': (g_minus_r + dg - dr, 'from the two above')},
            npoints=int(np.sum(idx_g | idx_r)),
            note='r is carried to g by the constant colour, so a variable '
                 'changing colour as it varies is mis-scaled there',
        )
    else:
        log_conversion(
            log, 'SkyMapper',
            'no conversion applied - each band is published as measured',
            {'system': ('AB', 'the SkyMapper photometric system')},
            npoints=len(table),
            note='not put on the common g scale: the catalogue has no mean '
                 'g, r and i for this star to take its colours from',
        )

    log_bands(log, 'SkyMapper', [
        {'label': band, 'kind': 'native', 'npoints': int(np.sum(bands == band)),
         'note': 'as measured by SkyMapper, AB'}
        for band in SKYMAPPER_BANDS if np.any(bands == band)
    ] + [
        {'label': f'g (from {band})', 'kind': 'derived',
         'npoints': int(np.sum(np.isfinite(np.asarray(table['mag_g'], dtype=float))
                               & (bands == band))),
         'note': "on the Pan-STARRS g scale, with the star's own mean colours"}
        for band in ('g', 'r')
        if np.any(np.isfinite(np.asarray(table['mag_g'], dtype=float)) & (bands == band))
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'skymapper_lc.png'),
                            figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        when = Time(np.asarray(table['mjd'], dtype=float), format='mjd').datetime

        for band in SKYMAPPER_BANDS:
            idx = bands == band

            if np.any(idx):
                plot_with_errors(ax, when[idx], table['mag'][idx],
                                 table['magerr'][idx], label=band,
                                 color=SKYMAPPER_COLORS[band])

        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        ax.legend()
        ax.set_ylabel('Magnitude (AB)')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - SkyMapper DR4")

    log("SkyMapper lightcurve plot saved to file:skymapper_lc.png")

    table.write(os.path.join(basepath, 'skymapper.vot'), format='votable', overwrite=True)
    table.write(os.path.join(basepath, 'skymapper.txt'),
                format='ascii.commented_header', overwrite=True)
    log("SkyMapper data written to file:skymapper.vot")
    log("SkyMapper data written to file:skymapper.txt")
