"""Palomar Gattini-IR lightcurve acquisition module.

Palomar Gattini-IR is a 30 cm telescope with a wide-field J-band camera,
surveying the sky visible from Palomar every night or two since 2018. Its
first release (Murakawa et al. 2024) is the J light curves of the 2MASS
sources brighter than J = 15.5 in its footprint, over its first four years:
hundreds of epochs a star, PSF photometry of the stack of each visit, served
by the Astro Data Lab.

What it is good for is set by its depth and its pixels. Photometry is good to
10 per cent only down to J = 13 in the Galactic plane and 15.5 away from it,
and the images are coarse enough that a neighbour within the PSF is common -
both flagged per epoch. The bright limit is saturation, J = 8.5 until 2020 and
about 6 after the readout was improved. So it is the survey for the red,
bright and slow: Miras, dusty and embedded stars, anything that varies over
weeks in the infrared.

Two things in the release are coarser than they look. The time of an epoch,
the Julian date the exposure began, is stored as a 32-bit float, which at
2.46 million days resolves only a quarter of a day - enough for the variables
the survey is for, not for anything periodic within a night. And there are
two errors, one from the pixel statistics, one from placing the PSF at random
blank positions in the image; the second, which takes in the confusion from
neighbours, is the one used here, the first being far smaller than the
scatter of a constant star.

A visit that did not detect the star leaves its magnitude empty and gives the
3 sigma limit instead. Those are not drawn. The band is 2MASS-calibrated J,
Vega, and none of it reaches the combined curve.
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
                    CLIP_RATIO_BY_LEVEL,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


# Matching radius, in arcsec, against the 2MASS position the release keys each
# source to - an epoch around 2000, so a fast-moving star has moved since
PGIR_SR = 2.0

PGIR_COLOR = '#2ca02c'

# The per-epoch flags, as Murakawa et al. (2024) define them:
#   1 - taken on the west side of the meridian
#   2 - brighter than the saturation magnitude of the exposure
#   4 - outside the magnitude range the photometry is reliable in
#   8 - airmass above 2
#  16 - zero point more than 0.75 mag off the median of its sub-quadrant
#  32 - a source in the PSF footprint within 2 mag of this one's brightness
# The paper recommends dropping 2, 4 and 32, which is the standard level; the
# zero point is dropped by both levels, as a magnitude that far off calibration
# says nothing about the star, and the relaxed level drops no more than it and
# saturation.
PGIR_FLAG_NAMES = {2: 'saturated', 4: 'outside the reliable magnitude range',
                   16: 'zero point more than 0.75 mag off',
                   32: 'a comparably bright neighbour in the PSF'}
PGIR_FLAGS_STANDARD = (2, 4, 16, 32)
PGIR_FLAGS_RELAXED = (2, 16)


def _column(table, name, dtype, fill):
    return np.ma.filled(np.ma.asarray(table[name], dtype=dtype), fill)


def _fetch(ra, dec, sr, log):
    """The epochs of the nearest 2MASS source, or None."""
    found = datalab_query("SELECT pts_key, tmcra, tmcdec, tmcjmag FROM "
                          "pgir_dr1.sources WHERE "
                          + datalab_cone('tmcra', 'tmcdec', ra, dec, sr),
                          'the Gattini-IR source catalogue')

    if not len(found):
        return None

    dist = SkyCoord(ra, dec, unit='deg').separation(
        SkyCoord(np.asarray(found['tmcra'], dtype=float),
                 np.asarray(found['tmcdec'], dtype=float), unit='deg')).arcsec
    i = int(np.argmin(dist))
    key = int(found['pts_key'][i])

    log(f"  Gattini-IR: 2MASS source {key}, J = {float(found['tmcjmag'][i]):.2f}, "
        f"{dist[i]:.2f} arcsec away"
        + (f", the nearest of {len(found)}" if len(found) > 1 else ""))

    lc = datalab_query(f"SELECT obsjd, magpsf, magpsferr, magpsfstaterr, "
                       f"magpsflim, flags FROM pgir_dr1.photometry "
                       f"WHERE pts_key = {key}",
                       'the Gattini-IR photometry')

    if not len(lc):
        return None

    return Table({
        'source': np.full(len(lc), str(key)),
        # Float32 in the release, a quarter of a day apart at this size;
        # widened before the subtraction, so that loses nothing more
        'mjd': _column(lc, 'obsjd', float, np.nan) - 2400000.5,
        'mag': _column(lc, 'magpsf', float, np.nan),
        'magerr': _column(lc, 'magpsferr', float, np.nan),
        'magerr_stat': _column(lc, 'magpsfstaterr', float, np.nan),
        'maglim': _column(lc, 'magpsflim', float, np.nan),
        'flags': _column(lc, 'flags', int, -1),
    })


def _judge(table, quality, log):
    """The mask of epochs to keep, with every removal logged."""
    mag = np.asarray(table['mag'], dtype=float)
    err = np.asarray(table['magerr'], dtype=float)

    detected = np.isfinite(mag)

    if np.any(~detected):
        log(f"    {int(np.sum(~detected)):5d} removed - not detected, only a limit")

    # No ceiling on the error, as other sources have: in a crowded field the
    # confusion error runs to a magnitude and more for a star whose scatter is
    # half that, and whether such an epoch is worth anything is what the
    # reliable-range and neighbour flags are for
    keep = detected & np.isfinite(err) & (err > 0)

    if np.any(detected & ~keep):
        log(f"    {int(np.sum(detected & ~keep)):5d} removed - no usable error")

    if quality == QUALITY_PUBLISHED:
        return keep

    flags = np.asarray(table['flags'], dtype=int)
    bits = PGIR_FLAGS_STANDARD if quality == QUALITY_STANDARD else PGIR_FLAGS_RELAXED

    for bit in bits:
        passed = (flags >= 0) & ((flags & bit) == 0)
        removed = int(np.sum(keep & ~passed))

        if removed:
            log(f"    {removed:5d} removed - {PGIR_FLAG_NAMES[bit]}")

        keep &= passed

    return keep


@survey_source(
    name='Palomar Gattini-IR',
    short_name='PGIR',
    state_acquiring='acquiring Gattini-IR lightcurve',
    state_acquired='Gattini-IR lightcurve acquired',
    log_file='pgir.log',
    output_files=['pgir.log', 'pgir_lc.png', 'pgir.vot', 'pgir.txt'],
    button_text='Get Gattini-IR lightcurve',
    form_fields={
        'pgir_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': PGIR_SR,
            'required': False,
        },
        'pgir_quality': quality_field({
            QUALITY_STANDARD: 'What the release recommends',
            QUALITY_RELAXED: 'Drop only saturated or miscalibrated epochs',
            QUALITY_PUBLISHED: 'None - every detection as published',
        }),
    },
    help_text='Palomar Gattini-IR DR1, J band, the sky visible from Palomar, '
              '2018-2022, J < 15.5',
    order=35,
    about=(
        "Palomar Gattini-IR, a wide-field J-band survey of the sky visible "
        "from Palomar every night or two: the J light curves of its first four "
        "years for 2MASS sources brighter than J = 15.5, served by the Astro "
        "Data Lab."),
    about_links=[
        ('PGIR DR1, Murakawa et al. 2024', 'https://arxiv.org/abs/2406.01720'),
        ('Palomar Gattini-IR, De et al. 2020',
         'https://ui.adsabs.harvard.edu/abs/2020PASP..132b5001D/abstract'),
        ('PGIR at Astro Data Lab', 'https://datalab.noirlab.edu/data/pgir'),
    ],
    # The release asks for no acknowledgement of its own; Data Lab does
    acknowledgement=(
        "This research uses services or data provided by the Astro Data Lab, "
        "which is part of the Community Science and Data Center (CSDC) Program "
        "of NSF NOIRLab. NOIRLab is operated by the Association of "
        "Universities for Research in Astronomy (AURA), Inc. under a "
        "cooperative agreement with the U.S. National Science Foundation. Cite "
        "Murakawa et al. (2024) for the light curves and De et al. (2020) for "
        "the survey."),
    # Lightcurve metadata
    votable_file='pgir.vot',
    lc_bands=[
        surveys.band('J', 'mag', 'magerr', surveys.BAND_NATIVE,
                     color=PGIR_COLOR, note='calibrated on 2MASS, Vega'),
    ],
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_color=PGIR_COLOR,
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata
    template_layout='with_cutout',
    show_cutout=True,
    cutout_hips='CDS/P/2MASS/color',
    requires_coordinates=True,
)
def target_pgir(config, basepath=None, verbose=True, show=False):
    """Acquire Palomar Gattini-IR lightcurve."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('pgir'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = float(config.get('pgir_sr') or PGIR_SR)

    with cached_votable_query(f"pgir_{ra:.5f}_{dec:.5f}_{sr:.1f}.vot",
                              basepath, log, 'Palomar Gattini-IR',
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
        log(f"Warning: No Gattini-IR light curve within {sr:.1f} arcsec - it "
            f"covers the sky visible from Palomar, for 2MASS sources brighter "
            f"than J = 15.5")
        return

    log(f"\nGattini-IR DR1: {len(table)} visits")

    quality = quality_level(config, 'pgir')
    table = table[_judge(table, quality, log)]

    if quality != QUALITY_PUBLISHED and len(table):
        clip = clip_noisy_points(table['mag'], table['magerr'], log=log,
                                 ratio=CLIP_RATIO_BY_LEVEL[quality])
        table = table[~clip]

    log(f"  {len(table)} epochs left ({quality} filtering)")

    if not len(table):
        log("Warning: No Gattini-IR epochs left at this level of filtering")
        return

    table.sort('mjd')

    log_conversion(
        log, 'Gattini-IR',
        'no conversion applied - J is published as measured',
        {'system': ('Vega', 'calibrated on 2MASS J')},
        npoints=len(table),
        note='times are exposure starts stored to a quarter of a day; not put '
             'on the common g scale, J having no route to g',
    )

    log_bands(log, 'Gattini-IR', [
        {'label': 'J', 'kind': 'native', 'npoints': len(table),
         'note': 'calibrated on 2MASS, Vega'},
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'pgir_lc.png'),
                            figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        when = Time(np.asarray(table['mjd'], dtype=float), format='mjd').datetime

        plot_with_errors(ax, when, table['mag'], table['magerr'], label='J',
                         color=PGIR_COLOR)

        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        ax.legend()
        ax.set_ylabel('J (Vega)')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - Palomar Gattini-IR")

    log("Gattini-IR lightcurve plot saved to file:pgir_lc.png")

    table.write(os.path.join(basepath, 'pgir.vot'), format='votable', overwrite=True)
    table.write(os.path.join(basepath, 'pgir.txt'),
                format='ascii.commented_header', overwrite=True)
    log("Gattini-IR data written to file:pgir.vot")
    log("Gattini-IR data written to file:pgir.txt")
