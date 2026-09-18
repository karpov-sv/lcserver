"""NOIRLab Source Catalog and DELVE-MC lightcurve acquisition module.

Every public exposure taken with DECam on the Blanco telescope, and with the
two northern cameras NOIRLab ran alongside it, has been measured and published
epoch by epoch at the Astro Data Lab, twice over where it matters:

- NSC DR2, the NOIRLab Source Catalog (Nidever et al. 2021): 68 billion
  measurements from DECam, Mosaic3 on the Mayall and 90Prime on the Bok, 2012
  to 2019, over some 35000 square degrees - most of the southern sky and the
  DESI footprint in the north. Source Extractor aperture photometry, each
  exposure calibrated on its own, in ugrizY and the wide VR.
- DELVE-MC, the Magellanic Clouds part of the DECam Local Volume Exploration
  survey's third release: DAOPHOT PSF photometry of the Clouds and their
  periphery, forced at the position of every object in every exposure, 2014
  to 2020.

Both are measurements of the same DECam exposures over the Clouds - the MJDs of
an exposure the two share agree to a microsecond - so they are not stacked
but merged: DELVE-MC's measurement of an exposure is taken wherever it has
one, being PSF photometry in the most crowded fields there are and running a
year later, and NSC fills in the exposures it alone has, from before DELVE-MC
began. Aperture and PSF magnitudes of the same star differ by several
hundredths, the aperture taking in some of the neighbours, so those NSC epochs
are first moved onto DELVE-MC's scale by the median difference over the
exposures the two share - measured for this star, band by band. A band in
which they share too few to measure it keeps whichever has more epochs there.

The bright limit is the camera's: a 90 s DECam exposure saturates at about
15-16 mag, which the flags say of each measurement. The magnitudes are AB on
each camera's own system; there is no relation here from DECam's g to the
Pan-STARRS g the combined curve is drawn on, so none of them reaches it.
"""

import os

import numpy as np

from astropy.table import Table, vstack
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


# Matching radius, in arcsec. DECam pixels are 0.26 arcsec and the seeing about
# one, and in the Clouds the next star is often little further away than that,
# so the nearest object within this is taken, not all of them.
NSC_SR = 1.0

# The bands, blue to red, VR being NSC's wide band
NSC_BANDS = ('u', 'g', 'r', 'i', 'z', 'Y', 'VR')
NSC_COLORS = {'u': '#9467bd', 'g': '#2ca02c', 'r': '#d62728', 'i': '#ff7f0e',
              'z': '#8c564b', 'Y': '#7f7f7f', 'VR': '#17becf'}

# Two measurements of one exposure agree in MJD to a microsecond, and two
# exposures are a minute apart at the least, so a second tells them apart
NSC_SAME_EXPOSURE = 1.0 / 86400

# How many exposures the two catalogues must share in a band before the offset
# between them is trusted to move NSC's other epochs onto DELVE-MC's scale
NSC_MIN_SHARED = 3

# NSC ---------------------------------------------------------------------
#
# Source Extractor flags: 1 - a neighbour or bad pixels in the aperture, 2 -
# deblended from one, 4 - saturated, 8 - truncated at the image edge, 16 and
# above - aperture or isophote data incomplete, or the extraction overflowing.
# The standard level keeps only the first two, which in any crowded field are
# most of the measurements there are; the relaxed level drops only what cannot
# be a measurement of the star's whole light, saturated or cut off.
NSC_FLAGS_STANDARD = ~np.int64(1 | 2)
NSC_FLAGS_RELAXED = np.int64(4 | 8)

# DELVE-MC ----------------------------------------------------------------
#
# No flags, but DAOPHOT's own judgement of each fit: chi near 1 for a star the
# PSF describes, sharp near 0 for a point source. A forced measurement landing
# on a star that is not there, or on the wing of a brighter one, shows as chi
# of tens. The relaxed level drops only fits that failed outright.
DELVE_CHI_STANDARD = 3.0
DELVE_SHARP_STANDARD = 1.0
DELVE_CHI_RELAXED = 10.0


def _nearest(table, ra, dec):
    """The row of a cone search nearest the target, and how far it is."""
    dist = SkyCoord(ra, dec, unit='deg').separation(
        SkyCoord(np.asarray(table['ra'], dtype=float),
                 np.asarray(table['dec'], dtype=float), unit='deg')).arcsec
    i = int(np.argmin(dist))

    return table[i], float(dist[i]), len(table)


def _column(table, name, dtype, fill):
    return np.ma.filled(np.ma.asarray(table[name], dtype=dtype), fill)


def _nsc(ra, dec, sr, log):
    """The NSC DR2 epochs of the nearest object, or None."""
    found = datalab_query(f"SELECT id, ra, dec FROM nsc_dr2.object WHERE "
                          + datalab_cone('ra', 'dec', ra, dec, sr),
                          'the NSC object catalogue')

    if not len(found):
        return None

    row, dist, n = _nearest(found, ra, dec)
    oid = str(row['id']).strip()

    log(f"  NSC: object {oid}, {dist:.2f} arcsec away"
        + (f", the nearest of {n}" if n > 1 else ""))

    lc = datalab_query(f"SELECT m.mjd, m.filter, m.mag_auto, m.magerr_auto,"
                       f" m.flags, e.instrument FROM nsc_dr2.meas AS m"
                       f" JOIN nsc_dr2.exposure AS e ON m.exposure = e.exposure"
                       f" WHERE m.objectid = '{oid}'",
                       'the NSC measurements')

    if not len(lc):
        return None

    return Table({
        'survey': np.full(len(lc), 'NSC'),
        'source': np.full(len(lc), oid),
        'instrument': np.char.strip(np.asarray(lc['instrument'], dtype=str)),
        'mjd': _column(lc, 'mjd', float, np.nan),
        'filter': np.char.strip(np.asarray(lc['filter'], dtype=str)),
        'mag': _column(lc, 'mag_auto', float, np.nan),
        'magerr': _column(lc, 'magerr_auto', float, np.nan),
        'flags': _column(lc, 'flags', np.int64, -1),
    })


def _delve(ra, dec, sr, log):
    """The DELVE-MC epochs of the nearest object, or None."""
    found = datalab_query(f"SELECT objid, ra, dec FROM delve_dr3.mc_object WHERE "
                          + datalab_cone('ra', 'dec', ra, dec, sr),
                          'the DELVE-MC object catalogue')

    if not len(found):
        return None

    row, dist, n = _nearest(found, ra, dec)
    oid = str(row['objid']).strip()

    log(f"  DELVE-MC: object {oid}, {dist:.2f} arcsec away"
        + (f", the nearest of {n}" if n > 1 else ""))

    lc = datalab_query(f"SELECT mjd, filter, mag, err, forced, chi, sharp"
                       f" FROM delve_dr3.mc_meas WHERE objid = '{oid}'",
                       'the DELVE-MC measurements')

    if not len(lc):
        return None

    return Table({
        'survey': np.full(len(lc), 'DELVE-MC'),
        'source': np.full(len(lc), oid),
        'instrument': np.full(len(lc), 'c4d'),
        'mjd': _column(lc, 'mjd', float, np.nan),
        'filter': np.char.strip(np.asarray(lc['filter'], dtype=str)),
        'mag': _column(lc, 'mag', float, np.nan),
        'magerr': _column(lc, 'err', float, np.nan),
        'forced': _column(lc, 'forced', int, -1),
        'chi': _column(lc, 'chi', float, np.nan),
        'sharp': _column(lc, 'sharp', float, np.nan),
    })


# What each catalogue is asked with, in the order they are asked, and the
# image of the sky shown beside a light curve it answered for. The Legacy
# Surveys' colour HiPS is built from the same DECam exposures and those of the
# two northern cameras, but breaks up in the Clouds, so a DELVE-MC target is
# shown in SkyMapper's.
NSC_SURVEYS = [
    ('delve', 'DELVE-MC', _delve,
     'CDS/P/Skymapper/DR4/color', 'SkyMapper DR4'),
    ('nsc', 'NSC DR2', _nsc,
     'CDS/P/DESI-Legacy-Surveys/DR10/color', 'Legacy Surveys DR10'),
]


def _usable(table):
    """Epochs with a magnitude and an error that mean something."""
    mag = np.asarray(table['mag'], dtype=float)
    err = np.asarray(table['magerr'], dtype=float)

    # NSC writes 99.99 for an aperture it could not measure
    return (np.isfinite(mag) & np.isfinite(err) & (mag > 0) & (mag < 50)
            & (err > 0) & (err < 1.0))


def _quality(key, table, quality):
    """Which epochs the level keeps, and what each criterion removed."""
    criteria = []

    if key == 'nsc':
        flags = np.asarray(table['flags'], dtype=np.int64)

        if quality == QUALITY_STANDARD:
            criteria.append(((flags >= 0) & ((flags & NSC_FLAGS_STANDARD) == 0),
                             'flagged beyond a neighbour or deblending'))
        elif quality == QUALITY_RELAXED:
            criteria.append(((flags >= 0) & ((flags & NSC_FLAGS_RELAXED) == 0),
                             'saturated or truncated at the image edge'))
    else:
        chi = np.asarray(table['chi'], dtype=float)
        sharp = np.asarray(table['sharp'], dtype=float)

        if quality == QUALITY_STANDARD:
            criteria.append((chi < DELVE_CHI_STANDARD,
                             f'PSF fit chi above {DELVE_CHI_STANDARD:g}'))
            criteria.append((np.abs(sharp) < DELVE_SHARP_STANDARD,
                             f'|sharp| above {DELVE_SHARP_STANDARD:g}, '
                             f'not a point source'))
        elif quality == QUALITY_RELAXED:
            criteria.append((chi < DELVE_CHI_RELAXED,
                             f'PSF fit chi above {DELVE_CHI_RELAXED:g}'))

    return criteria


def _judge(key, table, quality, log):
    """The mask of epochs to keep, with every removal logged."""
    keep = _usable(table)

    if np.any(~keep):
        log(f"    {int(np.sum(~keep)):5d} removed - no usable magnitude or error")

    if quality == QUALITY_PUBLISHED:
        return keep

    for passed, meaning in _quality(key, table, quality):
        removed = int(np.sum(keep & ~passed))

        if removed:
            log(f"    {removed:5d} removed - {meaning}")

        keep &= passed

    return keep


def _merge(delve, delve_good, nsc, nsc_good, log):
    """NSC epochs DELVE-MC lacks, on DELVE-MC's scale, and which to keep.

    Returns NSC with the epochs it alone has shifted, and the keep masks of
    both - NSC's without the exposures DELVE-MC measured too. A band sharing
    too few exposures to measure the offset keeps whichever catalogue has
    more epochs in it, and drops the other's.
    """
    log("\nMerging NSC into DELVE-MC:")

    nsc = nsc.copy()
    nsc_mjd = np.asarray(nsc['mjd'], dtype=float)
    nsc_filter = np.asarray(nsc['filter'], dtype=str)
    delve_mjd = np.asarray(delve['mjd'], dtype=float)
    delve_filter = np.asarray(delve['filter'], dtype=str)

    # For every NSC epoch, the DELVE-MC measurement of the same exposure, if any
    twin = np.full(len(nsc), -1)

    for band in np.unique(nsc_filter):
        idx_n = np.where(nsc_filter == band)[0]
        idx_d = np.where(delve_filter == band)[0]

        if not len(idx_d):
            continue

        order = idx_d[np.argsort(delve_mjd[idx_d])]
        sorted_mjd = delve_mjd[order]
        pos = np.clip(np.searchsorted(sorted_mjd, nsc_mjd[idx_n]), 1, len(order)) - 1

        for i, p in zip(idx_n, pos):
            for q in (p, min(p + 1, len(order) - 1)):
                if abs(sorted_mjd[q] - nsc_mjd[i]) < NSC_SAME_EXPOSURE:
                    twin[i] = order[q]

    shared = twin >= 0
    log(f"  {int(np.sum(shared))} of {len(nsc)} NSC epochs are exposures "
        f"DELVE-MC measured too, and are left to it")

    keep = nsc_good & ~shared
    delve_keep = delve_good.copy()
    offsets = {}

    for band in NSC_BANDS:
        own = keep & (nsc_filter == band)
        theirs = delve_keep & (delve_filter == band)

        # With nothing of DELVE-MC's in the band there is no scale to disagree
        # with, and NSC's epochs stand as they are
        if not np.any(own) or not np.any(theirs):
            continue

        pairs = shared & nsc_good & (nsc_filter == band)
        pairs[pairs] &= delve_good[twin[pairs]]
        npairs = int(np.sum(pairs))

        if npairs < NSC_MIN_SHARED:
            why = (f"{npairs} exposures shared, too few to tie the two "
                   f"together" if npairs else "no exposures shared to tie "
                   f"the two together")

            if np.sum(theirs) >= np.sum(own):
                log(f"  {band}: NSC's {int(np.sum(own))} epochs dropped for "
                    f"DELVE-MC's {int(np.sum(theirs))} - {why}")
                keep &= ~own
            else:
                log(f"  {band}: DELVE-MC's {int(np.sum(theirs))} epochs dropped "
                    f"for NSC's {int(np.sum(own))} - {why}")
                delve_keep &= ~theirs

            continue

        diff = (np.asarray(nsc['mag'], dtype=float)[pairs]
                - np.asarray(delve['mag'], dtype=float)[twin[pairs]])
        offset = float(np.median(diff))
        offsets[band] = (offset, npairs, int(np.sum(own)))

        nsc['mag'][own] -= offset

    if offsets:
        log_conversion(
            log, 'NSC', 'm = m_NSC - median(m_NSC - m_DELVE-MC), per band',
            {f"offset {band}": (offset, f"median over {n} shared exposures")
             for band, (offset, n, _) in offsets.items()},
            npoints=sum(_[2] for _ in offsets.values()),
            note='aperture magnitudes moved onto the PSF scale of the '
                 'catalogue with more of the epochs',
            heading=False)

    return nsc, keep, delve_keep


@survey_source(
    name='NOIRLab Source Catalog (NSC DR2, DELVE-MC)',
    short_name='NSC',
    state_acquiring='acquiring NSC lightcurve',
    state_acquired='NSC lightcurve acquired',
    log_file='nsc.log',
    output_files=['nsc.log', 'nsc_lc.png', 'nsc.vot', 'nsc.txt'],
    button_text='Get NSC lightcurve',
    form_fields={
        'nsc_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': NSC_SR,
            'required': False,
        },
        'nsc_quality': quality_field({
            QUALITY_STANDARD: 'Clean detections and good PSF fits',
            QUALITY_RELAXED: 'Drop only saturated, truncated or failed measurements',
            QUALITY_PUBLISHED: 'None - every epoch as published',
        }),
    },
    help_text='Every public DECam, Mosaic3 and 90Prime exposure, measured: '
              'NSC DR2 and, in the Magellanic Clouds, DELVE-MC, ugrizY, 2012-2020',
    order=33,
    about=(
        "Two catalogues of every measurement in the public exposures of "
        "NOIRLab's wide-field cameras, served by the Astro Data Lab: NSC DR2, "
        "the NOIRLab Source Catalog, over most of the southern sky and the "
        "DESI footprint in the north, and DELVE-MC, PSF photometry of the "
        "Magellanic Clouds - deep, sparse light curves in ugrizY for stars "
        "fainter than about 15 mag."),
    about_links=[
        ('NSC DR2, Nidever et al. 2021',
         'https://ui.adsabs.harvard.edu/abs/2021AJ....161..192N/abstract'),
        ('DELVE at Astro Data Lab', 'https://datalab.noirlab.edu/data/delve'),
        ('Astro Data Lab', 'https://datalab.noirlab.edu/'),
    ],
    acknowledgement=(
        "This research uses services or data provided by the Astro Data Lab, "
        "which is part of the Community Science and Data Center (CSDC) Program "
        "of NSF NOIRLab. NOIRLab is operated by the Association of "
        "Universities for Research in Astronomy (AURA), Inc. under a "
        "cooperative agreement with the U.S. National Science Foundation. "
        "The DECam Local Volume Exploration Survey (DELVE; Proposal ID: "
        "2019A-0305, PI: Drlica-Wagner) is partially supported by Fermilab "
        "LDRD (L2019-011), the NASA Fermi Guest Investigator Program (Cycle 9 "
        "No. 91201), the NSF AAG (AST-2108168, AST-2108169, AST-2307126, "
        "AST-2407526), and the NSF-Simonyi Scholars program. Cite Nidever et "
        "al. (2021), AJ 161, 192, for NSC DR2."),
    # Lightcurve metadata
    votable_file='nsc.vot',
    lc_bands=[
        surveys.band(band, 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value=band,
                     color=NSC_COLORS[band],
                     note='as calibrated by the survey, AB')
        for band in NSC_BANDS
    ],
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color=NSC_COLORS['g'],
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata - the Legacy Surveys' colour HiPS until a run has
    # found which catalogue answers, and SkyMapper's then in the Clouds, see
    # above
    template_layout='with_cutout',
    show_cutout=True,
    cutout_hips='CDS/P/DESI-Legacy-Surveys/DR10/color',
    cutout_name='Legacy Surveys DR10',
    requires_coordinates=True,
)
def target_nsc(config, basepath=None, verbose=True, show=False):
    """Acquire NSC DR2 and DELVE-MC lightcurves."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('nsc'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = float(config.get('nsc_sr') or NSC_SR)

    # The cutout is chosen by the catalogue that answered, so what an earlier
    # run chose is dropped until this one has found out
    config.pop('nsc_cutout_hips', None)
    config.pop('nsc_cutout_name', None)

    parts = {}

    # Each catalogue is cached on its own: one answering nothing says nothing
    # about the other
    for key, name, fetch, _, _ in NSC_SURVEYS:
        with cached_votable_query(f"nsc_{key}_{ra:.5f}_{dec:.5f}_{sr:.1f}.vot",
                                  basepath, log, name,
                                  refresh=refresh_cache) as cache:
            if not cache.hit:
                table = fetch(ra, dec, sr, log)

                if table is None or not len(table):
                    cache.save_empty()
                    log(f"  {name}: nothing within {sr:.1f} arcsec")
                    continue

                cache.save(table)

            if cache.data is not None and len(cache.data):
                parts[key] = cache.data

    if not parts:
        log("Warning: No NSC or DELVE-MC measurements at this position - NSC "
            "covers most of the southern sky and the DESI footprint in the "
            "north, DELVE-MC the Magellanic Clouds, and neither has stars "
            "brighter than about 15 mag")
        return

    # The first catalogue with anything here names the image shown beside it
    for key, name, _, hips, hips_name in NSC_SURVEYS:
        if key in parts:
            config['nsc_cutout_hips'] = hips
            config['nsc_cutout_name'] = hips_name
            break

    quality = quality_level(config, 'nsc')
    good = {}

    for key, name, _, _, _ in NSC_SURVEYS:
        if key in parts:
            log(f"\n{name}: {len(parts[key])} epochs")
            good[key] = _judge(key, parts[key], quality, log)

    if 'delve' in parts and 'nsc' in parts:
        parts['nsc'], good['nsc'], good['delve'] = _merge(
            parts['delve'], good['delve'], parts['nsc'], good['nsc'], log)

    columns = ['survey', 'source', 'instrument', 'mjd', 'filter', 'mag', 'magerr']
    kept = [parts[key][good[key]][columns] for key in parts if np.any(good[key])]

    if not kept:
        log("Warning: No NSC or DELVE-MC epochs left at this level of filtering")
        return

    nsc = vstack(kept)

    # Where there are no flags to go on, and as a last check where there are,
    # the errors themselves
    if quality != QUALITY_PUBLISHED:
        clip = clip_noisy_points(nsc['mag'], nsc['magerr'],
                                 np.asarray(nsc['filter'], dtype=str),
                                 log=log, group_name='band',
                                 ratio=CLIP_RATIO_BY_LEVEL[quality])
        nsc = nsc[~clip]

    log(f"\n{len(nsc)} epochs left ({quality} filtering)")

    if not len(nsc):
        log("Warning: No NSC or DELVE-MC epochs left at this level of filtering")
        return

    nsc.sort('mjd')

    bands = np.asarray(nsc['filter'], dtype=str)

    log_conversion(
        log, 'NSC',
        'no conversion applied - each band is published as calibrated',
        {'system': ('AB', "each camera's own bands, calibrated per exposure")},
        npoints=len(nsc),
        note="not put on the common g scale: there is no relation here from "
             "the DECam bands to Pan-STARRS g",
    )

    log_bands(log, 'NSC', [
        {'label': band, 'kind': 'native', 'npoints': int(np.sum(bands == band)),
         'note': 'as calibrated by the survey, AB'}
        for band in NSC_BANDS if np.any(bands == band)
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'nsc_lc.png'),
                            figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        when = Time(np.asarray(nsc['mjd'], dtype=float), format='mjd').datetime

        for band in NSC_BANDS:
            idx = bands == band

            if np.any(idx):
                plot_with_errors(ax, when[idx], nsc['mag'][idx],
                                 nsc['magerr'][idx], label=band,
                                 color=NSC_COLORS[band])

        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        ax.legend()
        ax.set_ylabel('Magnitude (AB)')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - "
                     f"{' and '.join(sorted(set(np.asarray(nsc['survey'], dtype=str))))}")

    log("NSC lightcurve plot saved to file:nsc_lc.png")

    nsc.write(os.path.join(basepath, 'nsc.vot'), format='votable', overwrite=True)
    nsc.write(os.path.join(basepath, 'nsc.txt'),
              format='ascii.commented_header', overwrite=True)
    log("NSC data written to file:nsc.vot")
    log("NSC data written to file:nsc.txt")
