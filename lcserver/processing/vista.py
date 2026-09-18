"""VISTA near-infrared survey lightcurve acquisition module.

Two of ESO's public surveys with the VISTA telescope and its VIRCAM camera
went back to the same fields for years, and both publish every epoch of every
source through ESO's catalogue service:

- VVV, the VISTA Variables in the Via Lactea, over the Galactic bulge and the
  southern disk - some 560 square degrees - as VIRAC2 (Smith et al. 2025):
  PSF photometry from 2010, with the VVVX epochs of 2016-2019 on the same
  fields. Mostly Ks, a hundred epochs and more of it, and a handful in Z, Y,
  J and H.
- VMC, the VISTA survey of the Magellanic Clouds, the Bridge and the Stream
  (Cioni et al. 2011), in its seventh release: a dozen or more Ks epochs from
  2009 on, and fewer in Y and J.

They share a camera, a photometric system and a way of being asked - a cone
search of the source catalogue, then the epochs of the nearest source by its
identifier - and their footprints do not overlap, so they are one source here
with a small adapter each. The fields VVVX added beyond the VVV footprint are
not in VIRAC2 and have no per-source time series at ESO.

The bright limit is the camera's: VIRCAM saturates at about Ks = 11, and a
star brighter than that is either missing from both catalogues or ringed by
spurious detections in its wings. The magnitudes are on the VISTA Vega system,
and no band here has a route to g, so none of them reaches the combined curve.
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
                    clip_noisy_points, quality_field, quality_level,
                    CLIP_RATIO_BY_LEVEL,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


VISTA_TAP = 'https://archive.eso.org/tap_cat/sync'

VISTA_TIMEOUT = 300
VISTA_MAXREC = 1000000

# Matching radius, in arcsec. VIRCAM pixels are 0.34 arcsec and the seeing
# under a second, and in a bulge field the next star is often little further
# away than that, so the nearest source within this is taken, not all of them.
VISTA_SR = 1.0

# The bands, blue to red
VISTA_BANDS = ('Z', 'Y', 'J', 'H', 'Ks')
VISTA_COLORS = {'Z': '#9467bd', 'Y': '#1f77b4', 'J': '#2ca02c',
                'H': '#ff7f0e', 'Ks': '#d62728'}

# VIRAC2 ------------------------------------------------------------------
#
# Smith et al. (2025), section 2.6: the statistics the catalogue publishes are
# computed from detections with a zero photometric flag, not shared with
# another source, and not an astrometric outlier above 5 sigma. That is what
# the standard level keeps. The photometric flag is about the calibration -
# too few reference stars for one of its coefficients, an edge bin of the
# illumination map - rather than about the detection, so the relaxed level
# keeps it and drops only what may be another star altogether: a detection
# the time series of a neighbour also claims, or one far from where the star
# is.
VIRAC_SOURCES = 'VVVX_VIRAC_V2_SOURCES'
VIRAC_LC = 'VVVX_VIRAC_V2_LC'

# ast_res_chisq is a chi-square of two degrees of freedom, and 5 sigma of one
# is this
VIRAC_AST_CHISQ_MAX = 28.74

# VMC ---------------------------------------------------------------------
#
# The per-epoch tables carry a column named for the post-processing error
# bits, but it does not behave as a bitmask: nearly every epoch of a source
# has a value of its own, in the millions, rising with the date of the
# observation - which is what a frame identifier does - and the sixth release
# the same. So nothing is read from it, and the only judgement made on VMC is
# the scatter of its own errors, as for any source with no flags to go on.
VMC_SOURCES = 'vmc_dr7_ksjy_V6'
VMC_LC = {'Y': ('vmc_dr7_mPhotY_V6', 'YMAG', 'YERR'),
          'J': ('vmc_dr7_mPhotJ_V6', 'JMAG', 'JERR'),
          'Ks': ('vmc_dr7_mPhotKs_V6', 'KSMAG', 'KSERR')}


def _tap(query, what):
    """One synchronous query of ESO's catalogue service, as a table."""
    try:
        res = requests.post(VISTA_TAP, timeout=VISTA_TIMEOUT, data={
            'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'votable',
            'MAXREC': VISTA_MAXREC, 'QUERY': query})
    except requests.RequestException as e:
        raise SourceError(f"could not query the ESO catalogue service for "
                          f"{what} - {type(e).__name__}: {e}")

    # A refused query comes back as a 400 with the reason inside a VOTable
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

    if b'value="OVERFLOW"' in res.content:
        raise SourceError(f"the ESO catalogue service returned {what} cut "
                          f"short at {VISTA_MAXREC} rows")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', VOWarning)
            table = Table.read(io.BytesIO(res.content), format='votable')
    except Exception as e:
        raise SourceError(f"could not read {what} from the ESO catalogue "
                          f"service - {type(e).__name__}: {e}")

    # A unit astropy cannot read would be complained about on every cache
    # write and read after this one
    for column in table.itercols():
        if isinstance(column.unit, u.UnrecognizedUnit):
            column.unit = None

    return table


def _nearest(table, ra, dec, ra_col, dec_col):
    """The row of a cone search nearest the target, and how far it is."""
    dist = SkyCoord(ra, dec, unit='deg').separation(
        SkyCoord(np.asarray(table[ra_col], dtype=float),
                 np.asarray(table[dec_col], dtype=float), unit='deg')).arcsec
    i = int(np.argmin(dist))

    return table[i], float(dist[i]), len(table)


def _cone(table, ra_col, dec_col, ra, dec, sr):
    return (f" FROM {table} WHERE CONTAINS(POINT('ICRS', {ra_col}, {dec_col}),"
            f" CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {sr / 3600.0:.9f}))=1")


def _virac(ra, dec, sr, log):
    """The VIRAC2 epochs of the nearest source, or None."""
    found = _tap(f"SELECT sourceid, ra, de"
                 + _cone(VIRAC_SOURCES, 'ra', 'de', ra, dec, sr),
                 'the VIRAC2 source catalogue')

    if not len(found):
        return None

    row, dist, n = _nearest(found, ra, dec, 'ra', 'de')
    sid = int(row['sourceid'])

    log(f"  VVV: VIRAC2 source {sid}, {dist:.2f} arcsec away"
        + (f", the nearest of {n}" if n > 1 else ""))

    lc = _tap(f"SELECT mjdobs, filter, mag, emag, phot_flag, ambiguous_match,"
              f" ast_res_chisq FROM {VIRAC_LC} WHERE sourceid = {sid}",
              'the VIRAC2 time series')

    if not len(lc):
        return None

    def column(name, dtype, fill):
        return np.ma.filled(np.ma.asarray(lc[name], dtype=dtype), fill)

    return Table({
        'survey': np.full(len(lc), 'VVV'),
        'source': np.full(len(lc), str(sid)),
        'mjd': column('mjdobs', float, np.nan),
        'filter': np.asarray(lc['filter'], dtype=str),
        'mag': column('mag', float, np.nan),
        'magerr': column('emag', float, np.nan),
        'phot_flag': column('phot_flag', int, -1),
        'ambiguous': column('ambiguous_match', int, 1),
        'ast_chisq': column('ast_res_chisq', float, np.nan),
    })


def _vmc(ra, dec, sr, log):
    """The VMC epochs of the nearest source, band by band, or None."""
    found = _tap("SELECT SOURCEID, RA2000, DEC2000"
                 + _cone(VMC_SOURCES, 'RA2000', 'DEC2000', ra, dec, sr),
                 'the VMC source catalogue')

    if not len(found):
        return None

    row, dist, n = _nearest(found, ra, dec, 'RA2000', 'DEC2000')
    sid = int(row['SOURCEID'])

    log(f"  VMC: source {sid}, {dist:.2f} arcsec away"
        + (f", the nearest of {n}" if n > 1 else ""))

    parts = []

    for band, (table, mag, err) in VMC_LC.items():
        lc = _tap(f"SELECT MJD, {mag}, {err} FROM {table} WHERE SOURCEID = {sid}",
                  f"the VMC {band} epochs")

        if not len(lc):
            continue

        parts.append(Table({
            'survey': np.full(len(lc), 'VMC'),
            'source': np.full(len(lc), str(sid)),
            'mjd': np.ma.filled(np.ma.asarray(lc['MJD'], dtype=float), np.nan),
            'filter': np.full(len(lc), band),
            'mag': np.ma.filled(np.ma.asarray(lc[mag], dtype=float), np.nan),
            'magerr': np.ma.filled(np.ma.asarray(lc[err], dtype=float), np.nan),
        }))

    return vstack(parts) if parts else None


# What each survey is asked with, in the order they are asked, and the image
# of the sky shown beside its light curve. VVV has a colour HiPS of its fourth
# release, in J, Y and Z - there is none newer, and none with Ks. VMC has no
# HiPS anywhere, at CDS, at ESO or at the survey's own archive, so a VMC
# target is shown in VHS, the VISTA hemisphere survey: the same camera and
# bands, shallower, and covering the whole of the Clouds and the Bridge. 2MASS
# is too shallow to show a VMC star at all, and the Legacy Surveys' colour
# HiPS breaks up in the SMC.
VISTA_SURVEYS = [
    ('vvv', 'VVV (VIRAC2)', _virac,
     'CDS/P/VISTA/VVV/DR4/ColorJYZ', 'VVV DR4, J Y Z'),
    ('vmc', 'VMC DR7', _vmc,
     'VOXASTRO/P/VHS/JKs/color', 'VHS J Ks - VMC has no HiPS'),
]


def _virac_quality(table, quality, log):
    """Which VIRAC2 epochs the level keeps, and what each criterion removed."""
    keep = np.ones(len(table), dtype=bool)

    if quality == QUALITY_PUBLISHED:
        return keep

    criteria = [
        (np.asarray(table['ambiguous']) == 0,
         'shared with a neighbouring source'),
        (np.asarray(table['ast_chisq'], dtype=float) < VIRAC_AST_CHISQ_MAX,
         'an astrometric outlier above 5 sigma'),
    ]

    if quality == QUALITY_STANDARD:
        criteria.append((np.asarray(table['phot_flag']) == 0,
                         'a photometric calibration flag set'))

    for passed, meaning in criteria:
        removed = int(np.sum(keep & ~passed))

        if removed:
            log(f"    {removed:5d} removed - {meaning}")

        keep &= passed

    return keep


@survey_source(
    name='VISTA surveys (VVV, VMC)',
    short_name='VISTA',
    state_acquiring='acquiring VISTA lightcurve',
    state_acquired='VISTA lightcurve acquired',
    log_file='vista.log',
    output_files=['vista.log', 'vista_lc.png', 'vista.vot', 'vista.txt'],
    button_text='Get VISTA lightcurve',
    form_fields={
        'vista_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': VISTA_SR,
            'required': False,
        },
        'vista_quality': quality_field({
            QUALITY_STANDARD: 'What the surveys use for their own statistics',
            QUALITY_RELAXED: 'Drop only detections that may be another star',
            QUALITY_PUBLISHED: 'None - every epoch as published',
        }),
    },
    help_text='VVV (VIRAC2) and VMC near-infrared time series from ESO, '
              'ZYJHKs, bulge, southern disk and Magellanic Clouds, 2009-2023',
    order=32,
    about=(
        "Two ESO public surveys with the VISTA telescope: VVV over the "
        "Galactic bulge and southern disk, as the VIRAC2 catalogue of its "
        "time series, and VMC over the Magellanic Clouds - near-infrared "
        "light curves, mostly in Ks, for stars fainter than about Ks = 11."),
    about_links=[
        ('VIRAC2, Smith et al. 2025', 'https://arxiv.org/abs/2501.06295'),
        ('VMC, Cioni et al. 2011', 'https://arxiv.org/abs/1103.5758'),
        ('ESO catalogue facility', 'https://archive.eso.org/scienceportal/'),
    ],
    acknowledgement=(
        "Based on data products from observations made with ESO Telescopes at "
        "the La Silla Paranal Observatory under programme IDs 179.B-2002 and "
        "198.B-2004 (VVV, VVVX) and 179.B-2003 (VMC). Cite Smith et al. "
        "(2025), MNRAS 536, 3707, for VIRAC2, and Cioni et al. (2011), A&A "
        "527, A116, for VMC."),
    # Lightcurve metadata
    votable_file='vista.vot',
    lc_bands=[
        surveys.band(band, 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value=band,
                     color=VISTA_COLORS[band],
                     note='VISTA system, Vega')
        for band in VISTA_BANDS
    ],
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color=VISTA_COLORS['Ks'],
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata - the cutout itself is chosen per target, see above
    template_layout='with_cutout',
    show_cutout=True,
    requires_coordinates=True,
)
def target_vista(config, basepath=None, verbose=True, show=False):
    """Acquire VISTA survey lightcurves."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('vista'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = float(config.get('vista_sr') or VISTA_SR)

    # The cutout is chosen by the survey that answered, so what an earlier run
    # chose is dropped until this one has found out
    config.pop('vista_cutout_hips', None)
    config.pop('vista_cutout_name', None)

    parts = {}

    # Each survey is cached on its own: they are separate catalogues, and one
    # answering nothing says nothing about the other
    for key, name, fetch, _, _ in VISTA_SURVEYS:
        with cached_votable_query(f"vista_{key}_{ra:.5f}_{dec:.5f}_{sr:.1f}.vot",
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
        log("Warning: No VISTA time series at this position - VVV covers the "
            "Galactic bulge and the southern disk, VMC the Magellanic Clouds, "
            "and both saturate at about Ks = 11")
        return

    # The first survey with anything here names the image shown beside it
    for key, name, _, hips, hips_name in VISTA_SURVEYS:
        if key in parts:
            config['vista_cutout_hips'] = hips
            config['vista_cutout_name'] = hips_name
            break

    quality = quality_level(config, 'vista')
    kept = []

    for key, name, _, _, _ in VISTA_SURVEYS:
        if key not in parts:
            continue

        table = parts[key]
        log(f"\n{name}: {len(table)} epochs")

        good = (np.isfinite(np.asarray(table['mag'], dtype=float))
                & np.isfinite(np.asarray(table['magerr'], dtype=float))
                & (np.asarray(table['mag'], dtype=float) > 0)
                & (np.asarray(table['magerr'], dtype=float) > 0)
                & (np.asarray(table['magerr'], dtype=float) < 1.0))

        if np.any(~good):
            log(f"    {int(np.sum(~good)):5d} removed - no usable magnitude or error")

        if key == 'vvv':
            good &= _virac_quality(table, quality, log)

        table = table[good]

        # Where there are no flags to go on - VMC - and as a last check where
        # there are, the errors themselves
        if quality != QUALITY_PUBLISHED and len(table):
            clip = clip_noisy_points(table['mag'], table['magerr'],
                                     np.asarray(table['filter'], dtype=str),
                                     log=log, group_name='band',
                                     ratio=CLIP_RATIO_BY_LEVEL[quality])
            table = table[~clip]

        log(f"  {len(table)} epochs left ({quality} filtering)")

        if len(table):
            kept.append(table[['survey', 'source', 'mjd', 'filter', 'mag', 'magerr']])

    if not kept:
        log("Warning: No VISTA epochs left at this level of filtering")
        return

    vista = vstack(kept)
    vista.sort('mjd')

    bands = np.asarray(vista['filter'], dtype=str)

    log_conversion(
        log, 'VISTA',
        'no conversion applied - each band is published as measured',
        {'system': ('Vega', 'the VISTA photometric system, calibrated on 2MASS')},
        npoints=len(vista),
        note='not put on the common g scale: no near-infrared band has a '
             'route to g',
    )

    log_bands(log, 'VISTA', [
        {'label': band, 'kind': 'native', 'npoints': int(np.sum(bands == band)),
         'note': 'VISTA system, Vega'}
        for band in VISTA_BANDS if np.any(bands == band)
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'vista_lc.png'),
                            figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        when = Time(np.asarray(vista['mjd'], dtype=float), format='mjd').datetime

        for band in VISTA_BANDS:
            idx = bands == band

            if np.any(idx):
                plot_with_errors(ax, when[idx], vista['mag'][idx],
                                 vista['magerr'][idx], label=band,
                                 color=VISTA_COLORS[band])

        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        ax.legend()
        ax.set_ylabel('Magnitude (Vega)')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - "
                     f"{' and '.join(sorted(set(np.asarray(vista['survey'], dtype=str))))}")

    log("VISTA lightcurve plot saved to file:vista_lc.png")

    vista.write(os.path.join(basepath, 'vista.vot'), format='votable', overwrite=True)
    vista.write(os.path.join(basepath, 'vista.txt'),
                format='ascii.commented_header', overwrite=True)
    log("VISTA data written to file:vista.vot")
    log("VISTA data written to file:vista.txt")
