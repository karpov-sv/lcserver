"""APPLAUSE lightcurve acquisition module.

Acquires APPLAUSE (Archives of Photographic PLates for Astronomical USE)
European plate archive lightcurves for Dec > -30 deg.
"""

import os
import numpy as np
import requests

from astropy.table import Table
from astropy.time import Time

import pyvo as vo

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (cleanup_paths, parse_votable_lenient, cached_votable_query,
                    quality_field, quality_level, log_bands, log_conversion,
                    plot_with_errors,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


# What SExtractor found wrong with the image of the star. The deblending bit,
# 2, is not among these: the plates carrying it measure the star as well as the
# ones that do not, and on a crowded plate almost everything is deblended.
APPLAUSE_SEXTRACTOR_BAD = 4 | 8 | 16 | 32   # saturated, truncated, aperture, isophotal

# Beyond this the plate's own estimate of its uncertainty is large enough that
# the magnitude adds little
APPLAUSE_MAX_ERROR = 0.3

# What the archive will not vouch for photometrically, from the DR4 schema:
# phot_range_flags is 0 where the star falls within the range of the plate's
# calibration stars, and 1 or 2 where it is brighter or fainter than all of
# them - in which case the magnitude is an extrapolation of the calibration
# curve rather than a reading from it.
#
# phot_calib_flags is deliberately not used. It says whether the star was
# itself accepted as one of the plate's calibrators, and a star is accepted
# only where its measured magnitude agrees with the one predicted from Gaia -
# so keeping the points that carry it would keep exactly the nights the star
# behaved, and drop the ones it did not. On a variable that is the light curve
# thrown away and the flat parts kept. Its outlier value is no better: on the
# eclipsing binary this was checked against, every point it marks is in
# eclipse.
APPLAUSE_MAX_NEIGHBOURS = 1


@survey_source(
    name='APPLAUSE',
    short_name='APPLAUSE',
    state_acquiring='acquiring APPLAUSE lightcurve',
    state_acquired='APPLAUSE lightcurve acquired',
    log_file='applause.log',
    output_files=['applause.log', 'applause_lc.png', 'applause.vot', 'applause.txt'],
    button_text='Get APPLAUSE lightcurve',
    form_fields={
        'applause_quality': quality_field({
            QUALITY_STANDARD: 'Drop extrapolated, blended and badly imaged plates',
            QUALITY_RELAXED: 'Drop only what the calibration could not reach',
            QUALITY_PUBLISHED: 'None - every plate as published',
        }),
    },
    help_text='European plate archive (Dec > -30 deg)',
    order=50,
    # Lightcurve metadata
    votable_file='applause.vot',
    lc_bands=[
        # The plates carry a per-plate colour term, so the natural magnitudes
        # are only comparable between plates once it has been taken out
        surveys.band('RP', 'mag_RP', 'magerr', surveys.BAND_CALIBRATED,
                     color='#8c564b',
                     note='natural plate magnitudes with the per-plate colour term removed'),
        surveys.band('g (conv.)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     color='#c49c94',
                     note='from RP through an assumed constant BP - RP and g - r',
                     combined=True),
        surveys.band('r (conv.)', 'mag_r', 'magerr', surveys.BAND_DERIVED,
                     color='#e7969c',
                     note='from RP through an assumed constant BP - RP and g - r'),
    ],
    lc_mag_column='mag_g',
    lc_err_column='magerr',
    lc_color='#9467bd',
    lc_mode='magnitude',
    lc_short=False,
    # Template metadata
    template_layout='with_cutout',
    declination_min=-30,
)
def target_applause(config, basepath=None, verbose=True, show=False):
    """
    Get APPLAUSE lightcurve.

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
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('applause'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    applause_sr = config.get('applause_sr', 2.0)
    cache_name = f"applause_{ra:.4f}_{dec:.4f}_{applause_sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'APPLAUSE', refresh=refresh_cache) as cache:
        if not cache.hit:

            log(f"for {config['target_name']} within {applause_sr:.1f} arcsec")

            url = 'https://www.plate-archive.org/tap'

            query = f"""
            SELECT
                s.*,
                DEGREES(spoint(RADIANS(s.ra_icrs), RADIANS(s.dec_icrs)) <-> spoint(RADIANS({config['target_ra']}), RADIANS({config['target_dec']}))) AS angdist,
                e.jd_start, e.jd_mid, e.jd_end
            FROM applause_dr4.source_calib s, applause_dr4.exposure e, applause_dr4.plate p
            WHERE
                s.pos @ scircle(spoint(RADIANS({config['target_ra']}), RADIANS({config['target_dec']})), RADIANS({applause_sr/3600}))
                AND
                s.plate_id = e.plate_id
                AND
                s.plate_id = p.plate_id
                AND
                p.numexp = 1
                AND
                s.match_radius > 0
                AND
                s.model_prediction > 0.9
                AND
                s.natmag_error > 0
            """

            tap_service = vo.dal.TAPService(url) # Anonymous access

            job = tap_service.submit_job(query, language='PostgreSQL')
            job.run()

            job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=120.)

            # TODO: more intelligent error handling?
            job.raise_if_error()

            # Parse VOTable with lenient error handling
            # The APPLAUSE TAP service sometimes returns malformed XML with undefined entities
            result_url = job.result_uri
            response = requests.get(result_url, timeout=300)

            # Use helper function to parse potentially malformed VOTable
            applause = parse_votable_lenient(response.content)

            if applause is not None and len(applause):
                cache.save(applause)
            else:
                cache.save_empty()

        applause = cache.data

    # Nothing here, and cached as nothing - the helper has said so already
    if applause is None:
        return

    log(f"{len(applause)} original data points")

    quality = quality_level(config, 'applause')

    # Nothing was ever judged here beyond what the query asked for. What the
    # archive records about each plate falls into two kinds: whether it could
    # calibrate the star at all, and whether it imaged it cleanly.
    if quality != QUALITY_PUBLISHED:
        drop = {}

        # The star lies outside the range of the plate's calibration stars, so
        # its magnitude is the calibration curve extrapolated rather than read
        if 'phot_range_flags' in applause.colnames:
            drop['extrapolated'] = \
                np.asarray(applause['phot_range_flags'], dtype=float) != 0

        if quality == QUALITY_STANDARD:
            # Another Gaia source falls in with it: the plate measures both
            if 'gaiaedr3_neighbors' in applause.colnames:
                drop['blended'] = (
                    np.asarray(applause['gaiaedr3_neighbors'], dtype=float)
                    > APPLAUSE_MAX_NEIGHBOURS)

            # Saturated, truncated, or measured through an incomplete aperture
            if 'sextractor_flags' in applause.colnames:
                flags = np.asarray(applause['sextractor_flags'], dtype=float).astype(int)
                drop['badly imaged'] = (flags & APPLAUSE_SEXTRACTOR_BAD) != 0

            if 'natmag_error' in applause.colnames:
                drop[f'error > {APPLAUSE_MAX_ERROR:.1f}'] = (
                    np.asarray(applause['natmag_error'], dtype=float)
                    > APPLAUSE_MAX_ERROR)

        bad = np.zeros(len(applause), dtype=bool)
        told = []

        for why, idx in drop.items():
            if np.any(idx & ~bad):
                told.append(f"{int(np.sum(idx & ~bad))} {why}")
            bad |= idx

        if np.any(bad):
            log(f"Warning: dropping {int(np.sum(bad))} plates: {', '.join(told)}")
            applause = applause[~bad]
            log(f"  {len(applause)} plates left")

        if not len(applause):
            log("Warning: No APPLAUSE plates left at this level of filtering")
            return

    applause['time'] = Time(applause['jd_start'], format='jd')
    applause['mjd'] = applause['time'].mjd
    applause.sort('time')

    BP_minus_RP = config.get('BP_minus_RP', np.nanmedian(applause['gaiaedr3_bp_rp']))
    g_minus_r = config.get('g_minus_r', 0.0)

    log(f"Using BP - RP = {BP_minus_RP:.2f} for converting natural magnitudes to Gaia Gmag")

    log_conversion(
        log, 'APPLAUSE',
        'RP = natmag - (BP - RP) * color_term',
        {
            '(BP - RP)': (float(BP_minus_RP),
                          'from config' if 'BP_minus_RP' in config
                          else 'median of the Gaia EDR3 colours of the matched stars'),
            'color_term': ('per plate, from the APPLAUSE archive', 'not assumed'),
        },
        npoints=len(applause),
        note='without this the plate zero points drift with the colour of the star',
    )

    RPmag = applause['natmag'] - BP_minus_RP*applause['color_term']
    BPmag = RPmag + BP_minus_RP # assuming constant color
    # Simple one-color fits based on Landolt standards
    gmag = BPmag - np.polyval([-0.11445168305534677, -0.20378930951540578, 0.0499368274565225], g_minus_r)
    rmag = BPmag - np.polyval([-0.13189831407771777, 0.8213890428750275, 0.04388161680503415], g_minus_r)

    applause['mag_RP'] = RPmag
    applause['magerr'] = applause['natmag_error']

    applause['mag_g'] = gmag
    applause['mag_r'] = rmag

    log_conversion(
        log, 'APPLAUSE',
        'BP = RP + (BP - RP);  g = BP - poly_g(g-r);  r = BP - poly_r(g-r)',
        {
            '(BP - RP)': (float(BP_minus_RP), 'assumed constant in time'),
            '(g - r)': (g_minus_r,
                        'from config' if 'g_minus_r' in config else 'default, no colour known'),
            'poly_g': '[-0.11445, -0.20379, 0.04994]  (Landolt standards)',
            'poly_r': '[-0.13190, 0.82139, 0.04388]  (Landolt standards)',
        },
        npoints=len(applause),
        note='a model rather than a measurement - RP above is the calibrated quantity',
    )

    log_bands(log, 'APPLAUSE', [
        {'label': 'RP', 'kind': 'calibrated', 'npoints': len(applause),
         'note': 'colour term removed'},
        {'label': 'g (conv.)', 'kind': 'derived', 'npoints': len(applause),
         'note': 'assumes a constant colour'},
        {'label': 'r (conv.)', 'kind': 'derived', 'npoints': len(applause),
         'note': 'assumes a constant colour'},
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'applause_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        plot_with_errors(ax, applause['time'].datetime, applause['mag_g'],
                         applause['magerr'], label='g')

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        # ax.legend()
        ax.set_ylabel('g')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - APPLAUSE")

    # Time cannot be serialized to VOTable
    applause[[_ for _ in applause.columns if _ != 'time']].write(
        os.path.join(basepath, 'applause.vot'),
        format='votable', overwrite=True
    )
    applause[[_ for _ in applause.columns if _ != 'time']].write(
        os.path.join(basepath, 'applause.txt'),
        format='ascii.commented_header', overwrite=True
    )
    log("APPLAUSE data written to file:applause.vot")
    log("APPLAUSE data written to file:applause.txt")
