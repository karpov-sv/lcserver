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

from ..surveys import survey_source, get_output_files
from .utils import cleanup_paths, parse_votable_lenient, cached_votable_query


@survey_source(
    name='APPLAUSE',
    short_name='APPLAUSE',
    state_acquiring='acquiring APPLAUSE lightcurve',
    state_acquired='APPLAUSE lightcurve acquired',
    log_file='applause.log',
    output_files=['applause.log', 'applause_lc.png', 'applause_color_mag.png'],
    button_text='Get APPLAUSE lightcurve',
    help_text='European plate archive (Dec > -30 deg)',
    order=50,
    # Lightcurve metadata
    votable_file='applause.vot',
    lc_mag_column='mag_g',
    lc_err_column='magerr',
    lc_color='#9467bd',
    lc_mode='magnitude',
    lc_short=False,
    # Template metadata
    template_layout='with_cutout',
    declination_min=-30,
    show_color_mag=True,
    color_mag_file='applause_color_mag.png',
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

    # Cleanup stale plots
    cleanup_paths(get_output_files('applause'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")


    cache_name = f"applause_{config['target_ra']:.4f}_{config['target_dec']:.4f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'APPLAUSE') as cache:
        if not cache.hit:
            applause_sr = config.get('applause_sr', 2.0)

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
            response = requests.get(result_url)

            # Use helper function to parse potentially malformed VOTable
            applause = parse_votable_lenient(response.content)

            cache.save(applause)

        applause = cache.data

    log(f"{len(applause)} original data points")

    if not len(applause):
        return

    applause['time'] = Time(applause['jd_start'], format='jd')
    applause['mjd'] = applause['time'].mjd
    applause.sort('time')

    BP_minus_RP = config.get('BP_minus_RP', np.nanmedian(applause['gaiaedr3_bp_rp']))
    g_minus_r = config.get('g_minus_r', 0.0)

    log(f"Using BP - RP = {BP_minus_RP:.2f} for converting natural magnitudes to Gaia Gmag")

    RPmag = applause['natmag'] - BP_minus_RP*applause['color_term']
    BPmag = RPmag + BP_minus_RP # assuming constant color
    # Simple one-color fits based on Landolt standards
    gmag = BPmag - np.polyval([-0.11445168305534677, -0.20378930951540578, 0.0499368274565225], g_minus_r)
    rmag = BPmag - np.polyval([-0.13189831407771777, 0.8213890428750275, 0.04388161680503415], g_minus_r)

    applause['mag_RP'] = RPmag
    applause['magerr'] = applause['natmag_error']

    applause['mag_g'] = gmag
    applause['mag_r'] = rmag

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'applause_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        ax.errorbar(applause['time'].datetime, applause['mag_g'], applause['magerr'], fmt='.', label='g')

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
