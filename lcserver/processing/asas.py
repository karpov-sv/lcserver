"""ASAS-SN lightcurve acquisition module.

Acquires ASAS-SN (All-Sky Automated Survey for Supernovae) lightcurves.
"""

import os
import numpy as np

from astropy.table import Table
from astropy.time import Time

from pyasassn.client import SkyPatrolClient

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import cleanup_paths


@survey_source(
    name='ASAS-SN',
    short_name='ASAS-SN',
    state_acquiring='acquiring ASAS-SN lightcurve',
    state_acquired='ASAS-SN lightcurve acquired',
    log_file='asas.log',
    output_files=['asas.log', 'asas_lc.png', 'asas_color_mag.png'],
    button_text='Get ASAS-SN lightcurve',
    help_text='All-Sky Automated Survey for Supernovae',
    order=20,
    # Lightcurve metadata
    votable_file='asas.vot',
    lc_mag_column='mag_g',
    lc_err_column='mag_err',
    lc_filter_column='phot_filter',
    lc_color='#1f77b4',
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata
    template_layout='with_cutout',
    show_color_mag=True,
    color_mag_file='asas_color_mag.png',
)
def target_asas(config, basepath=None, verbose=True, show=False):
    """
    Get ASAS-SN lightcurve.

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
    cleanup_paths(get_output_files('asas'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    if os.path.exists(os.path.join(basepath, 'asas.vot')):
        log(f"Loading ASAS-SN lightcurve from asas.vot")
        asas = Table.read(os.path.join(basepath, 'asas.vot'))
    else:
        asas_sr = config.get('asas_sr', 10.0)

        log(f"Requesting ASAS-SN lightcurve for {config['target_name']} within {asas_sr:.1f} arcsec")

        try:
            client = SkyPatrolClient()
            lcq = client.cone_search(ra_deg=config.get('target_ra'), dec_deg=config.get('target_dec'), radius=asas_sr/3600, catalog='master_list', download=True)
        except:
            import traceback
            traceback.print_exc()

            lcq = None

        if not lcq or not len(lcq.data):
            log("Warning: No ASAS-SN data found")
            return

        asas = Table.from_pandas(lcq.data)

    log(f"{len(asas)} ASAS-SN data points found")

    for fn in ['g', 'V']:
        idx = asas['phot_filter'] == fn
        idx_good = idx & (asas['quality'] == 'G') & (asas['mag_err'] < 0.05)

        log(f"  {fn}: {np.sum(idx)} total, {np.sum(idx_good)} good")

    log("Earliest: ", Time(np.min(asas['jd']), format='jd').datetime.strftime('%Y-%m-%s %H:%M:%S'))
    log("  Latest: ", Time(np.max(asas['jd']), format='jd').datetime.strftime('%Y-%m-%s %H:%M:%S'))

    asas['time'] = Time(asas['jd'], format='jd')
    asas['mjd'] = asas['time'].mjd

    asas['mag_V'] = np.nan
    asas['mag_g'] = np.nan

    g_minus_r = config.get('g_minus_r', 0.0)
    log(f"Will use (g - r) = {g_minus_r:.2f} for converting V to g magnitudes")

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'asas_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        idx = asas['quality'] == 'G'
        idx &= np.isfinite(asas['mag'])
        idx &= asas['mag_err'] < 0.05

        idx1 = idx & (asas['phot_filter'] == 'V')
        asas['mag_V'][idx1] = asas['mag'][idx1]
        asas['mag_g'][idx1] = asas['mag'][idx1] + 0.02 + 0.498*g_minus_r + 0.008*g_minus_r**2

        ax.errorbar(asas[idx1]['time'].datetime, asas[idx1]['mag_g'], asas[idx1]['mag_err'], fmt='.', label='V conv. to g')

        idx1 = idx & (asas['phot_filter'] == 'g')
        asas['mag_g'][idx1] = asas['mag'][idx1] - 0.013 - 0.145*g_minus_r - 0.019*g_minus_r**2

        ax.errorbar(asas[idx1]['time'].datetime, asas[idx1]['mag_g'], asas[idx1]['mag_err'], fmt='.', label='g')

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        ax.legend()
        ax.set_ylabel('g')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - ASAS-SN")

    # Time cannot be serialized to VOTable
    asas[[_ for _ in asas.columns if _ != 'time']].write(os.path.join(basepath, 'asas.vot'),
                                                         format='votable', overwrite=True)
    asas[[_ for _ in asas.columns if _ != 'time']].write(os.path.join(basepath, 'asas.txt'),
                                                         format='ascii.commented_header', overwrite=True)
    log("ASAS-SN data written to file:asas.vot")
    log("ASAS-SN data written to file:asas.txt")
