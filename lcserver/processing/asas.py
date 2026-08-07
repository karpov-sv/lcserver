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

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import cleanup_paths, cached_votable_query, log_bands, log_conversion


@survey_source(
    name='ASAS-SN',
    short_name='ASAS-SN',
    state_acquiring='acquiring ASAS-SN lightcurve',
    state_acquired='ASAS-SN lightcurve acquired',
    log_file='asas.log',
    output_files=['asas.log', 'asas_lc.png', 'asas.vot', 'asas.txt'],
    button_text='Get ASAS-SN lightcurve',
    help_text='All-Sky Automated Survey for Supernovae',
    order=20,
    # Lightcurve metadata
    votable_file='asas.vot',
    lc_bands=[
        surveys.band('V', 'mag_V', 'mag_err', surveys.BAND_NATIVE,
                     filter_column='phot_filter', filter_value='V',
                     color='#1f77b4', note='as reported by ASAS-SN'),
        surveys.band('g', 'mag_g_nat', 'mag_err', surveys.BAND_NATIVE,
                     filter_column='phot_filter', filter_value='g',
                     color='#17becf', note='as reported by ASAS-SN'),
        surveys.band('g (conv.)', 'mag_g', 'mag_err', surveys.BAND_DERIVED,
                     color='#9edae5',
                     note='V and g put on a common g scale using an assumed g - r'),
    ],
    # The combined light curve wants everything on one scale
    lc_mag_column='mag_g',
    lc_err_column='mag_err',
    lc_filter_column='phot_filter',
    lc_color='#1f77b4',
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata
    template_layout='with_cutout',
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

    # Check if processed data already exists
    if os.path.exists(os.path.join(basepath, 'asas.vot')):
        log(f"Loading processed ASAS-SN lightcurve from asas.vot")
        asas = Table.read(os.path.join(basepath, 'asas.vot'))
    else:
        # Cache raw query results before color conversion
        ra = config.get('target_ra')
        dec = config.get('target_dec')
        asas_sr = config.get('asas_sr', 10.0)
        cache_name = f"asas_{ra:.4f}_{dec:.4f}_{asas_sr:.1f}.vot"

        with cached_votable_query(cache_name, basepath, log, 'ASAS-SN') as cache:
            if not cache.hit:
                # Query ASAS-SN - only if not cached
                log(f"Requesting ASAS-SN lightcurve for {config['target_name']} within {asas_sr:.1f} arcsec")

                try:
                    client = SkyPatrolClient()
                    lcq = client.cone_search(
                        ra_deg=ra,
                        dec_deg=dec,
                        radius=asas_sr/3600,
                        catalog='master_list',
                        download=True
                    )
                except:
                    import traceback
                    traceback.print_exc()
                    lcq = None

                if not lcq or not len(lcq.data):
                    log("Warning: No ASAS-SN data found")
                    return

                # Cache raw query results
                asas_raw = Table.from_pandas(lcq.data)
                cache.save(asas_raw)

            # Use cached or freshly queried raw data
            asas = cache.data

    log(f"{len(asas)} ASAS-SN data points found")

    for fn in ['g', 'V']:
        idx = asas['phot_filter'] == fn
        idx_good = idx & (asas['quality'] == 'G') & (asas['mag_err'] < 0.05)

        log(f"  {fn}: {np.sum(idx)} total, {np.sum(idx_good)} good")

    log("Earliest: ", Time(np.min(asas['jd']), format='jd').datetime.strftime('%Y-%m-%s %H:%M:%S'))
    log("  Latest: ", Time(np.max(asas['jd']), format='jd').datetime.strftime('%Y-%m-%s %H:%M:%S'))

    asas['time'] = Time(asas['jd'], format='jd')
    asas['mjd'] = asas['time'].mjd

    # Native measurements, kept per band, and the common g scale built from them
    asas['mag_V'] = np.nan
    asas['mag_g_nat'] = np.nan
    asas['mag_g'] = np.nan

    g_minus_r = config.get('g_minus_r', 0.0)
    g_minus_r_origin = 'from config' if 'g_minus_r' in config else 'default, no colour known'

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'asas_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        idx = asas['quality'] == 'G'
        idx &= np.isfinite(asas['mag'])
        idx &= asas['mag_err'] < 0.05

        idx_V = idx & (asas['phot_filter'] == 'V')
        idx_g = idx & (asas['phot_filter'] == 'g')

        # Native magnitudes, exactly as ASAS-SN reports them
        asas['mag_V'][idx_V] = asas['mag'][idx_V]
        asas['mag_g_nat'][idx_g] = asas['mag'][idx_g]

        # ... and the same points carried onto a common g scale
        asas['mag_g'][idx_V] = asas['mag'][idx_V] + 0.02 + 0.498*g_minus_r + 0.008*g_minus_r**2
        asas['mag_g'][idx_g] = asas['mag'][idx_g] - 0.013 - 0.145*g_minus_r - 0.019*g_minus_r**2

        log_conversion(
            log, 'ASAS-SN',
            'g = V + 0.02 + 0.498*(g-r) + 0.008*(g-r)^2',
            {'(g - r)': (g_minus_r, g_minus_r_origin)},
            npoints=int(np.sum(idx_V)),
            note='the colour is assumed constant over the whole light curve',
        )
        log_conversion(
            log, 'ASAS-SN',
            'g = g_ASAS - 0.013 - 0.145*(g-r) - 0.019*(g-r)^2',
            {'(g - r)': (g_minus_r, g_minus_r_origin)},
            npoints=int(np.sum(idx_g)),
        )

        ax.errorbar(asas[idx_V]['time'].datetime, asas[idx_V]['mag_g'], asas[idx_V]['mag_err'], fmt='.', label='V conv. to g')
        ax.errorbar(asas[idx_g]['time'].datetime, asas[idx_g]['mag_g'], asas[idx_g]['mag_err'], fmt='.', label='g')

        log_bands(log, 'ASAS-SN', [
            {'label': 'V', 'kind': 'native', 'npoints': int(np.sum(idx_V)),
             'note': 'as reported by ASAS-SN'},
            {'label': 'g', 'kind': 'native', 'npoints': int(np.sum(idx_g)),
             'note': 'as reported by ASAS-SN'},
            {'label': 'g (conv.)', 'kind': 'derived', 'npoints': int(np.sum(idx_V | idx_g)),
             'note': 'both bands on the common g scale'},
        ])

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
