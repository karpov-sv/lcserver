"""Mini-MegaTORTORA lightcurve acquisition module.

Acquires Mini-MegaTORTORA (MMT9) Russian wide-field optical survey lightcurves.
"""

import os
import numpy as np
import requests

from astropy.table import Table
from astropy.time import Time

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import cleanup_paths, cached_votable_query


@survey_source(
    name='Mini-MegaTORTORA',
    short_name='MMT9',
    state_acquiring='acquiring Mini-MegaTORTORA lightcurve',
    state_acquired='Mini-MegaTORTORA lightcurve acquired',
    log_file='mmt9.log',
    output_files=['mmt9.log', 'mmt9_lc.png'],
    button_text='Get Mini-MegaTORTORA lightcurve',
    form_fields={
        'mmt9_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': 15.0,
            'required': False,
        }
    },
    help_text='Russian wide-field optical survey',
    order=60,
    # Lightcurve metadata
    votable_file='mmt9.vot',
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_color='#8c564b',
    lc_mode='magnitude',
    lc_short=False,
    # Template metadata
    template_layout='simple',
)
def target_mmt9(config, basepath=None, verbose=True, show=False):
    """
    Get Mini-MegaTORTORA lightcurve.

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
    cleanup_paths(get_output_files('mmt9'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    cache_name = f"mmt9_{config['target_ra']:.4f}_{config['target_dec']:.4f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'Mini-MegaTORTORA') as cache:
        if not cache.hit:
            mmt9_sr = config.get('mmt9_sr', 15.0)  # Search radius in arcsec

            log(f"for {config['target_name']} within {mmt9_sr:.1f} arcsec")

            # Convert search radius to degrees
            sr_deg = mmt9_sr / 3600.0

            # Query API
            api_url = f"http://survey.favor2.info/favor2/photometry/mjd"
            params = {
                "sr": sr_deg,
                "ra": config['target_ra'],
                "dec": config['target_dec']
            }

            log(f"Querying Mini-MegaTORTORA at RA={config['target_ra']:.4f}, Dec={config['target_dec']:.4f}")

            try:
                response = requests.get(api_url, params=params, timeout=60)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f'Error downloading Mini-MegaTORTORA lightcurve: {e}')

            # Parse whitespace-separated table with commented header
            from io import StringIO
            try:
                mmt9 = Table.read(response.text, format='ascii.commented_header', delimiter=' ')
                mmt9.rename_column('MJD', 'mjd')
                mmt9.rename_column('Mag', 'mag')
                mmt9.rename_column('Magerr', 'magerr')

            except Exception as e:
                raise RuntimeError(f'Error parsing Mini-MegaTORTORA data: {e}')

            if not len(mmt9):
                log("Warning: No Mini-MegaTORTORA data found")
                return

            log(f"Downloaded {len(mmt9)} data points from Mini-MegaTORTORA")

            cache.save(mmt9)

        mmt9 = cache.data

    log(f"{len(mmt9)} original data points")

    # Convert time to MJD if not already
    if 'mjd' not in mmt9.colnames:
        if 'time' in mmt9.colnames:
            mmt9['mjd'] = mmt9['time']
        else:
            raise RuntimeError('Cannot find time column in Mini-MegaTORTORA data')

    mmt9['time'] = Time(mmt9['mjd'], format='mjd')
    mmt9.sort('time')

    # Filter out bad data
    mmt9 = mmt9[np.isfinite(mmt9['mag'])]
    mmt9 = mmt9[mmt9['mag'] > 0]

    mmt9 = mmt9[np.isfinite(mmt9['magerr'])]
    mmt9 = mmt9[mmt9['magerr'] > 0]
    mmt9 = mmt9[mmt9['magerr'] < 1.0]  # Filter out large errors

    mmt9['filter'] = 'V'

    log(f"{len(mmt9)} data points after filtering")
    if not len(mmt9):
        return

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'mmt9_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        ax.errorbar(mmt9['time'].datetime, mmt9['mag'], mmt9['magerr'], fmt='.', label='V')

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        ax.set_ylabel('V magnitude')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - Mini-MegaTORTORA")

    # Time cannot be serialized to VOTable
    mmt9[[_ for _ in mmt9.columns if _ != 'time']].write(os.path.join(basepath, 'mmt9.vot'),
                                                          format='votable', overwrite=True)
    mmt9[[_ for _ in mmt9.columns if _ != 'time']].write(os.path.join(basepath, 'mmt9.txt'),
                                                          format='ascii.commented_header', overwrite=True)
    log("Mini-MegaTORTORA data written to file:mmt9.vot")
    log("Mini-MegaTORTORA data written to file:mmt9.txt")
