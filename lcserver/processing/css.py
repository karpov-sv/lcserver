"""Catalina Sky Survey lightcurve acquisition module.

Acquires CSS (Catalina Sky Survey) optical lightcurves in V band.
"""

import os
import requests
import numpy as np

from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord

# STDPipe
from stdpipe import plots

from ..surveys import survey_source
from .utils import cleanup_paths, cleanup_css


@survey_source(
    name='Catalina Sky Survey',
    short_name='CSS',
    state_acquiring='acquiring CSS lightcurve',
    state_acquired='CSS lightcurve acquired',
    log_file='css.log',
    output_files=['css.log', 'css_lc.png'],
    button_text='Get CSS lightcurve',
    form_fields={
        'css_radius': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': 2.0,
            'required': False,
        }
    },
    help_text='Catalina Sky Survey optical transient survey',
    order=22,
    # Lightcurve metadata
    votable_file='css.vot',
    lc_mag_column='mag_V',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color='#17becf',
    lc_mode='magnitude',
    lc_short=False,
)
def target_css(config, basepath=None, verbose=True, show=False):
    """Acquire Catalina Sky Survey lightcurve."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Cleanup stale plots
    cleanup_paths(cleanup_css, basepath=basepath)

    cachename = f"css_{config['target_ra']:.4f}_{config['target_dec']:.4f}.vot"

    if os.path.exists(os.path.join(basepath, 'cache', cachename)):
        log(f"Loading Catalina Sky Survey lightcurve from the cache")
        css = Table.read(os.path.join(basepath, 'cache', cachename))
    else:
        # Get search radius from config or use default
        radius_arcsec = config.get('css_radius', 2.0)

        log(f"Querying Catalina Sky Survey within {radius_arcsec} arcsec")

        # Build target coordinate
        target = SkyCoord(ra=config['target_ra'], dec=config['target_dec'], unit='deg')

        # Query CSS database
        try:
            res = requests.post(
                'http://nunuku.caltech.edu/cgi-bin/getcssconedb_release_img.cgi',
                {
                    'RADec': f"{target.ra.deg} {target.dec.deg}",
                    'Rad': radius_arcsec / 60,  # Convert arcsec to arcmin
                    'IMG': 'nun',
                    'DB': 'photcat',
                    '.submit': 'Submit',
                    'OUT': 'csv',
                    'SHORT': 'short',
                    'PLOT': 'plot'
                },
                timeout=30
            )
            res.raise_for_status()
        except requests.RequestException as e:
            log(f"Error querying CSS: {e}")
            raise RuntimeError(f"CSS query failed: {e}")

        # Parse response
        # CSS returns CSV data embedded in HTML with a specific format
        try:
            content = res.content.decode('utf-8', errors='ignore')

            # Extract the data array from the response
            # Format is: data: [[mjd1, mag1, err1], [mjd2, mag2, err2], ...]
            start_marker = 'data: [['
            end_marker = '],]'

            start_idx = content.find(start_marker)
            if start_idx == -1:
                log("No data found in CSS response")
                log("Response might indicate no objects in search radius")
                return

            start_idx += len(start_marker) - 2  # Include the opening [[
            end_idx = content.find(end_marker, start_idx)
            if end_idx == -1:
                log("Malformed CSS response - could not find data end marker")
                return

            end_idx += 3  # Include the closing ]]

            data_str = content[start_idx:end_idx]
            data_array = eval(data_str)  # Parse JavaScript array format

            if not data_array or len(data_array) == 0:
                log("No CSS data points found")
                return

            # Convert to astropy table
            css = Table(np.array(data_array), names=['mjd', 'mag', 'magerr'])

            try:
                os.makedirs(os.path.join(basepath, 'cache'))
            except:
                pass

            css.write(
                os.path.join(basepath, 'cache', cachename),
                format='votable', overwrite=True
            )

        except Exception as e:
            import traceback
            log(f"Error parsing CSS response: {e}")
            log(traceback.format_exc())
            raise RuntimeError(f"Failed to parse CSS data: {e}")

    # Filter out bad data
    css = css[np.isfinite(css['mag'])]
    css = css[np.isfinite(css['magerr'])]
    css = css[css['magerr'] > 0]
    css = css[css['magerr'] < 1.0]  # Filter out large errors

    log(f"{len(css)} data points after filtering")

    if not len(css):
        log("No valid CSS data points after filtering")
        return

    log(f"Found {len(css)} CSS data points")

    # Convert magnitudes
    css['filter'] = 'V'  # CSS uses variant of V band

    # V = V_CSS + 0.31*(B-V)^2 + 0.04 (sigma=0.059)
    css['mag_V'] = css['mag'] + 0.31*config.get('B_minus_V', 0) + 0.04
    # TODO: g mag?..

    # Add time column
    css['time'] = Time(css['mjd'], format='mjd')

    # Sort by time
    css.sort('mjd')

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'css_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        ax.errorbar(css['time'].datetime, css['mag'], css['magerr'], fmt='.', label='V')

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        ax.set_ylabel('CSS V magnitude')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - Catalina Sky Survey")

    log("CSS lightcurve plot saved to file:css_lc.png")

    # Save data
    # Time cannot be serialized to VOTable
    css[[_ for _ in css.columns if _ != 'time']].write(
        os.path.join(basepath, 'css.vot'),
        format='votable', overwrite=True
    )
    css[[_ for _ in css.columns if _ != 'time']].write(
        os.path.join(basepath, 'css.txt'),
        format='ascii.commented_header', overwrite=True
    )
    log("CSS data written to file:css.vot")
    log("CSS data written to file:css.txt")
