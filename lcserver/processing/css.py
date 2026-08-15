"""Catalina Sky Survey lightcurve acquisition module.

Acquires CSS (Catalina Sky Survey) optical lightcurves in V band.
"""

import os
import ast
import requests
import numpy as np

from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query, log_bands,
                    log_conversion, plot_with_errors,
                    assumed_color, v_to_g, V_TO_G_FORMULA)


@survey_source(
    name='Catalina Sky Survey',
    short_name='CSS',
    state_acquiring='acquiring CSS lightcurve',
    state_acquired='CSS lightcurve acquired',
    log_file='css.log',
    output_files=['css.log', 'css_lc.png', 'css.vot', 'css.txt'],
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
    lc_bands=[
        surveys.band('V', 'mag', 'magerr', surveys.BAND_NATIVE,
                     color='#17becf',
                     note='as reported by CSS, in its own unfiltered V-like band'),
        surveys.band('V (corr.)', 'mag_V', 'magerr', surveys.BAND_DERIVED,
                     color='#9edae5',
                     note='colour-corrected onto standard V using an assumed B - V'),
        surveys.band('g (conv.)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     color='#c5e8ed',
                     note='the corrected V put on the common g scale using an '
                          'assumed g - r',
                     combined=True),
    ],
    lc_mag_column='mag_V',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color='#17becf',
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata
    template_layout='simple',
)
def target_css(config, basepath=None, verbose=True, show=False):
    """Acquire Catalina Sky Survey lightcurve."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('css'), basepath=basepath)

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    radius_arcsec = config.get('css_radius', 2.0)
    cache_name = f"css_{ra:.4f}_{dec:.4f}_{radius_arcsec:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'Catalina Sky Survey', refresh=refresh_cache) as cache:
        if not cache.hit:

            log(f"within {radius_arcsec} arcsec")

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
                raise SourceError("could not query CSS - "
                                  f"{type(e).__name__}: {e}")

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
                    log("Warning: No data found in CSS response")
                    log("Response might indicate no objects in search radius")
                    return

                start_idx += len(start_marker) - 2  # Include the opening [[
                end_idx = content.find(end_marker, start_idx)
                if end_idx == -1:
                    raise SourceError("malformed CSS response - no data end marker")

                end_idx += 3  # Include the closing ]]

                data_str = content[start_idx:end_idx]

                # literal_eval rather than eval: this is a string cut out of a
                # reply fetched over plain HTTP, and the survey offers no TLS
                # to fetch it over instead. eval() would hand anyone able to
                # answer in its place - or the host itself, one day - the run
                # of the worker process.
                try:
                    data_array = ast.literal_eval(data_str)
                except (ValueError, SyntaxError) as e:
                    raise SourceError("could not parse the data array from CSS"
                                      f" - {type(e).__name__}: {e}")

                if not data_array or len(data_array) == 0:
                    cache.save_empty()
                    log("Warning: No CSS data points found")
                    return

                # Convert to astropy table
                css = Table(np.array(data_array), names=['mjd', 'mag', 'magerr'])

                cache.save(css)

            except SourceError:
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise SourceError("could not parse the CSS response - "
                                  f"{type(e).__name__}: {e}")

        css = cache.data

    # Nothing here, and cached as nothing - the helper has said so already
    if css is None:
        return

    # Filter out bad data
    css = css[np.isfinite(css['mag'])]
    css = css[np.isfinite(css['magerr'])]
    css = css[css['magerr'] > 0]
    css = css[css['magerr'] < 1.0]  # Filter out large errors

    log(f"{len(css)} data points after filtering")

    if not len(css):
        log("Warning: No valid CSS data points after filtering")
        return

    log(f"Found {len(css)} CSS data points")

    # Convert magnitudes
    css['filter'] = 'V'  # CSS uses variant of V band

    # V = V_CSS + 0.31*(B-V) + 0.04 (sigma=0.059)
    B_minus_V, B_minus_V_origin = assumed_color(config, 'B_minus_V')
    css['mag_V'] = css['mag'] + 0.31*B_minus_V + 0.04

    log_conversion(
        log, 'CSS',
        'V = V_CSS + 0.31*(B-V) + 0.04',
        {
            '(B - V)': (B_minus_V, B_minus_V_origin),
            'scatter of the relation': 0.059,
        },
        npoints=len(css),
        note='the colour is assumed constant; the native CSS magnitudes are kept as well',
    )

    # ... and on from there to the common g scale, the same second step the
    # other V-band surveys take. Two assumed colours deep, so the standing of
    # this band is that of the weaker of them.
    g_minus_r, g_minus_r_origin = assumed_color(config, 'g_minus_r')
    css['mag_g'] = v_to_g(css['mag_V'], g_minus_r)

    log_conversion(
        log, 'CSS',
        V_TO_G_FORMULA,
        {'(g - r)': (g_minus_r, g_minus_r_origin)},
        npoints=len(css),
        note='applied to the corrected V above, so both assumed colours enter',
    )

    log_bands(log, 'CSS', [
        {'label': 'V', 'kind': 'native', 'npoints': len(css),
         'note': 'unfiltered CSS magnitudes, as reported'},
        {'label': 'V (corr.)', 'kind': 'derived', 'npoints': len(css),
         'note': 'on the standard V scale'},
        {'label': 'g (conv.)', 'kind': 'derived', 'npoints': len(css),
         'note': 'on the common g scale, through V'},
    ])

    # Add time column
    css['time'] = Time(css['mjd'], format='mjd')

    # Sort by time
    css.sort('mjd')

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'css_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        plot_with_errors(ax, css['time'].datetime, css['mag'], css['magerr'],
                         label='V')

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
