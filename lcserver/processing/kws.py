"""Kamogata Wide-field Survey lightcurve acquisition module.

Acquires KWS (Kamogata Wide-field Survey) optical lightcurves.
"""

import os
import requests
import numpy as np

from astropy.table import Table
from astropy.time import Time

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import cleanup_paths, cached_votable_query


@survey_source(
    name='Kamogata Wide-field Survey',
    short_name='KWS',
    state_acquiring='acquiring KWS lightcurve',
    state_acquired='KWS lightcurve acquired',
    log_file='kws.log',
    output_files=['kws.log', 'kws_lc.png', 'kws.vot', 'kws.txt'],
    button_text='Get KWS lightcurve',
    help_text='Kamogata Wide-field Survey',
    order=23,
    # Lightcurve metadata
    votable_file='kws.vot',
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color='#e377c2',
    lc_mode='magnitude',
    lc_short=False,
    # Template metadata
    template_layout='simple',
    requires_coordinates=False,
)
def target_kws(config, basepath=None, verbose=True, show=False):
    """Acquire Kamogata Wide-field Survey lightcurve."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Cleanup stale plots
    cleanup_paths(get_output_files('kws'), basepath=basepath)

    # KWS uses target name, not coordinates
    target_name = config.get('target_name')
    if not target_name:
        log("Error: target_name not found in config")
        raise RuntimeError("Target name required for KWS query")

    # Create cache name based on target name
    # Sanitize the name for use in filename
    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in target_name)
    safe_name = safe_name.replace(' ', '_')
    cache_name = f"kws_{safe_name}.vot"

    with cached_votable_query(cache_name, basepath, log, 'Kamogata Wide-field Survey') as cache:
        if not cache.hit:
            log(f"for {target_name}")

            # Query KWS database
            # Note: KWS does not support coordinates, only object names resolved via SIMBAD
            try:
                res = requests.post(
                    "http://kws.cetus-net.org/~maehara/VSdata.py",
                    {
                        "object": target_name,  # KWS resolves object name via SIMBAD
                        "resolver": "simbad",
                        "p_band": "All",
                        "plot": "0",
                        "obs_ys": "",
                        "obs_ms": "",
                        "obs_ds": "",
                        "obs_ye": "",
                        "obs_me": "",
                        "obs_de": "",
                        "submit": "Send query"
                    },
                    timeout=30
                )
                res.raise_for_status()
            except requests.RequestException as e:
                log(f"Error: Error querying KWS: {e}")
                return

            # Parse response
            # KWS returns HTML with embedded table
            try:
                content = res.content.decode('utf-8', errors='ignore')

                # Extract table from HTML
                start_marker = '<table>'
                end_marker = '</table>'

                start_idx = content.find(start_marker)
                if start_idx == -1:
                    log("No table found in KWS response")
                    log("Object might not be in KWS database or name not resolved")
                    return

                end_idx = content.find(end_marker, start_idx)
                if end_idx == -1:
                    log("Malformed KWS response - could not find table end")
                    return

                end_idx += len(end_marker)

                table_html = content[start_idx:end_idx]

                # Parse HTML table
                # Expected columns: name, time, mag, magerr, filter, frame
                kws = Table.read(
                    table_html,
                    format='html',
                    names=['name', 'time', 'mag', 'magerr', 'filter', 'frame'],
                    data_start=1
                )

                if not len(kws):
                    log("No KWS data points found")
                    return

                # Convert time to MJD
                kws['mjd'] = Time(kws['time']).mjd

                cache.save(kws)

                log(f"Found {len(kws)} KWS data points")

            except Exception as e:
                import traceback
                log(f"Error: Error parsing KWS response: {e}")
                log(traceback.format_exc())
                return

        kws = cache.data

    # Filter out bad data
    kws = kws[np.isfinite(kws['mag'])]
    kws = kws[np.isfinite(kws['magerr'])]
    kws = kws[kws['magerr'] > 0]
    kws = kws[kws['magerr'] < 1.0]  # Filter out large errors

    log(f"{len(kws)} data points after filtering")

    if not len(kws):
        log("No valid KWS data points after filtering")
        return

    # Add time column for plotting
    kws['time_obj'] = Time(kws['mjd'], format='mjd')

    # Sort by time
    kws.sort('mjd')

    # Standardize magnitude column name for compatibility
    kws['mag_V'] = kws['mag']  # KWS typically uses V-band equivalent

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'kws_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        # Plot by filter if multiple filters present
        unique_filters = np.unique(kws['filter'])

        for filt in unique_filters:
            idx = kws['filter'] == filt
            if np.sum(idx):
                label = f'{filt}' if filt else 'unfiltered'
                ax.errorbar(
                    kws['time_obj'][idx].datetime,
                    kws['mag'][idx],
                    kws['magerr'][idx],
                    fmt='.',
                    label=label
                )

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        if len(unique_filters) > 1:
            ax.legend()

        ax.set_ylabel('Magnitude')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - Kamogata Wide-field Survey")

    log("KWS lightcurve plot saved to file:kws_lc.png")

    # Save data
    # Remove time_obj column (not serializable to VOTable)
    kws_save = kws[[_ for _ in kws.columns if _ != 'time_obj']]

    kws_save.write(
        os.path.join(basepath, 'kws.vot'),
        format='votable', overwrite=True
    )
    kws_save.write(
        os.path.join(basepath, 'kws.txt'),
        format='ascii.commented_header', overwrite=True
    )
    log("KWS data written to file:kws.vot")
    log("KWS data written to file:kws.txt")
