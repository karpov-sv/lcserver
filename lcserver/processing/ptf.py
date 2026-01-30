"""Palomar Transient Factory lightcurve acquisition module.

Acquires PTF optical lightcurves.
"""

import os
import numpy as np

from astropy.table import Table
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord

from astroquery.ipac.irsa import Irsa

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import cleanup_paths


@survey_source(
    name='Palomar Transient Factory',
    short_name='PTF',
    state_acquiring='acquiring PTF lightcurve',
    state_acquired='PTF lightcurve acquired',
    log_file='ptf.log',
    output_files=['ptf.log', 'ptf_lc.png'],
    button_text='Get PTF lightcurve',
    help_text='Palomar Transient Factory optical survey',
    order=11,
    # Lightcurve metadata
    votable_file='ptf.vot',
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color='#17becf',
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata
    template_layout='simple',
    requires_coordinates=True,
)
def target_ptf(config, basepath=None, verbose=True, show=False):
    """Acquire Palomar Transient Factory lightcurve."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Cleanup stale plots
    cleanup_paths(get_output_files('ptf'), basepath=basepath)

    # Get coordinates
    ra = config.get('target_ra')
    dec = config.get('target_dec')

    if ra is None or dec is None:
        log("Error: target_ra and target_dec are required for PTF query")
        raise RuntimeError("Coordinates required for PTF query")

    # Create cache name based on coordinates
    cachename = f"ptf_{ra:.4f}_{dec:.4f}.vot"

    if os.path.exists(os.path.join(basepath, 'cache', cachename)):
        log(f"Loading Palomar Transient Factory lightcurve from the cache")
        ptf = Table.read(os.path.join(basepath, 'cache', cachename))
    else:
        log(f"Querying Palomar Transient Factory at RA={ra:.4f}, Dec={dec:.4f}")

        # Query PTF catalog
        try:
            target = SkyCoord(ra, dec, unit='deg')
            table = Irsa.query_region(
                coordinates=target,
                spatial='Cone',
                catalog='ptf_lightcurves',
                radius=2 * u.arcsec
            )

            if not len(table):
                log("No PTF data points found within 2 arcsec")
                return

            # Create standardized columns
            table['mjd'] = table['obsmjd']
            table['mag'] = table['mag_autocorr']
            table['magerr'] = table['magerr_auto']
            table['filter'] = np.where(
                table['fid'] == 1, 'g',
                np.where(table['fid'] == 2, 'R', 'Ha')
            )

            # Quality filtering
            log("Applying quality filters...")

            # Basic filtering - positive magnitudes
            idx = table['mag_autocorr'] > 0

            # FWHM ratio filtering (reject elongated sources)
            fwhm_ratio = table['fwhm_image'] / table['fwhmsex']
            idx = np.logical_and(idx, fwhm_ratio < 1.5)

            # Edge of frame filtering (5px margin)
            idx = np.logical_and(idx, table['xpeak_image'] > 5)
            idx = np.logical_and(idx, table['xpeak_image'] < 2043)
            idx = np.logical_and(idx, table['ypeak_image'] > 5)
            idx = np.logical_and(idx, table['ypeak_image'] < 4091)

            # Photometry correction filtering (< 0.5 mag correction)
            idx = np.logical_and(idx, np.abs(table['mag_autocorr'] - table['mag_auto']) < 0.5)

            table = table[idx]

            if not len(table):
                log("No PTF data points remaining after quality filtering")
                return

            log(f"Found {len(table)} PTF data points after filtering")

            # Create cache directory if needed
            try:
                os.makedirs(os.path.join(basepath, 'cache'))
            except:
                pass

            # Save to cache
            table.write(
                os.path.join(basepath, 'cache', cachename),
                format='votable', overwrite=True
            )

            ptf = table

        except Exception as e:
            import traceback
            log(f"Error querying PTF: {e}")
            log(traceback.format_exc())
            raise RuntimeError(f"PTF query failed: {e}")

    # Filter out bad data
    ptf = ptf[np.isfinite(ptf['mag'])]
    ptf = ptf[np.isfinite(ptf['magerr'])]
    ptf = ptf[ptf['magerr'] > 0]
    ptf = ptf[ptf['magerr'] < 1.0]

    log(f"{len(ptf)} data points after error filtering")

    if not len(ptf):
        log("No valid PTF data points after filtering")
        return

    # Add time column for plotting
    ptf['time_obj'] = Time(ptf['mjd'], format='mjd')

    # Sort by time
    ptf.sort('mjd')

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'ptf_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        # Plot by filter
        unique_filters = np.unique(ptf['filter'])

        for filt in unique_filters:
            idx = ptf['filter'] == filt
            if np.sum(idx):
                ax.errorbar(
                    ptf['time_obj'][idx].datetime,
                    ptf['mag'][idx],
                    ptf['magerr'][idx],
                    fmt='.',
                    label=filt
                )

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        if len(unique_filters) > 1:
            ax.legend()

        ax.set_ylabel('Magnitude')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - Palomar Transient Factory")

    log("PTF lightcurve plot saved to file:ptf_lc.png")

    # Save data
    # Remove time_obj column (not serializable to VOTable)
    ptf_save = ptf[[_ for _ in ptf.columns if _ != 'time_obj']]

    ptf_save.write(
        os.path.join(basepath, 'ptf.vot'),
        format='votable', overwrite=True
    )
    ptf_save.write(
        os.path.join(basepath, 'ptf.txt'),
        format='ascii.commented_header', overwrite=True
    )
    log("PTF data written to file:ptf.vot")
    log("PTF data written to file:ptf.txt")
