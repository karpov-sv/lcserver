"""DASCH lightcurve acquisition module.

Acquires DASCH (Digital Access to a Sky Century @ Harvard) historical lightcurves
from Harvard plate archive.
"""

import os
import csv
import io
import numpy as np
import requests

from astropy.table import Table
from astropy.time import Time

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import cleanup_paths


@survey_source(
    name='DASCH',
    short_name='DASCH',
    state_acquiring='acquiring DASCH lightcurve',
    state_acquired='DASCH lightcurve acquired',
    log_file='dasch.log',
    output_files=['dasch.log', 'dasch_lc.png'],
    button_text='Get DASCH lightcurve',
    help_text='Harvard plate archive (historical data)',
    order=40,
    # Lightcurve metadata
    votable_file='dasch.vot',
    lc_mag_column='mag_g',
    lc_err_column='magerr',
    lc_color='#d62728',
    lc_mode='magnitude',
    lc_short=False,
)
def target_dasch(config, basepath=None, verbose=True, show=False):
    """
    Get DASCH lightcurve.

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
    cleanup_paths(get_output_files('dasch'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    cachename = f"dasch_{config['target_ra']:.4f}_{config['target_dec']:.4f}.vot"

    if os.path.exists(os.path.join(basepath, 'cache', cachename)):
        log(f"Loading DASCH lightcurve from the cache")
        dasch = Table.read(os.path.join(basepath, 'cache', cachename))
    else:
        dasch_sr = config.get('dasch_sr', 5.0)

        log(f"Requesting DASCH lightcurve for {config['target_name']} within {dasch_sr:.1f} arcsec")

        # New DASCH DR7 API
        base_url = "https://api.starglass.cfa.harvard.edu/public"
        refcat = "atlas"  # Could also use "apass"

        # Step 1: Query catalog to find source
        querycat_url = f"{base_url}/dasch/dr7/querycat"
        querycat_payload = {
            "refcat": refcat,
            "ra_deg": config['target_ra'],
            "dec_deg": config['target_dec'],
            "radius_arcsec": dasch_sr
        }

        log(f"Querying DASCH catalog at RA={config['target_ra']:.4f}, Dec={config['target_dec']:.4f}")

        try:
            response = requests.post(querycat_url, json=querycat_payload, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f'Error querying DASCH catalog: {e}')

        # Parse CSV response
        csv_lines = response.json()
        if not csv_lines or len(csv_lines) < 2:
            raise RuntimeError('No sources found in DASCH catalog')

        # Parse CSV to table
        csv_text = '\n'.join(csv_lines)
        reader = csv.DictReader(io.StringIO(csv_text))
        sources = list(reader)

        if not sources:
            raise RuntimeError('No sources found in DASCH catalog')

        # Find closest source based on angular separation
        separations = []
        for src in sources:
            dra = float(src['dra_asec'])
            ddec = float(src['ddec_asec'])
            sep = np.sqrt(dra**2 + ddec**2)
            separations.append(sep)

        closest_idx = np.argmin(separations)
        source = sources[closest_idx]
        ref_number = int(source['ref_number'])
        gsc_bin_index = int(source['gsc_bin_index'])

        log(f"Found {len(sources)} sources, using closest one (sep={separations[closest_idx]:.2f} arcsec)")
        log(f"Source: ref_number={ref_number}, gsc_bin_index={gsc_bin_index}, stdmag={source.get('stdmag', 'N/A')}")

        # Step 2: Get lightcurve for the source
        lightcurve_url = f"{base_url}/dasch/dr7/lightcurve"
        lightcurve_payload = {
            "refcat": refcat,
            "ref_number": ref_number,
            "gsc_bin_index": gsc_bin_index
        }

        log(f"Requesting lightcurve data...")

        try:
            response = requests.post(lightcurve_url, json=lightcurve_payload, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f'Error downloading DASCH lightcurve: {e}')

        # Parse CSV response
        csv_lines = response.json()
        if not csv_lines or len(csv_lines) < 2:
            raise RuntimeError('No lightcurve data returned from DASCH')

        # Parse CSV to table
        csv_text = '\n'.join(csv_lines)
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)

        if not rows:
            raise RuntimeError('No lightcurve data points found')

        # Convert to astropy Table with proper column names
        # Note: API returns snake_case column names
        dasch = Table()

        # Parse required columns (handle empty values gracefully)
        dasch['ExposureDate'] = [float(row['date_jd']) if row['date_jd'] else np.nan for row in rows]
        dasch['magcal_magdep'] = [float(row['magcal_magdep']) if row['magcal_magdep'] else np.nan for row in rows]
        dasch['magcal_local_rms'] = [float(row['magcal_local_rms']) if row['magcal_local_rms'] else np.nan for row in rows]
        dasch['AFLAGS'] = [int(row['aflags']) if row['aflags'] else 0 for row in rows]

        # Optional: add more columns if available
        if 'limiting_mag_local' in rows[0]:
            dasch['limiting_mag_local'] = [float(row['limiting_mag_local']) if row['limiting_mag_local'] else np.nan for row in rows]

        # Filter out rows with invalid data
        dasch = dasch[np.isfinite(dasch['ExposureDate'])]

        try:
            os.makedirs(os.path.join(basepath, 'cache'))
        except:
            pass

        dasch.write(os.path.join(basepath, 'cache', cachename),
                    format='votable', overwrite=True)

    log(f"{len(dasch)} original data points")

    dasch['time'] = Time(dasch['ExposureDate'].value, format='jd')
    dasch.sort('time')

    dasch['mjd'] = dasch['time'].mjd

    dasch = dasch[dasch['magcal_magdep'] > 0]
    # t = t[t['BFLAGS'] & 0x10000 > 0]

    # Criteria from https://github.com/barentsen/did-tabbys-star-fade/blob/master/data/data-preprocessing.py
    # dasch = dasch[dasch['AFLAGS'] <= 9000]
    dasch = dasch[dasch['AFLAGS'] < 524288]
    # dasch = dasch[dasch['AFLAGS'] & 33554432 == 0]
    # dasch = dasch[dasch['AFLAGS'] & 1048576 == 0]
    dasch = dasch[dasch['magcal_local_rms'] < 1]
    # dasch = dasch[dasch['magcal_local_rms'] < 0.33]
    # dasch = dasch[dasch['magcal_magdep'] < dasch['limiting_mag_local'] - 0.2]

    dasch['mag_g'] = dasch['magcal_magdep']
    dasch['magerr'] = dasch['magcal_local_rms']

    log(f"{len(dasch)} data points after filtering")
    if not len(dasch):
        return

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'dasch_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        ax.errorbar(dasch['time'].datetime, dasch['mag_g'], dasch['magerr'], fmt='.', label='g')

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        # ax.legend()
        ax.set_ylabel('g')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - DASCH")

    # Time cannot be serialized to VOTable
    dasch[[_ for _ in dasch.columns if _ != 'time']].write(os.path.join(basepath, 'dasch.vot'),
                                                           format='votable', overwrite=True)
    dasch[[_ for _ in dasch.columns if _ != 'time']].write(os.path.join(basepath, 'dasch.txt'),
                                                           format='ascii.commented_header', overwrite=True)
    log("DASCH data written to file:dasch.vot")
    log("DASCH data written to file:dasch.txt")
