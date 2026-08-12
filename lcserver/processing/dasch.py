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

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import SourceError, cleanup_paths, cached_votable_query, log_bands, log_conversion


@survey_source(
    name='DASCH',
    short_name='DASCH',
    state_acquiring='acquiring DASCH lightcurve',
    state_acquired='DASCH lightcurve acquired',
    log_file='dasch.log',
    output_files=['dasch.log', 'dasch_lc.png', 'dasch.vot', 'dasch.txt'],
    button_text='Get DASCH lightcurve',
    help_text='Harvard plate archive (historical data)',
    order=40,
    # Lightcurve metadata
    votable_file='dasch.vot',
    lc_bands=[
        surveys.band('phot', 'magcal_magdep', 'magerr', surveys.BAND_NATIVE,
                     color='#d62728',
                     note='DASCH calibrated photographic magnitudes, as reported'),
    ],
    lc_mag_column='mag_g',
    lc_err_column='magerr',
    lc_color='#d62728',
    lc_mode='magnitude',
    lc_short=False,
    # Template metadata
    template_layout='with_cutout',
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

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('dasch'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    dasch_sr = config.get('dasch_sr', 5.0)
    cache_name = f"dasch_{ra:.4f}_{dec:.4f}_{dasch_sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'DASCH', refresh=refresh_cache) as cache:
        if not cache.hit:

            log(f"for {config['target_name']} within {dasch_sr:.1f} arcsec")

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
                raise SourceError('could not query the DASCH catalog - '
                                  f'{type(e).__name__}: {e}')

            # Parse CSV response
            csv_lines = response.json()
            if not csv_lines or len(csv_lines) < 2:
                log('Warning: No sources found in DASCH catalog')
                return

            # Parse CSV to table
            csv_text = '\n'.join(csv_lines)
            reader = csv.DictReader(io.StringIO(csv_text))
            sources = list(reader)

            if not sources:
                log('Warning: No sources found in DASCH catalog')
                return

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
                raise SourceError('could not download the DASCH lightcurve - '
                                  f'{type(e).__name__}: {e}')

            # Parse CSV response
            csv_lines = response.json()
            if not csv_lines or len(csv_lines) < 2:
                log('Warning: No lightcurve data returned from DASCH')
                return

            # Parse CSV to table
            csv_text = '\n'.join(csv_lines)
            reader = csv.DictReader(io.StringIO(csv_text))
            rows = list(reader)

            if not rows:
                log('Warning: No lightcurve data points found')
                return

            # Convert to astropy Table with proper column names
            # Note: API returns snake_case column names
            dasch = Table()

            # Parse required columns (handle empty values gracefully)
            dasch['ExposureDate'] = [float(row['date_jd']) if row['date_jd'] else np.nan for row in rows]
            dasch['magcal_magdep'] = [float(row['magcal_magdep']) if row['magcal_magdep'] else np.nan for row in rows]
            dasch['magcal_local_rms'] = [float(row['magcal_local_rms']) if row['magcal_local_rms'] else np.nan for row in rows]
            dasch['AFLAGS'] = [int(row['aflags']) if row['aflags'] else 0 for row in rows]

            # DR7 judges its own measurements and reports the verdict, which is
            # what the quality cut below rests on
            dasch['reject_flag'] = [int(row['reject_flag']) if row.get('reject_flag') else 0
                                    for row in rows]
            # The per-point uncertainty. Unlike magcal_local_rms, which is a
            # scatter estimate missing for about two fifths of the points, this
            # one is given for every calibrated measurement.
            dasch['magcal_local_error'] = [float(row['magcal_local_error'])
                                           if row.get('magcal_local_error') else np.nan
                                           for row in rows]

            # Optional: add more columns if available
            if 'limiting_mag_local' in rows[0]:
                dasch['limiting_mag_local'] = [float(row['limiting_mag_local']) if row['limiting_mag_local'] else np.nan for row in rows]

            # Filter out rows with invalid data
            dasch = dasch[np.isfinite(dasch['ExposureDate'])]

            cache.save(dasch)

        dasch = cache.data

    log(f"{len(dasch)} original data points")

    dasch['time'] = Time(dasch['ExposureDate'].value, format='jd')
    dasch.sort('time')

    dasch['mjd'] = dasch['time'].mjd

    nraw = len(dasch)

    # Only the plates on which the star was actually measured
    dasch = dasch[dasch['magcal_magdep'] > 0]
    log(f"  {len(dasch)} of {nraw} plates carry a calibrated magnitude")

    # DASCH decides for itself which of its measurements to stand behind, and
    # reject_flag carries that verdict; zero means it is happy with the point.
    #
    # This used to cut on AFLAGS instead, keeping rows below 1 << 19 - a
    # criterion borrowed from a script written against an older data release.
    # Under DR7 that bit is set on precisely the points that *have* a
    # calibrated magnitude, so for a star whose plates carry it the cut threw
    # away the entire light curve and kept nothing else.
    if 'reject_flag' in dasch.colnames:
        ncal = len(dasch)
        rejected = np.asarray(dasch['reject_flag']) != 0
        if np.any(rejected):
            codes, counts = np.unique(np.asarray(dasch['reject_flag'])[rejected],
                                      return_counts=True)
            log(f"  DASCH rejects {int(np.sum(rejected))} of them: "
                + ', '.join(f"{c} for reason {int(v)}" for v, c in zip(codes, counts)))
        dasch = dasch[~rejected]
        log(f"  {len(dasch)} of {ncal} accepted by DASCH")
    else:
        log("Warning: cached data predates the quality flags, so nothing is "
            "filtered on quality. Re-query the survey to apply them.")

    # magcal_local_rms is a scatter estimate that DR7 leaves empty for a good
    # fraction of the points, so it is no use either as a cut - a missing value
    # would silently drop the point - or as an error bar
    if 'magcal_local_error' in dasch.colnames:
        dasch['magerr'] = dasch['magcal_local_error']
    else:
        dasch['magerr'] = dasch['magcal_local_rms']

    # No conversion is available for the photographic plates, so the common
    # g column is the native magnitude under another name. Said plainly here,
    # because the name would otherwise suggest a conversion that never happened.
    dasch['mag_g'] = dasch['magcal_magdep']

    log_conversion(
        log, 'DASCH',
        'g = magcal_magdep   (no conversion applied)',
        {'colour term': ('none available', 'photographic plates, blue-sensitive')},
        npoints=len(dasch),
        note='the photographic band is closer to B than to g; the combined light '
             'curve uses these magnitudes unchanged',
    )

    log_bands(log, 'DASCH', [
        {'label': 'phot', 'kind': 'native', 'npoints': len(dasch),
         'note': 'calibrated photographic magnitudes'},
    ])

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
