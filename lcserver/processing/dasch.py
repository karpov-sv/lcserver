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
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    quality_field, quality_level, log_bands, log_conversion,
                    assumed_color, b_to_g, B_TO_G_FORMULA,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


# What DASCH found wrong with the local-bin calibration - the patch of plate
# the star's magnitude was measured against - from the DR7 description of the
# photometric calibration. These speak against the calibration rather than
# against the star, and the survey acts on them itself.
DASCH_REJECT_FLAGS = [
    (0x1, 'HIZOUT', 'the local correction exceeds 0.5 mag, or its error 0.7'),
    (0x2, 'MEDIAN', 'the median star in the bin is within 0.5 mag of the limit'),
    (0x4, 'DRAD', 'astrometric scatter beyond 90 arcsec, or 3 pixels'),
]

# What the pipeline noted about the measurement itself, from the AFlags
# enumeration daschlab publishes. Its table numbers the bits from one while the
# values are 1 << (n - 1); these are the values, taken from the source.
DASCH_AFLAGS = [
    (1 << 6,  'HIGH_BACKGROUND', 'high background at the object'),
    (1 << 7,  'BAD_PLATE_QUALITY', 'the plate fails the general quality checks'),
    (1 << 8,  'MULT_EXP_UNMATCHED', 'unmatched, on a multiple-exposure plate'),
    (1 << 9,  'UNCERTAIN_DATE', 'the time is too uncertain to correct extinction'),
    (1 << 10, 'MULT_EXP_BLEND', 'a blend, on a multiple-exposure plate'),
    (1 << 11, 'LARGE_ISO_RMS', 'suspiciously large isophotal RMS'),
    (1 << 12, 'LARGE_LOCAL_SMOOTH_RMS', 'suspiciously large local-binning RMS'),
    (1 << 13, 'CLOSE_TO_LIMITING', 'too close to the local limiting magnitude'),
    (1 << 14, 'RADIAL_BIN_9', 'close to the plate edge'),
    (1 << 15, 'BIN_DRAD_UNKNOWN', 'the spatial bin has no measured drad'),
    (1 << 19, 'UNCERTAIN_CATALOG_MAG', 'the catalogue star is uncertain or variable'),
    (1 << 20, 'CASE_B_BLEND', 'several catalogue entries for one imaged star'),
    (1 << 21, 'CASE_C_BLEND', 'several imaged stars for one catalogue entry'),
    (1 << 22, 'CASE_BC_BLEND', 'entries and imaged stars mixed up together'),
    (1 << 23, 'LARGE_DRAD', 'drad large for its bin, or the bin is bad'),
    (1 << 24, 'PICKERING_WEDGE', 'a Pickering wedge image'),
    (1 << 25, 'SUSPECTED_DEFECT', 'a suspected plate defect'),
    (1 << 26, 'SXT_BLEND', 'SExtractor calls it a blend'),
    (1 << 27, 'REJECTED_BLEND', 'a rejected blend'),
    (1 << 28, 'LARGE_SMOOTHING_CORRECTION', 'suspiciously large smoothing correction'),
    (1 << 29, 'TOO_BRIGHT', 'too bright for the calibration to be accurate'),
    (1 << 30, 'LOW_ALTITUDE', 'within 23.5 degrees of the horizon'),
]

# Which is set on every measurement of a variable star - it describes the
# catalogue entry the plate was calibrated against, not the plate - and so is
# no reason to drop anything here. Cutting on it once emptied the light curve
# of every variable, which is what these are for.
DASCH_CATALOG_MAG = 1 << 19

# Beyond this the local calibration is scattered enough that the magnitude
# means little. Where DASCH could not compute it at all the star was too bright
# for its bin, which the flags say in as many words: on the star this was
# written for, the points with no value are exactly the ones flagged TOO_BRIGHT.
DASCH_MAX_LOCAL_RMS = 0.4


@survey_source(
    name='DASCH',
    short_name='DASCH',
    state_acquiring='acquiring DASCH lightcurve',
    state_acquired='DASCH lightcurve acquired',
    log_file='dasch.log',
    output_files=['dasch.log', 'dasch_lc.png', 'dasch.vot', 'dasch.txt'],
    button_text='Get DASCH lightcurve',
    form_fields={
        'dasch_quality': quality_field({
            QUALITY_STANDARD: 'Drop what the pipeline flagged as well',
            QUALITY_RELAXED: "Drop what DASCH's own calibration rejects",
            QUALITY_PUBLISHED: 'None - every plate with a calibrated magnitude',
        }),
    },
    help_text='Harvard plate archive (historical data)',
    order=40,
    # Lightcurve metadata
    votable_file='dasch.vot',
    lc_bands=[
        surveys.band('phot', 'magcal_magdep', 'magerr', surveys.BAND_NATIVE,
                     color='#d62728',
                     note='DASCH calibrated photographic magnitudes, as reported'),
        surveys.band('g (conv.)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     color='#ff9896',
                     note='the plates taken as Johnson B and put on the common '
                          'g scale using an assumed g - r',
                     combined=True),
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

    quality = quality_level(config, 'dasch')

    # DASCH decides for itself which local-bin calibrations to stand behind,
    # and reject_flag carries that verdict as a bitfield; zero means it is
    # happy with the point.
    if 'reject_flag' in dasch.colnames:
        ncal = len(dasch)
        flags = np.asarray(dasch['reject_flag']).astype(np.int64)
        rejected = flags != 0

        if np.any(rejected):
            told = ', '.join(f"{name} {int(np.sum((flags & value) != 0))}"
                             for value, name, _ in DASCH_REJECT_FLAGS
                             if np.any(flags & value))
            log(f"  DASCH rejects {int(np.sum(rejected))} calibrations: {told}")

            for value, name, meaning in DASCH_REJECT_FLAGS:
                if np.any(flags & value):
                    log(f"    {name:8s} {meaning}")

        if quality != QUALITY_PUBLISHED:
            dasch = dasch[~rejected]
            log(f"  {len(dasch)} of {ncal} accepted by DASCH")
    else:
        log("Warning: cached data predates the quality flags, so nothing is "
            "filtered on quality. Re-query the survey to apply them.")

    # What the pipeline made of the measurement itself, which the survey
    # leaves to the reader. All but one of these bits mark something wrong
    # with the plate or with the extraction; the exception describes the
    # catalogue star, is set on every measurement of a variable, and is the
    # reason an earlier cut on this column emptied the light curve.
    if quality == QUALITY_STANDARD and 'AFLAGS' in dasch.colnames:
        aflags = np.asarray(dasch['AFLAGS']).astype(np.int64)
        bad = (aflags & ~np.int64(DASCH_CATALOG_MAG)) != 0

        if np.any(aflags):
            log("  what the pipeline flagged, of these:")

            for value, name, meaning in DASCH_AFLAGS:
                count = int(np.sum((aflags & value) != 0))

                if count:
                    log(f"    {name:26s} {count:5d}  {meaning}"
                        + ('  [kept]' if value == DASCH_CATALOG_MAG else ''))

        if np.any(bad):
            log(f"Warning: dropping {int(np.sum(bad))} points the pipeline "
                f"flagged, of {len(dasch)}")
            dasch = dasch[~bad]

    # The scatter of the local calibration, where DASCH could compute one. It
    # grades the points beyond what the flags say, and where it is missing the
    # star was too bright for its bin - which is not a measurement to keep.
    if quality == QUALITY_STANDARD and 'magcal_local_rms' in dasch.colnames:
        rms = np.asarray(dasch['magcal_local_rms'], dtype=float)
        coarse = ~(rms < DASCH_MAX_LOCAL_RMS)

        if np.any(coarse):
            log(f"Warning: dropping {int(np.sum(coarse))} more whose local "
                f"calibration scatters past {DASCH_MAX_LOCAL_RMS:.1f} mag, or "
                f"could not be made at all")
            dasch = dasch[~coarse]

    log(f"  {len(dasch)} plates left")

    # magcal_local_rms is a scatter estimate that DR7 leaves empty for a good
    # fraction of the points, so it is no use either as a cut - a missing value
    # would silently drop the point - or as an error bar
    if 'magcal_local_error' in dasch.colnames:
        dasch['magerr'] = dasch['magcal_local_error']
    else:
        dasch['magerr'] = dasch['magcal_local_rms']

    # The plates are blue-sensitive and DASCH calibrates them against B, so the
    # way onto the common g scale is the B to g relation - not the identity
    # this column used to hold, which put a B magnitude on a g axis and called
    # it converted. What remains uncorrected is the colour term of the emulsion
    # itself, which varies from plate to plate and is larger than the relation's
    # own scatter, so this is the roughest of the conversions by some way.
    g_minus_r, g_minus_r_origin = assumed_color(config, 'g_minus_r')
    dasch['mag_g'] = b_to_g(np.asarray(dasch['magcal_magdep'], dtype=float),
                            g_minus_r)

    log_conversion(
        log, 'DASCH',
        B_TO_G_FORMULA,
        {'(g - r)': (g_minus_r, g_minus_r_origin),
         'emulsion colour term': ('not corrected for',
                                  'varies plate to plate, and is not published')},
        npoints=len(dasch),
        note='the photographic band is treated as Johnson B, which it resembles '
             'but is not; the native magnitudes are kept as the phot band',
    )

    log_bands(log, 'DASCH', [
        {'label': 'phot', 'kind': 'native', 'npoints': len(dasch),
         'note': 'calibrated photographic magnitudes'},
        {'label': 'g (conv.)', 'kind': 'derived', 'npoints': len(dasch),
         'note': 'the plates taken as B and put on the common g scale'},
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
