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
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    quality_field, quality_level, log_bands, log_conversion,
                    assumed_color, v_to_g, V_TO_G_FORMULA,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


# A non-detection is reported as a magnitude equal to the image's limit, with
# this error and no flux to speak of - and is labelled good all the same: of
# the eighty-four in the first target this was tried on, seventy-two said 'G'.
# Reading the columns for what they say is not a judgement, so this much is
# done at every level of filtering.
ASAS_NODATA_ERR = 99.0

# How much shallower than usual an image may be before what it says about the
# star is not worth having. Cloud and haze cost depth, and a star measured
# through them comes out too faint - the two correlate at -0.6 in the g band.
# Half a magnitude is the knee: a quarter gains two hundredths of scatter and
# costs another few per cent of the points.
ASAS_DEPTH_MARGIN = 0.5

# How much worse than usual the seeing may be before a saturated star's
# photometry is not to be trusted, in the pixels the survey quotes it in. Only
# applied where the star is saturated: elsewhere it costs a quarter of the
# points and buys almost nothing.
ASAS_SEEING_MARGIN = 0.35

# Where the survey's own catalogue papers put saturation. Nothing marks a
# saturated measurement, and the quoted errors say nothing about it - photon
# noise on a clipped profile, understating the real scatter three hundred
# times - so a word in the log is all that can be offered.
ASAS_SATURATION = 11.5


@survey_source(
    name='ASAS-SN',
    short_name='ASAS-SN',
    state_acquiring='acquiring ASAS-SN lightcurve',
    state_acquired='ASAS-SN lightcurve acquired',
    log_file='asas.log',
    output_files=['asas.log', 'asas_lc.png', 'asas.vot', 'asas.txt'],
    button_text='Get ASAS-SN lightcurve',
    form_fields={
        'asas_quality': quality_field({
            QUALITY_STANDARD: 'Drop what was measured through cloud as well',
            QUALITY_RELAXED: 'Drop what the survey calls bad',
            QUALITY_PUBLISHED: 'None - every detection as published',
        }),
    },
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
                     note='V and g put on a common g scale using an assumed g - r',
                     combined=True),
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

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('asas'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    # Cache raw query results before color conversion
    ra = config.get('target_ra')
    dec = config.get('target_dec')
    asas_sr = config.get('asas_sr', 10.0)
    cache_name = f"asas_{ra:.4f}_{dec:.4f}_{asas_sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'ASAS-SN', refresh=refresh_cache) as cache:
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
            except Exception as e:
                import traceback
                traceback.print_exc()
                # Raised rather than passed over: what followed was a report of
                # no data, which is what an archive that is down looks like
                # from the outside
                raise SourceError("could not query ASAS-SN - "
                                  f"{type(e).__name__}: {e}")

            if not lcq or not len(lcq.data):
                log("Warning: No ASAS-SN data found")
                return

            # Cache raw query results
            asas_raw = Table.from_pandas(lcq.data)
            cache.save(asas_raw)

        # Use cached or freshly queried raw data
        asas = cache.data

    log(f"{len(asas)} ASAS-SN data points found")

    quality = quality_level(config, 'asas')

    # What the survey did not measure at all. A non-detection carries the
    # image's limiting magnitude in the magnitude column, so left in it reads
    # as the star having faded by several magnitudes.
    good = np.isfinite(asas['mag']) & (asas['mag_err'] < ASAS_NODATA_ERR)

    if 'flux' in asas.colnames:
        good &= asas['flux'] > 0

    if not np.all(good):
        log(f"  {int(np.sum(~good))} of them are not detections but upper limits"
            + (f", {int(np.sum(~good & (asas['quality'] == 'G')))} of those"
               " labelled good" if 'quality' in asas.colnames else ''))

    if quality != QUALITY_PUBLISHED:
        # The survey's own letter, and the error cut this source has always
        # applied - which says little about a saturated star, whose errors are
        # photon noise on a profile that lost its peak
        good &= (asas['quality'] == 'G') & (asas['mag_err'] < 0.05)

    # Band by band, since the two cameras differ in what they reach and in what
    # they saturate at, and since the star has a different brightness in each
    for fn in ['g', 'V']:
        here = good & (asas['phot_filter'] == fn)

        if np.sum(here) < 10:
            continue

        # An image taken through cloud reaches shallower and measures the star
        # too faint. Judged against the depth this band usually reaches on this
        # star rather than against a fixed number.
        if quality == QUALITY_STANDARD and 'limit' in asas.colnames:
            usual = np.median(asas['limit'][here])
            shallow = here & (asas['limit'] < usual - ASAS_DEPTH_MARGIN)

            if np.any(shallow):
                log(f"Warning: dropping {int(np.sum(shallow))} {fn} points from "
                    f"images over {ASAS_DEPTH_MARGIN:.1f} mag shallower than "
                    f"the usual {usual:.1f}")
                good &= ~shallow
                here &= ~shallow

        # Nothing marks saturation, and nothing can undo it: the star has lost
        # flux to a clipped profile, so what is left scatters to the faint side
        # only. Said plainly, as the light curve looks like variability.
        median = np.median(np.asarray(asas['mag'])[here])

        if median >= ASAS_SATURATION:
            continue

        log(f"Warning: {fn} = {median:.1f} is past the {ASAS_SATURATION:.1f} "
            f"ASAS-SN saturates at - this scatter is the survey, not the star")

        # Where the survey corrects a saturated star it corrects a bleed trail,
        # and in poor seeing that correction is the likeliest to fail: on the
        # star this was found on, the points a magnitude and more below the
        # rest were taken in the worst seeing of the run and carry a tenth of
        # its flux. Only for a saturated star, and only at this level: seeing
        # costs an ordinary star nothing, and a third of the points is too much
        # to pay for nothing.
        if quality == QUALITY_STANDARD and 'fwhm' in asas.colnames:
            # The survey serves the seeing as text, unlike everything beside it
            try:
                fwhm = np.asarray(asas['fwhm'], dtype=float)
            except (TypeError, ValueError):
                continue

            usual = np.median(fwhm[here])
            blurred = here & (fwhm > usual + ASAS_SEEING_MARGIN)

            if np.any(blurred):
                log(f"Warning: dropping {int(np.sum(blurred))} of those {fn} "
                    f"points taken in seeing worse than "
                    f"{usual + ASAS_SEEING_MARGIN:.2f} px, where that "
                    f"correction fails")
                good &= ~blurred

    for fn in ['g', 'V']:
        idx = asas['phot_filter'] == fn
        log(f"  {fn}: {np.sum(idx)} total, {np.sum(idx & good)} kept")

    log("Earliest: ", Time(np.min(asas['jd']), format='jd').datetime.strftime('%Y-%m-%s %H:%M:%S'))
    log("  Latest: ", Time(np.max(asas['jd']), format='jd').datetime.strftime('%Y-%m-%s %H:%M:%S'))

    asas['time'] = Time(asas['jd'], format='jd')
    asas['mjd'] = asas['time'].mjd

    # Native measurements, kept per band, and the common g scale built from them
    asas['mag_V'] = np.nan
    asas['mag_g_nat'] = np.nan
    asas['mag_g'] = np.nan

    g_minus_r, g_minus_r_origin = assumed_color(config, 'g_minus_r')

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'asas_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        idx = good

        idx_V = idx & (asas['phot_filter'] == 'V')
        idx_g = idx & (asas['phot_filter'] == 'g')

        # Native magnitudes, exactly as ASAS-SN reports them
        asas['mag_V'][idx_V] = asas['mag'][idx_V]
        asas['mag_g_nat'][idx_g] = asas['mag'][idx_g]

        # ... and the same points carried onto a common g scale
        asas['mag_g'][idx_V] = v_to_g(asas['mag'][idx_V], g_minus_r)
        asas['mag_g'][idx_g] = asas['mag'][idx_g] - 0.013 - 0.145*g_minus_r - 0.019*g_minus_r**2

        log_conversion(
            log, 'ASAS-SN',
            V_TO_G_FORMULA,
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
