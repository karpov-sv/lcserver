"""Palomar Transient Factory lightcurve acquisition module.

Acquires PTF optical lightcurves.
"""

import os
import numpy as np

from astropy.table import Table
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord


# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query, irsa_client,
                    log_bands, log_conversion, assumed_color, r_to_g,
                    R_TO_G_FORMULA)


@survey_source(
    name='Palomar Transient Factory',
    short_name='PTF',
    state_acquiring='acquiring PTF lightcurve',
    state_acquired='PTF lightcurve acquired',
    log_file='ptf.log',
    output_files=['ptf.log', 'ptf_lc.png', 'ptf.vot', 'ptf.txt'],
    button_text='Get PTF lightcurve',
    help_text='Palomar Transient Factory optical survey',
    order=11,
    # Lightcurve metadata
    votable_file='ptf.vot',
    lc_bands=[
        surveys.band('g', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='g',
                     color='#17becf', note='as reported by PTF',
                     combined=True),
        surveys.band('R', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='R',
                     color='#bcbd22', note='as reported by PTF'),
        surveys.band('Ha', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='Ha',
                     color='#e377c2', note='as reported by PTF'),
        surveys.band('g (from R)', 'mag_g_from_R', 'magerr', surveys.BAND_DERIVED,
                     filter_column='filter', filter_value='R',
                     color='#dbdb8d',
                     note='R taken as SDSS r and moved onto g using an '
                          'assumed g - r',
                     combined=True),
    ],
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

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('ptf'), basepath=basepath)

    # Get coordinates
    ra = config.get('target_ra')
    dec = config.get('target_dec')

    if ra is None or dec is None:
        log("Error: target_ra and target_dec are required for PTF query")
        raise RuntimeError("Coordinates required for PTF query")

    # Query with caching
    ptf_sr = config.get('ptf_sr', 2.0)
    cache_name = f"ptf_{ra:.4f}_{dec:.4f}_{ptf_sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'Palomar Transient Factory', refresh=refresh_cache) as cache:
        if not cache.hit:
            # Query PTF catalog - only if not cached
            try:
                target = SkyCoord(ra, dec, unit='deg')
                table = irsa_client().query_region(
                    coordinates=target,
                    spatial='Cone',
                    catalog='ptf_lightcurves',
                    radius=ptf_sr * u.arcsec
                )

                if not len(table):
                    log(f"Warning: No PTF data points found within {ptf_sr:.1f} arcsec")
                    return

                # Save to cache
                cache.save(table)

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise SourceError("could not query PTF - "
                                  f"{type(e).__name__}: {e}")

        # Use cached or freshly queried data
        table = cache.data

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
        log("Warning: No PTF data points remaining after quality filtering")
        return

    log(f"Found {len(table)} PTF data points after filtering")
    ptf = table

    # Filter out bad data
    ptf = ptf[np.isfinite(ptf['mag'])]
    ptf = ptf[np.isfinite(ptf['magerr'])]
    ptf = ptf[ptf['magerr'] > 0]
    ptf = ptf[ptf['magerr'] < 1.0]

    log(f"{len(ptf)} data points after error filtering")

    log_conversion(
        log, 'PTF',
        'no conversion applied - each band is published as measured',
        {'colour term': ('none', 'aperture-corrected magnitudes from the PTF archive')},
        npoints=len(ptf),
    )

    # PTF's g needs nothing to join the combined light curve; its R does, and
    # is where the bulk of the points are. The Mould R it observes in sits
    # close enough to SDSS r to be converted as one, which is an approximation
    # on top of the assumed colour, though a small one beside it.
    g_minus_r, g_minus_r_origin = assumed_color(config, 'g_minus_r')

    idx_R = ptf['filter'] == 'R'
    ptf['mag_g_from_R'] = np.nan
    ptf['mag_g_from_R'][idx_R] = r_to_g(ptf['mag'][idx_R], g_minus_r)

    n_R = int(np.sum(idx_R))

    if n_R:
        log_conversion(
            log, 'PTF',
            R_TO_G_FORMULA,
            {'(g - r)': (g_minus_r, g_minus_r_origin),
             'R': ('taken as SDSS r', 'PTF observes in Mould R, which is close')},
            npoints=n_R,
            note='only so that the R points reach the combined light curve; '
                 'the native bands are untouched',
        )

    log_bands(log, 'PTF', [
        {'label': str(fn), 'kind': 'native',
         'npoints': int(np.sum(ptf['filter'] == fn)),
         'note': 'as reported by PTF'}
        for fn in np.unique(ptf['filter'])
    ] + [
        {'label': 'g (from R)', 'kind': 'derived', 'npoints': n_R,
         'note': 'R on the common g scale'},
    ])

    if not len(ptf):
        log("Warning: No valid PTF data points after filtering")
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
