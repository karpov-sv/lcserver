"""TESS lightcurve acquisition module.

Acquires TESS (Transiting Exoplanet Survey Satellite) lightcurves.
"""

import os
import shutil
import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import (break_at_gaps, cleanup_paths, cached_lightkurve_search,
                    download_tars_lightcurve, download_tequila_lightcurve,
                    drop_mast_downloads, mast_download_dir,
                    mission_quality_mask, log_quality, quality_field,
                    quality_level,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


# The reductions TESS light curves come from, in the order they are preferred
# when the choice is left open. Two more can be asked for by name but are not
# among them, as each answers a narrower question than "the light curve of
# this star". TEQUILA covers only the prime mission at half-hour cadence, its
# timestamps sit a quarter of an hour from everyone else's, and where it
# overlaps QLP it measures the same thing slightly less precisely; what it has
# that they do not is the faint end, as it looks for what varies in the
# difference images rather than at what the TIC lists, and so reaches the
# variables and the transients the others had no target for. TARS detrends
# against the other pixels of its own camera rather than fitting the star's
# own trend away, which is what keeps a rotation signal that spans days
# intact where the others flatten it - at the price of a bare light curve,
# with no uncertainties and no flags.
TESS_AUTHORS = ['TESS-SPOC', 'QLP', 'SPOC']

# The pipelines read here rather than by lightkurve, which cannot open either
TESS_OWN_READERS = {'TEQUILA': download_tequila_lightcurve,
                    'TARS': download_tars_lightcurve}


@survey_source(
    name='TESS',
    short_name='TESS',
    state_acquiring='acquiring TESS lightcurves',
    state_acquired='TESS lightcurves acquired',
    log_file='tess.log',
    output_files=['tess.log', 'tess_lc_*.vot', 'tess_lc_*.txt', 'tess_lc_*.png'],
    button_text='Get TESS lightcurves',
    form_fields={
        'tess_author': {
            'type': 'choice',
            'label': 'Pipeline',
            'choices': [('auto', 'Best available'),
                        ('TESS-SPOC', 'TESS-SPOC'),
                        ('QLP', 'QLP'),
                        ('SPOC', 'SPOC'),
                        ('TEQUILA', 'TEQUILA (faint variables, sectors 1-26)'),
                        ('TARS', 'TARS (rotation, flux only)')],
            'initial': 'auto',
            'required': False,
        },
        'tess_quality': quality_field({
            QUALITY_STANDARD: 'Drop every cadence the mission flagged',
            QUALITY_RELAXED: 'Drop only what the mission calls unusable',
            QUALITY_PUBLISHED: 'None - every cadence as published',
        }),
    },
    help_text='NASA TESS space telescope',
    order=30,
    # Lightcurve metadata
    votable_file='tess_lc_*.vot',
    lc_flux_column='flux',
    lc_err_column='flux_err',
    lc_quality_column='quality',
    lc_color='#e74c3c',
    lc_mode='flux',
    lc_short=False,
    lc_segment_name='Sector',
    # Template metadata
    template_layout='complex',
    show_cutout=True,
    cutout_skyview='TESS',
    cutout_hips='DSS2/color',
    cutout_fov=0.06,
    additional_plots=['tess_lc_*.png'],
)
def target_tess(config, basepath=None, verbose=True, show=False):
    """
    Get TESS lightcurves.

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

    # Two caches here: the search below, kept as a VOTable like every other
    # source's query, and the files themselves, which lightkurve skips if it
    # has them already. The flag is read rather than consumed - see the other
    # sources - and drops both.
    refresh_cache = bool(config.get('refresh_cache', False))
    if refresh_cache:
        drop_mast_downloads(basepath, 'tess', log)

    # Cleanup stale plots
    cleanup_paths(get_output_files('tess'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    tess_sr = config.get('tess_sr', 10.0)

    # Which reduction to take, in order of preference. Asking for one by name
    # keeps the others out, so that a sector is either that pipeline's or absent
    # rather than quietly falling back to another.
    author = str(config.get('tess_author', 'auto'))
    authors = TESS_AUTHORS if author == 'auto' else [author]

    ra, dec = config.get('target_ra'), config.get('target_dec')

    log(f"Requesting TESS data for {config['target_name']} within {tess_sr:.1f} arcsec"
        + ("" if author == 'auto' else f", from {author} only"))

    # The pipeline is chosen below, among what was found, so it is no part of
    # the search and every choice of it shares the one cache
    cache_name = f"tess_search_{ra:.4f}_{dec:.4f}_{tess_sr}.vot"
    res = cached_lightkurve_search(cache_name, basepath, log, 'TESS',
                                   refresh=refresh_cache,
                                   target=SkyCoord(ra, dec, unit='deg'),
                                   radius=tess_sr*u.arcsec, mission='TESS')

    if len(res):
        # Filter out CDIPS products
        res = res[res.author != 'CDIPS']

    if not len(res):
        log("Warning: No TESS data found")
        return
    else:
        log(f"{len(res)} data products found")

    # What counts as one target among the products found. Most pipelines name
    # the star and keep that name across its sectors, so its sectors gather
    # under it. TEQUILA names the position on the CCD it measured instead -
    # s0014-cam3-ccd2-x0284-y1546, a different name in every sector - so
    # grouping on what it publishes would make each sector a target of its own.
    tnames = np.array(res.target_name, dtype=object)
    tnames[np.asarray(res.author) == 'TEQUILA'] = 'TEQUILA source'

    for tname in np.unique(tnames):
        idx = tnames == tname
        # The nearest of them, as a group holds one match per sector rather
        # than one match
        log(f"\nTESS target {tname} at {np.min(res[idx].distance.value):.1f} arcsec")

        for mission in np.unique(res[idx].mission):
            idx1 = idx & (res.mission == mission)
            tmin = Time(np.min(res.table['t_min'][idx1]), format='mjd')
            tmax = Time(np.max(res.table['t_max'][idx1]), format='mjd')
            log(f"  {mission}: {tmin.datetime.strftime('%Y-%m-%d')} - {tmax.datetime.strftime('%Y-%m-%d')}")

            for prod in res[idx1].table:
                log(f"    {prod['author']:10s} {prod['exptime']} s exp")

            # Write one representative lightcurve per sector
            for author in authors:
                idx2 = idx1 & (res.author == author)
                is_done = False

                for row in res[idx2]:
                    if author in TESS_OWN_READERS:
                        lc = TESS_OWN_READERS[author](
                            row, mast_download_dir(basepath, 'tess'))
                    else:
                        lc = row.download(download_dir=mast_download_dir(basepath, 'tess'))

                    if not lc:
                        continue

                    # What the pipeline made of the sector itself, where it
                    # says so - TARS looks for rotation and publishes the
                    # period it found beside the light curve
                    if lc.meta.get('PERIOD'):
                        log(f"    {lc.meta['AUTHOR']} period {lc.meta['PERIOD']:.4f} d, "
                            f"amplitude {lc.meta['PERIOD_AMPLITUDE']:.4f}, "
                            f"S/N {lc.meta['PERIOD_SNR']:.1f}")

                    # Plot the lightcurve
                    lcname = f"tess_lc_{lc.meta['SECTOR']}_{lc.meta['AUTHOR']}_{row.exptime[0].value:.0f}.png"
                    with plots.figure_saver(os.path.join(basepath, lcname), figsize=(8, 4), show=show) as fig:
                        ax = fig.add_subplot(1, 1, 1)

                        time = lc.time.btjd
                        flux = lc.normalize().flux
                        # Not every pipeline reports one - K2SFF publishes
                        # eight columns and no quality among them
                        if 'quality' in lc.colnames:
                            keep = mission_quality_mask(
                                lc['quality'], 'TESS',
                                quality_level(config, 'tess'))
                            # What the mask costs, rather than what it marks:
                            # the SPOC pipelines have already emptied the flux
                            # of every cadence they flagged
                            had = np.isfinite(flux)
                            flux[~keep] = np.nan
                            log_quality(log, lc['quality'], had & ~keep, 'TESS')

                        ax.axhline(1, ls='--', color='gray', alpha=0.3)
                        ax.plot(*break_at_gaps(time, flux),
                                drawstyle='steps', lw=1)

                        ax.grid(alpha=0.2)

                        ax.set_ylabel('Normalized ' + lc.meta['FLUX_ORIGIN'])
                        ax.set_xlabel('Time - 2457000, BTJD days')
                        ax.set_title(f"{config['target_name']} - TESS Sector {lc.meta['SECTOR']} - {lc.meta['AUTHOR']} - {row.exptime[0].value:.0f} s")

                    # log(f"   Sector lightcurve written to file:{lcname}")

                    # Remove time column that cannot be serialized
                    lc1 = lc.to_table()
                    lc1['mjd'] = lc1['time'].mjd
                    lc1['btjd'] = lc1['time'].btjd
                    lc1.remove_column('time')

                    votname = os.path.splitext(lcname)[0] + '.vot'
                    txtname = os.path.splitext(lcname)[0] + '.txt'
                    lc1.write(os.path.join(basepath, votname), format='votable', overwrite=True)
                    lc1.write(os.path.join(basepath, txtname), format='ascii.commented_header', overwrite=True)
                    log(f"    Sector lightcurve written to file:{votname}")
                    log(f"    Sector lightcurve written to file:{txtname}")

                    is_done = True
                    break

                if is_done:
                    break
