"""TESS lightcurve acquisition module.

Acquires TESS (Transiting Exoplanet Survey Satellite) lightcurves.
"""

import os
import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

import lightkurve as lk

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import cleanup_paths


@survey_source(
    name='TESS',
    short_name='TESS',
    state_acquiring='acquiring TESS lightcurves',
    state_acquired='TESS lightcurves acquired',
    log_file='tess.log',
    output_files=['tess.log', 'tess_lc_*.vot', 'tess_lc_*.png'],
    button_text='Get TESS lightcurves',
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

    # Cleanup stale plots
    cleanup_paths(get_output_files('tess'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    tess_sr = config.get('tess_sr', 10.0)
    log(f"Requesting TESS data for {config['target_name']} within {tess_sr:.1f} arcsec")

    res = lk.search_lightcurve(SkyCoord(config.get('target_ra'), config.get('target_dec'), unit='deg'), radius=tess_sr*u.arcsec, mission='TESS')

    if len(res):
        # Filter out CDIPS products
        res = res[res.author != 'CDIPS']

    if not len(res):
        log("Warning: No TESS data found")
        return
    else:
        log(f"{len(res)} data products found")

    for tname in np.unique(res.target_name):
        idx = res.target_name == tname
        log(f"\nTESS target {tname} at {res[idx].distance[0].value:.1f} arcsec")

        for mission in np.unique(res[idx].mission):
            idx1 = idx & (res.mission == mission)
            tmin = Time(np.min(res.table['t_min'][idx1]), format='mjd')
            tmax = Time(np.max(res.table['t_max'][idx1]), format='mjd')
            log(f"  {mission}: {tmin.datetime.strftime('%Y-%m-%d')} - {tmax.datetime.strftime('%Y-%m-%d')}")

            for prod in res[idx1].table:
                log(f"    {prod['author']:10s} {prod['exptime']} s exp")

            # Write one representative lightcurve per sector
            for author in ['TESS-SPOC', 'QLP', 'SPOC']:
                idx2 = idx1 & (res.author == author)
                is_done = False

                for row in res[idx2]:
                    lc = row.download(download_dir=os.path.join(basepath, 'cache'))
                    if not lc:
                        continue

                    # Plot the lightcurve
                    lcname = f"tess_lc_{lc.meta['SECTOR']}_{lc.meta['AUTHOR']}_{row.exptime[0].value:.0f}.png"
                    with plots.figure_saver(os.path.join(basepath, lcname), figsize=(8, 4), show=show) as fig:
                        ax = fig.add_subplot(1, 1, 1)

                        time = lc.time.btjd
                        flux = lc.normalize().flux
                        flux[lc['quality'] != 0] = np.nan

                        ax.axhline(1, ls='--', color='gray', alpha=0.3)
                        ax.plot(time, flux, drawstyle='steps', lw=1)

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
