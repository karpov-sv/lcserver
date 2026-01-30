"""ZTF lightcurve acquisition module.

Acquires ZTF (Zwicky Transient Facility) optical lightcurves in g/r bands.
"""

import os
import numpy as np

from astropy.table import Table
from astropy.time import Time

from ztfquery import lightcurve

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import cleanup_paths


# Some convenience code for gaussian process based smoothing of unevenly spaced 1d data
import george
from george import kernels
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def gaussian_smoothing(x, y, dy, scale=100, nsteps=1000):
    """
    Gaussian process based smoothing of unevenly spaced 1d data.

    Parameters
    ----------
    x : array
        Input x coordinates
    y : array
        Input y values
    dy : array
        Input y errors
    scale : float, optional
        Initial scale parameter
    nsteps : int, optional
        Number of steps for prediction

    Returns
    -------
    callable
        Interpolation function
    """
    y0 = np.median(y)
    y = y - y0

    kernel = 10*np.var(y)*kernels.Matern32Kernel(100, ndim=1)
    gp = george.GP(kernel)
    gp.compute(x, dy)

    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)

    result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    gp.set_parameter_vector(result.x)

    x_pred = np.linspace(np.min(x), np.max(x), 1000)
    pred, pred_var = gp.predict(y, x_pred, return_var=True)

    return interp1d(x_pred, pred + y0, fill_value='extrapolate')


@survey_source(
    name='Zwicky Transient Facility',
    short_name='ZTF',
    state_acquiring='acquiring ZTF lightcurve',
    state_acquired='ZTF lightcurve acquired',
    log_file='ztf.log',
    output_files=['ztf.log', 'ztf_lc.png', 'ztf_color_mag.png'],
    button_text='Get ZTF lightcurve',
    form_fields={
        'ztf_color_model': {
            'type': 'choice',
            'label': 'Color model',
            'choices': [('constant', 'Constant'), ('gp', 'GP smoothing')],
            'initial': 'constant',
            'required': False,
        }
    },
    help_text='ZTF optical transient survey (g/r bands)',
    order=10,
    # Lightcurve metadata
    votable_file='ztf.vot',
    lc_mag_column='mag_g',
    lc_err_column='magerr',
    lc_filter_column='zg',
    lc_color='#ff7f0e',
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata
    template_layout='with_cutout',
    show_cutout=True,
    cutout_hips='CDS/P/ZTF/DR7/color',
    cutout_fov=0.03,
    show_color_mag=True,
    color_mag_file='ztf_color_mag.png',
)
def target_ztf(config, basepath=None, verbose=True, show=False):
    """
    Get ZTF lightcurve.

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
    cleanup_paths(get_output_files('ztf'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    if os.path.exists(os.path.join(basepath, 'ztf.vot')):
        log(f"Loading ZTF lightcurve from ztf.vot")
        ztf = Table.read(os.path.join(basepath, 'ztf.vot'))
    else:
        ztf_sr = config.get('ztf_sr', 2.0)

        log(f"Requesting ZTF lightcurve for {config['target_name']} within {ztf_sr:.1f} arcsec")
        lcq = lightcurve.LCQuery.from_position(config.get('target_ra'), config.get('target_dec'), ztf_sr)

        if not lcq or not len(lcq.data):
            log("Warning: No ZTF data found")
            return

        ztf = Table.from_pandas(lcq.data)

    log(f"{len(ztf)} ZTF data points found")
    for fn in ['zg', 'zr', 'zi']:
        idx = ztf['filtercode'] == fn
        idx_good = idx & (ztf['catflags'] == 0) & (ztf['magerr'] < 0.05)

        log(f"  {fn}: {np.sum(idx)} total, {np.sum(idx_good)} good")

    log("Earliest: ", Time(np.min(ztf['mjd']), format='mjd').datetime.strftime('%Y-%m-%s %H:%M:%S'))
    log("  Latest: ", Time(np.max(ztf['mjd']), format='mjd').datetime.strftime('%Y-%m-%s %H:%M:%S'))

    if not np.sum(ztf['filtercode'] == 'zr') and not np.sum(ztf['filtercode'] == 'zg'):
        log("No datapoints in zg or zr filters")
        return

    if np.nanmin(ztf['mag']) < 13.2:
        log(f"Warning: Max brightness is {np.nanmin(ztf['mag']):.2f}, object may be saturated")

    ztf['time'] = Time(ztf['mjd'], format='mjd') # Astropy Time object corresponding to given MJD
    ztf['mag_calib'] = np.nan # We will use this field to store Pan-STARRS calibrated magnitudes
    ztf['mag_g'] = np.nan
    ztf['mag_r'] = np.nan

    # Initial model for the color
    gr = lambda x: np.zeros_like(x)

    log("\n---- Reconstructing the color and Pan-STARRS magnitudes ----\n")

    ztf['time'] = Time(ztf['mjd'], format='mjd') # Astropy Time object corresponding to given MJD
    ztf['mag_calib'] = np.nan # We will use this field to store Pan-STARRS calibrated magnitudes
    ztf['mag_g'] = np.nan
    ztf['mag_r'] = np.nan

    # Initial model for the color
    gr = lambda x: np.ones_like(x) * config.get('g_minus_r', 0)

    u_mjd = np.linspace(np.min(ztf['time'].mjd), np.max(ztf['time'].mjd), 1000)

    for iter in range(20):
        # Select only good points in ZTF g filter
        idx = (ztf['filtercode'] == 'zg') & (ztf['magerr'] < 0.15) & (ztf['catflags'] == 0)
        tg,magg,dmagg,colg = ztf['time'][idx], ztf['mag'][idx], ztf['magerr'][idx], ztf['clrcoeff'][idx]
        cmagg = magg + colg * gr(tg.mjd)
        ztf['mag_calib'][idx] = cmagg
        ztf['mag_g'][idx] = cmagg

        # Select only good points in ZTF r filter
        idx = (ztf['filtercode'] == 'zr') & (ztf['magerr'] < 0.15) & (ztf['catflags'] == 0)
        tr,magr,dmagr,colr = ztf['time'][idx], ztf['mag'][idx], ztf['magerr'][idx], ztf['clrcoeff'][idx]
        cmagr = magr + colr * gr(tr.mjd)
        ztf['mag_calib'][idx] = cmagr
        ztf['mag_r'][idx] = cmagr

        # Compute the colors by associating nearby points
        iig,iir = [],[]

        if not len(tg) or not len(tr):
            break

        for i,tg1 in enumerate(tg.mjd):
            dist = np.abs((tr.mjd - tg1))
            # FIXME: make time delay configurable
            if np.min(dist) < 0.5:
                iig.append(i)
                iir.append(np.where(dist == np.min(dist))[0][0])

        if not len(iig):
            break

        gr_old = gr(u_mjd)

        # Let's fit for the next estimate of the color model!
        if len(iig) > 100 and config.get('ztf_color_model', 'constant') == 'gp':
            gr = gaussian_smoothing(tg[iig].mjd, cmagg[iig]-cmagr[iir], np.hypot(dmagg[iig], dmagr[iir]), scale=10)
        else:
            med = np.median(cmagg[iig]-cmagr[iir])
            gr = lambda x: np.ones_like(x)*med

        gr_new = gr(u_mjd)
        rms = np.sqrt(np.sum((gr_new-gr_old)**2)/(len(gr_new)-1))
        log(f"Iteration {iter}: mean (g-r) = {np.mean(gr_new):.3f}, rms difference {rms:.2g}")
        if rms < 1e-4:
            log(f"Converged")
            break

    # Time cannot be serialized to VOTable
    ztf[[_ for _ in ztf.columns if _ != 'time']].write(os.path.join(basepath, 'ztf.vot'),
                                                       format='votable', overwrite=True)
    ztf[[_ for _ in ztf.columns if _ != 'time']].write(os.path.join(basepath, 'ztf.txt'),
                                                       format='ascii.commented_header', overwrite=True)
    log("ZTF data written to file:ztf.vot")
    log("ZTF data written to file:ztf.txt")

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'ztf_lc.png'), figsize=(12, 8), show=show) as fig:
        ax = fig.add_subplot(3, 1, 1)
        ax.errorbar(tg.datetime, cmagg, dmagg, fmt='.', color='green',
                    label='g=%.2f +/- %.2f' % (np.mean(cmagg), np.std(cmagg)))
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        ax.set_ylabel('g')
        ax.legend()
        ax.set_title(f"{config['target_name']} - ZTF")

        ax = fig.add_subplot(3, 1, 2, sharex=ax)

        ax.errorbar(tr.datetime, cmagr, dmagr, fmt='.', color='red',
                    label='r=%.2f +/- %.2f' % (np.mean(cmagr), np.std(cmagr)))
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        ax.set_ylabel('r')
        ax.legend()

        ax = fig.add_subplot(3, 1, 3, sharex=ax)

        if len(iig):
            ax.plot(Time(u_mjd, format='mjd').datetime, gr(u_mjd), '--', color='red', alpha=0.3, label='Model')
            ax.errorbar(tg[iig].datetime, cmagg[iig]-cmagr[iir], np.hypot(dmagg[iig], dmagr[iir]), fmt='.', alpha=0.5, label='g-r=%.2g +/- %.2g' % (np.mean(cmagg[iig]-cmagr[iir]), np.std(cmagg[iig]-cmagr[iir])))
        ax.grid(alpha=0.3)
        ax.set_ylabel('g - r')
        ax.set_xlabel('Time')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels))

    # Plot color-magnitude diagram
    with plots.figure_saver(os.path.join(basepath, 'ztf_color_mag.png'), figsize=(10, 5), show=show) as fig:
        ax = fig.add_subplot(1, 2, 1)
        ax.errorbar(cmagg[iig]-cmagr[iir], cmagg[iig], xerr=np.hypot(dmagg[iig], dmagr[iir]), yerr=dmagg[iig],
                    fmt='.', color='green', alpha=0.5)
        ax.grid(alpha=0.3)
        ax.set_xlabel('g - r')
        ax.set_ylabel('g')
        ax.invert_yaxis()

        ax = fig.add_subplot(1, 2, 2, sharex=ax)
        ax.errorbar(cmagg[iig]-cmagr[iir], cmagr[iir], xerr=np.hypot(dmagg[iig], dmagr[iir]), yerr=dmagr[iir],
                    fmt='.', color='red', alpha=0.5)
        ax.grid(alpha=0.3)
        ax.set_xlabel('g - r')
        ax.set_ylabel('r')
        ax.invert_yaxis()

        fig.suptitle(f"{config['target_name']} - ZTF")

    log("\n---- Worst-case Pan-STARRS recalibration error ----\n")

    color_mean,color_std = np.mean(cmagg[iig]-cmagr[iir]), np.std(cmagg[iig]-cmagr[iir])
    log(f"(g - r) = {color_mean:.3f} +/- {color_std:.3f}")

    for fn in ['zg', 'zr', 'zi']:
        idx = ztf['filtercode'] == fn

        if np.sum(idx):
            mean,std = np.nanmean(ztf['clrcoeff'][idx]), np.nanstd(ztf['clrcoeff'][idx])

            log(f"{fn}: clrcoeff = {mean:.3f} +/- {std:.3f}  delta = {mean*color_mean:.3f} err = {mean*color_std:.3f}")

    # TODO: when should we update the color?
    config['g_minus_r'] = color_mean
