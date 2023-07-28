from django.conf import settings

import os, glob, shutil

from functools import partial

import numpy as np

from astropy.wcs import WCS
from astropy.io import fits as fits

from astropy.table import Table, vstack
from astropy.stats import mad_std
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

from ztfquery import lightcurve

import dill as pickle

# STDPipe
from stdpipe import astrometry, catalogs, resolve, plots

# Disable some annoying warnings from astropy
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter(action='ignore', category=FITSFixedWarning)
warnings.simplefilter(action='ignore', category=VerifyWarning)


files_info = [
    'info.log',
]

files_cache = [
    'ztf.vot',
]

files_ztf = [
    'ztf.log',
    'ztf_lc.png',
    'ztf_color_mag.png',
]

cleanup_info = files_info + files_cache + files_ztf
cleanup_ztf = files_ztf

def cleanup_paths(paths, basepath=None):
    for path in paths:
        fullpath = os.path.join(basepath, path)
        if os.path.exists(fullpath):
            if os.path.isdir(fullpath):
                shutil.rmtree(fullpath)
            else:
                os.unlink(fullpath)


def print_to_file(*args, clear=False, logname='out.log', **kwargs):
    if clear and os.path.exists(logname):
        print('Clearing', logname)
        os.unlink(logname)

    if len(args) or len(kwargs):
        print(*args, **kwargs)
        with open(logname, 'a+') as lfd:
            print(file=lfd, *args, **kwargs)


def pickle_to_file(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pickle_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Acquire basic info on the target
def target_info(config, basepath=None, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Cleanup stale plots
    cleanup_paths(cleanup_info, basepath=basepath)

    log(f"Acquiring the info on {config['target_name']}")

    # Resolve
    try:
        target = resolve.resolve(config['target_name'])
        config['target_ra'] = target.ra.deg
        config['target_dec'] = target.dec.deg

        log(f"Resolved to RA={config['target_ra']:.4f} Dec={config['target_dec']:.4f}")
    except:
        raise RuntimeError("Target not resolved")

    # Galactic coordinates
    log(f"Galactic l={target.galactic.l.deg:.4f} b={target.galactic.b.deg:.4f}")


# Some convenience code for gaussian process based smoothing of unevenly spaced 1d data
import george
from george import kernels
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def gaussian_smoothing(x, y, dy, scale=100, nsteps=1000):
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


# Get ZTF lightcurve
def target_ztf(config, basepath=None, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Cleanup stale plots
    cleanup_paths(cleanup_ztf, basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    if os.path.exists(os.path.join(basepath, 'ztf.vot')):
        log(f"Loading ZTF lightcurve from ztf.vot")
        ztf = Table.read(os.path.join(basepath, 'ztf.vot'))
    else:
        log(f"Requesting ZTF lightcurve for {config['target_name']}")

        # FIXME: configurable radius
        lcq = lightcurve.LCQuery.from_position(config.get('target_ra'), config.get('target_dec'), 2)

        if not lcq or not len(lcq.data):
            log("Warning: No ZTF data found")
            return

        ztf = Table.from_pandas(lcq.data)

    log(f"{len(ztf)} ZTF data points found")
    for fn in ['zg', 'zr', 'zi']:
        idx = ztf['filtercode'] == fn
        idx_good = idx & (ztf['catflags'] == 0) & (ztf['magerr'] < 0.05)

        log(f"  {fn}: {np.sum(idx)} total, {np.sum(idx_good)} good")

    log("Earliest: ", str(Time(np.min(ztf['mjd']), format='mjd').datetime))
    log("  Latest: ", str(Time(np.max(ztf['mjd']), format='mjd').datetime))

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
    gr = lambda x: np.zeros_like(x)

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
        if rms < 1e-6:
            log(f"Converged")
            break

    # Time cannot be serialized to VOTable
    ztf[[_ for _ in ztf.columns if _ != 'time']].write(os.path.join(basepath, 'ztf.vot'), format='votable', overwrite=True)
    log("ZTF data written to file:ztf.vot")

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'ztf_lc.png'), figsize=(12, 8), show=show) as fig:
        ax = fig.add_subplot(3, 1, 1)
        ax.errorbar(tg.datetime, cmagg, dmagg, fmt='.', label='g=%.2f +/- %.2f' % (np.mean(cmagg), np.std(cmagg)))
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        ax.set_ylabel('g')
        ax.legend()

        ax = fig.add_subplot(3, 1, 2, sharex=ax)

        ax.errorbar(tr.datetime, cmagr, dmagr, fmt='.', label='r=%.2f +/- %.2f' % (np.mean(cmagr), np.std(cmagr)))
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

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels))

    # Plot color-magnitude diagram
    with plots.figure_saver(os.path.join(basepath, 'ztf_color_mag.png'), figsize=(10, 5), show=show) as fig:
        ax = fig.add_subplot(1, 2, 1)
        ax.errorbar(cmagg[iig]-cmagr[iir], cmagg[iig], xerr=np.hypot(dmagg[iig], dmagr[iir]), yerr=dmagg[iig], fmt='.', alpha=0.5)
        ax.grid(alpha=0.3)
        ax.set_xlabel('g - r')
        ax.set_ylabel('g')
        ax.invert_yaxis()

        ax = fig.add_subplot(1, 2, 2, sharex=ax)
        ax.errorbar(cmagg[iig]-cmagr[iir], cmagr[iir], xerr=np.hypot(dmagg[iig], dmagr[iir]), yerr=dmagr[iir], fmt='.', alpha=0.5)
        ax.grid(alpha=0.3)
        ax.set_xlabel('g - r')
        ax.set_ylabel('r')
        ax.invert_yaxis()
