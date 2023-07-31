from django.conf import settings

import os, glob, shutil, re
import requests
from io import BytesIO

from functools import partial

import numpy as np

from astropy.wcs import WCS
from astropy.io import fits as fits

from astropy.table import Table, vstack
from astropy.stats import mad_std
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

from astroquery.simbad import Simbad

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
    'galaxy_map.png',
]

files_cache = [
    'ps1.vot',
    'ztf.vot',
    'asas.vot',
    'dasch.vot',
    'applause.vot',
]

files_ztf = [
    'ztf.log',
    'ztf_lc.png',
    'ztf_color_mag.png',
]

files_asas = [
    'asas.log',
    'asas_lc.png',
    'asas_color_mag.png',
]

files_tess = [
    'tess.log',
    'tess_lc_*.png',
    'tess_lc_*.vot',
]

files_dasch = [
    'dasch.log',
    'dasch_lc.png',
]

files_applause = [
    'applause.log',
    'applause_lc.png',
]

files_combined = [
    'combined.log',
    'combined_lc.png',
]

cleanup_info = files_info + files_cache + files_ztf + files_asas + files_tess + files_dasch + files_applause + files_combined
cleanup_ztf = files_ztf
cleanup_asas = files_asas
cleanup_tess = files_tess
cleanup_dasch = files_dasch
cleanup_applause = files_applause
cleanup_combined = files_combined


def cleanup_paths(paths, basepath=None):
    for path in paths:
        for fullpath in glob.glob(os.path.join(basepath, path)):
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

    config['target_l'] = target.galactic.l.deg
    config['target_b'] = target.galactic.b.deg

    # Galactic coordinates
    log(f"Galactic l={target.galactic.l.deg:.4f} b={target.galactic.b.deg:.4f}")

    log("\n---- SIMBAD ----\n")

    sim = Simbad()
    sim.add_votable_fields('otype', 'otypes', 'ids', 'distance', 'sptype')

    res = sim.query_region(target, radius=5*u.arcsec)

    if not len(res):
        log("No SIMBAD objects within 5 arcsec")
    else:
        for r in res:
            # TODO: select closest object?..
            log(f"{r['MAIN_ID']} = {r['OTYPE']}")
            log(f"{r['OTYPES']}")

            if r['SP_TYPE']:
                log(f"SpType = {r['SP_TYPE']}")
            # if r['Distance_distance']:
            #     log(f"Dist = {r['Distance_distance']:.2f} +{r['Distance_perr']:.2f} -{-r['Distance_merr']:.2f} {r['Distance_unit']}")

            break

    # Catalogues to get photometry
    for catname in ['gaiadr3syn', 'ps1', 'skymapper']:
        cat = catalogs.get_cat_vizier(config.get('target_ra'), config.get('target_dec'), 5/3600,
                                      catname, get_distance=True, verbose=False)
        if not cat or not len(cat):
            continue

        log(f"\n---- {catalogs.catalogs[catname]['name']} ----\n")

        star = dict(cat[cat['_r'] == np.min(cat['_r'])][0])

        for fn in ['B', 'V', 'R', 'I', 'g', 'r', 'i', 'z']:
            if star.get(f'{fn}mag'):
                log(f"{fn} = {star[f'{fn}mag']:.2f} +/- {star[f'e_{fn}mag']:.2f}")

        if star.get('Bmag') and star.get('Vmag'):
            B_minus_V = star['Bmag'] - star['Vmag']
            B_minus_V_err = np.hypot(star['e_Bmag'], star['e_Vmag'])
            log(f"(B - V) = {B_minus_V:.3f} +/- {B_minus_V_err:.3f}")
            if config.get('B_minus_V') is None:
                config['B_minus_V'] = B_minus_V

        if star.get('gmag') and star.get('rmag'):
            g_minus_r = star['gmag'] - star['rmag']
            g_minus_r_err = np.hypot(star['e_gmag'], star['e_rmag'])
            log(f"(g - r) = {g_minus_r:.3f} +/- {g_minus_r_err:.3f}")
            if config.get('g_minus_r') is None:
                config['g_minus_r'] = g_minus_r

        # if config.get('B_minus_V') is not None and config.get('g_minus_r') is not None:
            # break

    # Gaia DR3 photometry
    cat = catalogs.get_cat_vizier(config.get('target_ra'), config.get('target_dec'), 5/3600,
                                  'I/355/gaiadr3',
                                  extra=['_RAJ2000', '_DEJ2000', 'e_Gmag', 'e_BPmag', 'e_RPmag'],
                                  get_distance=True, verbose=False)
    if cat:
        star = dict(cat[cat['_r'] == np.min(cat['_r'])][0])

        log(f"\n---- Gaia DR3 ----\n")

        for fn in ['G', 'BP', 'RP']:
            if star.get(f'{fn}mag'):
                log(f"{fn} = {star[f'{fn}mag']:.2f} +/- {star[f'e_{fn}mag']:.2f}")

        if star.get('BPmag') and star.get('RPmag'):
            BP_minus_RP = star['BPmag'] - star['RPmag']
            BP_minus_RP_err = np.hypot(star['e_BPmag'], star['e_RPmag'])
            log(f"(BP - RP) = {BP_minus_RP:.3f} +/- {BP_minus_RP_err:.3f}")
            if config.get('BP_minus_RP') is None:
                config['BP_minus_RP'] = BP_minus_RP

    # Gaia DR3 distances by Bailer-Jones
    cat = catalogs.get_cat_vizier(config.get('target_ra'), config.get('target_dec'), 5/3600,
                                  'I/352/gedr3dis', extra=['_RAJ2000', '_DEJ2000'],
                                  get_distance=True, verbose=False)
    if cat:
        star = cat[cat['_r'] == np.min(cat['_r'])][0]

        if star['rgeo']:
            log(f"Gaia DR3 distance is {star['rgeo']:.1f} [{star['b_rgeo']:.1f} ... {star['B_rgeo']:.1f}] pc")

            log(f"Height above Galactic plane is {star['rgeo']*np.abs(np.sin(np.deg2rad(config.get('target_b')))):.1f} pc")

            # Galaxy map
            from matplotlib import image
            from PIL import Image
            import urllib

            url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Artist%27s_impression_of_the_Milky_Way_%28updated_-_annotated%29.jpg/1024px-Artist%27s_impression_of_the_Milky_Way_%28updated_-_annotated%29.jpg'
            galaxy_image = np.array(Image.open(urllib.request.urlopen(url)))

            # Plot Galaxy map
            with plots.figure_saver(os.path.join(basepath, 'galaxy_map.png'), figsize=(8, 8), show=show) as fig:
                ax = fig.add_subplot(1, 1, 1)
                ax.axis('off')

                img = 1 - galaxy_image / 255
                ax.imshow(img, origin='upper')

                x0,y0 = 998 * 1024/2000, 1381 * 1024/2000
                scale_ly = 295/20000 * 1024/2000 # pixels per lightyear
                scale = scale_ly * 1000 / 0.306601 # pixels per kpc
                # ax.plot(x0, y0, 'ro')
                # ax.plot(x0, y0 + scale_ly*20000, 'ro')

                l = config.get('target_l')
                b = config.get('target_b')
                r = star['rgeo'] / 1000
                x = x0 - scale * r * np.cos(np.deg2rad(b)) * np.sin(np.deg2rad(l))
                y = y0 - scale * r * np.cos(np.deg2rad(b)) * np.cos(np.deg2rad(l))

                z = scale * r * np.sin(np.deg2rad(b))

                ax.plot([x, x], [y-z, y], '.-', color='yellow', markeredgecolor='black')
                ax.scatter(x, y-z, marker='*', color='yellow', edgecolor='black', alpha=1, s=200, label=config.get('target_name'), zorder=100)

                ax.scatter(x0, y0, marker='o', color='lightgreen', edgecolor='black', s=30, label='Sun')
                ax.legend()

            log("Galaxy map with object position saved to file:galaxy_map.png")

    # Pan-STARRS DR2 warp photometry
    log("\n---- Pan-STARRS DR2 warp photometry ----\n")

    try:
        res = requests.get('https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/detection.csv', params={'ra':config['target_ra'], 'dec':config['target_dec'], 'radius':2/3600, 'format':'csv', 'columns':['obsTime', 'filterID', 'psfQfPerfect', 'psfFlux', 'psfFluxErr']})
        ps1 = Table.read(BytesIO(res.content), format='csv')
    except:
        import traceback
        traceback.print_exc()
        log("Error while downloading the data")
        ps1 = None

    if ps1 and len(ps1):
        ps1.sort('obsTime')
        ps1 = ps1[ps1['psfQfPerfect'] > 0.95] # Quality cut

        ps1['time'] = Time(ps1['obsTime'], format='mjd')
        ps1['mjd'] = ps1['time'].mjd
        ps1['mag'] = -2.5*np.log10(ps1['psfFlux']) + 8.90 # Janskys to AB?..
        ps1['magerr'] = 2.5/np.log(10)*(ps1['psfFluxErr']/ps1['psfFlux'])

        for fid,fn in [[1, 'g'], [2, 'r'], [3, 'i']]:
            idx = ps1['filterID'] == fid

            log(f"{fn}: {np.sum(idx)} good points")

            ps1['mag_' + fn] = np.nan
            ps1['mag_' + fn][idx] = ps1['mag'][idx]

        # Color?..
        ig,ir = np.where(ps1['filterID'] == 1)[0], np.where(ps1['filterID'] == 2)[0]
        mg,mr = [],[]

        for i in ig:
            dist = np.abs(ps1['mjd'][i] - ps1['mjd'][ir])
            if np.min(dist) < 1:
                mg.append(ps1['mag'][i])
                mr.append(ps1['mag'][ir[dist == np.min(dist)]][0])

        if len(mg):
            mg,mr = [np.array(_) for _ in (mg,mr)]

            log(f"{len(mg)} quasi-simultaneous measurements")
            log(f"(g - r) = {np.nanmean(mg-mr):.3f} +/- {np.nanstd(mg-mr):.3f}")

        # Time cannot be serialized to VOTable
        ps1[[_ for _ in ps1.columns if _ != 'time']].write(os.path.join(basepath, 'ps1.vot'), format='votable', overwrite=True)
        log("Pan-STARRS DR2 warp photometry written to file:ps1.vot")

    else:
        log("No Pan-STARRS DR2 warp data found")


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
    log("ZTF data written to file:ztf.vot")

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


from pyasassn.client import SkyPatrolClient # Installed via %pip install skypatrol

# Get ASAS-SN lightcurve
def target_asas(config, basepath=None, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Cleanup stale plots
    cleanup_paths(cleanup_asas, basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    if os.path.exists(os.path.join(basepath, 'asas.vot')):
        log(f"Loading ASAS-SN lightcurve from asas.vot")
        asas = Table.read(os.path.join(basepath, 'asas.vot'))
    else:
        asas_sr = config.get('asas_sr', 10.0)

        log(f"Requesting ASAS-SN lightcurve for {config['target_name']} within {asas_sr:.1f} arcsec")

        try:
            client = SkyPatrolClient()
            lcq = client.cone_search(ra_deg=config.get('target_ra'), dec_deg=config.get('target_dec'), radius=asas_sr/3600, catalog='master_list', download=True)
        except:
            import traceback
            traceback.print_exc()

            lcq = None

        if not lcq or not len(lcq.data):
            log("Warning: No ASAS-SN data found")
            return

        asas = Table.from_pandas(lcq.data)

    log(f"{len(asas)} ASAS-SN data points found")

    for fn in ['g', 'V']:
        idx = asas['phot_filter'] == fn
        idx_good = idx & (asas['quality'] == 'G') & (asas['mag_err'] < 0.05)

        log(f"  {fn}: {np.sum(idx)} total, {np.sum(idx_good)} good")

    log("Earliest: ", Time(np.min(asas['jd']), format='jd').datetime.strftime('%Y-%m-%s %H:%M:%S'))
    log("  Latest: ", Time(np.max(asas['jd']), format='jd').datetime.strftime('%Y-%m-%s %H:%M:%S'))

    asas['time'] = Time(asas['jd'], format='jd')
    asas['mjd'] = asas['time'].mjd

    asas['mag_V'] = np.nan
    asas['mag_g'] = np.nan

    g_minus_r = config.get('g_minus_r', 0.0)
    log(f"Will use (g - r) = {g_minus_r:.2f} for converting V to g magnitudes")

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'asas_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        idx = asas['quality'] == 'G'
        idx &= np.isfinite(asas['mag'])
        idx &= asas['mag_err'] < 0.05

        idx1 = idx & (asas['phot_filter'] == 'V')
        asas['mag_V'][idx1] = asas['mag'][idx1]
        asas['mag_g'][idx1] = asas['mag'][idx1] + 0.02 + 0.498*g_minus_r + 0.008*g_minus_r**2

        ax.errorbar(asas[idx1]['time'].datetime, asas[idx1]['mag_g'], asas[idx1]['mag_err'], fmt='.', label='V conv. to g')

        idx1 = idx & (asas['phot_filter'] == 'g')
        asas['mag_g'][idx1] = asas['mag'][idx1] - 0.013 - 0.145*g_minus_r - 0.019*g_minus_r**2

        ax.errorbar(asas[idx1]['time'].datetime, asas[idx1]['mag_g'], asas[idx1]['mag_err'], fmt='.', label='g')

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        ax.legend()
        ax.set_ylabel('g')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - ASAS-SN")

    # Time cannot be serialized to VOTable
    asas[[_ for _ in asas.columns if _ != 'time']].write(os.path.join(basepath, 'asas.vot'),
                                                         format='votable', overwrite=True)
    log("ASAS-SN data written to file:asas.vot")


import lightkurve as lk

# Get TESS lightcurves
def target_tess(config, basepath=None, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Cleanup stale plots
    cleanup_paths(cleanup_tess, basepath=basepath)

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
                    lc1.write(os.path.join(basepath, votname), format='votable', overwrite=True)
                    log(f"    Sector lightcurve written to file:{votname}")

                    is_done = True
                    break

                if is_done:
                    break


# Get DASCH lightcurve
def target_dasch(config, basepath=None, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Cleanup stale plots
    cleanup_paths(cleanup_dasch, basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    if config.get('target_b') < 0:
        log("DASCH is not currently available for Southern Galactic hemisphere")
        return


    cachename = f"dasch_{config['target_ra']:.4f}_{config['target_dec']:.4f}.vot"

    if os.path.exists(os.path.join(basepath, 'cache', cachename)):
        log(f"Loading DASCH lightcurve from the cache")
        dasch = Table.read(os.path.join(basepath, 'cache', cachename))
    else:
        dasch_sr = config.get('dasch_sr', 5.0)

        log(f"Requesting DASCH lightcurve for {config['target_name']} within {dasch_sr:.1f} arcsec")

        baseurl = "http://dasch.rc.fas.harvard.edu"
        url = f"{baseurl}/lightcurve_cat_input.php?nmin=1&box={dasch_sr}&source=atlas&coo={config['target_ra']}+{config['target_dec']}"
        res = requests.get(url)
        m = re.search(r'/tmp/(\w+)/object', res.text)
        if not m:
            raise RuntimeError('Error downloading DASCH data')
        token = m[1]
        url2 = f"http://dasch.rc.fas.harvard.edu/lightcurve_data.php?dbfilename=/tmp/{token}/object_{token}.db&vofilename=/tmp/{token}/object_{token}.xml.gz&tmpdir={token}&REF=object&source=atlas"
        res2 = requests.get(url2)

        m = re.search(r'"tmp/\w+/([^s][^/"]+\.xml\.gz)"', res2.text)
        if not m:
            raise RuntimeError('Error downloading DASCH data')
        fname = m[1]
        url3 = f"{baseurl}/tmp/{token}/{fname}"
        dasch = Table.read(url3)

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
    log("DASCH data written to file:dasch.vot")


import pyvo as vo

# Get APPLAUSE lightcurve
def target_applause(config, basepath=None, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Cleanup stale plots
    cleanup_paths(cleanup_applause, basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")


    cachename = f"applause_{config['target_ra']:.4f}_{config['target_dec']:.4f}.vot"

    if os.path.exists(os.path.join(basepath, 'cache', cachename)):
        log(f"Loading APPLAUSE lightcurve from the cache")
        applause = Table.read(os.path.join(basepath, 'cache', cachename))
    else:
        applause_sr = config.get('applause_sr', 2.0)

        log(f"Requesting APPLAUSE lightcurve for {config['target_name']} within {applause_sr:.1f} arcsec")

        url = 'https://www.plate-archive.org/tap'

        query = f"""
        SELECT
            s.*,
            DEGREES(spoint(RADIANS(s.ra_icrs), RADIANS(s.dec_icrs)) <-> spoint(RADIANS({config['target_ra']}), RADIANS({config['target_dec']}))) AS angdist,
            e.jd_start, e.jd_mid, e.jd_end
        FROM applause_dr4.source_calib s, applause_dr4.exposure e, applause_dr4.plate p
        WHERE
            s.pos @ scircle(spoint(RADIANS({config['target_ra']}), RADIANS({config['target_dec']})), RADIANS({5/3600}))
            AND
            s.plate_id = e.plate_id
            AND
            s.plate_id = p.plate_id
            AND
            p.numexp = 1
            AND
            s.match_radius > 0
            AND
            s.model_prediction > 0.9
        """

        tap_service = vo.dal.TAPService(url) # Anonymous access

        job = tap_service.submit_job(query, language='PostgreSQL')
        job.run()

        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=120.)

        # TODO: more intelligent error handling?
        job.raise_if_error()
        results = job.fetch_result()
        applause = results.to_table()

        try:
            os.makedirs(os.path.join(basepath, 'cache'))
        except:
            pass

        applause.write(os.path.join(basepath, 'cache', cachename),
                       format='votable', overwrite=True)

    log(f"{len(applause)} original data points")

    if not len(applause):
        return

    applause['time'] = Time(applause['jd_start'], format='jd')
    applause['mjd'] = applause['time'].mjd
    applause.sort('time')

    BP_minus_RP = config.get('BP_minus_RP', np.nanmedian(applause['gaiaedr3_bp_rp']))
    g_minus_r = config.get('g_minus_r', 0.0)

    log(f"Using BP - RP = {BP_minus_RP:.2f} for converting natural magnitudes to Gaia Gmag")

    RPmag = applause['natmag'] - BP_minus_RP*applause['color_term']
    BPmag = RPmag + BP_minus_RP # assuming constant color
    # Simple one-color fits based on Landolt standards
    gmag = BPmag - np.polyval([-0.11445168305534677, -0.20378930951540578, 0.0499368274565225], g_minus_r)
    rmag = BPmag - np.polyval([-0.13189831407771777, 0.8213890428750275, 0.04388161680503415], g_minus_r)

    applause['mag_RP'] = RPmag
    applause['magerr'] = applause['natmag_error']

    applause['mag_g'] = gmag
    applause['mag_r'] = rmag

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'applause_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        ax.errorbar(applause['time'].datetime, applause['mag_g'], applause['magerr'], fmt='.', label='g')

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        # ax.legend()
        ax.set_ylabel('g')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - APPLAUSE")

    # Time cannot be serialized to VOTable
    applause[[_ for _ in applause.columns if _ != 'time']].write(os.path.join(basepath, 'applause.vot'),
                                                                 format='votable', overwrite=True)
    log("APPLAUSE data written to file:applause.vot")


combined_lc_rules = {
    'asas': {'name': 'ASAS-SN', 'filename': 'asas.vot', 'mag': 'mag_g', 'err': 'mag_err', 'filter': 'phot_filter', 'short': True},
    'ztf': {'name': 'ZTF', 'filename': 'ztf.vot', 'mag': 'mag_g', 'err': 'magerr', 'filter': 'zg', 'short': True},
    'ps1': {'name': 'Pan-STARRS', 'filename': 'ps1.vot', 'mag': 'mag_g', 'err': 'magerr', 'filter': 'g', 'short': True},
    'dasch': {'name': 'DASCH', 'filename': 'dasch.vot', 'mag': 'mag_g', 'err': 'magerr'},
    'applause': {'name': 'APPLAUSE', 'filename': 'applause.vot', 'mag': 'mag_g', 'err': 'magerr'},
}

# Get combined lightcurve
def target_combined(config, basepath=None, verbose=True, show=False):
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Cleanup stale plots
    cleanup_paths(cleanup_combined, basepath=basepath)

    for short,lcname in [[True, 'combined_short_lc.png'], [False, 'combined_lc.png']]:
        log(f"\n---- Plotting {'short ' if short else ''}lightcurve ----\n")

        with plots.figure_saver(os.path.join(basepath, lcname), figsize=(12, 4), show=show) as fig:
            ax = fig.add_subplot(1, 1, 1)

            for name,rules in combined_lc_rules.items():
                if short and not rules.get('short'):
                    continue

                fullname = os.path.join(basepath, rules['filename'])
                if os.path.exists(fullname):
                    log(f"Reading {rules['name']} data from file:{rules['filename']}")
                    data = Table.read(fullname)

                    data['time'] = Time(data['mjd'], format='mjd')

                    x = data['time'].datetime
                    y = data[rules.get('mag', 'mag_g')]
                    dy = data[rules.get('err', 'magerr')]

                    if 'filter' in rules and rules['filter'] in data.colnames:
                        fnames = data[rules['filter']]
                    else:
                        fnames = np.repeat(rules.get('filter', ''), len(data))

                    for fn in np.unique(fnames):
                        idx = fnames == fn

                        label = rules['name']
                        if fn:
                            label += ' ' + fn

                        ax.errorbar(x[idx], y[idx], dy[idx], fmt='.', label=label)

            ax.invert_yaxis()
            ax.grid(alpha=0.2)

            ax.legend()
            ax.set_ylabel('g magnitude')
            ax.set_xlabel('Time')
            ax.set_title(f"{config['target_name']}")
