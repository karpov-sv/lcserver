"""ZTF lightcurve acquisition module.

Acquires ZTF (Zwicky Transient Facility) optical lightcurves in g/r bands.
"""

import os
import numpy as np

from astropy.table import Table
from astropy.time import Time

from ztfquery import lightcurve
import requests
import pandas as pd

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import SourceError, cleanup_paths, cached_votable_query, log_bands, log_conversion


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
    output_files=['ztf.log', 'ztf_lc.png', 'ztf_color_mag.png', 'ztf.vot', 'ztf.txt'],
    button_text='Get ZTF lightcurve',
    # Its measured colour replaces the catalogue one the info step derives,
    # and five other sources convert their photometry with it
    provides_config=['g_minus_r'],
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
    lc_bands=[
        # ZTF reports magnitudes in a zero point that moves with the colour of
        # the star, through the per-epoch clrcoeff, so only these corrected
        # values are comparable between epochs
        surveys.band('g', 'mag_g', 'magerr', surveys.BAND_CALIBRATED,
                     filter_column='filtercode', filter_value='zg',
                     color='#2ca02c',
                     note='zg corrected to Pan-STARRS g using the per-epoch clrcoeff',
                     combined=True),
        surveys.band('r', 'mag_r', 'magerr', surveys.BAND_CALIBRATED,
                     filter_column='filtercode', filter_value='zr',
                     color='#d62728',
                     note='zr corrected to Pan-STARRS r using the per-epoch clrcoeff'),
        # ZTF observes r far more often than g, so without this the combined
        # curve would be missing most of what ZTF knows about the star
        surveys.band('g (from r)', 'mag_g_from_r', 'magerr', surveys.BAND_DERIVED,
                     filter_column='filtercode', filter_value='zr',
                     color='#98df8a',
                     note='r moved onto g by the colour ZTF measured for this star',
                     combined=True),
        # No colour model is reconstructed for i, so these stay as measured
        surveys.band('i', 'mag_i', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filtercode', filter_value='zi',
                     color='#9467bd',
                     note='zi as reported, with no colour correction applied'),
    ],
    lc_mag_column='mag_g',
    lc_err_column='magerr',
    lc_filter_column='filtercode',
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

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('ztf'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    # Cache raw query results before color calibration
    ra = config.get('target_ra')
    dec = config.get('target_dec')
    ztf_sr = config.get('ztf_sr', 2.0)
    cache_name = f"ztf_raw_{ra:.4f}_{dec:.4f}_{ztf_sr:.1f}.vot"

    # Columns required for ZTF processing
    ztf_required_columns = {'filtercode', 'mag', 'magerr', 'catflags', 'clrcoeff', 'mjd'}

    with cached_votable_query(cache_name, basepath, log, 'ZTF raw data', refresh=refresh_cache) as cache:
        if cache.hit:
            # Validate cached data has expected columns
            if not ztf_required_columns.issubset(cache.data.colnames):
                log(f"Warning: Cached ZTF data is invalid (missing columns), re-querying")
                log(f"  Expected: {sorted(ztf_required_columns)}")
                log(f"  Got: {cache.data.colnames}")
                cache.invalidate()

        if not cache.hit:
            # Query ZTF - only if not cached
            log(f"Requesting ZTF lightcurve for {config['target_name']} within {ztf_sr:.1f} arcsec")
            # Original IRSA data access
            # lcq = lightcurve.LCQuery.from_position(ra, dec, ztf_sr)
            # SNAD data access
            r = requests.get(
                "https://db.ztf.snad.space/api/v3/data/latest/circle/full/json",
                params={"ra": ra, "dec": dec, "radius_arcsec": ztf_sr},
                timeout=180,
            )

            lcq = None
            if r.status_code == 200:
                lc = []
                for v in r.json().values():
                    lc1 = pd.DataFrame(v["lc"])
                    lc1["filtercode"] = v["meta"]["filter"]
                    lc.append(lc1)

                if len(lc):
                    lcq = lambda: None # Fake object able to have attributes
                    lcq.data = pd.concat(lc, ignore_index=True)
                    # SNAD serves no catflags of its own, having applied
                    # catflags == 0 when it ingested the data release: what
                    # arrives here is already through that cut, so the zeros
                    # are true rather than assumed. The filtering below reads
                    # the column and is therefore right either way, whether
                    # the data came from here or from IRSA.
                    lcq.data['catflags'] = np.zeros_like(lcq.data['clrcoeff'], dtype=int)

            else:
                raise SourceError(f"the ZTF archive answered {r.status_code}")

            if not lcq or not len(lcq.data):
                log("Warning: No ZTF data found")
                return

            # Validate query results before caching
            ztf_raw = Table.from_pandas(lcq.data)
            if not ztf_required_columns.issubset(ztf_raw.colnames):
                raise RuntimeError(
                    f"ZTF query returned invalid data (missing columns: "
                    f"{sorted(ztf_required_columns - set(ztf_raw.colnames))}). "
                    f"Got columns: {ztf_raw.colnames}"
                )

            cache.save(ztf_raw)

        # Use cached or freshly queried raw data
        ztf = cache.data

    log(f"{len(ztf)} ZTF data points found")
    for fn in ['zg', 'zr', 'zi']:
        idx = ztf['filtercode'] == fn
        idx_good = idx & (ztf['catflags'] == 0) & (ztf['magerr'] < 0.05)

        log(f"  {fn}: {np.sum(idx)} total, {np.sum(idx_good)} good")

    log("Earliest: ", Time(np.min(ztf['mjd']), format='mjd').datetime.strftime('%Y-%m-%s %H:%M:%S'))
    log("  Latest: ", Time(np.max(ztf['mjd']), format='mjd').datetime.strftime('%Y-%m-%s %H:%M:%S'))

    if not np.sum(ztf['filtercode'] == 'zr') and not np.sum(ztf['filtercode'] == 'zg'):
        log("Warning: No datapoints in zg or zr filters")
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

    # ZTF also observes in i on occasion. No colour model is reconstructed for
    # it, so it is published as measured rather than dropped - under the same
    # quality cut as the two calibrated bands.
    ztf['mag_i'] = np.nan
    idx_i = (ztf['filtercode'] == 'zi') & (ztf['magerr'] < 0.15) & (ztf['catflags'] == 0)
    ztf['mag_i'][idx_i] = ztf['mag'][idx_i]

    # Most of ZTF's points are in r, and for a good many targets they are the
    # only ones there are, so the combined light curve takes them too rather
    # than showing an empty stretch where ZTF has the best coverage of all.
    # Converted with the colour model reconstructed just above - measured, per
    # epoch, from this star - which makes this the one conversion in the whole
    # combined curve that does not rest on an assumed constant colour.
    ztf['mag_g_from_r'] = np.nan
    idx_from_r = np.isfinite(ztf['mag_r'])
    if np.any(idx_from_r):
        ztf['mag_g_from_r'][idx_from_r] = (ztf['mag_r'][idx_from_r]
                                           + gr(ztf['time'][idx_from_r].mjd))

    color_model = config.get('ztf_color_model', 'constant')
    n_g = int(np.sum(np.isfinite(ztf['mag_g'])))
    n_r = int(np.sum(np.isfinite(ztf['mag_r'])))
    n_i = int(np.sum(idx_i))

    log_conversion(
        log, 'ZTF',
        'mag = mag_ZTF + clrcoeff * (g - r)',
        {
            'colour model': (color_model, 'gp = smoothed in time, constant = single median'),
            'mean (g - r)': float(np.mean(gr(u_mjd))),
            'clrcoeff': ('per epoch, from the ZTF archive', 'not assumed'),
            'colour pairs used': len(iig),
        },
        npoints=n_g + n_r,
        note='without this the zero point drifts with the colour of the star, '
             'so the raw ZTF magnitudes are not published as a band',
    )

    if n_i:
        log_conversion(
            log, 'ZTF',
            'i = mag_ZTF   (no conversion applied)',
            {'clrcoeff': ('present in the data but unused', 'no colour model for i')},
            npoints=n_i,
            note='published as measured, so the points are not lost',
        )

    if n_r:
        log_conversion(
            log, 'ZTF',
            'g = r + (g - r)',
            {'(g - r)': (f"{color_model} model reconstructed above",
                         'measured per epoch, not assumed')},
            npoints=n_r,
            note='so that the r points, usually the most numerous, reach the '
                 'combined light curve as well',
        )

    log_bands(log, 'ZTF', [
        {'label': 'g', 'kind': 'calibrated', 'npoints': n_g,
         'note': 'zg on the Pan-STARRS g scale'},
        {'label': 'r', 'kind': 'calibrated', 'npoints': n_r,
         'note': 'zr on the Pan-STARRS r scale'},
        {'label': 'i', 'kind': 'native', 'npoints': n_i,
         'note': 'zi as measured, no colour correction'},
        {'label': 'g (from r)', 'kind': 'derived', 'npoints': n_r,
         'note': 'zr moved onto g by the measured colour'},
    ])

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

    if not len(cmagg):
        log("No multiband data in ZTF")
        return

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
