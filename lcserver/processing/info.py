"""Target info acquisition module.

Resolves target coordinates and fetches catalog photometry from SIMBAD,
Gaia DR3, Pan-STARRS DR2, and other catalogs.
"""

import os
import numpy as np
import requests
from io import BytesIO

from astropy.table import Table
from astropy import units as u
from astropy.time import Time

from astroquery.simbad import Simbad

# STDPipe
from stdpipe import catalogs, resolve, plots

from ..surveys import survey_source, get_all_output_files
from .utils import cleanup_paths, cached_votable_query, log_bands, log_conversion


# Gaia timestamps are barycentric JD in TCB, counted from 2010-01-01
GAIA_MJD0 = 2455197.5 - 2400000.5

# Cone radius for the epoch photometry, in arcsec
GAIA_EPPHOT_SR = 5.0

# Per band: name, and the columns holding its time, magnitude, flux, flux error
# and the flag raised when the variability analysis rejected the measurement
GAIA_EPPHOT_BANDS = [
    ('G',  'TimeG',  'Gmag',  'FG',  'e_FG',  'GrVFlag'),
    ('BP', 'TimeBP', 'BPmag', 'FBP', 'e_FBP', 'BPrVFlag'),
    ('RP', 'TimeRP', 'RPmag', 'FRP', 'e_FRP', 'RPrVFlag'),
]


@survey_source(
    name='Target Info',
    short_name='Info',
    state_acquiring='acquiring info',
    state_acquired='info acquired',
    log_file='info.log',
    output_files=['info.log', 'galaxy_map.png', 'ps1.vot', 'ps1.txt',
                  'gaia.vot', 'gaia.txt'],
    button_text='Get Target Info',
    button_class='btn-info',
    help_text='Resolve target coordinates and fetch catalog photometry',
    order=1,
    # Template metadata
    template_layout='custom',
)
def target_info(config, basepath=None, verbose=True, show=False):
    """
    Acquire basic info on the target.

    Parameters
    ----------
    config : dict
        Configuration dictionary with target_name
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
    cleanup_paths(get_all_output_files(), basepath=basepath)

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

    # Create safe name for caching
    import re
    safe_name = re.sub(r'[^\w\-.]', '_', config['target_name'])
    cache_name = f"simbad_{safe_name}.vot"

    with cached_votable_query(cache_name, basepath, log, 'SIMBAD', refresh=refresh_cache) as cache:
        if not cache.hit:
            # Query SIMBAD - only if not cached
            sim = Simbad()
            sim.add_votable_fields('otype', 'otypes', 'alltypes', 'ids', 'distance', 'sptype')

            res = sim.query_region(target, radius=5*u.arcsec)

            if res and len(res):
                cache.save(res)
            else:
                res = None
        else:
            res = cache.data

    if not res or not len(res):
        log("No SIMBAD objects within 5 arcsec")
    else:
        for r in res:
            # TODO: select closest object?..
            log(f"{r['main_id']} = {r['otype']}") #
            log(f"{r['alltypes.otypes']}")

            if r['sp_type']:
                log(f"SpType = {r['sp_type']}")
            # if r['Distance_distance']:
            #     log(f"Dist = {r['Distance_distance']:.2f} +{r['Distance_perr']:.2f} -{-r['Distance_merr']:.2f} {r['Distance_unit']}")

            break

    # Catalogues to get photometry
    for catname in ['gaiadr3syn', 'ps1', 'skymapper']:
        ra = config.get('target_ra')
        dec = config.get('target_dec')
        cache_name = f"{catname}_{ra:.4f}_{dec:.4f}.vot"

        # Before the query, so that the line reporting where the data came from
        # falls under the heading of the catalogue it belongs to
        log(f"\n---- {catalogs.catalogs[catname]['name']} ----\n")

        with cached_votable_query(cache_name, basepath, log, catalogs.catalogs[catname]['name'], refresh=refresh_cache) as cache:
            if not cache.hit:
                # Query catalog - only if not cached
                cat = catalogs.get_cat_vizier(ra, dec, 5/3600,
                                              catname, get_distance=True, verbose=False)
                if cat and len(cat):
                    cache.save(cat)
                else:
                    cat = None
            else:
                cat = cache.data

        if not cat or not len(cat):
            log("Nothing found")
            continue

        star = dict(cat[cat['_r'] == np.min(cat['_r'])][0])

        nmags = 0
        for fn in ['B', 'V', 'R', 'I', 'g', 'r', 'i', 'z']:
            if star.get(f'{fn}mag'):
                log(f"{fn} = {star[f'{fn}mag']:.2f} +/- {star[f'e_{fn}mag']:.2f}")
                nmags += 1

        # A match with none of the bands we look at would otherwise leave the
        # section looking as though something had gone wrong
        if not nmags:
            log(f"Matched a source, but it carries none of the magnitudes we use")

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
    ra = config.get('target_ra')
    dec = config.get('target_dec')
    cache_name = f"gaiadr3_phot_{ra:.4f}_{dec:.4f}.vot"

    log(f"\n---- Gaia DR3 ----\n")

    with cached_votable_query(cache_name, basepath, log, 'Gaia DR3 photometry', refresh=refresh_cache) as cache:
        if not cache.hit:
            # Query Gaia DR3 - only if not cached
            cat = catalogs.get_cat_vizier(ra, dec, 5/3600,
                                          'I/355/gaiadr3',
                                          extra=['_RAJ2000', '_DEJ2000', 'e_Gmag', 'e_BPmag', 'e_RPmag'],
                                          get_distance=True, verbose=False)
            if cat and len(cat):
                cache.save(cat)
            else:
                cat = None
        else:
            cat = cache.data

    if cat:
        star = dict(cat[cat['_r'] == np.min(cat['_r'])][0])

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
    ra = config.get('target_ra')
    dec = config.get('target_dec')
    cache_name = f"gaiadr3_dist_{ra:.4f}_{dec:.4f}.vot"

    log(f"\n---- Gaia DR3 distances ----\n")

    with cached_votable_query(cache_name, basepath, log, 'Gaia DR3 distances', refresh=refresh_cache) as cache:
        if not cache.hit:
            # Query Gaia DR3 distances - only if not cached
            cat = catalogs.get_cat_vizier(ra, dec, 5/3600,
                                          'I/352/gedr3dis', extra=['_RAJ2000', '_DEJ2000'],
                                          get_distance=True, verbose=False)
            if cat and len(cat):
                cache.save(cat)
            else:
                cat = None
        else:
            cat = cache.data

    if cat:
        star = cat[cat['_r'] == np.min(cat['_r'])][0]

        if star['rgeo']:
            log(f"Gaia DR3 distance is {star['rgeo']:.1f} [{star['b_rgeo']:.1f} ... {star['B_rgeo']:.1f}] pc")

            log(f"Height above Galactic plane is {star['rgeo']*np.abs(np.sin(np.deg2rad(config.get('target_b')))):.1f} pc")

            # Galaxy map
            from matplotlib import image
            from PIL import Image
            import urllib

            # url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Artist%27s_impression_of_the_Milky_Way_%28updated_-_annotated%29.jpg/1024px-Artist%27s_impression_of_the_Milky_Way_%28updated_-_annotated%29.jpg'
            # galaxy_image = np.array(Image.open(urllib.request.urlopen(url)))
            path = '1024px-Artist\'s_impression_of_the_Milky_Way_(updated_-_annotated).jpg'
            galaxy_image = np.array(Image.open(path))

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

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    cache_name = f"ps1_warp_{ra:.4f}_{dec:.4f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'Pan-STARRS DR2 warp', refresh=refresh_cache) as cache:
        if not cache.hit:
            # Query Pan-STARRS DR2 - only if not cached
            try:
                res = requests.get('https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/detection.csv',
                                   params={'ra': ra, 'dec': dec, 'radius': 2/3600, 'format': 'csv',
                                          'columns': ['obsTime', 'filterID', 'psfQfPerfect', 'psfFlux', 'psfFluxErr']})
                ps1_raw = Table.read(BytesIO(res.content), format='csv')

                if ps1_raw and len(ps1_raw):
                    cache.save(ps1_raw)
                else:
                    ps1_raw = None
            except:
                import traceback
                traceback.print_exc()
                log("Error while downloading the data")
                ps1_raw = None
        else:
            ps1_raw = cache.data

    # Apply quality cut
    ps1 = None
    if ps1_raw and len(ps1_raw):
        ps1 = ps1_raw[ps1_raw['psfQfPerfect'] > 0.95]  # Quality cut

    if ps1 and len(ps1):
        ps1.sort('obsTime')

        ps1['time'] = Time(ps1['obsTime'], format='mjd')
        ps1['mjd'] = ps1['time'].mjd
        ps1['mag'] = -2.5*np.log10(ps1['psfFlux']) + 8.90 # Janskys to AB?..
        ps1['magerr'] = 2.5/np.log(10)*(ps1['psfFluxErr']/ps1['psfFlux'])

        # Pan-STARRS reports each band on its own scale already, so the bands
        # are split apart rather than converted into one another
        ps1_filters = [[1, 'g'], [2, 'r'], [3, 'i'], [4, 'z'], [5, 'y']]

        # Explicit width, so that a band name is never silently truncated
        ps1['filter'] = np.full(len(ps1), '', dtype='<U2')
        for fid, fn in ps1_filters:
            idx = ps1['filterID'] == fid

            log(f"{fn}: {np.sum(idx)} good points")

            ps1['filter'][idx] = fn
            ps1['mag_' + fn] = np.nan
            ps1['mag_' + fn][idx] = ps1['mag'][idx]

        log_conversion(
            log, 'Pan-STARRS',
            'no conversion applied - every band is published as measured',
            {'colour term': ('none', 'PSF fluxes converted to AB magnitudes only'),
             'zero point': '-2.5*log10(psfFlux) + 8.90'},
            npoints=len(ps1),
            heading=False,
        )

        log_bands(log, 'Pan-STARRS', [
            {'label': fn, 'kind': 'native',
             'npoints': int(np.sum(ps1['filterID'] == fid)),
             'note': 'warp photometry, as reported'}
            for fid, fn in ps1_filters
        ], heading=False)

        # Color?..
        ig,ir = np.where(ps1['filterID'] == 1)[0], np.where(ps1['filterID'] == 2)[0]
        mg,mr = [],[]

        for i in ig:
            dist = np.abs(ps1['mjd'][i] - ps1['mjd'][ir])
            if len(dist) and np.min(dist) < 1:
                mg.append(ps1['mag'][i])
                mr.append(ps1['mag'][ir[dist == np.min(dist)]][0])

        if len(mg):
            mg,mr = [np.array(_) for _ in (mg,mr)]

            log(f"{len(mg)} quasi-simultaneous measurements")
            log(f"(g - r) = {np.nanmean(mg-mr):.3f} +/- {np.nanstd(mg-mr):.3f}")

        # Time cannot be serialized to VOTable
        ps1[[_ for _ in ps1.columns if _ != 'time']].write(os.path.join(basepath, 'ps1.vot'), format='votable', overwrite=True)
        ps1[[_ for _ in ps1.columns if _ != 'time']].write(os.path.join(basepath, 'ps1.txt'), format='ascii.commented_header', overwrite=True)
        log("Pan-STARRS DR2 warp photometry written to file:ps1.vot")
        log("Pan-STARRS DR2 warp photometry written to file:ps1.txt")

    else:
        log("No Pan-STARRS DR2 warp data found")

    # Gaia DR3 epoch photometry
    log("\n---- Gaia DR3 epoch photometry ----\n")

    cache_name = f"gaiadr3_epphot_{ra:.4f}_{dec:.4f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'Gaia DR3 epoch photometry',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            try:
                # Published for the sources Gaia flags as variable, so most
                # targets have none of it
                epphot = catalogs.get_cat_vizier(ra, dec, GAIA_EPPHOT_SR/3600,
                                                 'I/355/epphot',
                                                 augment_bands=False, verbose=False)
                if epphot and len(epphot):
                    cache.save(epphot)
                else:
                    epphot = None
            except:
                import traceback
                traceback.print_exc()
                log("Error while downloading the data")
                epphot = None
        else:
            epphot = cache.data

    if epphot and len(epphot):
        # A cone of a few arcsec may still catch a neighbour, and mixing two
        # stars' transits would be worse than missing one
        sources = np.unique(epphot['Source'])
        if len(sources) > 1:
            dist = np.hypot((np.asarray(epphot['RA_ICRS'], float) - ra)
                            * np.cos(np.deg2rad(dec)),
                            np.asarray(epphot['DE_ICRS'], float) - dec)
            closest = epphot['Source'][np.argmin(dist)]
            log(f"{len(sources)} Gaia sources within {GAIA_EPPHOT_SR:.0f} arcsec, "
                f"keeping the closest one ({closest})")
            epphot = epphot[epphot['Source'] == closest]

        # Gaia gives one row per transit, carrying a time and a magnitude for
        # each band at once. Unpacked here into one row per measurement, the way
        # every other source is stored.
        mjd, mag, magerr, filt, transit = [], [], [], [], []

        for band, tcol, mcol, fcol, ecol, rcol in GAIA_EPPHOT_BANDS:
            t_ = np.asarray(epphot[tcol], float)
            m_ = np.asarray(epphot[mcol], float)
            f_ = np.asarray(epphot[fcol], float)
            e_ = np.asarray(epphot[ecol], float)
            rejected = np.asarray(epphot[rcol]).astype(int) != 0

            idx = np.isfinite(t_) & np.isfinite(m_) & np.isfinite(f_) & (f_ > 0)

            log(f"{band}: {int(np.sum(idx))} epochs"
                + (f", {int(np.sum(idx & rejected))} rejected by Gaia"
                   if np.any(idx & rejected) else ''))

            idx &= ~rejected

            mjd.append(t_[idx] + GAIA_MJD0)
            mag.append(m_[idx])
            # Gaia quotes fluxes and their errors rather than magnitude errors
            magerr.append(2.5/np.log(10) * e_[idx]/f_[idx])
            filt.append(np.full(int(np.sum(idx)), band, dtype='<U2'))
            transit.append(np.asarray(epphot['TransitID'])[idx])

        gaia = Table({
            'mjd': np.concatenate(mjd),
            'mag': np.concatenate(mag),
            'magerr': np.concatenate(magerr),
            'filter': np.concatenate(filt),
            'TransitID': np.concatenate(transit),
        })
        gaia.sort('mjd')

        if len(gaia):
            log_conversion(
                log, 'Gaia DR3',
                'no conversion applied - each band is published as measured',
                {'colour term': ('none', 'G, BP and RP are already on the Gaia scale'),
                 'magnitude error': '2.5/ln(10) * e_F/F, as Gaia quotes fluxes'},
                npoints=len(gaia),
                heading=False,
            )

            log_bands(log, 'Gaia DR3', [
                {'label': band, 'kind': 'native',
                 'npoints': int(np.sum(gaia['filter'] == band)),
                 'note': 'per-transit photometry, as reported'}
                for band, *_ in GAIA_EPPHOT_BANDS
            ], heading=False)

            gaia.write(os.path.join(basepath, 'gaia.vot'), format='votable', overwrite=True)
            gaia.write(os.path.join(basepath, 'gaia.txt'), format='ascii.commented_header', overwrite=True)
            log("Gaia DR3 epoch photometry written to file:gaia.vot")
            log("Gaia DR3 epoch photometry written to file:gaia.txt")
        else:
            log("No usable Gaia DR3 epoch photometry")

    else:
        log("No Gaia DR3 epoch photometry found - it is published only for the "
            "sources Gaia treats as variable")


# Register lightcurve-only sources (no processing function)
# These sources have data files but no automated acquisition
from .. import surveys

surveys.register_lightcurve_source(
    source_id='ps1',
    name='Pan-STARRS',
    short_name='Pan-STARRS',
    votable_file='ps1.vot',
    lc_bands=[
        surveys.band(fn, 'mag_' + fn, 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value=fn, color=color,
                     note='Pan-STARRS DR2 warp photometry, as reported')
        for fn, color in [('g', '#2ca02c'), ('r', '#d62728'), ('i', '#9467bd'),
                          ('z', '#8c564b'), ('y', '#7f7f7f')]
    ],
    lc_mag_column='mag_g',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color='#2ca02c',
    lc_mode='magnitude',
    lc_short=True,
)

surveys.register_lightcurve_source(
    source_id='gaia',
    name='Gaia DR3',
    short_name='Gaia',
    votable_file='gaia.vot',
    lc_bands=[
        surveys.band(band, 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value=band, color=color,
                     note='Gaia DR3 per-transit photometry, as reported')
        for band, color in [('G', '#333333'), ('BP', '#1f77b4'), ('RP', '#d62728')]
    ],
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color='#333333',
    lc_mode='magnitude',
    lc_short=True,
)
