"""Target info acquisition module.

Resolves target coordinates and fetches catalog photometry from SIMBAD,
Gaia DR3, Pan-STARRS DR2, and other catalogs.
"""

import os
import numpy as np
import requests
from io import BytesIO

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

from astroquery.simbad import Simbad
from astroquery.ipac.irsa.irsa_dust import IrsaDust
from astroquery.vizier import Vizier

# STDPipe
from stdpipe import catalogs, resolve, plots

from ..surveys import survey_source, get_all_output_files
from .utils import cleanup_paths, cached_votable_query, log_bands, log_conversion


# Gaia timestamps are barycentric JD in TCB, counted from 2010-01-01
GAIA_MJD0 = 2455197.5 - 2400000.5

# Cone radius for the epoch photometry, in arcsec
GAIA_EPPHOT_SR = 5.0

# The sampled XP spectra, as Vizier serves them, and the radius to look in
GAIA_XP_CATALOGUE = 'I/355/xpsample'
GAIA_XP_SR = 5.0

# AAVSO VSX, the variable star index, and the radius to look in. Wider than the
# cones above: many VSX positions come from surveys with coarse astrometry, or
# from discovery papers a century old.
VSX_CATALOGUE = 'B/vsx/vsx'
VSX_SR = 10.0

# What VSX means by its variability flag
VSX_FLAGS = {
    0: 'variable',
    1: 'suspected variable',
    2: 'constant or non-existing',
    3: 'misidentification or duplicate',
    4: 'not found',
}

# HJD to MJD, the epochs being quoted the old way
JD_TO_MJD = 2400000.5

# Bands worth tabulating the extinction in: the ones the sources here actually
# measure. IRSA offers twenty-five, most of them for instruments we never see.
DUST_FILTERS = [
    'CTIO U', 'CTIO B', 'CTIO V', 'CTIO R', 'CTIO I',
    'SDSS u', 'SDSS g', 'SDSS r', 'SDSS i', 'SDSS z',
    '2MASS J', '2MASS H', '2MASS Ks',
    'WISE-1', 'WISE-2',
]

# Astronomical unit in km divided by the Julian year in seconds: turns a proper
# motion in arcsec per year at a distance in parsecs into km/s
PM_TO_KMS = 4.740470446

# The band the extinction law is anchored at when Gaia's A0 is scaled onto the
# others. A0 is monochromatic at 541.4 nm, which is V to well within the
# accuracy of any of this.
DUST_ANCHOR = 'CTIO V'

# Per band: name, and the columns holding its time, magnitude, flux, flux error
# and the flag raised when the variability analysis rejected the measurement
GAIA_EPPHOT_BANDS = [
    ('G',  'TimeG',  'Gmag',  'FG',  'e_FG',  'GrVFlag'),
    ('BP', 'TimeBP', 'BPmag', 'FBP', 'e_FBP', 'BPrVFlag'),
    ('RP', 'TimeRP', 'RPmag', 'FRP', 'e_FRP', 'RPrVFlag'),
]


def _has(row, key):
    """Whether a row - a dict or a table row - carries a column at all.

    A table Row defines no membership test of its own, so `in` would fall back
    to iterating its values and compare the name against each of them.
    """
    return key in (row.colnames if hasattr(row, 'colnames') else row)


def _number(row, key):
    """A column as a float, or None where Vizier has nothing to give.

    A missing cell arrives masked rather than absent or NaN, so it answers to
    neither of the usual tests.
    """
    if not _has(row, key):
        return None

    value = row[key]

    if value is None or value is np.ma.masked:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if np.isfinite(value) else None


def _text(row, key):
    """A column as a stripped string, blank where Vizier has nothing to give."""
    if not _has(row, key):
        return ''

    value = row[key]

    if value is None or value is np.ma.masked:
        return ''

    return str(value).strip()


def _vsx_magnitude(row, key):
    """A VSX magnitude with everything it is qualified by.

    Each of them carries a limit flag when the star was only seen to be
    brighter or fainter than that, a colon when the value itself is uncertain,
    and the passband it was measured in - which is as often unfiltered or
    photographic as it is a standard one.
    """
    value = _number(row, key)

    if value is None:
        return None

    text = f"{_text(row, 'l_' + key)}{value:.2f}{_text(row, 'u_' + key)}"
    band = _text(row, 'n_' + key)

    return f"{text} {band}" if band else text


@survey_source(
    name='Target Info',
    short_name='Info',
    state_acquiring='acquiring info',
    state_acquired='info acquired',
    log_file='info.log',
    output_files=['info.log', 'galaxy_map.png', 'ps1.vot', 'ps1.txt',
                  'gaia.vot', 'gaia.txt', 'gaia_xp.png', 'gaia_xp.vot',
                  'gaia_xp.txt'],
    button_text='Get Target Info',
    button_class='btn-info',
    # Coordinates every source queries by, and the colours they convert with
    provides_config=['target_ra', 'target_dec', 'B_minus_V', 'g_minus_r',
                     'BP_minus_RP'],
    # It begins by deleting every source's output files
    clears_other_sources=True,
    help_text='Resolve target coordinates and fetch catalog photometry',
    order=1,
    # It resolves the target and describes it, which it can always do; the
    # photometry and the spectrum below are what it finds along the way
    data_files=[],
    # Spectral viewer: Gaia publishes XP in nanometres
    spectrum_files='gaia_xp.txt',
    spectrum_label='Gaia XP',
    spectrum_wavelength_scale=10.0,
    spectrum_color='#2980b9',
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

    # AAVSO VSX, where a variable star is described by whoever has looked at it
    # most recently - so this is the place its period and type come from, if
    # anyone has ever published either
    log("\n---- AAVSO VSX ----\n")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    cache_name = f"vsx_{ra:.4f}_{dec:.4f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'AAVSO VSX',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            try:
                res = Vizier(columns=['**'], row_limit=-1).query_region(
                    SkyCoord(ra, dec, unit='deg'), radius=VSX_SR*u.arcsec,
                    catalog=VSX_CATALOGUE)

                vsx = res[0] if res and len(res) else None

                if vsx is not None and len(vsx):
                    cache.save(vsx)
                else:
                    vsx = None
            except Exception as e:
                log(f"Error: could not reach VizieR: {e}")
                vsx = None
        else:
            vsx = cache.data

    if vsx is not None and len(vsx):
        # Vizier's own _r is in the units the radius was given in, so the
        # separations are measured here rather than read off
        sep = SkyCoord(ra, dec, unit='deg').separation(
            SkyCoord(np.asarray(vsx['RAJ2000'], dtype=float),
                     np.asarray(vsx['DEJ2000'], dtype=float),
                     unit='deg')).arcsec
        vsx = vsx[np.argsort(sep)]
        sep = np.sort(sep)

        if len(vsx) > 1:
            log(f"{len(vsx)} entries within {VSX_SR:.0f} arcsec, nearest first")

        for i, row in enumerate(vsx):
            # A zero flag is the ordinary case - a variable star, which is what
            # the catalogue is made of - so only the rest is worth saying
            flag = _number(row, 'V')
            flag = VSX_FLAGS.get(int(flag)) if flag else None

            log(f"{_text(row, 'Name') or '(unnamed)'} at {sep[i]:.1f} arcsec"
                + (f" - {flag}" if flag else ""))

            vtype = _text(row, 'Type')
            if vtype:
                log(f"  Type = {vtype}")

            bright = _vsx_magnitude(row, 'max')
            faint = _vsx_magnitude(row, 'min')

            if bright and faint:
                # The second column holds an amplitude rather than a magnitude
                # whenever this flag is raised, which is how VSX records the
                # stars whose faint state nobody has measured
                if _text(row, 'f_min') == 'Y':
                    log(f"  Brightness = {bright}, amplitude {faint}")
                else:
                    log(f"  Brightness = {bright} ... {faint}")
            elif bright:
                log(f"  Brightness = {bright}")

            period = _number(row, 'Period')
            if period is not None:
                log(f"  Period = {_text(row, 'l_Period')}{period:.8g}"
                    f"{_text(row, 'u_Period')} d")

            epoch = _number(row, 'Epoch')
            if epoch is not None:
                # Quoted as an HJD, while every lightcurve here is in MJD
                log(f"  Epoch = HJD {epoch:.4f}{_text(row, 'u_Epoch')} "
                    f"= MJD {epoch - JD_TO_MJD:.4f}")

            sptype = _text(row, 'Sp')
            if sptype:
                log(f"  SpType = {sptype}")

            oid = _number(row, 'OID')
            if oid is not None:
                log("  https://www.aavso.org/vsx/index.php?view=detail.top&oid="
                    f"{int(oid)}")

            # The nearest entry only: a second one is a neighbour, not a better
            # description of the same star
            if not i:
                for key, name in [('Name', 'vsx_name'), ('Type', 'vsx_type')]:
                    if _text(row, key):
                        config[name] = _text(row, key)

                for key, name in [('Period', 'vsx_period'), ('Epoch', 'vsx_epoch')]:
                    if _number(row, key) is not None:
                        config[name] = _number(row, key)
    else:
        log(f"No VSX entries within {VSX_SR:.0f} arcsec")

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
                                          extra=['_RAJ2000', '_DEJ2000', 'e_Gmag', 'e_BPmag', 'e_RPmag',
                                                 'A0', 'b_A0', 'B_A0', 'AG', 'b_AG', 'B_AG',
                                                 'Source', 'XPsamp'],
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

        # Extinction to the star itself, fitted from its own spectrum and
        # parallax - unlike the maps below, which integrate the whole column
        # through the Galaxy. Only a subset of Gaia sources has it, and an
        # older cache was written before these columns were asked for.
        for name, label in [('A0', 'A0 (551 nm)'), ('AG', 'A(G)')]:
            value = star.get(name)

            if value is None or not np.isfinite(value):
                continue

            lo, hi = star.get(f'b_{name}'), star.get(f'B_{name}')
            span = (f" [{lo:.3f} ... {hi:.3f}]"
                    if lo is not None and hi is not None
                    and np.isfinite(lo) and np.isfinite(hi) else "")

            log(f"Gaia {label} = {value:.3f}{span} mag")
            config[f'gaia_{name}'] = float(value)

        if 'gaia_A0' not in config and 'A0' not in cat.colnames:
            log("Gaia extinction was not in the cached reply - "
                "tick 'Ignore cache' to fetch it")

        # Kept for the sections below, which need the star found here
        if _number(star, 'Source') is not None:
            config['gaia_source'] = int(star['Source'])

        # Said rather than left to silence: most sources have no XP spectrum,
        # and its absence below is otherwise indistinguishable from a failure
        if _number(star, 'XPsamp') == 0:
            log("No XP spectrum was published for this source")

        # Kept for the distances section below, which turns them into a
        # velocity once it knows how far away the star is
        for key, name in [('pmRA', 'gaia_pmra'), ('pmDE', 'gaia_pmdec'),
                          ('RV', 'gaia_rv')]:
            value = _number(star, key)
            if value is not None:
                config[name] = value

    # Gaia DR3 XP spectrum
    #
    # Published for a fraction of the catalogue only, and the catalogue says
    # which - so nothing is asked for that is known not to be there.
    if config.get('gaia_source') and _number(star, 'XPsamp'):
        log("\n---- Gaia DR3 XP spectrum ----\n")

        cache_name = f"gaiaxp_{ra:.4f}_{dec:.4f}.vot"

        with cached_votable_query(cache_name, basepath, log, 'Gaia DR3 XP spectrum',
                                  refresh=refresh_cache) as cache:
            if not cache.hit:
                try:
                    # Vizier serves the same spectra as the mission archive -
                    # checked point by point, and identical to the precision
                    # they are stored at - which keeps this on the one service
                    # the rest of the step already talks to
                    res = Vizier(columns=['**'], row_limit=-1).query_region(
                        SkyCoord(ra, dec, unit='deg'), radius=GAIA_XP_SR*u.arcsec,
                        catalog=GAIA_XP_CATALOGUE)

                    xp = res[0] if res and len(res) else None

                    if xp is not None and len(xp):
                        # A cone this size can hold more than one Gaia source,
                        # each with its own spectrum, so the nearest is taken
                        if 'Source' in xp.colnames and config.get('gaia_source'):
                            wanted = xp['Source'] == config['gaia_source']
                            if np.sum(wanted):
                                xp = xp[wanted]

                        xp = Table({
                            'wavelength': np.asarray(xp['lambda'], dtype=float),
                            'flux': np.asarray(xp['Flux'], dtype=float),
                            'flux_error': np.asarray(xp['e_Flux'], dtype=float),
                        })
                        xp.sort('wavelength')
                        cache.save(xp)
                    else:
                        xp = None
                except Exception as e:
                    log(f"Error: could not fetch the XP spectrum: {e}")
                    xp = None
            else:
                xp = cache.data

        if xp is not None and len(xp):
            lam = np.asarray(xp['wavelength'], dtype=float)
            flux = np.asarray(xp['flux'], dtype=float)
            err = np.asarray(xp['flux_error'], dtype=float)

            log(f"{len(xp)} points from {lam.min():.0f} to {lam.max():.0f} nm")
            log("Sampled from the basis functions Gaia publishes rather than "
                "observed as drawn, at a resolution of some tens - the wiggles "
                "are mostly the reconstruction, not lines")

            # The fluxes run to 1e-14 or so, which no axis should have to say
            scale = 1e-16

            with plots.figure_saver(os.path.join(basepath, 'gaia_xp.png'),
                                    figsize=(8, 4), show=show) as fig:
                ax = fig.add_subplot(1, 1, 1)

                ax.fill_between(lam, (flux - err)/scale, (flux + err)/scale,
                                alpha=0.25, color='#2980b9', lw=0)
                ax.plot(lam, flux/scale, '-', lw=1, color='#2980b9')

                ax.grid(alpha=0.2)
                ax.set_xlabel('Wavelength, nm')
                ax.set_ylabel(r'Flux, $10^{-16}$ W nm$^{-1}$ m$^{-2}$')
                ax.set_title(f"{config['target_name']} - Gaia DR3 XP spectrum")

            log("XP spectrum plot saved to file:gaia_xp.png")

            # Both forms, as everything else here is written: the VOTable
            # keeps the units and the column types, the text one opens
            # anywhere
            xp.write(os.path.join(basepath, 'gaia_xp.vot'),
                     format='votable', overwrite=True)
            xp.write(os.path.join(basepath, 'gaia_xp.txt'),
                     format='ascii.commented_header', overwrite=True)
            log("XP spectrum written to file:gaia_xp.vot")
            log("XP spectrum written to file:gaia_xp.txt")

    # Interstellar reddening from the two-dimensional maps
    ra = config.get('target_ra')
    dec = config.get('target_dec')
    cache_name = f"dust_{ra:.4f}_{dec:.4f}.vot"

    log("\n---- Interstellar reddening ----\n")

    with cached_votable_query(cache_name, basepath, log, 'IRSA dust maps',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            try:
                res = IrsaDust.get_query_table(
                    SkyCoord(ra, dec, unit='deg'), section='ebv')

                # IRSA names its columns with spaces in them, which a VOTable
                # turns into underscores on the way back out, so the few
                # numbers worth keeping are copied into a table of our own
                cat = Table({
                    'ebv_sfd': [float(res['ext SFD ref'][0])],
                    'ebv_sfd_std': [float(res['ext SFD std'][0])],
                    'ebv_sf11': [float(res['ext SandF ref'][0])],
                    'ebv_sf11_std': [float(res['ext SandF std'][0])],
                })
                cache.save(cat)
            except Exception as e:
                log(f"Error: could not reach the IRSA dust service: {e}")
                cat = None
        else:
            cat = cache.data

    if cat is not None and len(cat):
        row = cat[0]

        # The whole column through the Galaxy, so an upper limit for anything
        # inside it - which is most things. Schlafly & Finkbeiner recalibrated
        # Schlegel, Finkbeiner & Davis downwards by some 14 per cent.
        log(f"E(B-V) = {row['ebv_sf11']:.4f} +/- {row['ebv_sf11_std']:.4f} "
            f"(Schlafly & Finkbeiner 2011)")
        log(f"E(B-V) = {row['ebv_sfd']:.4f} +/- {row['ebv_sfd_std']:.4f} "
            f"(Schlegel, Finkbeiner & Davis 1998)")
        log("  integrated through the whole Galaxy, so an upper limit for a "
            "star inside it")

        config['ebv_sf11'] = float(row['ebv_sf11'])
        config['ebv_sfd'] = float(row['ebv_sfd'])

        # A0 is very nearly A(V), so the two are worth comparing: a star well
        # in front of the dust shows much less than the full column
        if config.get('gaia_A0') is not None:
            full = 3.1 * row['ebv_sf11']
            log(f"  Gaia sees A0 = {config['gaia_A0']:.3f} mag towards the star "
                f"itself, against {full:.3f} mag for the full column at R(V) = 3.1")

            # Nothing interstellar can redden a star by more than the whole
            # column in front of it. Gaia's figure is fitted from the spectrum
            # assuming an ordinary stellar atmosphere, and comes out too large
            # for anything that is not one - a star with a disc around it, a
            # blend, an unresolved binary - where it is also degenerate with
            # temperature.
            if config['gaia_A0'] > full + 0.1:
                log("  which is more than the Galaxy holds in that direction, so "
                    "the fit is not measuring interstellar dust alone - expected "
                    "for a star with circumstellar material, a blend or a binary")

    # Extinction band by band
    cache_name = f"dustext_{ra:.4f}_{dec:.4f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'IRSA extinction table',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            try:
                ext = IrsaDust.get_extinction_table(SkyCoord(ra, dec, unit='deg'))
                # A byte string in a VOTable comes back as bytes, so the names
                # are settled here rather than at every use
                ext['Filter_name'] = [str(_) for _ in ext['Filter_name']]
                cache.save(ext)
            except Exception as e:
                log(f"Error: could not fetch the extinction table: {e}")
                ext = None
        else:
            ext = cache.data

    if ext is not None and len(ext):
        rows = {str(r['Filter_name']).strip(): r for r in ext}
        anchor = rows.get(DUST_ANCHOR)

        # Gaia measures how much extinction there is towards the star; the map
        # measures how it is distributed across the bands. Scaling the second
        # by the first gives the star's own extinction band by band, on the
        # assumption that the law has the usual shape and only the amount
        # differs - which is the assumption behind any such table.
        scale = None
        if anchor is not None and config.get('gaia_A0') is not None:
            coeff = float(anchor['A_over_E_B_V_SandF'])
            if coeff > 0:
                scale = config['gaia_A0'] / coeff

        log("")
        log(f"  {'band':10s} {'A(Gaia)':>9s} {'A(SF11)':>9s} {'A(SFD)':>9s}")

        for name in DUST_FILTERS:
            row = rows.get(name)

            if row is None:
                continue

            gaia = (f"{float(row['A_over_E_B_V_SandF']) * scale:9.3f}"
                    if scale is not None else f"{'-':>9s}")

            log(f"  {name:10s} {gaia} {float(row['A_SandF']):9.3f} "
                f"{float(row['A_SFD']):9.3f}")

        log("")
        log("  A(Gaia) is the star's own extinction, from its A0 spread over the")
        log("  bands by the same law; the other two are the whole column through")
        log("  the Galaxy, on the two calibrations of the same map.")

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
            # Named, as it is not a parallax turned upside down but the
            # geometric distance Bailer-Jones et al. infer from it with a prior
            log(f"Gaia DR3 distance is {star['rgeo']:.1f} [{star['b_rgeo']:.1f} ... {star['B_rgeo']:.1f}] pc"
                f" (geometric, Bailer-Jones et al. 2021)")

            log(f"Height above Galactic plane is {star['rgeo']*np.abs(np.sin(np.deg2rad(config.get('target_b')))):.1f} pc")

            # How it moves. The proper motion came from the main Gaia table
            # further up; here it can finally be turned into a velocity, the
            # distance being what the two need to be combined.
            pmra, pmdec = config.get('gaia_pmra'), config.get('gaia_pmdec')

            if pmra is not None and pmdec is not None:
                pm = np.hypot(pmra, pmdec)

                # Position angle of the motion, from north through east
                angle = np.degrees(np.arctan2(pmra, pmdec)) % 360

                log(f"Proper motion is {pm:.3f} mas/yr at {angle:.0f} deg")
                log(f"  pmRA* = {pmra:+.3f} mas/yr")
                log(f"  pmDE = {pmdec:+.3f} mas/yr)")

                # 4.74 km/s is one arcsec per year at one parsec
                vt = PM_TO_KMS * pm * 1e-3 * star['rgeo']
                log(f"Transverse velocity is {vt:.1f} km/s")

                rv = config.get('gaia_rv')

                if rv is not None:
                    log(f"Radial velocity is {rv:+.1f} km/s (Gaia RVS)")
                    log(f"Full spatial velocity is {np.hypot(vt, rv):.1f} km/s")
                else:
                    log("No RVS radial velocity, so this is the whole of the "
                        "motion that can be seen")

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
                                   timeout=120,
                                   params={'ra': ra, 'dec': dec, 'radius': 2/3600, 'format': 'csv',
                                          'columns': ['obsTime', 'filterID', 'psfQfPerfect', 'psfFlux', 'psfFluxErr']})
                ps1_raw = Table.read(BytesIO(res.content), format='csv')

                if ps1_raw and len(ps1_raw):
                    cache.save(ps1_raw)
                else:
                    ps1_raw = None
            except Exception as e:
                import traceback
                traceback.print_exc()
                log(f"Error: could not download the data - {type(e).__name__}: {e}")
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
        log("Warning: No Pan-STARRS DR2 warp data found")

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
            except Exception as e:
                import traceback
                traceback.print_exc()
                log(f"Error: could not download the data - {type(e).__name__}: {e}")
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
            log("Warning: No usable Gaia DR3 epoch photometry")

    else:
        log("Warning: No Gaia DR3 epoch photometry found - it is published "
            "only for the sources Gaia treats as variable")


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
