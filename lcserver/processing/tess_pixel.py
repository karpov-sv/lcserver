"""TESS light curves measured in the full-frame images themselves.

For a star the pipelines give no light curve of their own - too faint to have
been a target, or so close to a brighter one that it falls into that star's
aperture. There its variability turns up in the neighbour's light curve,
diluted by the neighbour's light and often flattened by the detrending, which
fits away anything slower than a day or so.

Here a cutout of the full-frame images around the target is fitted frame by
frame instead. Each frame minus the median of the sector is described by:

- the target's image profile, whose amplitude is its change in flux, e/s;
- the profiles of the brighter neighbours near it, one amplitude each, so that
  their own variability is not taken for the target's;
- a shift of the pointing and a change of focus, shared by every star - what
  a constant star leaves in a difference image when the field moves under it.
  First as the derivatives of all their Gaussian profiles, weighted by their
  catalogue fluxes; then, the shift and focus of every frame known, as maps
  of how each pixel actually answers them, learned from the sector itself,
  and the frames fitted again with those;
- a plane of background, for the scattered light.

The profile is a Gaussian, its width fitted to the median image and then, for
a star that varies enough, to the variable light itself - whose centre is
also logged, as the check that what varies is the target and not a
neighbour. A Gaussian is not TESS's PSF, but a difference image only asks it
to separate the target from neighbours a pixel or more away, which it does.
The scale of the amplitude rests on it all the same, and is good to some
15-20%: against light curves made with the shape of the variation known, and
against simultaneous ground photometry, the amplitudes differ by that much
sector to sector, either way.

What comes out is the change of the target's flux, which is then put on the
flux the TIC magnitude implies - the mean itself cannot be measured in a
field this crowded with a profile this rough. The viewer normalizes it to its
median like any other TESS light curve.

What it is not for, and where it stops, from the targets it was tried on:

- A bright star alone in its pixels. With the Gaussian's own derivatives, a
  constant star of Tmag 12 followed its sector's pointing jumps in steps of
  up to 1.5%; with the learned maps 0.12% is left, against 0.08% in QLP's own
  aperture photometry - close, but the pipelines remain the better choice
  there, and the log says so.
- Near a very bright neighbour the learned maps matter most: a star of Tmag
  10.5 a minute of arc from a faint target left ramps of ~20 e/s over each
  orbit with the Gaussian derivatives, and ~1 e/s with the maps.
- Steps between the orbits of a sector, from a single median image as the
  reference: about 0.5-2 e/s for faint targets, 0.3% for a bright one.
- The maps are learned from how the pointing varied, so they are noisier in
  a quiet sector; in two of eight sectors of one target the frame-to-frame
  scatter rose by half.
- Scattered light beyond what the plane takes up: a few e/s, with the scatter
  doubled, while the background is high. Such frames get a quality bit of
  their own, which the standard filtering drops.

A target whose own light is lost in the noise - faint, or not in the TIC at
all, when the nearest catalogue star is a different one - keeps its light
curve, as a change of flux in e/s, but is kept out of the viewer, whose
normalization would make that noise into swings of hundreds of percent. A
sector where the target is off the detector, which TESScut still returns, is
skipped.
"""

import os
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u

from stdpipe import plots

from .utils import (break_at_gaps, cached_lightkurve_search, cached_votable_query,
                    mast_download_dir, mission_quality_mask, log_quality,
                    quality_level, bin_width, bin_sector)


# The cutout, in TESS pixels of 21 arcsec, and how much of it is fitted around
# the target. The cutout reaches past the fitted part so that the stars just
# outside it, whose wings fall inside, are in the model of the median image.
CUTOUT_SIZE = 15
FIT_RADIUS = 3.0

# Neighbours get an amplitude of their own if they are this close to the
# target, in pixels, and at least this bright relative to it; at most this
# many, the brightest first. A star nearer than the last limit cannot be told
# from the target at TESS resolution at all, and is only named in the log.
NEIGHBOUR_RADIUS = FIT_RADIUS + 1.5
NEIGHBOUR_MIN_RATIO = 0.2
NEIGHBOUR_MAX = 8
NEIGHBOUR_UNRESOLVED = 0.6

# How faint a catalogue star still counts towards the shift and focus terms
# and the model of the median image
TIC_FAINT_LIMIT = 17.5

# A TIC entry within this distance is the target itself, arcsec. Tight, as a
# faint target may well be missing from the TIC, and the nearest entry a few
# arcsec away is then another star - whose magnitude the light curve would be
# put on
TIC_MATCH = 2.5

# The quality bit set here on frames whose background is high enough to show
# in the target. Unused by the mission (bits 0-15) and by QLP (29, 30); the
# standard filtering, which keeps only unflagged frames, drops these too.
# What of the background leaks into the target was measured on 18 sectors of
# four targets: 0.4-4% of its excess over the sector's median, either way,
# 1.3% typically. A frame is flagged where that much of its excess would
# exceed the light curve's frame-to-frame scatter - which in the sectors
# examined picked out frames with twice the scatter of the rest, and left
# the quiet sectors alone.
QUALITY_HIGH_BACKGROUND = 1 << 20
BACKGROUND_LEAKAGE = 0.01

# The cadences of the full-frame images through the missions, s; a sector's
# measured spacing is named by the nearest
FFI_CADENCES = (200, 600, 1800)

# A target brighter than this, with no neighbour given an amplitude of its
# own, dominates its pixels - the pipelines measure such a star better
BRIGHT_TMAG = 13.5

# A target's own light below this many times the frame-to-frame scatter is
# not something a normalized light curve can show
MIN_FLUX_TO_SCATTER = 3.0

# The TESS zero point: 15000 e/s for a star of T = 10 (TESS Instrument
# Handbook), so a magnitude T gives 10**(-0.4*(T - TESS_ZP)) e/s
TESS_ZP = 10 + 2.5*np.log10(15000)

# Widths of the Gaussian profile tried against the median image, pixels
PSF_SIGMAS = np.linspace(0.45, 1.5, 22)

# BTJD is BJD - 2457000, MJD is JD - 2400000.5
BTJD_TO_MJD = 2457000 - 2400000.5


def tic_stars(ra, dec, radius, basepath, log, refresh=False):
    """The TIC stars around a position, as a Table with ID, ra, dec, Tmag."""
    from astroquery.mast import Catalogs

    cache_name = f"tess_tic_{ra:.4f}_{dec:.4f}_{radius*3600:.0f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'TIC',
                              refresh=refresh) as cache:
        if not cache.hit:
            res = Catalogs.query_region(SkyCoord(ra, dec, unit='deg'),
                                        radius=radius*u.deg, catalog='TIC')
            if res is not None and len(res):
                res = res[['ID', 'ra', 'dec', 'Tmag', 'objType']]
                res['ID'] = res['ID'].astype(str)
                res['objType'] = res['objType'].astype(str)
                cache.save(Table(res))
            else:
                cache.save_empty()

        tic = cache.data

    if tic is None or not len(tic):
        return None

    tic = Table(tic)
    keep = np.isfinite(np.ma.filled(np.asarray(tic['Tmag'], dtype=float), np.nan))
    # The TIC lists galaxies and artefacts beside the stars
    keep &= np.asarray(tic['objType']).astype(str) == 'STAR'

    return tic[keep]


def gaussian(xx, yy, x0, y0, sigma):
    """A circular Gaussian of unit integral, and its derivatives in x0, y0, sigma."""
    dx, dy = xx - x0, yy - y0
    r2 = dx**2 + dy**2
    g = np.exp(-0.5*r2/sigma**2)/(2*np.pi*sigma**2)

    return g, g*dx/sigma**2, g*dy/sigma**2, g*(r2/sigma**3 - 2/sigma)


def fit_psf_width(image, xs, ys, xx, yy, mask):
    """The Gaussian width that best describes the stars of the median image.

    Fluxes are free and kept positive, the background is free in sign; the
    stars are taken where the catalogue puts them.
    """
    from scipy.optimize import nnls

    best = None
    data = image[mask]

    for sigma in PSF_SIGMAS:
        cols = [gaussian(xx[mask], yy[mask], x, y, sigma)[0] for x, y in zip(xs, ys)]
        ones = np.ones(mask.sum())
        A = np.column_stack(cols + [ones, -ones])
        coef, _ = nnls(A, data, maxiter=10*A.shape[1])
        chi2 = np.sum((data - A @ coef)**2)

        if best is None or chi2 < best[0]:
            best = (chi2, sigma, coef)

    return best[1], best[2]


def refine_width(signal, dflux, X, Y, x0, y0, noise, log):
    """The profile width of whatever varies like the first light curve.

    Each pixel's difference images, with everything but the target's term
    taken out, regressed on that light curve give the share of the variation
    falling on that pixel; a Gaussian fitted to those shares has the width of
    the variable star's image, and its centre says where that star is.

    Returns the width, or None where the shares are too noisy to fit - a star
    that hardly varies - or the fit lands somewhere no profile could.
    """
    from scipy.optimize import least_squares

    dflux = dflux - np.median(dflux)
    power = np.sum(dflux**2)
    if not power > 0:
        return None

    share = signal.T @ dflux/power
    share_err = noise/np.sqrt(power)

    def resid(p):
        return (p[0]*gaussian(X, Y, p[1], p[2], p[3])[0] + p[4] - share)/share_err

    try:
        fit = least_squares(resid, [1.0, x0, y0, 0.7, 0.0],
                            bounds=([0, x0 - 1.5, y0 - 1.5, 0.3, -np.inf],
                                    [np.inf, x0 + 1.5, y0 + 1.5, 2.0, np.inf]))
        jac = fit.jac
        cov = np.linalg.pinv(jac.T @ jac)*max(np.sum(fit.fun**2)/max(len(X) - 5, 1), 1)
        err = np.sqrt(np.diag(cov))
    except Exception:
        return None

    snr = fit.x[0]/err[0] if err[0] > 0 else 0
    offset = np.hypot(fit.x[1] - x0, fit.x[2] - y0)*21.0

    if snr < 20 or not 0.35 < fit.x[3] < 1.9:
        log(f"    The variable light is too faint to measure its profile on (S/N {snr:.0f}): "
            f"the median image's width is kept")
        return None

    log(f"    The variable light sits {offset:.1f} arcsec from the target "
        f"(+- {np.hypot(err[1], err[2])*21.0:.1f}), with a profile {fit.x[3]:.2f} px wide - "
        f"the light curve is refitted with that width")
    if offset > 10.0:
        log("    Warning: that is further than a target's own light would be - "
            "the variability may belong to a neighbour")

    return fit.x[3]


def pixel_lightcurve(tpf, ra, dec, tic, level, log):
    """Measure the target's flux change in every frame of one sector's cutout.

    Parameters
    ----------
    tpf : `lightkurve.TessTargetPixelFile`
        The cutout, as TESScut gives it
    ra, dec : float
        The target, degrees
    tic : `astropy.table.Table` or None
        The TIC stars around it, from `tic_stars`
    level : str
        How hard to filter on the quality flags, for the reference image
    log : callable
        Logging function

    Returns
    -------
    table : `astropy.table.Table` or None
        mjd, btjd, flux, flux_err, quality and the fitted nuisance terms; None
        where the target is not on the cutout or nothing could be fitted
    """
    btjd = np.asarray(tpf.time.value, dtype=float)
    cube = np.asarray(tpf.flux.value, dtype=float)
    quality = np.asarray(tpf.quality, dtype=int)

    finite = np.isfinite(btjd) & np.all(np.isfinite(cube.reshape(len(cube), -1)), axis=1)
    if finite.sum() < 10:
        log("    Too few usable frames in the cutout")
        return None

    btjd, cube, quality = btjd[finite], cube[finite], quality[finite]
    ny, nx = cube.shape[1:]
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)

    wcs = tpf.wcs
    x0, y0 = (float(_) for _ in wcs.all_world2pix(ra, dec, 0))
    if not (FIT_RADIUS <= x0 <= nx - 1 - FIT_RADIUS and FIT_RADIUS <= y0 <= ny - 1 - FIT_RADIUS):
        log(f"    Target at pixel ({x0:.1f}, {y0:.1f}) is too close to the cutout edge")
        return None

    # Where the cutout reaches past the detector, TESScut fills it with values
    # about zero that never change - and a target just off the CCD still gets
    # its sector, as one of nothing but such pixels. Fitted, they give a
    # perfectly flat light curve, which is worse than none.
    live = ~((np.abs(np.median(cube, axis=0)) < 1) & (np.std(cube, axis=0) < 1))
    near_target = np.hypot(xx - x0, yy - y0) <= FIT_RADIUS
    if not np.all(live[near_target]):
        log(f"    The target is off the detector or at its edge in this sector: "
            f"{np.sum(~live[near_target])} of the {np.sum(near_target)} pixels around it "
            f"hold no data - skipped")
        return None

    # The catalogue stars on the cutout, and their fluxes from their magnitudes
    stars = []
    target_flux = None
    target_tmag = None
    if tic is not None and len(tic):
        sx, sy = wcs.all_world2pix(np.asarray(tic['ra'], float), np.asarray(tic['dec'], float), 0)
        tmag = np.asarray(tic['Tmag'], float)
        dist = np.hypot(sx - x0, sy - y0)*21.0

        on = (sx > -3) & (sx < nx + 2) & (sy > -3) & (sy < ny + 2) & (tmag < TIC_FAINT_LIMIT)
        nearest = np.argmin(dist)
        itarget = nearest if dist[nearest] < TIC_MATCH else None
        if itarget is not None:
            target_flux = 10**(-0.4*(tmag[itarget] - TESS_ZP))
            target_tmag = tmag[itarget]
            log(f"    Target is TIC {tic['ID'][itarget]}, Tmag {tmag[itarget]:.2f}, "
                f"{dist[itarget]:.1f} arcsec off: {target_flux:.0f} e/s expected")
        else:
            log(f"    No TIC entry within {TIC_MATCH:.1f} arcsec - the nearest, TIC {tic['ID'][nearest]} "
                f"(Tmag {tmag[nearest]:.2f}), is {dist[nearest]:.1f} arcsec away and is not taken "
                f"for the target")

        for i in np.where(on)[0]:
            if i != itarget:
                stars.append((sx[i], sy[i], 10**(-0.4*(tmag[i] - TESS_ZP)),
                              str(tic['ID'][i]), tmag[i]))

    # The reference: the median of the frames the flags allow
    keep = mission_quality_mask(quality, 'TESS', level)
    if keep.sum() < 10:
        keep = np.ones(len(btjd), bool)
    ref = np.median(cube[keep], axis=0)

    # The profile width, from the median image around the target
    local = (np.hypot(xx - x0, yy - y0) < FIT_RADIUS + 2) & live
    sigma, coef = fit_psf_width(ref, [s[0] for s in stars] + [x0], [s[1] for s in stars] + [y0],
                                xx, yy, local)
    log(f"    Profile width {sigma:.2f} px fitted to the median image")

    if target_flux is None:
        # No catalogue magnitude, and no mean to put the light curve on: what
        # the median image gives the target is the light of whatever faint
        # stars blend with it - for a quiescent dwarf nova of ~21 mag it gave
        # 15-60 e/s, against ~0.5 the star itself has - and put on that, any
        # change would read tens of times too small
        target_flux = 0.0
        log(f"    No mean flux for the target - its light in the median image "
            f"({coef[len(stars)]:.0f} e/s) is that of whatever blends with it - "
            f"so the light curve is the change of flux alone, in e/s")

    region = (np.hypot(xx - x0, yy - y0) <= FIT_RADIUS).ravel()
    X, Y = xx.ravel()[region], yy.ravel()[region]
    D = cube.reshape(len(cube), -1)[:, region] - ref.ravel()[region]

    # Each pixel weighted by its own scatter from frame to frame
    noise = 1.4826*np.median(np.abs(np.diff(D, axis=0)), axis=0)/np.sqrt(2)
    noise[noise <= 0] = np.median(noise[noise > 0]) if np.any(noise > 0) else 1.0
    w = 1/noise

    # The neighbours with amplitudes of their own
    near = []
    for s in sorted([s for s in stars if np.hypot(s[0] - x0, s[1] - y0) < NEIGHBOUR_RADIUS],
                    key=lambda s: -s[2]):
        r = np.hypot(s[0] - x0, s[1] - y0)
        if s[2] < NEIGHBOUR_MIN_RATIO*target_flux:
            continue
        if r < NEIGHBOUR_UNRESOLVED:
            log(f"    TIC {s[3]} (Tmag {s[4]:.2f}) is {r*21:.0f} arcsec from the target, "
                f"too close to tell apart: its variability is counted as the target's")
            continue
        if len(near) < NEIGHBOUR_MAX:
            near.append(s)
            log(f"    Neighbour TIC {s[3]}, Tmag {s[4]:.2f}, {r*21:.0f} arcsec away, fitted alongside")

    # A star that dominates its pixels is what the pipelines' apertures are
    # made for, and their photometry of it is still somewhat the better
    if target_tmag is not None and target_tmag < BRIGHT_TMAG and not near:
        log(f"    Note: at Tmag {target_tmag:.1f} the target dominates its pixels - the "
            f"pipelines' light curves (QLP, SPOC) will likely be somewhat more precise "
            f"than this fit")

    def design(sigma):
        """The frame model at a profile width: target, neighbours, then the nuisance terms."""
        cols = [gaussian(X, Y, x0, y0, sigma)[0]]
        cols += [gaussian(X, Y, s[0], s[1], sigma)[0] for s in near]

        # Pointing and focus, shared: every star's profile derivatives by its
        # flux. The gradients of the median image itself were tried instead,
        # as difference imaging does for better sampled images; TESS's are
        # too coarse for finite differences to describe a shift of a fraction
        # of a pixel, and they did worse in every test
        shift_x = np.zeros(len(X))
        shift_y = np.zeros(len(X))
        focus = np.zeros(len(X))
        for sx_, sy_, flux, _, _ in stars + [(x0, y0, target_flux, None, None)]:
            _, gx, gy, gs = gaussian(X, Y, sx_, sy_, sigma)
            shift_x += flux*gx
            shift_y += flux*gy
            focus += flux*gs

        # The scattered light, as a plane. Curvature terms were tried: across
        # the few pixels fitted they are too much like a profile, and took
        # amplitude from the target while adding noise
        u, v = X - x0, Y - y0
        cols += [shift_x, shift_y, focus, np.ones(len(X)), u, v]

        return np.column_stack(cols)

    def solve(sigma):
        A = design(sigma)
        Aw = A*w[:, None]
        sol, _, rank, _ = np.linalg.lstsq(Aw, (D*w).T, rcond=None)
        return A, Aw, sol, rank

    A, Aw, sol, rank = solve(sigma)

    # The width again, from where the variable light actually sits: each
    # pixel's change against the first light curve is the profile of what
    # varies, measured on that star alone rather than on the crowd of the
    # median image - and its centre is the check that the variability is the
    # target's. Taken only where that profile stands clear of the noise: for a
    # star that does not vary there is nothing to measure it on.
    sigma_var = refine_width(D - (A[:, 1:] @ sol[1:]).T, sol[0], X, Y, x0, y0, noise, log)
    if sigma_var is not None:
        sigma = sigma_var
        A, Aw, sol, rank = solve(sigma)

    # How the pixels really answer the pointing and the focus, learned from
    # the sector itself: each pixel's difference images regressed on the shift
    # and focus the Gaussian model found, frame by frame. The Gaussian answers
    # a shift a few percent differently from TESS's own profile, and that
    # difference went into the target - steps of 1.5% at the pointing jumps of
    # a constant Tmag 12 star, ramps over each orbit beside a bright neighbour.
    # The maps carry the real profile of every star at once, so the frames
    # refitted with them leave it out; a target varying on its own does not
    # move with the pointing, and its variability stays where it was.
    ishift = 1 + len(near)
    drivers = np.column_stack([sol[ishift], sol[ishift + 1], sol[ishift + 2], np.ones(len(sol[0]))])
    maps, *_ = np.linalg.lstsq(drivers[keep], D[keep], rcond=None)
    A = A.copy()
    A[:, ishift:ishift + 3] = maps[:3].T
    Aw = A*w[:, None]
    sol, _, rank, _ = np.linalg.lstsq(Aw, (D*w).T, rcond=None)

    if rank < A.shape[1]:
        log("    Warning: the frame model is degenerate - some of its terms cannot be told apart")

    resid = (D*w).T - Aw @ sol
    dof = max(A.shape[0] - A.shape[1], 1)
    chi2 = np.sum(resid**2, axis=0)/dof
    cov = np.linalg.pinv(Aw.T @ Aw)
    dflux = sol[0]
    dflux_err = np.sqrt(cov[0, 0]*np.maximum(chi2, 1))
    names = ['target'] + [s[3] for s in near]

    # About the median of the frames the flags allow, so that the flux below
    # is centred on what the magnitude says
    dflux = dflux - np.median(dflux[keep])

    nshift = len(names)
    background = sol[nshift + 3]

    # The frames the scattered light flooded, beyond what the plane takes up
    scatter = 1.4826*np.median(np.abs(np.diff(dflux[keep])))/np.sqrt(2)
    excess = background - np.median(background[keep])
    high = BACKGROUND_LEAKAGE*excess > scatter
    if np.any(high):
        quality = quality | np.where(high, QUALITY_HIGH_BACKGROUND, 0)
        log(f"    {np.sum(high)} frames flagged for a background more than "
            f"{scatter/BACKGROUND_LEAKAGE:.0f} e/s above its median, where its leaking "
            f"into the target would exceed the scatter (quality bit {QUALITY_HIGH_BACKGROUND})")

    table = Table({
        'mjd': btjd + BTJD_TO_MJD,
        'btjd': btjd,
        'flux': target_flux + dflux,
        'flux_err': dflux_err,
        'quality': quality,
        'flux_change': dflux,
        'shift_x': sol[nshift],
        'shift_y': sol[nshift + 1],
        'focus': sol[nshift + 2],
        'background': background,
    })
    table.meta['TARGET_FLUX'] = float(target_flux)
    table.meta['PSF_SIGMA'] = float(sigma)
    table.meta['NEIGHBOURS'] = ','.join(names[1:])

    table.meta['SCATTER'] = float(scatter)
    log(f"    {len(table)} frames; point-to-point scatter {scatter:.1f} e/s"
        + (f" ({100*scatter/target_flux:.1f}% of the target)" if target_flux > 0 else "")
        + f", full range {np.ptp(np.percentile(dflux[keep], [1, 99])):.0f} e/s")

    return table


def target_tess_pixel(config, basepath, log, show=False, refresh=False):
    """Pixel-level light curves of the target in every sector with full-frame images.

    Writes tess_lc_<sector>_PIXEL_<cadence>.vot / .txt / .png, the same way
    the pipelines' light curves are written, so the viewer reads them alike.
    """
    ra, dec = config['target_ra'], config['target_dec']
    level = quality_level(config, 'tess')

    log(f"Requesting TESS full-frame image cutouts for {config['target_name']}, "
        f"{CUTOUT_SIZE}x{CUTOUT_SIZE} pixels")

    res = cached_lightkurve_search(f"tess_tesscut_{ra:.4f}_{dec:.4f}.vot", basepath, log,
                                   'TESS cutout', refresh=refresh, kind='tesscut',
                                   target=SkyCoord(ra, dec, unit='deg'))
    if not len(res):
        log("Warning: No TESS full-frame images cover the target")
        return

    log(f"{len(res)} sectors with full-frame images")

    # The stars around it, once for all sectors: reaching the cutout's corners
    radius = (CUTOUT_SIZE/2 + 3)*np.sqrt(2)*21/3600
    tic = tic_stars(ra, dec, radius, basepath, log, refresh=refresh)
    if tic is None:
        log("Warning: no TIC stars found around the target - only it will be modelled")

    for row in res:
        mission = row.mission[0]
        log(f"\n  {mission}")

        try:
            tpf = row.download(cutout_size=CUTOUT_SIZE,
                               download_dir=mast_download_dir(basepath, 'tess'))
        except Exception as e:
            log(f"    Error: could not download the cutout: {e}")
            continue

        if tpf is None:
            log("    Error: no cutout returned")
            continue

        table = pixel_lightcurve(tpf, ra, dec, tic, level, log)
        if table is None:
            continue

        sector = int(tpf.sector)
        # Named by the nominal cadence, as the pipelines' files are: the
        # spacing measured comes out a second or two off it
        spacing = np.median(np.diff(table['btjd']))*86400
        cadence = min(FFI_CADENCES, key=lambda c: abs(c - spacing))

        keep = mission_quality_mask(table['quality'], 'TESS', level)
        log_quality(log, table['quality'], ~keep, 'TESS')

        table, keep, cadence = bin_sector(table, keep, cadence, bin_width(config, 'tess'), log)
        lcname = f"tess_lc_{sector}_PIXEL_{cadence:.0f}.png"

        # Whether the target's own light stands out of the noise, judged on
        # the light curve as written - averaging may be what lifts it out
        mean = table.meta['TARGET_FLUX']
        change = np.asarray(table['flux_change'], dtype=float)[keep]
        scatter = 1.4826*np.median(np.abs(np.diff(change)))/np.sqrt(2) if len(change) > 2 else np.inf
        normalized = mean > MIN_FLUX_TO_SCATTER*scatter

        with plots.figure_saver(os.path.join(basepath, lcname), figsize=(8, 4), show=show) as fig:
            ax = fig.add_subplot(1, 1, 1)
            if normalized:
                flux = np.array(table['flux'], dtype=float)
                flux /= np.median(flux[keep])
                ax.axhline(1, ls='--', color='gray', alpha=0.3)
                ax.set_ylabel('Flux, normalized at its mean')
            else:
                flux = np.array(table['flux_change'], dtype=float)
                ax.axhline(0, ls='--', color='gray', alpha=0.3)
                ax.set_ylabel('Change of flux, e/s')
            flux[~keep] = np.nan

            ax.plot(*break_at_gaps(np.array(table['btjd']), flux), drawstyle='steps', lw=1)

            ax.grid(alpha=0.2)
            ax.set_xlabel('Time - 2457000, BTJD days')
            ax.set_title(f"{config['target_name']} - TESS Sector {sector} - pixel-level - {cadence} s")

        votname = os.path.splitext(lcname)[0] + '.vot'
        txtname = os.path.splitext(lcname)[0] + '.txt'
        table.write(os.path.join(basepath, txtname), format='ascii.commented_header', overwrite=True)
        log(f"    Sector lightcurve written to file:{txtname}")

        # The viewer divides every TESS light curve by its median. Put on a
        # mean lost in the noise - or on none - that makes swings of hundreds
        # of percent out of the noise, so such a light curve stays out of it;
        # what it can still show is said instead
        if normalized:
            table.write(os.path.join(basepath, votname), format='votable', overwrite=True)
            log(f"    Sector lightcurve written to file:{votname}")
        else:
            floor = 5*scatter/np.sqrt(max(3600/cadence, 1))
            log(f"    Not offered to the light curve viewer: the target's own light, "
                f"{mean:.1f} e/s, is lost in the point-to-point scatter of {scatter:.1f} e/s. "
                f"A change of more than ~{floor:.1f} e/s over an hour - a source of "
                f"Tmag {TESS_ZP - 2.5*np.log10(floor):.1f} appearing - would still show "
                f"in the text file")
