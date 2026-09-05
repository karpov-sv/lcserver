"""Fitting a stellar photosphere to the catalogue photometry of a target.

What is fitted is one model atmosphere, reddened and put at a distance:

    interpolate a model grid at (Teff, log g, [Fe/H]), redden by Av, dilute by
    (R/d)^2, and compare with the observed flux band by band.

The grids began as astroARIADNE's - each is an HDF5 cube of per-filter fluxes on
a (log g, Teff, [Fe/H]) lattice, built by convolving model spectra through
ninety-odd passbands, and reproducing that would mean redoing several hundred
gigabytes of convolution. Nothing else of that package is used: the cubes are
read directly here, which is thirty lines, and the forward model above is
another twenty.

Three things are done deliberately, each of which the SED fitting we compared
against gets wrong in a way that showed up on our own targets:

  * The posterior is kept as rows. Averaging several grids by resampling each
    parameter separately - which is what astroARIADNE does - leaves every
    correlation exactly zero, and the temperature/extinction degeneracy that
    dominates a reddened hot star simply vanishes from the answer. Here the
    samples stay whole.

  * The parameters reported for drawing are one row of the posterior, not a
    vector of per-parameter peaks. On a curved degeneracy the marginal modes do
    not lie on the ridge, so a model drawn at them fits nothing - which is what
    puts every residual of a hot star on the same side of zero.

  * The temperature prior does not depend on which grid is loaded, so a grid's
    extent cannot leak into the answer, and no regime is excluded by a prior
    that stops short of it.

Two numbers come out beside the parameters and are worth as much:

  ``jitter``   the fractional model inadequacy the fit needed - how far the
               photometry is from anything a single photosphere explains. A
               few per cent is a clean SED; twenty per cent is a SED assembled
               out of catalogues that saw the star at different brightnesses.

  ``shrink``   how much narrower the temperature posterior is than its prior.
               Below about ninety per cent the answer is largely the prior
               rather than the data, and should be quoted as such.
"""

import glob
import os
import time

import numpy as np

import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy import stats

import extinction

from .filters import (FILTER_NAMES, get_filter, is_ab, pivot_aa, vega_zero_jy,
                      width_aa)
from .utils import SourceError


# Solar radius and parsec in cm, for the (R/d)^2 dilution
R_SUN = 6.957e10
PARSEC = 3.0856775814913673e18

# Speed of light in Angstrom/s, for f_lambda <-> f_nu
C_AA = 2.99792458e18

AB_ZERO_JY = 3631.0


class Grid:
    """One atmosphere grid, read straight out of its HDF5 file.

    Two layouts, because model grids come both ways.

    A *lattice* carries three axes and a dense (n_logg, n_teff, n_feh, n_filter)
    block of fluxes - every combination computed, which is what astroARIADNE's
    grids are.

    A *scattered* grid carries a parameter triple per model and a
    (n_model, n_filter) block. Hot-star grids are this shape and cannot be the
    other: a star of 56000 K at log g 2 does not exist to be computed, so half
    the rectangle is empty and filling it with nothing would refuse half the
    models that do exist. Interpolation is over the models themselves, and the
    support is the hull they span rather than a box.

    Either way, bands a grid does not reach are NaN; those are reported so a
    caller can decline to fit them rather than discover it as a likelihood of
    minus infinity.
    """

    def __init__(self, path, name=None):
        self.name = name or str(path)

        with h5py.File(path, 'r') as h:
            self.layout = str(h.attrs.get('layout', 'lattice'))
            # Almost always a surface gravity, and named for one throughout;
            # on PoWR's Wolf-Rayet grids it is the transformed radius, whose
            # models have one gravity per temperature and would collapse onto
            # a line if it were stored as one
            self.axis_logg = str(h.attrs.get('axis_logg', 'logg'))
            self.logg = np.asarray(h['logg'][:], dtype=float)
            self.teff = np.asarray(h['teff'][:], dtype=float)
            self.feh = np.asarray(h['feh'][:], dtype=float)
            cube = np.asarray(h['flux'][:], dtype=float)
            names = [b.decode() if isinstance(b, bytes) else str(b)
                     for b in h['filters'][:]]

        self.filters = names
        self.column = {b: i for i, b in enumerate(names)}

        if self.layout == 'scattered':
            axes = self._scattered(cube)
            finite = np.isfinite(cube).any(axis=0)
        else:
            axes = None
            self._interp = RegularGridInterpolator(
                (self.logg, self.teff, self.feh), cube,
                bounds_error=False, fill_value=np.nan)
            finite = np.isfinite(cube).any(axis=(0, 1, 2))

        self._varies = axes
        self.covers = frozenset(b for b, i in self.column.items() if finite[i])

    def _scattered(self, cube):
        """Interpolate over the models, in whichever parameters actually vary.

        A grid of one metallicity says nothing about metallicity, and asking an
        interpolator to work in a direction with one value in it is asking for
        an error rather than an answer. What does not vary is left out here and
        comes back as a prior of zero width, which is the honest reading of it.
        """
        from scipy.interpolate import interp1d
        from scipy.spatial import Delaunay

        varies = [n for n in ('logg', 'teff', 'feh')
                  if np.ptp(getattr(self, n)) > 0]
        points = np.column_stack([getattr(self, n) for n in varies])

        # Each axis onto the unit interval before triangulating. A temperature
        # runs to tens of thousands and a gravity to four, and a triangulation
        # of points twenty thousand times further apart in one direction than
        # the other is degenerate.
        self._offset = points.min(axis=0)
        self._span = np.where(np.ptp(points, axis=0) > 0,
                              np.ptp(points, axis=0), 1.0)

        if len(varies) >= 2:
            self._cube = cube
            scaled = (points - self._offset) / self._span
            self._mesh = Delaunay(scaled)
            self._compact = self._compactness(scaled)
            self._interp = None
        elif len(varies) == 1:
            order = np.argsort(points[:, 0])
            self._interp = interp1d(points[order, 0], cube[order], axis=0,
                                    bounds_error=False, fill_value=np.nan)
        else:
            raise SourceError(f'{self.name} has one model and nothing to vary')

        return varies

    # How many steps of the grid's own lattice a simplex may span before what
    # it crosses is taken to be a gap rather than a gap-free neighbourhood.
    # Two allows a simplex to step over a single missing model and refuses one
    # that crosses further.
    MESH_SPAN = 2.0

    def _compactness(self, points):
        """Which simplices join neighbours, and which bridge a gap.

        A hull is convex and a grid of models is not: the temperatures a star
        of a given gravity was computed at run out, and where they do the hull
        carries on. Interpolating there is not extrapolating by a node, it is
        answering where nothing was computed.

        What counts as far is measured against the grid's own step, taken from
        the values its axes actually have. Against the typical simplex instead
        it would mean nothing on a grid with few models, where the typical
        simplex already bridges: three of the ones here answer over their whole
        box that way, which is most of it invented.
        """
        # The diagonal of one cell of the lattice the models sit on, in the
        # same normalised axes the mesh was built in
        steps = []
        for column in range(points.shape[1]):
            values = np.unique(points[:, column])
            steps.append(np.median(np.diff(values)) if len(values) > 1 else 0.0)

        cell = float(np.linalg.norm(steps))
        if not cell > 0:
            return np.ones(len(self._mesh.simplices), dtype=bool)

        corners = self._mesh.points[self._mesh.simplices]
        edges = np.linalg.norm(corners[:, :, None, :] - corners[:, None, :, :],
                               axis=-1).max(axis=(1, 2))

        return edges <= self.MESH_SPAN * cell

    # How far outside a simplex a point may be and still be taken as inside,
    # in axes normalised to their own span. Models on the edge of the hull -
    # which for a hot-star grid is the whole main sequence, every model at the
    # highest gravity computed - sit exactly on a face of it, and asking for a
    # node stored as a 32-bit float in 64-bit arithmetic lands a hair outside.
    # A millionth of an axis is four hundredths of a kelvin here.
    MESH_TOLERANCE = 1e-6

    def _from_mesh(self, point):
        """Linear interpolation over the triangulated models."""
        n = len(point)
        where = self._mesh.find_simplex(point, tol=self.MESH_TOLERANCE)
        if where < 0 or not self._compact[where]:
            return np.full(self._cube.shape[1], np.nan)

        transform = self._mesh.transform[where]
        bary = transform[:n].dot(point - transform[n])
        weights = np.append(bary, 1 - bary.sum())

        return weights @ self._cube[self._mesh.simplices[where]]

    @property
    def limits(self):
        """The box the grid is defined over, as {parameter: (low, high)}.

        For a scattered grid this is the box around the models rather than the
        hull they fill; what falls in the box and outside the hull comes back
        as no flux, which the likelihood already knows what to do with.
        """
        return {'teff': (self.teff.min(), self.teff.max()),
                'logg': (self.logg.min(), self.logg.max()),
                'feh': (self.feh.min(), self.feh.max())}

    def flux(self, teff, logg, feh, columns):
        """Surface flux in the given filter columns, erg/s/cm2/um."""
        if self._varies is None:
            return self._interp(np.array([[logg, teff, feh]]))[0][columns]

        at = {'logg': logg, 'teff': teff, 'feh': feh}
        point = [at[n] for n in self._varies]

        if len(point) == 1:
            return np.atleast_2d(self._interp(point[0]))[0][columns]

        scaled = (np.array(point) - self._offset) / self._span
        return self._from_mesh(scaled)[columns]


class CompositeGrid:
    """Two cubes over the same models, answering as one.

    A grid's filters are in one file and its Gaia XP bins in another, because
    the bins were written later and from the spectra rather than from whatever
    the group published. Nothing else should have to know that. This presents
    the two as a single grid with one set of band names and one column
    numbering, so a likelihood, a residual table and a figure take an XP bin
    exactly as they take Johnson V, and none of them contains the word Gaia.

    Where the two disagree about what exists, nothing exists: the spectra of a
    grid are usually a subset of its models, so a fit that wanders where there
    are spectra but no cube - or the other way - gets no flux for the bands it
    cannot have, and the likelihood already knows what that means.
    """

    def __init__(self, base, extra):
        self.base, self.extra = base, extra
        self.name = base.name
        self.teff, self.logg, self.feh = base.teff, base.logg, base.feh
        self.axis_logg = base.axis_logg

        self.filters = list(base.filters) + [b for b in extra.filters
                                             if b not in base.column]
        self.column = {b: i for i, b in enumerate(self.filters)}
        self.covers = frozenset(base.covers | extra.covers)

        # For each of our columns, which grid answers for it and where in it
        self._where = [(0, base.column[b]) if b in base.column
                       else (1, extra.column[b]) for b in self.filters]

    @property
    def limits(self):
        """The box both are defined over, which is where both can answer."""
        one, other = self.base.limits, self.extra.limits

        return {p: (max(one[p][0], other[p][0]), min(one[p][1], other[p][1]))
                for p in one}

    def flux(self, teff, logg, feh, columns):
        """Surface flux in the given columns, from whichever cube holds them."""
        columns = np.atleast_1d(columns)
        out = np.full(len(columns), np.nan)

        for n, grid in enumerate((self.base, self.extra)):
            mine = np.array([self._where[c][0] == n for c in columns])
            if not mine.any():
                continue

            wanted = np.array([self._where[c][1] for c in columns[mine]])
            out[mine] = grid.flux(teff, logg, feh, wanted)

        return out


def attenuation(wave_um, av, law=extinction.fitzpatrick99, rv=3.1):
    """Extinction in magnitudes at each wavelength, for one Av."""
    return law(np.asarray(wave_um, dtype=float) * 1e4, av, rv)


def model_flux(theta, columns, ext_unit, grid):
    """Observed flux for (teff, logg, feh, dist, rad, Av), erg/s/cm2/um.

    ``ext_unit`` is the attenuation at Av = 1, which is linear in Av, so the
    extinction law is evaluated once per fit rather than once per likelihood
    call - it is the same curve every time and it is not cheap.
    """
    teff, logg, feh, dist, rad, av = theta[:6]

    flux = grid.flux(teff, logg, feh, columns)
    dilution = (rad * R_SUN / (dist * PARSEC)) ** 2

    return flux * dilution * 10 ** (-0.4 * av * ext_unit)


# ------------------------------------------------------------------- priors

def make_prior(spec, limits=None):
    """One parameter's prior as an inverse CDF, from a short specification.

    ('uniform', lo, hi), ('loguniform', lo, hi), ('normal', mu, sigma),
    ('truncnorm', mu, sigma, lo, hi), ('fixed', value), or ('grid',) for the
    extent of the grid axis itself - which is the default for the parameters
    the grid defines, since the honest statement about a temperature nothing
    else constrains is that it lies somewhere the models exist.
    """
    kind = spec[0]

    if kind == 'grid':
        lo, hi = limits
        return lambda u: lo + u * (hi - lo)
    if kind == 'uniform':
        lo, hi = spec[1:3]
        return lambda u: lo + u * (hi - lo)
    if kind == 'loguniform':
        lo, hi = np.log(spec[1]), np.log(spec[2])
        return lambda u: np.exp(lo + u * (hi - lo))
    if kind == 'normal':
        mu, sigma = spec[1:3]
        return lambda u: stats.norm.ppf(u, loc=mu, scale=sigma)
    if kind == 'truncnorm':
        mu, sigma, lo, hi = spec[1:5]
        a, b = (lo - mu) / sigma, (hi - mu) / sigma
        return lambda u: stats.truncnorm.ppf(u, a, b, loc=mu, scale=sigma)
    if kind == 'halfnormal':
        sigma = spec[1]
        return lambda u: stats.halfnorm.ppf(u, scale=sigma)
    if kind == 'fixed':
        value = spec[1]
        return lambda u: value

    raise ValueError(f'unknown prior {kind!r}')


PARAMETERS = ('teff', 'logg', 'feh', 'dist', 'rad', 'Av', 'jitter')


def default_priors(grid, distance=None, distance_err=None, av_max=1.0):
    """Priors that say what is actually known, and no more.

    Teff, log g and [Fe/H] get the grid's own extent: a grid is a statement
    about where models exist, and nothing outside it can be evaluated anyway.
    The radius is log-uniform, being a scale rather than a location. Av is
    uniform up to the line-of-sight total, which is what a dust map gives.
    """
    limits = grid.limits

    priors = {
        'teff': ('grid',),
        'logg': ('grid',),
        'feh': ('grid',),
        'rad': ('loguniform', 0.05, 100.0),
        'Av': ('uniform', 0.0, max(av_max, 1e-3)),
        # A fractional term, shared across bands: it stands for how well the
        # model describes the star, which is a property of the model and not
        # of how precisely any one catalogue measured its band. Per-band terms
        # would be one free parameter each and mostly unconstrained.
        'jitter': ('halfnormal', 0.05),
    }

    if distance:
        spread = distance_err or 0.1 * distance
        priors['dist'] = ('truncnorm', distance, 3 * spread,
                          max(1.0, distance - 10 * spread), distance + 10 * spread)
    else:
        priors['dist'] = ('loguniform', 1.0, 30000.0)

    return priors


# ---------------------------------------------------------------------- fit

# How often a sampler that is running says so, in seconds. The line goes into
# the run's own log, which is what a page watching a fit is reading, so it is
# sparse enough to be worth keeping afterwards: a fit of several minutes leaves
# a dozen lines saying how it got there, not a thousand saying it is alive.
PROGRESS_SECONDS = 10.0


def fit(bands, wave_um, flux, flux_err, grid, priors,
        nlive=500, dlogz=0.5, seed=None, verbose=True, progress=None):
    """Sample the posterior for one grid. Returns whole rows, never marginals.

    The result carries ``samples`` as (n, 7) - equal-weight draws from the
    joint posterior, so any function of the parameters can be evaluated
    per-row and summarised afterwards, which is the only way a ratio like R/d
    comes out right.
    """
    import dynesty
    from dynesty.utils import resample_equal

    missing = [b for b in bands if b not in grid.covers]
    if missing:
        raise ValueError(f'{grid.name} has no flux for {", ".join(missing)}')

    columns = np.array([grid.column[b] for b in bands])
    ext_unit = attenuation(wave_um, 1.0)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    limits = grid.limits
    transforms = [make_prior(priors[p], limits.get(p)) for p in PARAMETERS]

    def prior_transform(u):
        return np.array([t(v) for t, v in zip(transforms, u)])

    def log_likelihood(theta):
        model = model_flux(theta, columns, ext_unit, grid)
        if not np.all(np.isfinite(model)):
            return -1e300

        # The jitter is a fraction of the model, so it means the same thing in
        # a band measured to a millimagnitude and one measured to a tenth
        sigma2 = flux_err ** 2 + (theta[6] * model) ** 2
        return float(-0.5 * np.sum((flux - model) ** 2 / sigma2
                                   + np.log(2 * np.pi * sigma2)))

    # Nested sampling converges on the evidence rather than after a fixed
    # number of steps, so how long it will take is not known when it starts.
    # What is known at every iteration is how far the remaining evidence is
    # from the tolerance it is going to stop at, and that is what is reported.
    def announce(results, niter, ncall, *args, **kwargs):
        now = time.time()
        if now - announce.last < PROGRESS_SECONDS:
            return

        announce.last = now

        # The sampler's own record of how much evidence it thinks is still
        # out there, against the tolerance it will stop at. Above a million
        # it means nothing has been bracketed yet, which dynesty itself
        # prints as infinity.
        left = getattr(results, 'delta_logz', None)
        target = kwargs.get('dlogz')

        line = f"    {niter} iterations, {ncall} likelihood calls"
        if isinstance(left, float) and np.isfinite(left) and left < 1e6:
            line += f", dlogz {left:.2f} of {target or dlogz:.2f} to go"

        progress(line)

    announce.last = time.time()

    sampler = dynesty.NestedSampler(
        log_likelihood, prior_transform, len(PARAMETERS),
        nlive=nlive, bound='multi', sample='rwalk',
        rstate=np.random.default_rng(seed))
    sampler.run_nested(dlogz=dlogz, print_progress=bool(verbose or progress),
                       print_func=announce if progress else None)

    results = sampler.results
    weights = np.exp(results.logwt - results.logz[-1])
    samples = resample_equal(results.samples, weights / weights.sum())

    return {
        'grid': grid.name,
        'bands': list(bands),
        'wave_um': np.asarray(wave_um, dtype=float),
        'flux': flux,
        'flux_err': flux_err,
        'samples': samples,
        'logz': float(results.logz[-1]),
        'logz_err': float(results.logzerr[-1]),
        'columns': columns,
        'ext_unit': ext_unit,
    }


def model_average(runs):
    """Combine grids by evidence, drawing whole rows.

    The weights are the usual normalised evidences; what matters is that a
    draw takes a row from one grid's posterior rather than a value from each
    grid's marginal. Mixing marginals produces a cloud with no correlations at
    all, and a parameter set drawn from it is not one the data ever supported.
    """
    logz = np.array([r['logz'] for r in runs], dtype=float)
    weights = np.exp(logz - logz.max())
    weights /= weights.sum()

    counts = np.random.default_rng(0).multinomial(
        max(len(r['samples']) for r in runs), weights)

    rows = [r['samples'][np.random.default_rng(i).choice(len(r['samples']), n)]
            for i, (r, n) in enumerate(zip(runs, counts)) if n]

    return {
        'grid': ' + '.join(r['grid'] for r in runs),
        'bands': runs[0]['bands'],
        'wave_um': runs[0]['wave_um'],
        'flux': runs[0]['flux'],
        'flux_err': runs[0]['flux_err'],
        'samples': np.vstack(rows),
        'weights': dict(zip([r['grid'] for r in runs], weights)),
        'logz': dict(zip([r['grid'] for r in runs], logz)),
        'runs': runs,
    }


# ------------------------------------------------------------------ summary

def derived(samples):
    """Quantities that are functions of the row, evaluated on the row.

    The angular diameter is the point: it is what a spectral energy
    distribution actually measures, and median(R)/median(d) is not
    median(R/d) when the two are correlated - which for a degenerate fit is
    the difference between a model that fits and one that does not.
    """
    teff, rad, dist = samples[:, 0], samples[:, 4], samples[:, 3]

    return {
        'theta_mas': 2 * rad * R_SUN / (dist * PARSEC) * 206264806.2,
        'lum_lsun': rad ** 2 * (teff / 5772.0) ** 4,
    }


def summarise(result, grid=None):
    """Point estimates that are a place the posterior actually goes.

    ``best`` is the single highest-likelihood row: mutually consistent, so a
    model drawn at it is a model that fits, which is what it is for. It is not
    the answer - along a flat degeneracy it wanders from one end of the ridge
    to the other. The inference is the percentiles, which are the marginals.
    The two are different things and are not presented as if they were.
    """
    samples = result['samples']
    out = {'logz': result.get('logz'), 'grid': result.get('grid')}

    if grid is not None:
        best, chi2 = best_row(result, grid)
        out['best'] = dict(zip(PARAMETERS, best))
        out['chi2'] = chi2

        # The derived quantities of that same row, so the whole column
        # describes one place rather than a row for some parameters and a
        # blank for the rest
        for name, values in derived(best[None, :]).items():
            out['best'][name] = float(values[0])

    for i, name in enumerate(PARAMETERS):
        out[name] = quantiles(samples[:, i])

    for name, values in derived(samples).items():
        out[name] = quantiles(values)

    return out


# The quantiles reported for every parameter: the median, the central 68 per
# cent, and the central 99.7 per cent. The last is not decoration - a posterior
# running along a degeneracy is badly non-Gaussian, and the three-sigma
# interval is where that shows.
QUANTILES = (0.135, 16, 50, 84, 99.865)


def quantiles(values):
    """The spread of one parameter, as the numbers worth quoting."""
    lo3, lo, mid, hi, hi3 = np.percentile(np.asarray(values, dtype=float),
                                          QUANTILES)

    return {'median': float(mid), 'lo': float(lo), 'hi': float(hi),
            'lo3': float(lo3), 'hi3': float(hi3)}


def log_parameters(summary, log):
    """Write out every parameter the way it should be read.

    Two columns that are not the same thing and are not presented as if they
    were. The first is the row of the posterior the model is drawn at - one
    place the samples actually go, so the parameters in it are mutually
    consistent. The rest are the marginals, which is what an interval means.
    Along a flat degeneracy the first wanders from one end of the ridge to the
    other while the marginals do not move, and seeing both is the only way to
    notice that has happened.
    """
    best = summary.get('best') or {}

    # Kept inside 78 columns, which is what the log column of a target page is
    # asked to hold: wider and the table wraps every row onto two
    log(f"\n  {'parameter':<10}{'plot row':>11}{'median':>11}"
        f"{'-1 sigma':>11}{'+1 sigma':>11}{'3 sigma range':>22}")

    axis = summary.get('axis_logg') or 'logg'

    for name in PARAMETERS + ('theta_mas', 'lum_lsun'):
        row = summary.get(name)
        if not row:
            continue

        fmt = LABELS.get(name, (None, '{:.4g}'))[1]
        value = best.get(name)
        shown = axis if name == 'logg' else name

        log(f"  {shown:<10}"
            f"{(fmt.format(value) if value is not None else '-'):>11}"
            f"{fmt.format(row['median']):>11}"
            f"{fmt.format(row['median'] - row['lo']):>11}"
            f"{fmt.format(row['hi'] - row['median']):>11}"
            f"{fmt.format(row['lo3']) + ' ... ' + fmt.format(row['hi3']):>22}")


def best_row(result, grid):
    """The posterior row that fits best, and its chi-square."""
    columns = np.array([grid.column[b] for b in result['bands']])
    ext_unit = attenuation(result['wave_um'], 1.0)

    best, best_chi2 = None, np.inf
    for row in result['samples']:
        model = model_flux(row, columns, ext_unit, grid)
        if not np.all(np.isfinite(model)):
            continue
        chi2 = float(np.sum(((result['flux'] - model) / result['flux_err']) ** 2))
        if chi2 < best_chi2:
            best, best_chi2 = row, chi2

    return best, best_chi2


def residuals(result, theta, grid):
    """Observed minus model, in units of the catalogue error."""
    columns = np.array([grid.column[b] for b in result['bands']])
    model = model_flux(theta, columns, attenuation(result['wave_um'], 1.0), grid)

    return (result['flux'] - model) / result['flux_err'], model


# --------------------------------------------------------- infrared excess

# Where the excess is described from, and the shape conventions.
#
# The modified blackbody is the field-standard debris-disc form: a blackbody
# with an emission efficiency falling as 210 um / lambda beyond 210 um, so a
# temperature here is comparable with a temperature in the literature.
MODBB_LAMBDA0 = 210.0
MODBB_BETA = 1.0

# Sigma-Boltzmann in cgs, and the solar luminosity in erg/s, for turning a
# solid angle at a temperature into a fraction of the star
SIGMA_SB = 5.670374419e-5
L_SUN = 3.828e33

# The spectral index of the free-free hypothesis, as F_nu ~ nu^alpha. An
# ionised envelope runs from about -0.1 where it is optically thin, through
# the 0.6 of a constant-velocity wind, to the 2 of a body optically thick at
# every wavelength; the prior covers that and a little either side.
FREEFREE_ALPHA = (-0.5, 2.0)

# The dust temperatures a modified blackbody is allowed. Silicates do not
# survive above about 1500 K, and below 30 K a debris disc is colder than the
# interstellar radiation field would leave it.
DUST_TEMPERATURE = (30.0, 1500.0)

# How far the amplitude is allowed either side of the largest excess actually
# measured. It is the same prior for both shapes, so the Occam factor it
# carries cancels in the free-free against blackbody comparison and enters the
# comparison with the null once, identically for each.
AMPLITUDE_RANGE = (1e-4, 1e2)


def modbb_factor(wave_um):
    """The emission efficiency of the modified blackbody at each wavelength."""
    wave = np.asarray(wave_um, dtype=float)
    return np.where(wave > MODBB_LAMBDA0,
                    (MODBB_LAMBDA0 / wave) ** MODBB_BETA, 1.0)


def planck(wave_um, temperature):
    """B_lambda in erg/s/cm2/um/sr, which is the unit the fluxes are in."""
    from scipy.constants import h, c, k

    lam = np.asarray(wave_um, dtype=float) * 1e-6
    # A cold blackbody in the ultraviolet overflows the exponential, and the
    # zero that comes back out of it is the right answer
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        # W/m2/m/sr, then to erg/s/cm2/um/sr: 1e7 / 1e4 / 1e6
        value = (2 * h * c ** 2 / lam ** 5) \
            / np.expm1(h * c / (lam * k * temperature)) * 1e-3

    return np.where(np.isfinite(value), value, 0.0)


def modbb_bolometric_fraction(temperature):
    """How much of a blackbody survives the modification, as a fraction.

    The solid angle a fitted amplitude implies is turned into a luminosity by
    integrating the shape that was fitted, not the blackbody it started as.
    """
    lam = np.geomspace(1.0, 5000.0, 400)
    full = planck(lam, temperature)

    return float(np.trapz(full * modbb_factor(lam), lam) / np.trapz(full, lam))


def dust_ceiling(temperature):
    """The largest fractional luminosity a disc of that temperature is seen at.

    An empirical envelope rather than a physical bound: cold debris reaches
    1e-1 only after a giant impact, and sustained hot dust is observed at the
    1e-4 of an exozodi. Nothing is refused for exceeding it - the comparison
    here is about the shape of the excess - but a solution above it is saying
    something no debris disc has been seen to do, and that is worth printing.
    """
    t = np.clip(np.asarray(temperature, dtype=float), 500.0, 1500.0)
    return 10.0 ** (-1.0 - 3.0 * (np.log10(t) - np.log10(500.0))
                    / (np.log10(1500.0) - np.log10(500.0)))


def predict_bands(run, grid, bands, wave_um, ndraw=400, seed=0):
    """The photosphere in each band, as the posterior predicts it.

    The mean and spread over posterior rows, not the model at one row: a band
    the fit leaves loose has a wide prediction, and calling an excess
    significant means comparing with the width of what was predicted as well
    as with the error on what was measured.
    """
    columns = np.array([grid.column[b] for b in bands])
    ext_unit = attenuation(wave_um, 1.0)

    samples = run['samples']
    if len(samples) > ndraw:
        samples = samples[np.random.default_rng(seed).choice(len(samples),
                                                             ndraw, replace=False)]

    drawn = np.array([model_flux(row, columns, ext_unit, grid) for row in samples])
    good = np.isfinite(drawn).all(axis=1)
    if not good.any():
        return (np.full(len(bands), np.nan), np.full(len(bands), np.nan))

    drawn = drawn[good]
    return drawn.mean(axis=0), drawn.std(axis=0)


def excess_rows(rows, grids, chosen=None):
    """The points an excess is measured on.

    ``chosen`` names them; without it they are whatever ``default_excess`` made
    of the reading. Either way a band is only *quantified* where every grid in
    play both carries it and is believed there, which is the intersection of
    what they reach - the rest are carried through unquantified, so a point can
    be shown without a number being invented for it.
    """
    reach = _reach(rows, grids)

    wanted = (set(chosen) if chosen is not None
              else {r['id'] for r in rows if r.get('excess')})

    out = [dict(row, quantified=row['band'] in (reach or ()))
           for row in rows
           if row['id'] in wanted and row['fittable'] and not row['used']]

    out.sort(key=lambda r: r['wave_um'])
    return out


def _reach(rows, grids):
    """The bands every grid in play both carries and is believed at."""
    reach = None
    for grid in grids:
        covers = {b for b in grid.covers if _within_limit(grid.name, b, rows)}
        reach = covers if reach is None else (reach & covers)

    return reach or set()


def unused_rows(rows, grids, excess):
    """The points in neither set, one per band not already spoken for.

    Nothing is done with these - a point the reader left out of both says
    nothing about the star as far as the fit is concerned. They are read so
    they can be drawn, because how far a point sits from a photosphere fitted
    without it is the thing worth seeing before leaving it out for good.

    One per band, and only bands nothing else covers: a file that republishes
    the same measurement a dozen times would otherwise put a dozen markers on
    top of each other.
    """
    reach = _reach(rows, grids)
    seen = {r['band'] for r in rows if r['used']}
    seen |= {r['band'] for r in excess}

    out = []
    for row in rows:
        if (not row['fittable'] or row['used'] or row['band'] in seen):
            continue

        seen.add(row['band'])
        out.append(dict(row, quantified=row['band'] in reach))

    out.sort(key=lambda r: r['wave_um'])
    return out


def _within_limit(name, band, rows):
    """Whether a grid is believed at a band, by the wavelength it stops at."""
    entry = grid_registry().get(str(name).lower()) or {}
    limit = entry.get('reach_um')
    if limit is None:
        return True

    for row in rows:
        if row['band'] == band and row['wave_um']:
            return row['wave_um'] <= limit

    return True


def quantify_excess(rows, run, grid):
    """Observed against predicted, band by band, for the points beyond the fit.

    The significance is against both errors - the catalogue's on what was
    measured, and the posterior's on what was predicted - which is what makes
    a three-sigma excess at a band the fit barely constrains mean anything.
    """
    quantified = [r for r in rows if r['quantified']]
    table = []

    model = model_err = {}
    if quantified:
        bands = [r['band'] for r in quantified]
        waves = np.array([r['wave_um'] for r in quantified])
        predicted, spread = predict_bands(run, grid, bands, waves)
        model = dict(zip(bands, predicted))
        model_err = dict(zip(bands, spread))

    for row in rows:
        entry = {'id': row['id'], 'band': row['band'],
                 'wave_um': float(row['wave_um']),
                 'wave_drawn_um': float(row['wave_drawn_um']
                                        or row['wave_um']),
                 'observed': float(row['flux']), 'observed_err': float(row['err']),
                 'quantified': bool(row['quantified']),
                 'model': None, 'model_err': None,
                 'excess': None, 'ratio': None, 'sigma': None}

        m = model.get(row['band'])
        if entry['quantified'] and m is not None and np.isfinite(m) and m > 0:
            me = float(model_err[row['band']])
            entry['model'] = float(m)
            entry['model_err'] = me
            entry['excess'] = float(row['flux'] - m)
            entry['ratio'] = float(row['flux'] / m)
            entry['sigma'] = float((row['flux'] - m) / np.hypot(row['err'], me))
        else:
            entry['quantified'] = False

        table.append(entry)

    return table


def compare_excess(table, run, grid, summary):
    """Which shape the excess has, by the evidence for each.

    Three hypotheses, compared by integrating the likelihood over their priors
    on a grid. At two parameters that integral is exact to the resolution of
    the grid and carries no Monte Carlo error at all, which is worth more here
    than sampling would be - the whole comparison is a ratio of numbers that
    sampling would only add noise to.

      null        no excess, and no parameters to pay for it.

      free-free   a power law, F_nu ~ nu^alpha. The continuum of ionised gas -
                  a wind, or the decretion disc of a Be star - which has no
                  temperature to speak of and does not turn over anywhere in
                  the infrared.

      blackbody   a modified blackbody at one dust temperature, which is what
                  a debris disc is. It peaks, and where it peaks is the whole
                  of the information in it.

    Both shapes carry the same amplitude prior, so what the comparison between
    them tests is the shape and nothing else. The free-free law is defined
    only redward of the fit: a power law continued blueward diverges, and
    whatever the gas contributes among the fitted bands was absorbed into the
    photosphere when it was fitted. The blackbody is defined everywhere, and
    pays for what it predicts among the fitted bands - which is what keeps a
    1500 K solution from quietly out-shining the star in K.
    """
    used = [e for e in table if e['quantified']]
    if len(used) < 2:
        return None

    wave = np.array([e['wave_um'] for e in used])
    excess = np.array([e['excess'] for e in used])
    sigma = np.array([np.hypot(e['observed_err'], e['model_err']) for e in used])
    inv2 = 1.0 / sigma ** 2

    # The amplitude is the excess at this wavelength, which is put in the
    # middle of the measured ones so that it is interpolated rather than
    # extrapolated whatever the shape turns out to be
    reference = float(np.exp(np.mean(np.log(wave))))
    scale = float(np.max(np.abs(excess)))
    if not scale > 0:
        return None

    # What each shape would add to the bands the photosphere was fitted on
    fit_wave = np.asarray(run['wave_um'], dtype=float)
    fit_model = np.array([summary['best'][p] for p in PARAMETERS])
    fit_model = model_flux(fit_model, run['columns'], run['ext_unit'], grid)
    fit_sigma = np.hypot(run['flux_err'], summary['jitter']['median'] * fit_model)
    fit_inv2 = 1.0 / fit_sigma ** 2

    amplitude = np.geomspace(AMPLITUDE_RANGE[0] * scale,
                             AMPLITUDE_RANGE[1] * scale, 1024)

    def evidence(at_bands, at_fit):
        """Integrate exp(-chi2/2) over the amplitude and the shape.

        chi2 is quadratic in the amplitude, so the sums over bands are done
        once per shape and the whole amplitude axis follows from three numbers.
        """
        s0 = float(np.sum(excess ** 2 * inv2))
        s1 = at_bands @ (excess * inv2)
        s2 = (at_bands ** 2) @ inv2 + (at_fit ** 2) @ fit_inv2

        chi2 = (s0 - 2 * amplitude[None, :] * s1[:, None]
                + amplitude[None, :] ** 2 * s2[:, None])
        return chi2, s0

    # free-free: F_lambda ~ lambda^-(alpha + 2), and nothing blueward of the fit
    alpha = np.linspace(*FREEFREE_ALPHA, 61)
    ff_bands = (wave[None, :] / reference) ** -(alpha[:, None] + 2)
    ff_fit = np.zeros((alpha.size, fit_wave.size))
    chi2_ff, chi2_null = evidence(ff_bands, ff_fit)

    # blackbody: a solid angle set by the amplitude at the reference wavelength
    temperature = np.geomspace(*DUST_TEMPERATURE, 64)
    unit = np.array([planck(wave, t) * modbb_factor(wave) for t in temperature])
    unit_ref = np.array([float(planck(reference, t)) * float(modbb_factor(reference))
                         for t in temperature])
    bb_bands = unit / unit_ref[:, None]
    bb_fit = np.array([planck(fit_wave, t) * modbb_factor(fit_wave) / u
                       for t, u in zip(temperature, unit_ref)])
    chi2_bb, _ = evidence(bb_bands, bb_fit)

    # One offset for all three, so the ratios are the ratios
    floor = min(chi2_null, float(np.min(chi2_ff)), float(np.min(chi2_bb)))
    like_ff = np.exp(-(chi2_ff - floor) / 2)
    like_bb = np.exp(-(chi2_bb - floor) / 2)

    # The priors are uniform over each axis as it is gridded - alpha linearly,
    # the temperature and the amplitude in the log - so the integral over the
    # prior is the mean over the nodes
    z = {'none': float(np.exp(-(chi2_null - floor) / 2)),
         'freefree': float(np.mean(like_ff)),
         'blackbody': float(np.mean(like_bb))}
    total = sum(z.values()) or 1.0

    # The ratios themselves as well as the probabilities: a shape that wins by
    # forty in the log has a probability of one to every digit that fits, and
    # the number that says how decisively is the ratio
    def ln_ratio(a, b):
        return float(np.log(z[a]) - np.log(z[b])) if z[a] > 0 and z[b] > 0 else None

    out = {'reference_um': reference,
           'chi2_null': float(chi2_null),
           'bands': len(used),
           'probability': {k: v / total for k, v in z.items()},
           'ln_bayes': {'freefree_over_blackbody': ln_ratio('freefree', 'blackbody'),
                        'freefree_over_none': ln_ratio('freefree', 'none'),
                        'blackbody_over_none': ln_ratio('blackbody', 'none')}}

    out['freefree'] = _marginal(like_ff, alpha, amplitude,
                                ('alpha', 'amplitude'), chi2_ff)
    out['blackbody'] = _marginal(like_bb, temperature, amplitude,
                                 ('t_dust', 'amplitude'), chi2_bb)

    # What the fitted blackbody would be, as a fraction of the star
    best = out['blackbody']
    dust = _dust_luminosity(best['t_dust'], best['amplitude'], reference,
                            summary)
    if dust is not None:
        best['l_dust_over_lstar'] = dust
        best['above_ceiling'] = bool(dust > dust_ceiling(best['t_dust']))

    out['preferred'] = max(out['probability'], key=out['probability'].get)
    return out


def _marginal(like, shape, amplitude, names, chi2):
    """Mean and spread of each axis under the likelihood, and the best node."""
    weight = like / (like.sum() or 1.0)
    ws, wa = weight.sum(axis=1), weight.sum(axis=0)

    def moments(values, w):
        mean = float(np.sum(w * values))
        var = float(np.sum(w * (values - mean) ** 2))
        return mean, float(np.sqrt(max(var, 0.0)))

    i, j = np.unravel_index(np.argmin(chi2), chi2.shape)
    s_mean, s_err = moments(shape, ws)
    a_mean, a_err = moments(amplitude, wa)

    return {names[0]: s_mean, names[0] + '_err': s_err,
            names[1]: a_mean, names[1] + '_err': a_err,
            names[0] + '_best': float(shape[i]),
            names[1] + '_best': float(amplitude[j]),
            'chi2': float(chi2[i, j])}


def _dust_luminosity(temperature, amplitude, reference_um, summary):
    """L_dust / L_star for a modified blackbody of that amplitude.

    The amplitude is a flux at one wavelength; the solid angle follows from
    the shape, and the luminosity from the solid angle and the distance.
    """
    distance = summary.get('dist', {}).get('median')
    luminosity = summary.get('lum_lsun', {}).get('median')
    if not distance or not luminosity or luminosity <= 0:
        return None

    unit = float(planck(reference_um, temperature)) * float(modbb_factor(reference_um))
    if not unit > 0:
        return None

    omega = amplitude / unit
    d_cm = distance * PARSEC

    return float(4 * d_cm ** 2 * omega * SIGMA_SB * temperature ** 4
                 * modbb_bolometric_fraction(temperature)
                 / (luminosity * L_SUN))


def excess_flux(models, wave_um):
    """The fitted excess at each wavelength, in erg/s/cm2/um, or None.

    Evaluated at the best node rather than at the marginal means, for the
    reason the photosphere is drawn at one posterior row: a temperature and an
    amplitude taken from two different places on the likelihood are not a shape
    that fits anything.
    """
    if not models or models.get('preferred') == 'none':
        return None

    reference = models['reference_um']
    wave = np.asarray(wave_um, dtype=float)

    if models['preferred'] == 'freefree':
        best = models['freefree']
        flux = best['amplitude_best'] * (wave / reference) ** -(best['alpha_best'] + 2)
    else:
        best = models['blackbody']
        t = best['t_dust_best']
        unit = float(planck(reference, t)) * float(modbb_factor(reference))
        if not unit > 0:
            return None
        flux = best['amplitude_best'] * planck(wave, t) * modbb_factor(wave) / unit

    return flux


def add_excess_residuals(table, models):
    """What is left of the excess once the fitted shape is taken off it.

    The significance already on each row says how far the point is from the
    photosphere, which is whether there is an excess at all. This says how far
    it is from the photosphere and the excess together, which is whether the
    shape that won describes it - a different question, and the one that says
    which band a shape is missing.
    """
    used = [e for e in table if e['quantified']]
    if not used or not models:
        return

    flux = excess_flux(models, np.array([e['wave_um'] for e in used]))
    if flux is None:
        return

    for entry, model in zip(used, flux):
        total = entry['model'] + float(model)
        sigma = np.hypot(entry['observed_err'], entry['model_err'])
        entry['fit_model'] = total
        entry['fit_sigma'] = float((entry['observed'] - total) / sigma)


def log_excess(table, models, log):
    """The excess as a table, and what shape it came out as."""
    log('\n  infrared excess, against the photosphere the fit predicts')
    # Two significances, and they answer different questions: how far the
    # point is from the photosphere, which is whether there is an excess at
    # all, and how far it is from the photosphere and the fitted excess
    # together, which is whether the shape that won describes it.
    log(f"    {'band':<16}{'lam um':>8}{'observed':>11}{'photosphere':>12}"
        f"{'ratio':>8}{'sigma':>8}{'from fit':>10}")

    for entry in table:
        if not entry['quantified']:
            log(f"    {entry['band'] or '':<16}{entry['wave_um']:8.2f}"
                f"{entry['observed']:11.3e}{'-':>12}{'-':>8}{'-':>8}{'-':>10}")
            continue

        left = entry.get('fit_sigma')
        log(f"    {entry['band'] or '':<16}{entry['wave_um']:8.2f}"
            f"{entry['observed']:11.3e}{entry['model']:12.3e}"
            f"{entry['ratio']:8.2f}{entry['sigma']:+8.1f}"
            + (f"{left:+10.1f}" if left is not None else f"{'-':>10}"))

    if not models:
        log('    fewer than two bands to compare shapes on')
        return

    p = models['probability']
    log(f"\n    shape, on {models['bands']} band(s):  none {p['none']:.3f}"
        f"  free-free {p['freefree']:.3f}  blackbody {p['blackbody']:.3f}")

    ln = models['ln_bayes']['freefree_over_blackbody']
    if ln is not None:
        log(f"      log Bayes factor, free-free over blackbody {ln:+.1f}")

    # Which shape wins is one question, and whether it fits is another. Two
    # parameters through four catalogue points measured years apart will not
    # go through them all, and saying so is worth more than the ratio alone.
    best = models[models['preferred']] if models['preferred'] != 'none' else None
    if best is not None and models['bands'] > 2:
        log(f"      the preferred shape leaves chi2 {best['chi2']:.1f}"
            f" on {models['bands'] - 2} degree(s) of freedom")

    ff = models['freefree']
    log(f"      free-free   F_nu ~ nu^({ff['alpha']:+.2f} +/- {ff['alpha_err']:.2f}),"
        f"  {ff['amplitude']:.3e} at {models['reference_um']:.2f} um")

    bb = models['blackbody']
    line = (f"      blackbody   T_dust {bb['t_dust']:.0f}"
            f" +/- {bb['t_dust_err']:.0f} K")
    if 'l_dust_over_lstar' in bb:
        line += f",  L_dust/L_star {bb['l_dust_over_lstar']:.2e}"
    log(line)

    if bb.get('above_ceiling'):
        log('        - which is more than any debris disc is seen to have'
            ' at that temperature')


# --------------------------------------------------- the Gaia XP spectrum

# What the info step leaves behind where Gaia published one: the sampled BP/RP
# spectrum, 343 points from 336 to 1020 nm in erg/s/cm2/A. It is not fitted
# here. It is a second measurement of the same star over the same wavelengths
# the photometry covers, taken as one epoch-average and calibrated by somebody
# else, and what it is worth is the answer to whether it agrees with the model
# the photometry produced. That is a question to ask before letting it into a
# likelihood, not after.
XP_FILE = 'gaia_xp.vot'

# Why it is binned to be compared.
#
# The 343 samples are a resampling of 55 + 55 basis coefficients through a line
# spread function of order ten to fifteen nanometres, so neighbouring points
# are largely the same measurement said again and the published errors are per
# point rather than per resolution element. Binned to twice the width of that
# function, what comes out is insensitive to its shape - a boxcar wider than
# the kernel integrates the same flux whatever the kernel is - and the twenty
# or so bins that result are no more numbers than the spectrum ever held.
XP_BIN_NM = 30.0
XP_LSF_NM = 15.0

# The ends are trimmed rather than compared. The flux calibration of the
# sampled spectra is at its worst below 400 and above 950 nm, and outside the
# range the mission publishes the reconstruction has no basis functions left.
XP_RANGE_NM = (350.0, 980.0)

# The calibration systematic, added to every bin. Without it the comparison is
# against the photon noise of a spectrum whose absolute scale is known to a
# couple of per cent, and every bin of a bright star would come out tens of
# sigma from a model that is in fact right.
XP_SYSTEMATIC = 0.02


def xp_bands():
    """The pseudo-passbands the spectrum is binned into, blue to red.

    Fixed, and fixed for good: they are convolved into every grid's file, and a
    cube built on one set of bins cannot be read with another. Changing them
    means rebuilding every grid, which is what a filter set is like everywhere
    else here.

    Named as the grids name their filters, so that a bin is a band and needs no
    special case anywhere downstream - a point list, a residual table and a
    likelihood all take one exactly as they take Johnson V.
    """
    edges = np.arange(XP_RANGE_NM[0], XP_RANGE_NM[1] + 1e-6, XP_BIN_NM)

    return [{'band': f'GAIA_XP_{int(round(0.5 * (a + b)))}',
             'lo_nm': float(a), 'hi_nm': float(b),
             'wave_um': float(0.5 * (a + b) * 1e-3)}
            for a, b in zip(edges[:-1], edges[1:])]


def is_xp(band):
    """Whether a band is one of those bins rather than a real filter."""
    return str(band or '').startswith('GAIA_XP_')


def read_xp(basepath):
    """The Gaia XP spectrum, on the units the rest of this module works in.

    Written per Angstrom by the info step, as every spectrum on this site is,
    and read per micron, as every flux here is.
    """
    from astropy.table import Table

    path = os.path.join(basepath, XP_FILE)
    if not os.path.exists(path):
        return None

    table = Table.read(path)
    if 'wavelength' not in table.colnames or 'flux' not in table.colnames:
        return None

    wave = np.asarray(table['wavelength'], dtype=float)
    flux = np.asarray(table['flux'], dtype=float)
    err = (np.asarray(table['flux_error'], dtype=float)
           if 'flux_error' in table.colnames else np.full(len(wave), np.nan))

    order = np.argsort(wave)

    return {'wave_um': wave[order] * 1e-4,
            'flux': flux[order] / PER_AA,
            'err': err[order] / PER_AA}


def xp_binned(xp):
    """The spectrum in those bins, with what each one is worth.

    Two terms in the error and they answer different questions. Within one
    resolution element the samples are one measurement said several times, so
    the noise on a bin is the mean of theirs over the root of how many
    resolution elements it holds - not of how many samples, which would claim
    a precision the reconstruction never had. And the absolute scale of these
    spectra is known to a couple of per cent, which no amount of binning
    improves and which is what a bright star's bins are limited by.
    """
    wave_nm = np.asarray(xp['wave_um'], dtype=float) * 1e3
    flux = np.asarray(xp['flux'], dtype=float)
    err = np.asarray(xp['err'], dtype=float)

    out = []
    for band in xp_bands():
        inside = ((wave_nm >= band['lo_nm']) & (wave_nm < band['hi_nm'])
                  & np.isfinite(flux))
        if inside.sum() < 3:
            continue

        observed = float(np.mean(flux[inside]))
        if not np.isfinite(observed) or observed <= 0:
            continue

        elements = max(1.0, (band['hi_nm'] - band['lo_nm']) / XP_LSF_NM)
        stat = float(np.mean(err[inside])) / np.sqrt(elements)
        if not np.isfinite(stat):
            stat = 0.0

        out.append(dict(band, samples=int(inside.sum()), observed=observed,
                        stat_err=stat,
                        err=float(np.hypot(stat, XP_SYSTEMATIC * observed))))

    return out


def rebin(wave, flux, edges):
    """The mean of a spectrum over each interval, by integrating it.

    Sampling a model at the middle of an interval is not what a spectrograph
    does to it. These grids run at a resolving power of thousands where Gaia's
    spectra run at tens, so a line that would be half the depth of a bin can
    fall between two samples of it and vanish; integrated, it counts for what
    it is. The antiderivative is evaluated at the interval edges, which is
    exact wherever an edge falls on a point of the model and close between.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    integral = np.concatenate([[0.0], np.cumsum(np.diff(wave)
                                                * 0.5 * (flux[1:] + flux[:-1]))])

    return np.diff(np.interp(edges, wave, integral)) / np.diff(edges)


def at_xp_resolution(wave_um, flux, step_nm=1.0, lsf_nm=XP_LSF_NM):
    """A model spectrum reduced to what Gaia's spectra are able to say.

    Integrated onto a fine uniform axis and convolved with the instrument's
    width, so that what is then binned has been through the same two things
    the data have been through. Over most of a thirty-nanometre bin this
    changes nothing - which is the point of binning that wide - and where it
    matters is the one place it should, at a bin edge that falls on a Balmer
    jump, where the spectrograph puts flux across the edge and a model that
    had not been smeared would not.
    """
    lo, hi = XP_RANGE_NM[0] - 4 * lsf_nm, XP_RANGE_NM[1] + 4 * lsf_nm
    edges = np.arange(lo, hi + step_nm, step_nm)
    axis = 0.5 * (edges[:-1] + edges[1:])

    wave_nm = np.asarray(wave_um, dtype=float) * 1e3
    if wave_nm[0] > lo or wave_nm[-1] < hi:
        return None, None

    fine = rebin(wave_nm, flux, edges)

    sigma = lsf_nm / 2.3548 / step_nm
    half = int(np.ceil(4 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)

    return axis, np.convolve(fine, kernel / kernel.sum(), mode='same')


def compare_xp(xp, run, grid, theta, ndraw=400, seed=0):
    """The XP spectrum against what the fit predicts in the same bins.

    Nothing stands in for the fit here. The bins are columns of a cube over
    the grid's own models, so the prediction is that cube interpolated at the
    fitted parameters - the same operation, in the same function, that
    produces the model for Johnson V - and the posterior's width in a bin is
    the spread of that over posterior rows rather than an interpolation of
    what it was in the neighbouring filters.

    Which means this compares two things and only two: what Gaia measured, and
    what the fit says. It is a comparison and not part of the fit unless the
    bins were fitted, and it is reported the same way either way.

    Nothing is scaled to anything. Both are absolute, so an overall offset
    between them is one of the things worth finding out, and it is reported
    beside the chi2 - a fit three per cent low everywhere and a fit at the
    wrong temperature are different problems with the same chi2.
    """
    bands = [b['band'] for b in xp_bands() if b['band'] in grid.covers]
    if not bands:
        raise SourceError(f'{grid.name} has no Gaia XP bins - '
                          f'"manage.py sedgrid --xp" writes them')

    binned = [b for b in xp_binned(xp) if b['band'] in grid.covers]
    if not binned:
        raise SourceError('no usable bins in the XP spectrum')

    wave = np.array([b['wave_um'] for b in binned])
    columns = np.array([grid.column[b['band']] for b in binned])
    ext_unit = attenuation(wave, 1.0)

    model = model_flux(theta, columns, ext_unit, grid)

    # What the rest of the posterior would have predicted here, which is what
    # makes a deviation in a bin the fit barely constrains worth what it is
    samples = run['samples']
    if len(samples) > ndraw:
        samples = samples[np.random.default_rng(seed).choice(len(samples),
                                                             ndraw, replace=False)]
    cloud = np.array([model_flux(row, columns, ext_unit, grid) for row in samples])
    good = np.isfinite(cloud).all(axis=1)
    model_err = (cloud[good].std(axis=0) if good.any()
                 else np.zeros(len(binned)))

    # Where the fit had points and where it did not. A bin redward of the
    # reddest band fitted is not a test of the fit, it is a test of what the
    # model does past the data, and reporting the two together makes a model
    # extrapolating badly look like a model that does not fit.
    fitted = [w for w, b in zip(run['wave_um'], run['bands']) if not is_xp(b)]
    span = (float(np.min(fitted)), float(np.max(fitted))) if fitted else None
    inside = set(run['bands'])

    bins = []
    for entry, m, me in zip(binned, model, model_err):
        if not np.isfinite(m) or m <= 0:
            bins.append(dict(entry, model=None, model_err=None, ratio=None,
                             sigma=None, constrained=False, fitted=False))
            continue

        total = float(np.sqrt(entry['err'] ** 2 + me ** 2))
        bins.append(dict(
            entry, model=float(m), model_err=float(me),
            ratio=float(entry['observed'] / m),
            sigma=float((entry['observed'] - m) / total),
            total_err=total,
            fitted=entry['band'] in inside,
            constrained=bool(entry['band'] in inside
                             or span is None
                             or span[0] <= entry['wave_um'] <= span[1])))

    usable = [b for b in bins if b['ratio'] is not None]
    if not usable:
        raise SourceError(f'{grid.name} predicts no Gaia XP bin at this fit')

    ratio = np.array([b['ratio'] for b in usable])
    sigma = np.array([b['sigma'] for b in usable])
    offset = float(np.median(ratio))

    # What is left once the offset is taken out, which the absolute chi2 does
    # not answer: whether the shape is right
    shape = np.array([(b['observed'] - offset * b['model']) / b['total_err']
                      for b in usable])

    within = [b for b in usable if b['constrained']]
    beyond = [b for b in usable if not b['constrained']]
    worst = usable[int(np.argmax(np.abs(sigma)))]

    return {
        'bins': bins,
        'n': len(usable),
        'n_fitted': sum(1 for b in usable if b['fitted']),
        'chi2': float(np.sum(sigma ** 2)),
        'offset': offset,
        'offset_within': (float(np.median([b['ratio'] for b in within]))
                          if within else None),
        'n_within': len(within),
        'offset_beyond': (float(np.median([b['ratio'] for b in beyond]))
                          if beyond else None),
        'n_beyond': len(beyond),
        'scatter': float(1.4826 * np.median(np.abs(ratio - offset))),
        'chi2_shape': float(np.sum(shape ** 2)),
        'worst': {'wave_um': worst['wave_um'], 'sigma': worst['sigma']},
        'spread': float(np.median([b['model_err'] / b['model'] for b in usable])),
        'systematic': XP_SYSTEMATIC,
        'bin_nm': XP_BIN_NM,
    }


def log_xp(result, log):
    """The comparison as a table, and what the numbers in it mean."""
    fitted = result['n_fitted']
    log(f"\n  Gaia XP spectrum in {result['bin_nm']:.0f} nm bins, against what"
        f" the fit predicts in them")
    log(f"    (errors are the spectrum's own, {result['systematic']:.0%} of"
        f" calibration, and how loosely the posterior predicts the bin -"
        f" {100 * result['spread']:.1f}% of it in the median)")

    log(f"    {'lam nm':>8}{'observed':>12}{'model':>12}{'ratio':>8}{'sigma':>8}"
        f"{'fitted':>9}")
    for b in result['bins']:
        log(f"    {b['wave_um'] * 1e3:8.0f}{b['observed'] * PER_AA:12.3e}"
            + (f"{b['model'] * PER_AA:12.3e}{b['ratio']:8.2f}{b['sigma']:+8.1f}"
               if b['ratio'] else f"{'-':>12}{'-':>8}{'-':>8}")
            + f"{'yes' if b['fitted'] else '-':>9}")

    log(f"\n    chi2 {result['chi2']:.1f} on {result['n']} bins;"
        f" the spectrum sits {100 * (result['offset'] - 1):+.1f}% on the model,"
        f" scatter {100 * result['scatter']:.1f}%")

    if result.get('offset_within') is not None and result['n_beyond']:
        log(f"      {100 * (result['offset_within'] - 1):+.1f}% over the"
            f" {result['n_within']} bins the fitted bands span, and"
            f" {100 * (result['offset_beyond'] - 1):+.1f}% over the"
            f" {result['n_beyond']} beyond them - which is the model past its"
            f" data rather than the model against it")

    log(f"    with that offset taken out, chi2 {result['chi2_shape']:.1f}"
        f" - which is the shape, and is what a fit would be working to")
    log(f"    worst bin {result['worst']['wave_um'] * 1e3:.0f} nm at"
        f" {result['worst']['sigma']:+.1f} sigma")

    # Which of the two things this is has to be said, since a chi2 in a log
    # invites the assumption that it was minimised
    if fitted:
        log(f"    {fitted} of these bins were fitted, so this is partly the"
            f" fit describing what it was shown")
    else:
        log('    the spectrum was not fitted - this is a check on the fit,'
            ' not part of it')


# ------------------------------------------------------------ our photometry

# VizieR's designation for a band, as our SED step writes it in the second
# field of the comment, against the name the model grids know it by. Only what
# the SED step curates is listed, plus the few extra designations that turn up
# in sed_all.vot, so an unmapped band is a real gap rather than an oversight.
FILTER_MAP = {
    'GALEX:FUV': 'GALEX_FUV',
    'GALEX:NUV': 'GALEX_NUV',

    'SDSS:u': 'SDSS_u',
    'SDSS:g': 'SDSS_g',
    'SDSS:r': 'SDSS_r',
    'SDSS:i': 'SDSS_i',
    'SDSS:z': 'SDSS_z',

    'PAN-STARRS/PS1:g': 'PS1_g',
    'PAN-STARRS/PS1:r': 'PS1_r',
    'PAN-STARRS/PS1:i': 'PS1_i',
    'PAN-STARRS/PS1:z': 'PS1_z',
    'PAN-STARRS/PS1:y': 'PS1_y',

    'Johnson:U': 'GROUND_JOHNSON_U',
    'Johnson:B': 'GROUND_JOHNSON_B',
    'Johnson:V': 'GROUND_JOHNSON_V',
    'Cousins:U': 'GROUND_JOHNSON_U',
    'Cousins:B': 'GROUND_JOHNSON_B',
    'Cousins:V': 'GROUND_JOHNSON_V',
    'Cousins:R': 'GROUND_COUSINS_R',
    'Cousins:I': 'GROUND_COUSINS_I',

    # 2MASS publishes J, H, Ks; VizieR labels them as the Johnson system
    'Johnson:J': '2MASS_J',
    'Johnson:H': '2MASS_H',
    'Johnson:K': '2MASS_Ks',
    '2MASS:J': '2MASS_J',
    '2MASS:H': '2MASS_H',
    '2MASS:Ks': '2MASS_Ks',

    'WISE:W1': 'WISE_RSR_W1',
    'WISE:W2': 'WISE_RSR_W2',
    'WISE:W3': 'WISE_RSR_W3',
    'WISE:W4': 'WISE_RSR_W4',

    'GAIA/GAIA3:G': 'GaiaDR2v2_G',
    'GAIA/GAIA3:Gbp': 'GaiaDR2v2_BP',
    'GAIA/GAIA3:Grp': 'GaiaDR2v2_RP',
    'GAIA/GAIA2:G': 'GaiaDR2v2_G',
    'GAIA/GAIA2:Gbp': 'GaiaDR2v2_BP',
    'GAIA/GAIA2:Grp': 'GaiaDR2v2_RP',

    'SkyMapper:u': 'SkyMapper_u',
    'SkyMapper:v': 'SkyMapper_v',
    'SkyMapper:g': 'SkyMapper_g',
    'SkyMapper:r': 'SkyMapper_r',
    'SkyMapper:i': 'SkyMapper_i',
    'SkyMapper:z': 'SkyMapper_z',

    'HIP:BT': 'TYCHO_B_MvB',
    'HIP:VT': 'TYCHO_V_MvB',
    'TYCHO:BT': 'TYCHO_B_MvB',
    'TYCHO:VT': 'TYCHO_V_MvB',

    'Stromgren:u': 'STROMGREN_u',
    'Stromgren:v': 'STROMGREN_v',
    'Stromgren:b': 'STROMGREN_b',
    'Stromgren:y': 'STROMGREN_y',

    'Kepler:Kp': 'KEPLER_Kp',
    'TESS:T': 'TESS',
}

# Bands kept out of the photosphere by default. Not because a grid cannot
# predict them - most do, and the excess analysis relies on that - but because
# what is measured there is largely not the star, and a band of excess left in
# the fit biases the temperature and the radius of exactly the objects it is
# interesting on. Where a grid genuinely stops short is the grid's own reach
# instead. A reader who wants one of these fitted can name it, and get it.
NOT_PHOTOSPHERE = ('WISE_RSR_W3', 'WISE_RSR_W4', 'HERSCHEL_PACS_BLUE',
                   'HERSCHEL_PACS_GREEN', 'HERSCHEL_PACS_RED',
                   'SPITZER_IRAC_58', 'SPITZER_IRAC_80')

# What is known about the grids that arrive without saying anything about
# themselves - astroARIADNE's, which are named for their files and carry no
# metadata. A grid built here writes its own, and needs no entry.
#
# A grid is offered on the form when it can say what it is, from here or from
# its file. The rest of what a directory holds is still loadable by name; it is
# simply not put in front of a reader who has been told nothing about it.
KNOWN_GRIDS = {
    'btsettl': ('BT-Settl', None, None),
    'tlusty': ('TLUSTY', 'the hot-star models', None),
    'phoenix': ('Phoenix v2', None, None),
    'ck04': ('Castelli & Kurucz', None, 8.5),
    'kurucz': ('Kurucz', None, 8.5),
    'bosz': ('BOSZ', None, None),
    'koester': ('Koester', 'white dwarfs', 3.0),
}

# What astroARIADNE calls the files, for a directory laid out its way. A
# directory laid out ours names each file for its grid and needs none of this.
LEGACY_STEMS = {
    'btsettl': 'BTSettl', 'btcond': 'BTCond', 'btnextgen': 'BTNextGen',
    'tlusty': 'TLUSTY', 'phoenix': 'Phoenixv2', 'ck04': 'CK04',
    'kurucz': 'Kurucz', 'bosz': 'BOSZ', 'coelho': 'Coelho',
    'koester': 'Koester', 'newera': 'NewEra', 'sphinx': 'SPHINX',
    'atmo': 'ATMO2020', 'sonora': 'Sonora', 'stagger': 'Stagger',
    'btdusty': 'BTDusty',
}

# Two of those files are named for a version rather than for the grid
LEGACY_NAMES = {'phoenixv2': 'phoenix', 'atmo2020': 'atmo'}


def grids_dir():
    """Where the model cubes live.

    Ours by configuration if the SEDFIT_GRIDS setting says so. Failing that,
    wherever astroARIADNE was installed, which is where the first of these
    grids came from - the only thing that package is still wanted for, and
    only until its files have been read once and written out here.
    """
    from django.conf import settings

    configured = getattr(settings, 'SEDFIT_GRIDS', None)
    if configured:
        return configured

    try:
        from astroARIADNE.config import gridsdir
    except ImportError:
        raise SourceError('no model grids: set SEDFIT_GRIDS to the directory '
                          'they are in')

    return gridsdir


def grid_registry():
    """Every grid the directory holds, by name, and what is known of each.

    The directory is the registry: a file is a grid, its name is the file's,
    and what a reader is told about it is what the file says. Adding a grid is
    putting one there. Spectra sit beside their cube as ``<name>.spectra.h5``
    and are optional - without them a model is drawn per band and not as a line.

    astroARIADNE's own directory is read as well, which names its files for the
    grids in a different case and keeps every spectrum in one cache; both are
    understood, so a split can happen when it happens rather than being
    required first.
    """
    import h5py

    path = grids_dir()
    out = {}
    if not path or not os.path.isdir(path):
        return out

    for name in sorted(glob.glob(os.path.join(path, '*.h5'))):
        if name.endswith('.spectra.h5') or name.endswith('.xp.h5'):
            continue

        stem = os.path.splitext(os.path.basename(name))[0]
        entry = {'name': LEGACY_NAMES.get(stem.lower(), stem.lower()),
                 'path': name, 'label': None, 'description': None,
                 'reach_um': None, 'axis_logg': 'logg'}

        try:
            with h5py.File(name, 'r') as h:
                for key in ('name', 'label', 'description'):
                    if key in h.attrs:
                        entry[key] = str(h.attrs[key])
                entry['axis_logg'] = str(h.attrs.get('axis_logg', 'logg'))
                if 'reach_um' in h.attrs:
                    entry['reach_um'] = float(h.attrs['reach_um'])
                teff = np.asarray(h['teff'][:], dtype=float) if 'teff' in h else None
        except (OSError, KeyError):
            continue

        if teff is not None and len(teff):
            entry['teff_lo'] = float(teff.min())
            entry['teff_hi'] = float(teff.max())

        # What the file did not say, for the grids that came with no way to
        knows = KNOWN_GRIDS.get(entry['name'])
        if knows:
            entry['label'] = entry['label'] or knows[0]
            entry['description'] = entry['description'] or knows[1]
            if entry['reach_um'] is None:
                entry['reach_um'] = knows[2]

        spectra = os.path.join(path, entry['name'] + '.spectra.h5')
        entry['spectra'] = spectra if os.path.exists(spectra) else None

        # The Gaia XP bins, where they have been written: a cube of the same
        # shape as the grid's own over the models its spectra cover, which is
        # what lets the spectrum be fitted rather than only compared with
        bins = os.path.join(path, entry['name'] + '.xp.h5')
        entry['xp'] = bins if os.path.exists(bins) else None

        out[entry['name']] = entry

    return out


def offered_grids():
    """The grids a reader is offered, in temperature order.

    A grid is offered when it can say what it is. A directory may hold others -
    the cool-dwarf and specialist grids astroARIADNE ships - and those stay
    loadable by name without being put in front of someone who has been told
    nothing about them.
    """
    grids = [g for g in grid_registry().values() if g['label']]
    grids.sort(key=lambda g: g.get('teff_lo') or 0)

    out = []
    for grid in grids:
        span = (f"{grid['teff_lo']:.0f} - {grid['teff_hi']:.0f} K"
                if grid.get('teff_lo') else '')

        # Named, described and nothing else: where the file is is nobody's
        # business but this machine's
        out.append({'name': grid['name'], 'label': grid['label'],
                    'note': ', '.join(_ for _ in (span, grid['description']) if _),
                    'has_spectra': has_spectra(grid['name']),
                    'has_xp': bool(grid.get('xp')),
                    'axis_logg': grid.get('axis_logg') or 'logg'})

    return out


def spectra_cache_path():
    """astroARIADNE's one-file cache of spectra, if there is one.

    The layout everything came from: seven grids in one file of some gigabytes.
    A grid whose spectra sit beside it does not need this, and a directory that
    has been split does not need it at all.
    """
    from django.conf import settings

    # A file, not a directory: once the grids are split each carries its own
    # spectra and this is not wanted at all, and a directory given here would
    # otherwise be handed to h5py to fail on
    configured = getattr(settings, 'SEDFIT_SPECTRA', None)
    if configured:
        return configured if os.path.isfile(configured) else None

    try:
        from astroARIADNE.config import spectra_cache
    except ImportError:
        return None

    return spectra_cache if spectra_cache and os.path.exists(spectra_cache) else None


def spectra_source(name):
    """Where one grid's spectra are, as (path, group), or None.

    Beside the cube where the directory has been split, and otherwise the group
    of that name in the one-file cache.
    """
    entry = grid_registry().get(str(name).lower()) or {}
    if entry.get('spectra'):
        return entry['spectra'], None

    cache = spectra_cache_path()
    return (cache, str(name).lower()) if cache else None


# How far the nearest node of a grid's spectra may be from the fit and still be
# drawn as it. A tenth of a dex in temperature is a quarter of the way between
# any two nodes of the coarsest grid here; a dex of gravity or of metallicity
# changes a spectrum less than that does. Past these there is a line to draw
# and it is not this one.
SPECTRUM_REACH = {'teff': 0.04, 'logg': 1.0, 'feh': 1.0}


def has_spectra(name):
    """Whether spectra can be had for this grid, from wherever they are.

    Beside the cube once a directory is split, in the one-file cache before
    that, and for several grids nowhere at all - which is worth knowing before
    a fit rather than after, since it is the difference between a model drawn
    as a line and one drawn as a row of diamonds.
    """
    import h5py

    source = spectra_source(name)
    if source is None:
        return False

    path, group = source
    if group is None:
        return os.path.exists(path)

    try:
        with h5py.File(path, 'r') as h:
            return group in h
    except OSError:
        return False


def model_spectrum(name, teff, logg, feh):
    """The model spectrum nearest the given parameters, or None.

    What the cubes were convolved from. The cube the fit interpolates holds a
    flux per filter; this is the spectrum behind it, and drawing it says what a
    row of diamonds cannot - where the Balmer jump falls, which bands sit on a
    molecular band, how much of a colour is a line and how much a continuum.

    It is the nearest node and not an interpolation. The fit's parameters fall
    between nodes, and interpolating spectra would mean reading and averaging
    several hundred megabytes to move a line by a per cent; which node it is
    comes back with it, so that whatever draws it can say so.

    The distance is measured in what changes a spectrum: a fractional
    temperature, and a half dex of gravity or of metallicity. Teff first, since
    a hundred kelvin does more to the shape than a whole node of the others.

    Nearest is not the same as near. A grid's spectra need not cover all of its
    cube - what is published at full wavelength is often a subset of what was
    computed - and the nearest node to a fit outside that subset can be a
    thousand kelvin away. Drawn, it would be a line that is not the model, so
    beyond SPECTRUM_REACH nothing is returned and the caller says as much.
    """
    import h5py

    source = spectra_source(name)
    if source is None:
        return None

    path, group = source
    with h5py.File(path, 'r') as h:
        if group is not None and group not in h:
            return None

        node = h[group] if group is not None else h
        t = np.asarray(node['teff'][:], dtype=float)
        g = np.asarray(node['logg'][:], dtype=float)
        z = np.asarray(node['z'][:], dtype=float)

        distance = (((np.log10(t) - np.log10(teff)) / 0.02) ** 2
                    + ((g - logg) / 0.5) ** 2
                    + ((z - feh) / 0.5) ** 2)
        i = int(np.argmin(distance))

        # Near enough to be the model, or nothing at all
        if (abs(np.log10(t[i] / teff)) > SPECTRUM_REACH['teff']
                or abs(g[i] - logg) > SPECTRUM_REACH['logg']
                or abs(z[i] - feh) > SPECTRUM_REACH['feh']):
            return None

        return {'wave_um': np.asarray(node['wavelength'][:], dtype=float),
                'flux': np.asarray(node['flux'][i], dtype=float),
                'teff': float(t[i]), 'logg': float(g[i]), 'feh': float(z[i])}


def node_correction(grid, spectrum, values):
    """What puts a node's spectrum on the scale of the fit it stands for.

    The spectra are a coarser grid than the cube and the fit lands between
    their nodes, so the nearest one can be several per cent away in flux - two
    and a half on a TLUSTY fit half a node from its neighbour, and in the
    other direction on the next grid along. Drawn as it is, that is a line
    which is not the model the diamonds are, and measured against it a
    spectrum that agrees with the fit perfectly looks five per cent out.

    The cube knows the difference exactly: it can be evaluated at the fit and
    at the node, in every filter it covers, and the ratio of those is how much
    the node is wrong band by band. Interpolated across wavelength it is a
    smooth curve - it is an interpolation in temperature and gravity and
    nothing sharper - and multiplying the spectrum by it leaves every line
    where it was while putting the continuum where the fit puts it.

    Beyond the reddest and bluest filter the grid covers there is nothing to
    measure it with, so it is held at the value it had there rather than run
    off to somewhere it was never checked.
    """
    pivots, ratios = [], []

    for entry in known_bands():
        band = entry['band']
        column = grid.column.get(band)
        if column is None:
            continue

        columns = np.array([column])
        fit = grid.flux(values['teff'], values['logg'], values['feh'], columns)[0]
        node = grid.flux(spectrum['teff'], spectrum['logg'], spectrum['feh'],
                         columns)[0]

        if np.isfinite(fit) and np.isfinite(node) and fit > 0 and node > 0:
            pivots.append(entry['wavelength'] * 1e-4)
            ratios.append(fit / node)

    if len(pivots) < 2:
        return None

    order = np.argsort(pivots)
    x = np.log(np.asarray(pivots, dtype=float)[order])
    y = np.asarray(ratios, dtype=float)[order]

    return lambda wave_um: np.interp(np.log(np.asarray(wave_um, dtype=float)),
                                     x, y)


def observed_spectrum(name, theta, grid=None):
    """That spectrum as it would be seen: diluted by (R/d)^2 and reddened.

    The same three operations the per-band model is put through, so the line
    and the diamonds on it are the same model said two ways - and a fourth,
    which is what makes that true. The spectrum is the nearest node and the
    fit is between nodes, so it is put on the fit's own scale first, by the
    only thing that knows the difference: the cube, evaluated at both.
    """
    values = dict(zip(PARAMETERS, theta)) if not isinstance(theta, dict) else theta

    spectrum = model_spectrum(name, values['teff'], values['logg'], values['feh'])
    if spectrum is None:
        return None

    flux = spectrum['flux']
    spectrum['correction'] = None

    correction = (node_correction(grid, spectrum, values)
                  if grid is not None else None)
    if correction is not None:
        scale = correction(spectrum['wave_um'])
        flux = flux * scale
        spectrum['correction'] = float(np.median(scale[np.isfinite(scale)]))

    dilution = (values['rad'] * R_SUN / (values['dist'] * PARSEC)) ** 2
    reddening = 10 ** (-0.4 * attenuation(spectrum['wave_um'], values['Av']))

    spectrum['observed'] = flux * dilution * reddening
    return spectrum


def has_xp(name):
    """Whether this grid can answer for the Gaia XP bins."""
    return bool((grid_registry().get(str(name).lower()) or {}).get('xp'))


def load_grid(name, xp=False):
    """One grid by name, from the directory or from astroARIADNE's own.

    With ``xp``, the Gaia XP bins are loaded alongside the filters and the two
    answer as one grid - which is all there is to fitting a spectrum here.
    """
    name = str(name).lower()
    entry = grid_registry().get(name)
    if entry:
        grid = Grid(entry['path'], name=name)
        if xp and entry.get('xp'):
            grid = CompositeGrid(grid, Grid(entry['xp'], name=f'{name}-xp'))
        return grid

    # A directory laid out astroARIADNE's way, being read before it is split
    stem = LEGACY_STEMS.get(name)
    if not stem:
        raise SourceError(f'no grid called {name}')

    return Grid(os.path.join(grids_dir(), f'{stem}.h5'), name=name)


def read_sed_points(path, points=None, extra=None,
                    err_floor=0.03, err_unknown=0.05):
    """The rows of one of our SED files, on the convention the grids were built on.

    A point is named by the comment the SED step writes - the catalogue and the
    filter designation together, 'Pan-STARRS PAN-STARRS/PS1:g' - which is
    unique within a file and names a *measurement* rather than a band. That is
    what lets one Pan-STARRS point be dropped without dropping the survey, and
    what makes sed_all.vot selectable at all, since it carries every catalogue
    at the position rather than one per band.

    The grid columns are pyphot's photon-weighted mean flux density, and pivot
    wavelength is defined so that <f_lambda> = <f_nu> c / lambda_p^2 - so the
    whole conversion from what VizieR gave us is a factor of
    (lambda_VizieR / lambda_pivot)^2. No zero point enters it.

    Every row is returned, with three things said about it. ``fittable`` is
    whether it ever could be fitted: a band no grid models, or a row with no
    flux in it, is a property of the data and not a choice anyone can change.
    ``default`` is what would be chosen if nobody chose - one point per band,
    and nothing from the far infrared, which is right for a file where the same
    measurement is republished by a dozen catalogues and where a band of excess
    would bias the photosphere. ``used`` is what will actually be fitted, and
    ``note`` says how it came to be that.

    ``points``, when given, is the whole answer: those points are fitted and no
    others, whatever the defaults would have said. Two points on one band is
    then allowed, because two surveys that genuinely measured it are two
    measurements and a likelihood knows what to do with them. Nothing is
    promoted or demoted behind the caller's back.
    """
    from astropy.table import Table, vstack

    points = set(points) if points is not None else None

    table = Table.read(path)

    # Points added by hand sit in their own file and are read alongside
    # whichever of ours was asked for, first, so that a measurement someone
    # entered deliberately takes the band from a catalogue's version of it.
    if extra is not None and len(extra):
        table = vstack([extra[table.colnames], table])

    rows, seen = [], set()
    for row in table:
        comment = str(row['comment'])
        catalogue, _, designation = comment.rpartition(' ')
        band = (designation if catalogue == EXTRA_CATALOGUE
                else FILTER_MAP.get(designation))

        # Two wavelengths, and the difference matters. The fit needs the
        # filter's pivot, which is the one that makes the observed flux
        # comparable with the grid's. The viewer draws the point where its own
        # file says it is - VizieR's effective wavelength - and the two differ
        # by up to three per cent, so a model drawn at the pivot sits visibly
        # beside the measurement it belongs to.
        entry = {'id': comment, 'band': band, 'used': False, 'default': False,
                 'fittable': False, 'note': None,
                 'wave_um': None, 'wave_drawn_um': None,
                 'flux': None, 'err': None}
        rows.append(entry)

        if band is None:
            entry['note'] = 'no model filter of this name'
            continue

        pivot = pivot_aa(band)
        entry['wave_um'] = pivot * 1e-4
        entry['wave_drawn_um'] = float(row['wavelength']) * 1e-4

        flux = float(row['flux'])
        if not np.isfinite(flux) or flux <= 0:
            entry['note'] = 'no flux'
            continue

        entry['flux'] = flux * (float(row['wavelength']) / pivot) ** 2 * 1e4

        error = row['flux_error']
        error = float(error) if error is not None and np.isfinite(error) else np.nan
        if np.isfinite(error) and error > 0:
            relative = max(error / flux, err_floor / 1.0857)
        else:
            relative = (err_unknown or 0.10) / 1.0857
        entry['err'] = entry['flux'] * relative

        # Everything from here is a choice about the point rather than a fact
        # about it, so from here it could be fitted if it were asked for
        entry['fittable'] = True

        # What would be chosen if nobody chose, and why not where not
        if band in NOT_PHOTOSPHERE:
            entry['note'] = 'infrared excess, not photosphere'
        elif band in seen:
            entry['note'] = 'a second point for this band'
        else:
            seen.add(band)
            entry['default'] = True

        entry['used'] = entry['default'] if points is None else comment in points

        # And how it came to be that, where it was not the default
        if entry['used'] and not entry['default']:
            entry['note'] = 'chosen by hand'
        elif not entry['used'] and entry['default']:
            entry['note'] = 'turned off'

    rows.sort(key=lambda r: (r['wave_um'] is None, r['wave_um'] or 0))

    for row in rows:
        row['excess'] = False
    for row in default_excess(rows):
        row['excess'] = True

    return rows


def xp_points(basepath, points=None, xp=None):
    """The Gaia XP spectrum as points, in the shape the SED files come in.

    A bin is a band, so a bin is a row: the same keys, the same meaning of
    ``used`` and ``default`` and ``fittable``, and the same treatment
    everywhere downstream. Nothing here knows it came from a spectrum.

    They are their own points and not part of an SED file. That file is
    written by the SED step and the spectrum by the info step, and a fit that
    asks for both is asking for two measurements of the same star, not for one
    file with more rows in it.
    """
    xp = xp if xp is not None else read_xp(basepath)
    if xp is None:
        return []

    points = set(points) if points is not None else None
    rows = []

    for entry in xp_binned(xp):
        name = f"Gaia XP {entry['lo_nm']:.0f}-{entry['hi_nm']:.0f} nm"
        row = {'id': name, 'band': entry['band'], 'used': False,
               'default': True, 'fittable': True, 'note': None, 'xp': True,
               'wave_um': entry['wave_um'], 'wave_drawn_um': entry['wave_um'],
               'flux': entry['observed'], 'err': entry['err'],
               'excess': False}

        row['used'] = row['default'] if points is None else name in points
        if row['used'] and not row['default']:
            row['note'] = 'chosen by hand'
        elif not row['used'] and row['default']:
            row['note'] = 'turned off'

        rows.append(row)

    return rows


# The catalogues whose photometry is these very spectra integrated - Gaia's own
# synthetic photometry, and its broadband bands, which are the same photons in
# a wider filter. Fitting those and the bins together is fitting one
# measurement twice, so where the bins are fitted these are not.
XP_DERIVED_CATALOGUES = ('Gaia-syntphot',)
XP_DERIVED_BANDS = ('GaiaDR2v2_G', 'GaiaDR2v2_BP', 'GaiaDR2v2_RP')


def xp_derived(rows):
    """Which points of a list are the XP spectrum in another form."""
    out = []

    for row in rows:
        if is_xp(row['band']):
            continue

        catalogue = str(row['id']).rpartition(' ')[0]
        if (row['band'] in XP_DERIVED_BANDS
                or any(catalogue.startswith(c) for c in XP_DERIVED_CATALOGUES)):
            out.append(row['id'])

    return out


def drop_xp_derived(rows, log=None):
    """Turn off what the XP bins already say, and say which."""
    derived = set(xp_derived(rows))
    dropped = []

    for row in rows:
        if row['used'] and row['id'] in derived:
            row['used'] = False
            row['note'] = 'the XP bins are this measurement already'
            dropped.append(row['id'])

    if dropped and log:
        log(f"\n{len(dropped)} point(s) left out as the spectrum they came"
            f" from is being fitted:")
        for each in dropped:
            log(f'    {each}')

    return dropped


def default_excess(rows):
    """The points an excess would be measured on if nobody chose.

    Everything past the reddest band being fitted, which is where a photosphere
    stops being what was measured. A point short of that which is not fitted
    was set aside rather than left over - a second catalogue for a band, or one
    the reader does not believe - and an excess is not what it is.

    One point per band, on the same reasoning the fit uses: in a file where the
    same measurement is republished a dozen times, a dozen copies of it would
    weigh that band a dozen times over.
    """
    edge = max((r['wave_um'] for r in rows if r['used'] and r['wave_um']),
               default=None)
    if edge is None:
        return []

    out, seen = [], set()
    for row in rows:
        if (not row['fittable'] or row['used'] or row['band'] in seen
                or not row['wave_um'] or row['wave_um'] <= edge):
            continue

        seen.add(row['band'])
        out.append(row)

    return out


def target_sed_fit(config, basepath='.', outpath=None, selection=None,
                   options=None, verbose=None):
    """Fit the photosphere, and write the run where it can be found again.

    A fit is not a data product of the target the way a light curve is: it is
    an answer to a question that was asked with particular choices, and the
    next question will be asked with different ones. So each run gets its own
    directory holding what was asked as well as what came back, and no run
    overwrites another.
    """
    import glob
    import json

    log = verbose if callable(verbose) else (print if verbose else lambda *a, **k: None)

    selection = dict(selection or {})
    options = dict(options or {})
    outpath = outpath or basepath

    source = selection.get('source') or 'sed.vot'
    sed = os.path.join(basepath, source)
    if not os.path.exists(sed):
        raise SourceError(f'{source} is not there - run the SED step first')

    rows = read_sed_points(
        sed, points=selection.get('points'),
        extra=(read_extra_points(basepath)
               if selection.get('extra', True) else None),
        err_floor=options.get('err_floor', 0.03),
        err_unknown=options.get('err_unknown', 0.05))

    # Read once, ahead of everything: it is one measurement of the star, every
    # grid is compared against the same copy of it, and where it is being
    # fitted its bins are points in the list like any others
    xp = read_xp(basepath) if options.get('xp', True) else None
    fit_xp = bool(xp is not None and options.get('xp_fit'))

    if fit_xp:
        # What the bins replace goes out: Gaia's synthetic photometry is these
        # very spectra integrated, and fitting both is one measurement twice.
        #
        # Only where nobody chose, though. A selection given is the whole
        # answer here as it is everywhere else in this module - somebody who
        # has deliberately kept a synthetic point beside the bins is allowed
        # to, and is told what they have rather than quietly corrected.
        if selection.get('points') is None:
            drop_xp_derived(rows, log)
        else:
            kept = [r['id'] for r in rows if r['used']
                    and r['id'] in set(xp_derived(rows))]
            if kept:
                log(f"\n{len(kept)} point(s) are the XP spectrum in another"
                    f" form and are fitted beside it, having been asked for:")
                for each in kept:
                    log(f'    {each}')

        rows = rows + xp_points(basepath, points=selection.get('points'), xp=xp)
        rows.sort(key=lambda r: (r['wave_um'] is None, r['wave_um'] or 0))

    used = [r for r in rows if r['used']]
    log(f"\n{len(used)} of {len(rows)} points from {source}"
        + (f" and {sum(1 for r in rows if is_xp(r['band']))} bins of the Gaia"
           f" XP spectrum" if fit_xp else ''))
    for r in rows:
        mark = 'fit ' if r['used'] else '   -'
        log(f"  {mark} {r['id']:<34} {r['band'] or '':<18}"
            + (f"{r['wave_um']:7.3f} um" if r['wave_um'] else ' ' * 10)
            + (f"  {r['note']}" if r['note'] else ''))

    if len(used) < 5:
        raise SourceError(f'{len(used)} points is not enough to fit six parameters')

    bands = [r['band'] for r in used]
    wave = np.array([r['wave_um'] for r in used])
    drawn = np.array([r['wave_drawn_um'] or r['wave_um'] for r in used])
    flux = np.array([r['flux'] for r in used])
    err = np.array([r['err'] for r in used])

    distance = options.get('distance', config.get('gaia_distance'))
    distance_err = options.get('distance_err')
    if distance_err is None and config.get('gaia_distance_hi') and distance:
        distance_err = max(config['gaia_distance_hi'] - distance,
                           distance - config['gaia_distance_lo'])
    av_max = options.get('av_max')
    if av_max is None:
        av_max = 2.742 * config['ebv_sfd'] if config.get('ebv_sfd') else 1.0

    if xp is not None and not fit_xp:
        log(f"\nGaia XP spectrum: {len(xp['wave_um'])} points,"
            f" {xp['wave_um'][0] * 1e3:.0f} to {xp['wave_um'][-1] * 1e3:.0f} nm."
            f" It is compared with each fit and fitted by none of them")

    runs, summaries = [], {}
    for name in options.get('grids') or ['btsettl']:
        # The bins are loaded whether or not they are fitted: unfitted, they
        # are what the comparison is made against, and that needs the same cube
        grid = load_grid(name, xp=xp is not None)
        priors = default_priors(grid, distance, distance_err, av_max)
        # The temperature prior is deliberately the same whatever grid is
        # loaded, so that a grid's extent cannot become the answer
        priors['teff'] = tuple(options.get('teff_prior')
                               or ('loguniform', 2000.0, 70000.0))
        if options.get('logg_prior'):
            # On a grid whose second axis is not a gravity, a gravity prior is
            # not a statement about the star and would cut the grid where it
            # happened to fall. The axis keeps its own extent, and it is said.
            if grid.axis_logg == 'logg':
                priors['logg'] = tuple(options['logg_prior'])
            else:
                log(f"\n{name}: its second axis is {grid.axis_logg}, not a"
                    f" gravity - the gravity prior is left off it and the axis"
                    f" keeps the grid's own extent")

        missing = sorted({b for b in bands if b not in grid.covers})
        if missing:
            reason = ('no Gaia XP bins - "manage.py sedgrid --xp" writes them'
                      if all(is_xp(b) for b in missing)
                      else f"no model flux for {', '.join(missing)}")
            log(f"\n{name}: {reason} - skipped")
            continue

        log(f"\nfitting {name}: Teff {grid.teff.min():.0f}-{grid.teff.max():.0f} K")
        run = fit(bands, wave, flux, err, grid, priors,
                  nlive=options.get('nlive', 500), seed=options.get('seed', 0),
                  verbose=False, progress=log)
        run['wave_drawn_um'] = drawn
        run['axis_logg'] = grid.axis_logg
        summary = summarise(run, grid)
        summary['axis_logg'] = grid.axis_logg
        summary['shrink'] = shrinkage(run, priors['teff'])
        summary['residuals'] = residual_table(run, summary, grid)
        runs.append((run, grid))
        summaries[grid.name] = summary

        log(f"  log Z {run['logz']:.2f} +/- {run['logz_err']:.2f}"
            f"   chi2 {summary['chi2']:.1f} on {len(bands)} bands"
            f"   prior shrinkage {100*summary['shrink']:.0f}%")
        log_parameters(summary, log)

        # What the points beyond the fit do, which the fit itself says nothing
        # about - it was not shown them. Per grid, as everything else here is:
        # the excess is measured against a photosphere, and two grids predict
        # two of those.
        if options.get('excess', True):
            try:
                beyond = excess_rows(rows, [grid], selection.get('excess'))
                if beyond:
                    summary['excess'] = quantify_excess(beyond, run, grid)
                    summary['excess_models'] = compare_excess(
                        summary['excess'], run, grid, summary)
                    add_excess_residuals(summary['excess'],
                                         summary['excess_models'])
                    log_excess(summary['excess'], summary['excess_models'], log)

                # What was left out of both, for the figure to show. Measured
                # the same way, so that a point set aside can be seen to have
                # deserved it - or not.
                summary['unused'] = quantify_excess(
                    unused_rows(rows, [grid], beyond), run, grid)
            except Exception as e:
                log(f'  infrared excess failed: {type(e).__name__}: {e}')

    if not runs:
        raise SourceError('no grid covers every fitted band')

    result = {
        'source': source,
        'selection': selection,
        'options': options,
        'points': rows,
        'grids': summaries,
        # Grids are not averaged. Two that both cover the regime disagree by
        # an amount that is the systematic from the atmosphere physics, and
        # that is worth reporting; a weighted mean of them is not, the more so
        # as the evidences differ by less than the sampler's own noise.
        'spread': _grid_spread(summaries),
    }

    os.makedirs(outpath, exist_ok=True)
    for run, grid in runs:
        np.save(os.path.join(outpath, f'samples_{grid.name}.npy'), run['samples'])

        # What Gaia measured over the optical, against what this fit predicts
        # in the same bins. It needs no spectrum and no node: the bins are
        # columns of a cube, so the prediction is an interpolation at the
        # fitted parameters like any other.
        if xp is not None:
            try:
                theta = np.array([summaries[grid.name]['best'][p]
                                  for p in PARAMETERS])
                summaries[grid.name]['xp'] = compare_xp(xp, run, grid, theta)
                log_xp(summaries[grid.name]['xp'], log)
            except SourceError as e:
                log(f'\n  no Gaia XP comparison for {grid.name}: {e}')
            except Exception as e:
                log(f'\n  Gaia XP comparison failed: {type(e).__name__}: {e}')

        # The model as a spectrum, at the row it is drawn at, kept with the run
        # so that neither the figure nor the viewer has to go back to a cache
        # of several gigabytes to draw a line
        try:
            spectrum = observed_spectrum(grid.name, summaries[grid.name]['best'],
                                         grid=grid)
        except Exception as e:
            spectrum = None
            log(f'no model spectrum for {grid.name}: {type(e).__name__}: {e}')

        if spectrum is None:
            # Worth saying: a reader who has seen a line under one grid will
            # wonder where it went under the next
            near = 'no spectra' if not has_spectra(grid.name) else \
                'no spectrum near this fit' 
            log(f"\n  {near} for {grid.name} - it is drawn per band"
                f" and not as a line")
            continue

        np.save(os.path.join(outpath, f'model_{grid.name}.npy'),
                np.vstack([spectrum['wave_um'],
                           spectrum['observed']]).astype('float32'))
        log(f"\n  {grid.name} spectrum: nearest node is"
            f" {spectrum['teff']:.0f} K,"
            f" log g {spectrum['logg']:.1f}, [Fe/H] {spectrum['feh']:+.1f}")
        if spectrum.get('correction'):
            log(f"    put on the fit's own scale, which the cube says is"
                f" {100 * (spectrum['correction'] - 1):+.1f}% from that node")

    # Written after the runs rather than before them, since what those learn -
    # which node the spectrum came from, how the XP spectrum compares - belongs
    # in the file a reader opens
    with open(os.path.join(outpath, 'fit.json'), 'w') as f:
        json.dump(result, f, indent=1, default=float)

    # Drawn last, and never allowed to lose a run: the fit is the thing, and a
    # figure that will not render is not a reason to throw away an hour of
    # sampling. Whichever ones worked are on disk and the viewer finds them.
    if options.get('figures', True):
        for run, grid in runs:
            try:
                draw_sed(run, grid, summaries[grid.name], outpath, xp=xp)
            except Exception as e:
                log(f'SED plot for {grid.name} failed: {type(e).__name__}: {e}')

            try:
                draw_corner(run, outpath)
            except Exception as e:
                log(f'corner plot for {grid.name} failed: {type(e).__name__}: {e}')

        try:
            draw_histograms([run for run, _ in runs], outpath)
        except Exception as e:
            log(f'histograms failed: {type(e).__name__}: {e}')

        log(f"\ndrew {len(glob.glob(os.path.join(outpath, '*.png')))} figure(s)")

    if len(summaries) > 1:
        log(f"\ngrid-to-grid Teff spread {result['spread']['teff']:.0f} K")

    return result


def shrinkage(result, teff_prior):
    """How much narrower the temperature posterior is than its prior.

    Near one, the answer is the data. Near zero it is the prior wearing the
    data's clothes, which for a reddened star with no ultraviolet is what
    happens - and is worth saying rather than quoting a number.
    """
    kind = teff_prior[0]
    if kind == 'loguniform':
        prior_sd = np.log(teff_prior[2] / teff_prior[1]) / np.sqrt(12)
    elif kind in ('uniform', 'grid'):
        lo, hi = teff_prior[1:3]
        prior_sd = np.log(hi / lo) / np.sqrt(12)
    else:
        return float('nan')

    return float(1 - np.std(np.log(result['samples'][:, 0])) / prior_sd)


def residual_table(result, summary, grid):
    """Observed against model at the row the fit is drawn at.

    The model is that one row, which is what the figures draw and what makes
    the parameters beside it mean anything together. What it is divided by is
    both errors: the catalogue's on the measurement, and the posterior's on
    the prediction. A band the fit barely constrains is predicted loosely, and
    a residual there is worth fewer sigma than the same residual in a band the
    fit is pinned to - which dividing by the catalogue error alone will not say.
    """
    theta = np.array([summary['best'][p] for p in PARAMETERS])
    _, model = residuals(result, theta, grid)
    _, model_err = predict_bands(result, grid, result['bands'],
                                 result['wave_um'])

    drawn = result.get('wave_drawn_um')
    if drawn is None:
        drawn = result['wave_um']

    sigma = np.hypot(result['flux_err'], np.nan_to_num(model_err))
    res = (result['flux'] - model) / sigma

    return [{'band': b, 'wave_um': float(w), 'wave_drawn_um': float(d),
             'observed': float(f), 'model': float(m), 'model_err': float(me),
             'residual': float(r)}
            for b, w, d, f, m, me, r in zip(result['bands'], result['wave_um'],
                                            drawn, result['flux'], model,
                                            model_err, res)]


def _grid_spread(summaries):
    """What the choice of atmosphere physics is worth, as a second error bar."""
    if len(summaries) < 2:
        return {}

    out = {}
    for key in ('teff', 'Av', 'theta_mas'):
        values = [s[key]['median'] for s in summaries.values()]
        out[key] = float(max(values) - min(values))
    return out


# ------------------------------------------------------- points added by hand

# Where a point that came from neither of our SED files is kept. Beside them
# rather than inside them: those two are written by the SED step and cleared
# when it runs again, and something typed in by hand should survive that.
EXTRA_FILE = 'sed_extra.vot'

# The catalogue field a hand-added point carries, in place of the VizieR table
# a fetched one names. It is what tells the reader to take the designation as
# a filter name directly rather than looking it up.
EXTRA_CATALOGUE = 'added'


def known_bands():
    """Every band a point may be added in, blue to red.

    The grids' own filter set, less the ones no grid actually models - there is
    no use offering a band the fit would refuse.
    """
    bands = []
    for band in FILTER_NAMES:
        if band in NOT_PHOTOSPHERE:
            continue
        try:
            pivot = pivot_aa(band)
        except Exception:
            continue
        bands.append({'band': band, 'wavelength': pivot,
                      'system': 'AB' if is_ab(band) else 'Vega'})

    return sorted(bands, key=lambda b: b['wavelength'])


def magnitude_to_flux(band, mag, mag_err=None):
    """A magnitude in one band as f_lambda at its pivot, erg/s/cm2/A.

    Through f_nu and the band's zero point in Jansky, which is exact for an AB
    band and as good as the library's Vega spectrum for the others. The pivot
    wavelength is the one that makes <f_lambda> = <f_nu> c / lambda^2 true for
    a photon-counting filter, which is the convention the grids are on.
    """
    pivot = pivot_aa(band)
    zero = 3631.0 if is_ab(band) else vega_zero_jy(band)

    fnu = zero * 10 ** (-0.4 * float(mag)) * 1e-23
    flux = fnu * C_AA / pivot ** 2
    error = (flux * float(mag_err) / 1.0857) if mag_err else None

    return pivot, flux, error


def read_extra_points(basepath):
    """The hand-added points of a target, as a table or None."""
    import os

    from astropy.table import Table

    path = os.path.join(basepath, EXTRA_FILE)
    return Table.read(path) if os.path.exists(path) else None


def add_extra_point(basepath, band, value, error=None, unit='mag'):
    """Record a measurement that neither SED file has.

    One point per band: adding a band that is already there replaces it, which
    is what correcting a typo should do and saves a delete-then-add.
    """
    import os

    from astropy.table import Table, vstack

    if band not in {b['band'] for b in known_bands()}:
        raise SourceError(f'{band} is not a band any grid models')

    if unit == 'mag':
        wavelength, flux, flux_error = magnitude_to_flux(band, value, error)
    elif unit == 'flux':
        wavelength = pivot_aa(band)
        flux = float(value)
        flux_error = float(error) if error else None
    else:
        raise SourceError(f'{unit} is neither a magnitude nor a flux')

    if not np.isfinite(flux) or flux <= 0:
        raise SourceError('that is not a positive flux')

    row = Table({'wavelength': [wavelength],
                 'bandwidth': [float(_get_bandwidth(band))],
                 'flux': [flux],
                 'flux_error': [flux_error if flux_error else np.nan],
                 'comment': [f'{EXTRA_CATALOGUE} {band}']})

    existing = read_extra_points(basepath)
    if existing is not None:
        keep = [str(c) != f'{EXTRA_CATALOGUE} {band}' for c in existing['comment']]
        existing = existing[keep]
        row = vstack([existing, row]) if len(existing) else row

    row.sort('wavelength')
    row.write(os.path.join(basepath, EXTRA_FILE), format='votable',
              overwrite=True)

    return row


def remove_extra_point(basepath, band):
    """Drop a hand-added point again."""
    import os

    existing = read_extra_points(basepath)
    if existing is None:
        return None

    keep = [str(c) != f'{EXTRA_CATALOGUE} {band}' for c in existing['comment']]
    remaining = existing[keep]

    path = os.path.join(basepath, EXTRA_FILE)
    if len(remaining):
        remaining.write(path, format='votable', overwrite=True)
    else:
        os.remove(path)

    return remaining


def _get_bandwidth(band):
    """The band's width, for drawing it as the range of wavelength it is."""
    try:
        return width_aa(band)
    except Exception:
        return 0.0


# ------------------------------------------------------------------- figures

# What each sampled parameter is called on an axis, and how to write it
LABELS = {
    'teff': (r'$T_{\rm eff}$, K', '{:.0f}'),
    'logg': (r'$\log g$', '{:.2f}'),
    'feh': ('[Fe/H]', '{:+.2f}'),
    'dist': ('Distance, pc', '{:.0f}'),
    'rad': (r'Radius, $R_\odot$', '{:.2f}'),
    'Av': (r'$A_V$, mag', '{:.2f}'),
    'jitter': ('Jitter', '{:.3f}'),
    'theta_mas': (r'$\theta$, mas', '{:.4f}'),
    'lum_lsun': (r'Luminosity, $L_\odot$', '{:.3g}'),
}

# One colour per grid, in the order they were asked for. Deliberately the
# palette the spectral viewer draws its sources in, so a grid keeps the same
# colour whichever figure it appears in.
GRID_COLOURS = ['#2980b9', '#c0392b', '#16a085', '#8e44ad', '#e67e22',
                '#2c3e50', '#27ae60', '#d35400']


# The grids are tabulated per micron. Everything else on this site - the SED
# files, the spectral viewer - is per Angstrom, so that is what is drawn, and
# a number read off a figure is the number read off the viewer beside it.
PER_AA = 1e-4


# The excess is not the star, and is not drawn as though it were
EXCESS_COLOUR = '#c0392b'

# Gaia's spectrum is a measurement, and is drawn nearer the photometry's black
# than any grid's colour - but not the same, since nothing else on the figure
# was measured by one instrument at one epoch
XP_COLOUR = '#5d6d7e'


def _excess_label(models):
    """What the drawn excess shape is, in a few words for a legend."""
    if not models or models.get('preferred') == 'none':
        return 'excess'

    p = models['probability'][models['preferred']]
    if models['preferred'] == 'freefree':
        return (f"free-free, "
                rf"$\nu^{{{models['freefree']['alpha_best']:+.2f}}}$ (P = {p:.2f})")

    return (f"blackbody, {models['blackbody']['t_dust_best']:.0f} K"
            f" (P = {p:.2f})")


def draw_sed(run, grid, summary, path, name=None, colour=GRID_COLOURS[0],
             xp=None):
    """The photometry, the model that fits it, and what is left over.

    The model is drawn at the plot row - one place the posterior actually
    goes - and not at the marginal medians, which on a curved degeneracy lie
    off the ridge and fit nothing. Around it is what the rest of the posterior
    allows, band by band, so a band the fit is free to place anywhere is
    visibly that rather than a suspiciously good match.

    Two error bars per point, because the fit sees two: the catalogue's, and
    the catalogue's widened by the jitter the fit needed. The residual panel
    is in units of the first, as the point list in the viewer is, with the
    second drawn behind it as the envelope the fit was actually working to.

    Where Gaia published a spectrum it is drawn under all of it, and its bins
    appear in the residual panel with everything else. It was not fitted, and
    it is deliberately drawn in a way that says so - a thin line behind the
    model rather than a set of points the model passes through.
    """
    from stdpipe import plots

    theta = np.array([summary['best'][p] for p in PARAMETERS])

    order = np.argsort(np.asarray(run['wave_um']))
    wave = np.asarray(run['wave_um'])[order]
    bands = [run['bands'][i] for i in order]
    flux = run['flux'][order] * PER_AA
    err = run['flux_err'][order] * PER_AA
    model = model_flux(theta, run['columns'], run['ext_unit'], grid)[order] * PER_AA

    jitter = float(theta[PARAMETERS.index('jitter')])
    widened = np.hypot(err, jitter * model)

    # The XP bins are points like any other to the fit, but not to a figure:
    # twenty-one of them drawn as photometry would bury the photometry, and
    # the spectrum they were binned from is already the curve underneath. So
    # everything below draws the bands, and the bins are drawn once, as bins.
    shown = np.array([not is_xp(b) for b in bands])

    # What the rest of the posterior would have drawn. Two hundred rows is
    # enough for a 16-84 envelope and costs an interpolation each.
    draws = run['samples']
    if len(draws) > 200:
        draws = draws[np.random.default_rng(0).choice(len(draws), 200,
                                                      replace=False)]
    cloud = np.array([model_flux(row, run['columns'], run['ext_unit'], grid)[order]
                      for row in draws]) * PER_AA
    lo, hi = np.nanpercentile(cloud, [16, 84], axis=0)

    # Both errors, as the residual table has them: the catalogue's on the
    # measurement and the posterior's on the prediction
    total = np.hypot(err, np.nan_to_num(np.nanstd(cloud, axis=0)))
    residual = (flux - model) / total

    # The points beyond the fit, and the shape the excess came out as. They
    # are drawn because they are the reason the fit was told to leave them
    # out: an excess is a statement about the photosphere as much as the fit is.
    beyond = [e for e in (summary.get('excess') or []) if e['quantified']]
    left = [e for e in (summary.get('unused') or []) if e['quantified']]
    models = summary.get('excess_models')

    # The model as a spectrum, if the cache had one. Written beside the run by
    # the fit, so this only has to read it.
    spectrum = os.path.join(path, f'model_{name or run["grid"]}.npy')
    spectrum = np.load(spectrum) if os.path.exists(spectrum) else None

    filename = os.path.join(path, f'sed_{name or run["grid"]}.png')

    with plots.figure_saver(filename, figsize=(8, 6), tight_layout=False) as fig:
        top, low = fig.subplots(2, 1, sharex=True,
                                gridspec_kw={'height_ratios': [3, 1],
                                             'hspace': 0.06})

        # What Gaia measured over the optical, behind everything: it is the
        # one curve on this figure that is neither a model nor something the
        # fit was shown, and the whole of its value is how it lies against the
        # line drawn over it. Positive points only, the axis being logarithmic
        # and the blue end of a faint spectrum wandering below zero.
        checked = summary.get('xp')
        if xp is not None:
            drawable = np.isfinite(xp['flux']) & (xp['flux'] > 0)
            label = ('Gaia XP, fitted in bins' if checked
                     and checked.get('n_fitted') else 'Gaia XP, not fitted')
            if checked:
                quoted = checked.get('offset_within') or checked['offset']
                label += f", {100 * (quoted - 1):+.0f}% on the model"

            top.plot(xp['wave_um'][drawable], xp['flux'][drawable] * PER_AA,
                     '-', lw=1.0, color=XP_COLOUR, alpha=0.85, zorder=0.5,
                     label=label)

        # The spectrum the cubes were convolved from, at the nearest node the
        # cache carries, scaled to the fit the way the fit's own cube says that
        # node differs from it. Behind everything and thin, because it is the
        # model said at a resolution the fit never used - the diamonds are what
        # was fitted, and this is what they were summed from.
        if spectrum is not None:
            # A little either side of what was measured and no further. A grid
            # ingested from its own publication reaches from the ultraviolet to
            # the radio, and drawing all of it would spend most of the axis -
            # and most of the decades of the flux axis with it - on a stretch
            # nothing was ever measured in.
            red = max([wave[-1]] + [e['wave_um'] for e in beyond + left])
            near = ((spectrum[0] >= 0.6 * wave[0])
                    & (spectrum[0] <= 1.5 * red))
            top.plot(spectrum[0][near], spectrum[1][near] * PER_AA, '-',
                     lw=0.6, color=colour, alpha=0.5, zorder=0,
                     label=f'{run["grid"]} spectrum, node put on the fit')

        # Per band rather than a shaded curve across them: what the fit
        # produced is a flux in each filter, and a band drawn between them
        # would be claiming a spectrum it never computed
        top.vlines(wave[shown], lo[shown], hi[shown], color=colour,
                   alpha=0.35, lw=6, label='posterior, central 68%')
        top.plot(wave[shown], model[shown], 'D', mfc='none', ms=9, mew=1.6,
                 color=colour, ls='none',
                 label=f'{run["grid"]} at the plot row')
        # Behind the catalogue error and unlabelled: it is the same statement
        # the residual panel makes, and it is made there with room to say it
        top.errorbar(wave[shown], flux[shown], widened[shown], fmt='none',
                     ecolor='0.75', elinewidth=3, capsize=0)
        top.errorbar(wave[shown], flux[shown], err[shown], fmt='o', ms=4,
                     color='k', ecolor='k', elinewidth=1, capsize=2,
                     label='photometry')

        # The photosphere where the fit was not shown it, which is the same
        # statement for a point measured as an excess and one left out of both
        if beyond or left:
            aw = np.array([e['wave_um'] for e in beyond + left])
            am = np.array([e['model'] for e in beyond + left]) * PER_AA
            top.plot(aw, am, 'd', mfc='none', ms=7, mew=1.2, color=colour,
                     ls='none', label='photosphere, not fitted here')

        if left:
            lw = np.array([e['wave_um'] for e in left])
            lf = np.array([e['observed'] for e in left]) * PER_AA
            le = np.array([e['observed_err'] for e in left]) * PER_AA

            # Hollow, because it is the photometry marker and this is a
            # measurement the fit never saw
            top.errorbar(lw, lf, le, fmt='o', ms=5, mfc='none', color='0.45',
                         ecolor='0.45', elinewidth=1, capsize=2,
                         label='left out of both')

        if beyond:
            bw = np.array([e['wave_um'] for e in beyond])
            bf = np.array([e['observed'] for e in beyond]) * PER_AA
            be = np.array([e['observed_err'] for e in beyond]) * PER_AA
            bm = np.array([e['model'] for e in beyond]) * PER_AA

            top.errorbar(bw, bf, be, fmt='s', ms=5, color=EXCESS_COLOUR,
                         ecolor=EXCESS_COLOUR, elinewidth=1, capsize=2,
                         label='beyond the fit')

            # The excess alone, as a curve: unlike the photosphere it is a
            # function we have in closed form, so drawing it between the bands
            # claims nothing that was not fitted. Dashed, because on its own it
            # is a component and not a model of the measurement.
            span_um = np.geomspace(wave[-1], bw[-1] * 1.3, 200)
            curve = excess_flux(models, span_um)
            if curve is not None:
                top.plot(span_um, curve * PER_AA, '--', lw=1.3,
                         color=EXCESS_COLOUR, alpha=0.8,
                         label=_excess_label(models))

                # And the two of them together at each band, which is what the
                # measurement is to be read against
                top.plot(bw, bm + excess_flux(models, bw) * PER_AA,
                         'D', mfc='none', ms=10,
                         mew=1.4, color=EXCESS_COLOUR, ls='none',
                         label='photosphere + excess')

        top.set_xscale('log')
        top.set_yscale('log')
        top.set_ylabel(r'$F_\lambda$, erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$')
        top.set_title(f"{run['grid']}: "
                      rf"$T_{{\rm eff}}$ = {theta[0]:.0f} K, "
                      rf"$A_V$ = {theta[5]:.2f}, "
                      rf"$\chi^2$ = {summary['chi2']:.1f} on {len(bands)} bands",
                      fontsize=10)
        top.legend(fontsize=8.5, frameon=False)
        top.grid(alpha=0.15)

        # The band the fit was working to, which is why a three-sigma residual
        # against the catalogue error is not necessarily a bad fit
        envelope = jitter * model / total
        low.axhspan(-1, 1, color='0.85', zorder=0)
        low.vlines(wave[shown], -envelope[shown], envelope[shown], color=colour,
                   alpha=0.45, lw=2, zorder=1,
                   label=f'jitter, {jitter:.1%} of the model')
        low.axhline(0, color='0.4', lw=1, zorder=2)
        low.plot(wave[shown], residual[shown], 'o', ms=4, color='k', zorder=3)

        # Named where they are worth naming: every band labelled would be five
        # Pan-STARRS labels on top of each other, and the ones worth reading
        # are the ones the model misses
        # Alternating heights, since two adjacent bands that both miss - the
        # Pan-STARRS pair here - would otherwise write over each other; and
        # turned inwards near the right edge, where a name would run off
        turn = wave[0] * (wave[-1] / wave[0]) ** 0.75
        for n, (w, r, band) in enumerate(zip(wave[shown], residual[shown],
                                             [b for b, keep in zip(bands, shown)
                                              if keep])):
            if abs(r) >= 3:
                inward = w > turn
                low.annotate(band, (w, r), fontsize=7, color='0.3',
                             textcoords='offset points', va='center',
                             ha='right' if inward else 'left',
                             xytext=(-5 if inward else 5, 5 if n % 2 else -10))

        # The excess sigmas belong in this panel, but they are tens where the
        # fitted ones are ones, and letting them set the scale would flatten
        # the residuals the photosphere is judged on. So the scale stays with
        # the fit, and an excess off the top is marked at the edge by how far.
        if beyond or left:
            span = max(3.0, 1.25 * float(np.max(np.abs(residual[shown]))),
                       1.25 * float(np.max(envelope[shown])))
            low.set_ylim(-span, span)

            def mark(entries, colour, marker, face=None):
                for e in entries:
                    sigma = e['sigma']
                    inside = min(max(sigma, -span * 0.92), span * 0.92)
                    low.plot([e['wave_um']], [inside], marker=marker, ms=5,
                             color=colour, mfc=face or colour, zorder=4,
                             clip_on=abs(sigma) <= span)
                    if abs(sigma) > span:
                        # Inside the axes, since the panel above starts where
                        # this one ends and there is nowhere outside to write
                        low.annotate(f'{sigma:+.0f}', (e['wave_um'], inside),
                                     fontsize=7, color=colour,
                                     textcoords='offset points', ha='center',
                                     va='bottom' if sigma < 0 else 'top',
                                     xytext=(0, 6 if sigma < 0 else -6))

            mark(left, '0.45', 'o', face='none')
            mark(beyond, EXCESS_COLOUR, 's')

            # And what the fitted shape leaves, drawn as the open diamond the
            # total model carries above, so the two panels say the same thing
            # with the same marker: the filled square is the excess, the open
            # diamond is what is left of it once the shape is taken off.
            mark([dict(e, sigma=e['fit_sigma']) for e in beyond
                  if e.get('fit_sigma') is not None],
                 EXCESS_COLOUR, 'D', face='none')

        if beyond and any(e.get('fit_sigma') is not None for e in beyond):
            low.plot([], [], 's', ms=5, color=EXCESS_COLOUR, ls='none',
                     label='from the photosphere')
            low.plot([], [], 'D', ms=5, mfc='none', color=EXCESS_COLOUR,
                     ls='none', label='from it and the excess')

        # The XP bins, last and without a say in the scale. They are a
        # different measurement of the same star against the same model, and
        # they can be tens of sigma out where the fit itself is not - letting
        # them stretch this axis would flatten the residuals the fit is judged
        # on, which are the reason the panel is here.
        if checked and checked.get('bins'):
            usable = [b for b in checked['bins'] if b['ratio'] is not None]
            span = low.get_ylim()
            low.plot([b['wave_um'] for b in usable],
                     [b['sigma'] for b in usable], '.', ms=3.5,
                     color=XP_COLOUR, alpha=0.9, zorder=2,
                     label=f"Gaia XP, {checked['bin_nm']:.0f} nm bins"
                           + (' (fitted)' if checked.get('n_fitted') else ''))
            low.set_ylim(span)

        # Along the bottom, which is the one strip of this panel that is
        # reliably empty - the residuals it draws cluster about zero
        low.legend(fontsize=7.5, frameon=False, loc='lower center', ncol=3)

        low.set_xlabel(r'Wavelength, $\mu$m')
        low.set_ylabel(r'Residual, $\sigma$', fontsize=9)
        low.grid(alpha=0.15)
        low.tick_params(labelsize=8)
        top.tick_params(labelsize=8)

    return filename


def _spread(samples, column):
    """Whether a parameter varied at all - a fixed one has nothing to draw."""
    values = samples[:, column]
    return np.isfinite(values).all() and np.ptp(values) > 0


def draw_corner(run, path, name=None):
    """The joint posterior of one grid, pair by pair.

    Per grid rather than for all of them together: what two grids disagree
    about is a systematic, and merging their samples into one cloud would
    draw it as though it were a wider measurement.

    The point of drawing it at all is the shape. A reddened hot star has a
    long curved ridge in temperature against extinction, and no pair of error
    bars says so - which is exactly the case where a single number for the
    temperature is worth least.
    """
    import corner

    from stdpipe import plots

    columns = [i for i, _ in enumerate(PARAMETERS) if _spread(run['samples'], i)]
    if len(columns) < 2:
        return None

    axis = run.get('axis_logg') or 'logg'
    labels = [(axis if PARAMETERS[i] == 'logg' and axis != 'logg'
               else LABELS[PARAMETERS[i]][0]) for i in columns]
    data = run['samples'][:, columns]

    filename = os.path.join(path, f'corner_{name or run["grid"]}.png')
    side = 2.0 * len(columns)

    with plots.figure_saver(filename, figsize=(side, side),
                            tight_layout=False) as fig:
        # corner needs the axes to exist before it will draw into a figure of
        # someone else's making
        fig.subplots(len(columns), len(columns))
        corner.corner(data, labels=labels, fig=fig,
                      quantiles=[0.16, 0.5, 0.84], show_titles=True,
                      title_fmt='.4g', title_kwargs={'fontsize': 9},
                      label_kwargs={'fontsize': 9},
                      color=GRID_COLOURS[0], plot_datapoints=True,
                      fill_contours=False, plot_density=False)

    return filename


def draw_histograms(runs, path, name='histograms.png'):
    """Every grid's answer for each parameter, drawn over each other.

    This is the figure that shows what the grids disagree about, which is the
    systematic we decline to average away: where their histograms sit on top
    of each other the atmosphere physics does not matter for that parameter,
    and where they are side by side it does.
    """
    from matplotlib.ticker import MaxNLocator
    from stdpipe import plots

    shown = [p for p in PARAMETERS + ('theta_mas', 'lum_lsun')
             if any(_parameter_values(run, p) is not None for run in runs)]
    if not shown:
        return None

    columns = 4
    rows = int(np.ceil(len(shown) / columns))
    filename = os.path.join(path, name)

    with plots.figure_saver(filename, figsize=(3.2 * columns, 2.6 * rows)) as fig:
        axes = fig.subplots(rows, columns, squeeze=False).ravel()

        for ax, parameter in zip(axes, shown):
            label, fmt = LABELS[parameter]
            titles = []

            for n, run in enumerate(runs):
                values = _parameter_values(run, parameter)
                if values is None or not np.ptp(values) > 0:
                    continue

                colour = GRID_COLOURS[n % len(GRID_COLOURS)]
                ax.hist(values, bins=40, density=True, histtype='stepfilled',
                        alpha=0.35, color=colour, label=run['grid'])

                # The median and the central 68 per cent, dashed, as the
                # corner plots mark them - so the two figures are read the
                # same way
                lo, mid, hi = np.percentile(values, [16, 50, 84])
                for level in (lo, mid, hi):
                    ax.axvline(level, color=colour, lw=1.0, ls='--', alpha=0.9)

                titles.append(_quantile_title(mid, lo, hi, fmt,
                                              run['grid'] if len(runs) > 1
                                              else None))

            ax.set_xlabel(label, fontsize=9)
            # The numbers above the panel, written the way the corner plots
            # write them. Named by grid where there is more than one, since
            # two medians one above the other say nothing about which is which.
            if titles:
                ax.set_title('\n'.join(titles), fontsize=8.5)
            ax.tick_params(labelsize=8)
            ax.set_yticks([])

            # A luminosity runs to six figures, and four of those tick labels
            # side by side run into each other. Few ticks, and an exponent
            # where the numbers are long enough to need one.
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 4),
                                useMathText=True)
            ax.xaxis.get_offset_text().set_fontsize(8)

        for ax in axes[len(shown):]:
            ax.set_visible(False)

        # One legend for the figure: the grids are the same in every panel
        handles, labels_ = axes[0].get_legend_handles_labels()
        if len(handles) > 1:
            fig.legend(handles, labels_, loc='lower right', fontsize=9,
                       frameon=False)

    return filename


def _quantile_title(median, lo, hi, fmt, grid=None):
    """The median with its one-sigma bounds, as corner writes a title.

    Mathtext rather than plain text, so the two figures of a run carry the
    same notation and a reader moving between them does not have to translate.
    """
    # A parameter written with a forced sign - [Fe/H] is - wants it on the
    # value and not on the bounds, which are distances and always positive:
    # otherwise the title reads "+0.46" behind a "+" of its own.
    spread = fmt.replace(':+', ':')

    text = (f'${fmt.format(median)}'
            f'^{{+{spread.format(hi - median)}}}'
            f'_{{-{spread.format(median - lo)}}}$')

    return f'{grid}: {text}' if grid else text


def _parameter_values(run, parameter):
    """One parameter's samples, sampled or derived, or None if it has none."""
    if parameter in PARAMETERS:
        column = PARAMETERS.index(parameter)
        return run['samples'][:, column] if _spread(run['samples'], column) else None

    values = derived(run['samples']).get(parameter)
    return values if values is not None and np.isfinite(values).all() else None
