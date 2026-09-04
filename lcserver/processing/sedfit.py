"""Fitting a stellar photosphere to the catalogue photometry of a target.

What is fitted is one model atmosphere, reddened and put at a distance:

    interpolate a model grid at (Teff, log g, [Fe/H]), redden by Av, dilute by
    (R/d)^2, and compare with the observed flux band by band.

The grids come from astroARIADNE - each is an HDF5 cube of per-filter fluxes on
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

import os

import numpy as np

import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy import stats

import extinction

from .utils import SourceError


# Solar radius and parsec in cm, for the (R/d)^2 dilution
R_SUN = 6.957e10
PARSEC = 3.0856775814913673e18

# Speed of light in Angstrom/s, for f_lambda <-> f_nu
C_AA = 2.99792458e18

AB_ZERO_JY = 3631.0


class Grid:
    """One atmosphere grid, read straight out of its HDF5 cube.

    The file carries three axes and a dense (n_logg, n_teff, n_feh, n_filter)
    block of fluxes, which is all an interpolator needs. Bands a grid does not
    reach are NaN at every node; those are reported so a caller can decline to
    fit them rather than discover it as a likelihood of -inf.
    """

    def __init__(self, path, name=None):
        self.name = name or str(path)

        with h5py.File(path, 'r') as h:
            self.logg = np.asarray(h['logg'][:], dtype=float)
            self.teff = np.asarray(h['teff'][:], dtype=float)
            self.feh = np.asarray(h['feh'][:], dtype=float)
            cube = np.asarray(h['flux'][:], dtype=float)
            names = [b.decode() if isinstance(b, bytes) else str(b)
                     for b in h['filters'][:]]

        self.filters = names
        self.column = {b: i for i, b in enumerate(names)}
        self._interp = RegularGridInterpolator(
            (self.logg, self.teff, self.feh), cube,
            bounds_error=False, fill_value=np.nan)

        finite = np.isfinite(cube).any(axis=(0, 1, 2))
        self.covers = frozenset(b for b, i in self.column.items() if finite[i])

    @property
    def limits(self):
        """The box the grid is defined over, as {parameter: (low, high)}."""
        return {'teff': (self.teff.min(), self.teff.max()),
                'logg': (self.logg.min(), self.logg.max()),
                'feh': (self.feh.min(), self.feh.max())}

    def flux(self, teff, logg, feh, columns):
        """Surface flux in the given filter columns, erg/s/cm2/um."""
        return self._interp(np.array([[logg, teff, feh]]))[0][columns]


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

def fit(bands, wave_um, flux, flux_err, grid, priors,
        nlive=500, dlogz=0.5, seed=None, verbose=True):
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

    sampler = dynesty.NestedSampler(
        log_likelihood, prior_transform, len(PARAMETERS),
        nlive=nlive, bound='multi', sample='rwalk',
        rstate=np.random.default_rng(seed))
    sampler.run_nested(dlogz=dlogz, print_progress=verbose)

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

    for name in PARAMETERS + ('theta_mas', 'lum_lsun'):
        row = summary.get(name)
        if not row:
            continue

        fmt = LABELS.get(name, (None, '{:.4g}'))[1]
        value = best.get(name)

        log(f"  {name:<10}"
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


def excess_rows(rows, run, grids):
    """The points an excess would be measured on, one per band.

    Every point redward of the reddest band that was fitted, which the grids
    can predict a photosphere for. Redward of the fit is the whole of the
    condition: a point set aside for any other reason - a second catalogue for
    a band already taken, one the reader turned off in the optical - is not an
    excess, it is a point that was not used.
    """
    fitted = set(run['bands'])
    edge = max(run['wave_um'])

    # A band is quantified only where every grid in play both carries it and is
    # believed there, which is the intersection of what they reach
    reach = None
    for grid in grids:
        covers = {b for b in grid.covers
                  if _within_limit(grid.name, b, rows)}
        reach = covers if reach is None else (reach & covers)

    out, seen = [], set()
    for row in rows:
        band = row['band']
        if (band is None or band in fitted or band in seen
                or row['flux'] is None or row['wave_um'] is None
                or not row['wave_um'] > edge):
            continue

        seen.add(band)
        out.append(dict(row, quantified=band in (reach or ())))

    out.sort(key=lambda r: r['wave_um'])
    return out


def _within_limit(name, band, rows):
    """Whether a grid is believed at a band, by the wavelength it stops at."""
    limit = GRID_IR_LIMIT.get(name)
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


def log_excess(table, models, log):
    """The excess as a table, and what shape it came out as."""
    log('\n  infrared excess, against the photosphere the fit predicts')
    log(f"    {'band':<16}{'lam um':>8}{'observed':>11}{'photosphere':>12}"
        f"{'ratio':>8}{'sigma':>8}")

    for entry in table:
        if entry['quantified']:
            log(f"    {entry['band'] or '':<16}{entry['wave_um']:8.2f}"
                f"{entry['observed']:11.3e}{entry['model']:12.3e}"
                f"{entry['ratio']:8.2f}{entry['sigma']:+8.1f}")
        else:
            log(f"    {entry['band'] or '':<16}{entry['wave_um']:8.2f}"
                f"{entry['observed']:11.3e}{'-':>12}{'-':>8}{'-':>8}")

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

# Bands the grids carry a column for but did not model - the flux there was
# extrapolated when the grid was built. They belong to an infrared excess
# rather than to a photosphere, and are never fitted.
NOT_PHOTOSPHERE = ('WISE_RSR_W3', 'WISE_RSR_W4', 'HERSCHEL_PACS_BLUE',
                   'HERSCHEL_PACS_GREEN', 'HERSCHEL_PACS_RED',
                   'SPITZER_IRAC_58', 'SPITZER_IRAC_80')

# How far out a grid may be believed, in microns. A cube carries a column for
# every filter, but some grids were built from spectra that stop short of the
# reddest of them and the flux there is an extrapolation. Only the grids that
# need a limit have one; the rest are trusted to their whole reach.
GRID_IR_LIMIT = {'koester': 3.0, 'ck04': 8.5, 'kurucz': 8.5}

# Which grid file holds which grid
GRID_FILES = {
    'btsettl': 'BTSettl', 'btcond': 'BTCond', 'btnextgen': 'BTNextGen',
    'tlusty': 'TLUSTY', 'phoenix': 'Phoenixv2', 'ck04': 'CK04',
    'kurucz': 'Kurucz', 'bosz': 'BOSZ', 'coelho': 'Coelho',
    'koester': 'Koester', 'newera': 'NewEra', 'sphinx': 'SPHINX',
    'atmo': 'ATMO2020', 'sonora': 'Sonora', 'stagger': 'Stagger',
    'btdusty': 'BTDusty',
}


def grids_dir():
    """Where the model cubes live.

    Ours by configuration if the SEDFIT_GRIDS setting says so, and otherwise
    wherever astroARIADNE was installed - the package is a dependency for its
    grids and for pyphot's filter profiles, and for nothing else.
    """
    from django.conf import settings

    configured = getattr(settings, 'SEDFIT_GRIDS', None)
    if configured:
        return configured

    from astroARIADNE.config import gridsdir

    return gridsdir


def load_grid(name):
    """One grid by name."""
    return Grid(f'{grids_dir()}/{GRID_FILES[name.lower()]}.h5', name=name.lower())


def read_sed_points(path, points=None, exclude=None, extra=None,
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

    Every row is returned, with ``used`` saying whether it will be fitted and
    ``note`` saying why not, so a caller can show the whole file and mark what
    it did with each point.
    """
    from astropy.table import Table, vstack
    from astroARIADNE.phot_utils import _get_filter

    points = set(points) if points else None
    exclude = set(exclude or ())

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
        entry = {'id': comment, 'band': band, 'used': False, 'note': None,
                 'wave_um': None, 'wave_drawn_um': None,
                 'flux': None, 'err': None}
        rows.append(entry)

        if band is None:
            entry['note'] = 'no model filter of this name'
            continue

        pivot = _get_filter(band).lpivot.to('AA').value
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

        if band in NOT_PHOTOSPHERE:
            entry['note'] = 'infrared excess, not photosphere'
        elif comment in exclude:
            entry['note'] = 'excluded'
        elif points is not None and comment not in points:
            entry['note'] = 'not selected'
        elif band in seen:
            entry['note'] = 'another point already covers this band'
        else:
            seen.add(band)
            entry['used'] = True

    rows.sort(key=lambda r: (r['wave_um'] is None, r['wave_um'] or 0))
    return rows


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
        sed, points=selection.get('points'), exclude=selection.get('exclude'),
        extra=(read_extra_points(basepath)
               if selection.get('extra', True) else None),
        err_floor=options.get('err_floor', 0.03),
        err_unknown=options.get('err_unknown', 0.05))

    used = [r for r in rows if r['used']]
    log(f"{len(used)} of {len(rows)} points from {source}")
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

    runs, summaries = [], {}
    for name in options.get('grids') or ['btsettl']:
        grid = load_grid(name)
        priors = default_priors(grid, distance, distance_err, av_max)
        # The temperature prior is deliberately the same whatever grid is
        # loaded, so that a grid's extent cannot become the answer
        priors['teff'] = tuple(options.get('teff_prior')
                               or ('loguniform', 2000.0, 70000.0))
        if options.get('logg_prior'):
            priors['logg'] = tuple(options['logg_prior'])

        missing = sorted({b for b in bands if b not in grid.covers})
        if missing:
            log(f"\n{name}: no model flux for {', '.join(missing)} - skipped")
            continue

        log(f"\nfitting {name}: Teff {grid.teff.min():.0f}-{grid.teff.max():.0f} K")
        run = fit(bands, wave, flux, err, grid, priors,
                  nlive=options.get('nlive', 500), seed=options.get('seed', 0),
                  verbose=False)
        run['wave_drawn_um'] = drawn
        summary = summarise(run, grid)
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
                beyond = excess_rows(rows, run, [grid])
                if beyond:
                    summary['excess'] = quantify_excess(beyond, run, grid)
                    summary['excess_models'] = compare_excess(
                        summary['excess'], run, grid, summary)
                    log_excess(summary['excess'], summary['excess_models'], log)
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
    with open(os.path.join(outpath, 'fit.json'), 'w') as f:
        json.dump(result, f, indent=1, default=float)
    for run, grid in runs:
        np.save(os.path.join(outpath, f'samples_{grid.name}.npy'), run['samples'])

    # Drawn last, and never allowed to lose a run: the fit is the thing, and a
    # figure that will not render is not a reason to throw away an hour of
    # sampling. Whichever ones worked are on disk and the viewer finds them.
    if options.get('figures', True):
        for run, grid in runs:
            try:
                draw_sed(run, grid, summaries[grid.name], outpath)
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
    from astroARIADNE.config import filter_names
    from astroARIADNE.phot_utils import _get_filter

    bands = []
    for band in filter_names:
        if band in NOT_PHOTOSPHERE:
            continue
        try:
            pivot = float(_get_filter(band).lpivot.to('AA').value)
        except Exception:
            continue
        bands.append({'band': band, 'wavelength': pivot,
                      'system': 'AB' if _is_ab(band) else 'Vega'})

    return sorted(bands, key=lambda b: b['wavelength'])


def _is_ab(band):
    """Whether this band's magnitudes are AB rather than Vega.

    Asked of the photometry library rather than decided here: it splits the two
    on a list of name prefixes that has moved before now - SkyMapper was on the
    Vega branch until mid-2026 - so a copy kept here would go quietly stale.
    Convert magnitude zero and see which zero point comes back.
    """
    from astroARIADNE import phot_utils

    return not np.isclose(phot_utils.mag_to_flux(0.0, 0.0, band)[0],
                          phot_utils.get_zero_flux(band))


def magnitude_to_flux(band, mag, mag_err=None):
    """A magnitude in one band as f_lambda at its pivot, erg/s/cm2/A.

    Through f_nu and the band's zero point in Jansky, which is exact for an AB
    band and as good as the library's Vega spectrum for the others. The pivot
    wavelength is the one that makes <f_lambda> = <f_nu> c / lambda^2 true for
    a photon-counting filter, which is the convention the grids are on.
    """
    from astroARIADNE.phot_utils import _get_filter

    pivot = float(_get_filter(band).lpivot.to('AA').value)
    zero = 3631.0 if _is_ab(band) else float(_get_filter(band).Vega_zero_Jy.value)

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
        from astroARIADNE.phot_utils import _get_filter

        wavelength = float(_get_filter(band).lpivot.to('AA').value)
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
    from astroARIADNE.phot_utils import _get_filter

    try:
        return float(_get_filter(band).width.to('AA').value)
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


def _excess_curve(models, wave_um):
    """The fitted excess shape over a wavelength range, or None.

    Drawn at the best node rather than at the marginal means, for the reason
    the photosphere is drawn at one posterior row: a temperature and an
    amplitude taken from two different places on the likelihood are not a
    shape that fits anything.
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

    return wave, flux


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


def draw_sed(run, grid, summary, path, name=None, colour=GRID_COLOURS[0]):
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
    models = summary.get('excess_models')

    filename = os.path.join(path, f'sed_{name or run["grid"]}.png')

    with plots.figure_saver(filename, figsize=(8, 6), tight_layout=False) as fig:
        top, low = fig.subplots(2, 1, sharex=True,
                                gridspec_kw={'height_ratios': [3, 1],
                                             'hspace': 0.06})

        # Per band rather than a shaded curve across them: what the fit
        # produced is a flux in each filter, and a band drawn between them
        # would be claiming a spectrum it never computed
        top.vlines(wave, lo, hi, color=colour, alpha=0.35, lw=6,
                   label='posterior, central 68%')
        top.plot(wave, model, 'D', mfc='none', ms=9, mew=1.6, color=colour,
                 ls='none', label=f'{run["grid"]} at the plot row')
        # Behind the catalogue error and unlabelled: it is the same statement
        # the residual panel makes, and it is made there with room to say it
        top.errorbar(wave, flux, widened, fmt='none', ecolor='0.75',
                     elinewidth=3, capsize=0)
        top.errorbar(wave, flux, err, fmt='o', ms=4, color='k', ecolor='k',
                     elinewidth=1, capsize=2, label='photometry')

        if beyond:
            bw = np.array([e['wave_um'] for e in beyond])
            bf = np.array([e['observed'] for e in beyond]) * PER_AA
            be = np.array([e['observed_err'] for e in beyond]) * PER_AA
            bm = np.array([e['model'] for e in beyond]) * PER_AA

            top.plot(bw, bm, 'd', mfc='none', ms=7, mew=1.2, color=colour,
                     ls='none', label='photosphere, not fitted here')
            top.errorbar(bw, bf, be, fmt='s', ms=5, color=EXCESS_COLOUR,
                         ecolor=EXCESS_COLOUR, elinewidth=1, capsize=2,
                         label='beyond the fit')

            # The excess alone, as a curve: unlike the photosphere it is a
            # function we have in closed form, so drawing it between the bands
            # claims nothing that was not fitted. Dashed, because on its own it
            # is a component and not a model of the measurement.
            curve = _excess_curve(models, np.geomspace(wave[-1], bw[-1] * 1.3, 200))
            if curve is not None:
                top.plot(curve[0], curve[1] * PER_AA, '--', lw=1.3,
                         color=EXCESS_COLOUR, alpha=0.8,
                         label=_excess_label(models))

                # And the two of them together at each band, which is what the
                # measurement is to be read against
                at = _excess_curve(models, bw)
                top.plot(bw, bm + at[1] * PER_AA, 'D', mfc='none', ms=10,
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
        low.vlines(wave, -envelope, envelope, color=colour, alpha=0.45, lw=2,
                   zorder=1, label=f'jitter, {jitter:.1%} of the model')
        low.axhline(0, color='0.4', lw=1, zorder=2)
        low.plot(wave, residual, 'o', ms=4, color='k', zorder=3)

        # Named where they are worth naming: every band labelled would be five
        # Pan-STARRS labels on top of each other, and the ones worth reading
        # are the ones the model misses
        # Alternating heights, since two adjacent bands that both miss - the
        # Pan-STARRS pair here - would otherwise write over each other; and
        # turned inwards near the right edge, where a name would run off
        turn = wave[0] * (wave[-1] / wave[0]) ** 0.75
        for n, (w, r, band) in enumerate(zip(wave, residual, bands)):
            if abs(r) >= 3:
                left = w > turn
                low.annotate(band, (w, r), fontsize=7, color='0.3',
                             textcoords='offset points', va='center',
                             ha='right' if left else 'left',
                             xytext=(-5 if left else 5, 5 if n % 2 else -10))

        # The excess sigmas belong in this panel, but they are tens where the
        # fitted ones are ones, and letting them set the scale would flatten
        # the residuals the photosphere is judged on. So the scale stays with
        # the fit, and an excess off the top is marked at the edge by how far.
        if beyond:
            span = max(3.0, 1.25 * float(np.max(np.abs(residual))),
                       1.25 * float(np.max(envelope)))
            low.set_ylim(-span, span)

            for e in beyond:
                sigma = e['sigma']
                inside = min(max(sigma, -span * 0.92), span * 0.92)
                low.plot([e['wave_um']], [inside], marker='s', ms=5,
                         color=EXCESS_COLOUR, zorder=4,
                         clip_on=abs(sigma) <= span)
                if abs(sigma) > span:
                    # Inside the axes, since the panel above starts where this
                    # one ends and there is nowhere outside to write
                    low.annotate(f'{sigma:+.0f}', (e['wave_um'], inside),
                                 fontsize=7, color=EXCESS_COLOUR,
                                 textcoords='offset points', ha='center',
                                 va='bottom' if sigma < 0 else 'top',
                                 xytext=(0, 6 if sigma < 0 else -6))

        low.legend(fontsize=7.5, frameon=False, loc='upper left')

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

    labels = [LABELS[PARAMETERS[i]][0] for i in columns]
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
