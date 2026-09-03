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

    Ours by configuration if SEDFIT_GRIDS says so, and otherwise wherever
    astroARIADNE was installed - the package is a dependency for its grids and
    for pyphot's filter profiles, and for nothing else.
    """
    import os

    configured = os.environ.get('SEDFIT_GRIDS')
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
    """Observed against model at the row the fit is drawn at."""
    theta = np.array([summary['best'][p] for p in PARAMETERS])
    res, model = residuals(result, theta, grid)

    drawn = result.get('wave_drawn_um')
    if drawn is None:
        drawn = result['wave_um']

    return [{'band': b, 'wave_um': float(w), 'wave_drawn_um': float(d),
             'observed': float(f), 'model': float(m), 'residual': float(r)}
            for b, w, d, f, m, r in zip(result['bands'], result['wave_um'],
                                        drawn, result['flux'], model, res)]


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
    residual = (flux - model) / err

    # What the rest of the posterior would have drawn. Two hundred rows is
    # enough for a 16-84 envelope and costs an interpolation each.
    draws = run['samples']
    if len(draws) > 200:
        draws = draws[np.random.default_rng(0).choice(len(draws), 200,
                                                      replace=False)]
    cloud = np.array([model_flux(row, run['columns'], run['ext_unit'], grid)[order]
                      for row in draws]) * PER_AA
    lo, hi = np.nanpercentile(cloud, [16, 84], axis=0)

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
        envelope = jitter * model / err
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
