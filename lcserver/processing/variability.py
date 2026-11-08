"""How much a light curve varies, and on what timescales, period or none.

The computation behind the light curve viewer's variability panel: indices
of each survey's variation - its significance, its amplitude above the noise,
how smooth it is from one point to the next, which way it trends - and the
structure function, the scatter of the differences of pairs of points against
the time between them, with the damped random walk fitted to it for a
timescale and an amplitude.

Like `periods`, everything here takes plain arrays and knows nothing of
targets, files or requests.
"""

import numpy as np
from scipy import stats
from scipy.optimize import least_squares


# Points closer than this, in days, are taken as one visit by Stetson's index
# and paired in it - a night's points, not a season's
VARIABILITY_PAIR_DAYS = 0.02

# Days in a year, for the slope of the trend
YEAR_DAYS = 365.25

# Lags of the structure function, in bins so many per decade, from the
# shortest lag the data have - but no shorter than this - to the span
SF_BINS_PER_DECADE = 5
SF_MIN_LAG = 1e-3

# A bin with fewer pairs than this, or pairs of fewer distinct points, says
# too little to be drawn
SF_MIN_PAIRS = 10

# Pairs in a bin are taken all of them up to so many, and drawn at random from
# all of them, each pair as likely as any other, beyond: TESS has a hundred
# million pairs a sector
SF_MAX_PAIRS = 200000

# The scatter of the differences in a bin is that of a Gaussian of the same
# interquartile range - robust to the outliers the archives have plenty of
IQR_SIGMA = 0.7413

# The uncertainty of that width, as a fraction of it, times the square root
# of the number of differences
SIGMA_G_ERROR = 1.069

# Below this many bins of the structure function, no random walk is fitted
SF_MIN_FIT_BINS = 4

# A timescale is only measured if the data span it many times over; longer
# than this fraction of the span, it is a lower limit
DRW_TAU_SPAN = 0.1


def variability_indices(t, y, dy, magnitudes=True):
    """Indices of the variation of one series, or of a survey pooled.

    `chi2` and `log10_p` say whether it varies at all - the reduced
    chi-squared about the weighted mean, and the chance of one as large from
    the errors alone; `rms` is the scatter left once the errors are taken off
    (Vaughan et al. 2003), with its uncertainty, in magnitudes or, for flux,
    as a fraction of the mean. `eta` is von Neumann's ratio of the mean
    square of successive differences to the variance: about 2 for noise,
    near zero for a variation slow against the sampling. Stetson's `J` is
    near zero for noise and positive for a variation that points close in
    time share; `K` about 0.80 for a Gaussian, lower for a light curve that
    spends its time at the extremes. `slope` is the trend, per year.
    """
    t, y, dy = (np.asarray(_, dtype=float) for _ in (t, y, dy))
    order = np.argsort(t)
    t, y, dy = t[order], y[order], dy[order]
    n = len(t)

    out = {'n': int(n), 'span': float(t[-1] - t[0]) if n else 0.0}
    if n < 3:
        return out

    w = 1.0 / dy ** 2
    mean = np.sum(w * y) / np.sum(w)
    chi2 = np.sum(w * (y - mean) ** 2)
    var = np.var(y, ddof=1)
    noise = np.mean(dy ** 2)

    # The excess variance and its uncertainty, eq. 11 of Vaughan et al.
    excess = var - noise
    excess_err = np.sqrt(2.0 / n * noise ** 2 + 4.0 / n * noise * max(excess, 0.0))
    scale = 1.0 if magnitudes else abs(mean)
    rms = np.sqrt(max(excess, 0.0)) / scale
    rms_err = (excess_err / (2 * np.sqrt(excess)) if excess > 0 else np.sqrt(excess_err)) / scale

    # Stetson's J over visits: neighbours closer than VARIABILITY_PAIR_DAYS
    # are paired, each point at most once, and the rest stand alone
    delta = np.sqrt(n / (n - 1)) * (y - mean) / dy
    products = []
    k = 0
    while k < n:
        if k + 1 < n and t[k + 1] - t[k] < VARIABILITY_PAIR_DAYS:
            products.append(delta[k] * delta[k + 1])
            k += 2
        else:
            products.append(delta[k] ** 2 - 1)
            k += 1
    products = np.array(products)
    stetson_j = float(np.mean(np.sign(products) * np.sqrt(np.abs(products))))
    stetson_k = float(np.mean(np.abs(delta)) / np.sqrt(np.mean(delta ** 2)))

    out.update({
        'mean': float(mean),
        'median': float(np.median(y)),
        'std': float(np.sqrt(var)),
        'noise': float(np.sqrt(noise)),
        'chi2': float(chi2 / (n - 1)),
        'log10_p': _log10_p(chi2, n - 1),
        'rms': float(rms),
        'rms_err': float(rms_err),
        # Whether the excess is two of its sigma clear of none
        'rms_detected': bool(excess > 2 * excess_err),
        'eta': float(np.sum(np.diff(y) ** 2) / (n - 1) / var) if var > 0 else None,
        'stetson_j': stetson_j,
        'stetson_k': stetson_k,
        'skewness': float(stats.skew(y)),
        'iqr': float(np.subtract(*np.percentile(y, [75, 25]))),
        'amplitude': float(np.subtract(*np.percentile(y, [95, 5]))),
    })

    # The slope by weighted least squares, its error scaled by the scatter
    # about the line - which is mostly the variation itself, not the noise
    if out['span'] > 0:
        x = (t - t.mean()) / YEAR_DAYS
        design = np.vstack([np.ones(n), x]).T * np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(design, y * np.sqrt(w), rcond=None)
        resid = y * np.sqrt(w) - design @ coef
        cov = np.linalg.inv(design.T @ design) * max(1.0, np.sum(resid ** 2) / (n - 2))
        out['slope'] = float(coef[1] / scale)
        out['slope_err'] = float(np.sqrt(cov[1, 1]) / scale)

    return out


def _log10_p(chi2, dof):
    """Log10 of the chance of a chi-squared this large from noise alone.

    Where the exact tail underflows, from the normal one that the cube root
    of chi-squared approximately follows (Wilson & Hilferty 1931).
    """
    logp = stats.chi2.logsf(chi2, dof)
    if not np.isfinite(logp):
        a = 2.0 / (9 * dof)
        logp = stats.norm.logsf(((chi2 / dof) ** (1 / 3) - 1 + a) / np.sqrt(a))
    return float(logp / np.log(10))


def lag_bins(span, shortest):
    """Edges of the bins of lag, logarithmic, from `shortest` to `span`."""
    lo = np.floor(np.log10(max(shortest, SF_MIN_LAG)) * SF_BINS_PER_DECADE) / SF_BINS_PER_DECADE
    hi = np.ceil(np.log10(span) * SF_BINS_PER_DECADE) / SF_BINS_PER_DECADE
    return 10 ** np.arange(lo, hi + 0.5 / SF_BINS_PER_DECADE, 1.0 / SF_BINS_PER_DECADE)


def lag_pairs(t, lo, hi, rng, max_pairs=SF_MAX_PAIRS):
    """Pairs (i, j) of points of sorted `t` with lo <= t[j] - t[i] < hi.

    All of them if there are no more than `max_pairs`, and otherwise as many
    drawn at random, every pair as likely: the first point by how many
    partners it has in the bin, the second evenly among them.
    """
    first = np.searchsorted(t, t + lo, side='left')
    last = np.searchsorted(t, t + hi, side='left')
    count = np.maximum(last - first, 0)
    total = int(count.sum())
    if not total:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    if total <= max_pairs:
        i = np.repeat(np.arange(len(t)), count)
        start = np.repeat(np.cumsum(count) - count, count)
        j = np.repeat(first, count) + np.arange(total) - start
    else:
        i = rng.choice(len(t), size=max_pairs, p=count / total)
        j = first[i] + np.floor(rng.random(max_pairs) * count[i]).astype(int)

    return i, j


def structure_function(sets, seed=0):
    """The structure function of a survey, from pairs within each of its series.

    `sets` are (t, y, dy) of the series pooled, whose pairs are taken within
    each and never across - each is calibrated, or normalized, on its own.
    In every bin of lag the scatter of the differences is the robust one of
    their interquartile range, and the noise of the pair, the sum of the
    squares of its errors, is taken off it in quadrature: what is left is the
    variation of the star alone.

    Returns, for each bin with enough pairs, the median lag, the `sf` and its
    uncertainty, the number of pairs, and `detected` - whether it is two
    sigma clear of the noise, as far as the noise alone would scatter it. Where it is not, `sf` is an upper limit, two
    sigma above the measured value. `sf2` and `sf2_err` are the square, with
    noise taken off and negative where the noise is all there is, which is
    what the random walk is fitted to.
    """
    rng = np.random.default_rng(seed)
    sorted_sets = []
    for t, y, dy in sets:
        order = np.argsort(t)
        sorted_sets.append((np.asarray(t, float)[order], np.asarray(y, float)[order],
                            np.asarray(dy, float)[order]))

    shortest = min((np.min(np.diff(t)[np.diff(t) > 0]) for t, _, _ in sorted_sets
                    if np.any(np.diff(t) > 0)), default=None)
    span = max((t[-1] - t[0] for t, _, _ in sorted_sets), default=0.0)
    if shortest is None or not span > 0:
        return []

    out = []
    edges = lag_bins(span, shortest)
    for lo, hi in zip(edges[:-1], edges[1:]):
        lags, diffs, noise, distinct = [], [], [], 0
        for t, y, dy in sorted_sets:
            i, j = lag_pairs(t, lo, hi, rng)
            if not len(i):
                continue
            lags.append(t[j] - t[i])
            diffs.append(y[j] - y[i])
            noise.append(dy[i] ** 2 + dy[j] ** 2)
            distinct += len(np.union1d(i, j))
        if not lags:
            continue
        lags, diffs, noise = (np.concatenate(_) for _ in (lags, diffs, noise))
        npairs = len(lags)
        if npairs < SF_MIN_PAIRS or distinct < SF_MIN_PAIRS:
            continue

        observed = IQR_SIGMA * np.subtract(*np.percentile(diffs, [75, 25]))
        sf2 = observed ** 2 - np.mean(noise)
        # Pairs sharing points are not independent: the error is that of as
        # many differences as the bin has distinct points, at most, and of a
        # width from their interquartile range that many give to 1.069/sqrt(n)
        # of itself (Ivezic et al. 2014). Whether there is variation at all is
        # asked of the noise alone - of how far its own width would scatter
        # - so that a few points far above it are not taken for none
        n_eff = np.sqrt(min(npairs, distinct))
        sf2_err = 2 * SIGMA_G_ERROR * observed ** 2 / n_eff
        detected = sf2 > 2 * 2 * SIGMA_G_ERROR * np.mean(noise) / n_eff
        if detected:
            sf, sf_err = np.sqrt(sf2), sf2_err / (2 * np.sqrt(sf2))
        else:
            sf, sf_err = np.sqrt(max(sf2, 0.0) + 2 * sf2_err), None

        out.append({
            'lag': float(np.median(lags)),
            'lag_min': float(lo),
            'lag_max': float(hi),
            'sf': float(sf),
            'sf_err': None if sf_err is None else float(sf_err),
            'sf2': float(sf2),
            'sf2_err': float(sf2_err),
            'noise': float(np.sqrt(np.mean(noise))),
            'npairs': int(npairs),
            'detected': bool(detected),
        })

    return out


def drw_structure_function(lag, sf_inf, tau):
    """The structure function of a damped random walk: SF∞ √(1 - e^(-Δt/τ))."""
    return sf_inf * np.sqrt(1 - np.exp(-np.asarray(lag) / tau))


def fit_drw(bins, span):
    """The damped random walk fitted to a structure function.

    Fitted to the squares with the noise taken off, so that the bins the
    noise drowns weigh in too, with what they rule out. Returns `sf_inf` and
    `tau` with their uncertainties, `tau_limit` when the timescale is longer
    than the data can tell (Kozłowski 2017: a tenth of the span) - so that
    it is no more than a lower limit - or None when there is too little to fit,
    or the fit determines neither to better than itself.
    """
    if len(bins) < SF_MIN_FIT_BINS or not any(b['detected'] for b in bins):
        return None

    lag = np.array([b['lag'] for b in bins])
    sf2 = np.array([b['sf2'] for b in bins])
    err = np.array([b['sf2_err'] for b in bins])

    def residuals(p):
        return (drw_structure_function(lag, np.exp(p[0]), np.exp(p[1])) ** 2 - sf2) / err

    plateau = np.sqrt(max(np.max(sf2), np.min(err)))
    starts = [np.log(lag[len(lag) // 2]), np.log(lag[0]), np.log(lag[-1])]
    best = min((least_squares(residuals, [np.log(plateau), s],
                              bounds=([np.log(plateau) - 10, np.log(lag[0]) - 5],
                                      [np.log(plateau) + 10, np.log(lag[-1]) + 5]))
                for s in starts), key=lambda r: r.cost)

    chi2 = 2 * best.cost
    dof = max(len(lag) - 2, 1)
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac) * max(1.0, chi2 / dof)
        errors = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        errors = np.array([np.nan, np.nan])

    # Neither determined to better than itself is no fit at all, whatever
    # line it draws
    if not np.all(errors < 1):
        return None

    sf_inf, tau = np.exp(best.x)
    return {
        'sf_inf': float(sf_inf),
        # In log, so the uncertainty of each is fractional
        'sf_inf_err': float(sf_inf * errors[0]),
        'tau': float(tau),
        'tau_err': float(tau * errors[1]),
        'tau_limit': bool(tau > DRW_TAU_SPAN * span),
        'chi2': float(chi2 / dof),
    }
