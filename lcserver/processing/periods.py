"""Searching a light curve for its periods, and following one that changes.

The computation behind the light curve viewer's period tools, kept apart from
the requests that ask for it: detrending by a penalised spline, the joint fit
of each series' trend with harmonics shared within a survey and filter, the
periodogram and box least squares searches and the spectral window, the
refit of their highest peaks, prewhitening by several frequencies at once,
the phase of a period that changes and the count of its cycles, the O-C of
windows against a fixed template, red noise, and the periodogram in a window
sliding along the light curve.

Everything here takes plain arrays - (mjd, value, error, group) per series,
where a group is a survey in one filter - and knows nothing of targets,
files or requests.
"""

import warnings

import numpy as np
from astropy.timeseries import LombScargle, BoxLeastSquares
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from scipy.interpolate import BSpline


# Grid density, in samples per periodogram peak
PERIOD_SAMPLES_PER_PEAK = 5

# The grid is what the search costs - it grows with the time span and with the
# shortest period, and does not depend on the number of points - so cap it to
# keep a single request fast. DASCH baselines run to well over a century.
PERIOD_MAX_FREQUENCIES = 500000

# The full periodogram is far too big to ship, so it is sent decimated to this
# many points, keeping the maxima rather than sampling them away
PERIOD_PLOT_POINTS = 2000

PERIOD_NPEAKS = 5

# Periods the sampling itself imprints on any periodogram from the ground: the
# solar and sidereal day with their harmonics, and the year
PERIOD_ALIASES = [1.0, 0.5, 1/3, 0.99726957, 0.49863478, 365.25]

# The trend is a cubic B-spline with a knot every so many days - the window -
# fitted on each stretch of a series that no gap longer than the window breaks.
# A stretch shorter than the window gets a constant level only: a polynomial
# through one isolated night would follow the very signal it is meant to leave.
DETREND_SPLINE_ORDER = 3

# Outliers are clipped from the fit of the trend, though not from the data
DETREND_CLIP_SIGMA = 3.0
DETREND_ITERATIONS = 3

# Weight of the penalty on the curvature of the spline, in units of one point.
# Too small to bend the trend where there are points to follow, and all there
# is where there are none - between two nights observed a hundred times each,
# say - which plain least squares would leave free to fall to zero
DETREND_PENALTY = 0.1

# The highest peaks are fitted again, the trend and this many harmonics of the
# period together, rather than one after the other
PERIOD_HARMONICS = 3

# Most harmonics the periodogram itself may be computed with
PERIOD_MAX_NTERMS = 4

# Trial frequencies across each peak in that fit, spanning one peak width
PERIOD_REFINE_SAMPLES = 21

# Peaks whose fits differ by less than this, in chi-squared scaled by the
# reduced chi-squared of the better one, cannot be told apart by the data -
# five sigma, for two models with the same number of parameters
PERIOD_AMBIGUOUS_DCHI2 = 25.0


def trend_blocks(mjds, window_days):
    """The columns of a trend slower than `window_days`, a stretch at a time.

    Returns (indices, columns) pairs, one per stretch of the series that no gap
    longer than the window interrupts: the points of the stretch, and the
    B-spline basis evaluated at them, with a knot every `window_days` days. The
    trend over the whole series is block-diagonal in these, so each is fitted
    on its own.

    Days rather than points, as ground-based sampling is anything but even: a
    window counted in points, converted at the median cadence, covers no time
    at all inside a night observed a hundred times and months where a season
    is sparse. And no polynomial is fitted across a gap, where it has nothing
    to follow.
    """
    x = np.asarray(mjds, dtype=float)
    order = np.argsort(x)
    breaks = np.where(np.diff(x[order]) > window_days)[0] + 1

    blocks = []
    for idx in np.split(order, breaks):
        lo, hi = x[idx].min(), x[idx].max()
        nknots = int(round((hi - lo) / window_days))
        k = DETREND_SPLINE_ORDER

        if nknots < 1 or len(idx) <= nknots + k:
            blocks.append((idx, np.ones((len(idx), 1))))
            continue

        knots = np.r_[[lo] * k, np.linspace(lo, hi, nknots + 1), [hi] * k]
        blocks.append((idx, BSpline.design_matrix(x[idx], knots, k).toarray()))

    return blocks


def _curvature(ncols):
    """Second differences of spline coefficients, the rows the penalty is on."""
    if ncols < 3:
        return np.zeros((0, ncols))

    return np.diff(np.eye(ncols), 2, axis=0)


def _robust_trend(columns, values):
    """The penalised least-squares fit of `columns` to `values`, outliers
    clipped from it."""
    # About the median, so that whatever the penalty leaves undecided settles
    # at the level of the series rather than at zero
    offset = np.median(values)
    values = values - offset

    penalty = np.sqrt(DETREND_PENALTY) * _curvature(columns.shape[1])
    zeros = np.zeros(len(penalty))
    keep = np.ones(len(values), dtype=bool)

    for _ in range(DETREND_ITERATIONS):
        coeffs = np.linalg.lstsq(np.vstack([columns[keep], penalty]),
                                 np.r_[values[keep], zeros], rcond=None)[0]
        resid = values - columns @ coeffs

        sigma = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
        if not sigma > 0:
            break

        clipped = np.abs(resid) < DETREND_CLIP_SIGMA * sigma
        if clipped.sum() <= columns.shape[1] or np.array_equal(clipped, keep):
            break
        keep = clipped

    return columns @ coeffs + offset


def detrend_series(mjds, values, window_days, subtract):
    """Remove variation slower than `window_days`, and return what it removed.

    The trend is fitted as `trend_blocks` describes it: a cubic spline with a
    knot every `window_days` days, on each stretch between gaps longer than
    that, with outliers clipped from the fit.

    Slow variation is not noise, and on a target like K2-18 it is the star's
    rotation - the largest real signal there is. This exists so that a short
    period can be searched for underneath it, not because the trend is wrong,
    which is why nothing calls it unless asked.

    `subtract` picks the arithmetic: magnitudes are logarithmic, so their trend
    comes off by subtraction, while a flux is divided by it.
    """
    mjds = np.asarray(mjds, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(mjds) < 2 * DETREND_SPLINE_ORDER + 3 or not window_days > 0:
        return values, None

    trend = np.empty_like(values)
    for idx, columns in trend_blocks(mjds, window_days):
        trend[idx] = _robust_trend(columns, values[idx])

    if subtract:
        # Keep the level, so the series still plots where it did
        return values - trend + np.median(trend), trend

    return values / trend, trend


def harmonics(mjds, frequency, nharm):
    """Sines and cosines of the first `nharm` harmonics of a frequency."""
    phase = 2 * np.pi * frequency * mjds
    return np.column_stack([fn((h + 1) * phase)
                            for h in range(nharm) for fn in (np.sin, np.cos)])


class SeriesFit:
    """One series and its trend, for fitting harmonics to at any frequency.

    The trend does not depend on the frequency, so it is eliminated once: for a
    trend of columns B with the curvature penalty P of the detrending, the
    best trend for any residual r is B (B'B + P)^-1 B' r, and what is left of
    the data and of each trial set of harmonics after taking it out is all the
    fit at one frequency needs - a system of 2 * nharm unknowns rather than
    one of all of them. Weighted by the errors throughout.

    The harmonics themselves may be shared with other series - see
    `refine_peaks` - so what a series gives at one frequency is its share of
    the normal equations, not a fit of its own.
    """

    def __init__(self, mjds, values, errors, window_days, nharm):
        self.mjds = mjds
        self.weight = 1.0 / errors
        self.nharm = nharm

        if window_days:
            blocks = trend_blocks(mjds, window_days)
        else:
            # No trend asked for: the series' own level and nothing else
            blocks = [(np.arange(len(mjds)), np.ones((len(mjds), 1)))]

        # The penalty is in units of one point, and a point here weighs its
        # inverse variance
        scale = DETREND_PENALTY * np.median(self.weight ** 2)

        self.blocks = []
        self.penalties = []
        self.ntrend = 0
        for idx, columns in blocks:
            bw = columns * self.weight[idx, None]
            curv = _curvature(columns.shape[1])
            self.blocks.append((idx, bw, np.linalg.pinv(bw.T @ bw + scale * curv.T @ curv)))
            self.penalties.append(np.sqrt(scale) * curv)
            self.ntrend += columns.shape[1]

        self.yw = values * self.weight
        self.my = self._detrend(self.yw)
        self.chi2_trend = float(self.yw @ self.my)

    def _detrend(self, v):
        """What is left of weighted columns `v` once the best trend is taken out."""
        out = np.array(v, dtype=float)
        for idx, bw, inverse in self.blocks:
            out[idx] -= bw @ (inverse @ (bw.T @ v[idx]))
        return out

    def normal(self, frequency):
        """This series' share of the normal equations of the harmonics at
        `frequency`, once its trend is taken out."""
        return self.normal_at(frequency * self.mjds)

    def normal_at(self, phase):
        """The same, for harmonics of a phase in cycles - one that need not
        grow linearly with time, for a period that changes."""
        return self.normal_of(harmonics(phase, 1.0, self.nharm))

    def normal_of(self, columns):
        """The same, for any columns - the harmonics of several frequencies
        at once, say."""
        h = columns * self.weight[:, None]
        mh = self._detrend(h)

        return h.T @ mh, mh.T @ self.yw

    def residuals(self, phase, coeffs):
        """The weighted residuals of the harmonics `coeffs` at `phase` and of
        the best trend under them, followed by the rows of the curvature
        penalty - whose squares sum to the chi-squared `normal_at` minimises."""
        return self.residuals_of(harmonics(phase, 1.0, self.nharm), coeffs)

    def residuals_of(self, columns, coeffs):
        """The same, for any columns."""
        v = self.yw - (columns * self.weight[:, None]) @ coeffs
        out = np.array(v)
        penalty = []
        for (idx, bw, inverse), pen in zip(self.blocks, self.penalties):
            beta = inverse @ (bw.T @ v[idx])
            out[idx] -= bw @ beta
            penalty.append(pen @ beta)

        return np.concatenate([out] + penalty)


def refine_peaks(series, peaks, window_days, span, nharm=PERIOD_HARMONICS):
    """The periodogram's highest peaks, fitted again and ranked by that fit.

    `series` are (mjd, value, error, group) of each series as loaded, before
    any trend came off. At every peak, each series is fitted with its own trend
    - the same spline the detrending used, or only its level when none was
    asked for - and `nharm` harmonics of the one period all of them share,
    which is varied across the width of the peak.

    The harmonics are shared by every series of a group - a survey in one
    filter - and differ between groups, as the amplitude of a pulsation is not
    the same in g as in r. Not by every series on its own: a phase free to
    differ from one TESS sector to the next would throw away the coherence
    across years that pins a period down, leaving the precision of a month.

    Fitting them together is what the periodogram cannot do: there the trend
    came off first, taking whatever of the signal it could with it, and the
    signal was a single sinusoid, which treats a light curve of any other shape
    as noise. The peaks of a family of aliases come out of a periodogram with
    much the same power, and the fit is what tells them apart, where anything
    can.

    Returns the peaks sorted by chi-squared, best first, each with its refined
    period, its chi-squared and the fraction of the variance about the trend
    the harmonics explain. And the reduced chi-squared of the best, to scale
    the differences between them by.

    Each period carries its uncertainty too, from the curvature of the
    chi-squared at the minimum, scaled by the reduced chi-squared when above
    one and by `red_noise_factor` on a tenth of its cycle - the errors of most
    surveys understate the scatter, and night to night variability no
    harmonics follow does not average down. None where the minimum is not
    one the samples bracket.
    """
    groups = {}
    for t, y, dy, group in series:
        groups.setdefault(group, []).append(SeriesFit(t, y, dy, window_days, nharm))

    # A group with too few points for its harmonics contributes its trend alone
    fitted = [members for members in groups.values()
              if sum(len(_.mjds) - _.ntrend for _ in members) > 2 * nharm + 2]

    fits = [_ for members in groups.values() for _ in members]
    chi2_trend = sum(_.chi2_trend for _ in fits)

    npoints = sum(len(_.mjds) for _ in fits)
    nparams = sum(_.ntrend for _ in fits) + 2 * nharm * len(fitted)
    dof = max(npoints - nparams, 1)

    def chi2(frequency):
        total = chi2_trend
        for members in fitted:
            gram, proj = 0, 0
            for fit in members:
                g, p = fit.normal(frequency)
                gram, proj = gram + g, proj + p
            total -= float(np.linalg.lstsq(gram, proj, rcond=None)[0] @ proj)
        return total

    def red_noise(frequency):
        """How much worse than white the residuals at `frequency` are."""
        times, resid = [], []
        for members in fitted:
            gram, proj = 0, 0
            for fit in members:
                g, p = fit.normal(frequency)
                gram, proj = gram + g, proj + p
            alpha = np.linalg.lstsq(gram, proj, rcond=None)[0]
            for fit in members:
                times.append(fit.mjds)
                resid.append(fit.residuals(frequency * fit.mjds, alpha)[:len(fit.mjds)])
        return red_noise_factor(np.concatenate(times), np.concatenate(resid),
                                RED_NOISE_BIN_CYCLES / frequency)

    refined = []
    for peak in peaks:
        f0 = peak['frequency']
        grid = f0 + np.linspace(-0.5, 0.5, PERIOD_REFINE_SAMPLES) / span
        grid = grid[grid > 0]
        values = np.array([chi2(_) for _ in grid])
        i = int(np.argmin(values))

        # The minimum between the samples, from the parabola through the three
        # around it, whose curvature is what the frequency is known to
        f, c = grid[i], values[i]
        sigma = None
        if 0 < i < len(grid) - 1:
            a, b, d = values[i - 1], values[i], values[i + 1]
            denom = a - 2 * b + d
            if denom > 0:
                step = 0.5 * (a - d) / denom
                f = grid[i] + step * (grid[1] - grid[0])
                c = min(chi2(f), b)
                if c == b:
                    f = grid[i]
                sigma = (grid[1] - grid[0]) * np.sqrt(2.0 / denom) * red_noise(f)

        refined.append(dict(peak,
                            # Where the periodogram had it, before the fit
                            periodogram_period=peak['period'],
                            period=round(float(1.0 / f), 6),
                            frequency=float(f),
                            chi2=round(c, 2),
                            fit_power=round(1.0 - c / chi2_trend, 4)
                            if chi2_trend > 0 else None,
                            # Unscaled by the reduced chi-squared until all
                            # are in, the best's being what scales them
                            frequency_err=sigma))

    refined.sort(key=lambda _: _['chi2'])
    reduced = refined[0]['chi2'] / dof if refined else None

    for peak in refined:
        if peak['frequency_err'] is not None:
            peak['frequency_err'] *= np.sqrt(max(reduced, 1.0))
            peak['period_err'] = float(peak['frequency_err'] / peak['frequency'] ** 2)
        else:
            peak['period_err'] = None

    return refined, reduced


def alias_kind(f1, f2, span):
    """What makes two frequencies aliases of each other, or None.

    The day and the year are the sampling of the ground: a frequency shifted by
    a whole number of cycles per day or per year is seen at the same moments as
    the original, and so is one reflected about such a shift - f and 1 - f,
    as a sinusoid of negative frequency is the same sinusoid. Matched to the
    resolution of the data, one peak width.
    """
    tolerance = 1.0 / span

    for diff in (abs(f1 - f2), f1 + f2):
        for cycles in (1, 2):
            if abs(diff - cycles) < tolerance:
                return 'daily'

    for cycles in (1, 2, 3):
        if abs(abs(f1 - f2) - cycles / 365.25) < tolerance:
            return 'yearly'

    return None


def frequency_columns(mjds, frequencies, nharm):
    """Harmonics of every frequency, side by side."""
    return np.hstack([harmonics(mjds, f, nharm) for f in frequencies])


def fit_frequencies(series, window_days, frequencies, nharm=PERIOD_HARMONICS):
    """Several frequencies fitted together, for prewhitening.

    `series` are (mjd, value, error, group) as `refine_peaks` takes them, and
    the model is the one it fits at a single frequency - every series its own
    trend, the harmonics shared within a group - with the harmonics of all of
    `frequencies` at once, and the frequencies themselves free. Each is
    refined in the presence of the others rather than one after another,
    which is what keeps a frequency found early from carrying the leakage of
    one found later.

    Returns the frequencies, their uncertainties - from the covariance, scaled
    by the reduced chi-squared and, each, by `red_noise_factor` on a tenth of
    its own cycle: a slow frequency is as uncertain as the noise is red on its
    timescale, which bins sized for a fast one do not see - and {group:
    coefficients}, the harmonics of each frequency in turn, nharm sine and
    cosine pairs apiece.
    """
    from scipy.optimize import least_squares

    frequencies = np.asarray(frequencies, dtype=float)

    groups = {}
    for t, y, dy, group in series:
        groups.setdefault(group, []).append(SeriesFit(t, y, dy, window_days, nharm))

    ncols = 2 * nharm * len(frequencies)
    fitted = {g: members for g, members in groups.items()
              if sum(len(f.mjds) - f.ntrend for f in members) > ncols + 2}

    if not fitted:
        raise ValueError('Too few points for the harmonics of the removed frequencies')

    allfits = [f for members in fitted.values() for f in members]
    span = max(f.mjds.max() for f in allfits) - min(f.mjds.min() for f in allfits)
    npoints = sum(len(f.mjds) for f in allfits)
    dof = max(npoints - sum(f.ntrend for f in allfits) - ncols * len(fitted)
              - len(frequencies), 1)

    def coefficients(freqs):
        out = {}
        for g, members in fitted.items():
            gram, proj = 0, 0
            for fit in members:
                gm, pm = fit.normal_of(frequency_columns(fit.mjds, freqs, nharm))
                gram, proj = gram + gm, proj + pm
            out[g] = np.linalg.lstsq(gram, proj, rcond=None)[0]
        return out

    # A step that moves the phase by a fixed fraction of a cycle over the data
    scale = PERIOD_CHANGE_PHASE_STEP / span

    def residuals(x):
        freqs = frequencies + x * scale
        coeffs = coefficients(freqs)
        return np.concatenate([fit.residuals_of(frequency_columns(fit.mjds, freqs, nharm), coeffs[g])
                               for g, members in fitted.items() for fit in members])

    result = least_squares(residuals, np.zeros(len(frequencies)), method='trf',
                           diff_step=PERIOD_CHANGE_DIFF_STEP, x_scale=1.0)

    freqs = frequencies + result.x * scale
    coeffs = coefficients(freqs)
    chi2 = float(2 * result.cost)

    times = np.concatenate([fit.mjds for members in fitted.values() for fit in members])
    resid = np.concatenate([fit.residuals_of(frequency_columns(fit.mjds, freqs, nharm),
                                             coeffs[g])[:len(fit.mjds)]
                            for g, members in fitted.items() for fit in members])
    beta = np.array([red_noise_factor(times, resid, RED_NOISE_BIN_CYCLES / f) for f in freqs])

    cov = np.linalg.pinv(result.jac.T @ result.jac) * scale ** 2 * max(chi2 / dof, 1.0)

    return freqs, np.sqrt(np.diag(cov)) * beta, coeffs


# Search methods beyond the plain periodogram, which costs far less per trial
# frequency: their grids are capped lower
PERIOD_MAX_FREQUENCIES_SLOW = 100000

# Box least squares: the period range is searched in chunks spanning this
# factor in period, each with its own durations - a duration has to be shorter
# than every period it is tried at - from the first to the second fraction of
# the shortest period in the chunk, in so many steps
BLS_CHUNK_FACTOR = 3.0
BLS_DURATION_FRACTIONS = (0.01, 0.25)
BLS_DURATIONS = 6

# Trial frequencies across the width of the narrowest box's peak - far
# narrower than a sinusoid's, which the periodogram's grid would step over
BLS_SAMPLES_PER_PEAK = 2

# The running median the box spectrum is measured against spans this fraction
# of each chunk, and never fewer trial periods than so many
BLS_SDE_WINDOWS = 50
BLS_SDE_MIN_WINDOW = 51


def _bls_power(mjds, values, errors, pmin, pmax, span):
    """Box least squares over the period range, a chunk at a time.

    Scored by the signal detection efficiency: the signal to noise of the best
    box at each period, less its running median over the neighbouring periods
    of the chunk, in units of the scatter about that. The raw signal to noise
    drifts - longer boxes take in more of any variation, so it has a level of
    its own in every chunk and slopes within it - and a light curve varying by
    a magnitude gives any box a large one; what stands out of its
    surroundings is what matters.
    Returns the frequencies, ascending, the power, the depth and the duration
    at each, and whether the grid had to be made coarser than the narrowest
    box asks for.
    """
    bounds = [pmin]
    while bounds[-1] < pmax:
        bounds.append(min(bounds[-1] * BLS_CHUNK_FACTOR, pmax))

    chunks = []
    for pa, pb in zip(bounds[:-1], bounds[1:]):
        durations = np.geomspace(BLS_DURATION_FRACTIONS[0] * pa, BLS_DURATION_FRACTIONS[1] * pa,
                                 BLS_DURATIONS)
        df = durations[0] / pb / (BLS_SAMPLES_PER_PEAK * span)
        chunks.append((pa, pb, durations, (1.0 / pa - 1.0 / pb) / df))

    total = sum(c[3] for c in chunks)
    shrink = min(1.0, PERIOD_MAX_FREQUENCIES_SLOW / total)

    model = BoxLeastSquares(mjds, values, errors)
    freq, power, depth, duration = [], [], [], []
    for pa, pb, durations, n in chunks:
        f = np.linspace(1.0 / pb, 1.0 / pa, max(int(n * shrink), 10))
        result = model.power(1.0 / f, durations, objective='snr')
        snr = np.nan_to_num(np.asarray(result.power, dtype=float))
        size = max(BLS_SDE_MIN_WINDOW, len(snr) // BLS_SDE_WINDOWS) | 1
        excess = snr - median_filter(snr, size=size, mode='nearest')
        scatter = 1.4826 * np.median(np.abs(excess - np.median(excess))) or 1.0
        freq.append(f)
        power.append(excess / scatter)
        depth.append(np.asarray(result.depth, dtype=float))
        duration.append(np.asarray(result.duration, dtype=float))

    freq, power, depth, duration = (np.concatenate(_) for _ in (freq, power, depth, duration))
    order = np.argsort(freq)

    return freq[order], power[order], depth[order], duration[order], bool(shrink < 1.0)


def _decimate(x, y, nbins):
    """The largest y of every one of nbins bins, with its x - so that a narrow
    peak survives instead of falling between the samples."""
    edges = np.linspace(0, len(y), min(nbins, len(y)) + 1).astype(int)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            j = a + int(np.argmax(y[a:b]))
            xs.append(x[j])
            ys.append(y[j])
    return np.array(xs), np.array(ys)


def spectral_window(mjds, width, nfreq):
    """The periodogram of the sampling alone, at frequency offsets from zero to
    `width`.

    A signal at any frequency is seen also wherever this has a peak, shifted
    by it - the aliases of the day, the month and the year, and whatever else
    the dates imprint. Centred on the period folded at, it says which peaks of
    the periodogram are that period over again. Decimated for display, the
    highest value in each bin.
    """
    offsets = np.linspace(width / nfreq, width, nfreq)
    window = LombScargle(mjds, np.ones_like(mjds), fit_mean=False,
                         center_data=False).power(offsets)
    x, y = _decimate(offsets, window, PERIOD_PLOT_POINTS)
    return {'offsets': [round(float(_), 7) for _ in x],
            'power': [round(float(_), 4) for _ in y]}


def find_period(mjds, values, errors, pmin, pmax, method='ls', nterms=1, magnitudes=True):
    """Periodogram of a light curve and its highest peaks.

    `method` is 'ls', the Lomb-Scargle periodogram with `nterms` harmonics -
    one is the sinusoid of the classic periodogram; more follow a light curve
    of another shape, a pulsator's sawtooth or a contact binary's double wave
    - or 'bls', box least squares, for eclipses and transits: a dip of some
    duration once a cycle, which the periodogram, fitting sinusoids, may not
    see at all. Its power is the signal detection efficiency - see _bls_power.

    Single-band, on purpose: the bands are expected to be centred on their own
    medians by the caller and pooled, which keeps the analytic false alarm
    probability available - for the one-term periodogram, the only one it
    exists for. LombScargleMultiband does not implement it.
    """
    span = np.max(mjds) - np.min(mjds)
    fmin, fmax = 1.0/pmax, 1.0/pmin

    nfreq = int(np.ceil(PERIOD_SAMPLES_PER_PEAK * span * (fmax - fmin)))
    window = spectral_window(mjds, fmax - fmin, int(np.clip(nfreq, 100, PERIOD_MAX_FREQUENCIES)))

    depth = duration = None
    if method == 'bls':
        # A dip of the flux is a rise of the magnitude, and the box is a dip
        freq, power, depth, duration, truncated = _bls_power(
            mjds, -values if magnitudes else values, errors, pmin, pmax, span)
        nfreq = len(freq)
        ls = None
    else:
        cap = PERIOD_MAX_FREQUENCIES if nterms == 1 else PERIOD_MAX_FREQUENCIES_SLOW
        truncated = nfreq > cap
        nfreq = int(np.clip(nfreq, 100, cap))
        freq = np.linspace(fmin, fmax, nfreq)
        ls = LombScargle(mjds, values, errors, nterms=nterms)
        power = ls.power(freq)

    # Highest local maxima, so that the neighbouring samples of one and the same
    # peak are not reported as separate detections
    idx, _ = find_peaks(power)
    if not len(idx):
        idx = np.array([np.argmax(power)])
    idx = idx[np.argsort(power[idx])[::-1]][:PERIOD_NPEAKS]

    peaks = []
    for i in idx:
        fap = None
        # The analytic false alarm probability is not always computable, and
        # degenerates for a handful of points, so report it only when it is sane
        if ls is not None and nterms == 1:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fap = float(ls.false_alarm_probability(
                        power[i], method='baluev',
                        minimum_frequency=fmin, maximum_frequency=fmax))
                if not np.isfinite(fap):
                    fap = None
            except Exception:
                fap = None

        peak = {
            'period': round(float(1.0/freq[i]), 6),
            'power': round(float(power[i]), 4),
            'frequency': float(freq[i]),
            'fap': fap,
        }
        if depth is not None:
            peak['depth'] = round(float(depth[i]), 5)
            peak['duration'] = round(float(duration[i]), 5)
        peaks.append(peak)

    x, y = _decimate(freq, power, PERIOD_PLOT_POINTS)

    return {
        # Rounded, as the full precision only inflates the response
        'periods': [round(float(1.0/_), 6) for _ in x],
        'power': [round(float(_), 4) for _ in y],
        'peaks': peaks,
        'nfreq': nfreq,
        'truncated': truncated,
        'window': window,
        'aliases': [_ for _ in PERIOD_ALIASES if pmin <= _ <= pmax],
    }


# ------------------------------------------------------------ period change

# Highest order of the polynomial a period may change by, and most coefficients
# of a spline - beyond them the cycle count drifts away between the gaps
PERIOD_CHANGE_MAX_ORDER = 6
PERIOD_CHANGE_MAX_COEFFS = 40

# A spline starts from the polynomial of this order, fitted first
PERIOD_CHANGE_SPLINE_START = 2

# Change of phase, in cycles, a unit step in each coefficient makes somewhere
# in the data - what puts them all on one footing for the optimiser
PERIOD_CHANGE_PHASE_STEP = 0.01

# Finite-difference step, in those units: large enough to stand above the
# rounding of a phase of tens of thousands of cycles
PERIOD_CHANGE_DIFF_STEP = 1e-4

# How finely the period and the O-C of the model are sent to be drawn and
# folded with
PERIOD_CHANGE_CURVE_POINTS = 2000

# How far either side of the starting frequency, as a fraction of it, the
# period within each window is searched for, and in how many steps at most
PERIOD_CHANGE_LOCAL_RANGE = 0.05
PERIOD_CHANGE_LOCAL_SAMPLES = 2000

# Trial phase shifts across one cycle in each O-C window
OC_SHIFT_SAMPLES = 200

# Points a window needs beyond its trend for a phase shift to be measured
OC_MIN_POINTS = 10

# And how much of the cycle it needs to cover, in this many bins of phase:
# the scraps the windows leave at the ends of a TESS sector cover a tenth of
# a cycle of a slow rotator, which a template free to scale fits whatever its
# phase. Not whole cycles - a day of WISE on a three day period covers half
# of one, and is what counts the cycles across the years WISE had alone.
OC_MIN_COVERAGE = 0.4
OC_COVERAGE_BINS = 20

# Chi-squared per window of the O-C about the model above which its scatter
# is noted as more than the errors account for - twice them, in rms
OC_EXCESS_SCATTER = 4.0

# The residuals are binned by this fraction of a cycle to see whether they
# average down as white noise would - the timescale a phase is measured on
RED_NOISE_BIN_CYCLES = 0.1


def red_noise_factor(times, residuals, bin_days):
    """How much worse than white the noise of `residuals` is on `bin_days`.

    The residuals, weighted, are averaged in bins of that many days. White
    noise of variance s^2 leaves the mean of n points with s^2 / n, so the
    mean of n * mean^2 over the bins is s^2 again; anything slower than the
    bin - spots, granulation, the systematics of a sector - stays in the means
    and makes it more. The ratio of the two scatters is the factor any
    uncertainty taken from the residuals as if white has to grow by, the
    beta of Pont, Zucker and Queloz (2006).

    One when there are too few bins with more than one point to tell, as for
    a survey that sees the star once a night.
    """
    times = np.asarray(times, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if not bin_days > 0 or len(times) < 4:
        return 1.0

    residuals = residuals - np.mean(residuals)
    _, index, counts = np.unique(np.floor((times - times.min()) / bin_days),
                                 return_inverse=True, return_counts=True)
    sums = np.bincount(index, weights=residuals)

    full = counts > 1
    if full.sum() < 3:
        return 1.0

    inside = full[index]
    white = np.mean(residuals[inside] ** 2)
    binned = np.mean(sums[full] ** 2 / counts[full])
    if not white > 0:
        return 1.0

    return float(max(np.sqrt(binned / white), 1.0))


class PhaseModel:
    """A phase, in cycles, linear in the coefficients of a frequency that
    changes with time.

    The frequency is f(t) = sum c_j b_j(t) and the phase its integral,
    sum c_j B_j(t), counted from the middle of the data - its constant is the
    harmonics' business. Either a polynomial of the time, in units of half the
    span, or a cubic spline with a knot every so many days - which is what the
    period of target 74 needed over sixteen years, a polynomial running away
    at the ends long before it follows the middle.

    A spline is extended past the ends of the data at the frequency it has
    there, rather than by its cubic.
    """

    def __init__(self, tmin, tmax, order=0, knot_days=None):
        self.tmin, self.tmax = float(tmin), float(tmax)
        self.tref = 0.5 * (self.tmin + self.tmax)
        self.half = 0.5 * (self.tmax - self.tmin)
        self.knot_days = knot_days

        if knot_days:
            nknots = max(int(round((self.tmax - self.tmin) / knot_days)), 1)
            k = 3
            self.knots = np.r_[[self.tmin] * k, np.linspace(self.tmin, self.tmax, nknots + 1),
                               [self.tmax] * k]
            self.ncoef = len(self.knots) - k - 1
            self.anti = [BSpline(self.knots, np.eye(self.ncoef)[j], k).antiderivative()
                         for j in range(self.ncoef)]
            self.anti0 = np.array([_(self.tref) for _ in self.anti])
        else:
            self.order = order
            self.ncoef = order + 1

    def describe(self):
        if self.knot_days:
            return f'spline with a knot every {self.knot_days:g} d ({self.ncoef} coefficients)'
        return f'polynomial of order {self.order}'

    def frequency_basis(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self.knot_days:
            tc = np.clip(t, self.tmin, self.tmax)
            return BSpline.design_matrix(tc, self.knots, 3).toarray()

        tau = (t - self.tref) / self.half
        return np.column_stack([tau ** k for k in range(self.ncoef)])

    def phase_basis(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self.knot_days:
            tc = np.clip(t, self.tmin, self.tmax)
            out = np.column_stack([_(tc) for _ in self.anti]) - self.anti0
            # Straight on past the ends, at the frequency there
            outside = t != tc
            if np.any(outside):
                out[outside] += self.frequency_basis(tc[outside]) * (t - tc)[outside, None]
            return out

        tau = (t - self.tref) / self.half
        return np.column_stack([self.half * tau ** (k + 1) / (k + 1) for k in range(self.ncoef)])

    def initial(self, frequency):
        """Coefficients of a constant frequency."""
        if self.knot_days:
            # The B-splines sum to one
            return np.full(self.ncoef, float(frequency))
        return np.r_[float(frequency), np.zeros(self.ncoef - 1)]

    def from_frequencies(self, t, frequency, error):
        """Coefficients fitted to frequencies measured at times `t` - for a
        spline under a penalty on its curvature, against knots no window
        reaches.

        Outliers are clipped, as a window a few nights fall in can find a
        period well away from the rest, and claim to know it better than any.
        The errors are floored at their median first: one such window weighed
        at face value pulls the first fit so far towards it that the clipping
        then takes out the windows at the ends instead - those that carry the
        change of the period - and the start lands a cycle count away. A start
        has to be robust more than it has to be optimal.
        """
        basis = self.frequency_basis(t)
        error = np.maximum(error, np.median(error))
        keep = np.ones(len(t), dtype=bool)

        for _ in range(DETREND_ITERATIONS + 1):
            a = basis[keep] / error[keep, None]
            b = frequency[keep] / error[keep]
            if self.knot_days:
                penalty = (np.sqrt(DETREND_PENALTY * np.median(1 / error[keep] ** 2))
                           * _curvature(self.ncoef))
                a, b = np.vstack([a, penalty]), np.r_[b, np.zeros(len(penalty))]
            coeffs = np.linalg.lstsq(a, b, rcond=None)[0]

            resid = (frequency - basis @ coeffs) / error
            sigma = 1.4826 * np.median(np.abs(resid[keep]))
            clipped = np.abs(resid) < DETREND_CLIP_SIGMA * sigma
            if not sigma > 0 or clipped.sum() <= self.ncoef or np.array_equal(clipped, keep):
                break
            keep = clipped

        return coeffs

    def from_phases(self, t, phase, error):
        """Coefficients fitted to phases measured at times `t`, along with a
        constant of their own - for a spline under a penalty on its curvature,
        a frequency changing by so much over a knot costing as much as a phase
        out by as much at a point."""
        a = np.column_stack([np.ones(len(t)), self.phase_basis(t)]) / error[:, None]
        b = phase / error
        if self.knot_days:
            curv = _curvature(self.ncoef)
            penalty = (np.sqrt(DETREND_PENALTY) * np.median(1 / error) * self.knot_days
                       * np.column_stack([np.zeros(len(curv)), curv]))
            a, b = np.vstack([a, penalty]), np.r_[b, np.zeros(len(penalty))]
        return np.linalg.lstsq(a, b, rcond=None)[0][1:]

    def project(self, other, coeffs):
        """These coefficients nearest to the frequency `other` has with `coeffs`."""
        t = np.linspace(self.tmin, self.tmax, 20 * self.ncoef + 20)
        return np.linalg.lstsq(self.frequency_basis(t),
                               other.frequency_basis(t) @ coeffs, rcond=None)[0]


def rotate_harmonics(coeffs, shift, nharm):
    """Coefficients of the harmonics of phase - shift, from those of phase.

    Pairs of sine and cosine of each harmonic, as `harmonics` has them, for
    one shift or an array of them - then a row per shift.
    """
    shift = np.atleast_1d(shift)
    out = np.empty((len(shift), 2 * nharm))
    for h in range(nharm):
        a, b = coeffs[2 * h], coeffs[2 * h + 1]
        d = 2 * np.pi * (h + 1) * shift
        out[:, 2 * h] = a * np.cos(d) + b * np.sin(d)
        out[:, 2 * h + 1] = b * np.cos(d) - a * np.sin(d)
    return out


def fit_phase_model(series, window_days, model, coeffs, nharm=PERIOD_HARMONICS):
    """The phase of a period that changes, fitted with the trend and harmonics.

    `series` are (mjd, value, error, group) as `refine_peaks` takes them, and
    the fit is the one it makes - every series its own trend, the harmonics
    shared within a group - but for harmonics of the phase of `model` rather
    than of a constant frequency, its coefficients now free. Started from
    `coeffs`, which have to be near already: the cycle count between the
    seasons is a local thing to find.

    The trend and the harmonics, linear, are solved for at every trial of the
    phase rather than searched along with it, which leaves the optimiser only
    the coefficients and makes the covariance of those the one marginalised
    over the rest.

    Returns the coefficients, their covariance, the chi-squared and the
    degrees of freedom, the harmonics of every group as {group: coefficients},
    and how much worse than white the residuals are on a tenth of a cycle -
    `red_noise_factor`. The covariance is scaled by the reduced chi-squared
    and by the square of that factor, as white noise is what it otherwise
    assumes: a spotted star's light curve wanders from night to night in ways
    no harmonics follow, and a period fitted through it is known no better.
    """
    from scipy.optimize import least_squares

    groups = {}
    for t, y, dy, group in series:
        groups.setdefault(group, []).append((SeriesFit(t, y, dy, window_days, nharm),
                                             model.phase_basis(t)))

    # A group with too few points for its harmonics contributes its trend alone
    fitted = {g: members for g, members in groups.items()
              if sum(len(f.mjds) - f.ntrend for f, _ in members) > 2 * nharm + 2}

    if not fitted:
        raise ValueError('Too few points for the harmonics in any survey and filter')

    allfits = [f for members in groups.values() for f, _ in members]
    chi2_unfitted = sum(f.chi2_trend for g, members in groups.items()
                        if g not in fitted for f, _ in members)
    npoints = sum(len(f.mjds) for f in allfits)
    nparams = (sum(f.ntrend for f in allfits) + 2 * nharm * len(fitted) + model.ncoef)
    dof = max(npoints - nparams, 1)

    def group_coefficients(c):
        out = {}
        for g, members in fitted.items():
            gram, proj = 0, 0
            for fit, basis in members:
                gm, pm = fit.normal_at(basis @ c)
                gram, proj = gram + gm, proj + pm
            out[g] = np.linalg.lstsq(gram, proj, rcond=None)[0]
        return out

    # Each coefficient scaled to move the phase by a fixed step somewhere in
    # the data
    span = np.concatenate([b for members in fitted.values() for _, b in members])
    scale = PERIOD_CHANGE_PHASE_STEP / np.maximum(np.max(np.abs(span), axis=0), 1e-12)
    c0 = np.asarray(coeffs, dtype=float)

    def residuals(x):
        c = c0 + x * scale
        alpha = group_coefficients(c)
        return np.concatenate([fit.residuals(basis @ c, alpha[g])
                               for g, members in fitted.items()
                               for fit, basis in members])

    result = least_squares(residuals, np.zeros(model.ncoef), method='trf',
                           diff_step=PERIOD_CHANGE_DIFF_STEP, x_scale=1.0)

    c = c0 + result.x * scale
    chi2 = float(2 * result.cost) + chi2_unfitted

    alpha = group_coefficients(c)
    times = np.concatenate([fit.mjds for members in fitted.values() for fit, _ in members])
    resid = np.concatenate([fit.residuals(basis @ c, alpha[g])[:len(fit.mjds)]
                            for g, members in fitted.items() for fit, basis in members])
    frequency = float(model.frequency_basis(model.tref)[0] @ c)
    beta = red_noise_factor(times, resid, RED_NOISE_BIN_CYCLES / frequency)

    jac = result.jac
    cov = (np.linalg.pinv(jac.T @ jac) * scale[:, None] * scale[None, :]
           * (chi2 / dof) * beta ** 2)

    return c, cov, chi2, dof, alpha, beta


def _oc_windows(times, length):
    """Edges of the windows the O-C is measured in: each from the first point
    after the last window to `length` days later, so that they follow the
    seasons rather than a grid the data may straddle."""
    times = np.sort(times)
    edges = []
    i = 0
    while i < len(times):
        start = times[i]
        edges.append((start, start + length))
        i = np.searchsorted(times, start + length, side='left')
    return edges


def local_periods(series, window_days, frequency, length, nharm=PERIOD_HARMONICS):
    """The period within every window, searched for near `frequency`.

    Each window on its own, with the fit of `refine_peaks` - every series its
    trend, the harmonics shared within a group - across PERIOD_CHANGE_LOCAL_RANGE
    either side of the frequency. Known to no better than a window allows, but
    it needs no count of the cycles between the windows, which is what makes it
    the start of a fit that does: a period that changes by a few percent over
    the years slips by whole cycles from what a constant one predicts.

    Returns per window (mean MJD, first and last MJD, number of points,
    frequency, its uncertainty), the latter scaled by the reduced chi-squared
    of the window when above one. Windows whose best frequency is at an edge
    of the range, or that span fewer than three cycles, are left out.
    """
    alltimes = np.concatenate([t for t, _, _, _ in series])
    out = []

    for a, b in _oc_windows(alltimes, length):
        groups = {}
        times = []
        for t, y, dy, group in series:
            k = (t >= a) & (t < b)
            if np.any(k):
                groups.setdefault(group, []).append(SeriesFit(t[k], y[k], dy[k], window_days, nharm))
                times.append(t[k])

        times = np.concatenate(times)
        span = times.max() - times.min()
        if span * frequency < 3:
            continue

        fitted = [members for members in groups.values()
                  if sum(len(_.mjds) - _.ntrend for _ in members) > 2 * nharm + 2]
        if not fitted:
            continue

        fits = [_ for members in groups.values() for _ in members]
        chi2_trend = sum(_.chi2_trend for _ in fits)
        dof = max(len(times) - sum(_.ntrend for _ in fits) - 2 * nharm * len(fitted) - 1, 1)

        def chi2(f):
            total = chi2_trend
            for members in fitted:
                gram, proj = 0, 0
                for fit in members:
                    g, p = fit.normal(f)
                    gram, proj = gram + g, proj + p
                total -= float(np.linalg.lstsq(gram, proj, rcond=None)[0] @ proj)
            return total

        width = PERIOD_CHANGE_LOCAL_RANGE * frequency
        step = max(0.1 / span, 2 * width / PERIOD_CHANGE_LOCAL_SAMPLES)
        grid = np.arange(frequency - width, frequency + width + step / 2, step)
        values = np.array([chi2(_) for _ in grid])

        i = int(np.argmin(values))
        if not 0 < i < len(grid) - 1:
            continue

        prev, best, nxt = values[i - 1], values[i], values[i + 1]
        denom = prev - 2 * best + nxt
        if not denom > 0:
            continue

        f = grid[i] + 0.5 * step * (prev - nxt) / denom

        # As for the O-C, no better known than the noise allows when it does
        # not average down as white noise would
        resid, rtimes = [], []
        for members in fitted:
            gram, proj = 0, 0
            for fit in members:
                g, p = fit.normal(f)
                gram, proj = gram + g, proj + p
            alpha = np.linalg.lstsq(gram, proj, rcond=None)[0]
            for fit in members:
                resid.append(fit.residuals(f * fit.mjds, alpha)[:len(fit.mjds)])
                rtimes.append(fit.mjds)
        beta = red_noise_factor(np.concatenate(rtimes), np.concatenate(resid),
                                RED_NOISE_BIN_CYCLES / f)

        error = np.sqrt(max(best / dof, 1.0) / (denom / (2 * step ** 2))) * beta

        out.append((float(np.mean(times)), float(times.min()), float(times.max()),
                    int(len(times)), float(f), float(error)))

    return out


def connect_phases(model, t, phase, error, frequency):
    """Count the cycles between windows whose phases are known only modulo one.

    `phase` is the phase of the light curve in each window at times `t`, as
    measured against a fixed template, and `frequency` the one to start out
    with. Starting from the window measured best, the windows are taken in
    one at a time - always the nearest in time to those already in - each
    given the whole number of cycles that puts it nearest to where the fit of
    those already in predicts it. The fit is allowed more freedom as more
    windows come in: a constant frequency to begin with, then one changing as
    the target `model` does - a polynomial up to its order, a spline once the
    windows span enough of its knots, and a straight change of frequency
    before that - but a changing one only where the prediction reaches no
    further than the windows it rests on.

    This is how the period of a star is followed across the seasons, and the
    years when nobody looked: a model of the whole light curve started from a
    period that is wrong far from the middle has its cycle count wrong there,
    and no local optimiser counts it right again.

    Returns the coefficients of `model` fitted to all the windows so counted.
    """
    order = np.argsort(t)
    t, phase, error = t[order], phase[order], error[order]
    n = len(t)

    seed = int(np.argmin(error))
    lo = hi = seed
    counted = np.array(phase, dtype=float)

    def predictor(lo, hi, x):
        tt, pp, ee = t[lo:hi + 1], counted[lo:hi + 1], error[lo:hi + 1]
        if hi == lo:
            return pp[0] + frequency * (x - tt[0])

        # A change of the period is only extrapolated as far as the windows it
        # was fitted on reach: four windows over a season say little of the
        # curvature of the phase, and carried across the year to the next
        # season what little they say is out by a cycle. Further than that, a
        # constant period fitted to them, which carries much further
        span = tt[-1] - tt[0]
        reach = max(tt[0] - x, x - tt[-1])
        if reach > span:
            return _track(PhaseModel(model.tmin, model.tmax, order=0), tt, pp, ee)(x)

        if model.knot_days:
            local = PhaseModel(tt[0], tt[-1], knot_days=model.knot_days)
            if span >= 3 * model.knot_days and len(tt) >= local.ncoef + 3:
                return _track(local, tt, pp, ee)(x)
            cap = PERIOD_CHANGE_SPLINE_START
        else:
            cap = model.order

        local = PhaseModel(model.tmin, model.tmax,
                            order=min(cap, max(0, (len(tt) - 2) // 2)))
        return _track(local, tt, pp, ee)(x)

    while hi - lo + 1 < n:
        # The nearer of the two windows either side of those counted
        gap_lo = t[lo] - t[lo - 1] if lo > 0 else np.inf
        gap_hi = t[hi + 1] - t[hi] if hi < n - 1 else np.inf
        j = lo - 1 if gap_lo < gap_hi else hi + 1

        predicted = predictor(lo, hi, t[j])
        counted[j] = phase[j] + np.round(predicted - phase[j])
        lo, hi = min(lo, j), max(hi, j)

    return model.from_phases(t, counted, error)


def _track(model, t, phase, error):
    """The phase `model` fits to measured phases, as a function of time."""
    coeffs = model.from_phases(t, phase, error)
    const = np.median(phase - model.phase_basis(t) @ coeffs)
    return lambda x: float(const + model.phase_basis(x)[0] @ coeffs)


def measure_oc(series, window_days, phase, templates, length, nharm=PERIOD_HARMONICS):
    """O-C of every window, as a shift in phase of a fixed template.

    `phase(t)` is the phase in cycles the shift is measured from - the linear
    ephemeris - and `templates` the harmonics of each group's light curve at
    that phase. Within each window every series gets its trend, and the
    template of its group, shifted by the one phase shift and scaled by the
    one amplitude all of them share, is fitted on top - the amplitude scaled
    as well since a light curve changes its amplitude along with its period,
    and would push the shift about if held to it.

    Returns per window (mean MJD, first and last MJD, number of points, phase
    shift, its uncertainty, amplitude, red noise factor), the shift in
    [-0.5, 0.5). Its uncertainty is that of white noise, scaled by the reduced
    chi-squared of the window when above one and then by how much worse than
    white the residuals are on a tenth of a cycle - `red_noise_factor`. Night
    to night variability the template does not follow does not average down
    over the thousands of points of a TESS sector, and the shift is no better
    known than it allows.

    Windows covering less than OC_MIN_COVERAGE of the cycle are left out.
    """
    alltimes = np.concatenate([t for t, _, _, _ in series])
    grid = np.arange(OC_SHIFT_SAMPLES) / OC_SHIFT_SAMPLES - 0.5
    step = 1.0 / OC_SHIFT_SAMPLES
    out = []

    for a, b in _oc_windows(alltimes, length):
        # Cycles per day of the reference, which the phase is linear in
        frequency = float(phase(np.array([a + 1.0]))[0] - phase(np.array([a]))[0])

        parts = {}
        members = []
        chi2_trend, npts, ntrend = 0.0, 0, 0
        times = []
        for t, y, dy, group in series:
            k = (t >= a) & (t < b)
            if group not in templates or not np.any(k):
                continue
            fit = SeriesFit(t[k], y[k], dy[k], window_days, nharm)
            ph = phase(t[k])
            gram, proj = fit.normal_at(ph)
            g0, p0 = parts.get(group, (0, 0))
            parts[group] = (g0 + gram, p0 + proj)
            members.append((fit, ph, group))
            chi2_trend += fit.chi2_trend
            npts += k.sum()
            ntrend += fit.ntrend
            times.append(t[k])

        if npts < ntrend + 2 + OC_MIN_POINTS:
            continue

        times = np.concatenate(times)
        covered = np.unique(np.floor((phase(times) % 1.0) * OC_COVERAGE_BINS)).size
        if covered < OC_MIN_COVERAGE * OC_COVERAGE_BINS:
            continue

        num, den = 0.0, 0.0
        for group, (gram, proj) in parts.items():
            r = rotate_harmonics(templates[group], grid, nharm)
            num = num + r @ proj
            den = den + np.einsum('ij,jk,ik->i', r, gram, r)

        with np.errstate(divide='ignore', invalid='ignore'):
            chi2 = np.where((num > 0) & (den > 0), chi2_trend - num ** 2 / den, chi2_trend)

        i = int(np.argmin(chi2))
        prev, best, nxt = chi2[i - 1], chi2[i], chi2[(i + 1) % len(grid)]
        curv = (prev - 2 * best + nxt) / (2 * step ** 2)
        if not curv > 0 or not den[i] > 0:
            continue

        shift = grid[i] + 0.5 * step * (prev - nxt) / (prev - 2 * best + nxt)
        amplitude = float(num[i] / den[i])
        reduced = best / max(npts - ntrend - 2, 1)

        # The residuals of the template as fitted, to see how white they are
        residuals = np.concatenate([
            fit.residuals(ph, amplitude * rotate_harmonics(templates[group], shift, nharm)[0])[:len(ph)]
            for fit, ph, group in members])
        beta = red_noise_factor(np.concatenate([fit.mjds for fit, _, _ in members]),
                                residuals, RED_NOISE_BIN_CYCLES / frequency)

        error = np.sqrt(max(reduced, 1.0) / curv) * beta

        # Not measured, only guessed at: no better than a quarter of a cycle
        if error > 0.25:
            continue

        out.append((float(np.mean(times)), float(times.min()), float(times.max()), int(npts),
                    float((shift + 0.5) % 1.0 - 0.5), float(error), amplitude, beta))

    return out


# The time-resolved periodogram: at most so many windows, each searched on a
# grid of at most so many frequencies, and sent back decimated to so many
# bins, the highest power in each - a grid fine enough for a window of a
# century of DASCH is far more than a heat map can show
DYNAMIC_MAX_WINDOWS = 300
DYNAMIC_MAX_FREQUENCIES = 20000
DYNAMIC_PLOT_POINTS = 600

# The window when none is asked for, as a fraction of the span of the data -
# no one length suits both two sectors of TESS and a century of DASCH
DYNAMIC_AUTO_WINDOW = 0.1

# Points, and the fraction of its length, a window needs for its periodogram
DYNAMIC_MIN_POINTS = 10
DYNAMIC_MIN_COVERAGE = 0.25


def dynamic_power(series, pmin, pmax, window, step):
    """Lomb-Scargle periodograms in a window sliding along the light curve.

    `series` are (mjd, value, error) of each series, ready to be searched.
    Within each window every series is centred on its own median there - its
    level drifts, and pooled as it is that drift would be all the window
    showed - and the periodogram is of them pooled, on one frequency grid for
    every window, as fine as the window resolves.

    Returns the window centres, the number of points in each, the periods of
    the grid as decimated, and the power, a row per window - None for a
    window with too few points, or covering too little of itself, to say
    anything.
    """
    tmin = min(t.min() for t, _, _ in series)
    tmax = max(t.max() for t, _, _ in series)

    centres = np.arange(tmin + window / 2, tmax - window / 2 + step / 2, step)
    if not len(centres):
        centres = np.array([0.5 * (tmin + tmax)])
    if len(centres) > DYNAMIC_MAX_WINDOWS:
        centres = np.linspace(centres[0], centres[-1], DYNAMIC_MAX_WINDOWS)

    fmin, fmax = 1.0 / pmax, 1.0 / pmin
    # Never coarser than a few hundred, so that a narrow range is drawn smooth
    nfreq = int(np.clip(np.ceil(PERIOD_SAMPLES_PER_PEAK * window * (fmax - fmin)),
                        3 * DYNAMIC_PLOT_POINTS // 2, DYNAMIC_MAX_FREQUENCIES))
    freq = np.linspace(fmin, fmax, nfreq)

    # Bins of the grid kept for the plot, each its highest power
    nbins = min(DYNAMIC_PLOT_POINTS, nfreq)
    edges = np.linspace(0, nfreq, nbins + 1).astype(int)
    centre_index = (edges[:-1] + edges[1:]) // 2

    counts, rows = [], []
    for c in centres:
        tt, yy, ee = [], [], []
        for t, y, dy in series:
            k = np.abs(t - c) <= window / 2
            if k.sum() < 2:
                continue
            tt.append(t[k])
            yy.append(y[k] - np.median(y[k]))
            ee.append(dy[k])

        n = sum(len(_) for _ in tt)
        counts.append(int(n))
        if n < DYNAMIC_MIN_POINTS:
            rows.append(None)
            continue

        tt, yy, ee = np.concatenate(tt), np.concatenate(yy), np.concatenate(ee)
        if np.ptp(tt) < DYNAMIC_MIN_COVERAGE * window:
            rows.append(None)
            continue

        power = LombScargle(tt, yy, ee).power(freq)
        rows.append([round(float(np.max(power[a:b])), 4) if b > a else None
                     for a, b in zip(edges[:-1], edges[1:])])

    return {
        'mjd': [round(float(_), 3) for _ in centres],
        'n': counts,
        'periods': [round(float(1.0 / freq[i]), 6) for i in centre_index],
        'power': rows,
        'nfreq': nfreq,
    }
