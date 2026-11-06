from django.http import HttpResponse, JsonResponse
from django.template.response import TemplateResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods

import os
import json
import glob
import re
import warnings

import numpy as np
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.interpolate import BSpline

from . import models
from . import surveys
from .processing.utils import mission_quality_mask, quality_level


def json_safe(values):
    """An array as a list JSON can carry, with what it cannot as null.

    NaN is not JSON, however readily Python writes and reads it again: a
    browser's JSON.parse refuses the whole document over one of them. It
    reaches here through the uncertainties - a source may know a magnitude
    without knowing how well it is known, which is most of what the AAVSO
    holds from before uncertainties were asked for - so those become null,
    which every consumer of this already understands as a point with no bar.
    """
    return [None if not np.isfinite(v) else float(v) for v in values]


def load_magnitude_data(basepath):
    """Load magnitude-based light curve data from multiple surveys"""
    lightcurve_data = []

    # Iterate over registry directly
    for source_id, survey_config in surveys.SURVEY_SOURCES.items():
        # Skip if not magnitude mode
        if survey_config.get('lc_mode') != 'magnitude':
            continue

        votable_file = survey_config.get('votable_file')
        if not votable_file:
            continue

        fullname = os.path.join(basepath, votable_file)
        if not os.path.exists(fullname):
            continue

        try:
            data = Table.read(fullname)

            x = np.asarray(data['mjd'], dtype=float)
            color = survey_config.get('lc_color', '#000000')

            for band in surveys.get_source_bands(survey_config):
                if band['mag'] not in data.colnames or band['err'] not in data.colnames:
                    # An older file, written before this band existed
                    continue

                y = np.asarray(data[band['mag']], dtype=float)
                dy = np.asarray(data[band['err']], dtype=float)

                idx = np.isfinite(x) & np.isfinite(y)

                # A band may occupy only part of the table, selected by the
                # filter the measurement was taken through
                fcol = band.get('filter_column')
                if fcol and band.get('filter_value') is not None:
                    if fcol not in data.colnames:
                        continue
                    idx &= np.asarray(data[fcol]).astype(str) == str(band['filter_value'])

                if not np.sum(idx):
                    continue

                label = survey_config['short_name']
                if band['label']:
                    label += f" {band['label']}"

                # Note: datetime conversion moved to frontend for performance
                lightcurve_data.append({
                    'source_id': source_id,
                    'label': label,
                    'filter': band['label'],
                    'kind': band['kind'],
                    'note': band.get('note') or '',
                    # Measurements are shown by default; anything reached
                    # through an assumed colour waits until it is asked for
                    'default_visible': band['kind'] in surveys.BAND_KINDS_SHOWN,
                    # On the common g scale, as the combined curve draws it
                    'combined': bool(band.get('combined')),
                    'color': band.get('color') or color,
                    'mjd': x[idx].tolist(),
                    'mag': y[idx].tolist(),
                    'magerr': json_safe(dy[idx]),
                    'n_points': int(np.sum(idx)),
                })

        except Exception as e:
            # Skip files that can't be read
            continue

    return lightcurve_data


def load_flux_data(basepath, config=None):
    """Load flux-based light curve data from the space missions.

    The config is the target's, and carries how hard each source was asked
    to filter - the files hold every cadence and the mission flags with
    them, so the choice takes effect here rather than at acquisition.
    """
    config = config or {}
    lightcurve_data = []

    # Fallback palette, for a source that names no colours of its own
    default_palette = ['#e74c3c', '#8e44ad', '#3498db', '#e67e22', '#1abc9c']

    # Iterate over registry to find flux sources
    for source_id, survey_config in surveys.SURVEY_SOURCES.items():
        # Skip if not flux mode
        if survey_config.get('lc_mode') != 'flux':
            continue

        votable_pattern = survey_config.get('votable_file')
        if not votable_pattern:
            continue

        # Find all matching files
        pattern_path = os.path.join(basepath, votable_pattern)
        source_files = glob.glob(pattern_path)

        color_palette = survey_config.get('lc_color_palette') or default_palette

        # What the mission calls an observing segment - a TESS sector, a Kepler
        # quarter, a K2 campaign
        segment_name = survey_config.get('lc_segment_name') or 'Segment'

        # A source spanning several phases of one mission letters its segments
        # by phase, as each numbers them from one
        prefixes = survey_config.get('lc_segment_prefixes') or {}

        # Every flux source writes {source}_lc_{segment}_{author}_{exptime}.vot,
        # where the segment carries a phase letter only where there is one
        filename_pattern = re.compile(
            re.escape(source_id) + r'_lc_([A-Za-z]?)(\d+)_([^_]+)_(\d+)\.vot')

        def sortkey(filepath):
            """Phases in the order flown, and segments in order within them."""
            match = filename_pattern.match(os.path.basename(filepath))

            if not match:
                return (len(prefixes), 0)

            order = list(prefixes).index(match.group(1)) if match.group(1) in prefixes else -1
            return (order, int(match.group(2)))

        for i, filepath in enumerate(sorted(source_files, key=sortkey)):
            filename = os.path.basename(filepath)
            match = filename_pattern.match(filename)

            if not match:
                continue

            prefix = match.group(1)
            sector = int(match.group(2))
            author = match.group(3)
            exptime = int(match.group(4))

            # A phase names its own segments - K2's are K2 campaigns, not
            # Kepler ones, though both are the same source here
            label_name = prefixes.get(prefix) or f"{survey_config['short_name']} {segment_name}"

            try:
                data = Table.read(filepath)

                # Use registry values
                flux_col = survey_config.get('lc_flux_column', 'flux')
                err_col = survey_config.get('lc_err_column', 'flux_err')
                quality_col = survey_config.get('lc_quality_column')

                # Check for required columns
                if 'mjd' not in data.colnames or flux_col not in data.colnames:
                    continue

                # Convert MJD to datetime
                data['time'] = Time(data['mjd'], format='mjd')

                # Get flux data
                x = data['mjd']
                flux = data[flux_col]

                # Handle missing flux_err column
                if err_col in data.colnames:
                    flux_err = data[err_col]
                else:
                    flux_err = np.zeros_like(flux)

                # Filter bad data
                idx = np.isfinite(x) & np.isfinite(flux)

                # Filter by quality flag if present, as hard as the source was
                # asked to. The files keep every cadence and the flags with
                # them, so this is where the choice takes effect for the
                # missions - the step itself only applies it to its own plots.
                if quality_col and quality_col in data.colnames:
                    idx &= mission_quality_mask(
                        data[quality_col], source_id,
                        quality_level(config, source_id))

                if not np.sum(idx):
                    continue

                # Normalize flux to median = 1.0
                valid_flux = flux[idx]
                median_flux = np.median(valid_flux)
                if median_flux > 0:
                    flux_normalized = flux / median_flux
                    flux_err_normalized = flux_err / median_flux
                else:
                    flux_normalized = flux
                    flux_err_normalized = flux_err

                # Create label
                label = f"{label_name} {sector} ({author}, {exptime}s)"

                # Assign color from palette (cycle if >5 sectors)
                color = color_palette[i % len(color_palette)]

                # Note: datetime conversion moved to frontend for performance
                # Use numpy array indexing for efficiency
                series_data = {
                    'source_id': source_id,
                    'label': label,
                    'sector': sector,
                    'author': author,
                    'exptime': exptime,
                    # TESS fluxes are as measured, only normalized to their median
                    'kind': surveys.BAND_NATIVE,
                    'note': '',
                    'default_visible': True,
                    'color': color,
                    'mjd': x[idx].tolist(),
                    'flux': flux_normalized[idx].tolist(),
                    'flux_err': json_safe(flux_err_normalized[idx]),
                    'n_points': int(np.sum(idx)),
                }

                lightcurve_data.append(series_data)

            except Exception as e:
                # Skip files that can't be read
                continue

    return lightcurve_data


@login_required
def target_lightcurve(request, id):
    """Interactive light curve viewer using Plotly"""
    target = models.Target.objects.get(id=id)

    # Check permissions
    if not target.can_view(request.user):
        return HttpResponse('Forbidden', status=403)

    # Determine display mode for initial render
    mode = request.GET.get('mode', 'auto')

    # Auto-detect mode if not specified (lightweight check)
    if mode == 'auto':
        basepath = target.path()

        # Check for magnitude data from registry
        has_magnitude_data = False
        for survey_config in surveys.SURVEY_SOURCES.values():
            if survey_config.get('lc_mode') == 'magnitude':
                votable_file = survey_config.get('votable_file')
                if votable_file and os.path.exists(os.path.join(basepath, votable_file)):
                    has_magnitude_data = True
                    break

        # Check for flux data from registry
        has_flux_data = False
        for survey_config in surveys.SURVEY_SOURCES.values():
            if survey_config.get('lc_mode') == 'flux':
                votable_pattern = survey_config.get('votable_file')
                if votable_pattern:
                    pattern_path = os.path.join(basepath, votable_pattern)
                    if glob.glob(pattern_path):
                        has_flux_data = True
                        break

        # Prefer magnitude mode if both exist
        if has_magnitude_data:
            mode = 'magnitude'
        elif has_flux_data:
            mode = 'flux'
        else:
            mode = 'magnitude'  # Default fallback

    # Which missions the flux mode covers, named rather than assumed: it was
    # TESS alone for long enough that the button said so, and Kepler arriving
    # left it lying. Taken from the registry so the next one is not a third
    # place to remember.
    flux_sources = '/'.join(
        config['short_name']
        for _, config in sorted(surveys.SURVEY_SOURCES.items(),
                                key=lambda kv: kv[1].get('order', 99))
        if config.get('lc_mode') == 'flux' and config.get('processing_function'))

    context = {
        'target': target,
        'target_id': id,
        'data_mode': mode,
        'mode': mode,
        'flux_sources': flux_sources,
    }

    return TemplateResponse(request, 'lightcurve_viewer.html', context=context)


@login_required
@require_http_methods(["GET"])
def load_lightcurve_data(request, id):
    """Load lightcurve data asynchronously via AJAX"""
    target = models.Target.objects.get(id=id)

    # Check permissions
    if not target.can_view(request.user):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    try:
        basepath = target.path()
        mode = request.GET.get('mode', 'magnitude')

        # Load data based on mode
        if mode == 'flux':
            lightcurve_data = load_flux_data(basepath, target.config)
        else:
            lightcurve_data = load_magnitude_data(basepath)

        return JsonResponse({
            'data': lightcurve_data,
            'mode': mode,
            'no_data': len(lightcurve_data) == 0,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'error': str(e),
            'traceback': traceback.format_exc()
        }, status=500)


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


def _harmonics(mjds, frequency, nharm):
    """Sines and cosines of the first `nharm` harmonics of a frequency."""
    phase = 2 * np.pi * frequency * mjds
    return np.column_stack([fn((h + 1) * phase)
                            for h in range(nharm) for fn in (np.sin, np.cos)])


class _SeriesFit:
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
        h = _harmonics(phase, 1.0, self.nharm) * self.weight[:, None]
        mh = self._detrend(h)

        return h.T @ mh, mh.T @ self.yw

    def residuals(self, phase, coeffs):
        """The weighted residuals of the harmonics `coeffs` at `phase` and of
        the best trend under them, followed by the rows of the curvature
        penalty - whose squares sum to the chi-squared `normal_at` minimises."""
        v = self.yw - (_harmonics(phase, 1.0, self.nharm) * self.weight[:, None]) @ coeffs
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
    """
    groups = {}
    for t, y, dy, group in series:
        groups.setdefault(group, []).append(_SeriesFit(t, y, dy, window_days, nharm))

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

    refined = []
    for peak in peaks:
        f0 = peak['frequency']
        grid = f0 + np.linspace(-0.5, 0.5, PERIOD_REFINE_SAMPLES) / span
        grid = grid[grid > 0]
        values = np.array([chi2(_) for _ in grid])
        i = int(np.argmin(values))

        # The minimum between the samples, from the parabola through the three
        # around it
        f, c = grid[i], values[i]
        if 0 < i < len(grid) - 1:
            a, b, d = values[i - 1], values[i], values[i + 1]
            denom = a - 2 * b + d
            if denom > 0:
                step = 0.5 * (a - d) / denom
                f = grid[i] + step * (grid[1] - grid[0])
                c = min(chi2(f), b)
                if c == b:
                    f = grid[i]

        refined.append(dict(peak,
                            # Where the periodogram had it, before the fit
                            periodogram_period=peak['period'],
                            period=round(float(1.0 / f), 6),
                            frequency=float(f),
                            chi2=round(c, 2),
                            fit_power=round(1.0 - c / chi2_trend, 4)
                            if chi2_trend > 0 else None))

    refined.sort(key=lambda _: _['chi2'])

    return refined, (refined[0]['chi2'] / dof if refined else None)


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


def find_period(mjds, values, errors, pmin, pmax):
    """Lomb-Scargle periodogram of a light curve and its highest peaks.

    Single-band, on purpose: the bands are expected to be centred on their own
    medians by the caller and pooled, which keeps the analytic false alarm
    probability available. LombScargleMultiband does not implement it.
    """
    ls = LombScargle(mjds, values, errors)

    span = np.max(mjds) - np.min(mjds)
    fmin, fmax = 1.0/pmax, 1.0/pmin

    nfreq = int(np.ceil(PERIOD_SAMPLES_PER_PEAK * span * (fmax - fmin)))
    truncated = nfreq > PERIOD_MAX_FREQUENCIES
    nfreq = int(np.clip(nfreq, 100, PERIOD_MAX_FREQUENCIES))

    freq = np.linspace(fmin, fmax, nfreq)
    power = ls.power(freq)

    # Highest local maxima, so that the neighbouring samples of one and the same
    # peak are not reported as separate detections
    idx, _ = find_peaks(power)
    if not len(idx):
        idx = np.array([np.argmax(power)])
    idx = idx[np.argsort(power[idx])[::-1]][:PERIOD_NPEAKS]

    peaks = []
    for i in idx:
        # The analytic false alarm probability is not always computable, and
        # degenerates for a handful of points, so report it only when it is sane
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

        peaks.append({
            'period': round(float(1.0/freq[i]), 6),
            'power': round(float(power[i]), 4),
            'frequency': float(freq[i]),
            'fap': fap,
        })

    # Decimate for display by keeping the largest value of every bin, so that a
    # narrow peak survives instead of falling between the samples
    nbins = min(PERIOD_PLOT_POINTS, nfreq)
    edges = np.linspace(0, nfreq, nbins + 1).astype(int)
    pp, pw = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            j = a + int(np.argmax(power[a:b]))
            # Rounded, as the full precision only inflates the response
            pp.append(round(float(1.0/freq[j]), 6))
            pw.append(round(float(power[j]), 4))

    return {
        'periods': pp,
        'power': pw,
        'peaks': peaks,
        'nfreq': nfreq,
        'truncated': truncated,
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


class _PhaseModel:
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
        """
        basis = self.frequency_basis(t)
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


def _rotate(coeffs, shift, nharm):
    """Coefficients of the harmonics of phase - shift, from those of phase.

    Pairs of sine and cosine of each harmonic, as `_harmonics` has them, for
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
        groups.setdefault(group, []).append((_SeriesFit(t, y, dy, window_days, nharm),
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

    def harmonics(c):
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
        alpha = harmonics(c)
        return np.concatenate([fit.residuals(basis @ c, alpha[g])
                               for g, members in fitted.items()
                               for fit, basis in members])

    result = least_squares(residuals, np.zeros(model.ncoef), method='trf',
                           diff_step=PERIOD_CHANGE_DIFF_STEP, x_scale=1.0)

    c = c0 + result.x * scale
    chi2 = float(2 * result.cost) + chi2_unfitted

    alpha = harmonics(c)
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
                groups.setdefault(group, []).append(_SeriesFit(t[k], y[k], dy[k], window_days, nharm))
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
    before that.

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

    def predictor(lo, hi):
        tt, pp, ee = t[lo:hi + 1], counted[lo:hi + 1], error[lo:hi + 1]
        if hi == lo:
            return lambda x: pp[0] + frequency * (x - tt[0])

        if model.knot_days:
            local = _PhaseModel(tt[0], tt[-1], knot_days=model.knot_days)
            if tt[-1] - tt[0] >= 3 * model.knot_days and len(tt) >= local.ncoef + 3:
                return _track(local, tt, pp, ee)
            cap = PERIOD_CHANGE_SPLINE_START
        else:
            cap = model.order

        local = _PhaseModel(model.tmin, model.tmax,
                            order=min(cap, max(0, (len(tt) - 2) // 2)))
        return _track(local, tt, pp, ee)

    while hi - lo + 1 < n:
        # The nearer of the two windows either side of those counted
        gap_lo = t[lo] - t[lo - 1] if lo > 0 else np.inf
        gap_hi = t[hi + 1] - t[hi] if hi < n - 1 else np.inf
        j = lo - 1 if gap_lo < gap_hi else hi + 1

        predicted = predictor(lo, hi)(t[j])
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
            fit = _SeriesFit(t[k], y[k], dy[k], window_days, nharm)
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
            r = _rotate(templates[group], grid, nharm)
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
            fit.residuals(ph, amplitude * _rotate(templates[group], shift, nharm)[0])[:len(ph)]
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


class _BadRequest(Exception):
    """A request the fit cannot be run on, with what to tell the user."""

    def __init__(self, error, details=None):
        super().__init__(error)
        self.error = error
        self.details = details

    def response(self):
        return JsonResponse({'error': self.error, 'details': self.details}, status=400)


def _fit_options(data):
    """The detrending window and the dates of a period fit request.

    The window is in days, absent or zero leaving the light curve alone, which
    is the default: what it would remove is often the real signal. The dates
    are MJDs, either end of them open.
    """
    try:
        detrend_window = float(data.get('detrend_window') or 0)
    except (TypeError, ValueError):
        detrend_window = 0.0

    try:
        mjd_min, mjd_max = (None if data.get(_) in (None, '') else float(data[_])
                            for _ in ('mjd_min', 'mjd_max'))
    except (TypeError, ValueError):
        raise _BadRequest('Invalid date range', 'The dates must be MJDs')

    if mjd_min is not None and mjd_max is not None and mjd_max <= mjd_min:
        raise _BadRequest('Invalid date range',
                          f'The end of the date range (MJD {mjd_max}) must be after its start (MJD {mjd_min})')

    if detrend_window and detrend_window <= 0:
        raise _BadRequest('Invalid detrending window',
                          f'The window must be a positive number of days (got {detrend_window})')

    return detrend_window, mjd_min, mjd_max


def _requested_series(target, series_list, mode, mjd_min, mjd_max):
    """The series a period fit asked for, loaded and filtered as the viewer has them.

    The client sends only the selection - which series are visible and how
    they are filtered - and the data itself is loaded here from the same files
    the viewer was built from. Sending the points back would make the request
    grow with the light curve and exceed DATA_UPLOAD_MAX_MEMORY_SIZE (TESS
    alone is several MB per target).

    Returns a (index, series, t, y, dy, searched) tuple for every series with
    points in the dates, `searched` marking those points.
    """
    if not series_list:
        raise _BadRequest('No series data provided')

    basepath = target.path()
    if mode == 'flux':
        available = load_flux_data(basepath, target.config)
        value_key, error_key = 'flux', 'flux_err'
    else:
        available = load_magnitude_data(basepath)
        value_key, error_key = 'mag', 'magerr'

    labels = {}
    for i, series in enumerate(available):
        labels.setdefault(series['label'], i)

    out = []
    for i, requested in enumerate(series_list):
        label = requested.get('label')
        index = requested.get('index')

        # Prefer the index the viewer used, but only while it still points
        # at the same series - the files may have changed since page load
        if (isinstance(index, int) and 0 <= index < len(available)
                and (label is None or available[index]['label'] == label)):
            series = available[index]
        elif label in labels:
            series = available[labels[label]]
        else:
            raise _BadRequest('Series is no longer available',
                              f'Series {i} ({label}) was not found on the server. '
                              'Reload the page and try again.')

        t = np.array(series['mjd'], dtype=float)
        y = np.array(series[value_key], dtype=float)
        dy = np.array(series[error_key], dtype=float)

        # Same max error cut the viewer applies, magnitude mode only
        max_error = requested.get('max_error')
        if mode != 'flux' and max_error is not None:
            keep = dy <= max_error
            t, y, dy = t[keep], y[keep], dy[keep]

        # Filter out non-finite values
        valid = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)
        t, y, dy = t[valid], y[valid], dy[valid]

        searched = np.ones(len(t), dtype=bool)
        if mjd_min is not None:
            searched &= t >= mjd_min
        if mjd_max is not None:
            searched &= t <= mjd_max

        if np.any(searched):
            out.append((index, series, t, y, dy, searched))

    return out


def _series_group(series):
    """The survey and filter a series is in, which its harmonics are shared within."""
    return (series.get('source_id'), series.get('filter'), series.get('kind'))


@login_required
@require_http_methods(["POST"])
def fit_period(request, id):
    """Fit period using Lomb-Scargle multiband periodogram with comprehensive error handling"""
    target = models.Target.objects.get(id=id)

    # Check permissions
    if not target.can_view(request.user):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    try:
        data = json.loads(request.body)
        mode = data.get('mode', 'magnitude')
        period_min = data.get('period_min', 0.1)  # Default: 0.1 days
        period_max = data.get('period_max', 100)  # Default: 100 days

        try:
            detrend_window, mjd_min, mjd_max = _fit_options(data)
            requested = _requested_series(target, data.get('series', []), mode,
                                          mjd_min, mjd_max)
        except _BadRequest as e:
            return e.response()

        # Validate period range
        if period_min <= 0:
            return JsonResponse({
                'error': 'Invalid period range',
                'details': f'Minimum period must be positive (got {period_min})'
            }, status=400)

        if period_max <= period_min:
            return JsonResponse({
                'error': 'Invalid period range',
                'details': f'Maximum period ({period_max}) must be greater than minimum period ({period_min})'
            }, status=400)

        # Collect all data points from visible series
        times = []
        values = []
        errors = []

        # What detrending left behind, so the viewer can show and fold the same
        # points the periodogram was computed from
        detrended = []

        # And each series as it was before, for the fit of the highest peaks,
        # which fits the trend along with the period rather than before it
        loaded = []

        # Only the dates asked for are searched, while the trend is taken off
        # the whole series: the viewer shows every point, so it gets every
        # point back
        for index, series, t, y, dy, searched in requested:
            loaded.append((t[searched], y[searched], dy[searched],
                           _series_group(series)))

            # Detrended one series at a time, before they are pooled: each has
            # its own cadence and its own gaps, and a window in days means a
            # different number of points in each
            if detrend_window:
                y, trend = detrend_series(t, y, detrend_window,
                                          subtract=(mode != 'flux'))

                if trend is not None:
                    detrended.append({
                        'index': index if isinstance(index, int) else None,
                        'label': series['label'],
                        'mjd': [round(float(_), 6) for _ in t],
                        'value': [round(float(_), 6) for _ in y],
                    })

            t, y, dy = t[searched], y[searched], dy[searched]

            # Every series sits at a level of its own - magnitudes differ from
            # band to band, and the normalized fluxes of the TESS sectors are
            # around unity - so pooling them as they are would find the color of
            # the star, or its mean brightness, rather than its period. Centring
            # each on its own median puts them on a common scale. Their
            # amplitudes still differ, which the transform tolerates.
            y = y - np.median(y)

            times.append(t)
            values.append(y)
            errors.append(dy)

        if not times:
            return JsonResponse({
                'error': 'No valid data points to fit',
                'details': 'All data points were invalid (NaN, Inf, or non-positive errors)'
                + (' or outside the date range' if mjd_min is not None or mjd_max is not None else '')
            }, status=400)

        # Concatenate all data
        t_all = np.concatenate(times)
        y_all = np.concatenate(values)
        dy_all = np.concatenate(errors)

        # Validation: Check minimum number of data points
        n_points = len(t_all)
        if n_points < 10:
            return JsonResponse({
                'error': 'Insufficient data',
                'details': f'Need at least 10 data points for reliable period fitting (got {n_points})'
            }, status=400)

        # Validation: Check data time span
        time_span = np.max(t_all) - np.min(t_all)
        if time_span <= 0:
            return JsonResponse({
                'error': 'Invalid data',
                'details': 'All data points have the same time'
            }, status=400)

        # Warn if searching for periods longer than data span
        if period_max > time_span:
            # This is a warning, not an error - still allow the fit
            pass

        # Check if data span is too short compared to minimum period
        if time_span < 2 * period_min:
            return JsonResponse({
                'error': 'Insufficient time coverage',
                'details': f'Data span ({time_span:.2f} days) should be at least 2× minimum period ({period_min:.2f} days) for reliable fitting'
            }, status=400)

        if detrend_window:
            print(f'Detrended with a {detrend_window:g} d window, '
                  f'{len(detrended)} of {len(times)} series')

        print(f'Starting period fit: {period_min:.3f} - {period_max:.3f} days')
        print(f'Data: {n_points} points, span {time_span:.2f} days, {len(times)} series')

        try:
            pg = find_period(t_all, y_all, dy_all, period_min, period_max)
        except Exception as e:
            return JsonResponse({
                'error': 'Periodogram computation failed',
                'details': str(e)
            }, status=500)

        print('Fit finished')

        if not pg['peaks']:
            return JsonResponse({
                'error': 'Periodogram computation failed',
                'details': 'No peaks found in the periodogram'
            }, status=500)

        notes = []

        # Ranked again by fitting the trend and the harmonics together, which
        # can tell apart peaks the periodogram gives the same power. Not worth
        # failing the fit over: the periodogram's own order is still an answer
        highest = pg['peaks'][0]
        try:
            peaks, scale = refine_peaks(loaded, pg['peaks'], detrend_window,
                                        time_span)
        except Exception:
            import traceback
            traceback.print_exc()
            peaks, scale = pg['peaks'], None
            notes.append('The highest peaks could not be fitted again with the '
                         'trend and harmonics together, so they are ranked by '
                         'periodogram power alone.')

        best = peaks[0]
        best_period = best['period']
        best_power = best['power']
        best_freq = best['frequency']
        fap = best['fap']

        if scale is not None:
            if best.get('periodogram_period') != highest['period']:
                notes.append(
                    f'The highest periodogram peak is at {highest["period"]:.6f} days, '
                    f'but fitting the trend and {PERIOD_HARMONICS} harmonics of the period '
                    f'together prefers {best_period:.6f} days.')

            # Peaks the data cannot tell apart. Scaled by the reduced chi-squared,
            # as the errors of most surveys understate the scatter
            for other in peaks[1:]:
                dchi2 = (other['chi2'] - best['chi2']) / max(scale, 1.0)
                if dchi2 < PERIOD_AMBIGUOUS_DCHI2:
                    kind = alias_kind(best_freq, other['frequency'], time_span)
                    notes.append(
                        f'{other["period"]:.6f} days fits almost as well as {best_period:.6f} days '
                        f'(scaled chi-squared {dchi2:.1f} higher)'
                        + (f' - they are {kind} aliases of each other, which the sampling of these data cannot tell apart.'
                           if kind else '.'))

        # Converging to an edge of the searched range usually means the real
        # period lies outside it. Reported rather than raised: the periodogram
        # comes back along with it and shows the situation better than any
        # message could, and failing outright would hide it.
        period_tolerance = 0.05  # 5% tolerance
        if best_period < period_min * (1 + period_tolerance):
            notes.append(f'Best period ({best_period:.4f} days) is at the lower edge of the search range ({period_min:.4f} days). Try decreasing the minimum period - the signal may have a shorter one.')

        if best_period > period_max * (1 - period_tolerance):
            notes.append(f'Best period ({best_period:.4f} days) is at the upper edge of the search range ({period_max:.4f} days). Try increasing the maximum period - the signal may have a longer one.')

        # A filter narrower than a few times the period does not only remove
        # the trend, it removes the signal - which shows up as a period at half
        # the true one, or as a peak that fades as the window is narrowed
        if detrend_window and best_period > detrend_window / 3.0:
            notes.append(
                f'The detrending window ({detrend_window:g} days) is less than three times '
                f'the period found ({best_period:.4f} days), so the filter is removing part '
                f'of the signal along with the trend. Widen the window and refit - if the '
                f'period changes, it was the filter talking.')

        # Warn if best period is close to data span (aliasing risk)
        if best_period > 0.8 * time_span:
            notes.append(f'Best period ({best_period:.2f} days) is close to data span ({time_span:.2f} days). Period may be poorly constrained.')

        # Check if power is suspiciously low (weak or no periodicity)
        # Typical significant peaks have power > 0.1, but this is data-dependent
        if best_power < 0.05:
            notes.append(f'Low periodogram power ({best_power:.4f}). Signal may be very weak or non-periodic.')

        if fap is not None and fap > 0.01:  # 1% FAP threshold
            notes.append(f'High false alarm probability ({fap:.4f}). Detection may not be significant.')

        if pg['truncated']:
            notes.append(f'Frequency grid truncated to {pg["nfreq"]} samples. Narrow peaks may be missed - narrow the search range to resolve them.')

        # Estimate epoch (time of maximum) using phase folding
        # Use median time as initial guess
        epoch = np.median(t_all)

        # Build result
        result = {
            'period': float(best_period),
            'epoch': float(epoch),
            'power': float(best_power),
            'frequency': float(best_freq),
            'n_points': int(n_points),
            'n_series': len(times),
            'period_min': float(period_min),
            'period_max': float(period_max),
            'time_span': float(time_span),
            'mjd_min': mjd_min,
            'mjd_max': mjd_max,
            # Decimated periodogram for display, plus its highest peaks
            'periodogram': {'periods': pg['periods'], 'power': pg['power']},
            'peaks': peaks,
            'harmonics': PERIOD_HARMONICS,
            'aliases': pg['aliases'],
            'nfreq': pg['nfreq'],
            'truncated': pg['truncated'],
            'detrend_window': float(detrend_window),
            # The points the periodogram was actually computed from, so that
            # what is folded and drawn is what was fitted
            'detrended': detrended,
        }

        if fap is not None:
            result['fap'] = float(fap)

        if notes:
            result['warnings'] = notes

        return JsonResponse(result)

    except json.JSONDecodeError as e:
        return JsonResponse({
            'error': 'Invalid JSON in request body',
            'details': str(e)
        }, status=400)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'error': 'Unexpected error during period fitting',
            'details': str(e),
            'traceback': traceback.format_exc()
        }, status=500)


def _positive(data, key, default, kind=float):
    try:
        value = kind(data.get(key) if data.get(key) not in (None, '') else default)
    except (TypeError, ValueError):
        value = None
    if value is None or not value > 0:
        raise _BadRequest(f'Invalid {key.replace("_", " ")}',
                          f'It must be a positive number (got {data.get(key)!r})')
    return value


@login_required
@require_http_methods(["POST"])
def fit_period_change(request, id):
    # The fit is thousands of products of small matrices, which OpenBLAS
    # spreads over threads at a cost of milliseconds each - ten times what
    # the products themselves take
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api='blas'):
        return _fit_period_change(request, id)


def _fit_period_change(request, id):
    """Fit a period that changes with time, and the O-C of the data against
    the period being folded with.

    The period and T0 of the fold are the reference ephemeris, and the period
    is where the fit starts from - it has to be near already, as from "Fit
    Period". The phase of the model then folds the light curve, with T0 moved
    to the maximum of light nearest to where it was.
    """
    target = models.Target.objects.get(id=id)

    if not target.can_view(request.user):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    try:
        data = json.loads(request.body)
        mode = data.get('mode', 'magnitude')

        try:
            detrend_window, mjd_min, mjd_max = _fit_options(data)
            period = _positive(data, 'period', None)
            oc_window = _positive(data, 'oc_window', 100.0)

            kind = data.get('model', 'polynomial')
            if kind == 'spline':
                knot_days = _positive(data, 'knot_days', 365.0)
                order = PERIOD_CHANGE_SPLINE_START
            else:
                knot_days = None
                try:
                    order = int(data.get('order', 1))
                except (TypeError, ValueError):
                    order = -1
                if not 0 <= order <= PERIOD_CHANGE_MAX_ORDER:
                    raise _BadRequest('Invalid order',
                                      f'The order of the polynomial must be 0 to {PERIOD_CHANGE_MAX_ORDER}')

            requested = _requested_series(target, data.get('series', []), mode,
                                          mjd_min, mjd_max)
        except _BadRequest as e:
            return e.response()

        loaded = [(t[s], y[s], dy[s], _series_group(series))
                  for index, series, t, y, dy, s in requested]

        # The light curve with its trend taken off, as "Fit Period" sends it,
        # so that the fold shows the variation the phase was fitted to
        detrended = []
        if detrend_window:
            for index, series, t, y, dy, s in requested:
                y, trend = detrend_series(t, y, detrend_window, subtract=(mode != 'flux'))
                if trend is not None:
                    detrended.append({
                        'index': index if isinstance(index, int) else None,
                        'label': series['label'],
                        'mjd': [round(float(_), 6) for _ in t],
                        'value': [round(float(_), 6) for _ in y],
                    })

        if not loaded:
            return JsonResponse({'error': 'No valid data points to fit',
                                 'details': 'No series has points in the dates'}, status=400)

        alltimes = np.concatenate([_[0] for _ in loaded])
        npoints = len(alltimes)
        tmin, tmax = float(alltimes.min()), float(alltimes.max())
        span = tmax - tmin

        if span < 3 * period:
            return JsonResponse({
                'error': 'Insufficient time coverage',
                'details': f'The data span {span:.2f} days, too little for a period of {period:.4f} days to change over'
            }, status=400)

        try:
            epoch = float(data.get('epoch'))
        except (TypeError, ValueError):
            epoch = None

        notes = []

        # The period within each window, which needs no count of the cycles
        # between them
        local = local_periods(loaded, detrend_window, 1.0 / period, oc_window)
        lt, lf, le = (np.array([_[k] for _ in local]) for k in (0, 4, 5))

        def best_fit(m, starts):
            fits = [fit_phase_model(loaded, detrend_window, m, c) for c in starts]
            return min(fits, key=lambda _: _[2])

        # Order by order from a constant period, each started both from the
        # last and from the periods of the windows - a period that changes a
        # lot is out of reach of a start from a constant one, which ends a
        # cycle or more out at the ends. Whichever fits better is kept, and
        # the chi-squared of each order tells what it bought over the last
        stages = []
        model, coeffs = None, None
        for o in range(order + 1):
            m = _PhaseModel(tmin, tmax, order=o)
            starts = [m.initial(1.0 / period) if coeffs is None else np.r_[coeffs, 0.0]]
            if o > 0 and len(local) > o + 1:
                starts.append(m.from_frequencies(lt, lf, le))
            fitted = best_fit(m, starts)
            model, coeffs = m, fitted[0]
            stages.append((m, fitted))

        target_model = model
        if knot_days:
            target_model = _PhaseModel(tmin, tmax, knot_days=knot_days)
            if target_model.ncoef > PERIOD_CHANGE_MAX_COEFFS:
                return JsonResponse({
                    'error': 'Too many knots',
                    'details': f'A knot every {knot_days:g} days makes {target_model.ncoef} coefficients over '
                               f'{span:.0f} days; at most {PERIOD_CHANGE_MAX_COEFFS} are fitted'
                }, status=400)

        # And from the phases of the windows, against the template of the
        # last fit, with the cycles between them counted one window at a time
        # - the start that survives the gaps, where the others lose count
        starts = []
        if knot_days or order > 0:
            def reference(t):
                return (np.asarray(t, dtype=float) - model.tref) / period

            windows = measure_oc(loaded, detrend_window, reference, stages[-1][1][4], oc_window)
            if len(windows) > 2:
                wt, ws, we = (np.array([_[k] for _ in windows]) for k in (0, 4, 5))
                starts.append(connect_phases(target_model, wt, reference(wt) - ws, we,
                                             1.0 / period))

        if knot_days:
            starts.append(target_model.project(model, coeffs))
            if len(local) > 2:
                starts.append(target_model.from_frequencies(lt, lf, le))
            stages.append((target_model, best_fit(target_model, starts)))
            model = target_model
        elif starts:
            fitted = best_fit(model, starts)
            if fitted[2] < stages[-1][1][2]:
                stages[-1] = (model, fitted)

        coeffs, cov, chi2, dof, harmonics, red_noise = stages[-1][1]
        reduced = chi2 / dof

        # Each stage scored against the scatter the last one leaves, as the
        # errors of most surveys understate it
        scored = []
        for m, (c, _, x2, d, _, _) in stages:
            scored.append({'model': m.describe(), 'ncoef': m.ncoef,
                           'chi2': round(x2, 2),
                           'bic': round(x2 / max(reduced, 1e-12) + m.ncoef * np.log(npoints), 2)})
        preferred = min(range(len(scored)), key=lambda i: scored[i]['bic'])
        if preferred != len(scored) - 1:
            notes.append(f'The Bayesian information criterion prefers the {scored[preferred]["model"]} '
                         f'to the {scored[-1]["model"]} - the extra freedom is not bought by the data.')

        # The phase zero moved to the maximum of light of the template of the
        # group with most points, so that T0 is a time of maximum as an
        # ephemeris has it
        sizes = {}
        for t, _, _, g in loaded:
            sizes[g] = sizes.get(g, 0) + len(t)
        main = max(harmonics, key=lambda g: sizes.get(g, 0))
        g = np.arange(1000) / 1000.0
        curve = _harmonics(g, 1.0, PERIOD_HARMONICS) @ harmonics[main]
        zero = float(g[np.argmax(curve) if mode == 'flux' else np.argmin(curve)])
        templates = {k: _rotate(v_, -zero, PERIOD_HARMONICS)[0] for k, v_ in harmonics.items()}

        def model_phase(t):
            return model.phase_basis(t) @ coeffs - zero

        def model_frequency(t):
            return model.frequency_basis(t) @ coeffs

        # The maximum nearest to the T0 asked for, or to the middle of the data
        t0 = epoch if epoch is not None and tmin <= epoch <= tmax else model.tref
        cycle = float(np.round(model_phase(t0)[0]))
        for _ in range(5):
            t0 -= float((model_phase(t0)[0] - cycle) / model_frequency(t0)[0])

        def linear_phase(t):
            return (np.asarray(t, dtype=float) - t0) / period

        def offset(t):
            """O-C of the model, in cycles of the reference period."""
            return linear_phase(t) - (model_phase(t) - cycle)

        oc = measure_oc(loaded, detrend_window, linear_phase, templates, oc_window)

        # Each shift taken a whole number of cycles from where the model has it,
        # which is what counts the cycles across the gaps
        oc_points = {'mjd': [], 'mjd_min': [], 'mjd_max': [], 'n': [], 'oc': [],
                     'oc_err': [], 'amplitude': [], 'red_noise': []}
        for tm, ta, tb, n, shift, err, amp, beta in oc:
            shift += np.round(offset(tm)[0] - shift)
            for key, value in zip(oc_points, (round(tm, 4), round(ta, 4), round(tb, 4), n,
                                              round(float(shift * period), 6),
                                              round(err * period, 6), round(amp, 4),
                                              round(beta, 2))):
                oc_points[key].append(value)

        # How far the O-C strays from the model, against its errors. Well
        # above one, the phase wanders more than the noise accounts for -
        # which on a spotted star is the spots, growing, decaying and
        # migrating, and otherwise a model too simple for the period
        oc_scatter = None
        if len(oc_points['mjd']) > 1:
            dev = (np.array(oc_points['oc']) / period - offset(np.array(oc_points['mjd'])))
            oc_scatter = float(np.mean((dev * period / np.array(oc_points['oc_err'])) ** 2))
            if oc_scatter > OC_EXCESS_SCATTER:
                notes.append(f'The O-C scatters about the model {np.sqrt(oc_scatter):.1f} times more than '
                             f'its errors allow (chi-squared {oc_scatter:.1f} per window) - the phase '
                             f'wanders beyond the noise, as spots make it on a spotted star, or the '
                             f'model is too simple for how the period changes.')

        # A window the model misses by a good part of a cycle has its cycle
        # count in doubt, and with it the model around it
        missed = [tm for tm, o in zip(oc_points['mjd'], oc_points['oc'])
                  if abs(o / period - offset(tm)[0]) > 0.25]
        if missed:
            notes.append(f'The model misses {len(missed)} of the O-C windows by more than a quarter '
                         f'of a cycle (around MJD {", ".join(f"{_:.0f}" for _ in missed[:5])}'
                         f'{", ..." if len(missed) > 5 else ""}) - the count of cycles there is in '
                         f'doubt. Narrow the dates, or give the model more freedom.')

        if len(oc_points['mjd']) < 3:
            notes.append(f'Only {len(oc_points["mjd"])} windows of {oc_window:g} days had enough '
                         f'points for an O-C - try longer ones.')

        grid = np.linspace(tmin, tmax, PERIOD_CHANGE_CURVE_POINTS)
        fb = model.frequency_basis(grid)
        freq = fb @ coeffs
        freq_err = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', fb, cov, fb), 0))

        # And of its O-C, counted from T0 - pinned at a maximum of light, and
        # so known exactly there by construction, the uncertainty growing with
        # the cycles away from it
        pb = model.phase_basis(grid) - model.phase_basis(t0)[0]
        offset_err = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', pb, cov, pb), 0))

        # Rate of change of the period in the middle of the data
        h = 1.0
        db = (model.frequency_basis(model.tref + h) - model.frequency_basis(model.tref - h))[0] / (2 * h)
        bm = model.frequency_basis(model.tref)[0]
        f_mid = float(bm @ coeffs)
        f_mid_err = float(np.sqrt(max(bm @ cov @ bm, 0)))
        fdot = float(db @ coeffs)
        fdot_err = float(np.sqrt(max(db @ cov @ db, 0)))

        if detrend_window and 1.0 / f_mid > detrend_window / 3.0:
            notes.append(f'The trend window ({detrend_window:g} days) is less than three times the '
                         f'period, so the trend takes part of the signal along with it.')

        if reduced > 1e3:
            notes.append(f'Reduced chi-squared {reduced:.0f} - the model is nowhere near the data. '
                         'Is the starting period the right one?')

        result = {
            'model': model.describe(),
            'kind': 'spline' if knot_days else 'polynomial',
            'order': order if not knot_days else None,
            'knot_days': knot_days,
            'harmonics': PERIOD_HARMONICS,
            'stages': scored,
            'chi2': round(chi2, 2),
            'reduced_chi2': round(reduced, 4),
            'n_points': int(npoints),
            'n_series': len(loaded),
            'time_span': span,
            'detrend_window': float(detrend_window),
            'oc_window': oc_window,
            'mjd_min': mjd_min,
            'mjd_max': mjd_max,
            # The reference ephemeris the O-C is against
            'period': float(period),
            'epoch': round(float(t0), 6),
            # The model at the middle of the data
            'tref': model.tref,
            'period_tref': 1.0 / f_mid,
            'period_tref_err': f_mid_err / f_mid ** 2,
            'pdot': -fdot / f_mid ** 2,
            'pdot_err': fdot_err / f_mid ** 2,
            'curve': {
                'mjd': [round(float(_), 4) for _ in grid],
                'period': [round(float(1.0 / _), 8) for _ in freq],
                'period_err': [round(float(e / f ** 2), 8) for f, e in zip(freq, freq_err)],
                # O-C of the model, in cycles - what the viewer folds with
                'offset': [round(float(_), 8) for _ in offset(grid)],
                'offset_err': [round(float(_), 8) for _ in offset_err],
            },
            'oc': oc_points,
            # Chi-squared per window of the O-C about the model
            'oc_scatter': oc_scatter,
            # How much worse than white the residuals of the fit are, which the
            # errors of the model are scaled by
            'red_noise': round(red_noise, 2),
            'detrended': detrended,
            # The period within each window, on its own
            'local': {
                'mjd': [round(_, 4) for _ in lt],
                'period': [round(float(1.0 / f), 8) for f in lf],
                'period_err': [round(float(e / f ** 2), 8) for f, e in zip(lf, le)],
            },
        }

        if notes:
            result['warnings'] = notes

        return JsonResponse(result)

    except json.JSONDecodeError as e:
        return JsonResponse({'error': 'Invalid JSON in request body', 'details': str(e)}, status=400)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'error': 'Unexpected error during the period change fit',
            'details': str(e),
            'traceback': traceback.format_exc()
        }, status=500)
