from django.http import HttpResponse, JsonResponse
from django.template.response import TemplateResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods

import os
import json
import glob
import re

import numpy as np
from astropy.table import Table
from astropy.time import Time

from . import models
from . import surveys
from .processing.utils import mission_quality_mask, quality_level
from .processing.periods import (
    DYNAMIC_AUTO_WINDOW,
    OC_EXCESS_SCATTER,
    PERIOD_AMBIGUOUS_DCHI2,
    PERIOD_CHANGE_CURVE_POINTS,
    PERIOD_CHANGE_HARMONICS,
    PERIOD_CHANGE_MAX_COEFFS,
    PERIOD_CHANGE_MAX_ORDER,
    PERIOD_CHANGE_SPLINE_START,
    PERIOD_HARMONICS,
    PERIOD_MAX_NTERMS,
    MINIMA_CLIP_SIGMA,
    PhaseModel,
    alias_kind,
    choose_harmonics,
    connect_phases,
    detrend_series,
    dynamic_power,
    eclipse_minima,
    eclipse_profile,
    find_period,
    fit_frequencies,
    fit_minima,
    fit_phase_model,
    fourier_parameters,
    frequency_columns,
    group_coefficients,
    harmonics,
    local_periods,
    measure_oc,
    minima_periods,
    refine_peaks,
    rotate_harmonics,
)


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


def _group_name(group):
    """How a group - a survey in one filter - is named to a reader."""
    source, band, kind = group
    config = surveys.SURVEY_SOURCES.get(source, {})
    name = config.get('short_name') or source or '?'
    if band:
        name += ' ' + band
    if kind and kind != 'native':
        name += f' ({kind})'
    return name


def _prewhiten(requested, window_days, frequencies, nharm=PERIOD_HARMONICS):
    """Take the signal of `frequencies` off the series a fit asked for.

    The frequencies are fitted together on the points of the dates searched -
    see `fit_frequencies` - and the harmonics they come to subtracted from
    every point of every series, so that the viewer can show and fold what is
    left. The trend fitted along with them stays in: it is the detrending's
    to take off, when asked, as before.

    Returns the series in the form `_requested_series` gives them, and per
    frequency its refined value and uncertainty and its full amplitude in
    each group, peak to peak. A series whose group had too few points to fit
    is left as it was.
    """
    loaded = [(t[s], y[s], dy[s], _series_group(series))
              for index, series, t, y, dy, s in requested]

    freqs, errors, coeffs = fit_frequencies(loaded, window_days, frequencies, nharm)

    out = []
    for index, series, t, y, dy, s in requested:
        c = coeffs.get(_series_group(series))
        if c is not None:
            y = y - frequency_columns(t, freqs, nharm) @ c
        out.append((index, series, t, y, dy, s))

    phase = np.arange(200) / 200.0
    removed = []
    for k, (f, e) in enumerate(zip(freqs, errors)):
        amplitudes = {}
        for group, c in coeffs.items():
            curve = harmonics(phase, 1.0, nharm) @ c[2 * nharm * k:2 * nharm * (k + 1)]
            amplitudes[_group_name(group)] = round(float(np.ptp(curve)), 5)
        removed.append({
            'frequency': float(f),
            'frequency_err': float(e),
            'period': float(1.0 / f),
            'period_err': float(e / f ** 2),
            'amplitudes': amplitudes,
        })

    return out, removed


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


def _prewhiten_request(requested, window_days, data):
    """The series with the frequencies a request asks to remove taken off -
    see `_prewhiten` - and what was removed; as they were when none is."""
    try:
        frequencies = [float(_) for _ in data.get('prewhiten') or []]
    except (TypeError, ValueError):
        raise _BadRequest('Invalid frequencies to remove', 'They must be numbers')

    if not frequencies:
        return requested, []

    if not all(np.isfinite(_) and _ > 0 for _ in frequencies):
        raise _BadRequest('Invalid frequencies to remove', 'They must be positive')

    try:
        return _prewhiten(requested, window_days, frequencies)
    except ValueError as e:
        raise _BadRequest('Prewhitening failed', str(e))


def _series_group(series):
    """The survey and filter a series is in, which its harmonics are shared within."""
    return (series.get('source_id'), series.get('filter'), series.get('kind'))


@login_required
@require_http_methods(["POST"])
def fit_period(request, id):
    """Fit period using Lomb-Scargle multiband periodogram with comprehensive error handling"""
    # Thousands of products of small matrices, in the peak refit and in
    # prewhitening, which OpenBLAS spreads over threads at a cost of
    # milliseconds each - see fit_period_change
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api='blas'):
        return _fit_period(request, id)


def _fit_period(request, id):
    target = models.Target.objects.get(id=id)

    # Check permissions
    if not target.can_view(request.user):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    try:
        data = json.loads(request.body)
        mode = data.get('mode', 'magnitude')
        period_min = data.get('period_min', 0.1)  # Default: 0.1 days
        period_max = data.get('period_max', 100)  # Default: 100 days

        # How the period is searched for - see find_period
        method = data.get('method', 'ls')
        try:
            nterms = int(data.get('nterms') or 1)
        except (TypeError, ValueError):
            nterms = 0
        if method not in ('ls', 'bls') or not 1 <= nterms <= PERIOD_MAX_NTERMS:
            return JsonResponse({'error': 'Invalid search method',
                                 'details': f'The method is ls, with 1 to {PERIOD_MAX_NTERMS} harmonics, or bls'},
                                status=400)
        # The refit of the highest peaks follows a shape at least as detailed
        # as the search did
        nharm = max(PERIOD_HARMONICS, nterms)

        try:
            detrend_window, mjd_min, mjd_max = _fit_options(data)
            requested = _requested_series(target, data.get('series', []), mode,
                                          mjd_min, mjd_max)
            requested, removed = _prewhiten_request(requested, detrend_window, data)
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

        # What detrending and prewhitening left behind, so the viewer can
        # show and fold the same points the periodogram was computed from
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
            trend = None
            if detrend_window:
                y, trend = detrend_series(t, y, detrend_window,
                                          subtract=(mode != 'flux'))

            if trend is not None or removed:
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
            pg = find_period(t_all, y_all, dy_all, period_min, period_max,
                             method=method, nterms=nterms, magnitudes=(mode != 'flux'))
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
        # failing the fit over: the periodogram's own order is still an answer.
        # Not for a box either - harmonics describe an eclipse badly, and would
        # rank a box's periods by how sinusoidal the light curve is at them
        highest = pg['peaks'][0]
        try:
            if method == 'bls':
                peaks, scale = pg['peaks'], None
            else:
                peaks, scale = refine_peaks(loaded, pg['peaks'], detrend_window,
                                            time_span, nharm=nharm)
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

        # The shape of the light curve at the best period, band by band - not
        # of a box, which harmonics describe badly
        shape = {}
        if method != 'bls':
            try:
                shape = {_group_name(g): {k: round(v, 5) for k, v in fourier_parameters(c, nharm).items()}
                         for g, c in group_coefficients(loaded, detrend_window, best_freq, nharm).items()}
            except Exception:
                import traceback
                traceback.print_exc()

        if scale is not None:
            if best.get('periodogram_period') != highest['period']:
                notes.append(
                    f'The highest periodogram peak is at {highest["period"]:.6f} days, '
                    f'but fitting the trend and {nharm} harmonics of the period '
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
        # Typical significant peaks have power > 0.1, but this is data-dependent.
        # The power of a box is a detection efficiency, on another scale
        # altogether
        if method == 'ls' and best_power < 0.05:
            notes.append(f'Low periodogram power ({best_power:.4f}). Signal may be very weak or non-periodic.')

        if fap is not None and fap > 0.01:  # 1% FAP threshold
            notes.append(f'High false alarm probability ({fap:.4f}). Detection may not be significant.')

        if pg['truncated']:
            notes.append(f'Frequency grid truncated to {pg["nfreq"]} samples. Narrow peaks may be missed - narrow the search range to resolve them.')

        # Two eclipses alike make the light curve repeat at half the orbit
        if method == 'bls':
            notes.append('An eclipsing binary with two eclipses of similar depth is found at half '
                         'its orbital period - fold at twice the period (×2) to see whether the '
                         'eclipses alternate in depth.')

        # Estimate epoch (time of maximum) using phase folding
        # Use median time as initial guess
        epoch = np.median(t_all)

        # Build result
        result = {
            'period': float(best_period),
            'period_err': best.get('period_err'),
            # Amplitude and Fourier parameters per survey and filter
            'shape': shape,
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
            'method': method,
            'nterms': nterms,
            # The periodogram of the sampling, to be centred on the period
            # folded at - where it peaks, that period is seen again
            'window': pg['window'],
            'harmonics': nharm if method != 'bls' else None,
            'aliases': pg['aliases'],
            'nfreq': pg['nfreq'],
            'truncated': pg['truncated'],
            'detrend_window': float(detrend_window),
            # The points the periodogram was actually computed from, so that
            # what is folded and drawn is what was fitted
            'detrended': detrended,
            # The frequencies taken off before the search, refined together
            'prewhitened': removed,
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


def _change_from_light_curve(loaded, detrend_window, period, order, knot_days, oc_window, mode, tmin, tmax,
                             nharm=PERIOD_HARMONICS):
    """The period change fitted to the light curve itself, with `nharm`
    harmonics - or, for 'auto', as many as `choose_harmonics` finds it needs -
    and its O-C measured in windows of `oc_window` days.

    Returns what `_fit_period_change` makes its answer of: the model, its
    coefficients and their covariance, the chi-squared and degrees of
    freedom, each stage as (model, chi-squared) and the number of points
    their BIC counts, the phase of the maximum of light, the O-C as a
    function of the linear phase it is measured from, and the periods of the
    windows on their own.
    """
    notes = []

    def fit_stages(nharm):
        # The period within each window, which needs no count of the cycles
        # between them
        local = local_periods(loaded, detrend_window, 1.0 / period, oc_window, nharm=nharm)
        lt, lf, le = (np.array([_[k] for _ in local]) for k in (0, 4, 5))

        def best_fit(m, starts):
            fits = [fit_phase_model(loaded, detrend_window, m, c, nharm=nharm) for c in starts]
            return min(fits, key=lambda _: _[2])

        # Order by order from a constant period, each started both from the
        # last and from the periods of the windows - a period that changes a
        # lot is out of reach of a start from a constant one, which ends a
        # cycle or more out at the ends. Whichever fits better is kept, and
        # the chi-squared of each order tells what it bought over the last
        stages = []
        model, coeffs = None, None
        for o in range(order + 1):
            m = PhaseModel(tmin, tmax, order=o)
            starts = [m.initial(1.0 / period) if coeffs is None else np.r_[coeffs, 0.0]]
            if o > 0 and len(local) > o + 1:
                starts.append(m.from_frequencies(lt, lf, le))
            fitted = best_fit(m, starts)
            model, coeffs = m, fitted[0]
            stages.append((m, fitted))

        target_model = model
        if knot_days:
            target_model = PhaseModel(tmin, tmax, knot_days=knot_days)
            if target_model.ncoef > PERIOD_CHANGE_MAX_COEFFS:
                raise _BadRequest('Too many knots',
                                  f'A knot every {knot_days:g} days makes {target_model.ncoef} coefficients over '
                                  f'{tmax - tmin:.0f} days; at most {PERIOD_CHANGE_MAX_COEFFS} are fitted')

        # And from the phases of the windows, against the template of the
        # last fit, with the cycles between them counted one window at a time
        # - the start that survives the gaps, where the others lose count
        starts = []
        if knot_days or order > 0:
            def reference(t):
                return (np.asarray(t, dtype=float) - model.tref) / period

            windows = measure_oc(loaded, detrend_window, reference, stages[-1][1][4], oc_window, nharm=nharm)
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
        return stages, (lt, lf, le)

    def choose(m, c):
        try:
            return choose_harmonics(loaded, detrend_window, m, c)
        except ValueError as e:
            raise _BadRequest('Too few points for the harmonics', str(e))

    harmonic_scores = None
    if nharm == 'auto':
        # As many harmonics as the light curve needs at the starting period,
        # and again at the phase the fit found with them - which a period that
        # changes a lot is the first to need - fitted again if that is
        # another number
        m0 = PhaseModel(tmin, tmax, order=0)
        nharm, harmonic_scores = choose(m0, m0.initial(1.0 / period))
        stages, (lt, lf, le) = fit_stages(nharm)
        again, scores = choose(stages[-1][0], stages[-1][1][0])
        if again != nharm:
            nharm, harmonic_scores = again, scores
            stages, (lt, lf, le) = fit_stages(nharm)
    else:
        stages, (lt, lf, le) = fit_stages(nharm)
    model = stages[-1][0]

    coeffs, cov, chi2, dof, group_harmonics, red_noise = stages[-1][1]

    # The phase zero moved to the maximum of light of the template of the
    # group with most points, so that T0 is a time of maximum as an
    # ephemeris has it
    sizes = {}
    for t, _, _, g in loaded:
        sizes[g] = sizes.get(g, 0) + len(t)
    main = max(group_harmonics, key=lambda g: sizes.get(g, 0))
    g = np.arange(1000) / 1000.0
    curve = harmonics(g, 1.0, nharm) @ group_harmonics[main]
    zero = float(g[np.argmax(curve) if mode == 'flux' else np.argmin(curve)])
    templates = {k: rotate_harmonics(v_, -zero, nharm)[0] for k, v_ in group_harmonics.items()}

    def measure(linear_phase):
        return [w + ('window', True)
                for w in measure_oc(loaded, detrend_window, linear_phase, templates, oc_window, nharm=nharm)]

    return {
        'model': model, 'coeffs': coeffs, 'cov': cov, 'chi2': chi2, 'dof': dof,
        'stages': [(m, f[2]) for m, f in stages],
        'nbic': sum(len(t) for t, _, _, _ in loaded),
        'zero': zero, 'measure': measure, 'local': (lt, lf, le), 'notes': notes,
        'harmonics': nharm,
        'harmonic_scores': [{'harmonics': n, 'chi2': round(c, 2), 'bic': round(b, 2)}
                            for n, c, b in harmonic_scores] if harmonic_scores else None,
        'red_noise': red_noise,
    }


def _change_from_minima(loaded, detrend_window, period, epoch, order, knot_days, oc_window, mode, tmin, tmax):
    """The period change fitted to the times of the eclipse minima, each
    timed on its own - see `eclipse_minima` - rather than to the light
    curve, which no number of harmonics describes as well as a detached
    eclipsing binary's eclipses can be timed.

    The eclipses are looked for in the light curve folded at the reference
    ephemeris, with its trend taken off first when one is asked for: a slope
    across an eclipse would move its minimum. Returns what
    `_change_from_light_curve` does, the phase zero at a primary minimum,
    and the periods of the windows from the minima in each.
    """
    notes = []
    magnitudes = mode != 'flux'
    frequency = 1.0 / period
    reference = epoch if epoch is not None else tmin

    series = loaded
    if detrend_window:
        series = [(t, detrend_series(t, y, detrend_window, subtract=magnitudes)[0], dy, g)
                  for t, y, dy, g in loaded]

    try:
        profile = eclipse_profile(series, frequency, reference, magnitudes)
    except ValueError as e:
        raise _BadRequest('No eclipses to time', str(e))

    minima = eclipse_minima(series, frequency, reference, profile, magnitudes)
    mt, me, ms, mi = (np.array([_[k] for _ in minima]) for k in (0, 1, 2, 8))
    if len(minima) < max(order + 3, 4):
        raise _BadRequest('Too few eclipse minima',
                          f'Only {len(minima)} eclipses are covered well enough to time. Each needs '
                          f'points along both of its branches and across its bottom, as few but the '
                          f'space missions and dense photometry from the ground have them.')
    ms = ms.astype(bool)

    models = [PhaseModel(tmin, tmax, order=o)
              for o in range(PERIOD_CHANGE_SPLINE_START + 1 if knot_days else order + 1)]
    if knot_days:
        models.append(PhaseModel(tmin, tmax, knot_days=knot_days))
        if models[-1].ncoef > PERIOD_CHANGE_MAX_COEFFS:
            raise _BadRequest('Too many knots',
                              f'A knot every {knot_days:g} days makes {models[-1].ncoef} coefficients over '
                              f'{tmax - tmin:.0f} days; at most {PERIOD_CHANGE_MAX_COEFFS} are fitted')

    # The errors scaled and the outliers clipped about the model asked for,
    # and every order up to it fitted to the same minima with the same
    # errors, so that their chi-squared tells what each order bought
    settled = fit_minima(models[-1], mt, me, ms, frequency, classes=mi)
    errors, keep = settled[7], settled[6]
    stages = [(m, fit_minima(m, mt, errors, ms, frequency, keep=keep)) for m in models[:-1]]
    stages.append((models[-1], settled))

    model = stages[-1][0]
    const, coeffs, cov, chi2, dof, counted, keep, errors = stages[-1][1]
    nconst = len(const)
    delta = float(const[1]) if nconst > 1 else 0.0
    delta_err = float(np.sqrt(max(cov[1, 1], 0))) if nconst > 1 else None

    if not keep.all():
        notes.append(f'{int((~keep).sum())} of the {len(keep)} minima miss the fit by more than '
                     f'{MINIMA_CLIP_SIGMA:g} times the scatter of the rest, and are left out of it - '
                     f'drawn hollow.')

    # The period within each window, from the minima in it alone
    local = minima_periods(mt, errors, counted, keep, frequency, oc_window)
    lt, lf, le = (np.array([_[k] for _ in local]) for k in (0, 4, 5))

    frac = np.where(ms, 0.5 + delta, 0.0)

    def measure(linear_phase):
        out = []
        # With the errors as the fit scaled them to the scatter
        for i, (tm, _, secondary, ta, tb, n, level, beta, _) in enumerate(minima):
            shift = (linear_phase(np.array([tm]))[0] - frac[i] + 0.5) % 1.0 - 0.5
            out.append((tm, ta, tb, n, float(shift), errors[i] * frequency, level, beta,
                        'secondary' if secondary else 'primary', bool(keep[i])))
        return out

    def described(eclipse):
        if eclipse is None:
            return None
        phase, halfwidth, depth = eclipse
        return {'phase': round(phase, 4), 'halfwidth': round(halfwidth, 4), 'depth': round(depth, 5)}

    return {
        'model': model, 'coeffs': coeffs, 'cov': cov[nconst:, nconst:], 'chi2': chi2, 'dof': dof,
        'stages': [(m, f[3]) for m, f in stages],
        'nbic': int(keep.sum()),
        # The phase is a whole number of cycles at a primary minimum
        'zero': -float(const[0]),
        'measure': measure, 'local': (lt, lf, le), 'notes': notes,
        'minima': {
            'primary': int((~ms).sum()), 'secondary': int(ms.sum()), 'used': int(keep.sum()),
            'delta': round(delta, 6) if nconst > 1 else None,
            'delta_err': round(delta_err, 6) if delta_err is not None else None,
            'profile': {k: described(v) for k, v in profile.items()},
        },
    }


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
    to the maximum of light nearest to where it was - or, when the minima of
    an eclipsing binary are what is timed, to the primary minimum.
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

            # What the O-C and the model are measured from: windows of the
            # light curve fitted with its shape, or the eclipse minima
            timing = data.get('timing') or 'windows'
            if timing not in ('windows', 'minima'):
                raise _BadRequest('Invalid timing', 'The O-C is measured in windows or from eclipse minima')

            # How many harmonics describe the light curve in windows, or
            # 'auto' for as many as it needs
            nharm = data.get('harmonics') or PERIOD_HARMONICS
            if nharm != 'auto':
                try:
                    nharm = int(nharm)
                except (TypeError, ValueError):
                    nharm = 0
                if not 1 <= nharm <= max(PERIOD_CHANGE_HARMONICS):
                    raise _BadRequest('Invalid number of harmonics',
                                      f'It is auto, or 1 to {max(PERIOD_CHANGE_HARMONICS)}')

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
            # The frequencies removed for the search are removed here too, so
            # that what is left - a second mode, say - can be followed in turn
            requested, removed = _prewhiten_request(requested, detrend_window, data)
        except _BadRequest as e:
            return e.response()

        loaded = [(t[s], y[s], dy[s], _series_group(series))
                  for index, series, t, y, dy, s in requested]

        # The light curve with its trend taken off and the removed frequencies,
        # as "Fit Period" sends it, so that the fold shows the variation the
        # phase was fitted to
        detrended = []
        if detrend_window or removed:
            for index, series, t, y, dy, s in requested:
                trend = None
                if detrend_window:
                    y, trend = detrend_series(t, y, detrend_window, subtract=(mode != 'flux'))
                if trend is not None or removed:
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

        try:
            if timing == 'minima':
                fit = _change_from_minima(loaded, detrend_window, period, epoch, order, knot_days,
                                          oc_window, mode, tmin, tmax)
            else:
                fit = _change_from_light_curve(loaded, detrend_window, period, order, knot_days,
                                               oc_window, mode, tmin, tmax, nharm=nharm)
        except _BadRequest as e:
            return e.response()

        model, coeffs, cov = fit['model'], fit['coeffs'], fit['cov']
        chi2, dof = fit['chi2'], fit['dof']
        reduced = chi2 / dof
        zero = fit['zero']

        # Each stage scored against the scatter the last one leaves, as the
        # errors of most surveys understate it
        scored = []
        for m, x2 in fit['stages']:
            scored.append({'model': m.describe(), 'ncoef': m.ncoef,
                           'chi2': round(x2, 2),
                           'bic': round(x2 / max(reduced, 1e-12) + m.ncoef * np.log(fit['nbic']), 2)})
        preferred = min(range(len(scored)), key=lambda i: scored[i]['bic'])
        if preferred != len(scored) - 1:
            notes.append(f'The Bayesian information criterion prefers the {scored[preferred]["model"]} '
                         f'to the {scored[-1]["model"]} - the extra freedom is not bought by the data.')

        def model_phase(t):
            return model.phase_basis(t) @ coeffs - zero

        def model_frequency(t):
            return model.frequency_basis(t) @ coeffs

        # The maximum of light, or the primary minimum, nearest to the T0
        # asked for, or to the middle of the data
        t0 = epoch if epoch is not None and tmin <= epoch <= tmax else model.tref
        cycle = float(np.round(model_phase(t0)[0]))
        for _ in range(5):
            t0 -= float((model_phase(t0)[0] - cycle) / model_frequency(t0)[0])

        def linear_phase(t):
            return (np.asarray(t, dtype=float) - t0) / period

        def offset(t):
            """O-C of the model, in cycles of the reference period."""
            return linear_phase(t) - (model_phase(t) - cycle)

        oc = fit['measure'](linear_phase)

        # Each shift taken a whole number of cycles from where the model has it,
        # which is what counts the cycles across the gaps
        oc_points = {'mjd': [], 'mjd_min': [], 'mjd_max': [], 'n': [], 'oc': [],
                     'oc_err': [], 'amplitude': [], 'red_noise': [], 'kind': [], 'used': []}
        for tm, ta, tb, n, shift, err, amp, beta, kind, used in oc:
            shift += np.round(offset(tm)[0] - shift)
            for key, value in zip(oc_points, (round(tm, 6), round(ta, 4), round(tb, 4), n,
                                              round(float(shift * period), 6),
                                              round(err * period, 6), round(amp, 4),
                                              round(beta, 2), kind, used)):
                oc_points[key].append(value)

        used = np.array(oc_points['used'], dtype=bool)
        oc_mjd = np.array(oc_points['mjd'])

        # How far the O-C strays from the model, against its errors. Well
        # above one, the phase wanders more than the noise accounts for -
        # which on a spotted star is the spots, growing, decaying and
        # migrating, and otherwise a model too simple for the period
        oc_scatter = None
        if used.sum() > 1:
            dev = (np.array(oc_points['oc'])[used] / period - offset(oc_mjd[used]))
            oc_scatter = float(np.mean((dev * period / np.array(oc_points['oc_err'])[used]) ** 2))
            if oc_scatter > OC_EXCESS_SCATTER:
                notes.append(f'The O-C scatters about the model {np.sqrt(oc_scatter):.1f} times more than '
                             f'its errors allow (chi-squared {oc_scatter:.1f} per point) - the phase '
                             f'wanders beyond the noise, as spots make it on a spotted star, or the '
                             f'model is too simple for how the period changes.')

        # A point the model misses by a good part of a cycle has its cycle
        # count in doubt, and with it the model around it
        missed = [tm for tm, o, u in zip(oc_points['mjd'], oc_points['oc'], used)
                  if u and abs(o / period - offset(tm)[0]) > 0.25]
        if missed:
            notes.append(f'The model misses {len(missed)} of the O-C points by more than a quarter '
                         f'of a cycle (around MJD {", ".join(f"{_:.0f}" for _ in missed[:5])}'
                         f'{", ..." if len(missed) > 5 else ""}) - the count of cycles there is in '
                         f'doubt. Narrow the dates, or give the model more freedom.')

        if timing != 'minima' and len(oc_points['mjd']) < 3:
            notes.append(f'Only {len(oc_points["mjd"])} windows of {oc_window:g} days had enough '
                         f'points for an O-C - try longer ones.')

        notes.extend(fit['notes'])
        lt, lf, le = fit['local']

        grid = np.linspace(tmin, tmax, PERIOD_CHANGE_CURVE_POINTS)
        fb = model.frequency_basis(grid)
        freq = fb @ coeffs
        freq_err = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', fb, cov, fb), 0))

        # And of its O-C, counted from T0 - pinned at a maximum of light, or
        # a primary minimum, and so known exactly there by construction, the uncertainty growing with
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
            'timing': timing,
            # How many harmonics the light curve was fitted with, and what
            # each number tried scored - none when the minima were timed
            'harmonics': fit.get('harmonics'),
            'harmonic_scores': fit.get('harmonic_scores'),
            'minima': fit.get('minima'),
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
            'prewhitened': removed,
            # Chi-squared per point of the O-C about the model
            'oc_scatter': oc_scatter,
            # How much worse than white the residuals of the fit to the light
            # curve are, which the errors of the model are scaled by
            'red_noise': round(fit['red_noise'], 2) if fit.get('red_noise') is not None else None,
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


@login_required
@require_http_methods(["POST"])
def dynamic_periodogram(request, id):
    """The periodogram in a window sliding along the light curve, for how a
    period - or which of several - comes and goes with time.

    Of the same points "Fit Period" searches: the series and filters asked
    for, within the dates, with the removed frequencies taken off and the
    trend, if asked, too.
    """
    from threadpoolctl import threadpool_limits

    target = models.Target.objects.get(id=id)

    if not target.can_view(request.user):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    try:
        data = json.loads(request.body)
        mode = data.get('mode', 'magnitude')

        try:
            detrend_window, mjd_min, mjd_max = _fit_options(data)
            period_min = _positive(data, 'period_min', 0.1)
            period_max = _positive(data, 'period_max', 100)
            window = float(data['window']) if data.get('window') not in (None, '') else None
            if window is not None and not window > 0:
                raise _BadRequest('Invalid window', 'The window must be a positive number of days')
            if period_max <= period_min:
                raise _BadRequest('Invalid period range',
                                  f'Maximum period ({period_max}) must be greater than minimum period ({period_min})')
            step = float(data.get('step') or 0)
            if step < 0:
                raise _BadRequest('Invalid step', 'The step must be a positive number of days')

            requested = _requested_series(target, data.get('series', []), mode,
                                          mjd_min, mjd_max)
            with threadpool_limits(limits=1, user_api='blas'):
                requested, removed = _prewhiten_request(requested, detrend_window, data)
        except _BadRequest as e:
            return e.response()

        series = []
        for index, s, t, y, dy, searched in requested:
            if detrend_window:
                y, _ = detrend_series(t, y, detrend_window, subtract=(mode != 'flux'))
            series.append((t[searched], y[searched], dy[searched]))

        if not series:
            return JsonResponse({'error': 'No valid data points',
                                 'details': 'No series has points in the dates'}, status=400)

        span = max(t.max() for t, _, _ in series) - min(t.min() for t, _, _ in series)
        if window is None:
            window = round(float(DYNAMIC_AUTO_WINDOW * span), 1)
        step = step or window / 4

        if window > span:
            return JsonResponse({
                'error': 'Window too long',
                'details': f'The window ({window:g} d) is longer than the data ({span:.1f} d) - '
                           'there is nothing to slide it along'}, status=400)

        # A window says nothing of a period it does not hold two cycles of, so
        # the range stops there - the period search's own range, up to 100
        # days by default, would otherwise rule out any window shorter than
        # two hundred
        searched_max = min(period_max, window / 2)
        if searched_max <= period_min:
            return JsonResponse({
                'error': 'Window too short',
                'details': f'A window of {window:g} d holds two cycles only of periods up to '
                           f'{window / 2:g} d, below the shortest searched ({period_min:g} d) - '
                           'lengthen the window'}, status=400)

        result = dynamic_power(series, period_min, searched_max, window, step)
        result.update({
            'window': window,
            'step': (result['mjd'][1] - result['mjd'][0]) if len(result['mjd']) > 1 else step,
            'period_min': period_min,
            'period_max': searched_max,
            # Whether the range was cut short of the one asked for
            'period_max_limited': bool(searched_max < period_max),
            'n_points': int(sum(len(t) for t, _, _ in series)),
            'n_series': len(series),
            'detrend_window': float(detrend_window),
            'prewhitened': removed,
        })
        return JsonResponse(result)

    except json.JSONDecodeError as e:
        return JsonResponse({'error': 'Invalid JSON in request body', 'details': str(e)}, status=400)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'error': 'Unexpected error computing the time-resolved periodogram',
            'details': str(e),
            'traceback': traceback.format_exc()
        }, status=500)
