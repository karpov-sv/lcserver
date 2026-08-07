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

from . import models
from . import surveys


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
                    'color': band.get('color') or color,
                    'mjd': x[idx].tolist(),
                    'mag': y[idx].tolist(),
                    'magerr': dy[idx].tolist(),
                    'n_points': int(np.sum(idx)),
                })

        except Exception as e:
            # Skip files that can't be read
            continue

    return lightcurve_data


def load_flux_data(basepath):
    """Load flux-based light curve data from TESS"""
    lightcurve_data = []

    # Color palette for multiple TESS sectors
    color_palette = ['#e74c3c', '#8e44ad', '#3498db', '#e67e22', '#1abc9c']

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

        # Parse filename to extract sector, author, exptime (TESS-specific)
        filename_pattern = re.compile(r'tess_lc_(\d+)_([^_]+)_(\d+)\.vot')

        for i, filepath in enumerate(sorted(source_files)):
            filename = os.path.basename(filepath)
            match = filename_pattern.match(filename)

            if not match:
                continue

            sector = int(match.group(1))
            author = match.group(2)
            exptime = int(match.group(3))

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

                # Filter by quality flag if present
                if quality_col and quality_col in data.colnames:
                    quality = data[quality_col]
                    idx &= (quality == 0)

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
                label = f'TESS Sector {sector} ({author}, {exptime}s)'

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
                    'flux_err': flux_err_normalized[idx].tolist(),
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

    context = {
        'target': target,
        'target_id': id,
        'data_mode': mode,
        'mode': mode,
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
            lightcurve_data = load_flux_data(basepath)
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


@login_required
@require_http_methods(["POST"])
def fit_period(request, id):
    """Fit period using Lomb-Scargle multiband periodogram with comprehensive error handling"""
    target = models.Target.objects.get(id=id)

    # Check permissions
    if not target.can_view(request.user):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    try:
        # Parse request data. The client sends only the selection - which series
        # are visible and how they are filtered - and the data itself is loaded
        # here from the same files the viewer was built from. Sending the points
        # back would make the request grow with the light curve and exceed
        # DATA_UPLOAD_MAX_MEMORY_SIZE (TESS alone is several MB per target).
        data = json.loads(request.body)
        series_list = data.get('series', [])
        mode = data.get('mode', 'magnitude')
        period_min = data.get('period_min', 0.1)  # Default: 0.1 days
        period_max = data.get('period_max', 100)  # Default: 100 days

        if not series_list:
            return JsonResponse({'error': 'No series data provided'}, status=400)

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

        # Load the light curves, exactly as the viewer itself does
        basepath = target.path()
        if mode == 'flux':
            available = load_flux_data(basepath)
            value_key, error_key = 'flux', 'flux_err'
        else:
            available = load_magnitude_data(basepath)
            value_key, error_key = 'mag', 'magerr'

        labels = {}
        for i, series in enumerate(available):
            labels.setdefault(series['label'], i)

        # Collect all data points from visible series
        times = []
        values = []
        errors = []

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
                return JsonResponse({
                    'error': 'Series is no longer available',
                    'details': f'Series {i} ({label}) was not found on the server. '
                               'Reload the page and try again.'
                }, status=400)

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

            if not np.any(valid):
                continue

            t, y, dy = t[valid], y[valid], dy[valid]

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

        best = pg['peaks'][0]
        best_period = best['period']
        best_power = best['power']
        best_freq = best['frequency']
        fap = best['fap']

        notes = []

        # Converging to an edge of the searched range usually means the real
        # period lies outside it. Reported rather than raised: the periodogram
        # comes back along with it and shows the situation better than any
        # message could, and failing outright would hide it.
        period_tolerance = 0.05  # 5% tolerance
        if best_period < period_min * (1 + period_tolerance):
            notes.append(f'Best period ({best_period:.4f} days) is at the lower edge of the search range ({period_min:.4f} days). Try decreasing the minimum period - the signal may have a shorter one.')

        if best_period > period_max * (1 - period_tolerance):
            notes.append(f'Best period ({best_period:.4f} days) is at the upper edge of the search range ({period_max:.4f} days). Try increasing the maximum period - the signal may have a longer one.')

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
            # Decimated periodogram for display, plus its highest peaks
            'periodogram': {'periods': pg['periods'], 'power': pg['power']},
            'peaks': pg['peaks'],
            'aliases': pg['aliases'],
            'nfreq': pg['nfreq'],
            'truncated': pg['truncated'],
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
