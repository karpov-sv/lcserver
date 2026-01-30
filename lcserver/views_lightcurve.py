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
import nifty_ls
from astropy.timeseries import LombScargleMultiband

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

            # Convert MJD to datetime
            data['time'] = Time(data['mjd'], format='mjd')

            # Get time in MJD for plotting
            x = data['mjd']

            # Use registry values with defaults
            mag_col = survey_config.get('lc_mag_column', 'mag_g')
            err_col = survey_config.get('lc_err_column', 'magerr')
            filter_col_name = survey_config.get('lc_filter_column')
            color = survey_config.get('lc_color', '#000000')

            y = data[mag_col]
            dy = data[err_col]

            # Handle filters if present
            if filter_col_name and filter_col_name in data.colnames:
                filter_col = data[filter_col_name]
            else:
                filter_col = np.repeat('', len(data))

            # Group by filter
            unique_filters = np.unique(filter_col)

            for filt in unique_filters:
                idx = filter_col == filt
                idx &= np.isfinite(x)
                idx &= np.isfinite(y)

                if not np.sum(idx):
                    continue

                # Create label
                label = survey_config['short_name']
                if filt:
                    label += f' {filt}'

                # Prepare data for this series
                # Note: datetime conversion moved to frontend for performance
                # Use numpy array indexing for efficiency
                series_data = {
                    'source_id': source_id,
                    'label': label,
                    'filter': str(filt) if filt else '',
                    'color': color,
                    'mjd': x[idx].tolist(),
                    'mag': y[idx].tolist(),
                    'magerr': dy[idx].tolist(),
                    'n_points': int(np.sum(idx)),
                }

                lightcurve_data.append(series_data)

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
    if not (request.user.is_staff or target.user == request.user):
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
    if not (request.user.is_staff or target.user == request.user):
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


@login_required
@require_http_methods(["POST"])
def fit_period(request, id):
    """Fit period using Lomb-Scargle multiband periodogram with comprehensive error handling"""
    target = models.Target.objects.get(id=id)

    # Check permissions
    if not (request.user.is_staff or target.user == request.user):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    try:
        # Parse request data
        data = json.loads(request.body)
        series_list = data.get('series', [])
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

        # Collect all data points from visible series
        times = []
        values = []
        errors = []
        bands = []

        for i, series in enumerate(series_list):
            try:
                t = np.array(series['mjd'])
                y = np.array(series['values'])
                dy = np.array(series['errors'])
            except (KeyError, ValueError) as e:
                return JsonResponse({
                    'error': 'Invalid series data format',
                    'details': f'Series {i}: {str(e)}'
                }, status=400)

            # Filter out non-finite values
            valid = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)

            if not np.any(valid):
                continue

            times.append(t[valid])
            values.append(y[valid])
            errors.append(dy[valid])
            bands.append(np.full(np.sum(valid), i, dtype=int))

        if not times:
            return JsonResponse({
                'error': 'No valid data points to fit',
                'details': 'All data points were invalid (NaN, Inf, or non-positive errors)'
            }, status=400)

        # Concatenate all data
        t_all = np.concatenate(times)
        y_all = np.concatenate(values)
        dy_all = np.concatenate(errors)
        bands_all = np.concatenate(bands)

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

        # Create LombScargleMultiband object
        try:
            ls = LombScargleMultiband(t_all, y_all, bands_all, dy=dy_all)
        except Exception as e:
            return JsonResponse({
                'error': 'Failed to create periodogram object',
                'details': str(e)
            }, status=400)

        # Convert period range to frequency range
        # frequency = 1 / period, so:
        # minimum_frequency = 1 / period_max
        # maximum_frequency = 1 / period_min
        minimum_frequency = 1.0 / period_max
        maximum_frequency = 1.0 / period_min

        print(f'Starting period fit: {period_min:.3f} - {period_max:.3f} days ({minimum_frequency:.6f} - {maximum_frequency:.6f} 1/day)')
        print(f'Data: {n_points} points, span {time_span:.2f} days, {len(times)} series')

        # Compute periodogram with automatic frequency grid
        try:
            freq, power = ls.autopower(
                minimum_frequency=minimum_frequency,
                maximum_frequency=maximum_frequency,
                samples_per_peak=10,
                method="fast",
                sb_method="fastnifty",
            )
        except Exception as e:
            return JsonResponse({
                'error': 'Periodogram computation failed',
                'details': str(e)
            }, status=500)

        print('Fit finished')

        # Validate periodogram output
        if len(freq) == 0 or len(power) == 0:
            return JsonResponse({
                'error': 'Periodogram computation failed',
                'details': 'Empty frequency or power array'
            }, status=500)

        if not np.any(np.isfinite(power)):
            return JsonResponse({
                'error': 'Periodogram computation failed',
                'details': 'All power values are invalid (NaN or Inf)'
            }, status=500)

        # Find best period
        best_idx = np.argmax(power)
        best_freq = freq[best_idx]
        best_period = 1.0 / best_freq
        best_power = power[best_idx]

        # Error detection: Check if fit converged to period limits
        # Allow 5% tolerance at edges
        period_tolerance = 0.05  # 5% tolerance
        lower_threshold = period_min * (1 + period_tolerance)
        upper_threshold = period_max * (1 - period_tolerance)

        warnings = []

        if best_period < lower_threshold:
            return JsonResponse({
                'error': 'Period fit converged to lower limit',
                'details': f'Best period ({best_period:.4f} days) is at the lower edge of search range ({period_min:.4f} days). Try decreasing minimum period or the signal may have a shorter period.',
                'best_period': float(best_period),
                'power': float(best_power),
                'period_min': float(period_min),
                'period_max': float(period_max),
            }, status=400)

        if best_period > upper_threshold:
            return JsonResponse({
                'error': 'Period fit converged to upper limit',
                'details': f'Best period ({best_period:.4f} days) is at the upper edge of search range ({period_max:.4f} days). Try increasing maximum period or the signal may have a longer period.',
                'best_period': float(best_period),
                'power': float(best_power),
                'period_min': float(period_min),
                'period_max': float(period_max),
            }, status=400)

        # Warn if best period is close to data span (aliasing risk)
        if best_period > 0.8 * time_span:
            warnings.append(f'Best period ({best_period:.2f} days) is close to data span ({time_span:.2f} days). Period may be poorly constrained.')

        # Check if power is suspiciously low (weak or no periodicity)
        # Typical significant peaks have power > 0.1, but this is data-dependent
        if best_power < 0.05:
            warnings.append(f'Low periodogram power ({best_power:.4f}). Signal may be very weak or non-periodic.')

        # Compute false alarm probability (if single band, use LombScargle)
        # Note: LombScargleMultiband doesn't have FAP calculation implemented
        fap = None
        if len(times) == 1:
            try:
                # Use single-band LombScargle for FAP calculation
                from astropy.timeseries import LombScargle
                ls_single = LombScargle(t_all, y_all, dy=dy_all)
                fap = ls_single.false_alarm_probability(best_power)

                # Warn if FAP is high (not significant)
                if fap > 0.01:  # 1% FAP threshold
                    warnings.append(f'High false alarm probability ({fap:.4f}). Detection may not be significant.')
            except Exception as e:
                # FAP calculation failed, but don't fail the whole fit
                print(f'FAP calculation failed: {e}')
                pass
        else:
            # For multiband, FAP is not available
            # Could use bootstrap or other methods, but skip for now
            pass

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
        }

        if fap is not None:
            result['fap'] = float(fap)

        if warnings:
            result['warnings'] = warnings

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
