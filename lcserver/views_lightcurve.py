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


# Light curve sources configuration (matching processing.py combined_lc_rules)
# Note: TESS is excluded as it uses normalized flux instead of magnitudes
LIGHTCURVE_SOURCES = {
    'asas': {'name': 'ASAS-SN', 'filename': 'asas.vot', 'mag': 'mag_g', 'err': 'mag_err', 'filter': 'phot_filter', 'color': '#1f77b4'},
    'ztf': {'name': 'ZTF', 'filename': 'ztf.vot', 'mag': 'mag_g', 'err': 'magerr', 'filter': 'zg', 'color': '#ff7f0e'},
    'ps1': {'name': 'Pan-STARRS', 'filename': 'ps1.vot', 'mag': 'mag_g', 'err': 'magerr', 'filter': 'g', 'color': '#2ca02c'},
    'dasch': {'name': 'DASCH', 'filename': 'dasch.vot', 'mag': 'mag_g', 'err': 'magerr', 'filter': None, 'color': '#d62728'},
    'applause': {'name': 'APPLAUSE', 'filename': 'applause.vot', 'mag': 'mag_g', 'err': 'magerr', 'filter': None, 'color': '#9467bd'},
}

# Flux-based sources (TESS)
FLUX_SOURCES = {
    'tess': {
        'name': 'TESS',
        'pattern': 'tess_lc_*.vot',
        'flux': 'flux',
        'err': 'flux_err',
        'quality': 'quality',
        'color_palette': ['#e74c3c', '#8e44ad', '#3498db', '#e67e22', '#1abc9c'],
    }
}


def load_magnitude_data(basepath):
    """Load magnitude-based light curve data from multiple surveys"""
    lightcurve_data = []

    for source_id, rules in LIGHTCURVE_SOURCES.items():
        fullname = os.path.join(basepath, rules['filename'])

        if not os.path.exists(fullname):
            continue

        try:
            data = Table.read(fullname)

            # Convert MJD to datetime
            data['time'] = Time(data['mjd'], format='mjd')

            # Get time in MJD for plotting
            x = data['mjd']
            y = data[rules.get('mag', 'mag_g')]
            dy = data[rules.get('err', 'magerr')]

            # Handle filters if present
            if rules.get('filter') and rules['filter'] in data.colnames:
                filter_col = data[rules['filter']]
            else:
                filter_col = np.repeat(rules.get('filter', '') if rules.get('filter') else '', len(data))

            # Group by filter
            unique_filters = np.unique(filter_col)

            for filt in unique_filters:
                idx = filter_col == filt
                idx &= np.isfinite(x)
                idx &= np.isfinite(y)

                if not np.sum(idx):
                    continue

                # Create label
                label = rules['name']
                if filt:
                    label += f' {filt}'

                # Prepare data for this series
                # Convert times to ISO format strings for JavaScript
                time_iso = [data['time'][i].iso for i in range(len(data)) if idx[i]]

                series_data = {
                    'source_id': source_id,
                    'label': label,
                    'filter': str(filt) if filt else '',
                    'color': rules.get('color', '#000000'),
                    'mjd': x[idx].tolist(),
                    'datetime': time_iso,
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

    rules = FLUX_SOURCES['tess']
    color_palette = rules['color_palette']

    # Find all TESS lightcurve files
    pattern = os.path.join(basepath, rules['pattern'])
    tess_files = glob.glob(pattern)

    # Parse filename to extract sector, author, exptime
    filename_pattern = re.compile(r'tess_lc_(\d+)_([^_]+)_(\d+)\.vot')

    for i, filepath in enumerate(sorted(tess_files)):
        filename = os.path.basename(filepath)
        match = filename_pattern.match(filename)

        if not match:
            continue

        sector = int(match.group(1))
        author = match.group(2)
        exptime = int(match.group(3))

        try:
            data = Table.read(filepath)

            # Check for required columns
            if 'mjd' not in data.colnames or rules['flux'] not in data.colnames:
                continue

            # Convert MJD to datetime
            data['time'] = Time(data['mjd'], format='mjd')

            # Get flux data
            x = data['mjd']
            flux = data[rules['flux']]

            # Handle missing flux_err column
            if rules['err'] in data.colnames:
                flux_err = data[rules['err']]
            else:
                flux_err = np.zeros_like(flux)

            # Filter bad data
            idx = np.isfinite(x) & np.isfinite(flux)

            # Filter by quality flag if present
            if rules['quality'] in data.colnames:
                quality = data[rules['quality']]
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

            # Convert times to ISO format strings for JavaScript
            time_iso = [data['time'][i].iso for i in range(len(data)) if idx[i]]

            series_data = {
                'source_id': 'tess',
                'label': label,
                'sector': sector,
                'author': author,
                'exptime': exptime,
                'color': color,
                'mjd': x[idx].tolist(),
                'datetime': time_iso,
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

        # Check for magnitude data
        has_magnitude_data = any(
            os.path.exists(os.path.join(basepath, rules['filename']))
            for rules in LIGHTCURVE_SOURCES.values()
        )

        # Check for TESS flux data
        tess_pattern = os.path.join(basepath, FLUX_SOURCES['tess']['pattern'])
        has_flux_data = bool(glob.glob(tess_pattern))

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
    """Fit period using Lomb-Scargle multiband periodogram"""
    target = models.Target.objects.get(id=id)

    # Check permissions
    if not (request.user.is_staff or target.user == request.user):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    try:
        # Parse request data
        data = json.loads(request.body)
        series_list = data.get('series', [])

        if not series_list:
            return JsonResponse({'error': 'No series data provided'}, status=400)

        # Collect all data points from visible series
        times = []
        values = []
        errors = []
        bands = []

        for i, series in enumerate(series_list):
            t = np.array(series['mjd'])
            y = np.array(series['values'])
            dy = np.array(series['errors'])

            # Filter out non-finite values
            valid = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy)

            if not np.any(valid):
                continue

            times.append(t[valid])
            values.append(y[valid])
            errors.append(dy[valid])
            bands.append(np.full(np.sum(valid), i, dtype=int))

        if not times:
            return JsonResponse({'error': 'No valid data points to fit'}, status=400)

        # Concatenate all data
        t_all = np.concatenate(times)
        y_all = np.concatenate(values)
        dy_all = np.concatenate(errors)
        bands_all = np.concatenate(bands)

        # Create LombScargleMultiband object
        ls = LombScargleMultiband(t_all, y_all, bands_all, dy=dy_all)

        print('starting the fit')
        # Compute periodogram with automatic frequency grid
        # Use autopower for sensible defaults
        freq, power = ls.autopower(
            minimum_frequency=0.01,  # Minimum period ~100 days
            maximum_frequency=10.0,  # Maximum period ~0.1 days
            samples_per_peak=10,
            method="fast",
            sb_method="fastnifty",
        )

        print('fit finished')
        # Find best period
        best_idx = np.argmax(power)
        best_freq = freq[best_idx]
        best_period = 1.0 / best_freq
        best_power = power[best_idx]

        # Compute false alarm probability (if single band, use LombScargle)
        # Note: LombScargleMultiband doesn't have FAP calculation implemented
        fap = None
        if len(times) == 1:
            # Use single-band LombScargle for FAP calculation
            from astropy.timeseries import LombScargle
            ls_single = LombScargle(t_all, y_all, dy=dy_all)
            fap = ls_single.false_alarm_probability(best_power)
        else:
            # For multiband, FAP is not available
            # Could use bootstrap or other methods, but skip for now
            pass

        # Estimate epoch (time of maximum) using phase folding
        # Use median time as initial guess
        epoch = np.median(t_all)

        result = {
            'period': float(best_period),
            'epoch': float(epoch),
            'power': float(best_power),
            'frequency': float(best_freq),
            'n_points': int(len(t_all)),
            'n_series': len(times),
        }

        if fap is not None:
            result['fap'] = float(fap)

        return JsonResponse(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'error': str(e),
            'traceback': traceback.format_exc()
        }, status=500)
