from django.http import HttpResponse
from django.template.response import TemplateResponse
from django.contrib.auth.decorators import login_required

import os
import json
import glob
import re

import numpy as np
from astropy.table import Table
from astropy.time import Time

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

    basepath = target.path()

    # Determine display mode
    mode = request.GET.get('mode', 'auto')

    # Auto-detect mode if not specified
    if mode == 'auto':
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

    # Load data based on mode
    if mode == 'flux':
        lightcurve_data = load_flux_data(basepath)
        data_mode = 'flux'
    else:
        lightcurve_data = load_magnitude_data(basepath)
        data_mode = 'magnitude'

    # Check if we have any data
    no_data = len(lightcurve_data) == 0

    context = {
        'target': target,
        'lightcurve_data': json.dumps(lightcurve_data),
        'target_id': id,
        'data_mode': data_mode,
        'mode': mode,
        'no_data': no_data,
    }

    return TemplateResponse(request, 'lightcurve_viewer.html', context=context)
