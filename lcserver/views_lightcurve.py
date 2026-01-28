from django.http import HttpResponse
from django.template.response import TemplateResponse
from django.contrib.auth.decorators import login_required

import os
import json

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


@login_required
def target_lightcurve(request, id):
    """Interactive light curve viewer using Plotly"""
    target = models.Target.objects.get(id=id)

    # Check permissions
    if not (request.user.is_staff or target.user == request.user):
        return HttpResponse('Forbidden', status=403)

    basepath = target.path()

    # Load all available light curve data
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
                series_data = {
                    'source_id': source_id,
                    'label': label,
                    'filter': str(filt) if filt else '',
                    'color': rules.get('color', '#000000'),
                    'mjd': x[idx].tolist(),
                    'mag': y[idx].tolist(),
                    'magerr': dy[idx].tolist(),
                    'n_points': int(np.sum(idx)),
                }

                lightcurve_data.append(series_data)

        except Exception as e:
            # Skip files that can't be read
            continue

    context = {
        'target': target,
        'lightcurve_data': json.dumps(lightcurve_data),
        'target_id': id,
    }

    return TemplateResponse(request, 'lightcurve_viewer.html', context=context)
