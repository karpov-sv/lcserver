from django.http import JsonResponse, Http404
from django.template.response import TemplateResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404

import os
import glob

import numpy as np
from astropy.table import Table

from . import models
from . import surveys


# Lines worth having drawn on an optical spectrum, in Angstrom, air
# wavelengths. Kept short on purpose: a spectrum covered in labels is harder to
# read than one with none.
SPECTRAL_LINES = [
    ('Ca II K', 3933.7), ('Ca II H', 3968.5),
    ('H-delta', 4101.7), ('H-gamma', 4340.5), ('H-beta', 4861.3),
    ('Mg I b', 5175.0), ('Na I D', 5892.9), ('H-alpha', 6562.8),
    ('Ca II', 8542.1),
]

# Bands where the atmosphere absorbs rather than the star, which is worth
# saying before someone reads one as a feature of the object
TELLURIC_BANDS = [(6860, 6920), (7590, 7700), (8100, 8400), (9300, 9650)]

# A jump this many times the typical spacing is a gap between segments rather
# than a step along one - LAMOST's medium-resolution arms lie 890 A apart where
# the points within an arm are 0.14 A apart, so the two are not close
SPECTRUM_GAP_FACTOR = 20

# Spectra are drawn against each other, so what matters is their shape rather
# than their calibration - the fluxes of the three sources here differ by
# twelve orders of magnitude
SPECTRUM_PALETTE = ['#2980b9', '#c0392b', '#16a085', '#8e44ad', '#e67e22',
                    '#2c3e50', '#27ae60', '#d35400']


def load_spectrum_data(basepath):
    """Every spectrum written for a target, on one wavelength scale.

    Each source says in the registry where its spectra are and what its
    wavelengths are in; the tables carry a wavelength and a flux column by
    convention, and the wavelengths are put into Angstrom here so that a Gaia
    XP spectrum in nanometres can be laid over a DESI one.
    """
    spectra = []
    index = 0

    for source_id, survey_config in surveys.SURVEY_SOURCES.items():
        pattern = survey_config.get('spectrum_files')

        if not pattern:
            continue

        scale = survey_config.get('spectrum_wavelength_scale') or 1.0

        # A source with several spectra tells them apart by shade, counting
        # from its own first one rather than from everything drawn so far
        palette = (survey_config.get('spectrum_palette')
                   or ([survey_config['spectrum_color']]
                       if survey_config.get('spectrum_color') else None)
                   or SPECTRUM_PALETTE)

        for number, path in enumerate(sorted(glob.glob(os.path.join(basepath, pattern)))):
            try:
                data = Table.read(path, format='ascii.commented_header')
            except Exception:
                continue

            if 'wavelength' not in data.colnames or 'flux' not in data.colnames:
                continue

            wavelength = np.asarray(data['wavelength'], dtype=float) * scale
            flux = np.asarray(data['flux'], dtype=float)

            good = np.isfinite(wavelength) & np.isfinite(flux)

            if not np.sum(good):
                continue

            wavelength, flux = wavelength[good], flux[good]

            order = np.argsort(wavelength)
            wavelength, flux = wavelength[order], flux[order]

            # Where the spectrum comes in separate pieces - the two arms of a
            # medium-resolution LAMOST observation are 890 A apart - a null
            # breaks the line rather than letting it run straight across the
            # gap, which would draw a feature that is not there
            steps = np.diff(wavelength)
            breaks = (np.where(steps > SPECTRUM_GAP_FACTOR * np.median(steps))[0]
                      if len(steps) else np.array([], dtype=int))

            # What the file says beyond what the pattern already fixed: the
            # part of its name that the wildcard stood for, so that one
            # spectrum per source is named for the source alone and several
            # are told apart by whatever distinguishes them
            stem = os.path.splitext(os.path.basename(path))[0]
            fixed = os.path.splitext(pattern)[0].split('*')[0]
            rest = stem[len(fixed):] if '*' in pattern and stem.startswith(fixed) else ''

            label = survey_config.get('spectrum_label') or survey_config['short_name']

            if rest:
                label += ' ' + rest.strip('_').replace('_', ' ')

            # Divided by the median rather than left as measured: these are
            # compared by shape, and their calibrations have nothing in common
            median = float(np.median(flux))

            spectra.append({
                'source_id': source_id,
                'label': label,
                'file': os.path.basename(path),
                'color': palette[number % len(palette)],
                'wavelength': [round(float(_), 3) for _ in wavelength],
                # Significant figures rather than decimal places: Gaia
                # publishes XP in W/nm/m2, where six decimals is every value
                # rounded to zero and the spectrum drawn along the axis
                'flux': [float(f'{_:.7g}') for _ in flux],
                # Indices after which the line should be broken
                'breaks': [int(_) for _ in breaks],
                'median': median,
                'n_points': int(len(wavelength)),
                'wavelength_min': float(wavelength.min()),
                'wavelength_max': float(wavelength.max()),
            })

            index += 1

    return spectra


def has_spectra(basepath):
    """Whether a target has anything for the spectral viewer to show."""
    for survey_config in surveys.SURVEY_SOURCES.values():
        pattern = survey_config.get('spectrum_files')

        if pattern and glob.glob(os.path.join(basepath, pattern)):
            return True

    return False


@login_required
def target_spectrum(request, id):
    """Interactive viewer for every spectrum a target has."""
    target = get_object_or_404(models.Target, id=id)

    if not target.can_view(request.user):
        raise Http404

    return TemplateResponse(request, 'spectrum_viewer.html', context={
        'target': target,
        'target_id': id,
        'lines': SPECTRAL_LINES,
    })


@login_required
@require_http_methods(["GET"])
def load_spectrum_json(request, id):
    """The spectra themselves, fetched after the page so it appears at once."""
    target = get_object_or_404(models.Target, id=id)

    if not target.can_view(request.user):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    spectra = load_spectrum_data(target.path())

    return JsonResponse({
        'spectra': spectra,
        'lines': [{'label': _[0], 'wavelength': _[1]} for _ in SPECTRAL_LINES],
        'telluric': [{'from': _[0], 'to': _[1]} for _ in TELLURIC_BANDS],
        'count': len(spectra),
    })
