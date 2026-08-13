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


# Lines worth having drawn, in Angstrom, air wavelengths. Kept short on
# purpose: a spectrum covered in labels is harder to read than one with none.
#
# The optical ones are for the spectrographs. The infrared ones are for
# SPHEREx, which reaches five microns and whose range holds things an optical
# list has no way to mark - the hydrogen series past Paschen, the CO bandheads
# that say a star is cool, and the three-micron ice band that the mission has
# a whole survey named after. Only what is drawn within the range on show, so
# a LAMOST spectrum is not annotated with lines it stops well short of.
SPECTRAL_LINES = [
    ('Ca II K', 3933.7), ('Ca II H', 3968.5),
    ('H-delta', 4101.7), ('H-gamma', 4340.5), ('H-beta', 4861.3),
    ('Mg I b', 5175.0), ('Na I D', 5892.9), ('H-alpha', 6562.8),
    ('Ca II', 8542.1),
    ('He I', 10830.3), ('Pa-beta', 12818.1), ('Pa-alpha', 18751.0),
    ('Br-gamma', 21661.2), ('CO 2-0', 22935.0),
    ('H2O ice', 30500.0), ('PAH', 33000.0), ('CO 1-0', 46700.0),
]

# Bands where the atmosphere absorbs rather than the star, which is worth
# saying before someone reads one as a feature of the object.
#
# Optical only, and deliberately so. The infrared has far heavier telluric
# absorption - the windows between J, H and K are cut by water - but nothing
# here observes through it: the only infrared spectra are SPHEREx's, taken
# from orbit, and shading those would mark an absorption that was never in
# their light. The bands below are for the ground-based spectrographs, LAMOST
# and DESI, which is where the light did come through the air.
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
    """Every spectrum written for a target.

    Nothing is converted here. Each source says in the registry only where its
    spectra are; what is in them is the same whoever wrote it - wavelength in
    Angstrom, flux in erg/s/cm2/A - the sources having each done their own
    conversion when they wrote the file, so that the files are as comparable
    on disk as they are on the screen.
    """
    spectra = []
    index = 0

    for source_id, survey_config in surveys.SURVEY_SOURCES.items():
        # A source with several spectra tells them apart by shade, counting
        # from its own first one rather than from everything drawn so far
        palette = (survey_config.get('spectrum_palette')
                   or ([survey_config['spectrum_color']]
                       if survey_config.get('spectrum_color') else None)
                   or SPECTRUM_PALETTE)

        # What the source's own curve was divided by, so that its individual
        # measurements can be divided by the same thing. Normalised on their
        # own they would sit a little off the curve they came from, the two
        # having different numbers of points at each wavelength, and the whole
        # purpose of drawing them is that they lie about it.
        curve_median = None

        for kind, pattern in (('line', survey_config.get('spectrum_files')),
                              ('points', survey_config.get('spectrum_points'))):
            if not pattern:
                continue

            unticked = survey_config.get('spectrum_hidden')
            unticked = set(glob.glob(os.path.join(basepath, unticked))) if unticked else set()

            for number, path in enumerate(sorted(glob.glob(os.path.join(basepath, pattern)))):
                hidden = path in unticked

                try:
                    data = Table.read(path, format='ascii.commented_header')
                except Exception:
                    continue

                if 'wavelength' not in data.colnames or 'flux' not in data.colnames:
                    continue

                wavelength = np.asarray(data['wavelength'], dtype=float)
                flux = np.asarray(data['flux'], dtype=float)

                # Uncertainties are carried for points and not for curves. A
                # curve of several thousand samples drawn with an error bar on
                # each is a band of ink, and the sources that publish one draw
                # the band themselves; a measurement is not much use without
                # the error on it.
                error = (np.asarray(data['flux_error'], dtype=float)
                         if kind == 'points' and 'flux_error' in data.colnames
                         else None)

                # What each point is, where the source says so. A table of
                # broadband photometry is a dozen surveys in one series, and
                # the viewer shows this under the cursor rather than asking
                # for a colour and a legend entry per survey.
                comment = (np.asarray(data['comment'], dtype=str)
                           if 'comment' in data.colnames else None)

                good = np.isfinite(wavelength) & np.isfinite(flux)

                if not np.sum(good):
                    continue

                wavelength, flux = wavelength[good], flux[good]
                error = error[good] if error is not None else None
                comment = comment[good] if comment is not None else None

                order = np.argsort(wavelength)
                wavelength, flux = wavelength[order], flux[order]
                error = error[order] if error is not None else None
                comment = comment[order] if comment is not None else None

                # Where the spectrum comes in separate pieces - the two arms of a
                # medium-resolution LAMOST observation are 890 A apart - a null
                # breaks the line rather than letting it run straight across the
                # gap, which would draw a feature that is not there. Points are
                # not joined in the first place, so nothing has to be broken.
                steps = np.diff(wavelength)
                breaks = (np.where(steps > SPECTRUM_GAP_FACTOR * np.median(steps))[0]
                          if kind == 'line' and len(steps)
                          else np.array([], dtype=int))

                # What the file says beyond what the pattern already fixed: the
                # part of its name that the wildcard stood for, so that one
                # spectrum per source is named for the source alone and several
                # are told apart by whatever distinguishes them
                stem = os.path.splitext(os.path.basename(path))[0]
                fixed = os.path.splitext(pattern)[0].split('*')[0]
                rest = stem[len(fixed):] if '*' in pattern and stem.startswith(fixed) else ''

                label = (survey_config.get('spectrum_label')
                         or survey_config['short_name'])

                if rest:
                    label += ' ' + rest.strip('_').replace('_', ' ')


                # All four sources are now in the one unit, so this is no longer
                # about units: a spectrum of a fifteenth-magnitude quasar and one
                # of a sixth-magnitude star still differ by a factor of a
                # thousand, and drawn together the fainter is a line along the
                # axis. The viewer offers the division as a choice.
                median = float(np.median(flux))

                if kind == 'points' and curve_median is not None:
                    median = curve_median
                elif kind == 'line' and curve_median is None:
                    curve_median = median

                spectra.append({
                    'source_id': source_id,
                    'label': label,
                    'draw': kind,
                    'file': os.path.basename(path),
                    'color': palette[number % len(palette)],
                    'wavelength': [round(float(_), 3) for _ in wavelength],
                    # Significant figures rather than decimal places: Gaia
                    # publishes XP in W/nm/m2, where six decimals is every value
                    # rounded to zero and the spectrum drawn along the axis
                    'flux': [float(f'{_:.7g}') for _ in flux],
                    'flux_error': ([float(f'{_:.4g}') if np.isfinite(_) else None
                                    for _ in error]
                                   if error is not None else None),
                    'comment': ([str(_) for _ in comment]
                                if comment is not None else None),
                    # Whether it is shown without being asked for
                    'default_visible': not hidden,
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
        for key in ('spectrum_files', 'spectrum_points'):
            pattern = survey_config.get(key)

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
