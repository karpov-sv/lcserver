from django.http import JsonResponse, Http404
from django.template.response import TemplateResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404

import os
import glob
import re

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

                # How wide the band is, for the sources whose points are bands
                # rather than samples: the catalogue SED, whose W3 point covers
                # five and a half microns, and SPHEREx, whose every exposure is
                # a channel of finite width. Carried as the full width, and
                # halved by the viewer into a bar reaching that far either way.
                band = (np.asarray(data['bandwidth'], dtype=float)
                        if kind == 'points' and 'bandwidth' in data.colnames
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
                band = band[good] if band is not None else None
                comment = comment[good] if comment is not None else None

                order = np.argsort(wavelength)
                wavelength, flux = wavelength[order], flux[order]
                error = error[order] if error is not None else None
                band = band[order] if band is not None else None
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
                    # A source that names its spectra says what this one is;
                    # one that has a spectrum per observation lets the
                    # filename speak, that being what tells them apart
                    named = survey_config.get('spectrum_labels') or {}
                    label += ' ' + named.get(rest.strip('_'),
                                             rest.strip('_').replace('_', ' '))


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
                    'bandwidth': ([round(float(_), 3) if np.isfinite(_) else None
                                   for _ in band]
                                  if band is not None else None),
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


# What a caller may set, and what it becomes. Everything else in a posted
# options object is ignored rather than passed through: the fit reads its
# options straight out of this dictionary, and an unchecked key would be a way
# to reach into it from a request.
FIT_OPTIONS = {
    'grids': lambda v: [str(_) for _ in v][:6],
    'teff_prior': lambda v: _prior(v),
    'logg_prior': lambda v: _prior(v),
    'av_max': lambda v: max(0.0, float(v)),
    'distance': lambda v: float(v),
    'distance_err': lambda v: float(v),
    'nlive': lambda v: int(np.clip(int(v), 50, 4000)),
    'err_floor': lambda v: float(np.clip(float(v), 0, 1)),
    'err_unknown': lambda v: float(np.clip(float(v), 0, 1)),
    'seed': lambda v: int(v),
}

# Priors arrive as ['uniform', low, high] and are rebuilt here rather than
# eval'd or passed on: the kind has to be one the fitter knows, and the bounds
# have to be numbers.
PRIOR_KINDS = ('uniform', 'loguniform', 'normal', 'truncnorm', 'fixed')


def _prior(value):
    kind = str(value[0])
    if kind not in PRIOR_KINDS:
        raise ValueError(f'unknown prior {kind}')
    return [kind] + [float(_) for _ in value[1:]]


@login_required
@require_http_methods(["POST"])
def fit_sed(request, id):
    """Start a photosphere fit over a chosen subset of the SED.

    The client sends what to fit rather than the data - which points are in,
    which grids to use, what to assume - and the points themselves are read
    here from the same file the viewer drew them from.

    A fit takes tens of seconds, so it goes through Celery like an
    acquisition, and for the same reason it takes the target's single task
    slot: a fit running while a source rewrites sed.vot would be reading a
    file out from under itself.
    """
    import json
    import datetime

    from . import celery_tasks

    target = get_object_or_404(models.Target, id=id)

    if not target.can_edit(request.user):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    if target.celery_id is not None:
        return JsonResponse(
            {'error': 'Something is already running for this target'},
            status=409)

    try:
        data = json.loads(request.body or '{}')
    except ValueError:
        return JsonResponse({'error': 'Malformed request'}, status=400)

    source = str(data.get('source') or 'sed.vot')
    if source not in ('sed.vot', 'sed_all.vot'):
        return JsonResponse({'error': 'Unknown SED file'}, status=400)

    if not os.path.exists(os.path.join(target.path(), source)):
        return JsonResponse({'error': f'{source} is not there yet'}, status=400)

    selection = {'source': source}
    for key in ('points', 'exclude'):
        if data.get(key) is not None:
            selection[key] = [str(_) for _ in data[key]]

    options = {}
    try:
        for key, clean in FIT_OPTIONS.items():
            if data.get(key) is not None:
                options[key] = clean(data[key])
    except (TypeError, ValueError, IndexError, KeyError) as e:
        return JsonResponse({'error': f'Bad option: {e}'}, status=400)

    # Named for when it ran, which is how a list of runs wants to be sorted,
    # and settled here so the answer can carry it before the fit has started
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    # The id is recorded before the task is published, as everywhere else here:
    # the task asks on entry whether it still has a celery_id, and a worker can
    # reach that question before a save that came afterwards.
    signature = celery_tasks.task_sed_fit.subtask(
        args=[target.id, run_id], kwargs={'selection': selection,
                                          'options': options})
    target.celery_id = signature.freeze().id
    target.state = 'fitting the SED'
    target.save()

    signature.apply_async()

    return JsonResponse({'run_id': run_id, 'task_id': target.celery_id,
                         'state_url': f'/targets/{target.id}/state'})


# What a run may have drawn. Served through the target's own file view, which
# already sanitises the path and checks the permission, rather than by
# anything new here.
FIGURE_TYPES = ('*.png', '*.jpg', '*.svg', '*.pdf')


def _run_figures(target, run_id, path):
    """Any figure a run left behind, as URLs the page can use directly."""
    figures = []
    for pattern in FIGURE_TYPES:
        for figure in sorted(glob.glob(os.path.join(path, pattern))):
            name = os.path.basename(figure)
            figures.append({
                'name': name,
                'url': f'/targets/{target.id}/view/sedfit/{run_id}/{name}',
            })

    return figures


@login_required
@require_http_methods(["GET"])
def sed_fits(request, id):
    """The runs on disk: an index of them, or one of them in full.

    Two shapes from one endpoint because the two uses differ. A list wants
    enough to choose by and nothing more - the posteriors of a dozen runs are
    megabytes, and the page only ever draws one at a time. Asking for a run by
    name gets the whole of it, log included.
    """
    import json

    target = get_object_or_404(models.Target, id=id)

    if not target.can_view(request.user):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    wanted = request.GET.get('run')
    root = os.path.join(target.path(), 'sedfit')

    def read(path):
        result, log = None, None

        if os.path.exists(os.path.join(path, 'fit.json')):
            with open(os.path.join(path, 'fit.json')) as f:
                try:
                    result = json.load(f)
                except ValueError:
                    result = None

        if os.path.exists(os.path.join(path, 'fit.log')):
            with open(os.path.join(path, 'fit.log')) as f:
                log = f.read()

        return result, log

    if wanted:
        # Named by the caller, so it is checked rather than trusted: a run is
        # a timestamp and nothing else, and anything else does not name a run.
        if not re.fullmatch(r'[0-9]{8}-[0-9]{6}', wanted):
            return JsonResponse({'error': 'Not a run name'}, status=400)

        path = os.path.join(root, wanted)
        if not os.path.isdir(path):
            raise Http404

        result, log = read(path)

        return JsonResponse({'run_id': wanted, 'result': result, 'log': log,
                             'figures': _run_figures(target, wanted, path)})

    runs = []
    for path in sorted(glob.glob(os.path.join(root, '*')), reverse=True):
        if not os.path.isdir(path):
            continue

        run_id = os.path.basename(path)
        result, log = read(path)

        entry = {'run_id': run_id, 'finished': result is not None,
                 'figures': len(_run_figures(target, run_id, path)),
                 'has_log': log is not None}

        # Enough to tell one run from another in a list, and to see at a
        # glance which is worth opening
        if result:
            grids = result.get('grids') or {}
            entry['source'] = result.get('source')
            entry['grids'] = list(grids)
            entry['points'] = sum(1 for p in result.get('points') or []
                                  if p.get('used'))
            first = next(iter(grids.values()), None)
            if first:
                entry['teff'] = first['teff']['median']
                entry['jitter'] = first['jitter']['median']
                entry['shrink'] = first.get('shrink')
        elif log:
            # A run that failed says so in its last line, which is the one
            # worth putting in the list
            lines = [_ for _ in log.strip().split('\n') if _.strip()]
            entry['error'] = lines[-1][:200] if lines else None

        runs.append(entry)

    return JsonResponse({'runs': runs, 'count': len(runs)})


@login_required
@require_http_methods(["GET"])
def sed_points(request, id):
    """Every point of a SED file, and what a fit would do with each.

    The same reading the fit itself does, so what the panel lists is what would
    be fitted - including which points it would set aside, and why.
    """
    from .processing import sedfit

    target = get_object_or_404(models.Target, id=id)

    if not target.can_view(request.user):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    source = request.GET.get('source') or 'sed.vot'
    if source not in ('sed.vot', 'sed_all.vot'):
        return JsonResponse({'error': 'Unknown SED file'}, status=400)

    path = os.path.join(target.path(), source)
    if not os.path.exists(path):
        return JsonResponse({'error': f'{source} is not there yet',
                             'points': [], 'bands': []}, status=200)

    extra = sedfit.read_extra_points(target.path())
    points = sedfit.read_sed_points(path, extra=extra)

    return JsonResponse({
        'source': source,
        'points': points,
        # What may be added by hand, for the picker
        'bands': sedfit.known_bands(),
        'added': [str(_) for _ in extra['comment']] if extra is not None else [],
    })


@login_required
@require_http_methods(["POST"])
def edit_sed_points(request, id):
    """Add a measurement neither SED file has, or take one back out.

    Kept in a file of its own beside them, so that re-running the SED step -
    which rewrites what it fetched - does not silently drop what somebody
    entered deliberately.
    """
    import json

    from .processing import sedfit
    from .processing.utils import SourceError

    target = get_object_or_404(models.Target, id=id)

    if not target.can_edit(request.user):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    try:
        data = json.loads(request.body or '{}')
    except ValueError:
        return JsonResponse({'error': 'Malformed request'}, status=400)

    action = data.get('action')
    band = str(data.get('band') or '')

    try:
        if action == 'remove':
            sedfit.remove_extra_point(target.path(), band)
        elif action == 'add':
            unit = data.get('unit') or 'mag'
            if unit not in ('mag', 'flux'):
                return JsonResponse({'error': 'Neither a magnitude nor a flux'},
                                    status=400)
            sedfit.add_extra_point(target.path(), band,
                                   float(data['value']),
                                   float(data['error']) if data.get('error') else None,
                                   unit=unit)
        else:
            return JsonResponse({'error': 'Unknown action'}, status=400)
    except SourceError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except (TypeError, ValueError, KeyError) as e:
        return JsonResponse({'error': f'Bad value: {e}'}, status=400)

    extra = sedfit.read_extra_points(target.path())

    return JsonResponse({
        'added': ([{'band': str(r['comment']).rpartition(' ')[2],
                    'wavelength': float(r['wavelength']),
                    'flux': float(r['flux']),
                    'flux_error': (float(r['flux_error'])
                                   if np.isfinite(r['flux_error']) else None)}
                   for r in extra] if extra is not None else []),
    })


@login_required
@require_http_methods(["POST"])
def delete_sed_fits(request, id):
    """Throw away one run, or all of them.

    A run is a directory of a fit that has already happened, so removing it
    loses nothing that cannot be fitted again - but it is still a delete, and
    it is refused while something is running, since the run being written is
    the one most likely to be asked for by mistake.
    """
    import json
    import shutil

    target = get_object_or_404(models.Target, id=id)

    if not target.can_edit(request.user):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    if target.celery_id is not None:
        return JsonResponse(
            {'error': 'Something is running for this target - wait for it'},
            status=409)

    try:
        data = json.loads(request.body or '{}')
    except ValueError:
        return JsonResponse({'error': 'Malformed request'}, status=400)

    root = os.path.join(target.path(), 'sedfit')

    if data.get('all'):
        shutil.rmtree(root, ignore_errors=True)
        return JsonResponse({'deleted': 'all'})

    run = str(data.get('run') or '')
    # Checked rather than trusted, as everywhere a run is named: a run is a
    # timestamp, and anything else does not name one
    if not re.fullmatch(r'[0-9]{8}-[0-9]{6}', run):
        return JsonResponse({'error': 'Not a run name'}, status=400)

    path = os.path.join(root, run)
    if not os.path.isdir(path):
        raise Http404

    shutil.rmtree(path)

    return JsonResponse({'deleted': run})
