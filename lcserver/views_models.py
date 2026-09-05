"""The model atmosphere grids the SED fit is built on, laid open.

A third standalone utility, tied to no target at all. The fit chooses a grid
and reports six numbers; what the grid itself says - how a colour runs with
temperature, where one set of models disagrees with another, which bands a
grid can answer for and where it has no models at all - is not visible
anywhere, and it is what decides whether a fitted temperature means anything.

Nothing here is computed twice. The magnitudes, colours and corrections come
from processing.gridstats, which is the arithmetic the fit puts an observed
magnitude through, run over the models instead; the extinction slope a plot
reddens with is the one the likelihood is evaluated with. A page that said
something different from the fit would be worse than no page.

Two of the axes move without asking the server again, because both are exactly
linear and the slope is a property of the band rather than of the model: a
magnitude of Av moves a band by its attenuation at Av = 1, and a radius moves
every absolute magnitude by 5 log10(R/Rsun) and no colour at all. The values
are sent with those two slopes attached and the sliders are arithmetic in the
browser.
"""

import numpy as np

from django.core.cache import cache
from django.http import JsonResponse
from django.template.response import TemplateResponse

from .processing import gridstats
from .processing.sedfit import offered_grids
from .processing.utils import SourceError


# The coverage matrix asks every grid for its cube, which is a second or two
# the first time and nothing after. It changes when a grid is added to the
# directory, which is not something a visitor does.
COVERAGE_CACHE = 'models_coverage'
COVERAGE_CACHE_AGE = 24 * 3600

# Points beyond which a scatter of every model is drawn as one trace of dots
# rather than as tracks. BT-Settl has fourteen thousand models and a line per
# (log g, [Fe/H]) is a hundred and fifty traces; Plotly will draw them, slowly.
TRACK_LIMIT = 400


# What the explorer opens on: the plot the grids are actually read for, which
# is where a colour stops following temperature and starts following gravity.
DEFAULT_STATE = {
    'grids': ['ck04', 'tlusty'],
    'x': 'teff',
    'y': 'color:GROUND_JOHNSON_B-GROUND_JOHNSON_V',
    'c': 'logg',
}


# Plots worth having a name, so that the page offers the standard ones rather
# than leaving every reader to assemble them from four dropdowns
PRESETS = [
    {
        'id': 'colour_teff',
        'name': 'Colour against temperature',
        'note': "What every photometric temperature calibration is, and where "
                "it stops working: below about 4000 K the colour turns over "
                "and one B - V answers for two temperatures.",
        'state': {'x': 'teff', 'y': 'color:GROUND_JOHNSON_B-GROUND_JOHNSON_V',
                  'c': 'logg', 'xlog': True},
    },
    {
        'id': 'colour_colour',
        'name': 'Colour against colour',
        'note': "The locus a photosphere is confined to, and the direction "
                "reddening moves a star along it. Raise Av and watch the "
                "models slide up the arrow - that is the degeneracy an SED "
                "fit has to break.",
        'state': {'x': 'color:PS1_g-PS1_r', 'y': 'color:PS1_r-PS1_i',
                  'c': 'teff', 'xlog': False},
    },
    {
        'id': 'cmd',
        'name': 'Colour-magnitude',
        'note': "Absolute magnitude against colour at one solar radius. A real "
                "star of radius R sits 5 log10(R/Rsun) higher; the radius "
                "slider is that number and nothing else, so a white dwarf grid "
                "and a supergiant one can be put on the same plot honestly.",
        'state': {'x': 'color:GaiaDR2v2_BP-GaiaDR2v2_RP', 'y': 'mag:GaiaDR2v2_G',
                  'c': 'teff', 'xlog': False, 'yreverse': True},
    },
    {
        'id': 'bc',
        'name': 'Bolometric correction',
        'note': "How much of a star's light the band misses, which is the whole "
                "of what a single-band luminosity depends on. Nearly flat "
                "through the F and G stars and catastrophic at both ends.",
        'state': {'x': 'teff', 'y': 'bc:GROUND_JOHNSON_V', 'c': 'logg',
                  'xlog': True},
    },
    {
        'id': 'nodes',
        'name': 'Where the models are',
        'note': "The grid's own nodes in temperature and gravity. The gaps are "
                "where nothing was computed, and an interpolation that lands in "
                "one is answering about a star that was never modelled.",
        'state': {'x': 'teff', 'y': 'logg', 'c': 'feh', 'xlog': True,
                  'yreverse': True},
    },
]


def grid_list():
    """The grids on offer, with what the page needs to draw them."""
    out = []

    for entry in offered_grids():
        out.append(dict(entry, axis_note=('log Rt, the transformed radius, '
                                          'rather than a surface gravity')
                        if entry['axis_logg'] != 'logg' else None))

    return out


def models(request):
    """The grids, their photometry and their spectra."""
    context = {
        'grids': grid_list(),
        # Only what a picker needs: the zero point and the extinction slope
        # are what the answers were computed with, not what a page chooses
        'bands': [{'band': b['band'], 'system': b['system'],
                   'wavelength': round(b['wavelength'] * 1e-4, 4)}
                  for b in gridstats.bands()],
        'presets': PRESETS,
        'state': DEFAULT_STATE,
        'track_limit': TRACK_LIMIT,
    }

    return TemplateResponse(request, 'models.html', context=context)


def _round(values, digits):
    """An array as a list of plain floats, short enough to send.

    A magnitude to four decimals is a ten-thousandth of one, which is finer
    than any grid is right to, and it is half the bytes of a repr.
    """
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), np.round(values, digits), np.nan)

    return [None if not np.isfinite(v) else float(v) for v in values]


def models_data(request):
    """The models of the chosen grids, on the chosen axes.

    The three parameters come with every answer whatever the axes are: the
    page filters on them, colours by them and groups tracks by them, and all
    of that is meant to happen without another request.
    """
    names = [n for n in request.GET.get('grids', '').split(',') if n]
    specs = {key: request.GET.get(key) for key in ('x', 'y', 'c')}

    if not names:
        return JsonResponse({'error': 'no grid asked for'}, status=400)

    out = []

    for name in names:
        try:
            table = gridstats.photometry(name)
        except (SourceError, OSError) as e:
            out.append({'name': name, 'error': str(e)})
            continue

        entry = {
            'name': table['name'],
            'axis_logg': table['axis_logg'],
            'layout': table['layout'],
            'teff': _round(table['teff'], 1),
            'logg': _round(table['logg'], 3),
            'feh': _round(table['feh'], 3),
            'axes': {},
        }

        for key, spec in specs.items():
            if not spec:
                continue

            # A parameter axis is already in the answer, and sending it twice
            # would be half the payload again on a grid of fourteen thousand
            kind, _ = gridstats.parse_axis(spec)
            if kind in ('teff', 'logg', 'feh'):
                entry['axes'][key] = {'ref': kind, 'slope': 0.0,
                                      'radius': False,
                                      'label': gridstats.axis(table, spec)['label']}
                continue

            try:
                axis = gridstats.axis(table, spec)
            except SourceError as e:
                entry['axes'][key] = {'error': str(e)}
                continue

            entry['axes'][key] = {
                'values': _round(axis['values'], 4),
                'slope': axis['slope'],
                'radius': axis['radius'],
                'label': axis['label'],
            }

        out.append(entry)

    return JsonResponse({'grids': out})


def models_spectra(request):
    """One family of model spectra, as the grid holds them.

    Surface flux and no reddening: the shape is what is being looked at, and
    every normalisation the page offers is a scalar it applies itself.
    """
    name = request.GET.get('grid', '')
    along = request.GET.get('along', 'teff')

    if along not in ('teff', 'logg', 'feh'):
        return JsonResponse({'error': f'cannot step along {along}'}, status=400)

    at = {}
    for parameter in ('teff', 'logg', 'feh'):
        value = request.GET.get(parameter)
        if value not in (None, ''):
            try:
                at[parameter] = float(value)
            except ValueError:
                pass

    try:
        count = min(int(request.GET.get('count', 8)), 16)
    except ValueError:
        count = 8

    try:
        index, fixed = gridstats.spectrum_family(name, along, at, count)
        if not index:
            return JsonResponse({'error': 'no models at those parameters',
                                 'fixed': fixed}, status=404)

        wave, family = gridstats.spectra(name, index)
    except (SourceError, OSError) as e:
        return JsonResponse({'error': str(e)}, status=404)

    # Each spectrum on its own peak, with the peak sent beside it. A surface
    # flux runs to 1e13 and five significant figures of it is eleven characters
    # of JSON per point; normalised, it is seven, and nothing is lost - the
    # scale multiplies it straight back.
    out = []

    for entry in family:
        scale = np.nanmax(entry['flux'])
        scale = float(scale) if np.isfinite(scale) and scale > 0 else 1.0

        out.append({'teff': entry['teff'], 'logg': entry['logg'],
                    'feh': entry['feh'], 'scale': scale,
                    'flux': _round(entry['flux'] / scale, 5)})

    return JsonResponse({
        'grid': name,
        'along': along,
        'fixed': fixed,
        'wave_um': _round(wave, 6),
        'spectra': out,
    })


def models_coverage(request):
    """Which bands each grid predicts, and how its models are laid out."""
    refresh = bool(request.GET.get('refresh')) and request.user.is_staff
    payload = None if refresh else cache.get(COVERAGE_CACHE)

    if payload is None:
        payload = gridstats.coverage()

        for grid in payload['grids']:
            try:
                grid['support'] = gridstats.support(grid['name'])
            except Exception:
                grid['support'] = None

        cache.set(COVERAGE_CACHE, payload, COVERAGE_CACHE_AGE)

    return JsonResponse(payload)
