"""Standalone multi-wavelength cutout previewer.

Adapted from the cutouts section of the crossmatch web app. Deliberately
outside the target processing loop: the positions are resolved on the fly, and
the images themselves are fetched by the browser straight from the image
services, so the page needs neither a database entry, nor a Celery worker, nor
a logged-in user.
"""

from django.http import Http404
from django.template.response import TemplateResponse
from django.core.cache import cache
from django.shortcuts import redirect
from django.urls import reverse

from collections import OrderedDict
from urllib.parse import urlencode
import hashlib

import numpy as np
from astropy.table import Table

from stdpipe import resolve


HIPS_SERVICE = 'https://alasky.u-strasbg.fr/hips-image-services/hips2fits'
SKYVIEW_SERVICE = 'https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl'
SDSS_SERVICE = 'https://skyserver.sdss.org/dr18/SkyserverWS/ImgCutout/getjpeg'
PS1_FILENAMES_SERVICE = 'https://ps1images.stsci.edu/cgi-bin/ps1filenames.py'
PS1_CUTOUT_SERVICE = 'https://ps1images.stsci.edu/cgi-bin/fitscut.cgi'

# Size of the images requested from the services, in pixels - the first is what
# the page displays, the second what the click-to-enlarge popup loads
CUTOUT_SIZE = 256
CUTOUT_SIZE_LARGE = 768

# Positions are resolved one network request at a time, so the list a single
# request may ask for is bounded
MAX_POSITIONS = 50

# Field of view limits, degrees
FOV_MIN = 1 / 3600
FOV_MAX = 10

UNITS = OrderedDict([
    ('deg', 1),
    ('arcmin', 60),
    ('arcsec', 3600),
])

# Wavelength ranges the surveys are shown in, in the order the page lists them
CUTOUT_GROUPS = OrderedDict([
    ('uv', 'Ultraviolet'),
    ('optical', 'Optical'),
    ('halpha', 'Halpha'),
    ('ir', 'Infrared'),
])

# The surveys offered on the page. Every entry with a 'hips' key is served by
# the CDS hips2fits service and needs nothing else; the rest are special-cased
# in get_cutout(). 'mono' marks a single-channel HiPS, which is the only case
# where the colormap and the cuts mean anything - colour HiPS carry their own
# palette and are left alone.
CUTOUT_SURVEYS = OrderedDict([
    ('galex', {'name': 'GALEX', 'group': 'uv', 'hips': 'CDS/P/GALEXGR6/AIS/color', 'default': True}),
    ('galex_nuv', {'name': 'GALEX NUV', 'group': 'uv', 'skyview': 'galex near uv'}),
    ('galex_fuv', {'name': 'GALEX FUV', 'group': 'uv', 'skyview': 'galex far uv'}),

    ('dss2', {'name': 'DSS2', 'group': 'optical', 'hips': 'CDS/P/DSS2/color', 'default': True}),

    ('sdss', {'name': 'SDSS DR18', 'group': 'optical', 'default': True}),

    ('panstarrs', {'name': 'PanSTARRS', 'group': 'optical', 'hips': 'CDS/P/PanSTARRS/DR1/color', 'default': True}),
    ('ps1', {'name': 'PanSTARRS (STScI)', 'group': 'optical'}),

    ('ztf', {'name': 'ZTF DR7', 'group': 'optical', 'hips': 'CDS/P/ZTF/DR7/color', 'default': True}),

    ('skymapper', {'name': 'SkyMapper', 'group': 'optical', 'hips': 'CDS/P/skymapper-color', 'default': True}),

    ('mmt9', {'name': 'Mini-MegaTORTORA', 'group': 'optical', 'hips': 'http://survey.favor2.info/favor2/hips/', 'mono': True}),

    ('iphas_halpha', {'name': 'IPHAS Halpha', 'group': 'halpha', 'hips': 'CDS/P/IPHAS/DR2/halpha', 'mono': True}),
    ('vtss_halpha', {'name': 'VTSS Halpha', 'group': 'halpha', 'hips': 'CDS/P/VTSS/Ha', 'mono': True}),
    ('shassa_halpha', {'name': 'SHASSA Halpha', 'group': 'halpha', 'hips': 'CDS/P/SHASSA/H', 'mono': True}),
    ('shs_halpha', {'name': 'SHS Halpha', 'group': 'halpha', 'hips': 'CDS/P/SHS', 'mono': True}),

    ('wise', {'name': 'allWISE', 'group': 'ir', 'hips': 'CDS/P/allWISE/color', 'default': True}),
    ('wise_w1', {'name': 'WISE 3.4um', 'group': 'ir', 'hips': 'CDS/P/allWISE/W1', 'mono': True}),
    ('wise_w2', {'name': 'WISE 4.6um', 'group': 'ir', 'hips': 'CDS/P/allWISE/W2', 'mono': True}),
    ('wise_w3', {'name': 'WISE 12um', 'group': 'ir', 'hips': 'CDS/P/allWISE/W3', 'mono': True}),
    ('wise_w4', {'name': 'WISE 22um', 'group': 'ir', 'hips': 'CDS/P/allWISE/W4', 'mono': True}),

    ('glimpse360', {'name': 'GLIMPSE360', 'group': 'ir', 'hips': 'IPAC/P/GLIMPSE360'}),
])


def get_survey_groups():
    """The surveys as the page shows them - split into wavelength sections.

    The sections come out in the order CUTOUT_GROUPS lists them, and a survey
    belonging to no section of its own still gets one, so that adding a survey
    can never drop it from the form.
    """
    groups = OrderedDict()

    for gid, name in CUTOUT_GROUPS.items():
        groups[gid] = {'id': gid, 'name': name, 'surveys': []}

    for sid, survey in CUTOUT_SURVEYS.items():
        gid = survey.get('group', 'other')

        if gid not in groups:
            groups[gid] = {'id': gid, 'name': CUTOUT_GROUPS.get(gid, gid.title()), 'surveys': []}

        groups[gid]['surveys'].append(dict(survey, id=sid))

    return [_ for _ in groups.values() if _['surveys']]


def hips_url(survey, ra, dec, fov, size, format='jpg'):
    """URL of a hips2fits cutout, as an image or as the FITS behind it."""
    params = {
        'hips': survey['hips'],
        'ra': ra,
        'dec': dec,
        'width': size,
        'height': size,
        'fov': fov,
        'projection': 'TAN',
        'coordsys': 'icrs',
        'rotation_angle': 0.0,
        'format': format,
    }

    if format != 'fits' and survey.get('mono'):
        params.update({
            'stretch': survey.get('stretch', 'linear'),
            'cmap': survey.get('cmap', 'viridis'),
            'min_cut': survey.get('min_cut', '0.5%'),
            'max_cut': survey.get('max_cut', '99.5%'),
        })

    return HIPS_SERVICE + '?' + urlencode(params)


def skyview_url(survey, ra, dec, fov, size, format='GIF'):
    """URL of a SkyView cutout, as an image or as the FITS behind it."""
    params = {
        'Position': '%s,%s' % (ra, dec),
        'Size': fov,
        'Pixels': size,
        'Survey': survey['skyview'],
        'Return': format,
    }

    if format != 'FITS':
        params['LUT'] = 'colortables/blue-white.bin'
        params['scaling'] = 'histeq'

    return SKYVIEW_SERVICE + '?' + urlencode(params)


def sdss_url(ra, dec, fov, size):
    """URL of an SDSS SkyServer colour cutout. The service is scale driven, so
    the field of view is what fixes the pixel scale."""
    return SDSS_SERVICE + '?' + urlencode({
        'ra': ra,
        'dec': dec,
        'width': size,
        'height': size,
        'scale': fov * 3600 / size,
    })


def get_cutout(sid, ra, dec, fov):
    """Everything the page needs in order to show one survey at one position:
    the thumbnail, the larger version the popup loads, and the FITS if the
    service has one."""
    survey = CUTOUT_SURVEYS.get(sid)

    if survey is None:
        return None

    cutout = {'id': sid, 'name': survey['name']}

    if survey.get('hips'):
        cutout['url'] = hips_url(survey, ra, dec, fov, CUTOUT_SIZE)
        cutout['url_large'] = hips_url(survey, ra, dec, fov, CUTOUT_SIZE_LARGE)
        cutout['fits'] = hips_url(survey, ra, dec, fov, CUTOUT_SIZE_LARGE, format='fits')

    elif survey.get('skyview'):
        cutout['url'] = skyview_url(survey, ra, dec, fov, CUTOUT_SIZE)
        cutout['url_large'] = skyview_url(survey, ra, dec, fov, CUTOUT_SIZE_LARGE)
        cutout['fits'] = skyview_url(survey, ra, dec, fov, CUTOUT_SIZE_LARGE, format='FITS')

    elif sid == 'sdss':
        cutout['url'] = sdss_url(ra, dec, fov, CUTOUT_SIZE)
        cutout['url_large'] = sdss_url(ra, dec, fov, CUTOUT_SIZE_LARGE)

    elif sid == 'ps1':
        # The image URL cannot be built without asking the archive which plates
        # cover the position, so it is deferred to a view of our own
        params = {'ra': ra, 'dec': dec, 'fov': fov}
        cutout['url'] = reverse('cutouts_ps1') + '?' + urlencode(dict(params, size=CUTOUT_SIZE))
        cutout['url_large'] = reverse('cutouts_ps1') + '?' + urlencode(dict(params, size=CUTOUT_SIZE_LARGE))

    else:
        return None

    return cutout


def ps1_url(ra, dec, fov, size):
    """Colour cutout from the Pan-STARRS image service at STScI.

    Unlike everything else on the page this needs a query of its own first, to
    learn the names of the images covering the position.
    """
    # The service works in its native 0.25 arcsec pixels, and refuses anything
    # larger than 6000 of them
    size_px = min(int(fov * 3600 / 0.25), 6000)

    table = Table.read(PS1_FILENAMES_SERVICE + '?' + urlencode({
        'ra': ra, 'dec': dec, 'size': size_px, 'format': 'fits', 'filters': 'grizy',
    }), format='ascii')

    if not len(table):
        return None

    # Sort the filters from red to blue, then keep three of them for the channels
    table = table[np.argsort(['yzirg'.find(_) for _ in table['filter']])]

    if len(table) > 3:
        table = table[[0, len(table) // 2, len(table) - 1]]
    elif len(table) < 3:
        return None

    params = {'ra': ra, 'dec': dec, 'size': size_px, 'format': 'jpg', 'output_size': size}
    params.update({_: table['filename'][i] for i, _ in enumerate(['red', 'green', 'blue'])})

    return PS1_CUTOUT_SERVICE + '?' + urlencode(params)


def cutouts_ps1(request):
    """Redirect to the Pan-STARRS cutout for the requested position.

    The lookup behind it is cached, as the browser asks for the image once per
    position on the page and again for every popup.
    """
    ra, dec, fov, size = [request.GET.get(_) for _ in ['ra', 'dec', 'fov', 'size']]

    try:
        ra, dec, fov = float(ra), float(dec), float(fov)
        size = int(size or CUTOUT_SIZE)
    except (TypeError, ValueError):
        raise Http404("Malformed position")

    cid = 'cutouts_ps1_%.6f_%.6f_%g_%d' % (ra, dec, fov, size)
    url = cache.get(cid)

    if url is None:
        try:
            url = ps1_url(ra, dec, fov, size)
        except Exception:
            url = None

        # Failures are cached too, and for less, so that a position outside the
        # survey does not query the archive again on every reload
        cache.set(cid, url or '', 3600 if url else 300)

    if not url:
        raise Http404("No Pan-STARRS images at this position")

    return redirect(url)


def resolve_position(string):
    """Resolve a name or a coordinate string, caching what came back.

    Names go to SIMBAD, TNS and Sesame in turn, so the same list of positions
    resubmitted - which is what changing the size or the survey selection does
    - should not repeat any of it.
    """
    cid = 'cutouts_resolve_' + hashlib.md5(string.encode('utf-8')).hexdigest()
    result = cache.get(cid)

    if result is None:
        try:
            target = resolve.resolve(string)
        except Exception:
            target = None

        if target is not None:
            result = {
                'ra': target.ra.deg,
                'dec': target.dec.deg,
                'l': target.galactic.l.deg,
                'b': target.galactic.b.deg,
            }
        else:
            result = {}

        cache.set(cid, result, 24 * 3600 if result else 600)

    return result or None


def cutouts(request):
    """Bulk preview of multi-wavelength images of arbitrary sky positions."""
    context = {
        'groups': get_survey_groups(),
        'fov_value': 3,
        'fov_units': 'arcmin',
        'selected': [_ for _ in CUTOUT_SURVEYS if CUTOUT_SURVEYS[_].get('default')],
        'max_positions': MAX_POSITIONS,
    }

    # The form itself posts, so that a list of positions is not carried in the
    # URL. A position may still be passed in a link, though - which is how
    # anything elsewhere would point at this page
    if request.method == 'POST':
        data = request.POST
    elif request.GET.get('coords') or request.GET.get('multicoords'):
        data = request.GET
    else:
        return TemplateResponse(request, 'cutouts.html', context=context)

    context['coords'] = data.get('coords', '')
    context['multicoords'] = data.get('multicoords', '')

    # An unticked survey is simply absent from what the form posts, so an empty
    # selection is only taken at face value when it came from the form itself
    if request.method == 'POST' or 'surveys' in data:
        context['selected'] = data.getlist('surveys')

    context['fov_units'] = data.get('fov_units', 'arcmin')
    if context['fov_units'] not in UNITS:
        context['fov_units'] = 'arcmin'

    try:
        context['fov_value'] = float(data.get('fov_value'))
    except (TypeError, ValueError):
        pass

    fov = context['fov_value'] / UNITS[context['fov_units']]
    fov = min(max(fov, FOV_MIN), FOV_MAX)
    context['fov'] = fov

    # Single position and the multi-line list are just concatenated, so that
    # the one is a convenience and not a separate mode
    positions = [_.strip() for _ in
                 [context['coords']] + context['multicoords'].splitlines()
                 if _.strip()]

    if len(positions) > MAX_POSITIONS:
        context['truncated'] = len(positions)
        positions = positions[:MAX_POSITIONS]

    targets = []

    for position in positions:
        target = {'query': position}
        result = resolve_position(position)

        if result is None:
            target['message'] = "Cannot resolve the position: " + position
        else:
            target.update(result)
            target['cutouts'] = [_ for _ in
                                 [get_cutout(sid, result['ra'], result['dec'], fov)
                                  for sid in CUTOUT_SURVEYS if sid in context['selected']]
                                 if _ is not None]

        targets.append(target)

    context['targets'] = targets

    return TemplateResponse(request, 'cutouts.html', context=context)
