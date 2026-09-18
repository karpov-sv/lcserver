"""ATLAS forced photometry acquisition module.

The Asteroid Terrestrial-impact Last Alert System is four 0.5m telescopes -
two on Hawaii, one in Chile, one in South Africa - covering the whole sky
every night or two since 2015, to about 19.5 in o. It is the main
high-cadence optical survey of the southern sky, where ZTF does not reach.

There is no catalogue of light curves to query. The forced photometry server
measures a position on every image ATLAS has taken of it, on request, and
the requests go into one queue shared by every user of the service - over a
thousand of them waiting at times. So this step does not simply ask and
wait. It submits the request, notes the task it was given in the target's
cache, and waits a few minutes; if the queue has not reached it by then the
step ends saying so, and running it again picks up the same task rather than
queueing another. Once a result has been fetched the task is deleted from
the server, as the service asks.

For the same reason it is a manual source, left out of "Run everything": a
request that holds a worker for minutes and then usually ends still queued
is not something to spend on every target that happens to be acquired.

Photometry is forced on the reduced images rather than the difference ones,
so that the light curve is the star's total brightness, as every other
source here reports it, rather than its change from a template - which also
jumps at each of the template replacements. The service warns that its
single-PSF fit on reduced images can misjudge the sky in a crowded field.
"""

import os
import json
import time

import requests
import numpy as np

from astropy.table import Table
from astropy.time import Time

from django.conf import settings

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    log_bands, log_conversion, plot_with_errors,
                    quality_field, quality_level, assumed_color,
                    atlas_c_to_g, atlas_o_to_g,
                    ATLAS_C_TO_G_FORMULA, ATLAS_O_TO_G_FORMULA,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


ATLAS_URL = 'https://fallingstar-data.com/forcedphot'

ATLAS_TIMEOUT = 60

# How long one run waits for the queue to reach its request and the server to
# finish it, in seconds. Well inside the Celery time limit, and short enough
# not to hold a worker for long; a request still queued after this is picked
# up by the next run rather than lost.
ATLAS_WAIT = 300

# How often to ask, as the service's own guide does
ATLAS_POLL = 10

# The longest the server may ask a submission to wait before it is retried
# here; longer than this and the step ends, to be run again later
ATLAS_MAX_RETRY_AFTER = 60

# The earliest MJD asked for - before the survey began, so all of it
ATLAS_MJD_MIN = 57000.0

# The service's zero point: an AB magnitude of a flux in microjansky
ATLAS_AB_UJY = 23.9

# The signal to noise below which a flux is not given a magnitude. The flux is
# kept; a magnitude of a flux consistent with nothing is not a measurement.
ATLAS_MIN_SNR = 3.0

# The bands, as the result names them. H is a narrow H-alpha filter used now
# and then, and is not drawn.
ATLAS_BANDS = ('c', 'o')
ATLAS_COLORS = {'c': '#17becf', 'o': '#ff7f0e'}

# The reduced chi-square above which the PSF fit is not measuring the star.
# Nothing in the result says a star saturated, and the FAQ's cuts do not catch
# it, but the fit does: for an isolated G = 15.4 star it is 1.5 in c and 4 in o
# at the median and 13 at worst, while for a V = 10 star - past the bright
# limit of about 11 that Heinze et al. (2018) give - it is a quarter of a
# million, or no number at all. The magnitudes of such a star are wrong by
# a magnitude and more, and scattered by as much.
ATLAS_MAX_CHI = 100.0

# What a measurement no level can keep: the fit failed or saturated
ATLAS_SATURATION_CUT = ('chi_N', '<', ATLAS_MAX_CHI,
                        'the PSF fit is sound - a saturated star fails it')

# The cuts the service's own FAQ recommends, after Rest et al., for points
# that are not what they seem: saturated or masked stars, failed fits, cloud,
# moonlight, the edge of the chip
ATLAS_STANDARD_CUTS = [
    ATLAS_SATURATION_CUT,
    ('duJy', '<', 10000, 'flux error under 10000 uJy'),
    ('err', '==', 0, 'the PSF fit reported no error'),
    ('x', '>', 100, 'away from the chip edge'),
    ('x', '<', 10460, 'away from the chip edge'),
    ('y', '>', 100, 'away from the chip edge'),
    ('y', '<', 10460, 'away from the chip edge'),
    ('maj', '>', 1.6, 'a PSF of plausible size'),
    ('maj', '<', 5, 'a PSF of plausible size'),
    ('min', '>', 1.6, 'a PSF of plausible size'),
    ('min', '<', 5, 'a PSF of plausible size'),
    ('apfit', '>', -1, 'a plausible aperture correction'),
    ('apfit', '<', -0.1, 'a plausible aperture correction'),
    ('mag5sig', '>', 17, 'the frame reached 17th magnitude'),
    ('Sky', '>', 17, 'the sky no brighter than 17 mag/arcsec2'),
]

# The one cut the same FAQ calls certain - a frame without a photometric
# solution - and saturation, which is all the relaxed level removes
ATLAS_RELAXED_CUTS = [
    ATLAS_SATURATION_CUT,
    ('mag5sig', '>=', 10, 'the frame has a photometric solution'),
]

ATLAS_CUTS = {
    QUALITY_STANDARD: ATLAS_STANDARD_CUTS,
    QUALITY_RELAXED: ATLAS_RELAXED_CUTS,
    QUALITY_PUBLISHED: [],
}

ATLAS_OPERATORS = {
    '<': np.less, '>': np.greater, '==': np.equal, '>=': np.greater_equal,
}


def _headers(token):
    return {'Authorization': f'Token {token}', 'Accept': 'application/json'}


def _refused(res, what):
    """A request the server answered with something other than what was asked.

    Its own words where it gave any - DRF answers with {"detail": ...}.
    """
    if res.status_code == 401:
        return SourceError("the ATLAS server refused the key in ATLAS_TOKEN - "
                           "it answered 401")

    try:
        detail = res.json().get('detail') or res.text
    except ValueError:
        detail = res.text

    return SourceError(f"could not {what} - HTTP {res.status_code}: "
                       + ' '.join(str(detail).split())[:300])


def _submit(ra, dec, token, log):
    """Queue a request, and return the URL of the task it became."""
    data = {'ra': ra, 'dec': dec, 'mjd_min': ATLAS_MJD_MIN,
            'use_reduced': True, 'send_email': False,
            'comment': 'lcserver'}

    while True:
        try:
            res = requests.post(f"{ATLAS_URL}/queue/", headers=_headers(token),
                                data=data, timeout=ATLAS_TIMEOUT)
        except requests.RequestException as e:
            raise SourceError("could not reach the ATLAS server - "
                              f"{type(e).__name__}: {e}")

        if res.status_code == 201:
            return res.json()['url']

        if res.status_code == 429:
            try:
                wait = int(res.headers.get('Retry-After') or 0)
            except ValueError:
                wait = 0

            if 0 < wait <= ATLAS_MAX_RETRY_AFTER:
                log(f"The ATLAS server asks to wait {wait} s before queueing")
                time.sleep(wait)
                continue

            raise SourceError("the ATLAS server is throttling this key"
                              + (f" - it asks for another {wait // 60} minutes"
                                 if wait else ""))

        raise _refused(res, "queue the ATLAS request")


def _task(url, token):
    """The task as the server describes it now, or None where it is gone."""
    try:
        res = requests.get(url, headers=_headers(token), timeout=ATLAS_TIMEOUT)
    except requests.RequestException as e:
        raise SourceError("could not reach the ATLAS server - "
                          f"{type(e).__name__}: {e}")

    if res.status_code == 404:
        return None

    if res.status_code != 200:
        raise _refused(res, "ask the ATLAS server about the request")

    return res.json()


def _delete(url, token, log):
    """Remove a task from the server, now that its result is kept here."""
    try:
        res = requests.delete(url, headers=_headers(token), timeout=ATLAS_TIMEOUT)
    except requests.RequestException as e:
        log(f"Could not delete the ATLAS task {url} - {type(e).__name__}: {e}")
        return

    if res.status_code not in (204, 404):
        log(f"Could not delete the ATLAS task {url} - HTTP {res.status_code}")


def _result(url, token):
    """The result file of a finished task, as a table.

    Space-separated text whose header line starts with ###. None where it
    holds no measurement at all.
    """
    try:
        res = requests.get(url, headers=_headers(token), timeout=ATLAS_TIMEOUT)
        res.raise_for_status()
    except requests.RequestException as e:
        raise SourceError("could not fetch the ATLAS result - "
                          f"{type(e).__name__}: {e}")

    text = res.text.replace('###', '', 1)
    lines = [line for line in text.splitlines() if line.strip()]

    if len(lines) < 2:
        return None

    try:
        table = Table.read(lines, format='ascii.basic', delimiter=' ',
                           guess=False, fast_reader=False)
    except Exception as e:
        raise SourceError("could not read the ATLAS result - "
                          f"{type(e).__name__}: {e}")

    # One name the VOTable writer would have to mangle
    if 'chi/N' in table.colnames:
        table.rename_column('chi/N', 'chi_N')

    return table


def _state_path(basepath, cache_name):
    """Where a request still in the server's queue is remembered."""
    return os.path.join(basepath, 'cache', cache_name.replace('.vot', '.task.json'))


def _fetch(ra, dec, token, state_path, log):
    """The measurements at a position: from the task already queued for it if
    there is one, or from a new one.

    Returns the result table, or None for a position ATLAS has no measurement
    of. Raises SourceError while the request is still waiting, having kept
    its task so that the next run resumes it.
    """
    state = None

    if os.path.exists(state_path):
        try:
            with open(state_path) as f:
                state = json.load(f)
        except (OSError, ValueError):
            state = None

    task = _task(state['url'], token) if state else None

    if state and task is None:
        log("The ATLAS task queued by an earlier run is no longer on the "
            "server, so the request is queued again")

    if task is None:
        url = _submit(ra, dec, token, log)
        state = {'url': url, 'submitted': Time.now().isot}

        os.makedirs(os.path.dirname(state_path), exist_ok=True)

        with open(state_path, 'w') as f:
            json.dump(state, f)

        log(f"Queued as {url}")
    else:
        log(f"Resuming {state['url']}, queued at {state['submitted']}")

    url = state['url']
    started = time.time()

    while True:
        task = _task(url, token)

        if task is None:
            os.remove(state_path)
            raise SourceError("the ATLAS task disappeared from the server "
                              "while waiting for it - run the step again")

        if task.get('finishtimestamp'):
            break

        if time.time() - started > ATLAS_WAIT:
            position = task.get('queuepos')
            where = ("running" if task.get('starttimestamp') else
                     f"number {position + 1} in the queue" if position is not None
                     else "still queued")

            raise SourceError(
                f"the ATLAS request is {where} after {ATLAS_WAIT // 60} "
                "minutes of waiting - it is kept on the server, and running "
                "the step again later collects it")

        time.sleep(ATLAS_POLL)

    log(f"The ATLAS server finished the request in "
        f"{task.get('runtime') or 0:.0f} s, after "
        f"{task.get('waittime') or 0:.0f} s in the queue")

    if task.get('error_msg'):
        # Nothing to collect, and nothing to resume: the next run starts over
        _delete(url, token, log)
        os.remove(state_path)
        raise SourceError(f"the ATLAS server could not measure the position: "
                          f"{task['error_msg']}")

    table = _result(task['result_url'], token)

    # Deleted only now the result is in hand: until then the server's copy is
    # the only one
    _delete(url, token, log)
    os.remove(state_path)

    return table


def _cut(table, cuts, log):
    """The rows passing every cut, with how many each one removed."""
    good = np.ones(len(table), dtype=bool)

    for column, op, value, meaning in cuts:
        if column not in table.colnames:
            continue

        values = np.asarray(table[column], dtype=float)
        passed = ATLAS_OPERATORS[op](values, value)
        removed = int(np.sum(good & ~passed))

        if removed:
            log(f"    {column} {op} {value:<8g} {removed:6d} removed - {meaning}")

        good &= passed

    return good


@survey_source(
    name='ATLAS forced photometry',
    short_name='ATLAS',
    state_acquiring='acquiring ATLAS lightcurve',
    state_acquired='ATLAS lightcurve acquired',
    log_file='atlas.log',
    output_files=['atlas.log', 'atlas_lc.png', 'atlas.vot', 'atlas.txt'],
    button_text='Get ATLAS lightcurve',
    form_fields={
        'atlas_quality': quality_field({
            QUALITY_STANDARD: "The service's recommended cuts - cloud, "
                              "moonlight, chip edges, failed fits",
            QUALITY_RELAXED: 'Only frames with no photometric solution',
            QUALITY_PUBLISHED: 'None - every measurement as published',
        }),
    },
    help_text='ATLAS forced photometry on reduced images, c and o, whole sky, '
              'since 2015; queued, needs an API key, run by hand only',
    order=13,
    about=(
        "The Asteroid Terrestrial-impact Last Alert System - four 0.5m "
        "telescopes in Hawaii, Chile and South Africa covering the whole sky "
        "every night or two since 2015, in cyan and orange. Its forced "
        "photometry server measures any position on every image, on request, "
        "through a queue shared by all its users."),
    about_links=[
        ('ATLAS forced photometry', 'https://fallingstar-data.com/forcedphot/'),
        ('Tonry et al. 2018', 'https://arxiv.org/abs/1802.00879'),
        ('Heinze et al. 2018', 'https://arxiv.org/abs/1804.02132'),
        ('Shingles et al. 2021', 'https://ui.adsabs.harvard.edu/abs/2021TNSAN...7....1S/abstract'),
    ],
    acknowledgement=(
        "Cite Tonry et al. (2018), and for variable object science Heinze et "
        "al. (2018); the service itself is Shingles et al. (2021). This work "
        "has made use of data from the Asteroid Terrestrial-impact Last Alert "
        "System (ATLAS) project. The Asteroid Terrestrial-impact Last Alert "
        "System (ATLAS) project is primarily funded to search for near earth "
        "asteroids through NASA grants NN12AR55G, 80NSSC18K0284, and "
        "80NSSC18K1575; byproducts of the NEO search include images and "
        "catalogs from the survey area. This work was partially funded by "
        "Kepler/K2 grant J1944/80NSSC19K0112 and HST GO-15889, and STFC "
        "grants ST/T000198/1 and ST/S006109/1. The ATLAS science products "
        "have been made possible through the contributions of the University "
        "of Hawaii Institute for Astronomy, the Queen's University Belfast, "
        "the Space Telescope Science Institute, the South African "
        "Astronomical Observatory, and The Millennium Institute of "
        "Astrophysics (MAS), Chile."),
    # Lightcurve metadata
    votable_file='atlas.vot',
    lc_bands=[
        surveys.band('c', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='c',
                     color=ATLAS_COLORS['c'], note='as measured by ATLAS, AB'),
        surveys.band('o', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='o',
                     color=ATLAS_COLORS['o'], note='as measured by ATLAS, AB'),
        surveys.band('g (from c)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     filter_column='filter', filter_value='c',
                     color='#9edae5',
                     note='c put on the common g scale using an assumed g - r',
                     combined=True),
        surveys.band('g (from o)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     filter_column='filter', filter_value='o',
                     color='#ffbb78',
                     note='o put on the common g scale using an assumed g - r',
                     combined=True),
    ],
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color=ATLAS_COLORS['o'],
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata
    template_layout='simple',
    requires_coordinates=True,
    manual=True,
)
def target_atlas(config, basepath=None, verbose=True, show=False):
    """Acquire ATLAS forced photometry lightcurve."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('atlas'), basepath=basepath)

    token = getattr(settings, 'ATLAS_TOKEN', '')

    if not token:
        # An installation without a key, rather than a position without data
        log("No ATLAS_TOKEN is set, and the forced photometry server cannot "
            "be asked without one. Register at "
            "fallingstar-data.com/forcedphot, get a token from its "
            "api-token-auth endpoint, and put it in the environment as "
            "ATLAS_TOKEN.")
        return

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')

    cache_name = f"atlas_{ra:.5f}_{dec:.5f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'ATLAS forced photometry',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            log(f"at RA={ra:.5f} Dec={dec:.5f}, on reduced images")

            table = _fetch(ra, dec, token,
                           _state_path(basepath, cache_name), log)

            if table is None or not len(table):
                cache.save_empty()
                log("Warning: ATLAS has no measurement at this position")
                return

            cache.save(table)

        atlas = cache.data

    # Nothing here, and cached as nothing - the helper has said so already
    if atlas is None:
        return

    log(f"\n{len(atlas)} measurements")

    bands = np.asarray(atlas['F'], dtype=str)
    other = ~np.isin(bands, ATLAS_BANDS)

    if np.any(other):
        log(f"{int(np.sum(other))} measurements in other filters "
            f"({', '.join(sorted(set(bands[other])))}) are not used")

    atlas = atlas[~other]

    quality = quality_level(config, 'atlas')
    cuts = ATLAS_CUTS[quality]

    if cuts:
        log(f"  cuts ({quality}):")

    good = _cut(atlas, cuts, log)

    # A star most of whose measurements fail the fit is not a star with some
    # bad frames but one too bright to measure at all, and that is worth
    # saying in so many words
    if ATLAS_SATURATION_CUT in cuts and len(atlas):
        chi = np.asarray(atlas['chi_N'], dtype=float)
        failed = np.mean(~(chi < ATLAS_MAX_CHI))

        if failed > 0.5:
            log(f"Warning: the PSF fit fails on {100*failed:.0f}% of the frames - "
                "the star is most likely too bright for ATLAS, which saturates "
                "at about 11th magnitude")
    atlas = atlas[good]

    log(f"{len(atlas)} measurements left ({quality} filtering)")

    flux = np.asarray(atlas['uJy'], dtype=float)
    flux_err = np.asarray(atlas['duJy'], dtype=float)

    # The magnitudes the server writes carry a sign for a negative flux, and
    # a value for a flux that is noise; they are made again here, from the
    # flux, only where it is a detection
    detected = np.isfinite(flux) & (flux > ATLAS_MIN_SNR*flux_err) & (flux_err > 0)

    if np.any(~detected):
        log(f"{int(np.sum(~detected))} measurements under {ATLAS_MIN_SNR:g} "
            "sigma, which are kept as fluxes but given no magnitude")

    atlas = Table({
        'mjd': np.asarray(atlas['MJD'], dtype=float),
        'filter': np.asarray(atlas['F'], dtype=str),
        'flux': flux,
        'flux_err': flux_err,
        'mag': np.where(detected, ATLAS_AB_UJY - 2.5*np.log10(np.where(detected, flux, 1.0)), np.nan),
        'magerr': np.where(detected, 2.5/np.log(10)*flux_err/np.where(detected, flux, 1.0), np.nan),
        'mag5sig': np.asarray(atlas['mag5sig'], dtype=float),
        'sky': np.asarray(atlas['Sky'], dtype=float),
        'chi_N': np.asarray(atlas['chi_N'], dtype=float),
        'obs': np.asarray(atlas['Obs'], dtype=str),
    })

    atlas.sort('mjd')

    if not np.any(np.isfinite(atlas['mag'])):
        log("Warning: No ATLAS detections left at this level of filtering")
        return

    g_minus_r, g_minus_r_origin = assumed_color(config, 'g_minus_r')

    idx_c = np.asarray(atlas['filter'] == 'c') & np.isfinite(atlas['mag'])
    idx_o = np.asarray(atlas['filter'] == 'o') & np.isfinite(atlas['mag'])

    atlas['mag_g'] = np.nan
    atlas['mag_g'][idx_c] = atlas_c_to_g(atlas['mag'][idx_c], g_minus_r)
    atlas['mag_g'][idx_o] = atlas_o_to_g(atlas['mag'][idx_o], g_minus_r)

    log_conversion(
        log, 'ATLAS',
        f'm = {ATLAS_AB_UJY} - 2.5*log10(uJy)',
        {'system': ('AB', 'as the service calibrates it, against ATLAS-RefCat2')},
        npoints=int(np.sum(idx_c | idx_o)),
    )

    for band, idx, formula in (('c', idx_c, ATLAS_C_TO_G_FORMULA),
                               ('o', idx_o, ATLAS_O_TO_G_FORMULA)):
        if np.any(idx):
            log_conversion(
                log, 'ATLAS', formula,
                {'(g - r)': (g_minus_r, g_minus_r_origin)},
                npoints=int(np.sum(idx)),
                note='Tonry et al. (2018), equation 2 - an approximation for '
                     'stellar spectra, with the colour assumed constant',
            )

    log_bands(log, 'ATLAS', [
        {'label': 'c', 'kind': 'native', 'npoints': int(np.sum(idx_c)),
         'note': 'as measured by ATLAS, AB'},
        {'label': 'o', 'kind': 'native', 'npoints': int(np.sum(idx_o)),
         'note': 'as measured by ATLAS, AB'},
        {'label': 'g (from c)', 'kind': 'derived', 'npoints': int(np.sum(idx_c)),
         'note': 'c on the common g scale'},
        {'label': 'g (from o)', 'kind': 'derived', 'npoints': int(np.sum(idx_o)),
         'note': 'o on the common g scale'},
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'atlas_lc.png'),
                            figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        when = Time(np.asarray(atlas['mjd'], dtype=float), format='mjd').datetime

        for band, idx in (('c', idx_c), ('o', idx_o)):
            if np.any(idx):
                plot_with_errors(ax, when[idx], atlas['mag'][idx],
                                 atlas['magerr'][idx], label=band,
                                 color=ATLAS_COLORS[band])

        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        ax.legend()
        ax.set_ylabel('AB magnitude')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - ATLAS forced photometry")

    log("ATLAS lightcurve plot saved to file:atlas_lc.png")

    atlas.write(os.path.join(basepath, 'atlas.vot'), format='votable', overwrite=True)
    atlas.write(os.path.join(basepath, 'atlas.txt'),
                format='ascii.commented_header', overwrite=True)
    log("ATLAS data written to file:atlas.vot")
    log("ATLAS data written to file:atlas.txt")
