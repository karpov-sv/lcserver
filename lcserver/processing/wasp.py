"""SuperWASP lightcurve acquisition module.

Acquires photometry from the public release of the Wide Angle Search for
Planets, as served by the CERIT-SC archive.
"""

import os

import re
import requests
import numpy as np

from astropy.table import Table
from astropy.time import Time

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    clip_noisy_points, log_bands, log_conversion)


WASP_SEARCH_URL = 'https://wasp.cerit-sc.cz/search'
WASP_CSV_URL = 'https://wasp.cerit-sc.cz/csv'

# Matching radius, in arcsec. WASP pixels are some 14 arcsec across, so a
# tighter cone would start missing the object rather than excluding neighbours.
WASP_SR = 30.0

# Fewest points an object must carry to be worth returning
WASP_NMIN = 10

# The archive quotes HJD
WASP_MJD0 = -2400000.5

# How much worse than the star's own typical error a camera may be before it
# is dropped whole. Across the stars this was checked on, an ordinary camera
# quotes between a tenth and twice the median error for its star, while the
# ones that saw it badly start at ten times - so five separates them with room
# on either side. A camera holding most of the points cannot exceed the median
# of a sample it dominates, so this can never empty a light curve.
WASP_CAMERA_MAX_ERR_RATIO = 5.0


def _search(ra, dec, sr, nmin, log):
    """Which SuperWASP objects sit at this position.

    The archive answers in HTML, so the identifiers are read out of the result
    table. Each row carries the object name, its number of points and its
    distance from the requested position.
    """
    res = requests.get(WASP_SEARCH_URL, timeout=90, params={
        'ra': f'{ra:.6f}',
        'dec': f'{dec:.6f}',
        'radius': f'{sr:.1f}',
        'radiusUnit': 'sec',
        'ptsmin': str(int(nmin)),
        'limit': '10',
    })
    res.raise_for_status()

    if 'No objects matching' in res.text:
        return []

    objects = []
    for row in re.findall(r'<tr[^>]*>(.*?)</tr>', res.text, re.S | re.I):
        cells = [re.sub(r'<[^>]*>', ' ', c).strip()
                 for c in re.findall(r'<td[^>]*>(.*?)</td>', row, re.S | re.I)]

        name = next((c for c in cells if c.startswith('1SWASP')), None)
        if not name:
            continue

        # The count sits in the cell after the name
        npts = None
        try:
            npts = int(cells[cells.index(name) + 1])
        except (ValueError, IndexError):
            pass

        objects.append({'name': name, 'npts': npts})

    return objects


def _download_lightcurve(name, log):
    """The photometry of one SuperWASP object, as a table."""
    res = requests.get(WASP_CSV_URL, timeout=180, params={'object': name})
    res.raise_for_status()

    # As a list of lines rather than a stream, which the fast reader declines
    return Table.read(res.text.splitlines(), format='ascii.csv')


@survey_source(
    name='SuperWASP',
    short_name='WASP',
    state_acquiring='acquiring SuperWASP lightcurve',
    state_acquired='SuperWASP lightcurve acquired',
    log_file='wasp.log',
    output_files=['wasp.log', 'wasp_lc.png', 'wasp.vot', 'wasp.txt'],
    button_text='Get SuperWASP lightcurve',
    form_fields={
        'wasp_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': WASP_SR,
            'required': False,
        }
    },
    help_text='Wide Angle Search for Planets, broad band, 2004-2008',
    order=22,
    # Lightcurve metadata
    votable_file='wasp.vot',
    lc_bands=[
        surveys.band('V', 'mag', 'magerr', surveys.BAND_NATIVE,
                     color='#ff7f0e',
                     note='broad WASP band, calibrated against Tycho-2 V'),
        surveys.band('g (conv.)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     color='#ffbb78',
                     note='V put on the common g scale using an assumed g - r'),
    ],
    lc_mag_column='mag_g',
    lc_err_column='magerr',
    lc_color='#ff7f0e',
    lc_mode='magnitude',
    lc_short=False,
    # Template metadata
    template_layout='simple',
)
def target_wasp(config, basepath=None, verbose=True, show=False):
    """
    Get SuperWASP lightcurve.

    Parameters
    ----------
    config : dict
        Configuration dictionary with target coordinates
    basepath : str, optional
        Base path for output files
    verbose : bool or callable, optional
        Verbose logging mode or log function
    show : bool, optional
        Show plots interactively
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('wasp'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    wasp_sr = config.get('wasp_sr', WASP_SR)

    cache_name = f"wasp_{ra:.4f}_{dec:.4f}_{wasp_sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'SuperWASP',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            log(f"within {wasp_sr:.0f} arcsec")

            try:
                objects = _search(ra, dec, wasp_sr, WASP_NMIN, log)

                if not objects:
                    log("Warning: No SuperWASP object at this position - its coverage, "
                        "though wide, is not the whole sky")
                    return

                if len(objects) > 1:
                    log(f"{len(objects)} SuperWASP objects nearby, "
                        f"using the first ({objects[0]['name']})")

                obj = objects[0]
                log(f"Requesting the light curve of {obj['name']}"
                    + (f", {obj['npts']} points" if obj['npts'] else ''))

                wasp = _download_lightcurve(obj['name'], log)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise SourceError("could not download the data - "
                                  f"{type(e).__name__}: {e}")

            if wasp is None or not len(wasp):
                log("Warning: No SuperWASP data points found")
                return

            cache.save(wasp)

        wasp = cache.data

    log(f"{len(wasp)} original data points")

    # The archive names its columns for people rather than for programs, and
    # 'magnitude error' comes back as 'magnitude_error' once it has been
    # through a VOTable, so spaces and underscores are treated alike
    columns = {c.lower().strip().replace(' ', '_'): c for c in wasp.colnames}
    hjd = columns.get('hjd')
    mag = columns.get('magnitude')
    err = columns.get('magnitude_error')

    if not (hjd and mag and err):
        log(f"Unexpected columns in the reply: {wasp.colnames}")
        return

    wasp['mjd'] = np.asarray(wasp[hjd], dtype=float) + WASP_MJD0
    wasp['mag'] = np.asarray(wasp[mag], dtype=float)
    wasp['magerr'] = np.asarray(wasp[err], dtype=float)

    idx = np.isfinite(wasp['mjd']) & np.isfinite(wasp['mag']) & np.isfinite(wasp['magerr'])
    idx &= wasp['magerr'] > 0
    # A handful of points carry uncertainties of several magnitudes, which say
    # nothing about the star. The same cut the other surveys here apply.
    idx &= wasp['magerr'] < 1.0

    log(f"{int(np.sum(idx))} data points after filtering")

    wasp = wasp[idx]
    wasp.sort('mjd')

    if not len(wasp):
        log("Warning: No valid SuperWASP data points")
        return

    # WASP observed with several cameras, and one of them may have seen this
    # particular star badly - saturated, out of focus for something this
    # bright, or on a poor part of the chip. It shows in the errors it quotes,
    # and it affects everything that camera took rather than a stretch of it,
    # so the camera goes as a whole. Which one it is differs from star to star
    # - the camera worst here is the best elsewhere - so it is found from the
    # data each time rather than known in advance.
    if 'camera' in columns:
        camera = np.asarray(wasp[columns['camera']]).astype(str)
        cameras = sorted(set(camera))
        log(f"{len(cameras)} cameras contributed: {', '.join(cameras)}")

        typical = np.median(wasp['magerr'])
        keep = np.ones(len(wasp), dtype=bool)

        for name in cameras:
            idx = camera == name
            ratio = np.median(wasp['magerr'][idx]) / typical

            if ratio > WASP_CAMERA_MAX_ERR_RATIO:
                log(f"Warning: dropping camera {name} - its {int(np.sum(idx))} "
                    f"points carry errors {ratio:.0f} times the median for this "
                    f"star, which is the camera rather than the star")
                keep &= ~idx

        dropped = not np.all(keep)

        if dropped:
            wasp = wasp[keep]
            camera = camera[keep]

        # What is left to catch inside a camera is the single ruined frame,
        # which the archive reports honestly as a large error
        clip = clip_noisy_points(wasp['mag'], wasp['magerr'], camera,
                                 log=log, group_name='camera')

        if np.any(clip):
            wasp = wasp[~clip]
            dropped = True

        if dropped:
            log(f"{len(wasp)} data points left")

    # The common g scale, as for the other V-band surveys
    g_minus_r = config.get('g_minus_r', 0.0)
    g_minus_r_origin = 'from config' if 'g_minus_r' in config else 'default, no colour known'

    wasp['mag_g'] = wasp['mag'] + 0.02 + 0.498*g_minus_r + 0.008*g_minus_r**2

    log_conversion(
        log, 'SuperWASP',
        'g = V + 0.02 + 0.498*(g-r) + 0.008*(g-r)^2',
        {'(g - r)': (g_minus_r, g_minus_r_origin)},
        npoints=len(wasp),
        note='the WASP band is broad and only approximately V, so this is a '
             'rougher footing than the same conversion applied to ASAS',
    )

    log_bands(log, 'SuperWASP', [
        {'label': 'V', 'kind': 'native', 'npoints': len(wasp),
         'note': 'broad WASP band, calibrated against Tycho-2 V'},
        {'label': 'g (conv.)', 'kind': 'derived', 'npoints': len(wasp),
         'note': 'on the common g scale'},
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'wasp_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        time = Time(np.asarray(wasp['mjd'], dtype=float), format='mjd')
        ax.errorbar(time.datetime, wasp['mag'], wasp['magerr'],
                    fmt='.', ms=3, alpha=0.3, color='#ff7f0e')

        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        ax.set_ylabel('WASP V')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - SuperWASP")

    log("SuperWASP lightcurve plot saved to file:wasp_lc.png")

    wasp.write(os.path.join(basepath, 'wasp.vot'), format='votable', overwrite=True)
    wasp.write(os.path.join(basepath, 'wasp.txt'), format='ascii.commented_header', overwrite=True)
    log("SuperWASP data written to file:wasp.vot")
    log("SuperWASP data written to file:wasp.txt")
