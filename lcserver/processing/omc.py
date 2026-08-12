"""INTEGRAL OMC lightcurve acquisition module.

Acquires V-band photometry from the Optical Monitoring Camera aboard INTEGRAL,
as served by the OMC Archive at CAB (INTA-CSIC).
"""

import os
import re
import io
import requests
import numpy as np

from astropy.table import Table
from astropy.io import fits
from astropy.time import Time

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import cleanup_paths, cached_votable_query, log_bands, log_conversion


OMC_FORM_URL = 'https://sdc.cab.inta-csic.es/omc/secure/form_busqueda.jsp'
OMC_FETCH_URL = 'https://sdc.cab.inta-csic.es/omc/secure/fetch_lcurve.jsp'

# The archive answers a partial submission by quietly re-rendering the form, so
# every field the search page carries is sent, defaults and all. The ones that
# matter for getting an answer at all are submit, output_format,
# results_per_page and page_to_show.
OMC_FORM_DEFAULTS = {
    'obj_id': '', 'obj_list': '', 'submit': 'Submit',
    'vmag_init': '', 'vmag_end': '',
    'ra': '', 'de': '', 'rad': '',
    'dateinit_d': '', 'dateinit_m': '', 'dateinit_y': '',
    'dateend_d': '', 'dateend_m': '', 'dateend_y': '',
    # Ten-minute binning with the source coordinates as centroid, which is what
    # the form offers by default; stated here rather than inherited silently
    'lct_tstep': '630',
    'lct_wcsflag': 'Y',
    'obj_sstar': 'S',
    'crv_numpoints': '1',
    'obj_prio': '1',
    'output_format': 'html',
    'order_by': '',
    'results_per_page': '50',
    'page_to_show': '1',
}

# Matching radius, in arcsec. The form wants arcminutes - a value meant as
# arcseconds would ask for a cone a hundred times too wide, and OMC is crowded
# enough at 17.5 arcsec per pixel for that to matter.
OMC_SR = 30.0

# INTEGRAL counts its days from 2000-01-01
OMC_MJD0 = 51544.0

# The apertures OMC reports, and the column each lives in. MAG_V is the
# survey's own choice and equals the 3x3 one.
OMC_APERTURES = {
    'standard': ('MAG_V', 'ERRMAG_V'),
    '1': ('MAG_V1', 'ERMAG_V1'),
    '3': ('MAG_V3', 'ERMAG_V3'),
    '5': ('MAG_V5', 'ERMAG_V5'),
}


def _search(session, log, **query):
    """Ask the archive which OMC objects match, and how to fetch each.

    Returns a list of dicts carrying the name, the OMC identifier, the position
    and the pair of ids the light curve is fetched with.
    """
    fields = dict(OMC_FORM_DEFAULTS)
    fields.update(query)

    res = session.post(OMC_FORM_URL, timeout=180,
                       files={k: (None, v) for k, v in fields.items()})
    res.raise_for_status()

    found = re.search(r'(\d+)\s*Object[s]?\s*found', res.text)
    if found:
        log(f"{found.group(1)} objects found")

    objects = []

    for row in re.findall(r'<tr[^>]*>(.*?)</tr>', res.text, re.S | re.I):
        link = re.search(r'fetch_lcurve\.jsp\?obj_id=(\d+)&lct_id=(\w+)', row)
        if not link:
            continue

        cells = [re.sub(r'<[^>]*>', ' ', c) for c in
                 re.findall(r'<td[^>]*>(.*?)</td>', row, re.S | re.I)]
        cells = [re.sub(r'&nbsp;?', ' ', c).strip() for c in cells]
        cells = [c for c in cells if c]

        entry = {'obj_id': link.group(1), 'lct_id': link.group(2),
                 'name': cells[0] if cells else '', 'omc_id': '',
                 'ra': np.nan, 'dec': np.nan, 'npoints': None}

        # name, OMC id, RA, Dec, V, ... with the counts further along
        numbers = []
        for c in cells[1:]:
            try:
                numbers.append(float(c.split()[0]))
            except (ValueError, IndexError):
                numbers.append(None)

        if len(cells) > 1:
            entry['omc_id'] = cells[1]
        if len(numbers) > 2 and numbers[1] is not None and numbers[2] is not None:
            entry['ra'], entry['dec'] = numbers[1], numbers[2]

        objects.append(entry)

    return objects


def _votable_safe(table):
    """Widen the integer columns a VOTable cannot hold.

    FITS lets a column be unsigned; VOTable only has unsignedByte, so anything
    wider comes back as an error when the table is cached. Widening to the next
    signed type keeps every value exactly.
    """
    for name in table.colnames:
        if table[name].dtype.kind == 'u' and table[name].dtype.itemsize > 1:
            table[name] = table[name].astype(f'i{2*table[name].dtype.itemsize}')

    return table


def _download_lightcurve(session, obj_id, lct_id):
    """The light curve of one OMC object, as the FITS table the archive sends."""
    res = session.get(OMC_FETCH_URL, timeout=300,
                      params={'obj_id': obj_id, 'lct_id': lct_id})
    res.raise_for_status()

    with fits.open(io.BytesIO(res.content)) as hdus:
        return _votable_safe(Table(hdus[1].data))


@survey_source(
    name='INTEGRAL OMC',
    short_name='OMC',
    state_acquiring='acquiring OMC lightcurve',
    state_acquired='OMC lightcurve acquired',
    log_file='omc.log',
    output_files=['omc.log', 'omc_lc.png', 'omc.vot', 'omc.txt'],
    button_text='Get OMC lightcurve',
    form_fields={
        'omc_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': OMC_SR,
            'required': False,
        },
        'omc_aperture': {
            'type': 'choice',
            'label': 'Aperture',
            'choices': [('standard', 'Standard (MAG_V)'),
                        ('1', '1x1 pixel'), ('3', '3x3 pixels'), ('5', '5x5 pixels')],
            'initial': 'standard',
            'required': False,
        },
    },
    help_text='Optical Monitoring Camera aboard INTEGRAL, V band, since 2003',
    order=24,
    # Lightcurve metadata
    votable_file='omc.vot',
    lc_bands=[
        surveys.band('V', 'mag', 'magerr', surveys.BAND_NATIVE,
                     color='#17becf', note='as reported by OMC'),
        surveys.band('g (conv.)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     color='#9edae5',
                     note='V put on the common g scale using an assumed g - r'),
    ],
    lc_mag_column='mag_g',
    lc_err_column='magerr',
    lc_color='#17becf',
    lc_mode='magnitude',
    lc_short=False,
    # Template metadata. No cutout: OMC publishes no HiPS of its own, and a
    # sky survey's image would say nothing about what OMC itself saw.
    template_layout='simple',
)
def target_omc(config, basepath=None, verbose=True, show=False):
    """
    Get INTEGRAL OMC lightcurve.

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
    cleanup_paths(get_output_files('omc'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    omc_sr = config.get('omc_sr', OMC_SR)

    cache_name = f"omc_{ra:.4f}_{dec:.4f}_{omc_sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'INTEGRAL OMC',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            log(f"within {omc_sr:.0f} arcsec")

            try:
                session = requests.Session()
                # The archive hands out a session on the form page, and expects
                # to see it again on the search and the download
                session.get(OMC_FORM_URL, params={'resetForm': 'true'}, timeout=60)

                objects = _search(session, log, ra=f'{ra:.6f}', de=f'{dec:.6f}',
                                  rad=f'{omc_sr/60:.4f}')

                if not objects:
                    log("Warning: No OMC object at this position - INTEGRAL "
                        "only observed where it was pointed")
                    return

                # The cone may hold several, so take the one actually asked for
                distance = [np.hypot((_['ra'] - ra)*np.cos(np.deg2rad(dec)),
                                     _['dec'] - dec)
                            if np.isfinite(_['ra']) else np.inf
                            for _ in objects]
                closest = objects[int(np.argmin(distance))]

                if len(objects) > 1:
                    log(f"{len(objects)} OMC objects within the cone, using "
                        f"{closest['omc_id'] or closest['name']} at "
                        f"{np.min(distance)*3600:.1f} arcsec")

                log(f"Requesting the light curve of OMC {closest['omc_id']}"
                    + (f" ({closest['name']})" if closest['name'] else ''))

                omc = _download_lightcurve(session, closest['obj_id'], closest['lct_id'])
            except:
                import traceback
                traceback.print_exc()
                log("Error: could not download the data")
                return

            if omc is None or not len(omc):
                log("Warning: No OMC data points found")
                return

            cache.save(omc)

        omc = cache.data

    log(f"{len(omc)} original data points")

    # Which aperture to believe
    setting = str(config.get('omc_aperture', 'standard'))
    magcol, errcol = OMC_APERTURES.get(setting, OMC_APERTURES['standard'])

    if magcol not in omc.colnames or errcol not in omc.colnames:
        log(f"Error: aperture column {magcol} is missing from the reply")
        return

    log(f"Using {magcol}" + (" (the survey's own choice, a 3x3 pixel aperture)"
                             if setting == 'standard' else ''))

    # INTEGRAL counts its days from 2000-01-01 rather than quoting MJD
    omc['mjd'] = np.asarray(omc['BARYTIME'], dtype=float) + OMC_MJD0
    omc['mag'] = np.asarray(omc[magcol], dtype=float)
    omc['magerr'] = np.asarray(omc[errcol], dtype=float)

    idx = np.isfinite(omc['mjd']) & np.isfinite(omc['mag']) & np.isfinite(omc['magerr'])
    idx &= omc['magerr'] > 0

    # OMC flags what it is unhappy with, and zero means it is content
    if 'PROBLEMS' in omc.colnames:
        problems = np.asarray(omc['PROBLEMS'], dtype=int)
        flagged = problems != 0
        if np.any(flagged):
            codes, counts = np.unique(problems[flagged], return_counts=True)
            log("  flagged by OMC: "
                + ', '.join(f"{c} with PROBLEMS={int(v)}" for v, c in zip(codes, counts)))
        idx &= ~flagged

    log(f"{int(np.sum(idx))} data points after filtering")

    omc = omc[idx]
    omc.sort('mjd')

    if not len(omc):
        log("Warning: No valid OMC data points")
        return

    # The common g scale, as for the other V-band surveys
    g_minus_r = config.get('g_minus_r', 0.0)
    g_minus_r_origin = 'from config' if 'g_minus_r' in config else 'default, no colour known'

    omc['mag_g'] = omc['mag'] + 0.02 + 0.498*g_minus_r + 0.008*g_minus_r**2

    log_conversion(
        log, 'OMC',
        'g = V + 0.02 + 0.498*(g-r) + 0.008*(g-r)^2',
        {'(g - r)': (g_minus_r, g_minus_r_origin),
         'aperture': magcol},
        npoints=len(omc),
        note='the colour is assumed constant over the whole light curve',
    )

    log_bands(log, 'OMC', [
        {'label': 'V', 'kind': 'native', 'npoints': len(omc),
         'note': f'as reported by OMC, {magcol}'},
        {'label': 'g (conv.)', 'kind': 'derived', 'npoints': len(omc),
         'note': 'on the common g scale'},
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'omc_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        time = Time(np.asarray(omc['mjd'], dtype=float), format='mjd')
        ax.errorbar(time.datetime, omc['mag'], omc['magerr'],
                    fmt='.', alpha=0.5, color='#17becf')

        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        ax.set_ylabel('V')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - INTEGRAL OMC")

    log("OMC lightcurve plot saved to file:omc_lc.png")

    omc.write(os.path.join(basepath, 'omc.vot'), format='votable', overwrite=True)
    omc.write(os.path.join(basepath, 'omc.txt'), format='ascii.commented_header', overwrite=True)
    log("OMC data written to file:omc.vot")
    log("OMC data written to file:omc.txt")
