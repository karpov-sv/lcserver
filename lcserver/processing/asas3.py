"""ASAS-3 lightcurve acquisition module.

Acquires photometry from the All Sky Automated Survey, which watched the sky
south of +28 degrees between 2000 and 2009 - the decade before ASAS-SN.
"""

import os
import re
import requests
import numpy as np

from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query, log_bands,
                    log_conversion, assumed_color, v_to_g, V_TO_G_FORMULA)


ASAS_CATALOG_URL = 'https://www.astrouw.edu.pl/cgi-asas/asas_cat_input'
ASAS_DATA_URL = 'https://www.astrouw.edu.pl/cgi-asas/asas_cgi_get_data'

# ASAS measures through five apertures, and which one to believe depends on how
# bright the star is: a wide aperture collects sky around a faint star, a narrow
# one clips the wings of a bright one. Upper magnitude bound for each aperture,
# smallest first.
ASAS_APERTURE_LIMITS = [(11.0, 0), (10.0, 1), (9.0, 2), (8.0, 3)]
ASAS_APERTURE_BRIGHT = 4

# A is the best data and B the average; C carries points that were not measured
# at all, and D is described by the survey itself as probably useless
ASAS_GRADES = ('A', 'B')

# Search box for the catalogue query, in arcsec
ASAS_SR = 15.0

# Fewest points a catalogue entry must have to be worth returning
ASAS_NMIN = 4

# Data rows are HJD-2450000
ASAS_MJD0 = 2450000 - 2400000.5


def _choose_aperture(mag):
    """The aperture ASAS recommends for a star of this brightness."""
    if mag is None or not np.isfinite(mag):
        return 2  # the middle one, when the brightness is not known

    for limit, aperture in ASAS_APERTURE_LIMITS:
        if mag > limit:
            return aperture

    return ASAS_APERTURE_BRIGHT


def _query_catalogue(coords, sr, nmin, log):
    """Ask the catalogue which ASAS object sits at this position.

    Returns its designation, or None. The reply is a small HTML page whose
    links carry the designations; the first, emblazoned on the queried
    coordinates themselves, is the survey's own best match.
    """
    # The form wants sexagesimal - decimal degrees come back as 'Not found'
    coo = coords.to_string('hmsdms', sep=':', precision=1)

    res = requests.post(ASAS_CATALOG_URL, timeout=60, data={
        'source': 'asas3',
        'coo': coo,
        'equinox': '2000',
        'nmin': str(int(nmin)),
        'box': f'{sr:.0f}',
    })
    res.raise_for_status()

    designations = re.findall(r'/cgi-asas/asas_variable/([^,]+),asas3', res.text)

    if not designations:
        log(f"Warning: No ASAS-3 object within {sr:.0f} arcsec of {coo}")
        return None

    # Keeping the order, so that the survey's own best match comes first
    unique = list(dict.fromkeys(designations))
    if len(unique) > 1:
        log(f"{len(unique)} ASAS-3 objects nearby ({', '.join(unique)}), "
            f"using {unique[0]}")

    return unique[0]


def _download_lightcurve(designation, log):
    """The photometry of one ASAS object, as the raw text the survey serves.

    The server ends the response without terminating the stream, so a plain
    request raises rather than returning the body. The payload that arrives is
    complete - its own #ndata headers account for every row - so it is read as
    a stream and a premature end is accepted rather than losing all of it.
    """
    url = f"{ASAS_DATA_URL}?{designation},asas3"

    chunks = []
    try:
        with requests.post(url, data={'desig': designation},
                           timeout=120, stream=True) as res:
            res.raise_for_status()
            for chunk in res.iter_content(chunk_size=65536):
                chunks.append(chunk)
    except requests.exceptions.ChunkedEncodingError:
        pass

    return b''.join(chunks).decode('utf-8', errors='ignore')


def _parse_lightcurve(text, log):
    """Turn the served text into a table.

    ASAS keeps its photometry per observed field, so a star covered by more
    than one comes back as several datasets whose mean magnitudes, as the
    header warns, may differ slightly. The dataset is kept per point.
    """
    mjd, mags, errs, grade, dataset = [], [], [], [], []
    current = ''
    declared = []

    for line in text.splitlines():
        line = line.strip()

        if line.startswith('#dataset='):
            current = line.split('=', 1)[1].strip()
            continue

        if line.startswith('#ndata='):
            declared.append(int(line.split('=', 1)[1]))
            continue

        if not line or line.startswith('#'):
            continue

        fields = line.split()
        # HJD, five magnitudes, five errors, grade, frame
        if len(fields) < 12:
            continue

        try:
            mjd.append(float(fields[0]) + ASAS_MJD0)
            mags.append([float(_) for _ in fields[1:6]])
            errs.append([float(_) for _ in fields[6:11]])
        except ValueError:
            continue

        grade.append(fields[11])
        dataset.append(current)

    if declared and sum(declared) != len(mjd):
        log(f"Warning: {len(mjd)} rows parsed but {sum(declared)} declared - "
            "the download may be incomplete")

    if not mjd:
        return None

    table = Table({
        'mjd': np.array(mjd),
        'grade': np.array(grade),
        'dataset': np.array(dataset, dtype='<U32'),
    })

    mags, errs = np.array(mags), np.array(errs)
    for i in range(5):
        table[f'mag_{i}'] = mags[:, i]
        table[f'magerr_{i}'] = errs[:, i]

    return table


@survey_source(
    name='ASAS-3',
    short_name='ASAS-3',
    state_acquiring='acquiring ASAS-3 lightcurve',
    state_acquired='ASAS-3 lightcurve acquired',
    log_file='asas3.log',
    output_files=['asas3.log', 'asas3_lc.png', 'asas3.vot', 'asas3.txt'],
    button_text='Get ASAS-3 lightcurve',
    form_fields={
        'asas3_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': ASAS_SR,
            'required': False,
        },
        'asas3_aperture': {
            'type': 'choice',
            'label': 'Aperture',
            'choices': [('auto', 'Automatic (by brightness)')] + [
                (str(i), f'MAG_{i}') for i in range(5)],
            'initial': 'auto',
            'required': False,
        },
    },
    help_text='All Sky Automated Survey, V band, 2000-2009, south of +28 deg',
    order=21,
    # Lightcurve metadata
    votable_file='asas3.vot',
    lc_bands=[
        surveys.band('V', 'mag', 'magerr', surveys.BAND_NATIVE,
                     color='#2ca02c', note='as reported by ASAS-3'),
        surveys.band('g (conv.)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     color='#98df8a',
                     note='V put on the common g scale using an assumed g - r',
                     combined=True),
    ],
    lc_mag_column='mag_g',
    lc_err_column='magerr',
    lc_color='#2ca02c',
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata
    template_layout='simple',
    declination_max=28.0,
)
def target_asas3(config, basepath=None, verbose=True, show=False):
    """
    Get ASAS-3 lightcurve.

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
    cleanup_paths(get_output_files('asas3'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    asas3_sr = config.get('asas3_sr', ASAS_SR)
    coords = SkyCoord(ra, dec, unit='deg')

    if dec > 28:
        log(f"Declination {dec:.1f} is north of +28, which ASAS-3 never observed")
        return

    cache_name = f"asas3_{ra:.4f}_{dec:.4f}_{asas3_sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'ASAS-3',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            log(f"within {asas3_sr:.0f} arcsec")

            try:
                designation = _query_catalogue(coords, asas3_sr, ASAS_NMIN, log)

                if designation is None:
                    return

                log(f"Requesting the light curve of ASAS {designation}")
                text = _download_lightcurve(designation, log)
                asas3 = _parse_lightcurve(text, log)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise SourceError("could not download the data - "
                                  f"{type(e).__name__}: {e}")

            if asas3 is None or not len(asas3):
                log("Warning: No ASAS-3 data points found")
                return

            cache.save(asas3)

        asas3 = cache.data

    log(f"{len(asas3)} original data points")

    # Which aperture to believe
    setting = str(config.get('asas3_aperture', 'auto'))
    if setting == 'auto':
        aperture = _choose_aperture(np.nanmedian(np.asarray(asas3['mag_2'], dtype=float)))
        origin = 'automatic, by the brightness of the star'
    else:
        aperture = int(setting)
        origin = 'from config'

    log(f"Using aperture MAG_{aperture} ({origin})")

    asas3['mag'] = asas3[f'mag_{aperture}']
    asas3['magerr'] = asas3[f'magerr_{aperture}']
    asas3['aperture'] = aperture

    # Quality: ASAS grades every point, and only the best two are worth keeping
    grades = np.asarray(asas3['grade']).astype(str)
    keep = np.isin(grades, ASAS_GRADES)
    keep &= np.isfinite(np.asarray(asas3['mag'], dtype=float))
    # Points that could not be measured are marked with this rather than a null
    keep &= np.asarray(asas3['mag'], dtype=float) < 29.0

    for g in sorted(set(grades)):
        n = int(np.sum(grades == g))
        log(f"  grade {g}: {n} points" + ('' if g in ASAS_GRADES else ', dropped'))

    asas3 = asas3[keep]

    if not len(asas3):
        log("Warning: No ASAS-3 data points left after filtering")
        return

    datasets = sorted(set(np.asarray(asas3['dataset']).astype(str)))
    if len(datasets) > 1:
        log(f"{len(datasets)} datasets, one per observed field; their mean "
            "magnitudes may differ slightly, and are kept apart in the dataset column")

    # The common g scale, the same way ASAS-SN gets there from its own V
    g_minus_r, g_minus_r_origin = assumed_color(config, 'g_minus_r')

    asas3['mag_g'] = v_to_g(asas3['mag'], g_minus_r)

    log_conversion(
        log, 'ASAS-3',
        V_TO_G_FORMULA,
        {'(g - r)': (g_minus_r, g_minus_r_origin),
         'aperture': (f'MAG_{aperture}', origin)},
        npoints=len(asas3),
        note='the colour is assumed constant over the whole light curve',
    )

    log_bands(log, 'ASAS-3', [
        {'label': 'V', 'kind': 'native', 'npoints': len(asas3),
         'note': f'as reported by ASAS-3, aperture MAG_{aperture}'},
        {'label': 'g (conv.)', 'kind': 'derived', 'npoints': len(asas3),
         'note': 'on the common g scale'},
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'asas3_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        time = Time(np.asarray(asas3['mjd'], dtype=float), format='mjd')

        for name in datasets:
            idx = np.asarray(asas3['dataset']).astype(str) == name
            ax.errorbar(time[idx].datetime, asas3['mag'][idx], asas3['magerr'][idx],
                        fmt='.', alpha=0.5, label=name or 'ASAS-3')

        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        if len(datasets) > 1:
            ax.legend(fontsize='small')
        ax.set_ylabel('V')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - ASAS-3")

    log("ASAS-3 lightcurve plot saved to file:asas3_lc.png")

    asas3.write(os.path.join(basepath, 'asas3.vot'), format='votable', overwrite=True)
    asas3.write(os.path.join(basepath, 'asas3.txt'), format='ascii.commented_header', overwrite=True)
    log("ASAS-3 data written to file:asas3.vot")
    log("ASAS-3 data written to file:asas3.txt")
