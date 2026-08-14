"""NSVS lightcurve acquisition module.

Acquires photometry from the Northern Sky Variability Survey, which ROTSE-I
recorded in 1999 and 2000 - the earliest of the modern wide-field surveys here.

SkyDOT at Los Alamos, which used to serve these light curves, no longer answers
on any port, so the data is taken from the copy CDS holds as catalogue II/287.
That copy is distributed a survey field at a time rather than an object at a
time, which shapes everything below.
"""

import os
import zlib
import requests
import numpy as np

from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

from astroquery.vizier import Vizier

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    quality_field, quality_level, log_bands, log_conversion,
                    assumed_color, rotse_to_v, v_to_g,
                    ROTSE_TO_V_FORMULA, V_TO_G_FORMULA,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


NSVS_BASE_URL = 'https://cdsarc.cds.unistra.fr/ftp/II/287'

# The object catalogue and the field list, both small enough to query remotely
NSVS_OBJECTS = 'II/287/skydot'
NSVS_FIELDS = 'II/287/fields'

# Matching radius, in arcsec. ROTSE-I pixels are some 14 arcsec across.
NSVS_SR = 20.0

# Magnitudes and their errors are stored as millimagnitudes
NSVS_MMAG = 1e-3

# Offsets from the median centroid are in units of 1/32000 degree
NSVS_OFFSET = 1.0/32000

# How much of a field file to read before giving up, in bytes. The largest is
# some 190 MB, and the rows are ordered by object, so a search normally stops
# far sooner than this.
NSVS_MAX_READ = 220 << 20

# What the survey attaches to each measurement, from Table 2 of Wozniak et al.
# 2004 (AJ 127, 2436). The low byte is what SExtractor made of the object on
# the frame; the high byte is how the relative photometry of that part of the
# sky went.
NSVS_FLAGS = [
    (0x0001, 'NEIGHBORS', 'a neighbour or bad pixels over a tenth of the object'),
    (0x0002, 'BLENDED', 'deblended from another object'),
    (0x0004, 'SATURATED', 'at least one saturated pixel'),
    (0x0008, 'ATEDGE', 'truncated by the image boundary'),
    (0x0010, 'APINCOMPL', 'aperture data incomplete or corrupted'),
    (0x0020, 'ISINCOMPL', 'isophotal data incomplete or corrupted'),
    (0x0040, 'DBMEMOVR', 'memory overflow while deblending'),
    (0x0080, 'EXMEMOVR', 'memory overflow while extracting'),
    (0x0100, 'NOCORR', 'no relative photometry correction could be calculated'),
    (0x0200, 'PATCH', 'the correction map was patched to derive one'),
    (0x0400, 'LONPTS', 'fewer than ten points in the macro-pixel'),
    (0x0800, 'HISCAT', 'macro-pixel scatter above 0.2 mag'),
    (0x1000, 'HICORR', 'correction above 0.1 mag'),
    (0x2000, 'HISIGCORR', 'corrections across the map scattered by over 0.1 mag'),
    (0x4000, 'RADECFLIP', 'mount flip near the pole'),
]

# What the survey itself will not call a good photometric point, in Table 3 of
# the same paper: saturation, and every sign that the relative photometry of
# that macro-pixel or that frame cannot be trusted. Blending is not among them
# - a blend measures the wrong star rather than measuring badly, and whether
# that matters depends on what is being asked.
NSVS_BAD_FLAGS = (0x0004 | 0x0100 | 0x0400 | 0x0800 | 0x1000 | 0x2000 | 0x4000)

# The one nobody argues about, and all that the relaxed level removes: a
# saturated star is not being measured at all. Dropping only this keeps twice
# the points and nearly three times the baseline on a bright star, which is
# the better set to look for a period in, if not to measure an amplitude.
NSVS_SATURATED = 0x0004

# The same table's range checks, which are there to catch SExtractor's error
# codes rather than to judge the photometry
NSVS_MAG_RANGE = (5.0, 16.0)
NSVS_MAGERR_MAX = 0.4


def _find_object(ra, dec, sr, log):
    """The NSVS object at a position, from the catalogue CDS serves.

    Its record number is the object identifier the light curve files are keyed
    by, which is the only way to find a light curve in a file holding a whole
    survey field.
    """
    v = Vizier(columns=['**', 'recno'], row_limit=50)
    res = v.query_region(SkyCoord(ra, dec, unit='deg'), radius=sr*u.arcsec,
                         catalog=NSVS_OBJECTS)

    if not res or not len(res[0]):
        return None

    table = res[0]
    # Nearest first, so that a crowded field does not hand back a neighbour
    if '_r' in table.colnames:
        table = table[np.argsort(np.asarray(table['_r'], dtype=float))]

    row = table[0]
    if len(table) > 1:
        log(f"{len(table)} NSVS objects within {sr:.0f} arcsec, using the closest")

    return {
        'id': int(row['recno']),
        'ra': float(row['RAJ2000']),
        'dec': float(row['DEJ2000']),
        'mag': float(row['mag']),
        'nobs': int(row['Nobs']) if 'Nobs' in table.colnames else None,
        'ndet': int(row['Ndet']) if 'Ndet' in table.colnames else None,
    }


def _find_field(ra, dec, log):
    """Which survey field covers a position.

    The light curves are distributed one file per field, so this decides which
    file has to be fetched.
    """
    fields = Vizier(columns=['**'], row_limit=-1).get_catalogs(NSVS_FIELDS)[0]

    dist = np.hypot((np.asarray(fields['RAJ2000'], dtype=float) - ra)
                    * np.cos(np.deg2rad(dec)),
                    np.asarray(fields['DEJ2000'], dtype=float) - dec)
    closest = int(np.argmin(dist))

    return str(fields['Field'][closest]), float(dist[closest])


def _stream_lightcurve(field, object_id, log):
    """The measurements of one object, out of the file holding its whole field.

    The file is decompressed as it arrives and abandoned as soon as the object
    has gone by - its rows are ordered by object, so there is no reason to read
    the remaining tens of megabytes.
    """
    url = f"{NSVS_BASE_URL}/id/skydotID_{field}.dat.gz"

    decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
    tail = b''
    rows = []
    read = 0
    passed = False

    with requests.get(url, stream=True, timeout=600) as res:
        res.raise_for_status()

        for chunk in res.iter_content(1 << 20):
            read += len(chunk)
            buffer = tail + decompressor.decompress(chunk)
            *lines, tail = buffer.split(b'\n')

            for line in lines:
                parts = line.split()
                if len(parts) < 7:
                    continue

                try:
                    current = int(parts[0])
                except ValueError:
                    continue

                if current == object_id:
                    try:
                        rows.append([int(_) for _ in parts[:7]])
                    except ValueError:
                        continue
                elif current > object_id:
                    passed = True

            if passed and rows:
                break

            if read > NSVS_MAX_READ:
                log("Giving up on the field file before finding the object")
                break

    log(f"Read {read/1e6:.0f} MB of the field file")

    return np.array(rows) if rows else None


def _frame_times(log):
    """When each exposure was taken, indexed by the frame identifier.

    The light curves carry a frame number rather than a time, and the frame
    list is a separate file. Its own documentation describes a fixed-width
    layout the file does not use, and calls its first column the frame ID when
    that is the field number; the frame identifier is simply the position of
    the row in the file, which is why the whole list has to be read.
    """
    url = f"{NSVS_BASE_URL}/frames.dat.gz"

    log("Fetching the list of exposures for their times")

    res = requests.get(url, timeout=600)
    res.raise_for_status()

    times = [np.nan]  # frame identifiers start at one
    for line in zlib.decompress(res.content, zlib.MAX_WBITS | 16).decode(
            'utf-8', errors='ignore').splitlines():
        parts = line.split('|')
        if len(parts) < 4:
            continue
        try:
            times.append(float(parts[3]))
        except ValueError:
            times.append(np.nan)

    return np.array(times)


@survey_source(
    name='NSVS',
    short_name='NSVS',
    state_acquiring='acquiring NSVS lightcurve',
    state_acquired='NSVS lightcurve acquired',
    log_file='nsvs.log',
    output_files=['nsvs.log', 'nsvs_lc.png', 'nsvs.vot', 'nsvs.txt'],
    button_text='Get NSVS lightcurve',
    form_fields={
        'nsvs_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': NSVS_SR,
            'required': False,
        },
        'nsvs_quality': quality_field({
            QUALITY_STANDARD: "The survey's own definition of a good point",
            QUALITY_RELAXED: 'Saturated measurements only',
            QUALITY_PUBLISHED: 'None - every measurement as published',
        }),
    },
    help_text='Northern Sky Variability Survey, unfiltered, 1999-2000',
    order=25,
    # Lightcurve metadata
    votable_file='nsvs.vot',
    lc_bands=[
        surveys.band('R', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='R', color='#e377c2',
                     note='unfiltered ROTSE-I, closest to R'),
        surveys.band('V (conv.)', 'mag_V', 'magerr', surveys.BAND_DERIVED,
                     filter_column='filter', filter_value='R', color='#f7b6d2',
                     note="the survey's own colour term undone using an "
                          'assumed B - V, which puts it on Johnson V'),
        surveys.band('g (conv.)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     filter_column='filter', filter_value='R', color='#fbd4e4',
                     note='and on from V to the common g scale using an '
                          'assumed g - r',
                     combined=True),
    ],
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color='#e377c2',
    lc_mode='magnitude',
    lc_short=False,
    # Template metadata. No cutout: ROTSE-I published no imaging of its own.
    template_layout='simple',
)
def target_nsvs(config, basepath=None, verbose=True, show=False):
    """
    Get NSVS lightcurve.

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
    cleanup_paths(get_output_files('nsvs'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    nsvs_sr = config.get('nsvs_sr', NSVS_SR)

    cache_name = f"nsvs_{ra:.4f}_{dec:.4f}_{nsvs_sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'NSVS',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            log(f"within {nsvs_sr:.0f} arcsec")

            try:
                obj = _find_object(ra, dec, nsvs_sr, log)

                if obj is None:
                    log("Warning: No NSVS object at this position")
                    return

                log(f"NSVS object {obj['id']} at {obj['mag']:.2f} mag"
                    + (f", {obj['ndet']} detections of which the catalogue "
                       f"calls {obj['nobs']} good" if obj['ndet'] else ''))

                field, dist = _find_field(ra, dec, log)
                log(f"Covered by field {field}, {dist:.1f} deg from its centre")

                # Tens of megabytes, so it is worth saying so before the wait
                log("The survey is distributed a field at a time, so this "
                    "means fetching that whole field")

                rows = _stream_lightcurve(field, obj['id'], log)

                if rows is None:
                    raise SourceError("the object was not found in the field file")

                times = _frame_times(log)

                frames = rows[:, 1]
                mjd = np.where(frames < len(times), times[np.clip(frames, 0, len(times)-1)], np.nan)

                nsvs = Table({
                    'mjd': mjd,
                    'mag': rows[:, 4] * NSVS_MMAG,
                    'magerr': rows[:, 5] * NSVS_MMAG,
                    'filter': np.full(len(rows), 'R', dtype='<U2'),
                    'frame': frames,
                    'flags': rows[:, 6],
                    # Where the measurement fell relative to the median centroid
                    'ora': rows[:, 2] * NSVS_OFFSET,
                    'odec': rows[:, 3] * NSVS_OFFSET,
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise SourceError("could not download the data - "
                                  f"{type(e).__name__}: {e}")

            if not len(nsvs):
                log("Warning: No NSVS data points found")
                return

            cache.save(nsvs)

        nsvs = cache.data

    log(f"{len(nsvs)} original data points")

    idx = np.isfinite(nsvs['mjd']) & np.isfinite(nsvs['mag']) & np.isfinite(nsvs['magerr'])
    idx &= nsvs['magerr'] > 0

    # The range checks of Table 3, which catch SExtractor's error codes
    idx &= ((nsvs['mag'] > NSVS_MAG_RANGE[0]) & (nsvs['mag'] < NSVS_MAG_RANGE[1])
            & (nsvs['magerr'] < NSVS_MAGERR_MAX))

    log(f"{int(np.sum(idx))} data points after filtering")

    nsvs = nsvs[idx]
    nsvs.sort('mjd')

    if not len(nsvs):
        log("Warning: No valid NSVS data points")
        return

    # The survey says which of its own measurements it would call good, and
    # the flags carry what the errors do not: the points a magnitude off the
    # star quote a hundredth of a magnitude and are flagged HISCAT, their
    # macro-pixel being one the relative photometry could not settle.
    quality = quality_level(config, 'nsvs')
    dropping = {QUALITY_STANDARD: NSVS_BAD_FLAGS,
                QUALITY_RELAXED: NSVS_SATURATED,
                QUALITY_PUBLISHED: 0}[quality]

    flags = np.asarray(nsvs['flags'], dtype=int)
    bad = (flags & dropping) != 0

    if np.any(flags):
        log(f"  what the survey flagged, of these ({quality} filtering):")

        for value, name, meaning in NSVS_FLAGS:
            count = int(np.sum((flags & value) != 0))

            if count:
                log(f"    {name:10s} {count:5d}  {meaning}"
                    + ('' if value & dropping else '  [kept]'))

    if np.any(bad):
        log(f"Warning: dropping {int(np.sum(bad))} points"
            + (" the survey would not call good (Wozniak et al. 2004, Table 3)"
               if quality == QUALITY_STANDARD else " measured through saturation"))
        nsvs = nsvs[~bad]
        log(f"{len(nsvs)} data points left")

        if not len(nsvs):
            log("Warning: No NSVS measurements left at this level of filtering")
            return

    log_conversion(
        log, 'NSVS',
        'no conversion applied - the band is published as measured',
        {'colour term': ('none', 'unfiltered ROTSE-I, 450-1000nm, closest to R'),
         'magnitudes': 'stored as millimagnitudes, scaled here'},
        npoints=len(nsvs),
    )

    # Onto the common g scale, in two steps. The band is unfiltered and wide,
    # but its zero point is not: NSVS magnitudes are defined against V with a
    # colour term already built into them, so undoing that term is what puts
    # them on V, and the usual V to g conversion follows.
    B_minus_V, B_minus_V_origin = assumed_color(config, 'B_minus_V')
    g_minus_r, g_minus_r_origin = assumed_color(config, 'g_minus_r')

    nsvs['mag_V'] = rotse_to_v(np.asarray(nsvs['mag'], dtype=float), B_minus_V)
    nsvs['mag_g'] = v_to_g(nsvs['mag_V'], g_minus_r)

    log_conversion(
        log, 'NSVS',
        ROTSE_TO_V_FORMULA + ',  then  ' + V_TO_G_FORMULA,
        {'(B - V)': (B_minus_V, B_minus_V_origin),
         '(g - r)': (g_minus_r, g_minus_r_origin),
         'definition': ('m_ROTSE = V - (B - V)/1.875',
                        'Wozniak et al. 2004, how the survey set its zero point')},
        npoints=len(nsvs),
        note='both colours are assumed constant; the native magnitudes are kept',
    )

    log_bands(log, 'NSVS', [
        {'label': 'R', 'kind': 'native', 'npoints': len(nsvs),
         'note': 'unfiltered, as reported'},
        {'label': 'V (conv.)', 'kind': 'derived', 'npoints': len(nsvs),
         'note': "the survey's own colour term undone, putting it on V"},
        {'label': 'g (conv.)', 'kind': 'derived', 'npoints': len(nsvs),
         'note': 'on the common g scale, through V'},
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'nsvs_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        time = Time(np.asarray(nsvs['mjd'], dtype=float), format='mjd')
        ax.errorbar(time.datetime, nsvs['mag'], nsvs['magerr'],
                    fmt='.', alpha=0.5, color='#e377c2')

        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        ax.set_ylabel('R (unfiltered)')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - NSVS")

    log("NSVS lightcurve plot saved to file:nsvs_lc.png")

    nsvs.write(os.path.join(basepath, 'nsvs.vot'), format='votable', overwrite=True)
    nsvs.write(os.path.join(basepath, 'nsvs.txt'), format='ascii.commented_header', overwrite=True)
    log("NSVS data written to file:nsvs.vot")
    log("NSVS data written to file:nsvs.txt")
