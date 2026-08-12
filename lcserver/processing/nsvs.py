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
from .utils import cleanup_paths, cached_votable_query, log_bands, log_conversion


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
        }
    },
    help_text='Northern Sky Variability Survey, unfiltered, 1999-2000',
    order=25,
    # Lightcurve metadata
    votable_file='nsvs.vot',
    lc_bands=[
        surveys.band('R', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='R', color='#e377c2',
                     note='unfiltered ROTSE-I, closest to R'),
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
                    log("Error: the object was not found in the field file")
                    return

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
            except:
                import traceback
                traceback.print_exc()
                log("Error: could not download the data")
                return

            if not len(nsvs):
                log("Warning: No NSVS data points found")
                return

            cache.save(nsvs)

        nsvs = cache.data

    log(f"{len(nsvs)} original data points")

    idx = np.isfinite(nsvs['mjd']) & np.isfinite(nsvs['mag']) & np.isfinite(nsvs['magerr'])
    idx &= nsvs['magerr'] > 0

    log(f"{int(np.sum(idx))} data points after filtering")

    nsvs = nsvs[idx]
    nsvs.sort('mjd')

    if not len(nsvs):
        log("Warning: No valid NSVS data points")
        return

    # The survey flags its measurements, but what the bits mean is not in the
    # documentation CDS carries, and no combination of them reproduces the
    # count of good points the catalogue quotes. So they are kept and reported
    # rather than acted upon.
    flags = np.asarray(nsvs['flags'], dtype=int)
    log(f"  measurement flags: {int(np.sum(flags == 0))} of {len(flags)} unflagged, "
        f"{len(set(flags[flags != 0].tolist()))} distinct non-zero values kept as they are")

    log_conversion(
        log, 'NSVS',
        'no conversion applied - the band is published as measured',
        {'colour term': ('none', 'unfiltered ROTSE-I, 450-1000nm, closest to R'),
         'magnitudes': 'stored as millimagnitudes, scaled here'},
        npoints=len(nsvs),
    )

    log_bands(log, 'NSVS', [
        {'label': 'R', 'kind': 'native', 'npoints': len(nsvs),
         'note': 'unfiltered, as reported'},
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
