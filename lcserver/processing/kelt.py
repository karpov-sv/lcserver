"""KELT lightcurve acquisition module.

The Kilodegree Extremely Little Telescope was two 42mm lenses - one in
Arizona, one in South Africa - watching whole tens of degrees of sky at once
from 2006 to 2019. What that buys is the bright end: a pixel is twenty-three
arcseconds across, so nothing faint is resolved, but stars of eighth to
eleventh magnitude, which ZTF and ATLAS saturate on and which the deep surveys
here never measure, are followed for a decade.

The catalogue of what it observed is at the NASA Exoplanet Archive and is
queried by position. The light curves themselves sit in a directory tree whose
paths are a fill order rather than anything derivable from an identifier - the
only public statement of which file is where is the archive's own bulk
download script. So that script is fetched once per installation, kept beside
the target directories rather than inside one of them, and read to find the
few files a target needs.
"""

import os
import io
import re
import tarfile

import requests
import numpy as np

from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    clip_noisy_points, quality_field, quality_level,
                    log_bands, log_conversion, plot_with_errors,
                    shared_cache_dir, CLIP_RATIO_BY_LEVEL,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


# What the survey observed, one row per source per field, orientation and
# processing. Metadata only - it carries no path to the data.
KELT_TAP = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync'
KELT_TABLE = 'kelttimeseries'

# The bulk download script, which is the index of where every light curve is.
# 68 megabytes compressed, holding one shell script per survey field.
KELT_INDEX_URL = ('https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/'
                  'KELT_wget.tar.gz')
KELT_INDEX_CACHE = 'kelt_wget.tar.gz'
KELT_INDEX_MEMBER = 'KELT_{field}_wget.bat'

KELT_TIMEOUT = 600

# Matching radius, in arcsec. Generous because the pixels are: at twenty-three
# arcseconds each, a KELT source is a good deal less well placed than anything
# else here, and its photometry is blended with whatever else is in the pixel.
KELT_SR = 12.0

# A star in several overlapping fields, each seen east and west of the
# meridian, is a curve apiece
KELT_MAX_CURVES = 6

# What to ask the archive for
KELT_COLUMNS = ['kelt_sourceid', 'kelt_field', 'kelt_orientation', 'proc_type',
                'ra', 'dec', 'kelt_mag', 'npts', 'bjdstart', 'bjdstop',
                'median', 'stddevwrtmedian']

# A wget line of the index, from which only the URL is wanted
KELT_LINE = re.compile(rb"wget -O '(?P<name>[^']+)' '(?P<url>[^']+)'")

# KELT-North observed each field twice a night, east and west of the meridian,
# and the archive lists both as separate rows - but every row of its metadata
# table names the east file, whatever side the row is for. There are 4.1
# million rows and not one identifier ending in 'west'.
#
# The west files do exist, under the same field and a source number of their
# own, and that numbering is independent of the east one: the west file with
# our target's east number is a different star by two and a half magnitudes,
# and so are the next two checked. Nothing published maps a west row to its
# file - not the metadata table, which has no separate identifier for it, and
# not the download index, which has no positions in it.
#
# So only the east curves are fetched. That is about three fifths of what the
# survey measured; the alternative would be to attach a light curve of one
# star to another, which is worse than not having it.
KELT_ORIENTATION = 'east'

# The archive dates its cadences in barycentric Julian days
KELT_BJD_TO_MJD = 2400000.5


def _query(ra, dec, sr, proc_type):
    """What KELT observed at a position, as the archive lists it.

    Asked for as a box in right ascension and declination and cut to a circle
    afterwards, rather than as a cone: this table carries no spatial index the
    service will accept a CONTAINS against, and the geometric form comes back
    as an error with no message in it. The box is widened in right ascension
    by the declination, a degree of it being shorter the further from the
    equator.
    """
    half = sr / 3600.0
    half_ra = half / max(np.cos(np.deg2rad(dec)), 1e-6)

    query = (f"SELECT {', '.join(KELT_COLUMNS)} FROM {KELT_TABLE}"
             f" WHERE proc_type = '{proc_type}'"
             f" AND kelt_orientation = '{KELT_ORIENTATION}'"
             f" AND ra BETWEEN {ra - half_ra:.7f} AND {ra + half_ra:.7f}"
             f" AND dec BETWEEN {dec - half:.7f} AND {dec + half:.7f}")

    try:
        res = requests.get(KELT_TAP, timeout=KELT_TIMEOUT, params={
            'request': 'doQuery', 'lang': 'ADQL', 'format': 'csv',
            'query': query})
        res.raise_for_status()
    except requests.RequestException as e:
        raise SourceError("could not query the Exoplanet Archive - "
                          f"{type(e).__name__}: {e}")

    # A refused query comes back as text rather than as a table of rows
    if res.text.lstrip().startswith('<') or 'ERROR' in res.text[:200].upper():
        raise SourceError("the Exoplanet Archive refused the query: "
                          + res.text[:200])

    try:
        table = Table.read(res.text.splitlines(), format='csv')
    except Exception as e:
        raise SourceError("could not read the Exoplanet Archive answer - "
                          f"{type(e).__name__}: {e}")

    if not len(table):
        return None

    # The box cut down to the circle that was asked for
    separation = SkyCoord(ra, dec, unit='deg').separation(
        SkyCoord(np.asarray(table['ra'], dtype=float),
                 np.asarray(table['dec'], dtype=float), unit='deg'))

    table = table[separation.arcsec <= sr]

    return table if len(table) else None


def _index(basepath, log, refresh=False):
    """The bulk download script, fetched once and kept for every target.

    It describes the survey rather than any position, so it goes in the shared
    cache beside the target directories rather than into whichever target
    happened to want it first - it is 68 megabytes, and a copy per target
    would be absurd.
    """
    path = os.path.join(shared_cache_dir(basepath), KELT_INDEX_CACHE)

    if os.path.exists(path) and not refresh:
        return path

    os.makedirs(shared_cache_dir(basepath), exist_ok=True)

    log(f"Fetching the KELT file index ({KELT_INDEX_CACHE}, 68 MB) - once for "
        "every target, not once per target")

    try:
        res = requests.get(KELT_INDEX_URL, timeout=KELT_TIMEOUT)
        res.raise_for_status()
    except requests.RequestException as e:
        raise SourceError("could not fetch the KELT file index - "
                          f"{type(e).__name__}: {e}")

    # Written aside and renamed, so that an interrupted fetch cannot leave a
    # half-file that every later run then reads
    with open(path + '.part', 'wb') as f:
        f.write(res.content)

    os.replace(path + '.part', path)

    return path


def _urls(index_path, field, wanted, log):
    """Where the light curves of one field's wanted sources are.

    The script is one line per file, and a field's may run to a hundred
    megabytes, so it is read as a stream and only the lines naming a source
    that was asked for are kept.
    """
    member = KELT_INDEX_MEMBER.format(field=field)
    found = {}

    try:
        with tarfile.open(index_path, 'r:gz') as tar:
            try:
                stream = tar.extractfile(member)
            except KeyError:
                stream = None

            if stream is None:
                log(f"  the index has no script for field {field}")
                return found

            targets = {name.encode(): name for name in wanted}

            for line in stream:
                for key, name in targets.items():
                    if key in line:
                        match = KELT_LINE.match(line)

                        if match:
                            found[name] = match.group('url').decode()
                        break

                if len(found) == len(targets):
                    break
    except (tarfile.TarError, OSError) as e:
        raise SourceError("could not read the KELT file index - "
                          f"{type(e).__name__}: {e}")

    return found


def _filename(source, proc):
    """The file a row of the metadata table stands for.

    The identifier already carries the orientation - always east, see above -
    so only the processing has to be added, which is not in it at all.
    """
    return f"{source}_{proc}_lc.tbl"


def _lightcurve(url):
    """One light curve, as the archive stores it: an IPAC table of BJD,
    magnitude and error."""
    try:
        res = requests.get(url, timeout=KELT_TIMEOUT)
        res.raise_for_status()
    except requests.RequestException as e:
        raise SourceError(f"could not fetch {os.path.basename(url)} - "
                          f"{type(e).__name__}: {e}")

    try:
        # A list of lines rather than a stream: the IPAC reader wants
        # something it can iterate, and declines a file-like object of text
        table = Table.read(res.text.splitlines(), format='ipac')
    except Exception as e:
        raise SourceError("could not read the light curve - "
                          f"{type(e).__name__}: {e}")

    if not len(table) or 'TIME' not in table.colnames:
        return None

    return Table({
        'mjd': np.asarray(table['TIME'], dtype=float) - KELT_BJD_TO_MJD,
        'mag': np.asarray(table['MAG'], dtype=float),
        'magerr': np.asarray(table['MAG_ERR'], dtype=float),
    })


@survey_source(
    name='KELT',
    short_name='KELT',
    state_acquiring='acquiring KELT lightcurve',
    state_acquired='KELT lightcurve acquired',
    log_file='kelt.log',
    output_files=['kelt.log', 'kelt_lc.png', 'kelt.vot', 'kelt.txt'],
    button_text='Get KELT lightcurve',
    form_fields={
        'kelt_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': KELT_SR,
            'required': False,
        },
        'kelt_proc': {
            'type': 'choice',
            'label': 'Processing',
            'choices': [('raw', 'Raw - the measurements as made'),
                        ('tfa', 'TFA - trends common to the field removed')],
            'initial': 'raw',
            'required': False,
        },
        'kelt_quality': quality_field({
            QUALITY_STANDARD: 'Drop the frames a field measured worst',
            QUALITY_RELAXED: 'Drop only the very worst frames',
            QUALITY_PUBLISHED: 'None - every measurement as published',
        }),
    },
    help_text='Kilodegree Extremely Little Telescope, broad R, 2006-2019, '
              'bright stars',
    order=27,
    # Lightcurve metadata
    votable_file='kelt.vot',
    lc_bands=[
        surveys.band('R', 'mag', 'magerr', surveys.BAND_NATIVE,
                     color='#c0392b',
                     note='the broad KELT band, calibrated onto Johnson R'),
    ],
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_color='#c0392b',
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata
    template_layout='simple',
)
def target_kelt(config, basepath=None, verbose=True, show=False):
    """Acquire KELT lightcurve."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('kelt'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = float(config.get('kelt_sr', KELT_SR))
    proc = config.get('kelt_proc', 'raw')

    if proc not in ('raw', 'tfa'):
        proc = 'raw'

    cache_name = f"kelt_{ra:.4f}_{dec:.4f}_{sr:.1f}_{proc}.vot"

    with cached_votable_query(cache_name, basepath, log, 'KELT',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            log(f"within {sr:.1f} arcsec, {proc} photometry")

            found = _query(ra, dec, sr, proc)

            if found is None:
                cache.save_empty()
                log("Warning: No KELT lightcurve at this position - the survey "
                    "covers most of the sky but measures only what is bright "
                    "enough to stand out in a 23 arcsec pixel")
                return

            log(f"{len(found)} KELT lightcurve(s) found, east of the meridian:")

            for row in found:
                log(f"  {row['kelt_sourceid']}"
                    f"  {row['npts']} points  KELT mag {row['kelt_mag']:.2f}")

            log("The survey also measured this field west of the meridian, and "
                "the archive lists those rows - but it names the east file for "
                "them too, and nothing published says which west file they are. "
                "See the note in the source.")

            # Brightest first, being the best measured of a blended pixel
            found.sort('kelt_mag')

            if len(found) > KELT_MAX_CURVES:
                log(f"{len(found)} curves, of which the brightest "
                    f"{KELT_MAX_CURVES} are fetched")
                found = found[:KELT_MAX_CURVES]

            index_path = _index(basepath, log, refresh=refresh_cache)

            parts = []

            # A field at a time, each being one member of the index to read
            for field in sorted(set(str(_) for _ in found['kelt_field'])):
                rows = found[np.asarray(found['kelt_field'], dtype=str) == field]
                wanted = [_filename(row['kelt_sourceid'], proc) for row in rows]

                urls = _urls(index_path, field, wanted, log)

                for row in rows:
                    source = str(row['kelt_sourceid'])
                    name = _filename(source, proc)
                    url = urls.get(name)

                    if not url:
                        log(f"  {name}: not in the index for field {field}")
                        continue

                    try:
                        lc = _lightcurve(url)
                    except SourceError as e:
                        log(f"  {source}: {e}")
                        continue

                    if lc is None or not len(lc):
                        continue

                    lc['field'] = field
                    lc['source'] = source

                    parts.append(lc)

                    log(f"  {name}: {len(lc)} points")

            if not parts:
                cache.save_empty()
                log("Warning: None of the KELT lightcurves here could be fetched")
                return

            from astropy.table import vstack

            cache.save(vstack(parts))

        kelt = cache.data

    # Nothing here, and cached as nothing - the helper has said so already
    if kelt is None:
        return

    log(f"\n{len(kelt)} data points")

    # Filter out bad data
    kelt = kelt[np.isfinite(kelt['mag'])]
    kelt = kelt[np.isfinite(kelt['magerr'])]
    kelt = kelt[kelt['magerr'] > 0]
    kelt = kelt[kelt['magerr'] < 1.0]

    log(f"{len(kelt)} data points after filtering")

    if not len(kelt):
        log("Warning: No valid KELT data points after filtering")
        return

    # Judged within each field and orientation at once: the two sides of the
    # meridian are different optical paths and are not calibrated together
    quality = quality_level(config, 'kelt')
    groups = np.asarray(kelt['field'], dtype=str)
    clip = (clip_noisy_points(kelt['mag'], kelt['magerr'], groups,
                              log=log, group_name='field',
                              ratio=CLIP_RATIO_BY_LEVEL[quality])
            if quality != QUALITY_PUBLISHED
            else np.zeros(len(kelt), dtype=bool))

    if np.any(clip):
        kelt = kelt[~clip]
        log(f"{len(kelt)} data points left")

        if not len(kelt):
            log("Warning: No KELT measurements left at this level of filtering")
            return

    kelt.sort('mjd')

    # Published as measured. The band is a Kodak Wratten 8 - everything redder
    # than 500nm - calibrated onto Johnson R, and its colour term runs from
    # -0.3 magnitudes for a very blue star to +0.8 for a very red one. Putting
    # that on the common g scale would need a relation nobody has published
    # for this filter, so the measurement is left as the measurement.
    log_conversion(
        log, 'KELT',
        'no conversion applied - the band is published as measured',
        {'colour term': ('none', 'broad Wratten 8, calibrated onto Johnson R'),
         'the survey quotes': ('-0.3 to +0.8 mag',
                               'its own colour term, blue star to red')},
        npoints=len(kelt),
        note='not put on the common g scale: no relation for this filter has '
             'been published, and one invented here would be worse than none',
    )

    log_bands(log, 'KELT', [
        {'label': 'R', 'kind': 'native', 'npoints': len(kelt),
         'note': 'the broad KELT band, as reported'},
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'kelt_lc.png'),
                            figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        time = Time(np.asarray(kelt['mjd'], dtype=float), format='mjd').datetime
        labels = np.asarray(kelt['field'], dtype=str)

        for label in sorted(set(labels.tolist())):
            idx = labels == label

            if np.any(idx):
                plot_with_errors(ax, time[idx], kelt['mag'][idx],
                                 kelt['magerr'][idx], label=label)

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        if len(set(labels.tolist())) > 1:
            ax.legend()

        ax.set_ylabel('KELT R')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - KELT")

    log("KELT lightcurve plot saved to file:kelt_lc.png")

    kelt.write(os.path.join(basepath, 'kelt.vot'), format='votable', overwrite=True)
    kelt.write(os.path.join(basepath, 'kelt.txt'),
               format='ascii.commented_header', overwrite=True)
    log("KELT data written to file:kelt.vot")
    log("KELT data written to file:kelt.txt")
