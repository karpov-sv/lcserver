"""OGLE lightcurve acquisition module.

The Optical Gravitational Lensing Experiment has watched the Galactic bulge,
the southern Galactic disk and the Magellanic Clouds from Las Campanas since
1992, in I and, less often, V. Its photometry is not public position by
position: what OGLE releases is the OGLE Collection of Variable Stars, a set
of catalogues - one per class of variable and per region - each with a list of
its stars and a light curve file for every one of them. A star nobody has
classified as variable has no public OGLE light curve, however long OGLE has
watched it.

VizieR mirrors most of the catalogues but none of the light curves, and not
all of the catalogues either - the OGLE-IV long-period variables of the bulge
and the disk are missing from it - so everything here is read from OGLE's own
FTP site. The collections have no position search, only a star list each, so
those lists are gathered once into an index kept beside the target
directories, and the index is what a target is looked up in.
"""

import os
import re
import json
import time

from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np

from astropy.table import Table, vstack
from astropy.time import Time
from astropy.coordinates import SkyCoord

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    log_bands, log_conversion, plot_with_errors,
                    shared_cache_dir, assumed_color, v_to_g, V_TO_G_FORMULA)


OGLE_FTP = 'https://ftp.astrouw.edu.pl/ogle/'

# Where the collections are: OGLE-IV's, and the OGLE-III ones it has not
# superseded - long-period variables in the Clouds, the OGLE-III disk fields
OGLE_ROOTS = ['ogle4/OCVS/', 'ogle3/OIII-CVS/']

# How deep below a root a collection may sit - ogle4/OCVS/blg/cep/ is two
OGLE_MAX_DEPTH = 3

# The file listing a collection's stars. Almost always ident.dat; the BLAPs
# have their parameters and positions in one file instead. Two collections have
# neither and are not indexed: M54, whose light curves are named after the
# cluster's own variable numbers rather than OGLE identifiers, and the
# cataclysmic variables, published only as a tarball.
OGLE_INDEX_FILES = ('ident.dat', 'blap.dat')

# Directories that are never a collection and never hold one
OGLE_SKIP_DIRS = {'fcharts', 'spectra', 'model', 'fig', 'images'}

# The photometry of a collection is under phot/, or split by survey phase
# into phot_ogle2/, phot_ogle3/, phot_ogle4/ - or both, the BLAPs keeping
# phot_ogleN/ inside phot/. Below that a directory per band, again sometimes
# split by phase as I_o3/, I_o4/.
OGLE_PHOT_DIR = re.compile(r'^phot(?:_ogle[234])?$')
OGLE_BAND_DIR = re.compile(r'^(?P<band>[IV])(?P<phase>_o[234])?$')

# A position in a star list: sexagesimal, with colons or with spaces
OGLE_COORDS = re.compile(
    r'(?<![\d.])(\d{1,2})[: ](\d\d)[: ](\d\d\.\d+)\s+([+-])\s*(\d{1,2})[: ]'
    r'(\d\d)[: ](\d\d(?:\.\d+)?)')

OGLE_INDEX_CACHE = 'ogle_ocvs_index.npz'

# The collections grow - a new class of variable every year or so - so the
# index is built again once it is this old, as well as on a refresh
OGLE_INDEX_MAX_AGE = 90 * 86400

OGLE_TIMEOUT = 300
OGLE_WORKERS = 4

# Matching radius, in arcsec. OGLE positions are good to a fraction of an
# arcsecond; this is wide enough for a proper motion over twenty years and
# narrow enough that a bulge field does not hand back the neighbours.
OGLE_SR = 2.0

# One star can be in several collections at once - an eclipsing binary with a
# pulsating component, say - and in a crowded field more than one star can be
# within the radius. Past this many the rest are left.
OGLE_MAX_STARS = 5

# Times are HJD, written in full by some collections and as HJD - 2450000 by
# the rest
OGLE_HJD_OFFSET = 2450000.0
OGLE_HJD_TO_MJD = 2400000.5

# How far apart, in seconds, two copies of one frame may be dated. The
# collections do not all use one time scale - see _deduplicate - and the
# difference between them is TT - UTC, a little over a minute. Consecutive
# frames of one field are never closer than 105 seconds, and only copies from
# different files are compared in any case.
OGLE_DUPLICATE_WINDOW = 90.0

# Which phase of the survey a measurement came from, by its date. The files do
# not all say - a plain phot/I/ may hold OGLE-III and OGLE-IV together - and
# the phases do not overlap in time, so the date settles it.
OGLE_PHASES = [
    # (first HJD, name)
    (2450000.0, 'OGLE-II'),     # 1997
    (2452000.0, 'OGLE-III'),    # 2001
    (2455200.0, 'OGLE-IV'),     # 2010
]

OGLE_COLORS = {'I': '#8c564b', 'V': '#2ca02c'}


def _listing(url):
    """The entries of an FTP directory, as its HTML index lists them.

    Directories keep their trailing slash, so the two are told apart by name.
    The sorting links and the one back up the tree are dropped.
    """
    res = requests.get(url, timeout=OGLE_TIMEOUT)
    res.raise_for_status()

    return [h for h in re.findall(r'href="([^"?#]+)"', res.text)
            if not h.startswith(('/', 'http', '..', 'mailto:'))]


def _band_dirs(url, path):
    """Where a collection keeps its light curves, as (path, band) pairs.

    Where a band is split by phase and also given whole - I_o3/ and I_o4/
    beside I/ - the whole one is the others put together, and only the split
    ones are kept.
    """
    found, nested = [], []
    entries = _listing(url + path)

    # The OGLE-III disk dwarf novae keep their light curves in phot/ itself,
    # with no directory per band - their README calls them I-band, and I is
    # what OGLE observes in unless it says otherwise
    if (OGLE_PHOT_DIR.match(path.rstrip('/').rsplit('/', 1)[-1])
            and any(e.endswith('.dat') for e in entries)):
        found.append((path, 'I', False))

    for entry in entries:
        name = entry.rstrip('/')

        if not entry.endswith('/'):
            continue

        if OGLE_PHOT_DIR.match(name):
            nested += _band_dirs(url, path + entry)
            continue

        match = OGLE_BAND_DIR.match(name)

        if match:
            found.append((path + entry, match.group('band'),
                          bool(match.group('phase'))))

    # Judged within this directory only - phot/I/ and phot_ogle4/I/ are not
    # one another's halves
    split = {band for _, band, phased in found if phased}

    return [(p, band) for p, band, phased in found
            if phased or band not in split] + nested


def _collections(log):
    """Every collection on the FTP site, as the paths of their directories.

    Walked rather than listed here: a collection is a directory with a star
    list in it, and OGLE adds them without announcing it anywhere this could
    read.
    """
    found = []
    queue = [(root, 0) for root in OGLE_ROOTS]

    while queue:
        path, depth = queue.pop(0)

        try:
            entries = _listing(OGLE_FTP + path)
        except requests.RequestException as e:
            raise SourceError(f"could not list the OGLE FTP site at {path} - "
                              f"{type(e).__name__}: {e}")

        index = [f for f in OGLE_INDEX_FILES if f in entries]

        if index:
            found.append((path, index[0]))
            continue

        if depth >= OGLE_MAX_DEPTH:
            continue

        for entry in entries:
            name = entry.rstrip('/')

            if (entry.endswith('/') and name not in OGLE_SKIP_DIRS
                    and not OGLE_PHOT_DIR.match(name)):
                queue.append((path + entry, depth + 1))

    return found


def _collection_stars(path, index_file):
    """The stars of one collection and where its light curves are."""
    res = requests.get(OGLE_FTP + path + index_file, timeout=OGLE_TIMEOUT)
    res.raise_for_status()

    ids, ras, decs = [], [], []

    for line in res.text.splitlines():
        match = OGLE_COORDS.search(line)

        if not match:
            continue

        h, m, s, sign, d, dm, ds = match.groups()

        ids.append(line.split()[0])
        ras.append(15.0*(int(h) + int(m)/60 + float(s)/3600))
        decs.append((-1 if sign == '-' else 1)
                    * (int(d) + int(dm)/60 + float(ds)/3600))

    return ids, ras, decs, _band_dirs(OGLE_FTP, path)


def _build_index(path, log):
    """Every star of every collection, with its position, in one file."""
    log("Building the OGLE collection index - once for every target, not once "
        "per target. It means reading the star list of every collection, "
        "some 130 MB, and takes a few minutes.")

    collections = _collections(log)

    log(f"{len(collections)} collections on the OGLE FTP site")

    def fetch(item):
        coll_path, index_file = item

        try:
            return item, _collection_stars(coll_path, index_file)
        except requests.RequestException as e:
            raise SourceError(f"could not read the OGLE collection {coll_path}"
                              f" - {type(e).__name__}: {e}")

    ids, ras, decs, colls, meta = [], [], [], [], []

    with ThreadPoolExecutor(OGLE_WORKERS) as pool:
        for (coll_path, _), (c_ids, c_ras, c_decs, dirs) in pool.map(
                fetch, collections):
            if not c_ids or not dirs:
                log(f"  {coll_path}: skipped - "
                    + ("no positions in its star list" if not c_ids
                       else "no light curve directories"))
                continue

            colls += [len(meta)] * len(c_ids)
            meta.append({'path': coll_path, 'dirs': dirs})
            ids += c_ids
            ras += c_ras
            decs += c_decs

    if not ids:
        raise SourceError("the OGLE FTP site listed no stars at all")

    log(f"{len(ids)} stars indexed")

    # Written aside and renamed, so that an interrupted build cannot leave a
    # half-file that every later run then reads - and aside under a name of
    # its own, as two targets acquired at once may both be building it
    part = f"{path}.{os.getpid()}.part.npz"

    np.savez_compressed(
        part,
        id=np.array(ids, dtype='S'),
        ra=np.array(ras, dtype=float),
        dec=np.array(decs, dtype=float),
        collection=np.array(colls, dtype=np.int16),
        collections=np.array(json.dumps(meta)),
    )

    os.replace(part, path)


def _index(basepath, log, refresh=False):
    """The collection index, built if it is missing, stale or to be refreshed."""
    path = os.path.join(shared_cache_dir(basepath), OGLE_INDEX_CACHE)

    stale = (os.path.exists(path)
             and time.time() - os.path.getmtime(path) > OGLE_INDEX_MAX_AGE)

    if refresh or stale or not os.path.exists(path):
        if stale:
            log("The OGLE collection index is over "
                f"{OGLE_INDEX_MAX_AGE // 86400} days old")

        os.makedirs(shared_cache_dir(basepath), exist_ok=True)
        _build_index(path, log)

    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _find(index, ra, dec, sr):
    """The stars of the index within sr arcsec, nearest first."""
    # Cut down by declination before anything is computed on a million rows
    near = np.abs(index['dec'] - dec) < sr/3600.0
    rows = np.flatnonzero(near)

    if not len(rows):
        return []

    dist = SkyCoord(ra, dec, unit='deg').separation(
        SkyCoord(index['ra'][rows], index['dec'][rows], unit='deg')).arcsec

    rows, dist = rows[dist <= sr], dist[dist <= sr]
    collections = json.loads(str(index['collections']))

    return [{'id': index['id'][i].decode(),
             'distance': float(d),
             **collections[int(index['collection'][i])]}
            for i, d in sorted(zip(rows, dist), key=lambda _: _[1])]


def _lightcurve(url):
    """One light curve file: HJD, magnitude and error, whatever else follows.

    None where the star has no file there - a collection lists every band
    directory it has, and not every star is in each of them.
    """
    res = requests.get(url, timeout=OGLE_TIMEOUT)

    if res.status_code == 404:
        return None

    res.raise_for_status()

    rows = []

    for line in res.text.splitlines():
        parts = line.split()

        try:
            rows.append([float(_) for _ in parts[:3]])
        except (ValueError, IndexError):
            continue

    rows = [r for r in rows if len(r) == 3]

    if not rows:
        return None

    hjd, mag, err = np.array(rows).T
    hjd = np.where(hjd < OGLE_HJD_OFFSET, hjd + OGLE_HJD_OFFSET, hjd)

    return hjd, mag, err


def _phase(hjd):
    """Which phase of the survey each time belongs to."""
    phase = np.full(len(hjd), 'OGLE-I', dtype='U8')

    for start, name in OGLE_PHASES:
        phase[hjd >= start] = name

    return phase


def _star(star, log):
    """Every light curve of one star, in every band and phase its collection
    has, as one table."""
    parts = []

    # Each directory is a path from the top of the FTP site, collection and all
    for path, band in star['dirs']:
        url = OGLE_FTP + path + star['id'] + '.dat'

        try:
            lc = _lightcurve(url)
        except requests.RequestException as e:
            log(f"  {star['id']}: could not fetch {path}{star['id']}.dat - "
                f"{type(e).__name__}: {e}")
            continue

        if lc is None:
            continue

        hjd, mag, err = lc

        parts.append(Table({
            'mjd': hjd - OGLE_HJD_TO_MJD,
            'mag': mag,
            'magerr': err,
            'filter': np.full(len(hjd), band),
            'phase': _phase(hjd),
            'id': np.full(len(hjd), star['id']),
        }))

    return vstack(parts) if parts else None


def _deduplicate(table):
    """The table without the frames it holds twice.

    The same frames reach it by more than one route - a star in two
    collections under two identifiers, or an OGLE-III collection and its
    OGLE-IV successor both carrying the OGLE-III data. Neither the numbers nor
    the times need agree between the copies. The OGLE-IV long-period
    variables carry their OGLE-III points recalibrated, 0.14 mag away from
    what the OGLE-III collection says of the same frames, and other
    collections rescale the errors; and most collections date a frame in HJD
    where a few - the BLAPs - use BJD_TDB, which is a minute later.

    So a copy is a point in the same band from another file within
    OGLE_DUPLICATE_WINDOW, and of the copies the one from the earliest file is
    kept - the files being in order of preference, see _preference.
    """
    table = table[np.argsort(np.asarray(table['mjd']), kind='stable')]

    bands = np.asarray(table['filter'], dtype=str)
    files = np.asarray(table['file'], dtype=int)
    mjd = np.asarray(table['mjd'], dtype=float)

    keep = np.ones(len(table), dtype=bool)
    window = OGLE_DUPLICATE_WINDOW / 86400.0

    for i in range(len(table)):
        j = i - 1

        while j >= 0 and mjd[i] - mjd[j] <= window:
            if keep[j] and bands[j] == bands[i] and files[j] != files[i]:
                if files[i] < files[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

            j -= 1

    return table[keep]


def _preference(star):
    """Which collection's copy of a frame to keep: OGLE-IV's before OGLE-III's.

    The newer collections put the older photometry on the calibration of the
    rest of their light curve, which is the one the OGLE-IV points are on.
    Nearest first within that, as the likelier to be the star asked for.
    """
    root = next((i for i, r in enumerate(OGLE_ROOTS)
                 if star['path'].startswith(r)), len(OGLE_ROOTS))

    return root, star['distance']


@survey_source(
    name='OGLE Collection of Variable Stars',
    short_name='OGLE',
    state_acquiring='acquiring OGLE lightcurve',
    state_acquired='OGLE lightcurve acquired',
    log_file='ogle.log',
    output_files=['ogle.log', 'ogle_lc.png', 'ogle.vot', 'ogle.txt'],
    button_text='Get OGLE lightcurve',
    form_fields={
        'ogle_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': OGLE_SR,
            'required': False,
        },
    },
    help_text='OGLE I and V photometry of the variables it has classified, '
              'bulge, southern disk and Magellanic Clouds, 1997 onwards',
    order=28,
    about=(
        "The Optical Gravitational Lensing Experiment, watching the Galactic "
        "bulge, the southern Galactic disk and the Magellanic Clouds from Las "
        "Campanas in I and V. Only the stars in its Collection of Variable "
        "Stars have public light curves - over a million of them, but a "
        "star OGLE has not classified has none."),
    about_links=[
        ('OGLE', 'https://ogle.astrouw.edu.pl/'),
        ('OGLE Collection of Variable Stars',
         'https://ogledb.astrouw.edu.pl/~ogle/OCVS/'),
    ],
    acknowledgement=(
        "OGLE asks that work using the Collection of Variable Stars cite the "
        "paper describing each collection used, listed in the README of its "
        "directory at ftp.astrouw.edu.pl/ogle."),
    # Lightcurve metadata
    votable_file='ogle.vot',
    lc_bands=[
        surveys.band('I', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='I',
                     color=OGLE_COLORS['I'],
                     note='Cousins I, as published by OGLE'),
        surveys.band('V', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='V',
                     color=OGLE_COLORS['V'],
                     note='Johnson V, as published by OGLE'),
        surveys.band('g (conv.)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     filter_column='filter', filter_value='V',
                     color='#98df8a',
                     note='V put on the common g scale using an assumed g - r',
                     combined=True),
    ],
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color=OGLE_COLORS['I'],
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata
    template_layout='simple',
    requires_coordinates=True,
)
def target_ogle(config, basepath=None, verbose=True, show=False):
    """Acquire OGLE lightcurve."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('ogle'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = float(config.get('ogle_sr') or OGLE_SR)

    cache_name = f"ogle_{ra:.4f}_{dec:.4f}_{sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'OGLE',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            log(f"within {sr:.1f} arcsec")

            index = _index(basepath, log, refresh=refresh_cache)
            stars = _find(index, ra, dec, sr)

            if not stars:
                cache.save_empty()
                log("Warning: No star of the OGLE Collection of Variable Stars "
                    "at this position. OGLE may well have observed it - only "
                    "the stars it has classified as variable have public "
                    "light curves.")
                return

            for star in stars:
                log(f"  {star['id']} in {star['path']}, "
                    f"{star['distance']:.2f} arcsec away")

            if len(stars) > OGLE_MAX_STARS:
                log(f"{len(stars)} stars, of which the nearest "
                    f"{OGLE_MAX_STARS} are fetched")
                stars = stars[:OGLE_MAX_STARS]

            parts = []

            for star in sorted(stars, key=_preference):
                lc = _star(star, log)

                if lc is None:
                    log(f"  {star['id']}: no light curve files")
                    continue

                log(f"  {star['id']}: " + ', '.join(
                    f"{np.sum(lc['filter'] == b)} in {b}"
                    for b in ('I', 'V') if np.any(lc['filter'] == b)))

                lc['file'] = len(parts)
                parts.append(lc)

            if not parts:
                cache.save_empty()
                log("Warning: The OGLE collections list a star here, but none "
                    "of its light curves could be found")
                return

            ogle = _deduplicate(vstack(parts))
            ogle.remove_column('file')

            if len(ogle) < sum(len(_) for _ in parts):
                log(f"  {sum(len(_) for _ in parts) - len(ogle)} points are "
                    "the same frames reaching here from more than one "
                    "collection, and are counted once")

            cache.save(ogle)

        ogle = cache.data

    # Nothing here, and cached as nothing - the helper has said so already
    if ogle is None:
        return

    log(f"\n{len(ogle)} data points")

    # Filter out bad data
    good = np.isfinite(ogle['mag']) & np.isfinite(ogle['magerr'])
    good &= (ogle['magerr'] > 0) & (ogle['magerr'] < 1.0) & (ogle['mag'] < 30)
    ogle = ogle[good]

    log(f"{len(ogle)} data points after filtering")

    if not len(ogle):
        log("Warning: No valid OGLE data points after filtering")
        return

    ogle.sort('mjd')

    bands = np.asarray(ogle['filter'], dtype=str)
    phases = np.asarray(ogle['phase'], dtype=str)

    for phase in ('OGLE-I', 'OGLE-II', 'OGLE-III', 'OGLE-IV'):
        idx = phases == phase

        if np.any(idx):
            log(f"  {phase}: {np.sum(idx)} points, "
                f"{Time(ogle['mjd'][idx].min(), format='mjd').datetime.year}-"
                f"{Time(ogle['mjd'][idx].max(), format='mjd').datetime.year}")

    # I is left as it is - there is no route from it to g without a colour
    # this code does not have - and V is carried onto the common g scale, as
    # every other source measuring V does
    g_minus_r, g_minus_r_origin = assumed_color(config, 'g_minus_r')

    idx_I = bands == 'I'
    idx_V = bands == 'V'

    ogle['mag_g'] = np.nan
    ogle['mag_g'][idx_V] = v_to_g(ogle['mag'][idx_V], g_minus_r)

    log_conversion(
        log, 'OGLE',
        'no conversion applied - each band is published as measured',
        {'colour term': ('none', 'OGLE calibrates onto Cousins I and Johnson V')},
        npoints=len(ogle),
    )

    if np.any(idx_V):
        log_conversion(
            log, 'OGLE',
            V_TO_G_FORMULA,
            {'(g - r)': (g_minus_r, g_minus_r_origin)},
            npoints=int(np.sum(idx_V)),
            note='only V reaches the combined light curve; I, where most of '
                 'the points are, has no route to g without a colour this '
                 'code does not have',
        )

    log_bands(log, 'OGLE', [
        {'label': 'I', 'kind': 'native', 'npoints': int(np.sum(idx_I)),
         'note': 'Cousins I, as published by OGLE'},
        {'label': 'V', 'kind': 'native', 'npoints': int(np.sum(idx_V)),
         'note': 'Johnson V, as published by OGLE'},
        {'label': 'g (conv.)', 'kind': 'derived', 'npoints': int(np.sum(idx_V)),
         'note': 'V on the common g scale'},
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'ogle_lc.png'),
                            figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        time_ = Time(np.asarray(ogle['mjd'], dtype=float), format='mjd').datetime

        for band, idx in (('I', idx_I), ('V', idx_V)):
            if np.any(idx):
                plot_with_errors(ax, time_[idx], ogle['mag'][idx],
                                 ogle['magerr'][idx], label=band,
                                 color=OGLE_COLORS[band])

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        if np.any(idx_I) and np.any(idx_V):
            ax.legend()

        ax.set_ylabel('Magnitude')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - OGLE")

    log("OGLE lightcurve plot saved to file:ogle_lc.png")

    ogle.write(os.path.join(basepath, 'ogle.vot'), format='votable', overwrite=True)
    ogle.write(os.path.join(basepath, 'ogle.txt'),
               format='ascii.commented_header', overwrite=True)
    log("OGLE data written to file:ogle.vot")
    log("OGLE data written to file:ogle.txt")
