"""Common utilities for processing astronomical data."""

import os
import glob
import shutil
import re
import requests
import numpy as np
import dill as pickle
from io import BytesIO
from contextlib import contextmanager

from astropy.io.votable import parse as votable_parse
from astropy.table import Table


# How hard a source is asked to judge its own measurements. The names are
# shared, so that one choice can govern a whole run; what each means is the
# source's own affair, and a source offers only the levels it can tell apart.
# There is no fourth, stricter level: where one was tried it either changed
# nothing or began removing the variability it was meant to reveal.
QUALITY_STANDARD = 'standard'    # what the survey, or this code, calls good
QUALITY_RELAXED = 'relaxed'      # only what is indefensible
QUALITY_PUBLISHED = 'published'  # nothing judged, the data as they arrive

QUALITY_LEVELS = [QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED]

# How far above what its own kind achieved at the same brightness a point may
# sit before it is dropped: a single ruined frame, rather than a survey that
# is simply less precise than another
CLIP_MAX_ERR_RATIO = 5.0

# The same, per level. Between three and ten the light curve hardly changes -
# the junk has gone by ten and the rest is shaving - so these are the two
# worth offering, and the third level clips nothing at all.
CLIP_RATIO_BY_LEVEL = {QUALITY_STANDARD: CLIP_MAX_ERR_RATIO,
                       QUALITY_RELAXED: 10.0}

# Brightness bins the comparison is made in, and the fewest points a group
# needs before there is anything to bin
CLIP_BINS = 5
CLIP_NMIN = 20


def quality_field(labels, label='Quality filtering'):
    """A source's filtering selector, as the registry wants a form field.

    The words are the source's own, since what it can distinguish differs -
    NSVS has the survey's own definition to appeal to, where another source
    has only the scatter of its own measurements.
    """
    return {
        'type': 'choice',
        'label': label,
        'choices': [(level, labels[level])
                    for level in QUALITY_LEVELS if level in labels],
        'initial': QUALITY_STANDARD,
        'required': False,
    }


def quality_level(config, source_id):
    """How hard this source was asked to filter.

    What was asked of it, or of the run as a whole, or the standard.
    """
    level = (config.get(f'{source_id}_quality')
             or config.get('quality')
             or QUALITY_STANDARD)

    return level if level in QUALITY_LEVELS else QUALITY_STANDARD


def typical_error(mag, err, nbins=CLIP_BINS):
    """What this photometry's error usually is at each of these brightnesses.

    A point cannot be judged against the median error of the set it belongs
    to, as photometry grows less certain the fainter the star is: on a
    variable that would condemn every measurement of its faint state and leave
    a light curve which only ever shows the star bright. A Mira loses a
    quarter of its minima that way. The comparison is with what the same
    camera, or band, or night achieved at the same brightness instead, taken
    as the median error within bins holding equal numbers of points, and held
    flat beyond the ends.
    """
    edges = np.quantile(mag, np.linspace(0, 1, nbins + 1))
    brightness, typical = [], []

    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (mag >= lo) & (mag <= hi)

        # A bin of one or two says more about those points than about the
        # instrument, and the bins are quantiles, so a repeated magnitude can
        # empty one altogether
        if np.sum(idx) >= 3:
            brightness.append(np.median(mag[idx]))
            typical.append(np.median(err[idx]))

    if len(brightness) < 2:
        return np.full(len(mag), np.median(err))

    return np.interp(mag, brightness, typical)


def mission_quality_mask(quality, mission='TESS', level=QUALITY_STANDARD):
    """Which cadences a mission's own quality flags allow, at this level.

    The space missions flag a cadence for anything from a safe-mode entry to
    stray light off the Earth, and the two are not comparable: the first
    leaves nothing to measure, while the second is largely taken out by the
    pipeline's background model. So 'standard' keeps only what carries no flag
    at all, which is what this code did before there was a choice, and
    'relaxed' defers to the mission's own reading - lightkurve's default
    bitmask, which passes stray light and stops at what it calls unusable.

    Which of them applies depends on the pipeline, and only one of the three
    offered here is affected at all. SPOC and TESS-SPOC write NaN into the
    flux of every cadence they flag, so their flagged cadences carry nothing
    to keep or drop and this changes nothing for them. QLP leaves the flux in
    place - all 273 flagged cadences of the sector this was checked on - so
    for QLP light curves the choice decides whether they are plotted.

    Returns a boolean array, True where the cadence is to be kept.
    """
    quality = np.asarray(quality)

    if level == QUALITY_PUBLISHED:
        return np.ones(len(quality), dtype=bool)

    if level != QUALITY_RELAXED:
        return quality == 0

    try:
        from lightkurve.utils import TessQualityFlags, KeplerQualityFlags
    except ImportError:
        return quality == 0

    flags = (TessQualityFlags if str(mission).upper().startswith('TESS')
             else KeplerQualityFlags)

    return flags.create_quality_mask(quality, bitmask='default')


def log_quality(log, quality, dropped, mission='TESS'):
    """Say which of the mission's flags cost this segment anything.

    Told what was actually lost rather than what was flagged: a pipeline that
    writes NaN into the flux of the cadences it flags has already dropped
    them, and saying so again would credit this code with work the mission
    did. Only the flags behind a real loss are named.
    """
    quality = np.asarray(quality)
    dropped = np.asarray(dropped)

    if not np.any(dropped):
        return

    named = ''

    try:
        from lightkurve.utils import TessQualityFlags, KeplerQualityFlags
        flags = (TessQualityFlags if str(mission).upper().startswith('TESS')
                 else KeplerQualityFlags)

        told = []
        for n in range(32):
            bit = 1 << n
            here = ((quality & bit) != 0) & dropped

            if np.any(here):
                for name in flags.decode(bit):
                    told.append(f"{name} {int(np.sum(here))}")

        named = ', '.join(told)
    except Exception:
        pass

    log(f"    {int(np.sum(dropped))} of {len(quality)} cadences dropped"
        + (f" on the mission flags: {named}" if named else " on the mission flags"))


def clip_noisy_points(mag, err, groups=None, log=None, group_name='group',
                      ratio=CLIP_MAX_ERR_RATIO, nmin=CLIP_NMIN):
    """Which points are noisier than their own kind manages at that brightness.

    The single ruined frame - cloud, moon, a trail across the star - which a
    survey reports honestly as a large error. Judged within each group, since
    what counts as a large error differs between one camera and another, or
    one band and another, and judged against that group's own error at the
    same brightness, so that a variable seen faint is not read as a run of bad
    measurements.

    Parameters
    ----------
    mag, err : array
        The measurements and their quoted uncertainties
    groups : array, optional
        What each point was measured with - a camera, a band. Judged as one
        set when not given.
    log : callable, optional
        Told what was dropped, and from where
    group_name : str
        What to call a group when saying so

    Returns
    -------
    array of bool
        True where the point should be dropped
    """
    mag = np.asarray(mag, dtype=float)
    err = np.asarray(err, dtype=float)
    groups = (np.full(len(mag), '') if groups is None
              else np.asarray(groups).astype(str))

    clip = np.zeros(len(mag), dtype=bool)

    for name in sorted(set(groups)):
        idx = np.where(groups == name)[0]

        # Too few to say what is usual for them, and dropping any would be
        # guesswork
        if len(idx) < nmin:
            continue

        clip[idx] = err[idx] > ratio * typical_error(mag[idx], err[idx])

    if log is not None and np.any(clip):
        # Where they fell, unless there was nothing to divide them by
        named = [f"{name}: {int(np.sum(clip[groups == name]))}"
                 for name in sorted(set(groups))
                 if name and np.any(clip[groups == name])]

        log(f"Warning: dropping {int(np.sum(clip))} noisy points"
            + (f" ({', '.join(named)})" if named else '')
            + f" - errors over {ratio:.0f}x their {group_name}'s at that "
            f"brightness")

    return clip


class SourceError(RuntimeError):
    """A source could not get its data.

    Not to be confused with finding none: the archive refused, stalled, or
    answered with something that could not be read, and the step ends without
    data because of that. Raised rather than logged and returned, so that the
    run records the step as failed instead of as a survey with nothing to
    show; and reported as its message alone, there being no bug here to trace
    - the reason is the whole of the story.
    """


# How long any one request to IRSA may take. Their TAP service answers a cone
# search in seconds when it answers at all, so this is long enough to be no
# constraint on a working service and short enough to give up on a dead one.
IRSA_TIMEOUT = 120


class _TimeoutSession(requests.Session):
    """A session that will not wait for an answer forever.

    requests takes its timeout per call and offers nothing session-wide, so a
    library that asks for none - pyvo does, and that is what astroquery hands
    this to - would wait indefinitely on a service that accepts the connection
    and then says nothing. The timeout covers the wait between bytes as well as
    the connection, which matters for a body that is streamed.
    """

    def __init__(self, timeout):
        super().__init__()
        self.timeout = timeout

    def request(self, *args, **kwargs):
        kwargs.setdefault('timeout', self.timeout)
        return super().request(*args, **kwargs)


def irsa_client(timeout=IRSA_TIMEOUT):
    """astroquery's IRSA client, made to give up rather than hang.

    astroquery builds its TAP service around a session of its own making and
    never gives it conf.timeout - that setting governs the older paths, and is
    referenced nowhere in the class the queries here go through - so the
    request pyvo ends up making carries no timeout at all. A service that
    stalls would then hold the step until Celery's own limit killed it half an
    hour later. The session is replaced once, and the TAP service dropped so
    that it is built again around the new one.
    """
    from astroquery.ipac.irsa import Irsa

    if not isinstance(Irsa._session, _TimeoutSession):
        Irsa._session = _TimeoutSession(timeout)
        Irsa._tap = None

    return Irsa


def parse_votable_lenient(xml_content):
    """
    Parse a VOTable from raw XML content with lenient error handling.

    This function handles malformed XML that contains undefined entities,
    which is common in some TAP service responses (e.g., APPLAUSE DR4).

    Parameters
    ----------
    xml_content : bytes
        Raw XML content as bytes

    Returns
    -------
    astropy.table.Table
        Parsed table from the VOTable

    Notes
    -----
    Uses two strategies in order:
    1. lxml recovery mode (if available) - automatically fixes malformed XML
    2. Regex-based entity removal (fallback) - strips undefined entities

    Examples
    --------
    >>> response = requests.get(tap_service_url)
    >>> table = parse_votable_lenient(response.content)
    """
    try:
        from lxml import etree
        # Primary method: Use lxml's recovery parser to fix malformed XML
        parser = etree.XMLParser(recover=True, encoding='utf-8')
        tree = etree.fromstring(xml_content, parser=parser)
        # Convert back to bytes for astropy
        fixed_xml = etree.tostring(tree, encoding='utf-8')
        votable = votable_parse(BytesIO(fixed_xml), verify='ignore')
    except ImportError:
        # Fallback method: Manually clean undefined entities with regex
        import re
        xml_str = xml_content.decode('utf-8', errors='ignore')
        # Remove undefined entities (keep only standard XML entities)
        xml_str = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)[a-zA-Z0-9_]+;', '', xml_str)
        votable = votable_parse(BytesIO(xml_str.encode('utf-8')), verify='ignore')

    return votable.get_first_table().to_table()


@contextmanager
def cached_votable_query(cache_name, basepath, log, description, refresh=False):
    """Context manager for cached VOTable queries.

    Automatically handles cache checking, directory creation, and saving.
    Reduces duplication across all survey processing modules.

    Usage:
        with cached_votable_query('source_123.4567_45.6789.vot',
                                  basepath, log, 'Source Name') as cache:
            if not cache.hit:
                # Query code here - only runs if not cached
                data = external_api.query(...)
                cache.save(data)

            # Use cache.data (either from cache or just saved)
            result = cache.data

    Parameters
    ----------
    cache_name : str
        Cache filename (e.g., 'ptf_123.4567_45.6789.vot')
    basepath : str
        Base directory containing cache/ subdirectory
    log : callable
        Logging function
    description : str
        Human-readable name for logging (e.g., 'Palomar Transient Factory')
    refresh : bool, optional
        Drop whatever is cached and query the source again. Never automatic -
        a survey may be knowingly down, in which case an old cache is still
        better than no data at all - so this is only ever set by an explicit
        request from the user.

    Yields
    ------
    cache : CacheHelper
        Helper object with:
        - cache.hit : bool - True if data loaded from cache
        - cache.data : Table - Cached data (if hit=True) or None
        - cache.save(data) : Save data to cache
        - cache.path : str - Full cache file path

    Examples
    --------
    Coordinate-based caching:
        >>> cache_name = f"ptf_{ra:.4f}_{dec:.4f}.vot"
        >>> with cached_votable_query(cache_name, basepath, log, 'PTF') as cache:
        ...     if not cache.hit:
        ...         data = query_ptf(ra, dec)
        ...         cache.save(data)
        ...     ptf = cache.data

    Name-based caching:
        >>> safe_name = target_name.replace(' ', '_')
        >>> cache_name = f"kws_{safe_name}.vot"
        >>> with cached_votable_query(cache_name, basepath, log, 'KWS') as cache:
        ...     if not cache.hit:
        ...         data = query_kws(target_name)
        ...         cache.save(data)
        ...     kws = cache.data
    """
    cache_path = os.path.join(basepath, 'cache', cache_name)

    class CacheHelper:
        """Helper class for cache operations."""

        def __init__(self):
            self.hit = False
            self.data = None
            self.path = cache_path
            self._saved = False

        def save(self, data):
            """Save data to cache.

            Parameters
            ----------
            data : Table
                Astropy table to cache
            """
            if self._saved:
                return  # Already saved

            # Create cache directory
            os.makedirs(os.path.join(basepath, 'cache'), exist_ok=True)

            # Save to cache
            data.write(cache_path, format='votable', overwrite=True)
            self.data = data
            self._saved = True
            log(f"Cached {description} data to {cache_name}")

        def invalidate(self):
            """Remove invalid cached data and reset state for re-query."""
            if os.path.exists(cache_path):
                os.remove(cache_path)
            self.hit = False
            self.data = None
            self._saved = False

    cache = CacheHelper()

    if refresh and os.path.exists(cache_path):
        log(f"Dropping cached {description} data ({cache_name}) as requested")
        cache.invalidate()

    # Try loading from cache
    if os.path.exists(cache_path):
        log(f"Loading {description} from cache ({cache_name})")
        cache.data = Table.read(cache_path)
        cache.hit = True
        cache._saved = True  # Already have data
    else:
        log(f"Querying {description}...")
        cache.hit = False

    try:
        yield cache
    finally:
        pass  # Could add cleanup here if needed


def cleanup_paths(paths, basepath=None):
    """Remove files matching patterns in paths list."""
    for path in paths:
        for fullpath in glob.glob(os.path.join(basepath, path)):
            if os.path.exists(fullpath):
                if os.path.isdir(fullpath):
                    shutil.rmtree(fullpath)
                else:
                    os.unlink(fullpath)


def print_to_file(*args, clear=False, logname='out.log', **kwargs):
    """Print to both stdout and a log file."""
    if clear and os.path.exists(logname):
        print('Clearing', logname)
        os.unlink(logname)

    if len(args) or len(kwargs):
        print(*args, **kwargs)
        with open(logname, 'a+') as lfd:
            print(file=lfd, *args, **kwargs)


def pickle_to_file(filename, obj):
    """Save object to file using dill pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pickle_from_file(filename):
    """Load object from pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def log_bands(verbose, source, bands, heading=True):
    """Report which bands a source is publishing, and where each comes from.

    Called once per source, after the columns have been filled in, so that the
    log says plainly what is a measurement and what is a model.

    Sources with a log of their own get a heading; those writing into a shared
    log, where the section above already names them, pass heading=False rather
    than nesting one heading inside another.
    """
    log = verbose if callable(verbose) else (print if verbose else lambda *a, **kw: None)

    if heading:
        log("\n---- Bands published ----\n")
    for b in bands:
        n = b.get('npoints')
        log(f"  {source} {b['label']:<4s} [{b['kind']}]"
            + (f"  {n} points" if n is not None else '')
            + (f"  - {b['note']}" if b.get('note') else ''))


def log_conversion(verbose, source, formula, params=None, npoints=None, note=None,
                   heading=True):
    """Report a photometric conversion and every parameter that went into it.

    Anything that changes the magnitudes - a colour term, an assumed colour, a
    polynomial fit - is printed here, so that a light curve can be traced back
    to the numbers that produced it.

    Parameters
    ----------
    verbose : callable or bool
        The usual logging callable of the processing functions.
    source : str
        Survey name, for the heading.
    formula : str
        Human-readable formula, e.g. 'g = V + 0.02 + 0.498*(g-r) + ...'.
    params : dict, optional
        Parameter values, mapped to either the value itself or a
        (value, origin) pair - the origin saying where the number came from.
    npoints : int, optional
        How many points the conversion was applied to.
    note : str, optional
        Any caveat worth stating, e.g. that a colour is assumed constant.
    heading : bool, optional
        Whether to head the block. Sources writing into a shared log, already
        under a heading naming them, pass False.
    """
    log = verbose if callable(verbose) else (print if verbose else lambda *a, **kw: None)

    if heading:
        log(f"\n---- {source}: photometric conversion ----\n")
    log(f"  {formula}")

    for key, value in (params or {}).items():
        if isinstance(value, tuple) and len(value) == 2:
            value, origin = value
            origin = f"  [{origin}]"
        else:
            origin = ''

        if isinstance(value, float):
            log(f"    {key} = {value:.4f}{origin}")
        else:
            log(f"    {key} = {value}{origin}")

    if npoints is not None:
        log(f"  applied to {npoints} points")

    if note:
        log(f"  note: {note}")


def mast_download_dir(basepath, source_id):
    """Where lightkurve should put a source's downloads.

    A directory of its own for each source, rather than the single tree
    lightkurve arranges by mission. The missions used to share that tree, and
    its HLSP folder holds the community products of all of them named for the
    pipeline rather than the mission - so a source had to pick its own out of
    the others', and two of them acquiring at once wrote into the same place.
    """
    return os.path.join(basepath, 'cache', f'mast_{source_id}')


def drop_mast_downloads(basepath, source_id, log):
    """Drop a source's lightkurve downloads.

    A source owns its download directory outright, so this is all of them.
    """
    path = mast_download_dir(basepath, source_id)

    if not os.path.exists(path):
        return

    shutil.rmtree(path, ignore_errors=True)
    log("Dropped the cached downloads as requested")
