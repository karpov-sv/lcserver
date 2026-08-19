"""Common utilities for processing astronomical data."""

import os
import glob
import shutil
import re
import warnings
import requests
import numpy as np
import dill as pickle
from io import BytesIO
from contextlib import contextmanager

from astropy.io.votable import parse as votable_parse
from astropy.table import Table
from astropy.units import UnitsWarning


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


# How every spectrum here is stored, whatever the survey publishes: wavelength
# in Angstrom, and flux per unit wavelength in erg/s/cm2/A. The surveys agree
# on none of this between them - Gaia XP comes in W/nm/m2 against wavelengths
# in nm, LAMOST and DESI in units of 1e-17 erg/s/cm2/A, SPHEREx in uJy against
# microns, which is not even per unit wavelength - so each converts on the way
# out. The files can then be read against each other, and exported, without
# carrying a table of what each one meant.
SPECTRUM_WAVELENGTH_UNIT = 'Angstrom'
SPECTRUM_FLUX_UNIT = 'erg / (s cm2 Angstrom)'

# The columns those two apply to, wherever a spectrum has them
SPECTRUM_WAVELENGTH_COLUMNS = ('wavelength', 'bandwidth')
SPECTRUM_FLUX_COLUMNS = ('flux', 'flux_error', 'scatter')

# Speed of light in Angstrom/s, and a microjansky in erg/s/cm2/Hz, which is
# what turns a flux per unit frequency into one per unit wavelength
SPEED_OF_LIGHT = 2.99792458e18
MICROJANSKY = 1e-29


def flambda_from_fnu(flux, wavelength):
    """A flux per unit frequency, in uJy, as one per unit wavelength.

    f_lambda = f_nu c / lambda^2, with the wavelength in Angstrom and the
    answer in erg/s/cm2/A. Not a change of units but of variable: the two
    differ by a factor of lambda squared, so a spectrum converted from one to
    the other changes shape and not merely scale.
    """
    wavelength = np.asarray(wavelength, dtype=float)

    return (np.asarray(flux, dtype=float) * MICROJANSKY
            * SPEED_OF_LIGHT / wavelength ** 2)


def write_spectrum(table, basepath, name, calibrated=True):
    """Write a spectrum out, in the one form they are all written in.

    Every source goes through here rather than writing its own pair of files,
    so that the format cannot drift between one and another: the units are
    stamped on the table, and both a VOTable and a text table are left, as
    everything else here leaves. The VOTable carries the units, so a spectrum
    exported from here describes itself without anyone having to be told what
    the survey behind it published; the text form keeps only the column names,
    which is why those units are the same for every source rather than
    something each one declares for itself.

    `calibrated` says whether the flux is on a physical scale at all. Most of
    the archives here publish one; three quarters of the ESO spectra do not,
    their instruments having been built to measure where a line sits rather
    than how much light arrived - and HARPS, which is most of them, reports
    detector counts. Those are still worth reading, the shape of a line being
    the whole point of a spectrum at R = 115000, but stamping erg/s/cm2/A on
    counts would be a lie the exported file then carries everywhere. So a
    source with no flux scale says so, its flux columns are left without a
    unit, and it is expected to have divided by something of its own - the
    median of the spectrum - so that the numbers still mean one thing.

    The wavelength is always stamped: every source knows what that is.

    Returns the two file names, for the source to say where it put them.
    """
    for column in SPECTRUM_WAVELENGTH_COLUMNS:
        if column in table.colnames:
            table[column].unit = SPECTRUM_WAVELENGTH_UNIT

    for column in SPECTRUM_FLUX_COLUMNS:
        if column in table.colnames:
            table[column].unit = SPECTRUM_FLUX_UNIT if calibrated else None

    # Angstrom and erg are both deprecated in the strict VOUnit standard, in
    # favour of 0.1nm and cm2.g.s-2, and astropy says so on every write. The
    # units astronomers read are worth more than the standard's preference,
    # and the warning is not worth a line of a worker's log for each spectrum.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnitsWarning)

        table.write(os.path.join(basepath, name + '.vot'),
                    format='votable', overwrite=True)

    table.write(os.path.join(basepath, name + '.txt'),
                format='ascii.commented_header', overwrite=True)

    return name + '.vot', name + '.txt'


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


# How faint a measurement's error bar is drawn beneath its own point
ERRORBAR_ALPHA = 0.2


def plot_with_errors(ax, x, y, yerr=None, xerr=None, color=None, label=None,
                     marker='.', error_alpha=ERRORBAR_ALPHA, **kwargs):
    """Draw measurements over their error bars, with the bars faint.

    A survey with small errors and many points draws its bars over its own
    measurements: each bar is a stroke as wide as the point it belongs to and
    darkest where the points are densest, so a well-sampled light curve reads
    as a band with no points in it. Drawn faint and underneath, the bars still
    say how well each point is known without taking the light curve over.

    The points go down first and the bars take their colour from them: left to
    itself a second call would draw the next colour in the axes' cycle, and
    the bars would not match the points they belong to.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Where to draw
    x, y : array
        The measurements
    yerr, xerr : array, optional
        What is known of their uncertainty. Bars are drawn only for those
        given.
    color : str, optional
        The colour of the points, taken from the axes' cycle when not given
    label : str, optional
        For the legend, carried by the points
    marker : str, optional
        The marker to draw the points with
    error_alpha : float, optional
        How faint the bars are
    kwargs : dict
        Passed on to the points, so `ms` and `alpha` mean what they usually do

    Returns
    -------
    line : `matplotlib.lines.Line2D`
        The points, as `ax.plot` returns them
    """
    line, = ax.plot(x, y, marker=marker, ls='none', color=color, label=label,
                    **kwargs)

    if yerr is not None or xerr is not None:
        # Just under the points, so that a bar never draws over the point it
        # belongs to nor over its neighbours'
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='none',
                    ecolor=line.get_color(), alpha=error_alpha,
                    zorder=line.get_zorder() - 0.5)

    return line


def break_at_gaps(time, flux, tolerance=1.5):
    """Keep a line from being drawn across a gap in the sampling.

    A pipeline that publishes the cadences it could not measure, with nothing
    in their flux, breaks its own line - matplotlib draws through no NaN. One
    that leaves them out instead hands over two runs of points with a jump
    between them, and the line is then drawn straight across the gap, which
    reads as data where there is none. TEQUILA does this at every mid-sector
    downlink; QLP does it too, and is saved from it only when the quality mask
    happens to empty the cadences on either side.

    So a gap wider than a cadence and a half gets an empty point of its own
    for the line to break at. It goes at the far side of the gap, a cadence
    before the point that resumes it, because these are drawn as steps: a step
    is horizontal into its own point, so an empty point at the near side would
    leave that horizontal run to be drawn clear across the gap anyway. Put at
    the far side it leaves a step of the usual width instead.

    Returns the two arrays with those points inserted.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if len(time) < 3:
        return time, flux

    steps = np.diff(time)
    cadence = np.median(steps[steps > 0]) if np.any(steps > 0) else 0

    if not cadence > 0:
        return time, flux

    gaps = np.nonzero(steps > tolerance * cadence)[0]

    if not len(gaps):
        return time, flux

    return (np.insert(time, gaps + 1, time[gaps + 1] - cadence),
            np.insert(flux, gaps + 1, np.nan))


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


def shared_cache_dir(basepath):
    """Where something the same for every target is kept.

    Beside the per-target caches rather than inside one of them: a table that
    describes the surveys rather than the position - VizieR's list of filters
    is the only one so far - does not belong to whichever target happened to
    be the first to want it, and copying it into each would be a few hundred
    kilobytes per target of the identical thing.

    Derived from the target's own path so that nothing here has to know the
    Django settings: targets/{id}/ has targets/ above it, and the shared cache
    sits there. It therefore follows TARGETS_PATH wherever it points.
    """
    return os.path.join(os.path.dirname(os.path.abspath(basepath)), 'cache')


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
                if data is not None and len(data):
                    cache.save(data)
                else:
                    cache.save_empty()

            # Use cache.data - the query's rows, or None where there were none
            result = cache.data

    An answer of no rows is cached like any other, as an empty file, and comes
    back as `cache.data` of None. It is an answer: a star outside a survey's
    footprint is outside it for good, and a service that has nothing on a
    target usually has nothing on it next month either. Left uncached, those
    are the queries that run on every single acquisition - SkyMapper alone
    took a minute of every info run it could never have data for.

    Only the source knows it was an answer, though, which is why this is a
    call of its own rather than something `save` infers from an empty table: a
    query that raised, timed out, or came back malformed must not reach it.
    Cache a failure and the target carries it until someone asks for a
    refresh.

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
        - cache.hit : bool - True if the source has been asked before
        - cache.data : Table - What it answered, or None where that was nothing
        - cache.save(data) : Save data to cache
        - cache.save_empty() : Remember that the answer was nothing
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
            log(f"Cached {description} data to cache:{cache_name}")

        def save_empty(self):
            """Remember that the source answered, and had nothing.

            Only ever for an answer. A query that raised or returned something
            unreadable has said nothing about the target, and caching that
            would keep saying it until the cache is refreshed by hand.
            """
            if self._saved:
                return

            os.makedirs(os.path.join(basepath, 'cache'), exist_ok=True)

            # An empty file rather than an empty table: there is generally no
            # table to write - a catalogue with no match returns nothing at
            # all rather than a row-less one - and a file of no bytes cannot
            # be mistaken for data by anything that reads the cache directory
            open(cache_path, 'w').close()
            self.data = None
            self._saved = True
            log(f"Cached the empty {description} answer to cache:{cache_name}")

        def invalidate(self):
            """Remove invalid cached data and reset state for re-query."""
            if os.path.exists(cache_path):
                os.remove(cache_path)
            self.hit = False
            self.data = None
            self._saved = False

    cache = CacheHelper()

    if refresh and os.path.exists(cache_path):
        log(f"Dropping cached {description} data ({cache_name})")
        cache.invalidate()

    # Try loading from cache
    if os.path.exists(cache_path):
        if not os.path.getsize(cache_path):
            log(f"{description} cache is empty (cache:{cache_name})")
            cache.data = None
        else:
            log(f"Loading {description} from cache (cache:{cache_name})")
            cache.data = Table.read(cache_path)

        cache.hit = True
        cache._saved = True  # Already answered, one way or the other
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


# ---------------------------------------------------------------------------
# Photometric conversions onto the common g scale.
#
# The combined light curve draws every survey on one axis, so everything on it
# has to be in one band, and the surveys measured in a dozen. These are the
# relations that bring them together, kept in one place rather than repeated in
# each module, so that the combined curve is built from one set of numbers and
# a correction to any of them reaches every source at once.
#
# Each is paired with the formula it prints, so that what the log says and what
# the code does cannot drift apart.
#
# All of them but one assume a single fixed colour for the star, taken from the
# config. That is a model rather than a measurement, and it costs most exactly
# where the light curve is most interesting: when a variable changes colour as
# it changes brightness, a constant colour mis-scales the amplitude by roughly
# the colour coefficient times how far the colour actually moved. ZTF is the
# exception, reconstructing its own colour per epoch.
# ---------------------------------------------------------------------------

V_TO_G_FORMULA = 'g = V + 0.02 + 0.498*(g - r) + 0.008*(g - r)^2'


def v_to_g(mag, g_minus_r):
    """Johnson V onto the Pan-STARRS g scale, through an assumed (g - r)."""
    return mag + 0.02 + 0.498*g_minus_r + 0.008*g_minus_r**2


B_TO_G_FORMULA = 'g = B - 0.3130*(g - r) - 0.2271'


def b_to_g(mag, g_minus_r):
    """Johnson B onto g - Lupton (2005), inverted.

    Published as B = g + 0.3130*(g - r) + 0.2271, with a scatter of 0.011 for
    the stars it was fitted to. The photographic plates it is used on here are
    a good deal further from Johnson B than that, so the number to expect is
    the plates' own colour term, not this one.
    """
    return mag - 0.3130*g_minus_r - 0.2271


R_TO_G_FORMULA = 'g = r + (g - r)'


def r_to_g(mag, g_minus_r):
    """An r-band magnitude onto g, by the colour itself."""
    return mag + g_minus_r


ROTSE_TO_V_FORMULA = 'V = m_ROTSE + (B - V)/1.875'


def rotse_to_v(mag, b_minus_v):
    """Unfiltered ROTSE-I onto Johnson V.

    The NSVS magnitudes are defined against V with a colour term already in
    them - m_ROTSE = V - (B - V)/1.875 (Wozniak et al. 2004) - so the band is
    on the V scale for a star of zero colour and drifts from it for any other.
    Inverted here to put it back on V.
    """
    return mag + b_minus_v/1.875


GAIA_G_TO_G_FORMULA = ('g = G - (0.2199 - 0.6365*x - 0.1548*x^2 + 0.0064*x^3),'
                       '  x = BP - RP')

# Where the relation above was fitted, and so where it means anything
GAIA_G_TO_G_RANGE = (0.3, 3.0)
GAIA_G_TO_G_SIGMA = 0.0745


def gaia_g_to_g(mag, bp_minus_rp):
    """Gaia G onto SDSS g - Gaia EDR3 documentation, Table 5.6.

    G is used rather than BP, which sits closer to g and would need a smaller
    correction: G is the more precisely measured of the two, and the relation
    for it is the better determined. Valid over GAIA_G_TO_G_RANGE in BP - RP.
    """
    x = bp_minus_rp
    return mag - (0.2199 - 0.6365*x - 0.1548*x**2 + 0.0064*x**3)


def assumed_color(config, key, default=0.0):
    """The colour a conversion is to use, and where it came from.

    Returns a (value, origin) pair, the origin being what log_conversion
    prints beside the number. A key that is present but empty counts as
    absent: the info step writes the colours it could not determine as None.
    """
    try:
        value = float(config.get(key))
    except (TypeError, ValueError):
        return default, 'default, no colour known'

    if not np.isfinite(value):
        return default, 'default, no colour known'

    return value, 'from config'


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


def cached_lightkurve_search(cache_name, basepath, log, description,
                             refresh=False, **kwargs):
    """A MAST product search, kept on disk like every other source's query.

    lightkurve caches its searches too, but in the process and nowhere else,
    with no expiry and no way in from outside. A worker that lives for weeks
    would answer from that memo forever - a sector released after it started
    would never be seen, and dropping the downloads could not force a fresh
    look - while a worker that restarts loses it entirely and pays the several
    seconds again. So the query goes to the undecorated function and what it
    returns is cached here instead, where refresh reaches it.

    Parameters
    ----------
    cache_name : str
        Cache filename, as for `cached_votable_query`
    basepath : str
        Base directory containing the cache/ subdirectory
    log : callable
        Logging function
    description : str
        Human-readable name of what is being searched for, for the log
    refresh : bool, optional
        Drop what is cached and ask MAST again
    kwargs : dict
        Passed on to `lightkurve.search_lightcurve`

    Returns
    -------
    res : `lightkurve.SearchResult`
        The products found, ready to download from
    """
    import lightkurve as lk

    cache_path = os.path.join(basepath, 'cache', cache_name)

    if refresh and os.path.exists(cache_path):
        log(f"Dropping cached {description} search (cache:{cache_name})")
        os.unlink(cache_path)

    if os.path.exists(cache_path):
        log(f"Loading {description} search from cache (cache:{cache_name})")
        return lk.SearchResult(Table.read(cache_path))

    log(f"Querying {description}...")

    # Undecorated, so that the in-process memo is neither consulted nor
    # filled: the file below is the only cache, and it can be dropped
    search = getattr(lk.search_lightcurve, '__wrapped__', lk.search_lightcurve)
    res = search(**kwargs)

    table = res.table.copy()
    # Row numbers for display, which a SearchResult makes for itself
    if '#' in table.colnames:
        table.remove_column('#')

    os.makedirs(os.path.join(basepath, 'cache'), exist_ok=True)
    table.write(cache_path, format='votable', overwrite=True)
    log(f"Cached {description} search to cache:{cache_name}")

    return res


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


def download_mast_product(row, download_dir):
    """Fetch one product from MAST, and say where it landed.

    Deliberately the download lightkurve would have made, into the same place,
    so that a pipeline read here is cached, reused and refreshed exactly as
    the ones read by lightkurve are.

    Parameters
    ----------
    row : `lightkurve.SearchResult`
        A single product to download
    download_dir : str
        Where this source keeps its downloads, as for `row.download()`

    Returns
    -------
    path : str
        Where the file is on disk
    """
    table = row.table[:1]

    # Where lightkurve's downloader would have left it - hard-coded there too,
    # to save asking MAST for the size of a file that is already on disk
    path = os.path.join(download_dir.rstrip('/'), 'mastDownload',
                        str(table['obs_collection'][0]),
                        str(table['obs_id'][0]),
                        str(table['productFilename'][0]))

    if os.path.exists(path):
        return path

    from astroquery.mast import Observations

    response = Observations.download_products(table, mrp_only=False,
                                              download_dir=download_dir)[0]
    if response['Status'] != 'COMPLETE':
        raise RuntimeError(
            f"Could not download {table['dataURI'][0]}: "
            f"{response['Status']}: {response['Message']}")

    return response['Local Path']


def download_tequila_lightcurve(row, download_dir):
    """Fetch one TEQUILA light curve, which lightkurve cannot open itself.

    lightkurve finds these products - they carry an author like any other -
    but dies on opening them. The files were written with `LightCurve.to_fits`,
    so its detector reads the CREATOR card, takes them for SPOC light curves,
    and the SPOC reader then fails on a missing 'quality' column, which
    TEQUILA calls SAP_QUALITY. It raises rather than returning nothing, which
    would end the whole step, so the file is fetched and read here.

    Parameters
    ----------
    row : `lightkurve.SearchResult`
        A single product to download
    download_dir : str
        Where this source keeps its downloads, as for `row.download()`

    Returns
    -------
    lc : `lightkurve.TessLightCurve`
        The light curve, carrying the metadata the plotting expects
    """
    import lightkurve as lk
    from astropy.io import fits
    from astropy.time import Time
    from astropy import units as u

    path = download_mast_product(row, download_dir)

    with fits.open(path) as hdus:
        header, data = hdus[0].header, hdus[1].data
        flux_unit = u.electron / u.s

        # FLUX is the master frame's reference flux plus the differential one,
        # so it stands on the same footing as what the other pipelines
        # publish; d_FLUX is what the difference imaging actually measured,
        # and is kept beside it. TIME is BTJD offset by 2457000, whatever the
        # TIMESYS card says - it is a quarter of an hour from where the other
        # pipelines put the same cadence, so this is not the light curve to
        # phase a precise ephemeris on.
        lc = lk.TessLightCurve(
            time=Time(np.asarray(data['TIME'], dtype=float),
                      format='jd', scale='tdb'),
            flux=np.asarray(data['FLUX'], dtype=float) * flux_unit,
            flux_err=np.asarray(data['FLUX_ERR'], dtype=float) * flux_unit,
            # Published in every file, and zero in every one seen so far
            quality=np.asarray(data['SAP_QUALITY'], dtype=int),
            meta={'SECTOR': header['SECTOR'],
                  'AUTHOR': header['HLSPID'],
                  'MISSION': 'TESS',
                  'FLUX_ORIGIN': 'flux',
                  'LABEL': header['HLSPTARG'],
                  'TARGETID': header['HLSPTARG'],
                  'CAMERA': header['CAMERA'],
                  'CCD': header['CCD']})

        lc['d_flux'] = np.asarray(data['d_FLUX'], dtype=float) * flux_unit
        lc['d_flux_err'] = np.asarray(data['d_FLUX_ERR'], dtype=float) * flux_unit

    return lc


def download_tars_lightcurve(row, download_dir):
    """Fetch one TARS light curve, which lightkurve cannot open either.

    A different failure from TEQUILA's, and a plainer one: the files carry no
    CREATOR card at all, so lightkurve's detector recognises nothing and its
    reader refuses the file outright. It raises, as ever, so this is read here.

    There is not much to read. TARS publishes two columns - the time, and a
    flux already normalised by its own median - with no quality flags, so the
    quality control does not reach these light curves. What it does publish
    beside them is its periodogram, which is the survey's point: the periods
    of its five strongest peaks, and the amplitude of a sine fit at the first.
    Those are carried into the metadata for the log.

    Parameters
    ----------
    row : `lightkurve.SearchResult`
        A single product to download
    download_dir : str
        Where this source keeps its downloads, as for `row.download()`

    Returns
    -------
    lc : `lightkurve.TessLightCurve`
        The light curve, carrying the metadata the plotting expects
    """
    import lightkurve as lk
    from astropy.io import fits
    from astropy.time import Time

    path = download_mast_product(row, download_dir)

    with fits.open(path) as hdus:
        header, data = hdus[0].header, hdus[1].data

        flux = np.asarray(data['FLUX'], dtype=float)

        # No per-point uncertainty is published either, only the sector's
        # point-to-point noise, and every point is given that. An empty column
        # would be worse than a rough one: everything downstream asks for a
        # finite error and drops what has none, so the light curve would go
        # missing from the folding and the periodograms rather than merely
        # carrying no error bars.
        rms = float(hdus[1].header.get('P2PRMS') or 0.0)
        if not rms > 0:
            rms = float(np.nanstd(np.diff(flux)) / np.sqrt(2))

        # TIME is BTJD in TDB, as the header says and as it agrees with QLP
        lc = lk.TessLightCurve(
            time=Time(np.asarray(data['TIME'], dtype=float) + 2457000.0,
                      format='jd', scale='tdb'),
            flux=flux,
            flux_err=np.full(len(flux), rms),
            meta={'SECTOR': header['SECTOR'],
                  'AUTHOR': header['HLSPID'],
                  'MISSION': 'TESS',
                  'FLUX_ORIGIN': 'flux',
                  'LABEL': header['HLSPTARG'],
                  'TARGETID': header.get('TICID'),
                  'CAMERA': header['CAMERA'],
                  'CCD': header['CCD'],
                  # From the table's own header rather than the primary one,
                  # and only ever read together, so absent is zero
                  'PERIOD': float(hdus[1].header.get('PER1') or 0.0),
                  'PERIOD_AMPLITUDE': float(hdus[1].header.get('AMPFIT') or 0.0),
                  'PERIOD_SNR': float(hdus[1].header.get('SNR') or 0.0)})

    return lc
