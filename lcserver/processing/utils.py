"""Common utilities for processing astronomical data."""

import os
import glob
import shutil
import re
import dill as pickle
from io import BytesIO
from contextlib import contextmanager

from astropy.io.votable import parse as votable_parse
from astropy.table import Table


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
def cached_votable_query(cache_name, basepath, log, description):
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

    cache = CacheHelper()

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
