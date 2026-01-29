"""Common utilities for processing astronomical data."""

import os
import glob
import shutil
import re
import dill as pickle
from io import BytesIO

from astropy.io.votable import parse as votable_parse


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


# File lists for cleanup operations
files_info = [
    'info.log',
    'galaxy_map.png',
]

files_cache = [
    'ps1.vot',
    'ztf.vot',
    'asas.vot',
    'css.vot',
    'dasch.vot',
    'applause.vot',
]

files_cache += [_.replace('.vot', '.txt') for _ in files_cache if '.vot' in _]

files_ztf = [
    'ztf.log',
    'ztf_lc.png',
    'ztf_color_mag.png',
]

files_asas = [
    'asas.log',
    'asas_lc.png',
    'asas_color_mag.png',
]

files_tess = [
    'tess.log',
    'tess_lc_*.png',
    'tess_lc_*.vot',
]

files_dasch = [
    'dasch.log',
    'dasch_lc.png',
]

files_applause = [
    'applause.log',
    'applause_lc.png',
]

files_mmt9 = [
    'mmt9.log',
    'mmt9_lc.png',
]

files_css = [
    'css.log',
    'css_lc.png',
]

files_combined = [
    'combined.log',
    'combined_lc.png',
]

cleanup_info = files_info + files_cache + files_ztf + files_asas + files_tess + files_dasch + files_applause + files_mmt9 + files_css + files_combined
cleanup_ztf = files_ztf
cleanup_asas = files_asas
cleanup_tess = files_tess
cleanup_dasch = files_dasch
cleanup_applause = files_applause
cleanup_mmt9 = files_mmt9
cleanup_css = files_css
cleanup_combined = files_combined


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
