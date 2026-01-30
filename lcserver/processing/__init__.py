"""Processing module for LCServer.

This module contains all processing functions for acquiring lightcurves
from various astronomical surveys.

The module is organized into separate files:
- utils.py: Common utilities (cleanup_paths, print_to_file, etc.)
- info.py: Target info acquisition (coordinates, catalog photometry)
- ztf.py: ZTF lightcurve acquisition
- asas.py: ASAS-SN lightcurve acquisition
- tess.py: TESS lightcurve acquisition
- dasch.py: DASCH lightcurve acquisition
- applause.py: APPLAUSE lightcurve acquisition
- mmt9.py: Mini-MegaTORTORA lightcurve acquisition
- css.py: Catalina Sky Survey lightcurve acquisition
- kws.py: Kamogata Wide-field Survey lightcurve acquisition
- combined.py: Combined lightcurve plotting
"""

# Import all utilities
from .utils import (
    parse_votable_lenient,
    cleanup_paths,
    print_to_file,
    pickle_to_file,
    pickle_from_file,
    files_info,
    files_cache,
    files_ztf,
    files_asas,
    files_tess,
    files_dasch,
    files_applause,
    files_mmt9,
    files_css,
    files_kws,
    files_combined,
    cleanup_info,
    cleanup_ztf,
    cleanup_asas,
    cleanup_tess,
    cleanup_dasch,
    cleanup_applause,
    cleanup_mmt9,
    cleanup_css,
    cleanup_kws,
    cleanup_combined,
)

# Import all processing functions
from .info import target_info, gaussian_smoothing
from .ztf import target_ztf
from .asas import target_asas
from .mmt9 import target_mmt9
from .css import target_css
from .kws import target_kws
from .tess import target_tess
from .dasch import target_dasch
from .applause import target_applause
from .combined import target_combined

# Register lightcurve-only sources (no processing function)
# These sources have data files but no automated acquisition
from .. import surveys

surveys.register_lightcurve_source(
    source_id='ps1',
    name='Pan-STARRS',
    short_name='Pan-STARRS',
    votable_file='ps1.vot',
    lc_mag_column='mag_g',
    lc_err_column='magerr',
    lc_filter_column='g',
    lc_color='#2ca02c',
    lc_mode='magnitude',
    lc_short=True,
)

# Export all functions and utilities
__all__ = [
    # Utilities
    'parse_votable_lenient',
    'cleanup_paths',
    'print_to_file',
    'pickle_to_file',
    'pickle_from_file',
    'gaussian_smoothing',
    # File lists
    'files_info',
    'files_cache',
    'files_ztf',
    'files_asas',
    'files_tess',
    'files_dasch',
    'files_applause',
    'files_mmt9',
    'files_css',
    'files_kws',
    'files_combined',
    # Cleanup lists
    'cleanup_info',
    'cleanup_ztf',
    'cleanup_asas',
    'cleanup_tess',
    'cleanup_dasch',
    'cleanup_applause',
    'cleanup_mmt9',
    'cleanup_css',
    'cleanup_kws',
    'cleanup_combined',
    # Processing functions
    'target_info',
    'target_ztf',
    'target_asas',
    'target_css',
    'target_kws',
    'target_tess',
    'target_dasch',
    'target_applause',
    'target_mmt9',
    'target_combined',
]
