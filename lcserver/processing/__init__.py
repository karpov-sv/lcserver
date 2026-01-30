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
- ptf.py: Palomar Transient Factory lightcurve acquisition
- combined.py: Combined lightcurve plotting
"""

# Import all utilities
from .utils import (
    parse_votable_lenient,
    cleanup_paths,
    print_to_file,
    pickle_to_file,
    pickle_from_file,
    cached_votable_query,
)

# Import all processing functions
from .info import target_info
from .ztf import target_ztf, gaussian_smoothing
from .asas import target_asas
from .mmt9 import target_mmt9
from .css import target_css
from .kws import target_kws
from .ptf import target_ptf
from .tess import target_tess
from .dasch import target_dasch
from .applause import target_applause
from .combined import target_combined

# Export all functions and utilities
__all__ = [
    # Utilities
    'parse_votable_lenient',
    'cleanup_paths',
    'print_to_file',
    'pickle_to_file',
    'pickle_from_file',
    'cached_votable_query',
    'gaussian_smoothing',
    # Processing functions
    'target_info',
    'target_ztf',
    'target_asas',
    'target_css',
    'target_kws',
    'target_ptf',
    'target_tess',
    'target_dasch',
    'target_applause',
    'target_mmt9',
    'target_combined',
]
