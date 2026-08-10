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
- wise.py: WISE / NEOWISE infrared epoch photometry
- asas3.py: ASAS-3 lightcurve acquisition
- wasp.py: SuperWASP lightcurve acquisition
- omc.py: INTEGRAL OMC lightcurve acquisition
- nsvs.py: NSVS lightcurve acquisition
- hipparcos.py: Hipparcos epoch photometry acquisition
- lamost.py: LAMOST DR11 spectra
- kepler.py: Kepler lightcurve acquisition, both the original mission and K2
- combined.py: Combined lightcurve plotting
"""

# astroquery keeps a cache of its own under ~/.astropy/cache, holding pickled
# responses for a week. It is turned off here, before anything can query
# through it, for three reasons:
#
#  - the sources cache their results themselves, per target, under
#    targets/{id}/cache, so it duplicates a layer we already have;
#  - that layer invalidates only when asked to, which is the point of it - and
#    an expiring cache underneath quietly overrode the request, so 'Ignore
#    cache' would delete our copy, re-run the query and be handed a reply up
#    to seven days old;
#  - it writes each response with a plain open() and no rename, so two
#    equal queries running at once - the NSVS field list is the same for
#    every target - can leave a half-written pickle that the next read
#    raises on.
from astroquery import cache_conf
from astroquery import query as astroquery_query

cache_conf.cache_active = False

# That switch alone is not enough. astroquery only consults it when the caller
# leaves the flag as None, and Vizier - the one client here that uses this
# cache - passes cache=True from its own method defaults instead. Passing
# cache=False at each call site would not cover it either, as stdpipe queries
# Vizier on our behalf. So the two ends of the cache are stubbed out directly;
# should astroquery ever rename them, this raises on import rather than
# quietly starting to cache again.
assert hasattr(astroquery_query, 'to_cache')
assert hasattr(astroquery_query.AstroQuery, 'from_cache')

astroquery_query.to_cache = lambda response, cache_file: None
astroquery_query.AstroQuery.from_cache = lambda self, cache_location, cache_timeout: None

# Import all utilities
from .utils import (
    parse_votable_lenient,
    cleanup_paths,
    print_to_file,
    pickle_to_file,
    pickle_from_file,
    cached_votable_query,
    log_bands,
    log_conversion,
)

# Import all processing functions
from .info import target_info
from .ztf import target_ztf, gaussian_smoothing
from .asas import target_asas
from .mmt9 import target_mmt9
from .css import target_css
from .kws import target_kws
from .ptf import target_ptf
from .wise import target_wise
from .asas3 import target_asas3
from .wasp import target_wasp
from .omc import target_omc
from .nsvs import target_nsvs
from .hipparcos import target_hipparcos
from .lamost import target_lamost
from .kepler import target_kepler
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
