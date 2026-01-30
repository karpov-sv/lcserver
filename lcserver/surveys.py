"""Survey source registry for LCServer.

This module provides the @survey_source decorator for registering data sources.
Survey sources are registered by decorating processing functions in processing.py.

Adding a new survey source requires:
1. Implementing processing function in processing.py
2. Decorating it with @survey_source(...) with metadata
"""

# Global registry populated by @survey_source decorator
SURVEY_SOURCES = {}


def survey_source(
    name,
    short_name,
    state_acquiring,
    state_acquired,
    log_file,
    output_files,
    button_text,
    button_class='btn-primary',
    form_fields=None,
    help_text='',
    order=50,
    # Lightcurve metadata
    votable_file=None,
    lc_mag_column=None,
    lc_err_column=None,
    lc_filter_column=None,
    lc_flux_column=None,
    lc_quality_column=None,
    lc_color=None,
    lc_mode=None,  # 'magnitude' or 'flux'
    lc_short=False,
):
    """
    Decorator to register a survey data source.

    Automatically extracts source_id from function name (target_xxx -> xxx)
    and registers the function in SURVEY_SOURCES.

    Example:
        @survey_source(
            name='ZTF',
            short_name='ZTF',
            state_acquiring='acquiring ZTF lightcurve',
            state_acquired='ZTF lightcurve acquired',
            log_file='ztf.log',
            output_files=['ztf.log', 'ztf_lc.png'],
            button_text='Get ZTF lightcurve',
            help_text='Zwicky Transient Facility',
            order=10,
            # Lightcurve metadata
            votable_file='ztf.vot',
            lc_mag_column='mag_g',
            lc_err_column='magerr',
            lc_filter_column='zg',
            lc_color='#ff7f0e',
            lc_mode='magnitude',
            lc_short=True,
        )
        def target_ztf(config, basepath='.', verbose=None, show=False):
            # processing code
            pass

    Parameters:
    -----------
    name : str
        Full display name (e.g., "Zwicky Transient Facility")
    short_name : str
        Short name for messages (e.g., "ZTF")
    state_acquiring : str
        State while processing (e.g., "acquiring ZTF lightcurve")
    state_acquired : str
        State after success (e.g., "ZTF lightcurve acquired")
    log_file : str
        Log filename (e.g., "ztf.log")
    output_files : list
        Expected output files (e.g., ["ztf.log", "ztf_lc.png"])
    button_text : str
        Button label (e.g., "Get ZTF lightcurve")
    button_class : str, optional
        Bootstrap button class (default: "btn-primary")
    form_fields : dict, optional
        Custom form fields (default: {})
    help_text : str, optional
        Brief description (default: "")
    order : int, optional
        Display order (default: 50)
    votable_file : str, optional
        VOTable filename or pattern (e.g., 'ztf.vot' or 'tess_lc_*.vot')
    lc_mag_column : str, optional
        Column name for magnitude (default: None)
    lc_err_column : str, optional
        Column name for error (default: None)
    lc_filter_column : str, optional
        Column name for filter/band (default: None)
    lc_flux_column : str, optional
        Column name for flux (default: None)
    lc_quality_column : str, optional
        Column name for quality flags (default: None)
    lc_color : str, optional
        Plotly display color (default: None)
    lc_mode : str, optional
        'magnitude' or 'flux' (default: None)
    lc_short : bool, optional
        Include in short lightcurves (default: False)
    """
    def decorator(func):
        # Extract source_id from function name (target_xxx -> xxx)
        func_name = func.__name__
        if not func_name.startswith('target_'):
            raise ValueError(f"Survey source function must be named 'target_xxx', got '{func_name}'")

        source_id = func_name.replace('target_', '')

        # Build registry entry
        SURVEY_SOURCES[source_id] = {
            'name': name,
            'short_name': short_name,
            'processing_function': func_name,
            'state_acquiring': state_acquiring,
            'state_acquired': state_acquired,
            'log_file': log_file,
            'output_files': output_files,
            'button_text': button_text,
            'button_class': button_class,
            'form_fields': form_fields or {},
            'help_text': help_text,
            'order': order,
            # Lightcurve metadata
            'votable_file': votable_file,
            'lc_mag_column': lc_mag_column,
            'lc_err_column': lc_err_column,
            'lc_filter_column': lc_filter_column,
            'lc_flux_column': lc_flux_column,
            'lc_quality_column': lc_quality_column,
            'lc_color': lc_color,
            'lc_mode': lc_mode,
            'lc_short': lc_short,
        }

        # Return function unchanged
        return func

    return decorator


def get_survey_source(source_id):
    """Get survey metadata by ID."""
    return SURVEY_SOURCES.get(source_id)


def get_all_survey_sources():
    """Get all survey sources sorted by order."""
    return dict(sorted(SURVEY_SOURCES.items(), key=lambda x: x[1]['order']))


def register_lightcurve_source(
    source_id,
    name,
    short_name,
    votable_file,
    lc_mag_column='mag_g',
    lc_err_column='magerr',
    lc_filter_column=None,
    lc_flux_column=None,
    lc_quality_column=None,
    lc_color='#000000',
    lc_mode='magnitude',
    lc_short=False,
):
    """
    Register a lightcurve-only source (no processing function).

    Use this for sources where lightcurve data exists (e.g., from manual upload
    or external process) but there's no automated acquisition function.

    These sources will:
    - Appear in the lightcurve viewer
    - Be included in combined lightcurve plots
    - Be excluded from "Acquire Everything" batch operations

    Example:
        register_lightcurve_source(
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

    Parameters:
    -----------
    source_id : str
        Source identifier (e.g., 'ps1', 'gaia')
    name : str
        Full display name (e.g., "Pan-STARRS")
    short_name : str
        Short name for displays (e.g., "Pan-STARRS")
    votable_file : str
        VOTable filename or pattern (e.g., 'ps1.vot' or 'ps1_*.vot')
    lc_mag_column : str, optional
        Column name for magnitude (default: 'mag_g')
    lc_err_column : str, optional
        Column name for error (default: 'magerr')
    lc_filter_column : str, optional
        Column name for filter/band (default: None)
    lc_flux_column : str, optional
        Column name for flux (default: None)
    lc_quality_column : str, optional
        Column name for quality flags (default: None)
    lc_color : str, optional
        Plotly display color (default: '#000000')
    lc_mode : str, optional
        'magnitude' or 'flux' (default: 'magnitude')
    lc_short : bool, optional
        Include in short lightcurves (default: False)
    """
    SURVEY_SOURCES[source_id] = {
        'name': name,
        'short_name': short_name,
        'processing_function': None,  # No processing function
        'state_acquiring': None,
        'state_acquired': None,
        'log_file': None,
        'output_files': [],
        'button_text': None,
        'button_class': None,
        'form_fields': {},
        'help_text': '',
        'order': 999,  # Sort to end
        # Lightcurve metadata
        'votable_file': votable_file,
        'lc_mag_column': lc_mag_column,
        'lc_err_column': lc_err_column,
        'lc_filter_column': lc_filter_column,
        'lc_flux_column': lc_flux_column,
        'lc_quality_column': lc_quality_column,
        'lc_color': lc_color,
        'lc_mode': lc_mode,
        'lc_short': lc_short,
    }


def get_survey_ids_for_everything():
    """Get list of survey IDs for 'everything' batch operation."""
    # Exclude 'info' and 'combined' initially, add them at start/end
    # Also exclude sources without processing functions (lightcurve-only sources)
    surveys = [
        k for k in SURVEY_SOURCES.keys()
        if k not in ['info', 'combined']
        and SURVEY_SOURCES[k].get('processing_function') is not None
    ]
    return ['info'] + sorted(surveys, key=lambda k: SURVEY_SOURCES[k]['order']) + ['combined']


def get_combined_lc_rules():
    """Get rules for combined lightcurve plotting (magnitude sources only)."""
    rules = {}
    for source_id, config in SURVEY_SOURCES.items():
        if config.get('lc_mode') == 'magnitude' and config.get('votable_file'):
            rules[source_id] = {
                'name': config['short_name'],
                'filename': config.get('votable_file'),
                'mag': config.get('lc_mag_column'),
                'err': config.get('lc_err_column'),
                'filter': config.get('lc_filter_column'),
                'short': config.get('lc_short', False),
            }
    return rules


def get_output_files(source_id):
    """Get list of output files for a survey source from registry."""
    config = SURVEY_SOURCES.get(source_id)
    return config.get('output_files', []) if config else []


def get_cache_files():
    """Get list of cache files/patterns used by all surveys.

    Cache files live in targets/{id}/cache/ directory and are shared
    across processing runs. Returns glob patterns to match coordinate-based
    and name-based cache files.

    Note: Processing functions use patterns like:
    - dasch_{ra}_{dec}.vot (coordinate-based)
    - kws_{safe_name}.vot (name-based)
    - mastDownload/ directory (TESS)
    """
    cache_patterns = [
        # Glob patterns for coordinate/name-based cache files
        'cache/applause_*.vot',
        'cache/css_*.vot',
        'cache/dasch_*.vot',
        'cache/mmt9_*.vot',
        'cache/kws_*.vot',
        # TESS mastDownload directory
        'cache/mastDownload',
    ]
    # Include .txt versions for VOTable caches
    txt_patterns = [p.replace('.vot', '.txt') for p in cache_patterns if '.vot' in p]
    return cache_patterns + txt_patterns


def get_all_output_files():
    """Get all output files from all survey sources.

    Used by info step to clean up everything when re-run.
    Includes all output files from all sources plus cache files.
    """
    all_files = []

    # Add all output files from all sources
    for source_id, config in SURVEY_SOURCES.items():
        all_files.extend(config.get('output_files', []))

    # Add cache files
    all_files.extend(get_cache_files())

    # Add info-specific files not in output_files
    all_files.extend(['galaxy_map.png'])

    return all_files
