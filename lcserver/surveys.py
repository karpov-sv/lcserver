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


def get_survey_ids_for_everything():
    """Get list of survey IDs for 'everything' batch operation."""
    # Exclude 'info' and 'combined' initially, add them at start/end
    surveys = [k for k in SURVEY_SOURCES.keys() if k not in ['info', 'combined']]
    return ['info'] + sorted(surveys, key=lambda k: SURVEY_SOURCES[k]['order']) + ['combined']
