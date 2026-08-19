"""Survey source registry for LCServer.

This module provides the @survey_source decorator for registering data sources.
Survey sources are registered by decorating processing functions in processing.py.

Adding a new survey source requires:
1. Implementing processing function in processing.py
2. Decorating it with @survey_source(...) with metadata
"""

import os
import fnmatch

# Global registry populated by @survey_source decorator
SURVEY_SOURCES = {}


# How a band relates to what the telescope actually measured.
#
#   native      - the measurement as the survey reports it, in its own band.
#                 Always preferred for display, and shown by default.
#   calibrated  - the survey's raw magnitudes sit in a zero point that moves
#                 with the colour of the star (ZTF's per-epoch clrcoeff, the
#                 per-plate colour term of APPLAUSE), so they are only
#                 comparable between epochs after the correction. Shown by
#                 default, as the raw values would be misleading.
#   derived     - obtained from another band through an assumed, constant
#                 colour. Useful for stitching surveys onto one scale, but it
#                 is a model rather than a measurement, so it is hidden until
#                 asked for.
BAND_NATIVE = 'native'
BAND_CALIBRATED = 'calibrated'
BAND_DERIVED = 'derived'

# The kinds a viewer shows unless the user says otherwise
BAND_KINDS_SHOWN = (BAND_NATIVE, BAND_CALIBRATED)


def band(label, mag, err, kind=BAND_NATIVE, filter_column=None, filter_value=None,
         color=None, note=None, combined=False):
    """One displayable band of a source.

    Parameters
    ----------
    label : str
        Band name, appended to the source name to label the series ('V', 'g').
    mag, err : str
        Columns holding the magnitude and its uncertainty.
    kind : str
        One of BAND_NATIVE, BAND_CALIBRATED, BAND_DERIVED - see above.
    filter_column, filter_value : str, optional
        Column to select this band's rows by, and the value to match. Omitted
        when the whole table belongs to a single band.
    color : str, optional
        Plotly colour. Falls back to the colour of the source.
    note : str, optional
        Short description of where a calibrated or derived band comes from,
        shown to the user next to the series.
    combined : bool, optional
        Whether this band belongs on the combined multi-survey light curve,
        which is drawn on one common g scale. Only bands that are in g - as
        measured, or converted there - set it; a source's own V, or W1, or an
        Ic does not, however good a measurement it is. A source may set it on
        more than one band, where it reaches g by more than one route: ZTF
        contributes both the g it measured and the r it converted.
    """
    return {
        'label': label,
        'mag': mag,
        'err': err,
        'kind': kind,
        'filter_column': filter_column,
        'filter_value': filter_value,
        'color': color,
        'note': note,
        'combined': combined,
    }


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
    # What the source is here to bring back. A run that ends without any of
    # these found nothing, however cleanly it finished - which is how a step
    # that came back empty is told apart from one that came back with data.
    # Defaults to the lightcurve or the spectra the source declares below;
    # give an empty list where the step is not about acquiring data of its own.
    data_files=None,
    # Lightcurve metadata
    votable_file=None,
    lc_bands=None,        # List of band() entries - what the viewer displays,
                          # and which of them the combined g-band curve is
                          # built from, through band(combined=True)
    lc_mag_column=None,   # Fallback single column, for a source with no bands
    lc_err_column=None,
    lc_filter_column=None,
    lc_flux_column=None,
    lc_quality_column=None,
    lc_color=None,
    lc_mode=None,  # 'magnitude' or 'flux'
    lc_short=False,
    # Flux sources arrive one file per observing segment, which the mission
    # names for itself - a TESS sector, a Kepler quarter, a K2 campaign
    lc_segment_name='Segment',
    # Where one source spans phases that number their segments independently,
    # the letter its filenames carry maps to what that phase calls a segment
    lc_segment_prefixes=None,
    lc_color_palette=None,
    # Spectra, for the spectral viewer. A source writing them says only where
    # they are; what is in them is fixed, and the same for every source:
    #
    #   wavelength   Angstrom
    #   flux         erg/s/cm2/A, a flux per unit wavelength
    #   flux_error   the same, where the survey gives an uncertainty
    #
    # The surveys agree on none of that between them - Gaia XP publishes
    # W/nm/m2 against nanometres, LAMOST and DESI units of 1e-17 erg/s/cm2/A,
    # SPHEREx uJy against microns, which is per unit frequency and so differs
    # in the shape of the curve and not merely its height - so each converts
    # on the way out and the files can be read against each other, and
    # exported, as they stand. See processing/utils.py for the conversion.
    spectrum_files=None,             # glob pattern, one table per spectrum
    # The same, for measurements that are not a curve. SPHEREx builds its
    # spectrum out of several hundred separate exposures, each at one
    # wavelength, and the individual ones are worth seeing about the binned
    # curve - the scatter between them at one wavelength is the target's own
    # variability between visits, which the curve averages away. Drawn as
    # points rather than joined, a line through them being a zigzag and not a
    # spectrum, and normalised with the curve they belong to rather than on
    # their own, so that the two sit on top of each other.
    spectrum_points=None,            # glob pattern, drawn as points
    # Spectra that are there to be reached for rather than read by default.
    # The SED writes both the catalogues it was told to trust and every
    # catalogue at the position; the second is the larger and the noisier, and
    # is worth having without its being the first thing seen.
    spectrum_hidden=None,            # glob pattern, loaded but unticked
    spectrum_label=None,             # what to call them, if not the short name
    spectrum_color=None,             # a source with one spectrum
    spectrum_palette=None,           # a source with several, told apart by shade
    # Config keys this source writes for others to convert with. A source that
    # provides one runs before the sources that read it, rather than alongside
    # them - see run_target_steps().
    provides_config=None,
    # Set where a source deletes what every other source produced, so that
    # their recorded states stop describing anything that is still on disk
    clears_other_sources=False,
    # Template metadata
    template_layout='simple',  # 'simple', 'with_cutout', 'complex', 'custom'
    requires_coordinates=True,  # False for name-based sources like KWS
    declination_min=None,       # Minimum declination (e.g., -30 for APPLAUSE)
    declination_max=None,       # Maximum declination
    show_cutout=False,          # Show HiPS/SkyView cutout image
    cutout_hips=None,           # HiPS survey for cutout: a CDS id, or the base URL of any HiPS
    cutout_name=None,           # Caption for the cutout, for when the HiPS is named by URL
    # Rendering of a single-channel HiPS. Colour HiPS carry their own palette
    # and are left alone, so these default to letting the service decide.
    cutout_cmap=None,           # Matplotlib colormap, e.g. "Blues_r"
    cutout_min_cut=None,        # Low cut, e.g. "2.5%"
    cutout_max_cut=None,        # High cut, e.g. "99.5%"
    cutout_skyview=None,        # SkyView survey for cutout (e.g., "TESS")
    cutout_fov=0.03,            # Field of view for cutout (degrees)
    show_color_mag=False,       # Show color-magnitude diagram
    color_mag_file=None,        # Color-magnitude diagram filename
    main_plot=None,             # Main lightcurve plot (auto-detected if None)
    additional_plots=None,      # Additional plots to display (list or pattern)
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
    lc_bands : list, optional
        Bands the light curve viewer offers for this source, as band() entries.
        A source keeps every band it measures, so that the original photometry
        stays reachable; conversions between bands are additional entries of
        kind BAND_DERIVED rather than replacements. The entries carrying
        combined=True are what the combined multi-survey curve draws.
    lc_mag_column : str, optional
        Single magnitude column, for a source that declares no bands at all -
        get_source_bands() makes one band of it. Both display and the combined
        curve go through lc_bands otherwise. (default: None)
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
            'data_files': data_files,
            # Lightcurve metadata
            'votable_file': votable_file,
            'lc_bands': lc_bands or [],
            'lc_mag_column': lc_mag_column,
            'lc_err_column': lc_err_column,
            'lc_filter_column': lc_filter_column,
            'lc_flux_column': lc_flux_column,
            'lc_quality_column': lc_quality_column,
            'lc_color': lc_color,
            'lc_mode': lc_mode,
            'lc_short': lc_short,
            'lc_segment_name': lc_segment_name,
            'lc_segment_prefixes': lc_segment_prefixes or {},
            'lc_color_palette': lc_color_palette,
            'spectrum_files': spectrum_files,
            'spectrum_points': spectrum_points,
            'spectrum_hidden': spectrum_hidden,
            'spectrum_label': spectrum_label,
            'spectrum_color': spectrum_color,
            'spectrum_palette': spectrum_palette,
            'provides_config': provides_config or [],
            'clears_other_sources': clears_other_sources,
            # Template metadata
            'template_layout': template_layout,
            'requires_coordinates': requires_coordinates,
            'declination_min': declination_min,
            'declination_max': declination_max,
            'show_cutout': show_cutout,
            'cutout_hips': cutout_hips,
            'cutout_name': cutout_name,
            'cutout_cmap': cutout_cmap,
            'cutout_min_cut': cutout_min_cut,
            'cutout_max_cut': cutout_max_cut,
            'cutout_skyview': cutout_skyview,
            'cutout_fov': cutout_fov,
            'show_color_mag': show_color_mag,
            'color_mag_file': color_mag_file,
            'main_plot': main_plot or f'{source_id}_lc.png',  # Auto-detect main plot
            'additional_plots': additional_plots or [],
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
    lc_bands=None,
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
        'lc_bands': lc_bands or [],
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


# Colours for the combined light curve, as (darker, lighter) pairs - one pair
# per source, the darker for the band it reaches g by first and the lighter for
# a second route to the same place, so that ZTF's measured g and its converted
# r read as one survey seen twice rather than as two.
#
# The combined figure cannot use the colours the bands declare for themselves.
# Those are picked per source, to tell one source's bands apart in the viewer,
# and shades of one hue serve that well; here every source appears at once and
# only its g band does, so the shades collide across sources instead. Assigned
# by position in the registry, so that a source keeps its colour between the
# short figure and the long one, and between one target and the next.
COMBINED_COLOR_PAIRS = [
    ('#1f77b4', '#aec7e8'), ('#ff7f0e', '#ffbb78'), ('#2ca02c', '#98df8a'),
    ('#d62728', '#ff9896'), ('#9467bd', '#c5b0d5'), ('#8c564b', '#c49c94'),
    ('#e377c2', '#f7b6d2'), ('#7f7f7f', '#c7c7c7'), ('#bcbd22', '#dbdb8d'),
    ('#17becf', '#9edae5'), ('#393b79', '#6b6ecf'), ('#637939', '#b5cf6b'),
    ('#8c6d31', '#e7ba52'), ('#843c39', '#d6616b'), ('#7b4173', '#ce6dbd'),
]


def get_combined_series(short=False):
    """The series the combined light curve draws, in registry order.

    One entry per band a source has marked as belonging on the common g scale,
    rather than one per source. That distinction is the whole point: a source
    reaching g by two routes contributes both - ZTF its measured g and its r
    converted through the colour it measured itself - and a source with
    nothing in g contributes none, which is how WISE's W1 and KWS's Ic stay
    off an axis they would only mislead on.

    Each entry says which rows to take as well as which column, so that a
    table holding several bands at once gives up only the rows the band was
    measured in.
    """
    series = []
    nsources = 0

    for source_id, config in get_all_survey_sources().items():
        if not config.get('votable_file'):
            continue

        bands = [b for b in (config.get('lc_bands') or []) if b.get('combined')]
        if not bands:
            continue

        # Counted whether or not this source is drawn, so that dropping to the
        # short figure does not recolour the sources that remain
        pair = COMBINED_COLOR_PAIRS[nsources % len(COMBINED_COLOR_PAIRS)]
        nsources += 1

        if short and not config.get('lc_short'):
            continue

        for i, b in enumerate(bands):
            series.append({
                'source_id': source_id,
                'name': config['short_name'],
                'filename': config['votable_file'],
                'label': f"{config['short_name']} {b['label']}".strip(),
                'mag': b['mag'],
                'err': b['err'],
                'filter_column': b.get('filter_column'),
                'filter_value': b.get('filter_value'),
                'color': pair[min(i, len(pair) - 1)],
                'kind': b['kind'],
            })

    return series


def get_output_files(source_id):
    """Get list of output files for a survey source from registry."""
    config = SURVEY_SOURCES.get(source_id)

    files = config.get('output_files', []) if config else []
    # Also .txt versions of .vot files
    txt_files = [p.replace('.vot', '.txt') for p in files if '.vot' in p]

    return files + txt_files


def get_data_files(source_id):
    """The patterns a source's own data lands in, as globs.

    What it declared, or else the lightcurve or the spectra it is registered
    with, or else its main plot - the last being all a source like Combined
    leaves behind. An empty list means the step brings back no data of its
    own, and so can never be said to have come back without any.
    """
    config = SURVEY_SOURCES.get(source_id) or {}

    if config.get('data_files') is not None:
        files = config['data_files']
    else:
        # spectrum_points before main_plot, so that a source publishing only
        # measurements - an SED merged out of catalogues has no curve to draw
        # - is still judged by its data rather than by whether a picture of it
        # came out
        files = (config.get('votable_file') or config.get('spectrum_files')
                 or config.get('spectrum_points') or config.get('main_plot')
                 or [])

    return [files] if isinstance(files, str) else list(files)


def source_has_data(source_id, files):
    """Whether a source's data is among the file names given."""
    patterns = get_data_files(source_id)

    if not patterns:
        return True

    return any(fnmatch.filter(files, _) for _ in patterns)


def get_source_states(target):
    """How every source stands, with the empty-handed ones told apart.

    A source that ran cleanly and found nothing is recorded as done - nothing
    went wrong, there was simply nothing there - so the two are separated here,
    by what the source left on disk, rather than stored. That way it holds for
    the targets acquired before this was drawn, and stops holding if the files
    are cleaned out from under it.
    """
    states = dict(target.source_states or {})

    try:
        files = os.listdir(target.path())
    except OSError:
        return states

    return {source_id: ('empty' if state == 'done'
                        and not source_has_data(source_id, files) else state)
            for source_id, state in states.items()}


def get_cache_files():
    """Get list of cache files/patterns used by all surveys.

    Cache files live in targets/{id}/cache/ directory and are shared
    across processing runs. Returns glob patterns to match cache files.
    """
    cache_patterns = [
        # Glob patterns for coordinate/name-based cache files
        'cache/*.vot',
        # lightkurve downloads, a directory per source
        'cache/mast_*',
    ]

    return cache_patterns


def get_all_output_files(cache=False):
    """Get all output files from all survey sources, and optionally a cache.

    Used by info step to clean up everything when re-run.
    Includes all output files from all sources plus cache files.
    """
    all_files = []

    # Add all output files from all sources
    for source_id, config in SURVEY_SOURCES.items():
        all_files.extend(config.get('output_files', []))

    if cache:
        # Add cache files
        all_files.extend(get_cache_files())

    # Add info-specific files not in output_files
    all_files.extend(['galaxy_map.png'])

    return all_files


def get_source_bands(config):
    """Bands a source publishes, for display.

    Sources that have not declared any fall back to the single column the
    combined light curve is built from, which reproduces the earlier
    behaviour of showing one series per source.
    """
    bands = config.get('lc_bands')
    if bands:
        return bands

    mag = config.get('lc_mag_column')
    err = config.get('lc_err_column')
    if not mag or not err:
        return []

    return [band('', mag, err, BAND_NATIVE, color=config.get('lc_color'))]


# Cache filenames are built by the processing modules from a source-specific
# prefix; this maps them back, so that a cache entry can be shown next to the
# survey it belongs to. Longest prefix wins.
CACHE_PREFIXES = {
    'asas_': 'asas',
    'applause_': 'applause',
    'css_': 'css',
    'dasch_': 'dasch',
    'kws_': 'kws',
    'mmt9_': 'mmt9',
    'ptf_': 'ptf',
    'bgds_': 'bgds',
    'ztf_raw_': 'ztf',
    'wise_': 'wise',
    'asas3_': 'asas3',
    'wasp_': 'wasp',
    'omc_': 'omc',
    'nsvs_': 'nsvs',
    'hipparcos_': 'hipparcos',
    'sdss_': 'sdss',
    'lamost_': 'lamost',
    'apogee_': 'apogee',
    'eso_': 'eso',
    'desi_': 'desi',
    'spherex_': 'spherex',
    'sed_': 'sed',
    # Everything the info step collects
    'simbad_': 'info',
    'gaiadr3_phot_': 'info',
    'gaiadr3_dist_': 'info',
    'gaiadr3syn_': 'info',
    'ps1_': 'info',
    'dust_': 'info',
    'dustext_': 'info',
    'gaiaxp_': 'info',
    'skymapper_': 'info',
    # What MAST was asked for, and what it sent back: the product search of
    # each, and the downloads themselves in a directory per source
    'tess_search_': 'tess',
    'kepler_search_': 'kepler',
    'mast_tess': 'tess',
    'mast_kepler': 'kepler',
}


def cache_source_for(name):
    """Which source a cache entry belongs to, or None if it is unrecognized."""
    for prefix in sorted(CACHE_PREFIXES, key=len, reverse=True):
        if name.startswith(prefix):
            return CACHE_PREFIXES[prefix]

    # Older entries were named after the source alone, with no coordinates
    stem = name.split('.')[0].split('_')[0]
    if stem in SURVEY_SOURCES:
        return stem

    return None

