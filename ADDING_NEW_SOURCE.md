# Adding a New Survey Source to LCServer

This guide shows how to add a new astronomical survey data source to LCServer using the decorator-based registration system.

## Overview

Adding a new source takes **2 steps**:

1. Write the processing function with the `@survey_source` decorator, in its own
   module under `lcserver/processing/` - one file per source
2. Import it in `lcserver/processing/__init__.py`, since the package imports its
   modules explicitly rather than discovering them; a module nobody imports
   registers nothing

Everything else (Celery tasks, forms, view handlers, template blocks, the
"Acquire Everything" chain) is **automatically generated** from the decorator.

## Step 1: Implement Processing Function with Decorator

Edit `lcserver/processing/` and add your processing function with the `@survey_source` decorator:

```python
from .surveys import survey_source

# ... existing code ...

# Get Gaia data
@survey_source(
    name='Gaia DR3',
    short_name='Gaia',
    state_acquiring='acquiring Gaia data',
    state_acquired='Gaia data acquired',
    log_file='gaia.log',
    output_files=['gaia.log', 'gaia_lc.png', 'gaia_pm.png'],
    button_text='Get Gaia data',
    button_class='btn-primary',  # Optional, defaults to 'btn-primary'
    form_fields={  # Optional, custom form fields
        'gaia_radius': {
            'type': 'float',
            'label': 'Search radius (arcsec)',
            'initial': 5.0,
            'required': False,
        },
        'gaia_data_release': {
            'type': 'choice',
            'label': 'Data Release',
            'choices': [('DR3', 'Gaia DR3'), ('DR2', 'Gaia DR2')],
            'initial': 'DR3',
            'required': True,
        },
    },
    help_text='ESA Gaia mission photometry and astrometry',
    order=35,  # Display order (1-100)
)
def target_gaia(config, basepath='.', verbose=None, show=False):
    """
    Fetch Gaia DR3 data for the target.

    Parameters:
    -----------
    config : dict
        Target configuration (includes target_ra, target_dec, gaia_radius, etc.)
    basepath : str
        Directory to save output files
    verbose : callable
        Logging function (call with messages to log)
    show : bool
        If True, display plots interactively (for debugging)
    """
    # Your processing code here...
    pass
```

### Decorator Parameter Reference

#### Required Fields

- **`name`** (str): Full display name (e.g., "Zwicky Transient Facility")
- **`short_name`** (str): Short name for messages (e.g., "ZTF")
- **`processing_function`** (str): Function name, as `processing/__init__.py` exports it (e.g. "target_ztf")
- **`state_acquiring`** (str): State while processing (e.g., "acquiring ZTF lightcurve")
- **`state_acquired`** (str): State after success (e.g., "ZTF lightcurve acquired")
- **`log_file`** (str): Log filename (e.g., "ztf.log")
- **`output_files`** (list): Expected output files (e.g., ["ztf.log", "ztf_lc.png"])
- **`button_text`** (str): Button label (e.g., "Get ZTF lightcurve")
- **`button_class`** (str): Bootstrap button class (e.g., "btn-primary", "btn-success")
- **`help_text`** (str): Short description for documentation
- **`order`** (int): Display order (1-100, determines position in UI)

#### Optional Fields

- **`kind`** (str): What the source brings back - `KIND_PHOTOMETRY` (the
  default) or `KIND_SPECTROSCOPY`. The page groups its sections under the two
  headings and badges each with it, and the checkboxes beside *Run everything*
  let a run ask for either on its own. `KIND_ALWAYS` is for a step belonging to
  every run whichever kind it asked for; the info step is the only one.
- **`form_fields`** (dict): Custom form fields (see below)

### Form Fields

Form fields are automatically generated from the `form_fields` dictionary. Two field types are supported:

#### Float Field

```python
'field_name': {
    'type': 'float',
    'label': 'Search radius (arcsec)',
    'initial': 5.0,
    'required': False,
}
```

Generates: `forms.FloatField(label='...', initial=5.0, required=False)`

#### Choice Field

```python
'field_name': {
    'type': 'choice',
    'label': 'Data Release',
    'choices': [('value1', 'Display 1'), ('value2', 'Display 2')],
    'initial': 'value1',
    'required': True,
}
```

Generates: `forms.ChoiceField(label='...', choices=[...], initial='value1', required=True)`

**Note**: For ZTF's color model field, the form uses `RadioSelect` widget. This is a special case handled in the factory.

If you need more complex form fields, you can manually define the form in `forms.py` as a special case (see `TargetInfoForm` for an example).

### Processing Function Implementation

The function decorated by `@survey_source` should follow this pattern:

```python
def target_gaia(config, basepath='.', verbose=None, show=False):
    """
    Fetch Gaia DR3 data for the target.

    Parameters:
    -----------
    config : dict
        Target configuration (includes target_ra, target_dec, gaia_radius, etc.)
    basepath : str
        Directory to save output files
    verbose : callable
        Logging function (call with messages to log)
    show : bool
        If True, display plots interactively (for debugging)
    """
    if verbose:
        verbose("Fetching Gaia DR3 data...")

    # Get target coordinates from config
    ra = config.get('target_ra')
    dec = config.get('target_dec')

    # Get custom parameters from form
    radius = config.get('gaia_radius', 5.0)
    data_release = config.get('gaia_data_release', 'DR3')

    if verbose:
        verbose(f"Search radius: {radius} arcsec")
        verbose(f"Data release: {data_release}")

    # Query Gaia archive
    from astroquery.gaia import Gaia
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')

    # Query Gaia
    if verbose:
        verbose(f"Querying Gaia {data_release} at {coord.to_string('hmsdms')}...")

    if data_release == 'DR3':
        table_name = 'gaiadr3.gaia_source'
    else:
        table_name = 'gaiadr2.gaia_source'

    query = f"""
    SELECT *
    FROM {table_name}
    WHERE CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {ra}, {dec}, {radius/3600.}))=1
    """

    job = Gaia.launch_job_async(query)
    results = job.get_results()

    if verbose:
        verbose(f"Found {len(results)} sources")

    # Save results
    votable_path = os.path.join(basepath, 'gaia.vot')
    results.write(votable_path, format='votable', overwrite=True)

    if verbose:
        verbose(f"Saved VOTable: {votable_path}")

    # Create plots
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: G magnitude vs time
    ax1.errorbar(results['phot_g_mean_mag'], results['phot_g_mean_flux'],
                 yerr=results['phot_g_mean_flux_error'],
                 fmt='o', label='G band')
    ax1.set_xlabel('G magnitude')
    ax1.set_ylabel('Flux')
    ax1.set_title('Gaia G-band photometry')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Proper motion
    ax2.quiver(results['ra'], results['dec'],
               results['pmra'], results['pmdec'])
    ax2.set_xlabel('RA (deg)')
    ax2.set_ylabel('Dec (deg)')
    ax2.set_title('Proper motion')
    ax2.grid(True)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(basepath, 'gaia_lc.png')
    plt.savefig(plot_path, dpi=100)

    if verbose:
        verbose(f"Saved plot: {plot_path}")

    if show:
        plt.show()
    else:
        plt.close()

    # Store results in config
    config['gaia_nsources'] = len(results)
    config['gaia_dr'] = data_release

    if verbose:
        verbose("Gaia processing complete")
```

### Processing Function Guidelines

1. **Function signature**: Must match `target_xxx(config, basepath='.', verbose=None, show=False)`

2. **Config dict**: Contains all target data and form field values
   - Standard fields: `target_name`, `target_ra`, `target_dec`, `target_simbad_type`, etc.
   - Custom fields: Any fields defined in `form_fields` will be in config

3. **Logging**: Call `verbose(message)` to log progress (logs to file and optionally console)
   - Use `verbose(clear=True)` to clear log file at start

4. **File output**: Save files to `basepath` directory
   - Log file: `{source_id}.log` (automatically created)
   - Data files: VOTables, FITS, CSV, etc.
   - Plots: PNG images

5. **Config updates**: Store results in `config` dict (will be saved to database)
   - Use simple types: str, int, float, list, dict
   - Avoid numpy types (use `float(value)` to convert)

6. **Error handling**: Raise exceptions on error (will be caught by task wrapper)

7. **Show parameter**: If True, call `plt.show()` to display plots interactively

## Testing Your Implementation

### Using Management Command (Recommended)

Test synchronously without Celery:

```bash
# Create a test target
python manage.py test_target "Kepler-11" --new -s gaia --verbose

# Test on existing target
python manage.py test_target 42 -s gaia --verbose

# Debug mode (with breakpoints)
python manage.py test_target 42 -s gaia --debug --verbose

# Interactive debugging
python manage.py test_target 42 -s gaia --pdb --verbose
```

### Using Web Interface

1. Start Django development server:
   ```bash
   python manage.py runserver
   ```

2. Start Celery worker:
   ```bash
   ./run_celery.sh
   ```

3. Open browser to http://localhost:8000/

4. Navigate to target detail page

5. Find your new source section and click "Get [Source] data"

## Verification

Your new source should automatically appear in:

✅ **Target detail page** - Form and button to trigger processing

✅ **Management command** - Available as a step choice
```bash
python manage.py test_target --help
# Should show 'gaia' in step choices
```

✅ **"Acquire Everything" button** - Included in batch processing chain

✅ **Queue monitor** - Task shows in `/queue/` with proper name

✅ **State tracking** - Target state updates correctly during processing

## Advanced: Special Cases

### Custom Form Layout

If you need complex form layout beyond simple fields, create a special case in `forms.py`:

```python
def create_survey_form(source_id, survey_config):
    # Special case for custom form
    if source_id == 'gaia':
        class TargetGaiaForm(forms.Form):
            form_type = forms.CharField(initial='target_gaia', widget=forms.HiddenInput())
            # Custom field definitions
            # Custom layout

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.helper = FormHelper()
                # Custom helper configuration

        return TargetGaiaForm

    # ... rest of factory function
```

### Custom View Logic

If you need special handling in the view (like 'info' does for name/title updates), add a special case:

```python
# In views.py, in the action handler
if source_id == 'gaia':
    # Special pre-processing logic
    config['custom_field'] = process_something()
```

### Multiple Output Files

For sources that create many output files (like TESS with per-sector plots):

```python
'output_files': [
    'tess.log',
    'tess_lc_*.vot',    # Wildcards supported
    'tess_lc_*.png',
],
```

### Conditional Requirements

If your source has specific requirements (coordinates, hemisphere, magnitude limits, etc.), check them in the processing function:

```python
def target_gaia(config, basepath='.', verbose=None, show=False):
    # Check coordinates are available
    ra = config.get('target_ra')
    dec = config.get('target_dec')
    if ra is None or dec is None:
        raise ValueError("Gaia requires target coordinates (RA/Dec)")

    # Check hemisphere requirement
    if dec <= -30:
        raise ValueError("This survey only covers northern hemisphere (Dec > -30 deg)")

    # Check magnitude requirement
    if config.get('target_vmag', 99) > 15:
        raise ValueError("Source only reliable for V < 15")

    # ... rest of processing
```

## Example: Complete Working Source

See the MMT9 (Mini-MegaTORTORA) implementation as a complete example:

1. Registry entry: `lcserver/surveys.py` line ~95
2. Processing function: `lcserver/processing/` search for `def target_mmt9`
3. Custom form field: `mmt9_sr` (search radius)

## Troubleshooting

### "Unknown step" error

Check that:
- Entry is in `SURVEY_SOURCES` dict
- Source ID matches action name (e.g., 'gaia' for action 'target_gaia')

### Form doesn't appear

Check that the survey source entry exists in the registry and is properly configured.

### Task doesn't run

Check that:
- Celery worker is running (`./run_celery.sh`)
- No Python syntax errors in processing function
- Function name matches `processing_function` in registry

### State doesn't update

Check that:
- `state_acquiring` and `state_acquired` are defined in registry
- No exception raised in processing function
- Processing function completes successfully

## Questions?

See the full implementation in:
- `lcserver/surveys.py` - Registry and helper functions
- `lcserver/celery_tasks.py` - Task factory and registration
- `lcserver/forms.py` - Form factory and generation
- `lcserver/views.py` - View handlers
- `lcserver/processing/` - All processing functions
- `REFACTORING_SUMMARY.md` - Complete refactoring documentation
