# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LCServer** ("Lightcurves for the Web") is a Django web application for acquiring and analyzing astronomical lightcurves from multiple sky surveys. Users can search for celestial objects and automatically retrieve multi-wavelength time-series photometry from ZTF, ASAS-SN, TESS, DASCH, APPLAUSE, and other surveys. The system uses Celery for background processing and generates visualizations and data products.

## Development Commands

### Running the Application

```bash
# Start Django development server
python manage.py runserver

# Start Celery worker with auto-reload (required for background tasks)
./run_celery.sh

# Or manually (includes MacOS compatibility fix):
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
watchmedo auto-restart -d lcserver --recursive -p '**/*.py' -- \
  python3 -m celery -- -A lcserver worker --loglevel=info
```

### Database Management

```bash
# Apply migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Make migrations after model changes
python manage.py makemigrations
```

### Testing & Debugging

```bash
# Test processing steps synchronously (without Celery)
python manage.py test_target 42 -s info --verbose
python manage.py test_target "Kepler-11" --new -s all --verbose

# Interactive debugging with pdb
python manage.py test_target 42 -s info --pdb --verbose

# IDE debugging (PyCharm, VSCode)
python manage.py test_target 42 -s info --debug --verbose

# List targets
python manage.py targets --list
python manage.py targets --info 42

# System maintenance and checks
python manage.py maintenance --check
python manage.py maintenance --stats
python manage.py maintenance --fix-stale
```

See the *Management commands* section of [README.md](README.md) for complete
documentation, including the `sedgrid` command that builds the model grids.
See [DEBUGGING.md](DEBUGGING.md) for comprehensive debugging guide.

### Static Files

```bash
# Collect static files for production
python manage.py collectstatic
```

### Dependencies

```bash
# Install requirements
pip install -r requirements.txt
```

**Note**: Redis must be running on `localhost:6379` for Celery to function. The application uses database `1` for broker, results, and cache.

## Architecture

### High-Level Structure

LCServer follows a **Django + Celery** architecture where:

1. **Django Web App** handles user interface, authentication, and database
2. **Celery Workers** execute long-running astronomical data acquisition tasks
3. **Redis** serves as message broker, result backend, and cache
4. **Filesystem** stores per-target data products in `targets/{id}/` directories

### Request Flow

```
User submits form → Django view → Celery task queued → Background worker executes
                                                      ↓
User polls task status ← AJAX updates ← Task writes logs/files → Database updated
```

### Key Components

#### Models (`lcserver/models.py`)

**Target Model** - Core entity representing an astronomical target:
- `name`: Target identifier (coordinates or catalog name like "Kepler-11")
- `state`: Processing state ('initial', 'info acquired', 'ztf acquired', 'failed', etc.)
- `celery_id`: Tracks currently running task (NULL when idle)
- `celery_chain_ids`: JSONField list of all task IDs in a chain (for tracking/cancellation)
- `celery_pid`: Process ID of the currently running worker (cleared to avoid accidental termination in thread pool mode)
- `config`: JSONField storing parameters and results from all processing stages
- `user`: Foreign key for multi-tenant support
- `path()`: Returns `targets/{id}/` directory for file storage

**Signals**:
- `post_save`: Auto-creates target directory on creation
- `pre_delete`: Cleans up filesystem when target deleted

#### Views (`lcserver/views.py`)

**Main endpoints**:
- `index()`: Homepage with target creation form
- `targets(id=None)`:
  - Without ID: List/filter all targets
  - With ID: Target detail page with processing forms
  - Handles form submissions to trigger Celery tasks
- `target_download()` / `target_preview()`: Serve generated files

**Permission model**:
- Owners can modify their targets
- Staff users can access all targets and admin queue
- Non-authenticated users have read-only or no access

**File browsing endpoints**:
- `target_files(id, path='')`: Browse files within target directory
  - Lists directories and files with type detection
  - Supports viewing text files, FITS files, images, tables (VOTable/Parquet)
  - Breadcrumb navigation for nested directories
  - Permission check: owner or staff only
- `list_files(path='', base=...)`: Generic file browser function
  - Detects MIME types and displays appropriate preview
  - Modes: list (directory), text, image, table, fits, download
- `preview(path, ...)`: Generate preview images for FITS files
  - Uses matplotlib to render FITS data with percentile scaling
  - Cached for 15 minutes via `@cache_page`
- `download(path, attachment=True, ...)`: Serve files with security checks
  - Uses `sanitize_path()` to prevent directory traversal attacks

**Interactive light curve viewer** (`lcserver/views_lightcurve.py`):
- `target_lightcurve(id)`: Plotly-based interactive viewer
  - Loads all available .vot files (ASAS-SN, ZTF, Pan-STARRS, DASCH, APPLAUSE)
  - Toggle individual data sources on/off
  - Adjust magnitude offsets for each source
  - Interactive zoom, pan, and hover with Plotly.js
  - No background tasks - loads data directly and renders client-side
  - Permission check: owner or staff only

**Spectral energy distribution and SED fitting** (`lcserver/views_spectrum.py`,
`lcserver/processing/sedfit.py`):
- `target_spectrum(id)`: the SED viewer - catalogue photometry, any Gaia XP
  spectrum, the model fitted to them, and the residuals
- The fit interpolates a grid of model atmospheres at (Teff, log g, [Fe/H]),
  reddens it, dilutes it by (R/d)^2 and samples the posterior with dynesty.
  The posterior is kept as whole rows rather than per-parameter marginals, so
  the temperature/extinction degeneracy survives into the answer
- Points may be added or dropped by hand; hand-added ones live in their own
  file so re-running the SED step does not discard them
- The grids are HDF5 cubes in the directory `SEDFIT_GRIDS` names, one file per
  grid, built by the `sedgrid` management command

**Standalone utility pages**, tied to no target:
- `/cutouts/` (`views_cutouts.py`) - image cutouts by position
- `/passbands/` (`views_passbands.py`) - the photometric conversions the
  combined lightcurve is built on, and the transmission curves behind them
- `/models/` (`views_models.py`, `processing/gridstats.py`) - what the model
  grids say: parameters against colours and magnitudes, the model spectra, and
  which bands each grid covers

**URL patterns for file browsing and visualization**:
- `/targets/{id}/lightcurve/` - Interactive light curve viewer
- `/targets/{id}/files/` - Browse target root directory
- `/targets/{id}/files/{path}` - Browse subdirectory or view file
- `/targets/{id}/preview/{path}` - FITS preview image
- `/targets/{id}/view/{path}` - View file inline (no download prompt)
- `/targets/{id}/download/{path}` - Download file as attachment
- `/targets/{id}/spectrum/` - SED viewer and photosphere fitting

#### Forms (`lcserver/forms.py`)

Three hand-written crispy-forms classes:
- `TargetNewForm`: Create new target
- `TargetsFilterForm`: Filter the target list
- `TargetsActionsForm`: Bulk cleanup and delete

Every per-survey form is built from that source's `form_fields` in its
`@survey_source` decorator rather than declared here. All forms POST to
`/targets/{id}/` with a hidden `form_type` field identifying which was
submitted.

#### Celery Tasks (`lcserver/celery_tasks.py`)

One task per registered source, none of them written by hand.
`create_survey_task()` builds it and the loop over `surveys.SURVEY_SOURCES`
registers one for every source declaring a processing function, named
`lcserver.celery_tasks.task_{source_id}`. Named references - `task_info`,
`task_ztf` and the rest - are bound afterwards for callers that want them.

The helper tasks below (`task_finalize`, `task_set_state`, ...) are the only
ones written out.

**Task pattern** (improved with TaskProcessContext):
```python
@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_xxx(self, id, finalize=True):
    with TaskProcessContext(self, id, finalize=finalize) as ctx:
        if ctx.cancelled:
            return

        target = ctx.target
        basepath = ctx.basepath
        config = target.config

        log = partial(processing.print_to_file,
                      logname=os.path.join(basepath, 'xxx.log'))
        log(clear=True)

        try:
            processing.target_xxx(config, basepath=basepath, verbose=log)
            target.state = 'xxx acquired'
        except:
            import traceback
            log("\nError!\n", traceback.format_exc())
            target.state = 'failed'

        fix_config(config)
        # Context manager handles finalize and save
```

**TaskProcessContext** benefits (thread pool safe):
- Cancellation detection before execution (checks if `celery_id` was cleared)
- Automatic cleanup of `celery_pid` field to prevent accidental worker termination
- Finalization control for chain execution
- Proper state management on failure
- Chain breaking on error by clearing `celery_id`
- Safe for both thread pool and prefork execution models

**Thread Pool Safe Architecture**:

The implementation is designed to work safely in thread pool mode, where traditional process group operations would terminate the entire worker process:

- **Cooperative Cancellation**: Tasks check `target.celery_id` on entry and exit early if it's been cleared
- **No Process Groups**: Avoids `os.setpgrp()` and process group kills that would affect the worker
- **Soft Revocation**: Uses `control.revoke(task_id, terminate=False)` to request graceful shutdown
- **PID Field Cleanup**: Clears `celery_pid` on task entry to prevent accidental termination attempts
- **Chain Breaking**: Failed or cancelled tasks clear `celery_id` to stop subsequent chain execution

This design allows external processes spawned by tasks to continue running (they must be managed by the processing code itself), but ensures the worker process remains stable.

**Helper tasks**:
- `task_finalize()`: Marks task as complete and clears chain IDs
- `task_set_state()`: Updates target state
- `task_break_if_failed()`: Breaks chain if target was cancelled
- `kill_task_processes()`: No-op in thread pool mode (killing process groups would terminate the entire worker)

**Task chaining**: The "Acquire Everything" button builds its canvas in
`run_target_steps()`, which acquires the sources together rather than in turn:

```python
chain(
    task_info,                      # gates the run - everything needs its coordinates
    task_ztf,                       # declares provides_config, so it precedes its readers
    chord(
        group(task_asas, task_tess, task_dasch, ...),
        chain(task_combined, task_finalize),
    ),
).apply_async()
```

The prologue is derived from the registry: `GATING_STEPS`, plus any source
declaring `provides_config` (a config key that other sources convert their
photometry with - ZTF's `g_minus_r` is read by five of them). Everything else
runs in the group; `combined` is the chord callback, as it reads what all of
them wrote.

A source that fails does not stop the others - it records itself in
`source_states` and `task_finalize` reports how many failed. Only cancellation,
or a gating step failing, ends a run early. How many sources actually run at
once is `worker_concurrency` (default 4), not the size of the group.

#### Processing (`lcserver/processing/`)

A package with a module per source - `ztf.py`, `asas.py`, `dasch.py` and some
thirty others - plus `utils.py` for what they share and `sedfit.py` for the SED
fitting. `processing/__init__.py` re-exports each `target_xxx()`, so
`processing.target_ztf` works regardless of which module it lives in. Each
`target_xxx()` function:
- Queries external APIs/services (SIMBAD, Vizier, ZTF archive, TESS, etc.)
- Generates plots using matplotlib
- Saves data products to `targets/{id}/`
- Updates `config` dict with results
- Logs progress to `{stage}.log` file

**Key dependencies**:
- `astropy`: FITS, coordinates, tables, time handling
- `astroquery`: SIMBAD and Vizier queries
- `ztfquery`: ZTF survey API
- `lightkurve`: TESS data access
- `skypatrol`: DASCH/APPLAUSE access
- `george`: Gaussian Process smoothing
- `scipy`: Scientific computing

**Helper functions**:
- `print_to_file()`: Dual logging (console + file)
- `cleanup_paths()`: Remove files by pattern (useful before reprocessing)
- `gaussian_smoothing()`: GP-based lightcurve smoothing
- `parse_votable_lenient()`: Parse VOTables with malformed XML (handles undefined entities)

#### Queue Monitoring (`lcserver/views_celery.py`)

Provides `/queue/` interface for viewing active/scheduled tasks with improved chain management:
- Lists all Celery tasks (active, reserved, scheduled)
- Individual task status at `/queue/{task_id}/`
- Chain position tracking: shows "3/7" for tasks in a chain
- Admin actions:
  - `terminatealltasks`: Terminates all running tasks with proper chain revocation
  - `cleanuplinkedtasks`: Clears stale celery_id references
  - `terminatetask`: Terminates a specific task chain (uses SIGTERM for graceful shutdown)
- JSON endpoint `/queue/{task_id}/state` for AJAX polling

**Thread pool safe chain handling**:
- `find_target_by_chain_id()`: Locates targets by any chain task ID
- `revoke_task_chain()`: Cooperatively revokes all tasks in a chain
- Uses `terminate=False` with SIGTERM signal for cooperative cancellation
- Clears `celery_id` to signal cancellation to TaskProcessContext
- Safe for thread pool execution (avoids process group operations that would kill the worker)

### Data Storage

**Database** (SQLite by default):
- `lcserver_target`: Main target table
- Django auth tables: `auth_user`, etc.
- Celery results (optional): `django_celery_results_*`

**Filesystem** (`targets/` directory):
```
targets/
├── 1/                      # Target ID 1
│   ├── info.log           # Coordinate resolution log
│   ├── galaxy_map.png     # Galactic position plot
│   ├── ztf_lc.png         # ZTF lightcurve
│   ├── ztf_color_mag.png  # ZTF color-magnitude diagram
│   ├── asas_lc.png        # ASAS-SN lightcurve
│   ├── tess_lc_*.png      # TESS per-sector plots
│   ├── dasch_lc.png       # DASCH historical lightcurve
│   ├── applause_lc.png    # APPLAUSE lightcurve
│   ├── combined_lc.png    # Combined multi-survey plot
│   └── *.vot              # VOTable catalog caches
├── 2/
└── ...
```

Files are served via `target_download()` and `target_preview()` views with security checks.

### State Machine

Target processing follows a state machine:
```
created → acquiring info → info acquired → acquiring {survey} → {survey} acquired → ...
                                                                                     ↓
                                                                                  completed/failed
```

- `celery_id` field is set when task running, NULL when idle
- Views check task state via Celery API before allowing resubmission
- AJAX polling (`updater.js`) checks task status every 3 seconds

### Configuration

**Environment variables** (via `python-decouple`):
- `SECRET_KEY`: Django secret (required for production)
- `DEBUG`: Debug mode (default: False)
- `TARGETS_PATH`: Base directory for target data (default: 'targets/')

**Celery settings** (`lcserver/settings.py`):
```python
CELERY_BROKER_URL = 'redis://localhost/1'
CELERY_RESULT_BACKEND = 'redis://localhost/1'
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 minutes
```

**Celery app configuration** (`lcserver/celery.py`):
```python
app.conf.update(
    task_reject_on_worker_lost=True,      # Re-queue if worker crashes
    task_track_started=True,              # Track task start state
    worker_prefetch_multiplier=1,         # One task per worker at a time
    broker_transport_options={
        "visibility_timeout": 3600,       # 1 hour timeout
    },
    worker_term_signal='SIGTERM',         # Graceful shutdown
)
```

## Adding a New Survey Source

A source is one module in `lcserver/processing/` with one decorated function.
The Celery task, the form, the button, the view action, the template block and
the "Acquire Everything" chain are all generated from the decorator - none of
them is written by hand, and adding any of them by hand is the old way and
wrong.

1. **Write the module**: `lcserver/processing/xyz.py`

   ```python
   from ..surveys import survey_source

   @survey_source(
       name='XYZ Survey',
       short_name='XYZ',
       state_acquiring='acquiring XYZ lightcurve',
       state_acquired='XYZ lightcurve acquired',
       log_file='xyz.log',
       output_files=['xyz.log', 'xyz_lc.png', 'xyz.vot'],
       button_text='Get XYZ lightcurve',
       order=50,
       # What the lightcurve viewer is to make of what it writes
       votable_file='xyz.vot',
       lc_bands=[...],          # surveys.band() entries
   )
   def target_xyz(config, basepath='.', verbose=None, show=False):
       # Query the survey, plot, write basepath/xyz.vot and xyz_lc.png
   ```

   See [ADDING_NEW_SOURCE.md](ADDING_NEW_SOURCE.md) for every option the
   decorator takes, and `processing/ptf.py` for a short worked example.

2. **Export it**: add the import to `lcserver/processing/__init__.py`

   ```python
   from .xyz import target_xyz
   ```

   The package imports its modules explicitly rather than discovering them, so
   a module nobody imports registers nothing and the source simply will not
   appear.

Two decorator options decide where the source runs in the chain. `GATING_STEPS`
and anything declaring `provides_config` run in the prologue, before the rest;
everything else runs in the group. A source that needs ZTF's `g_minus_r` is a
reader of it and belongs in the group.

## Testing

### Management Commands for Testing

Four management commands are available for testing and debugging:

1. **test_target** - Synchronous processing testing (bypasses Celery)
   ```bash
   # Test specific step with verbose output
   python manage.py test_target 42 -s ztf --verbose

   # Create new target and test full pipeline
   python manage.py test_target "Kepler-11" --new -s all --verbose

   # Debug with interactive plots
   python manage.py test_target 42 -s ztf --show --verbose
   ```

2. **targets** - Target management
   ```bash
   # List all targets
   python manage.py targets --list

   # Show detailed info
   python manage.py targets --info 42

   # Filter by state or user
   python manage.py targets --list --state failed --user admin
   ```

3. **maintenance** - System maintenance
   ```bash
   # Check system status
   python manage.py maintenance --check

   # Show statistics
   python manage.py maintenance --stats

   # Fix stale tasks
   python manage.py maintenance --fix-stale
   ```

4. **sedgrid** - The model atmosphere grids on disk
   ```bash
   # What the grid directory holds
   python manage.py sedgrid --list

   # Build a cube from a downloaded model set. --ingest-cdbs reads an atlas
   # in the layout STScI distributes them in, which is where ck04 and kurucz
   # come from; there are also --ingest-btsettl, --ingest-bosz,
   # --ingest-tlusty, --ingest-koester and --ingest-powr
   python manage.py sedgrid --ingest-cdbs ck04models --to data/grids/
   ```

See the *Management commands* section of [README.md](README.md) for complete
documentation.

### Manual Testing Workflow

1. Create target via web UI or command line
2. For debugging, use synchronous testing:
   ```bash
   python manage.py test_target 42 -s info --verbose
   ```
3. Monitor task execution via `/queue/` for Celery tasks
4. Check logs in `targets/{id}/*.log`
5. Verify plots generated correctly

### Common Test Targets

- "Kepler-11" - Well-known exoplanet host star
- "M31" - Andromeda Galaxy
- "HD 209458" - Hot Jupiter host star
- "RA=10.0 Dec=20.0" - Coordinate-based target

## Important Notes

- **Numpy serialization**: Use `fix_config()` before saving JSONField to convert `numpy.float32` → `float`
- **File security**: All file downloads use `sanitize_path()` to prevent directory traversal attacks
- **Task cancellation**: Setting `target.celery_id = None` signals cancellation; TaskProcessContext checks this on entry
- **Thread pool safe design**: Implementation avoids process group operations that would kill the entire worker process
- **Cooperative cancellation**: Tasks are revoked using `terminate=False` to allow graceful shutdown
- **celery_pid handling**: PID field is cleared on task entry to prevent accidental worker termination
- **Chain tracking**: `celery_chain_ids` stores all task IDs in a chain for proper revocation
- **Task decorators**: Use `acks_late=True, reject_on_worker_lost=True` for reliability
- **Cache**: Views use `@cache_page(15*60)` decorator for preview images
- **AJAX updates**: Target detail page polls task status every 3 seconds via JavaScript
- **Permissions**: Check `user.is_staff` or `user == target.user` before allowing modifications
- **MacOS compatibility**: Export `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` before running Celery workers
- **APPLAUSE XML parsing**: The APPLAUSE TAP service returns malformed XML with undefined entities; uses lxml recovery mode or regex cleaning as fallback
