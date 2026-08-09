# lcserver

**Lightcurves for the Web** — a Django application for assembling long-term
optical light curves of a celestial object from as many sky surveys as will
answer for it.

You give it a name or a position; it resolves the target, then queries every
survey in turn for whatever photometry they hold, from Hipparcos transits of
1989 to last week's ZTF. Each survey's data is kept in its own band as
published, plots and tables are written per source, and an interactive viewer
overlays them so a variable star's behaviour over thirty years can be looked at
in one place.

## Data sources

| | Coverage |
| --- | --- |
| Hipparcos | Hp, 1989–1993, brighter than V ≈ 12 |
| NSVS | unfiltered ROTSE-I, 1999–2000 |
| ASAS-3 | V, 2000–2009, south of +28° |
| SuperWASP | broad band, 2004–2008 |
| CSS | Catalina Sky Survey, V-like unfiltered |
| PTF | Palomar Transient Factory |
| ZTF | g, r — colour-corrected per epoch |
| ASAS-SN | V and g |
| KWS | Kamogata, V and Ic |
| OMC | INTEGRAL Optical Monitoring Camera, V, since 2003 |
| MMT9 | Mini-MegaTORTORA, white light |
| WISE | W1, W2 infrared, including NEOWISE |
| Kepler | flux, quarters and K2 campaigns, 2009–2018 |
| TESS | flux, per sector |
| DASCH | Harvard plate archive, historical |
| APPLAUSE | European plate archive, Dec > −30° |

Bands are kept as each survey measured them. Where a conversion onto a common
scale is applied it is an *additional* series rather than a replacement, and
every conversion prints its formula and parameters into that source's log.

## Requirements

- Python 3.11
- Redis on `localhost`, database `1` — Celery's broker, result backend and cache
- The packages in `requirements.txt`

## Setup

```sh
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
```

## Running

Two processes. The web application:

```sh
python manage.py runserver
```

and a Celery worker, which does all of the acquisition:

```sh
./run_celery.sh
```

`run_celery.sh` runs the worker under `watchmedo`, so it restarts when the code
changes, and exports `OBJC_DISABLE_INITIALIZE_FORK_SAFETY` for macOS.

## Configuration

Read from the environment or a `.env` file, via `python-decouple`:

| | |
| --- | --- |
| `SECRET_KEY` | Django secret. Set it for anything but local use. |
| `DEBUG` | default `False` |
| `TARGETS_PATH` | where per-target data is written, default `targets/` |
| `CELERY_CONCURRENCY` | how many sources are acquired at once, default `4` |

The sources of one target are acquired in parallel, so `CELERY_CONCURRENCY` is
also the number of simultaneous queries pointed at external services. Several
sources go to CDS; raising it is not free of consequence for them.

## Management commands

### `targets` — inspect and tidy targets

```sh
python manage.py targets --list
python manage.py targets --list --state failed --user someone --sort modified
python manage.py targets --info 42
python manage.py targets --delete 42          # removes the record and its files
python manage.py targets --cleanup 42         # removes the files, keeps the record
```

### `test_target` — run processing without Celery

Runs a step synchronously in the current process, which is what you want when
something is misbehaving: exceptions surface where you can see them.

```sh
python manage.py test_target 42 --step ztf --verbose
python manage.py test_target "Kepler-11" --new --step all --verbose
python manage.py test_target 42 --step tess --show          # plots interactively
python manage.py test_target 42 --step info --pdb           # debugger on exception
python manage.py test_target 42 --step info --debug         # let exceptions escape
```

### `maintenance` — system checks and housekeeping

```sh
python manage.py maintenance --check      # Redis, workers, disk, templates
python manage.py maintenance --stats
python manage.py maintenance --fix-stale  # targets left marked as running
python manage.py maintenance --cleanup-orphans
python manage.py maintenance --cleanup-old --days 30
```

Anything destructive takes `--dry-run` first.

`--check` includes a scan for Django template comments written across more than
one line. `{# … #}` closes at the newline whether or not anything closed it, so
the remainder of such a comment is rendered into the page; nothing else in the
stack warns about it.

## Layout

```
lcserver/
├── processing/     one module per survey, each registering itself
├── surveys.py      the registry - metadata, bands, form fields, layout
├── celery_tasks.py task generation, and the canvas a full run is built into
├── views*.py       pages, file browser, light curve viewer, queue
└── templates/
targets/{id}/       per-target logs, plots, VOTables, and cache/
```

Adding a survey means writing `processing/xxx.py` with a `target_xxx()`
function and decorating it with `@survey_source(...)`. The form, the button,
the task, the section on the page, its place in a full run and its entry in the
cache panel are all derived from that registration; nothing else needs editing.
