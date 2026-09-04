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

Alongside the light curves it collects what else is on record for the same
position: archival spectra, a spectral energy distribution built from the
catalogues, and what Gaia and SIMBAD know of the star itself. A photosphere
can be fitted to that distribution, from the spectral viewer, against as many
model atmosphere grids as are worth asking.

## Data sources

### Time-domain photometry

| | Coverage |
| --- | --- |
| Hipparcos | Hp, 1989–1993, brighter than V ≈ 12 |
| NSVS | unfiltered ROTSE-I, 1999–2000 |
| ASAS-3 | V, 2000–2009, south of +28° |
| SuperWASP | broad band, 2004–2008 |
| KELT | broad R, 2006–2019, bright stars |
| CoRoT | flux, two fields on the Galactic plane from space, 2007–2012 |
| CSS | Catalina Sky Survey, V-like unfiltered |
| PTF | Palomar Transient Factory |
| BGDS | Bochum Galactic Disk Survey DR2, r′ and i′ along the southern plane, 2010–2019 |
| ZTF | g, r — colour-corrected per epoch |
| ASAS-SN | V and g |
| KWS | Kamogata, V and Ic |
| OMC | INTEGRAL Optical Monitoring Camera, V, since 2003 |
| MMT9 | Mini-MegaTORTORA, white light |
| FRAM | B, V, R, I from the Pierre Auger Observatory and CTAO telescopes |
| WISE | W1–W4 in 2010, W1 and W2 through the NEOWISE years |
| Kepler | flux, quarters and K2 campaigns, 2009–2018 |
| TESS | flux, per sector |
| DASCH | Harvard plate archive, historical |
| APPLAUSE | European plate archive, Dec > −30° |

Bands are kept as each survey measured them. Where a conversion onto a common
scale is applied it is an *additional* series rather than a replacement, and
every conversion prints its formula and parameters into that source's log.

A final `combined` step draws everything that reached a common scale on one
pair of axes — the whole record, and the recent years on their own.

### Spectra

| | Coverage |
| --- | --- |
| SDSS | DR17, 3600–10400 Å |
| LAMOST | DR11, northern sky, 3700–9100 Å |
| DESI | DR1, 3600–9800 Å, northern sky |
| APOGEE | DR17, H band, 1.51–1.70 µm |
| SPHEREx | QR2 spectrophotometry, all sky, 0.75–5 µm |
| ESO archive | reduced UVES, X-shooter, FEROS, HARPS, GIRAFFE and the rest |
| Gaia | DR3 XP and RVS, fetched by the info step |

Every spectrum is stored on one physical scale — wavelength in Å, flux in
erg/s/cm²/Å — whatever the survey published in, so they can be read against
each other. A source with no flux calibration says so and is divided by its own
median instead.

### Catalogue photometry

The `sed` source builds a spectral energy distribution from whatever VizieR
catalogues cover the target, 0.15 to 22 µm, each point carrying the width of
the band it was measured in and a note of where on the sky it was measured —
GALEX and AllWISE do not centroid alike, and a point several arcsec away may
belong to something else.

## Fitting a photosphere

From the spectral viewer, the distribution the `sed` step built can be fitted
with one model atmosphere: interpolated from a grid at (Teff, log g, [Fe/H]),
reddened by A_V, and diluted by (R/d)². Nested sampling gives the posterior,
and it is kept as whole rows rather than as a parameter at a time, so the
temperature/extinction ridge that dominates a reddened star survives into the
answer instead of being averaged flat.

Which points are fitted is yours to say: a single Pan-STARRS band can be
dropped without dropping the survey, and a measurement the SED step cannot
place — a magnitude off a paper, say — can be typed in by hand.

Sixteen grids are available, several at once; BT-Settl, TLUSTY, Phoenix,
Castelli & Kurucz, Kurucz, BOSZ and Koester are on the form. They are not
averaged. Two grids that both cover the regime disagree by an amount that is
the systematic from the atmosphere physics, and that is reported as it stands.

Each run is written to `targets/{id}/sedfit/{timestamp}/` and stays there —
what was asked as well as what came back, so a run can be compared with the one
before it. Every run leaves its log, the SED with the model and the residuals
against it, a corner plot per grid, and the parameters of all the grids drawn
over each other.

Two numbers come out beside the parameters and are worth as much as they are:
the **jitter**, the fractional model inadequacy the fit needed, which says how
far the photometry is from anything one photosphere explains; and the
**shrinkage**, how much narrower the temperature posterior is than its prior,
below which the answer is largely the prior rather than the data — which for a
reddened star with no ultraviolet is what happens, and is worth saying rather
than quoting a number.

## Quality filtering

Many sources take a **quality level**, on their own form:

| | |
| --- | --- |
| `standard` | what the survey, or this code, calls good |
| `relaxed` | only what is indefensible |
| `published` | nothing judged, the data as they arrive |

The names are shared; what each means is the source's own affair, and a source
offers only the levels it can actually tell apart — DASCH has a rejection
bitfield to appeal to, KWS only the scatter of its own measurements. Whichever
was applied is written into that source's log along with what it came to.

## What the info step collects

Before any survey is queried, `info` resolves the target and gathers what is
known of it: SIMBAD identifiers and classification, the AAVSO VSX entry if
there is one, Gaia DR3 astrometry, photometry and distances, its kinematics,
and the interstellar reddening along the line of sight. It also picks up the
epoch photometry that comes for free with the catalogues — Pan-STARRS DR2 warp
and Gaia DR3 — and the Gaia XP and RVS spectra.

Reddening is reported both from the two-dimensional IRSA maps and, where the
three-dimensional maps are installed, resolved in distance, so that only the
dust actually in front of the star is counted. Edenhofer et al. (2023) is read
wherever it reaches, and Bayestar2019 takes over beyond its 1.25 kpc. Both are
optional and several gigabytes apiece; where they are absent nothing is said of
them.

## Pages

| | |
| --- | --- |
| `/targets/{id}/` | the target: every source, its form, its log and its plots |
| `/targets/{id}/lightcurve/` | interactive viewer — series toggled and offset, folded, and a Lomb-Scargle period search that can look underneath a slow trend |
| `/targets/{id}/spectrum/` | spectral viewer, on the physical scale, with the lines marked that fall inside the range on show — and the photosphere fit, its runs and their figures |
| `/targets/{id}/files/` | file browser over the target directory, previewing FITS, images and tables |
| `/cutouts/` | multi-wavelength cutouts of any position, resolved on the fly — no target, no worker, no login |
| `/passbands/` | the photometric conversions written out and calculable, over the transmission curves they run between |
| `/queue/` | Celery queue, chain positions, and the controls to stop a run |

Plots follow the theme the reader is in.

## Requirements

- Python 3.11
- Redis on `localhost`, database `1` — Celery's broker, result backend and cache
- The packages in `requirements.txt`
- For the photosphere fit, `astroARIADNE`, which is not on PyPI and is
  installed from its repository. The model grids and pyphot's filter profiles
  come with it, and nothing else of it is used — the cubes are read directly.
- Optionally `dustmaps`, with `dustmaps.edenhofer2023.fetch()` and
  `dustmaps.bayestar.fetch()`, for extinction resolved in distance

## Setup

```sh
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
```

For the photosphere fit, the model grids as well:

```sh
pip install git+https://github.com/jvines/astroARIADNE.git
```

That is 46 MB of HDF5 cubes, installed beside the package, and they are found
there without being told. Without the package the rest of the application is
unaffected; the fit is the only thing that asks for it.

`astroARIADNE.fetch.fetch_spectra_cache()` fetches a further 2.8 GB of the
spectra those cubes were convolved from, which is what lets a model be drawn as
a line rather than as a flux per band. It is optional, and a grid without it
says so.

### The grid directory

A grid is a file, and the directory is the registry: what is installed is what
is there, and adding one is putting one there. Each grid is `<name>.h5`, the
cube the fit interpolates, optionally beside `<name>.spectra.h5`, the spectra it
was convolved from. Both carry what a reader should be told - a label, a
description, how far out the grid may be believed - so nothing about a grid is
written down anywhere else.

astroARIADNE's own layout, which names its files differently and keeps every
spectrum in one cache, is read as it stands. To lay it out the other way:

```sh
python manage.py sedgrid --list
python manage.py sedgrid --split --to /where/the/grids/should/live
```

and point `SEDFIT_GRIDS` at that directory. Splitting is worth it once you
build grids of your own: adding one is then a file rather than an edit to a
2.8 GB cache.

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
| `SEDFIT_GRIDS` | the grid directory; unset, astroARIADNE's own is read |
| `SEDFIT_SPECTRA` | astroARIADNE's single spectra cache, for grids whose spectra do not sit beside them |
| `CELERY_CONCURRENCY` | how many sources are acquired at once, default `4` |

The sources of one target are acquired in parallel, so `CELERY_CONCURRENCY` is
also the number of simultaneous queries pointed at external services. Several
sources go to CDS; raising it is not free of consequence for them.

Every external answer is cached under `targets/{id}/cache/`, an empty answer as
well as a full one, so that re-running a step does not re-ask. Every source
that talks to an external service carries the switch that refreshes it.

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

The step's log file is written either way; `--verbose` echoes it to the
terminal as it goes.

### `sedgrid` — the model grids on disk

```sh
python manage.py sedgrid --list
python manage.py sedgrid --split --to ~/sedgrids
python manage.py sedgrid --split --to ~/sedgrids --only btsettl tlusty
```

`--list` says what the grid directory holds, where each grid's spectra are, and
which of them are offered on the fitting form — a grid is offered when it can
say what it is, so the specialist grids astroARIADNE ships stay loadable by
name without being put in front of someone who has been told nothing about them.

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
│   └── sedfit.py   the photosphere fit - grids, priors, sampling, figures
├── surveys.py      the registry - metadata, bands, form fields, layout
├── celery_tasks.py task generation, and the canvas a full run is built into
├── views.py        pages, file browser
├── views_lightcurve.py  the viewer, and the period search behind it
├── views_spectrum.py    the spectral viewer, and the photosphere fit behind it
├── views_cutouts.py     cutouts of any position, outside the target loop
├── views_passbands.py   the conversions, and the passbands they run between
├── views_celery.py      the queue
└── templates/
targets/{id}/       per-target logs, plots, VOTables, and cache/
targets/{id}/sedfit/{timestamp}/   one photosphere fit, as asked and as answered
```

Adding a survey means writing `processing/xxx.py` with a `target_xxx()`
function and decorating it with `@survey_source(...)`. The form, the button,
the task, the section on the page, its place in a full run and its entry in the
cache panel are all derived from that registration; nothing else needs editing.
