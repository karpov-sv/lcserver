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

Where Gaia published an XP spectrum for the source, it is drawn under the SED
and compared with the fitted photosphere bin by bin — in bins of 30 nm, which is
about twice the width of the instrument's line spread function, so that what
comes out does not depend on the shape of that function. The log says how far
the spectrum sits from the model, separately over the bins the fitted bands span
and over any beyond them — the second being what the model does past its data
rather than against it — and what is left once that offset is taken out, which
is the shape.

It can also be **fitted**, with one switch on the form. A bin is then a point
like any other: it appears in the list, it can be turned off on its own, and it
is interpolated at the fitted parameters out of a cube of the same layout as the
grid's own — `<name>.xp.h5`, written by `sedgrid --xp` from that grid's spectra.
Nothing stands in for the fit and nothing is corrected. Gaia's synthetic
photometry leaves the fit when the bins enter it, being the same spectra
integrated, and a grid with no bins written is skipped rather than guessed at.

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
python manage.py sedgrid --ingest-powr ~/Downloads/griddl-gal-ob-vd3-sed \
    --to data/grids --label 'PoWR Gal OB Vd3' --description 'Galactic OB stars'
python manage.py sedgrid --ingest-powr ~/Downloads/griddl-wne-sed \
    --to data/grids --label 'PoWR WNE' --description 'early WN, Galactic'
python manage.py sedgrid --ingest-tlusty ~/Downloads/obstar_merged_3d.ascii \
    --to data/grids
python manage.py sedgrid --ingest-cdbs ~/tmp/kurucz/.../k93models --to data/grids
python manage.py sedgrid --ingest-cdbs ~/tmp/kurucz/.../ck04models --to data/grids
python manage.py sedgrid --ingest-koester ~/Downloads/models_.../koester2 \
    --to data/grids
python manage.py sedgrid --ingest-bosz ~/tmp/bosz --to data/grids
python manage.py sedgrid --ingest-btsettl ~/Downloads/bt-settl.medres.grid.fits \
    --to data/grids
python manage.py sedgrid --scatter            # or --scatter btsettl bosz
python manage.py sedgrid --xp                 # or --xp tlusty bosz
```

`--ingest-powr` converts a [PoWR](https://www.astro.physik.uni-potsdam.de/PoWR/)
download - one file per model, the star as seen from ten parsecs - into a cube
convolved through the same passbands every other grid was built with, plus the
spectra to draw it from. It appears on the form as soon as it is written; there
is nothing else to edit. It reads the OB grids and the Wolf-Rayet ones alike,
and works out from the download's own table which quantity it varies beside the
temperature: gravity for the OB grids, and for the WR grids the **transformed
radius**, a wind density in disguise, since those hold luminosity fixed and give
one gravity per temperature. Where that happens the file says so, the fit leaves
any gravity prior off that axis, and every log and figure names it `log Rt`
rather than reporting a wind density as a surface gravity.

`--ingest-tlusty` reads a [TLUSTY](https://tlusty.oca.eu/) OB-star grid from the
single merged file it is published in for Cloudy - which is what astroARIADNE's
own TLUSTY cube was built from, so the new one is compared against the installed
one band by band and the ratios printed. What it adds is the spectra, which that
cube has none of, and the low-gravity edge a lattice with holes in it refuses.

`--ingest-cdbs` reads an atlas in the layout STScI distributes them in — a
directory per metallicity, one FITS per temperature, a flux column per gravity.
The 1993 Kurucz atlas and the 2004 Castelli & Kurucz one are both this shape,
and astroARIADNE's cubes for both came from them and stop at 12000 K, which is
less than half of either: 7618 and 3808 models reaching 50000 K. Past about ten
microns the atlases are a Rayleigh-Jeans tail rather than a model, so those
grids are written believing themselves only to 8.5 µm.

`--ingest-btsettl` writes BT-Settl's spectra from the medium-resolution grid
published for [pystellibs](https://github.com/mfouesneau/pystellibs), which run
from a nanometre to a millimetre where astroARIADNE's cache stopped at 4.63 µm.
Only the spectra: that file covers 8132 of the cube's 14183 models, starting at
2600 K where the cube starts at 400, so a cube from it would cost the cool and
metal-poor ends. The two files are allowed to hold different models, and a fit
outside the spectra is drawn per band as it was before.

`--ingest-bosz` reads a [BOSZ](https://archive.stsci.edu/hlsp/bosz) download.
That library is organised by rather more than the three axes a fit varies here —
alpha enhancement, carbon abundance and microturbulence besides metallicity, and
eight instrumental broadenings — so only one composition at one broadening is
wanted, and only about one file in eight hundred of the archive. The cube here
already came from the solar-abundance, 2 km/s slice; what the download adds is
its spectra, out to 32 µm, which is further than any other grid here reaches at
those temperatures. The ingest refuses a download holding more than one
composition rather than average two of them together.

`--ingest-koester` reads Koester's white-dwarf models as SVO hands them out, one
file per model. The cube here was built from the same models and has no spectra;
these are finely sampled through the optical, where a DA's Balmer lines are the
whole of what there is to see. They stop at three microns, where a
Rayleigh-Jeans tail takes over for the four bands just beyond — which the models
are measurably already in, and which the cube it replaces got wrong, having W2
brighter than W1 on a white dwarf.

`--xp` writes each grid's Gaia XP bins beside it, from its spectra: a small
cube — under a megabyte — holding a flux per 30 nm bin per model, which is what
lets a fit take the spectrum as points rather than only compare with it. A grid
with no spectra has nothing to bin and is skipped.

`--scatter` rewrites a grid stored as a lattice as the models it actually holds.
A cube on a lattice has a place for every combination of its three axes and a
real grid never computed most of them, so a box whose corners are not all models
cannot be interpolated in and the fit refuses a band of parameter space all the
way around the region that exists. Written as the models and triangulated over,
that band comes back — a sixth of the reachable space on BT-Settl — and where
both layouts answer they agree to about one per cent. Nothing is downloaded and
no flux is recomputed.

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
├── ingest/         turning somebody else's data into ours, once and by hand
│   ├── passbands.py  the convolution every grid here was built with
│   ├── store.py      the two files a grid is, written out
│   ├── ariadne.py    astroARIADNE's grids, laid out as this reads them
│   ├── powr.py       a PoWR download, one file per model at ten parsecs
│   ├── tlusty.py     a TLUSTY grid, out of its merged file for Cloudy
│   ├── cdbs.py       the Kurucz and Castelli atlases, as STScI ships them
│   ├── bosz.py       a BOSZ download, one composition of the library
│   ├── btsettl.py    BT-Settl's medium-resolution spectra, from their grid file
│   ├── koester.py    white-dwarf models, one file each, as SVO hands them out
│   └── scatter.py    a lattice rewritten as the models it actually holds
├── surveys.py      the registry - metadata, bands, form fields, layout
├── celery_tasks.py task generation, and the canvas a full run is built into
├── views.py        pages, file browser
├── views_lightcurve.py  the viewer, and the period search behind it
├── views_spectrum.py    the spectral viewer, and the photosphere fit behind it
├── views_cutouts.py     cutouts of any position, outside the target loop
├── views_passbands.py   the conversions, and the passbands they run between
├── views_celery.py      the queue
└── templates/
data/grids/         one model grid per file, the register the fitter reads
targets/{id}/       per-target logs, plots, VOTables, and cache/
targets/{id}/sedfit/{timestamp}/   one photosphere fit, as asked and as answered
```

`processing/` runs per target, in a worker, whenever somebody asks. `ingest/`
runs once, by hand, to prepare something the application then reads for the
rest of its life; nothing there is on the request path.

Adding a survey means writing `processing/xxx.py` with a `target_xxx()`
function and decorating it with `@survey_source(...)`. The form, the button,
the task, the section on the page, its place in a full run and its entry in the
cache panel are all derived from that registration; nothing else needs editing.
