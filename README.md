# lcserver

**Lightcurves for the Web** — a Django application for assembling long-term
optical light curves of a celestial object from as many sky surveys as will
answer for it.

You give it a name or a position; it resolves the target, then queries every
survey in turn for whatever photometry they hold, from Hipparcos transits of
1989 to last week's ZTF. Each survey's data is kept in its own band as
published, plots and tables are written per source, and an interactive viewer
overlays them so a variable star's behaviour over thirty years can be looked at
in one place, searched for a period, and followed when that period changes.

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
| OGLE | I and V since 1997, bulge, southern disk and Magellanic Clouds; only stars in its Collection of Variable Stars |
| NSVS | unfiltered ROTSE-I, 1999–2000 |
| ASAS-3 | V, 2000–2009, south of +28° |
| SuperWASP | broad band, 2004–2008 |
| KELT | broad R, 2006–2019, bright stars |
| CoRoT | flux, two fields on the Galactic plane from space, 2007–2012 |
| NGTS | broad 520–890 nm, 13 s cadence published in bins, 72 southern fields, 2015–2018 |
| CSS | Catalina Sky Survey, V-like unfiltered |
| PTF | Palomar Transient Factory |
| BGDS | Bochum Galactic Disk Survey DR2, r′ and i′ along the southern plane, 2010–2019 |
| ZTF | g, r — colour-corrected per epoch |
| ATLAS | forced photometry, c and o, whole sky since 2015; queued, needs an API key, run by hand only |
| ASAS-SN | V and g |
| KWS | Kamogata, V and Ic |
| AAVSO | a century of observer-contributed photometry, by name; needs an API key |
| OMC | INTEGRAL Optical Monitoring Camera, V, since 2003 |
| MMT9 | Mini-MegaTORTORA, white light |
| FRAM | B, V, R, I from the Pierre Auger Observatory and CTAO telescopes |
| VISTA | VVV (VIRAC2) and VMC from ESO: Z, Y, J, H, Ks, bulge, southern disk and Magellanic Clouds, 2009–2023, fainter than Ks ≈ 11 |
| NSC | NOIRLab Source Catalog DR2 and, in the Magellanic Clouds, DELVE-MC: every public DECam, Mosaic3 and 90Prime exposure, ugrizY, 2012–2020, fainter than ≈ 15 mag |
| SkyMapper | DR4 detections, uvgriz, southern sky, 2014–2021 |
| PGIR | Palomar Gattini-IR, J, sky visible from Palomar, 2018–2022, J < 15.5 |
| WISE | W1–W4 in 2010, W1 and W2 through the NEOWISE years |
| Kepler | flux, quarters and K2 campaigns, 2009–2018; the corrected flux, or the raw aperture photometry for variability the correction would flatten |
| TESS | flux, per sector; the pipelines' light curves - corrected, or their raw aperture photometry for variability the correction would flatten - or measured in the full-frame images for faint or blended stars; optionally averaged to 10 min - 2 h |
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
| IUE | INES archive, 1978–1996, 1150–3350 Å, low resolution and echelle rebinned to it |
| SDSS | DR17, 3600–10400 Å |
| LAMOST | DR11, northern sky, 3700–9100 Å |
| DESI | DR1, 3600–9800 Å, northern sky |
| APOGEE | DR17, H band, 1.51–1.70 µm |
| SPHEREx | QR2+QR3 spectrophotometry, all sky, 0.75–5 µm |
| ESO archive | reduced UVES, X-shooter, FEROS, HARPS, GIRAFFE and the rest |
| IACOB | Galactic OB stars at R = 25000–85000, 3500–9200 Å, continuum normalised |
| Gaia | DR3 XP and RVS, fetched by the info step |

Every spectrum is stored on one physical scale — wavelength in Å, flux in
erg/s/cm²/Å — whatever the survey published in, so they can be read against
each other. A source with no flux calibration says so and is divided by its own
median instead. IACOB's spectra are normalised to the continuum, and the
viewer loads them unticked.

### Catalogue photometry

The `sed` source builds a spectral energy distribution from whatever VizieR
catalogues cover the target, 0.15 to 22 µm, each point carrying the width of
the band it was measured in and a note of where on the sky it was measured —
GALEX and AllWISE do not centroid alike, and a point several arcsec away may
belong to something else.

## Period analysis

The light curve viewer folds any selection of series at a period, and searches
for one, the series pooled, each about its own median. The search is the
Lomb-Scargle periodogram, with up to four harmonics for a light curve that is
no sinusoid — a pulsator's sawtooth, a contact binary's double wave — or box
least squares, for the eclipse or transit a periodogram of sinusoids may not
see at all, scored by its signal detection efficiency. The highest peaks of a
periodogram are fitted again — each series with its own trend, each survey and
filter with its own harmonics — and ranked by that fit, which is what tells
apart the aliases a periodogram gives the same power, and what gives the
period its uncertainty. A slow trend can be fitted alongside, so that a short
period can be looked for underneath it, and the search limited to a range of
dates. The trend has no knots closer than the visits allow, and isolated
points — single Gaia or Pan-STARRS epochs — share one slow trend rather than
each taking a level of its own, and all of its variation with it; the light
curve is then shown with the trend fitted along with the period.

The periodogram of the sampling alone is drawn under the periodogram, centred
on the period folded at: wherever it rises, that period is seen again, through
the day, the month or the year. The folded light curve carries the median of
each series in bins of phase, which shows its shape at any period, whatever
found it.

A period found can be removed and the search run again on what is left —
prewhitening, for a star with more than one. Every removed period is fitted
again together with the others each time, so that one found early does not
keep the leakage of one found later, and the light curve shown is then the
residuals. And the periodogram can be taken in a window sliding along the
light curve, drawn as power against period and time, for how a period — or
which of several — comes and goes.

A period that changes is fitted from there: the phase as the integral of a
frequency that is a polynomial in time, or a spline with a knot every so many
days, kept smooth where no data say otherwise so that it bridges a gap rather
than swinging about across it, started from the period being folded at. Counting the cycles across the
seasonal gaps is what such a fit lives or dies by, so it starts both from the
phases of windows that follow the seasons, linked one window at a time, and
from the periods found within each window, and keeps whichever fits best. The
answer is drawn as an O−C diagram against the period and T₀ folded at, with
the model and its band, the residuals from it, and the period with time beside
the period each window finds on its own — the one check on the count of
cycles that does not depend on it. The light curve can then be folded with the
model's phase.

The light curve's shape is described in that fit by three harmonics, or as
many as chosen — or, for auto, as the Bayesian information criterion asks
for: one or two for a sinusoidal pulsator, a dozen and more for a detached
eclipsing binary, whose narrow eclipses three harmonics put hours out. Each
survey's errors are scaled to its scatter about the fit and the fit made
again, so that a few points with errors of millimagnitudes that no model
follows — Gaia's, in years when little else was looking — do not outweigh the
thousands that count the cycles; the summary names the surveys scaled most. For an eclipsing binary seen often
enough, the space missions above all, the O−C can instead come from the
eclipse minima: each eclipse timed on its own by a curve symmetric about its
minimum, and the model fitted to those times — primaries at whole cycles,
secondaries half a cycle on with their offset from it fitted too, the errors
of each series' primaries and secondaries scaled to their scatter, and the
minima far off the fit left out.

The errors throughout allow for noise that does not average down as white
noise would, which over the thousands of points of a TESS sector is most of
it, and the summary says when the O−C wanders beyond them — as on a spotted
star it will. What each fit said stays on the page, and the light curve, the
periodogram, the O−C and the model can be downloaded as CSV.

## Variability without a period

Not every star has a period to find, and the viewer says how much one varies
and on what timescales either way. For each survey in one filter, it gives
the reduced chi-squared about the mean and the chance of one as large from
the errors alone, the scatter with the errors taken off, von Neumann's ratio
and Stetson's J and K (noise, a slow variation, or one that spends its time at
the extremes), the skewness that tells dips from outbursts, and the trend per
year. Beside them is the structure function, the scatter of the differences of
pairs of points against the time between them, with the noise of each pair
taken off. Pairs are taken within each series, which is calibrated on its
own, or, if asked, across all the series at the offsets they are drawn at,
out to the century between a plate and ZTF. A damped random walk is fitted
to it for a timescale and an amplitude, and a timescale longer than a tenth of
the span is reported as no more than a lower limit.

The colour of the star comes from any two sources measured close enough in
time - matched epoch by epoch, or interpolated across a gap for surveys that
never take two bands on one night - with errors carried through, and with a
diagram against a magnitude or a second colour. The line through the diagram
is fitted with the errors of both axes and their correlation, and with the
scatter about it beyond them. It is drawn against the reddening by dust and a
blackbody heating and cooling, each put through the passbands of the sources,
so that its slope can be read as one or the other. Folded, the colour is also
the difference of each source's harmonic model, which needs no simultaneous
measurements at all.

## Fitting a photosphere

From the spectral viewer, the distribution the `sed` step built can be fitted
with one model atmosphere: interpolated from a grid at (Teff, log g, [Fe/H]),
reddened by A_V, and diluted by (R/d)². Nested sampling gives the posterior,
and it is kept as whole rows rather than as a parameter at a time, so the
temperature/extinction ridge that dominates a reddened star survives into the
answer instead of being averaged flat.

Which points are fitted is yours to say: a single Pan-STARRS band can be
dropped without dropping the survey, and a measurement the SED step cannot
place — a magnitude off a paper, say — can be typed in by hand. Two
submillimetre bands take a flux and not a magnitude: SPIRE PMW and PLW lie past
the red end of the reference Vega spectrum, so there is no zero point to read a
magnitude against, and the picker says so.

Several grids can be fitted at once — BT-Settl, Phoenix, BOSZ, Castelli &
Kurucz, Kurucz, TLUSTY and Koester, and PoWR for the Wolf-Rayet and OB stars
with winds, as far as they are installed. They are not averaged. Two grids that
both cover the regime disagree by an amount that is the systematic from the
atmosphere physics, and that is reported as it stands.
What each of them actually says — where a colour stops following temperature,
where two of them part company, which bands one can answer for at all — is the
[`/models/`](#pages) page.

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

The **IUE** spectra are fitted the same way, with a switch of their own: 100 A
bins from 1250 to 3150 A, out of `<name>.iue.h5` written by `sedgrid --iue`.
Only the low dispersion spectra through the large aperture are binned, every
exposure reaching into a bin contributing to it, and each bin carries a 5%
calibration error besides its noise and the scatter between exposures. They
are what pins the extinction of a hot star: the 2175 A bump is five bins wide.

So is **SPHEREx**, in 38 bins of R = 20 from 0.75 to 5 µm, out of
`<name>.spherex.h5` written by `sedgrid --spherex`. Each bin is the mean of the
exposures in it, each moved to the middle of the bin along the local slope of
the spectrum, against a model integrated through the bin and the filter's own
width; it carries a 3% calibration error besides its noise and the scatter
between visits. Unlike the other spectra its bins can be measured as an
**excess** instead of fitted, one by one - five microns is as often a disc or
a wind as it is the star. Grids whose spectra stop short of five microns get
the bins they reach, and are skipped by a fit that asks for the rest.

A star with a disc needs more than a photosphere, and three options give it.
**Dust** - one blackbody, or a hot one (600–2000 K) and a warm one (30–600 K) -
is fitted *with* the star rather than measured after it, each component's
luminosity a fraction of the star's; at one micron a Herbig star's inner rim is
already a quarter of the light, so the two cannot be fitted one after the
other. **R<sub>V</sub>** can be 3.1, 5, or free between 2 and 6, circumstellar
dust being greyer than the interstellar medium. And the bins of the spectra
holding a strong **emission line** - H-alpha, the Paschen and Brackett lines,
O I and the Ca II triplet - can be left out with one switch. Where nothing
pins the temperature down, a Gaussian prior on it can be taken from Gaia.

Each spectrum - Gaia XP, SPHEREx, IUE - can also be **scaled**: multiplied by
the factor that puts it on the scale of the catalogue photometry, which is
never scaled. The factor is set in that spectrum's row of the viewer, where it
moves the data on the plot, and the fit either takes it as it is or finds its
own - which the plot then takes up, so that the plot shows what the fit did. A star that changed between
epochs needs one, and so does a SPHEREx spectrum with a neighbour in it: its
aperture is nine arcseconds, and the SPHEREx step says, from Gaia, how much of
the aperture's flux is other stars.

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

SPHEREx offers one more, `saturated`: the standard filtering, but keeping the
saturated core of a star too bright for the detector, which still holds usable
values. Its log then says, detector by detector, whether those pixels hold what
the PSF of the unsaturated ones around them predicts, relative to the same
prediction for the target's own unsaturated exposures.

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
| `/targets/{id}/lightcurve/` | interactive viewer — series toggled and offset, folded, a period search, and a period that changes with its O−C diagram |
| `/targets/{id}/spectrum/` | spectral viewer, on the physical scale, with the lines marked that fall inside the range on show — and the photosphere fit, its runs and their figures |
| `/targets/{id}/references/` | every paper SIMBAD links to the target, ranked by how much each is about it |
| `/targets/{id}/catalogues/` | every VizieR catalogue with an entry at the target, those of the papers about it first |
| `/targets/{id}/files/` | file browser over the target directory, previewing FITS, images and tables |
| `/cutouts/` | multi-wavelength cutouts of any position, resolved on the fly — no target, no worker, no login |
| `/passbands/` | the photometric conversions written out and calculable, over the transmission curves they run between |
| `/models/` | what the model grids say — any parameter against any colour, magnitude or bolometric correction, over as many grids at once as you ask for, with the reddening and the radius as sliders; the families of spectra behind the cubes; and which bands each grid can answer for |
| `/queue/` | Celery queue, chain positions, and the controls to stop a run |
| `/about/` | where the data comes from — each archive and grid as it describes itself, with the acknowledgements they ask for |

Plots follow the theme the reader is in.

## Requirements

- Python 3.11
- Redis on `localhost`, database `1` — Celery's broker, result backend and cache
- The packages in `requirements.txt`
- For the photosphere fit, a directory of model grids - see
  [Model grids](#model-grids)
- Optionally `dustmaps`, with `dustmaps.edenhofer2023.fetch()` and
  `dustmaps.bayestar.fetch()`, for extinction resolved in distance

## Setup

```sh
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
```

### Model grids

The photosphere fit reads a directory of grids, `SEDFIT_GRIDS`. A grid is a
file, and the directory is the registry: each grid is `<name>.h5`, the cube the
fit interpolates, optionally beside `<name>.spectra.h5`, the spectra it was
convolved from. Both carry what a reader should be told - a label, a
description, how far out the grid may be believed - so nothing about a grid is
written down anywhere else, and adding one is putting one there.

`sedgrid --ingest-*` builds them from what the groups publish. The quickest way
to a first set is astroARIADNE, which is not on PyPI and ships 46 MB of cubes:

```sh
pip install git+https://github.com/jvines/astroARIADNE.git
python manage.py sedgrid --split --to data/grids
```

and point `SEDFIT_GRIDS` at `data/grids`, after which the package can go:
nothing here imports it, and the passbands the cubes are keyed by are kept in
`lcserver/processing/filters.py` rather than borrowed from it. Left unset,
`SEDFIT_GRIDS` reads astroARIADNE's own directory as it stands. `astroARIADNE.fetch.fetch_spectra_cache()` fetches a further
2.8 GB of the spectra those cubes were convolved from, which is what lets a
model be drawn as a line rather than as a flux per band; it is optional, and a
grid without it says so.

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
| `AAVSO_TOKEN` | key for the AAVSO API; without it the AAVSO step finds nothing |
| `ATLAS_TOKEN` | key for the ATLAS forced photometry server; without it the ATLAS step finds nothing |
| `ADS_TOKEN` | key for the ADS API, for citation counts, arXiv links and the archives holding each paper's data in the reference list - one request per target, of the hundred a day ADS allows; without it the list is SIMBAD's alone |

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
python manage.py sedgrid --split --to ~/sedgrids --only btsettl tlusty
python manage.py sedgrid --ingest-cdbs ~/tmp/kurucz/.../ck04models --to data/grids
python manage.py sedgrid --ingest-powr ~/Downloads/griddl-wne-sed \
    --to data/grids --label 'PoWR WNE' --description 'early WN, Galactic'
python manage.py sedgrid --scatter            # or --scatter btsettl bosz
python manage.py sedgrid --xp                 # or --xp tlusty bosz
```

| | |
| --- | --- |
| `--list` | what the grid directory holds, where each grid's spectra are, and which are offered on the form - a grid is offered when it can say what it is |
| `--split` | astroARIADNE's grids, one cube and one spectrum file each |
| `--ingest-cdbs` | the 1993 Kurucz and 2004 Castelli & Kurucz atlases as STScI distributes them, to 50000 K where astroARIADNE's cubes stop at 12000; believed only to 8.5 µm, past which they are a Rayleigh-Jeans tail |
| `--ingest-powr` | a [PoWR](https://www.astro.physik.uni-potsdam.de/PoWR/) download, OB or Wolf-Rayet; the WR grids vary the transformed radius rather than gravity, and say `log Rt` wherever it shows |
| `--ingest-tlusty` | the [TLUSTY](https://tlusty.oca.eu/) OB grid from its merged file for Cloudy, with the spectra astroARIADNE's cube lacks |
| `--ingest-bosz` | the spectra of one [BOSZ](https://archive.stsci.edu/hlsp/bosz) composition and broadening, out to 32 µm; a download holding more than one is refused |
| `--ingest-btsettl` | BT-Settl's spectra from a nanometre to a millimetre, from the grid published for [pystellibs](https://github.com/mfouesneau/pystellibs) |
| `--ingest-koester` | Koester's white-dwarf spectra, one file per model as SVO hands them out |
| `--scatter` | a grid stored as a lattice rewritten as the models it actually holds, which gives back the band of parameter space a lattice with holes refuses - a sixth of it on BT-Settl |
| `--xp`, `--iue`, `--spherex` | each grid's Gaia XP bins (30 nm), IUE bins (100 Å, 1250–3150 Å) or SPHEREx bins (R = 20, 0.75–5 µm), from its spectra, for the fit to take as points |

A new grid is offered on the form as soon as it is written; there is nothing
else to edit. What each ingest reads and why is in its module under
`lcserver/ingest/`.

### `maintenance` — system checks and housekeeping

```sh
python manage.py maintenance --check      # Redis, workers, disk, templates
python manage.py maintenance --stats
python manage.py maintenance --fix-stale  # targets left marked as running
python manage.py maintenance --cleanup-orphans
python manage.py maintenance --cleanup-old --days 30
```

Anything destructive takes `--dry-run` first.

`--check` also catches Django template comments written across more than one
line, which render the rest of the comment into the page.

## Tests

```sh
python manage.py test lcserver.tests
```

The period analysis is tested on synthetic light curves, sampled a few points
a night as the ground samples a star: the true period against its daily alias,
the uncertainty of a period against the scatter it should describe, an
eclipsing binary at half its period and at its period, close frequencies
fitted together, red noise, a drifting period followed across seasonal gaps,
and O−C. No database is needed, and they take a few seconds.

## Layout

```
lcserver/
├── processing/     one module per survey, each registering itself
│   ├── periods.py  the period search, prewhitening and the period change fit
│   ├── variability.py  variability indices and the structure function
│   ├── colours.py  the passbands of the light curves, and blackbody and reddening through them
│   ├── sedfit.py   the photosphere fit - grids, priors, sampling, figures
│   └── gridstats.py  what a grid says as colours and magnitudes, for the page
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
│   ├── scatter.py    a lattice rewritten as the models it actually holds
│   ├── xp.py         a grid's spectra binned as Gaia XP
│   ├── iue.py        a grid's spectra binned as IUE
│   └── spherex.py    a grid's spectra binned as SPHEREx
├── surveys.py      the registry - metadata, bands, form fields, layout
├── celery_tasks.py task generation, and the canvas a full run is built into
├── views.py        pages, file browser
├── views_lightcurve.py  the viewer, and the requests of its period tools
├── views_spectrum.py    the spectral viewer, and the photosphere fit behind it
├── views_cutouts.py     cutouts of any position, outside the target loop
├── views_passbands.py   the conversions, and the passbands they run between
├── views_models.py      the grids themselves - parameters, colours, spectra
├── views_references.py  the papers about the target, and their catalogues
├── views_about.py       where the data comes from, from the registry
├── views_celery.py      the queue
├── templates/
└── tests/          the period analysis, on synthetic light curves
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

## Licence

MIT — see [LICENSE](LICENSE).

One part of this is somebody else's and says so where it sits: the passband
curves in `lcserver/processing/filters.h5` were collected by
[astroARIADNE](https://github.com/jvines/astroARIADNE) (MIT, © 2019 Jose Vines)
from the [SVO Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/),
whose curves they are. `lcserver/processing/filters.py` carries the notice.

No astronomical data is kept here. The model grids and everything under
`targets/` are fetched and converted, not redistributed.
