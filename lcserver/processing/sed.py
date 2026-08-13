"""Spectral energy distribution out of the catalogue photometry at a position.

Not a spectrum any instrument took: a handful of broadband points, each from a
different survey, spanning from the ultraviolet to the mid-infrared - four
decades of wavelength where a spectrograph covers less than one. Beside the
spectra it says what the whole object is doing, and beside a SPHEREx spectrum
it says whether the near-infrared continues the optical or breaks from it.

The photometry comes from VizieR's SED service, which is what stands behind
its photometry viewer: one query at a position and it returns every catalogue
it has there, each measurement already converted to a flux in Jansky at an
effective wavelength. That conversion is the whole reason for using it. Done
here it would mean a table of zero points and effective wavelengths for every
filter of every catalogue, maintained by hand and wrong in the places nobody
looked; VizieR keeps that table for the whole of its holdings, and the filters
are named in the reply.

What it does not do is choose. The reply is everything at the position - two
hundred rows over sixty catalogues for an ordinary star, several of them the
same measurement republished, and some belonging to a neighbour rather than to
the target. So the catalogues are named here rather than taken as they come,
and each is matched to its own nearest source.
"""

import os
import io
import collections

import numpy as np
import requests

from astropy.table import Table, vstack

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    flambda_from_fnu, write_spectrum)


# VizieR's SED service, which its photometry viewer is a page around
SED_URL = 'https://vizier.cds.unistra.fr/viz-bin/sed'

# How far out to accept a catalogue's source, in arcsec. The service takes a
# radius of its own but barely honours it - asking for two arcsec against five
# changed a reply of 448 rows by nine - so the cut that matters is this one,
# made here against the positions the reply carries.
SED_SR = 3.0

# How far from its nearest row another row of the same catalogue may sit and
# still be the same source, in arcsec. Catalogue positions of one object differ
# between bands by a little; a neighbour differs by a lot. Pan-STARRS returned
# three sources within five arcsec of one target, and the two that were not it
# were fainter by a factor of fifty.
SED_SAME_SOURCE = 0.5

# The catalogues, in the order their wavelengths run. Each is named by its
# VizieR table, and by the name its file and its series take - which is what
# the viewer shows, so it is spelled as the survey spells it.
#
# 'filters' names the bands the catalogue measured itself, where it also
# republishes someone else's: AllWISE carries the 2MASS magnitudes of whatever
# it matched to, which are not AllWISE measurements and would otherwise be
# drawn twice, once here and once from 2MASS proper.
SED_CATALOGUES = [
    {'table': 'II/335/galex_ais', 'name': 'GALEX', 'colour': '#8e44ad'},
    {'table': 'I/360/syntphot', 'name': 'Gaia-syntphot', 'colour': '#2980b9'},
    {'table': 'II/349/ps1', 'name': 'Pan-STARRS', 'colour': '#16a085'},
    {'table': 'II/246/out', 'name': '2MASS', 'colour': '#d35400'},
    {'table': 'II/328/allwise', 'name': 'AllWISE', 'colour': '#922b21',
     'filters': ('WISE:',)},
]

# The colours run with the wavelength, violet to red, so that a point's colour
# says roughly where it sits before the legend is read. Gaia's synthetic
# photometry is deliberately the blue the Gaia XP spectrum is drawn in, being
# the same data summarised; AllWISE is a darker red than SPHEREx, which it
# would otherwise overlap in both wavelength and colour.

# The same colours, keyed by the name each catalogue's file carries, which is
# how the viewer is given them. By name and not in order: a target has whatever
# catalogues cover it, so counting through a list would colour Pan-STARRS one
# way where GALEX was also found and another way where it was not, and neither
# would match the plot written here.

# A Jansky in microjansky, the conversion out of f_nu being written for uJy
SED_JY = 1e6

# Speed of light in Angstrom/s, the service quoting frequencies in GHz
SED_C = 2.99792458e18


def _column(table, *names):
    """The first of these columns the table has, whatever it called it.

    The service names its columns with a leading underscore, which the VOTable
    reader keeps for some and strips from others, so both are looked for.
    """
    for name in names:
        for candidate in (name, '_' + name, name.lstrip('_')):
            if candidate in table.colnames:
                return np.asarray(table[candidate])

    raise SourceError(f"the SED service returned no {names[0]} column")


def _query(ra, dec, sr, basepath, log, refresh):
    """Everything VizieR has at a position, cached as it arrived.

    Cached whole rather than per catalogue: it is one request either way, and
    keeping the reply entire means the choice of catalogues can be changed
    without going back to the service.
    """
    cache_name = f"sed_{ra:.4f}_{dec:.4f}_{sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log,
                              'VizieR SED photometry', refresh=refresh) as cache:
        if not cache.hit:
            res = requests.get(SED_URL, timeout=180,
                               params={'-c': f'{ra} {dec:+f}', '-c.rs': sr})

            if res.status_code != 200:
                raise SourceError(f"the SED service answered {res.status_code}")

            try:
                from astropy.io.votable import parse
                table = parse(io.BytesIO(res.content)).get_first_table().to_table()
            except Exception as e:
                raise SourceError(f"the SED service returned something that is "
                                  f"not a VOTable ({e})")

            if len(table):
                cache.save(table)
            else:
                cache.data = None
        else:
            table = cache.data

    return cache.data


def _separations(table, ra, dec):
    """How far each row's source is from the target, in arcsec."""
    row_ra = np.asarray(_column(table, '_RAJ2000'), dtype=float)
    row_dec = np.asarray(_column(table, '_DEJ2000'), dtype=float)

    return 3600.0 * np.hypot((row_ra - ra) * np.cos(np.deg2rad(dec)),
                             row_dec - dec), row_ra, row_dec


def _one_source(table, rows, ra, dec, sr, log, name):
    """The rows of one catalogue that belong to the target rather than a neighbour.

    The nearest row fixes where the catalogue thinks the source is, and the
    rest of that source's bands are the rows sitting on the same place. Anything
    else at the position is another object, and its photometry would be read as
    the target's own were it kept.
    """
    separation, row_ra, row_dec = _separations(table[rows], ra, dec)

    inside = separation <= sr

    if not np.any(inside):
        return None, None

    nearest = np.argmin(np.where(inside, separation, np.inf))

    together = (3600.0 * np.hypot((row_ra - row_ra[nearest])
                                  * np.cos(np.deg2rad(dec)),
                                  row_dec - row_dec[nearest])) <= SED_SAME_SOURCE

    dropped = int(np.sum(inside & ~together))

    if dropped:
        log(f"  {name}: {dropped} row(s) belong to another source within"
            f" {sr:.1f} arcsec and are left out")

    return rows[together], float(separation[nearest])


def _points(table, entry, ra, dec, sr, log):
    """One catalogue's photometry at the target, as wavelength and flux."""
    tables = np.asarray([str(_) for _ in _column(table, '_tabname')])
    rows = np.where(tables == entry['table'])[0]

    if not len(rows):
        return None

    rows, separation = _one_source(table, rows, ra, dec, sr, log, entry['name'])

    if rows is None or not len(rows):
        return None

    here = table[rows]

    filters = np.asarray([str(_) for _ in _column(here, 'sed_filter')])

    # Only the bands this catalogue measured, where it also carries another's
    if entry.get('filters'):
        own = np.array([f.startswith(entry['filters']) for f in filters])

        if not np.all(own):
            log(f"  {entry['name']}: {int(np.sum(~own))} band(s) it republishes"
                f" from elsewhere are left to the catalogue they came from")

        here, filters = here[own], filters[own]

    if not len(here):
        return None

    frequency = np.asarray(_column(here, 'sed_freq'), dtype=float)
    flux = np.asarray(_column(here, 'sed_flux'), dtype=float)
    error = np.asarray(_column(here, 'sed_eflux'), dtype=float)

    # GHz to Angstrom, and Jansky to a flux per unit wavelength
    wavelength = SED_C / (frequency * 1e9)

    good = np.isfinite(wavelength) & np.isfinite(flux) & (wavelength > 0)

    if not np.any(good):
        return None

    # An uncertainty of exactly zero is one the catalogue did not give, rather
    # than a measurement known perfectly - Pan-STARRS publishes a few - and is
    # carried as absent so that nothing draws it as a point beyond doubt
    error = np.where(np.isfinite(error) & (error > 0), error, np.nan)

    rows = Table({
        'wavelength': wavelength[good],
        'flux': flambda_from_fnu(flux[good] * SED_JY, wavelength[good]),
        'flux_error': flambda_from_fnu(error[good] * SED_JY, wavelength[good]),
        'filter': filters[good],
        # Where the point came from, in one column. The viewer shows it when
        # the cursor is over the point, which is how a table of measurements
        # from a dozen surveys says which is which without a dozen colours.
        'comment': np.array([f"{entry['name']} {_}" for _ in filters[good]]),
    })

    return _deduplicate(rows, entry['name'], log), separation


def _deduplicate(rows, name, log):
    """One point per band, and an uncertainty that owns up to any disagreement.

    A catalogue can list a filter twice at the one source - Gaia's synthetic
    photometry gives two values for SDSS i, and 2MASS two for J - and drawn as
    they come the two sit above each other looking like a fault in the plot.
    The median stands for them.

    But a median alone would hide how far apart they were, and they are not
    always close: one target's two J values differ by a factor of two and a
    half, which is most of the near-infrared. So the spread widens the error
    bar where it exceeds what the catalogue quoted, and the point is drawn as
    badly determined - which it is - rather than as a confident average of two
    numbers that disagree.
    """
    keep = []

    for band in dict.fromkeys(np.asarray(rows['filter'], dtype=str)):
        here = np.asarray(rows['filter'], dtype=str) == band

        if np.sum(here) == 1:
            keep.append(rows[here][0])
            continue

        values = np.asarray(rows['flux'][here], dtype=float)
        errors = np.asarray(rows['flux_error'][here], dtype=float)

        middle = float(np.median(values))
        spread = float(np.max(values) - np.min(values)) / 2.0
        quoted = float(np.median(errors)) if np.any(np.isfinite(errors)) else np.nan

        log(f"  {name}: {band} is given {int(np.sum(here))} times, spanning"
            f" {np.max(values) / max(np.min(values), 1e-300):.2f}x - the median"
            f" stands for them, with the spread as its uncertainty")

        row = rows[here][0]
        row['flux'] = middle
        row['flux_error'] = (spread if not np.isfinite(quoted)
                             else max(spread, quoted))
        keep.append(row)

    out = Table(rows=keep, names=rows.colnames)
    out.sort('wavelength')

    return out


@survey_source(
    name='Catalogue SED',
    short_name='SED',
    state_acquiring='acquiring catalogue photometry',
    state_acquired='catalogue SED acquired',
    log_file='sed.log',
    output_files=['sed.log', 'sed.png', 'sed*.vot', 'sed*.txt'],
    button_text='Get catalogue SED',
    form_fields={
        'sed_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': SED_SR,
            'required': False,
        },
    },
    help_text='Broadband photometry from VizieR, 0.15 to 22 um',
    order=83,
    # Measurements and no curve, so the viewer draws them as points and they
    # keep the full weight of anything else it shows
    spectrum_points='sed*.txt',
    # The whole of what VizieR had is loaded but not opened on
    spectrum_hidden='sed_all.txt',
    spectrum_palette=['#16a085', '#7f8c8d'],
    template_layout='complex',
    main_plot='sed.png',
)
def target_sed(config, basepath=None, verbose=True, show=False):
    """
    Get the catalogue photometry at a position, as a spectral energy distribution.

    Parameters
    ----------
    config : dict
        Configuration dictionary with target coordinates
    basepath : str, optional
        Base path for output files
    verbose : bool or callable, optional
        Verbose logging mode or log function
    show : bool, optional
        Show plots interactively
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('sed'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = float(config.get('sed_sr') or SED_SR)

    log(f"within {sr:.1f} arcsec")

    table = _query(ra, dec, sr, basepath, log, refresh_cache)

    if table is None or not len(table):
        log("\nWarning: VizieR has no photometry at this position")
        return

    everything = collections.Counter(str(_) for _ in _column(table, '_tabname'))
    log(f"\n{len(table)} measurements at this position, from"
        f" {len(everything)} catalogues")

    # The named few, which is what the SED normally means here
    log("\n---- The catalogues taken ----\n")

    found = []

    for entry in SED_CATALOGUES:
        got = _points(table, entry, ra, dec, sr, log)

        if got is None:
            log(f"  {entry['name']:12s} nothing")
            continue

        rows, separation = got

        log(f"  {entry['name']:12s} {len(rows):2d} band(s),"
            f" {rows['wavelength'].min() / 1e4:.2f}-{rows['wavelength'].max() / 1e4:.2f} um,"
            f" {separation:.2f} arcsec away")

        found.append((entry, rows))

    # And everything else VizieR had there, gathered the same way. Kept
    # because the named few are a judgement, and one worth being able to look
    # behind: a target whose interest is in the far infrared, or in a survey
    # nobody thought to name here, has the rest of its photometry in the
    # second file rather than nowhere. It is written every time and shown on
    # request, the two being one query either way.
    rest = []

    for name in sorted(everything):
        if any(name == _['table'] for _ in SED_CATALOGUES):
            continue

        got = _points(table, {'table': name, 'name': name}, ra, dec, sr,
                      lambda *args, **kwargs: None)

        if got is not None:
            rest.append(got[0])

    curated = vstack([rows for _, rows in found]) if found else None

    pieces = ([curated] if curated is not None else []) + rest
    complete = vstack(pieces) if pieces else None

    if complete is None:
        log("\nWarning: none of the catalogues has this target")
        return

    if curated is not None:
        curated.sort('wavelength')
    if complete is not None:
        complete.sort('wavelength')

    log("\n---- Photometry ----\n")
    log(f"{'catalogue and band':32s} {'lambda, um':>11s} {'flux':>12s} {'error':>11s}")

    for row in (curated if curated is not None else complete):
        error = float(row['flux_error'])
        # A catalogue that publishes no uncertainty is common enough - Gaia's
        # synthetic photometry and half of Pan-STARRS - and is said as a dash
        # rather than as a number that is not there
        shown = f"{error:11.4g}" if np.isfinite(error) else f"{'-':>11s}"

        log(f"{str(row['comment']):32s} {float(row['wavelength']) / 1e4:11.4f}"
            f" {float(row['flux']):12.5g} {shown}")

    log("\nfluxes per unit wavelength, in erg/s/cm2/A")

    # Two tables of the same shape: the named catalogues, and every catalogue
    # at the position. One row per measurement, and a column saying which
    # survey and which band it came from - which is what lets the second file
    # hold a hundred points from forty surveys and still be read.
    columns = ['wavelength', 'flux', 'flux_error', 'comment']

    if curated is not None:
        write_spectrum(curated[columns], basepath, 'sed')
        log(f"\n{len(curated)} points from the named catalogues written to file:sed.vot")
        log(f"\n{len(curated)} points from the named catalogues written to file:sed.txt")

    if complete is not None:
        write_spectrum(complete[columns], basepath, 'sed_all')
        log(f"{len(complete)} points from all {len(everything)} catalogues written to file:sed_all.vot")
        log(f"{len(complete)} points from all {len(everything)} catalogues written to file:sed_all.txt")

    # Coloured by catalogue here, where the viewer tells them apart by what it
    # shows under the cursor - a plot has no cursor, and five colours is a
    # legend where forty would be a wall
    with plots.figure_saver(os.path.join(basepath, 'sed.png'),
                            figsize=(9, 5), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        for entry, rows in found:
            error = np.asarray(rows['flux_error'], dtype=float)

            ax.errorbar(np.asarray(rows['wavelength']) / 1e4,
                        np.asarray(rows['flux']),
                        np.where(np.isfinite(error), error, 0.0),
                        fmt='o', ms=5, lw=1, capsize=0, color=entry['colour'],
                        label=entry['name'])

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=0.2, which='both')
        ax.legend(fontsize=9)
        ax.set_xlabel('Wavelength, um')
        ax.set_ylabel(r'Flux, erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$')
        ax.set_title(f"{config['target_name']} - catalogue SED")

    log("Plotted in file:sed.png")
