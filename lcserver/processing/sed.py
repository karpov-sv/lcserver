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
are named in the reply. The same table, fetched separately, says how wide each
band is, which is the difference between drawing these as points and drawing
them as what they are - a flux averaged across a range of wavelength.

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
from astroquery.vizier import Vizier

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    flambda_from_fnu, shared_cache_dir, write_spectrum)


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

# VizieR's own two tables of the filters it quotes: METAfltr is one row per
# filter, with the effective wavelength and the effective width; METAphot names
# the photometric systems those filters belong to. The pair is what the SED
# service itself works from, and joining them gives back exactly the
# designations it puts in sed_filter - 'photoSystem:filterName', which is
# METAphot's name and METAfltr's filter with a colon between.
#
# The alternative was the SVO filter profile service, which has these widths
# and more besides. It was not taken: its identifiers are not VizieR's, so half
# the bands would need an alias table maintained here by hand, and every band
# would rest on a second service being up. This way the widths come from the
# same place as the wavelengths they belong to, in one query, and agree with
# them by construction - lambda0 here and the wavelength derived from sed_freq
# differ by under 1.5 per cent across every band of every target tried, which
# is why the width can simply be hung on the wavelength the reply gave.
SED_FILTER_TABLES = ('METAphot', 'METAfltr')

# The join of those two, kept as the one small table it reduces to. Cached
# whole and shared between targets, being the same table for all of them; the
# name carries what is in it, so that changing what is kept means a new file
# rather than a stale one read as the new thing.
SED_FILTER_CACHE = 'vizier_filter_widths.vot'

# And held in the process as well, since every target wants the same four
# thousand rows and reading them off disk for each is work for nothing
_filter_widths_cache = None


def _filled(column):
    """A column as plain floats, whatever it left unset becoming a nan."""
    return np.asarray(column.filled(np.nan) if hasattr(column, 'filled')
                      else column, dtype=float)


def _fetch_filter_widths(log):
    """VizieR's filter table, as designation and width in Angstrom."""
    log("Querying VizieR for its filter table (METAphot and METAfltr)...")

    vizier = Vizier(columns=['**'], row_limit=-1)

    systems = vizier.get_catalogs(SED_FILTER_TABLES[0])[0]
    filters = vizier.get_catalogs(SED_FILTER_TABLES[1])[0]

    names = dict(zip(np.asarray(systems['photid'], dtype=int),
                     [str(_).strip() for _ in systems['name']]))

    # Filled rather than read row by row: a few hundred filters have no width
    # given, and converting a masked element one at a time is a warning apiece
    # for something this expects and drops
    band_width = _filled(filters['dlambda'])

    designation, width = [], []

    for photid, band, this in zip(np.asarray(filters['photid'], dtype=int),
                                  [str(_).strip() for _ in filters['filter']],
                                  band_width * 1e4):  # microns to Angstrom
        system = names.get(photid, '')

        if not system or not band or not np.isfinite(this) or this <= 0:
            continue

        designation.append(f'{system}:{band}')
        width.append(this)

    if not designation:
        raise SourceError("VizieR's filter table came back with nothing usable")

    return Table({'filter': np.array(designation), 'width': np.array(width)})


def _filter_widths(basepath, log, refresh=False):
    """How wide each band is, keyed as the SED service names it.

    Never fatal: a target whose photometry is in hand is not worth failing
    over the width of the bands it was measured in, so anything going wrong
    here leaves the SED to be drawn without them.
    """
    global _filter_widths_cache

    if _filter_widths_cache is not None and not refresh:
        return _filter_widths_cache

    path = os.path.join(shared_cache_dir(basepath), SED_FILTER_CACHE)

    table = None

    if os.path.exists(path) and not refresh:
        try:
            table = Table.read(path)
            log(f"Loading VizieR's filter table from cache ({SED_FILTER_CACHE})")
        except Exception as e:
            # The class of it and not the message: astropy answers an
            # unreadable table with its whole list of supported formats
            log(f"Cannot read the cached filter table ({type(e).__name__}),"
                " asking VizieR again")

    if table is None:
        try:
            table = _fetch_filter_widths(log)
        except Exception as e:
            log(f"\nWarning: cannot get VizieR's filter table ({e})"
                " - the SED will be drawn without band widths")
            _filter_widths_cache = {}
            return _filter_widths_cache

        # Written where it lands rather than in place: several targets are
        # acquired at once, and two of them finding the cache missing at the
        # same moment would otherwise leave each other half a file to read
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temporary = f'{path}.{os.getpid()}.tmp'
            table.write(temporary, format='votable', overwrite=True)
            os.replace(temporary, path)
            log(f"Cached VizieR's filter table to {SED_FILTER_CACHE}")
        except Exception as e:
            log(f"Cannot cache the filter table ({e}), which is not fatal")

    # First of a designation wins. Thirty-seven of four thousand are given
    # twice - narrow-band HST and Gemini filters, mostly the same number twice
    # over, and none of them a band any catalogue here reports in.
    widths = {}

    for name, this in zip(np.asarray(table['filter'], dtype=str),
                          np.asarray(table['width'], dtype=float)):
        if np.isfinite(this) and this > 0:
            widths.setdefault(str(name), float(this))

    log(f"{len(widths)} filters have a width")

    _filter_widths_cache = widths

    return widths


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
                cache.save_empty()
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


def _points(table, entry, ra, dec, sr, log, widths):
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

    # Where each of these was measured, kept alongside what was measured. Two
    # catalogues agreeing on a flux while disagreeing on where it came from is
    # the thing worth seeing, and it cannot be seen once the positions are
    # dropped. Left out of the spectrum files, which are written by column.
    row_ra = np.asarray(_column(here, '_RAJ2000'), dtype=float)
    row_dec = np.asarray(_column(here, '_DEJ2000'), dtype=float)

    # GHz to Angstrom, and Jansky to a flux per unit wavelength
    wavelength = SED_C / (frequency * 1e9)

    good = np.isfinite(wavelength) & np.isfinite(flux) & (wavelength > 0)

    if not np.any(good):
        return None

    # An uncertainty of exactly zero is one the catalogue did not give, rather
    # than a measurement known perfectly - Pan-STARRS publishes a few - and is
    # carried as absent so that nothing draws it as a point beyond doubt
    error = np.where(np.isfinite(error) & (error > 0), error, np.nan)

    # How wide the band is, where VizieR's filter table names it. A broadband
    # point is not a measurement at a wavelength but one across a range of
    # them, and the ranges are not small next to the spacing - W3 is five and a
    # half microns wide, half the wavelength it sits at, and drawn as a dot it
    # claims a precision in wavelength that nothing about it has. Carried as
    # the full effective width, the same as SPHEREx carries its own, and halved
    # by whatever draws the bar.
    #
    # A band the table does not have gets no width rather than a guessed one,
    # and is drawn as a plain point.
    bandwidth = np.array([widths.get(str(_), np.nan) for _ in filters])

    rows = Table({
        'wavelength': wavelength[good],
        'bandwidth': bandwidth[good],
        'flux': flambda_from_fnu(flux[good] * SED_JY, wavelength[good]),
        'flux_error': flambda_from_fnu(error[good] * SED_JY, wavelength[good]),
        'filter': filters[good],
        'ra': row_ra[good],
        'dec': row_dec[good],
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


def _offsets(row_ra, row_dec, ra, dec):
    """Where these sit relative to the target, in arcsec east and north."""
    return (3600.0 * (row_ra - ra) * np.cos(np.deg2rad(dec)),
            3600.0 * (row_dec - dec))


def _position_groups(row_ra, row_dec, dec, tolerance=SED_SAME_SOURCE):
    """The distinct positions among these rows, as lists of row indices.

    Grouped by the same distance that decides whether two rows of a catalogue
    are one source, so that what the plot separates and what the photometry
    separates are the same thing. Greedy and order-dependent, which is enough
    for telling two objects an arcsecond apart from one object measured twice.
    """
    groups, centres = [], []

    for i in range(len(row_ra)):
        for g, (centre_ra, centre_dec) in enumerate(centres):
            near = 3600.0 * np.hypot((row_ra[i] - centre_ra)
                                     * np.cos(np.deg2rad(dec)),
                                     row_dec[i] - centre_dec)
            if near <= tolerance:
                groups[g].append(i)
                break
        else:
            centres.append((row_ra[i], row_dec[i]))
            groups.append([i])

    return groups, centres


def _report_positions(table, found, ra, dec, sr, log):
    """Say how many distinct objects the reply actually describes.

    The photometry is drawn against wavelength, where a neighbour's flux looks
    like the target's own with nothing to mark it. Counted here instead: if the
    rows fall on three positions rather than one, there are three objects here,
    and what matters then is which of them each catalogue was matched to.
    """
    _, row_ra, row_dec = _separations(table, ra, dec)

    groups, centres = _position_groups(row_ra, row_dec, dec)

    tables = np.asarray([str(_) for _ in _column(table, '_tabname')])

    log(f"\n{len(groups)} distinct position(s) among {len(table)} measurements,"
        f" within {SED_SAME_SOURCE:.1f} arcsec of each other:\n")

    spots = []

    for g, (centre_ra, centre_dec) in enumerate(centres):
        east, north = _offsets(centre_ra, centre_dec, ra, dec)
        spots.append((float(np.hypot(east, north)), east, north, groups[g]))

    for distance, east, north, rows in sorted(spots):
        log(f"  {distance:5.2f} arcsec away"
            f" ({east:+6.2f}, {north:+6.2f}): {len(rows):4d} measurement(s)"
            f" from {len(set(tables[np.asarray(rows)]))} catalogue(s)")

    if not found:
        return

    # Which of those objects the SED was actually built from. Each named
    # catalogue was matched to its own nearest source, independently of the
    # others, so nothing has yet required them to agree - and where they do
    # not, the SED is two objects' photometry drawn as one spectrum.
    log("\nThe catalogues the SED is built from sit at:\n")

    places = []

    for entry, rows in found:
        east, north = _offsets(float(np.median(rows['ra'])),
                               float(np.median(rows['dec'])), ra, dec)

        log(f"  {entry['name']:14s} {np.hypot(east, north):5.2f} arcsec away"
            f" ({east:+6.2f}, {north:+6.2f})")

        places.append((entry['name'], east, north))

    apart = max((float(np.hypot(a[1] - b[1], a[2] - b[2]))
                 for a in places for b in places), default=0.0)

    # Judged against the radius searched rather than against SED_SAME_SOURCE.
    # That half-arcsecond separates two rows of one catalogue, where the same
    # instrument measured both; between catalogues it means nothing, since they
    # do not centroid alike - GALEX and AllWISE see with beams several arcsec
    # across, and will disagree with Gaia by a fraction of one on an object
    # neither is confused about. Warning at that distance would warn on nearly
    # every target, which is the same as not warning at all.
    if apart > sr:
        log(f"\nWarning: these span {apart:.2f} arcsec, more than the"
            f" {sr:.1f} arcsec searched - two of them"
            "\nhave been matched to different objects, and the SED is of both."
            "\nSee file:sed_positions.png.")
    elif apart > SED_SAME_SOURCE:
        log(f"\nThese span {apart:.2f} arcsec. The coarser catalogues do not"
            " centroid to better than\nthat, so it need not be a second object"
            " - file:sed_positions.png is where to tell.")
    else:
        log(f"\nAll within {apart:.2f} arcsec of each other, so they are one"
            " object.")


def _plot_positions(table, found, ra, dec, sr, basepath, config, log, show=False):
    """Where every measurement was made, against where the target is.

    The SED itself cannot show this. It is drawn against wavelength, and a
    point belonging to a neighbour an arcsecond away sits in it looking exactly
    like one belonging to the target - which is how a blend is read as an
    infrared excess. Here the same measurements are drawn against the sky
    instead: everything the service returned in the background, and the points
    the SED was actually built from over it. One tight knot is one object;
    several knots are several, and the SED is only as good as which knot the
    catalogues each landed on.
    """
    separation, row_ra, row_dec = _separations(table, ra, dec)
    east, north = _offsets(row_ra, row_dec, ra, dec)

    # Scaled to the circle, not to the furthest row. The service returns rows
    # well outside the radius it was asked for, and one of those at eight
    # arcsec would squeeze everything this plot is about into a thumbprint at
    # the centre. Rows beyond the circle were never candidates for the SED
    # anyway; they are simply left off the edge, and counted in the log. Where
    # the whole field sits well inside the circle there is nothing to see at
    # that scale either, so then it zooms in instead.
    inside = separation <= sr
    spread = (float(np.max(np.abs(np.concatenate([east[inside], north[inside]]))))
              if np.any(inside) else sr)

    reach = max(spread * 1.6, 0.3) if spread < sr / 3 else sr * 1.15

    with plots.figure_saver(os.path.join(basepath, 'sed_positions.png'),
                            figsize=(5.5, 5.5), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        # Everything at the position, including the rows the catalogues were
        # not matched to - those are the neighbours, and leaving them out would
        # hide the very thing this plot is for
        ax.plot(east, north, 'o', ms=7, alpha=0.15, color='#7f8c8d',
                mew=0, zorder=1,
                label=f'all measurements ({len(table)})')

        # Drawn large to small, so that catalogues agreeing on a position stay
        # visible through one another as a bullseye rather than the last one
        # drawn hiding the rest. Which makes disagreement look like what it is:
        # the rings come apart.
        for i, (entry, rows) in enumerate(found):
            entry_east, entry_north = _offsets(np.asarray(rows['ra'], dtype=float),
                                               np.asarray(rows['dec'], dtype=float),
                                               ra, dec)

            ax.plot(entry_east, entry_north, 'o',
                    ms=12 - 7.0 * i / max(len(found) - 1, 1), alpha=1.0,
                    color=entry['colour'], mew=0.5, mec='white', zorder=3 + i,
                    label=entry['name'])

        # The position everything is measured against
        # ax.plot(0, 0, '+', ms=13, mew=1.5, color='black', zorder=4, label='target')

        # What was asked for. Rows outside it are in the reply because the
        # service barely honours the radius, and are dropped by the cut made
        # here - so the circle says which of these points could be taken at all.
        # Named in the legend only where the view reaches it, a legend entry
        # for something off the edge of the plot being a puzzle rather than a
        # help.
        turn = np.linspace(0, 2 * np.pi, 200)
        ax.plot(sr * np.cos(turn), sr * np.sin(turn), '--', lw=1,
                color='#7f8c8d', alpha=0.6, zorder=2,
                label=f'search radius, {sr:.1f}"' if reach >= sr else None)

        ax.set_xlim(reach, -reach)  # east to the left, as the sky is drawn
        ax.set_ylim(-reach, reach)
        ax.set_aspect('equal')
        ax.grid(alpha=0.2)
        # Below the field rather than in it: the points are about the centre,
        # where every corner a legend could take is somewhere a neighbour might
        # be, and a neighbour hidden by the legend is the one thing this plot
        # must not do
        ax.legend(fontsize=8, ncol=3, loc='upper center',
                  bbox_to_anchor=(0.5, -0.12), frameon=False)
        ax.set_xlabel(r'$\Delta$RA $\cos\delta$, arcsec  (east left)')
        ax.set_ylabel(r'$\Delta$Dec, arcsec')
        ax.set_title(f"{config['target_name']} - where the photometry is")

    log("Positions plotted in file:sed_positions.png")


def _gaia_xp(basepath, log):
    """The Gaia XP spectrum, where the info step left one.

    Drawn under the SED as a reference: it is a measured spectrum over the
    optical, in the same units, and the broadband points ought to lie along it.
    Where they do not, either the photometry belongs to something else or the
    object varies between the epochs the two were taken at.
    """
    path = os.path.join(basepath, 'gaia_xp.vot')

    if not os.path.exists(path):
        return None

    try:
        xp = Table.read(path)
    except Exception as e:
        # Never fatal: the SED is the point of this step, and the reference
        # curve is worth none of it
        log(f"Cannot read the Gaia XP spectrum ({type(e).__name__}),"
            " leaving it off the plot")
        return None

    if 'wavelength' not in xp.colnames or 'flux' not in xp.colnames:
        return None

    wavelength = np.asarray(xp['wavelength'], dtype=float)
    flux = np.asarray(xp['flux'], dtype=float)

    # Both axes are logarithmic, where a flux that came out negative at the
    # noisy end of the spectrum has nowhere to be drawn
    good = np.isfinite(wavelength) & np.isfinite(flux) & (wavelength > 0) & (flux > 0)

    if not np.any(good):
        return None

    return wavelength[good], flux[good]


@survey_source(
    name='Catalogue SED',
    short_name='SED',
    state_acquiring='acquiring catalogue photometry',
    state_acquired='catalogue SED acquired',
    log_file='sed.log',
    output_files=['sed.log', 'sed*.png', 'sed*.vot', 'sed*.txt'],
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
    # Last of the spectral block: it is assembled out of other catalogues'
    # photometry rather than observed, and spans all of them
    order=89,
    # Measurements and no curve, so the viewer draws them as points and they
    # keep the full weight of anything else it shows
    spectrum_points='sed*.txt',
    # The whole of what VizieR had is loaded but not opened on
    spectrum_hidden='sed_all.txt',
    spectrum_palette=['#16a085', '#7f8c8d'],
    template_layout='complex',
    main_plot='sed.png',
    # Where the photometry came from on the sky, which the SED cannot show
    additional_plots=['sed_positions.png'],
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

    log("\n---- Band widths ----\n")

    widths = _filter_widths(basepath, log, refresh_cache)

    # The named few, which is what the SED normally means here
    log("\n---- The catalogues taken ----\n")

    found = []

    for entry in SED_CATALOGUES:
        got = _points(table, entry, ra, dec, sr, log, widths)

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
                      lambda *args, **kwargs: None, widths)

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
    log(f"{'catalogue and band':32s} {'lambda, um':>11s} {'width, um':>10s}"
        f" {'flux':>12s} {'error':>11s}")

    for row in (curated if curated is not None else complete):
        error = float(row['flux_error'])
        # A catalogue that publishes no uncertainty is common enough - Gaia's
        # synthetic photometry and half of Pan-STARRS - and is said as a dash
        # rather than as a number that is not there
        shown = f"{error:11.4g}" if np.isfinite(error) else f"{'-':>11s}"

        band = float(row['bandwidth'])
        wide = f"{band / 1e4:10.4f}" if np.isfinite(band) else f"{'-':>10s}"

        log(f"{str(row['comment']):32s} {float(row['wavelength']) / 1e4:11.4f}"
            f" {wide} {float(row['flux']):12.5g} {shown}")

    log("\nfluxes per unit wavelength, in erg/s/cm2/A")
    log("widths are the effective width of the band, VizieR's own")

    log("\n---- Where the photometry is ----")

    _report_positions(table, found, ra, dec, sr, log)

    # Two tables of the same shape: the named catalogues, and every catalogue
    # at the position. One row per measurement, and a column saying which
    # survey and which band it came from - which is what lets the second file
    # hold a hundred points from forty surveys and still be read.
    columns = ['wavelength', 'bandwidth', 'flux', 'flux_error', 'comment']

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

        # Underneath, where it is something to read the points against rather
        # than another series among them. In the blue Gaia's synthetic
        # photometry is drawn in, those points being this same spectrum
        # summarised into bands.
        xp = _gaia_xp(basepath, log)

        if xp is not None:
            xp_wavelength, xp_flux = xp
            ax.plot(xp_wavelength / 1e4, xp_flux, '-', lw=1, alpha=0.4,
                    color='#2980b9', zorder=1, label='Gaia XP')

        for entry, rows in found:
            error = np.asarray(rows['flux_error'], dtype=float)
            band = np.asarray(rows['bandwidth'], dtype=float)

            ax.errorbar(np.asarray(rows['wavelength']) / 1e4,
                        np.asarray(rows['flux']),
                        np.where(np.isfinite(error), error, 0.0),
                        # Half the width to either side, so that the bar spans
                        # the band. In microns, as the axis is.
                        xerr=np.where(np.isfinite(band), band, 0.0) / 2e4,
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

    _plot_positions(table, found, ra, dec, sr, basepath, config, log, show=show)
