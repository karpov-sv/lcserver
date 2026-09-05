"""IACOB spectroscopic database acquisition module.

High resolution spectra of Galactic massive OB stars, collected by the IACOB
project over more than a decade from FIES at the Nordic Optical Telescope,
HERMES at Mercator, and FEROS at La Silla. Around 2500 stars and 17000
spectra, at resolutions from 25000 to 85000 over roughly 3500-9200 A.

What it is for is the massive stars - O types, B supergiants, LBVs - that the
photometric surveys here see as one more variable point of light. Where a star
is in it at all there are usually several spectra of it, sometimes hundreds
over a decade, which is a record of how its wind and its line profiles moved.

The spectra are continuum normalised rather than flux calibrated. Each file
holds two planes, the normalised one and the raw counts it was divided by, and
it is the normalised plane that is taken: it is what the database exists to
provide, and it is what the lines are read from. So these are written without
a flux unit, and the spectral viewer loads them unticked - a curve running
about one would otherwise be the only thing visible beside spectra running at
1e-14, exactly as the Gaia RVS spectrum would.

There is no VO service. The database is a web form, and this fills it in: a
GET for the session and its CSRF token, a POST of the coordinates, and the
result table read out of the HTML that comes back. That is more fragile than a
TAP query and cannot be helped - it is the only way in.

Publishing anything based on these spectra carries an acknowledgement, which
IACOB_ACKNOWLEDGEMENT below records and every run of this step logs.
"""

import os
import io
import re
import ssl

import requests
from requests.adapters import HTTPAdapter

import numpy as np

import lxml.html

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files, KIND_SPECTROSCOPY
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    break_at_gaps, write_spectrum)


IACOB_URL = 'https://research.iac.es/proyecto/iacob/iacobcat/'

# One spectrum per call, by the database's own row id for it
IACOB_FILE = 'https://research.iac.es/proyecto/iacob/iacobcat/myspec/download-file/{}/0'

IACOB_TIMEOUT = 300

# What every paper using these spectra has to say, quoted here so that a user
# who exports one from this server finds it without having to go looking
IACOB_ACKNOWLEDGEMENT = (
    "The IACOB spectroscopic database is based on observations made with the"
    " Nordic Optical Telescope, operated by the Nordic Optical Telescope"
    " Scientific Association, and the Mercator Telescope, operated by the"
    " Flemish Community, both at the Observatorio del Roque de los Muchachos"
    " (La Palma, Spain) of the Instituto de Astrofisica de Canarias."
    " In addition, the paper must reference Simon-Diaz et al."
    " (2011a, 2011b, 2015)."
)

# Matching radius, in arcsec. Wider than it looks like it should be, and wider
# than the ESO step's five: IACOB carries a catalogue position for each star
# rounded to a tenth of a second in RA and a second in declination, which puts
# HR Car eighteen arcsec from where it actually is. A radius that would be
# generous for a pointing is too tight for that. Confusion is not much of a
# risk at this distance either - the database holds some 2500 massive stars
# over the whole sky, so two of them within an arcminute is rare.
IACOB_SR = 60.0

# How many to fetch, in total and from any one instrument. One star can have
# several hundred spectra here - epsilon Ori has 343 - at a few megabytes
# each, so a limit is not optional. Spread across instruments for the same
# reason the ESO step spreads across its own: a star seen by FIES, HERMES and
# FEROS is better described by some of each than by the nine best of whichever
# happened to observe it most.
IACOB_MAX_SPECTRA = 9
IACOB_MAX_PER_INSTRUMENT = 3

# Points kept per spectrum. R=85000 over 3800-9000 A is a third of a million
# samples, which no browser will draw and no one will look at that closely in
# a preview. The stored file is binned to this and the original stays in the
# archive, which is where anyone wanting the line profile itself should go.
IACOB_MAX_POINTS = 24000


class _Adapter(HTTPAdapter):
    """An HTTPS adapter that will still speak to research.iac.es.

    The server offers a Diffie-Hellman key below the size OpenSSL accepts by
    default, and a plain request fails outright with DH_KEY_TOO_SMALL. Only
    the cipher policy is relaxed, and only for this host: the certificate is
    verified as it would be anywhere else, so this weakens what the connection
    is encrypted with and not whether the server is who it claims to be.
    """

    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.set_ciphers('DEFAULT@SECLEVEL=1')
        kwargs['ssl_context'] = context

        return super().init_poolmanager(*args, **kwargs)


def _session():
    """A session holding the cookie the database hands out, and its token.

    Both the search and the downloads need it - the search because the form
    carries a CSRF token tied to the session, the downloads because they are
    refused without the cookie.
    """
    session = requests.Session()
    session.mount('https://', _Adapter())

    return session


def _token(session):
    """The CSRF token out of the search form, fetched fresh.

    Read from the page rather than remembered: it is minted per session, and a
    stale one is refused with a page that looks like an empty result rather
    than like an error.
    """
    try:
        page = session.get(IACOB_URL, timeout=IACOB_TIMEOUT)
        page.raise_for_status()
    except requests.RequestException as e:
        raise SourceError(f"could not reach IACOB - {type(e).__name__}: {e}")

    found = re.search(r'name="_csrfToken"[^>]*value="([^"]+)"', page.text)

    if not found:
        raise SourceError("unrecognised IACOB search page - it carries no"
                          " CSRF token, so the form cannot be submitted")

    return found.group(1)


def _stars(html):
    """The stars a result page describes, and the spectra under each.

    The page is one block per star - its name, position and spectral type in a
    heading, its spectra in a table below - so it is walked as that rather
    than as one flat table. A spectrum with no download link is dropped here:
    not everything compiled is public, and a row that cannot be fetched is
    only a row that would fail later.
    """
    tree = lxml.html.fromstring(html)
    stars = []

    for heading in tree.xpath("//div[contains(@class, 'alert-info')]"):
        link = heading.xpath(".//a[contains(@href, '/stars/view/')]")

        if not link:
            continue

        name = link[0].text_content().strip()
        star_id = link[0].get('href', '').rstrip('/').split('/')[-1]

        text = ' '.join(heading.text_content().split())

        position = re.search(r'\u03b1:\s*([\d:.+-]+)\s*\u03b4:\s*([\d:.+-]+)', text)
        sptype = re.search(r'SpC\s*:\s*(\S+)', text)
        vmag = re.search(r'V:\s*([-\d.]+)', text)

        # The spectra sit in a panel of their own, keyed by the star's id
        panel = tree.xpath(f"//div[@id='star-{star_id}']")

        rows = []

        for row in (panel[0].xpath('.//tr') if panel else []):
            cells = row.xpath('./td')

            if len(cells) < 6:
                continue

            download = row.xpath(".//a[contains(@href, 'download-file/')]")

            if not download:
                # Compiled for reference, but not public
                continue

            file_id = download[0].get('href').rstrip('/').split('/')[-2]

            values = [c.text_content().strip() for c in cells]

            rows.append({
                'instrument': values[0],
                'date': values[2],
                'snr': _number(values[3]),
                'exptime': _number(values[4]),
                'resolution': _number(values[5]),
                'file_id': file_id,
            })

        if rows:
            stars.append({
                'name': name,
                'position': position.groups() if position else None,
                'sptype': sptype.group(1) if sptype else '',
                'vmag': _number(vmag.group(1)) if vmag else np.nan,
                'rows': rows,
            })

    return stars


def _number(value):
    """A float out of a table cell, or NaN where it said nothing."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return np.nan


def _query(ra, dec, sr, log):
    """What IACOB has at a position, as one table of spectra.

    Where several stars fall inside the radius only the nearest is taken. The
    database is sparse enough that this is rare, and a second OB star an
    arcminute away is a different object rather than a second look at this
    one.
    """
    session = _session()

    data = {
        '_csrfToken': _token(session),
        '_method': 'POST',
        'name': '',
        'Coords': f'{ra:.6f} {dec:+.6f}',
        'radius': f'{sr:.0f}',
        'search_units': 'arcsec',
        'sp_class': '',
        'dr': '',
        'instrument': '',
        'minVmag': '', 'maxVmag': '',
        'minHJD': '', 'maxHJD': '',
    }

    try:
        answer = session.post(IACOB_URL, data=data, timeout=IACOB_TIMEOUT)
        answer.raise_for_status()
    except requests.RequestException as e:
        raise SourceError(f"could not query IACOB - {type(e).__name__}: {e}")

    stars = _stars(answer.text)

    if not stars:
        return None, session

    here = SkyCoord(ra, dec, unit='deg')

    def separation(star):
        if not star['position']:
            return np.inf

        try:
            where = SkyCoord(star['position'][0], star['position'][1],
                             unit=(u.hourangle, u.deg))
        except Exception:
            return np.inf

        return float(here.separation(where).to(u.arcsec).value)

    stars.sort(key=separation)

    if len(stars) > 1:
        log(f"\n{len(stars)} stars inside {sr:.0f} arcsec; taking the nearest,"
            f" and leaving " + ', '.join(_['name'] for _ in stars[1:]))

    star = stars[0]

    table = Table({
        key: [row[key] for row in star['rows']]
        for key in ('instrument', 'date', 'snr', 'exptime', 'resolution', 'file_id')
    })

    table['name'] = star['name']
    table['sptype'] = star['sptype']
    table['vmag'] = star['vmag']
    table['separation'] = separation(star)

    return table, session


def _chosen(found, log):
    """Which of the spectra to fetch, and in what order.

    The best signal to noise from each instrument first, so that a cut at the
    limit leaves a spread of instruments rather than the whole of one.
    """
    snr = np.asarray([_ if np.isfinite(_) else -1.0
                      for _ in np.asarray(found['snr'], dtype=float)])

    order = np.argsort(-snr, kind='stable')

    taken, seen = [], {}

    for index in order:
        row = found[int(index)]
        instrument = str(row['instrument'])

        if seen.get(instrument, 0) >= IACOB_MAX_PER_INSTRUMENT:
            continue

        seen[instrument] = seen.get(instrument, 0) + 1
        taken.append(row)

        if len(taken) >= IACOB_MAX_SPECTRA:
            break

    if len(taken) < len(found):
        log(f"\nFetching {len(taken)} of {len(found)}, the best signal to"
            f" noise of each instrument first")

    return taken


def _bin(table, maxpoints=IACOB_MAX_POINTS):
    """A spectrum binned down to a length worth keeping, and by how much.

    Whole blocks of neighbouring pixels are averaged, the wavelengths with
    them, so the grid stays the grid of what was measured. A trailing
    part-block is dropped rather than averaged over fewer points than the
    rest, being at most a few pixels at one end.
    """
    n = len(table)

    if n <= maxpoints:
        return table, 1

    factor = int(np.ceil(n / maxpoints))
    kept = (n // factor) * factor

    def blocks(values):
        return np.asarray(values, dtype=float)[:kept].reshape(-1, factor)

    binned = Table({
        'wavelength': np.mean(blocks(table['wavelength']), axis=1),
        'flux': np.mean(blocks(table['flux']), axis=1),
    })

    return binned, factor


def _fetch(session, file_id):
    """One spectrum, as its normalised plane against wavelength.

    The file is a two by N image: the first plane continuum normalised, the
    second the raw counts it was divided by. The first is what is taken - see
    the module docstring - and the wavelength grid comes from the header,
    there being no wavelength plane to read it from.
    """
    try:
        answer = session.get(IACOB_FILE.format(file_id), timeout=IACOB_TIMEOUT)
        answer.raise_for_status()
    except requests.RequestException as e:
        raise SourceError(f"could not download - {type(e).__name__}: {e}")

    if not answer.content:
        raise SourceError("the database returned an empty file")

    try:
        hdu = fits.open(io.BytesIO(answer.content))[0]
    except Exception as e:
        raise SourceError(f"unreadable FITS - {type(e).__name__}: {e}")

    data = np.asarray(hdu.data, dtype=float)

    if data.ndim == 2:
        flux = data[0]
    elif data.ndim == 1:
        flux = data
    else:
        raise SourceError(f"unexpected data shape {data.shape}")

    header = hdu.header

    step = header.get('CDELT1', header.get('CD1_1'))

    if not header.get('CRVAL1') or not step:
        raise SourceError("no wavelength scale in the header")

    wavelength = (header['CRVAL1']
                  + step * (np.arange(len(flux)) + 1 - header.get('CRPIX1', 1)))

    good = np.isfinite(wavelength) & np.isfinite(flux)

    if not np.any(good):
        raise SourceError("nothing measurable in it")

    return _bin(Table({'wavelength': wavelength[good], 'flux': flux[good]}))


@survey_source(
    name='IACOB',
    short_name='IACOB',
    state_acquiring='acquiring IACOB spectra',
    state_acquired='IACOB spectra acquired',
    log_file='iacob.log',
    output_files=['iacob.log', 'iacob_*.png', 'iacob_*.vot', 'iacob_*.txt'],
    button_text='Get IACOB spectra',
    form_fields={
        'iacob_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': IACOB_SR,
            'required': False,
        },
    },
    help_text='High resolution spectra of Galactic massive OB stars - '
              'FIES, HERMES and FEROS',
    order=86,
    about=(
        "The IACOB spectroscopic database - high resolution spectra of "
        "Galactic massive OB stars from FIES at the Nordic Optical "
        "Telescope, HERMES at Mercator and FEROS at La Silla. The spectra "
        "are continuum normalised rather than flux calibrated."),
    about_links=[
        ('IACOB database', 'https://research.iac.es/proyecto/iacob/iacobcat/'),
    ],
    acknowledgement=IACOB_ACKNOWLEDGEMENT,
    kind=KIND_SPECTROSCOPY,
    spectrum_files='iacob_*.txt',
    # Continuum normalised, so every one of them runs about one where the
    # calibrated spectra beside them run at 1e-14. Loaded and left unticked,
    # as the Gaia RVS spectrum is, until there is a viewer mode that puts
    # normalised spectra on an axis of their own.
    spectrum_hidden='iacob_*.txt',
    spectrum_palette=['#7d6608', '#9a7d0a', '#b7950b', '#d4ac0d', '#f1c40f',
                      '#f4d03f', '#f7dc6f'],
    template_layout='complex',
    additional_plots=['iacob_*.png'],
)
def target_iacob(config, basepath=None, verbose=True, show=False):
    """Acquire spectra from the IACOB spectroscopic database."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('iacob'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = float(config.get('iacob_sr', IACOB_SR))

    log(f"within {sr:.0f} arcsec")

    cache_name = f"iacob_{ra:.4f}_{dec:.4f}_{sr:.0f}.vot"

    session = None

    with cached_votable_query(cache_name, basepath, log, 'IACOB',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            found, session = _query(ra, dec, sr, log)

            if found is None:
                cache.save_empty()
                log("\nWarning: No IACOB spectra at this position - it holds "
                    "Galactic massive OB stars, and nothing else")
                return

            cache.save(found)

        found = cache.data

    # Nothing here, and cached as nothing - the helper has said so already
    if found is None:
        return

    # The search may have been answered from the cache, in which case no
    # session was opened; the downloads still need the cookie one carries
    if session is None:
        session = _session()
        session.get(IACOB_URL, timeout=IACOB_TIMEOUT)

    name = str(found['name'][0])
    separation = float(found['separation'][0])

    log(f"\n{name} at {separation:.1f} arcsec"
        + (f", {found['sptype'][0]}" if str(found['sptype'][0]) else '')
        + (f", V = {found['vmag'][0]:.2f}" if np.isfinite(found['vmag'][0]) else ''))

    log(f"{len(found)} public spectr{'um' if len(found) == 1 else 'a'}")

    log("\n---- What the database holds ----\n")

    for instrument in sorted(set(str(_) for _ in found['instrument'])):
        rows = found[np.asarray(found['instrument'], dtype=str) == instrument]

        snr = np.asarray(rows['snr'], dtype=float)
        snr = snr[np.isfinite(snr)]

        power = np.asarray(rows['resolution'], dtype=float)
        power = power[np.isfinite(power)]

        dates = sorted(str(_)[:10] for _ in rows['date'])

        log(f"{instrument:<10s} {len(rows):4d} spectra"
            + (f"  R ~ {np.nanmedian(power):.0f}" if len(power) else '')
            + (f"  S/N up to {snr.max():.0f}" if len(snr) else '')
            + (f"  {dates[0]} to {dates[-1]}" if dates else ''))

    log("")

    for row in _chosen(found, log):
        instrument = str(row['instrument'])
        when = str(row['date'])

        # The date is an ISO timestamp, which does not belong in a filename
        stem = ('iacob_' + instrument + '_'
                + re.sub(r'[^0-9]', '', when)[:14])

        with cached_votable_query(stem + '.vot', basepath, log,
                                  f'IACOB spectrum {row["file_id"]}',
                                  refresh=refresh_cache) as cache:
            if not cache.hit:
                try:
                    spectrum, factor = _fetch(session, str(row['file_id']))

                    if factor > 1:
                        log(f"  {stem}: binned by {factor} to"
                            f" {len(spectrum)} points")

                    # Inside the try, so that a fetch which failed is not
                    # remembered as a spectrum that does not exist
                    if spectrum is not None and len(spectrum):
                        cache.save(spectrum)
                    else:
                        cache.save_empty()
                        spectrum = None
                except SourceError as e:
                    log(f"  {stem}: {e}")
                    spectrum = None
                except Exception as e:
                    log(f"  {stem}: {type(e).__name__}: {e}")
                    spectrum = None
            else:
                spectrum = cache.data

        if spectrum is None or not len(spectrum):
            continue

        wavelength = np.asarray(spectrum['wavelength'], dtype=float)
        flux = np.asarray(spectrum['flux'], dtype=float)

        snr = float(row['snr'])

        log(f"  {instrument} {when[:19]}: {len(wavelength)} points from"
            f" {wavelength.min():.0f} to {wavelength.max():.0f} A"
            + (f", S/N = {snr:.0f}" if np.isfinite(snr) else ''))

        with plots.figure_saver(os.path.join(basepath, stem + '.png'),
                                figsize=(10, 4), show=show) as fig:
            ax = fig.add_subplot(1, 1, 1)

            # An echelle spectrum merged onto one grid can leave gaps between
            # its orders, which are not to be drawn across
            ax.plot(*break_at_gaps(wavelength, flux), '-', lw=0.5,
                    color='#b7950b')

            # Where the continuum is, by construction - the one line that says
            # what this spectrum is normalised to
            ax.axhline(1.0, ls=':', lw=1, color='#7f8c8d', alpha=0.7)

            ax.grid(alpha=0.2)
            ax.set_xlabel('Wavelength, A')
            ax.set_ylabel('Flux, normalised to the continuum')
            ax.set_title(f"{config['target_name']} - IACOB {instrument},"
                         f" {when[:10]}")

        write_spectrum(Table({'wavelength': wavelength, 'flux': flux}),
                       basepath, stem, calibrated=False)

        log(f"    Spectrum plotted in file:{stem}.png")
        log(f"    Spectrum written to file:{stem}.vot")
        log(f"    Spectrum written to file:{stem}.txt")

    log(f"\n{IACOB_ACKNOWLEDGEMENT}")
