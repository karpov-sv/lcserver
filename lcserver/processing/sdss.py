"""SDSS spectra acquisition module.

Not a light curve: SDSS is the largest optical spectroscopic survey there is,
and what it offers a target is one spectrum per time a fibre was put on it -
often more than one, since plates overlap and a field observed for the legacy
survey may have been observed again for BOSS or eBOSS years later. Every one
of them is fetched and drawn, and what the pipeline made of each is logged.

The spectra come through astroquery, which knows where a given plate, MJD and
fibre live and how the legacy and BOSS layouts differ. The cone search that
finds them returns the pipeline's classification, redshift and signal to
noise alongside, so the log can say what was observed without a second query.
"""

import os

import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

from astroquery.sdss import SDSS

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    write_spectrum)


# Which release to ask for. DR17 is the final SDSS-IV release and holds every
# optical spectrum the survey has taken - legacy, SEGUE, BOSS and eBOSS - so
# nothing is lost by not asking a later one.
#
# DR18 is not asked, though it exists: its SkyServer reports a reduction
# version for the eBOSS plates that its own file tree does not store them
# under, so every spectrum comes back a 404. The files are all present and
# correct under DR17, which also returns more rows for the same position,
# and reaching them means only naming the release that is consistent with
# itself rather than assembling paths here by hand.
SDSS_DR = 17

# Matching radius, in arcsec. An SDSS fibre subtends 3 arcsec and a BOSS one
# 2, so a match beyond this is a different object rather than a looser match
# to the same one.
SDSS_SR = 3.0

# astroquery refuses a cone wider than this, and there would be no sense in
# one: at three arcminutes the answer is a catalogue of the field
SDSS_MAX_SR = 170.0

# A field observed many times over is a spectrum and a plot per epoch, so only
# this many are fetched
SDSS_MAX_SPECTRA = 20

# What the pipeline reports, where it reports it. Asked for by name rather
# than taken as they come: the default set carries the photometry of the
# object and almost nothing of the spectrum.
SDSS_PHOTO_FIELDS = ['ra', 'dec']
SDSS_SPEC_FIELDS = ['specobjid', 'plate', 'mjd', 'fiberID', 'run2d',
                    'instrument', 'class', 'subclass', 'z', 'zErr', 'snMedian']

# SDSS publishes its fluxes in units of 1e-17 erg/s/cm2/A, where every
# spectrum here is written in erg/s/cm2/A outright
SDSS_TO_CGS = 1e-17

# Speed of light in km/s, to read a redshift as a velocity
SDSS_C = 299792.458


def _designation(row):
    """The plate-MJD-fibre a spectrum is known by, as SDSS writes it."""
    return f"{int(row['plate']):04d}-{int(row['mjd']):05d}-{int(row['fiberID']):04d}"


def _value(row, key):
    """One field of a row, or None where it is absent or not a number."""
    if key not in row.colnames:
        return None

    value = row[key]

    if value is np.ma.masked:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if np.isfinite(value) else None


def _text(row, key):
    """One field of a row as a word, or None where it says nothing.

    The pipeline leaves a subclass empty for anything it did not subdivide,
    and the column then arrives typed as a number and wholly masked, so an
    absent value has to be recognised rather than merely printed.
    """
    if key not in row.colnames:
        return None

    value = row[key]

    if value is np.ma.masked:
        return None

    value = str(value).strip()

    return value or None


def _votable_safe(table):
    """The query's answer, in types a VOTable can hold.

    SDSS identifiers are unsigned 64-bit integers, which VOTable has no type
    for - it stops at a signed long. They are names rather than quantities,
    nothing here does arithmetic on them, and the largest of them would not
    fit a signed long anyway, so they are kept as the digits they are.
    """
    table = table.copy()

    for name in table.colnames:
        if table[name].dtype.kind == 'u' and table[name].dtype.itemsize >= 8:
            table[name] = np.asarray([str(_) for _ in table[name]])

    return table


def _fetch_spectrum(row):
    """One spectrum, as the survey coadded it.

    The file holds the coadd and every individual exposure behind it; the
    coadd is the one worth drawing, and the one the pipeline classified.
    Returned as published - fluxes in units of 1e-17 erg/s/cm2/A and an
    inverse variance - with the conversion left to the caller, so that what
    is cached is what the survey said.
    """
    # A one-row table of exactly the columns astroquery matches on
    match = Table({key: [row[key]] for key in
                   ('plate', 'mjd', 'fiberID', 'run2d', 'instrument')
                   if key in row.colnames})

    got = SDSS.get_spectra(matches=match, data_release=SDSS_DR)

    if not got:
        return None

    with got[0] as hdus:
        if 'COADD' not in hdus:
            return None

        data = hdus['COADD'].data

        return Table({
            # Stored as the log of the wavelength, which is what makes the
            # sampling uniform in velocity rather than in wavelength
            'wavelength': 10 ** np.asarray(data['loglam'], dtype=float),
            'flux': np.asarray(data['flux'], dtype=float),
            'ivar': np.asarray(data['ivar'], dtype=float),
            'and_mask': np.asarray(data['and_mask'], dtype=np.int64),
        })


@survey_source(
    name='SDSS',
    short_name='SDSS',
    state_acquiring='acquiring SDSS spectra',
    state_acquired='SDSS spectra acquired',
    log_file='sdss.log',
    output_files=['sdss.log', 'sdss_*.png', 'sdss_*.vot', 'sdss_*.txt'],
    button_text='Get SDSS spectra',
    form_fields={
        'sdss_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': SDSS_SR,
            'required': False,
        },
    },
    help_text=f'SDSS DR{SDSS_DR} optical spectra, 3600-10400 A',
    # First of the spectra, being the largest collection of them
    order=79,
    spectrum_files='sdss_*.txt',
    # A target may have several, one per time a fibre was put on it
    spectrum_palette=['#5b2c6f', '#7d3c98', '#9b59b6', '#af7ac5', '#c39bd3'],
    template_layout='complex',
    additional_plots=['sdss_*.png'],
)
def target_sdss(config, basepath=None, verbose=True, show=False):
    """Acquire SDSS spectra."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('sdss'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = min(float(config.get('sdss_sr', SDSS_SR)), SDSS_MAX_SR)

    log(f"within {sr:.1f} arcsec")

    cache_name = f"sdss_{ra:.4f}_{dec:.4f}_{sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'SDSS spectra',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            try:
                found = SDSS.query_region(
                    SkyCoord(ra, dec, unit='deg'), radius=sr * u.arcsec,
                    spectro=True, data_release=SDSS_DR,
                    photoobj_fields=SDSS_PHOTO_FIELDS,
                    specobj_fields=SDSS_SPEC_FIELDS)
            except Exception as e:
                raise SourceError("could not query SDSS - "
                                  f"{type(e).__name__}: {e}")

            if found is None or not len(found):
                cache.save_empty()
                log("\nWarning: No SDSS spectra at this position - the survey "
                    "took one only where a fibre was put on the object, which "
                    "is far from everywhere it imaged")
                return

            cache.save(_votable_safe(found))

        found = cache.data

    # Nothing here, and cached as nothing - the helper has said so already
    if found is None:
        return

    log(f"\n{len(found)} SDSS spectr{'um' if len(found) == 1 else 'a'}")

    # What the pipeline made of each of them
    log("\n---- What SDSS made of it ----\n")

    for row in found:
        redshift, error = _value(row, 'z'), _value(row, 'zErr')
        snr = _value(row, 'snMedian')

        kind = _text(row, 'class')
        subkind = _text(row, 'subclass')

        log(f"{_designation(row)}  {_text(row, 'instrument') or 'SDSS'}"
            + (f"  {kind}" if kind else '')
            + (f" {subkind}" if subkind else '')
            + (f"  S/N = {snr:.0f}" if snr is not None else ''))

        if redshift is not None:
            # A redshift is what the pipeline fits whatever it is looking at,
            # so it is read back as a velocity only where the thing is a star
            # - quoting one for a galaxy at z = 0.17 would be nonsense
            log(f"    z = {redshift:.6f}"
                + (f" +/- {error:.6f}" if error is not None else '')
                + (f", which is {redshift * SDSS_C:+.1f} km/s"
                   if kind == 'STAR' else ''))

    wanted = list(found)

    if len(wanted) > SDSS_MAX_SPECTRA:
        log(f"\n{len(wanted)} spectra, of which the first {SDSS_MAX_SPECTRA} are fetched")
        wanted = wanted[:SDSS_MAX_SPECTRA]
    else:
        log(f"\nFetching {len(wanted)} spectr{'um' if len(wanted) == 1 else 'a'}")

    for row in wanted:
        designation = _designation(row)
        spec_cache = f"sdss_spec_{designation}.vot"

        with cached_votable_query(spec_cache, basepath, log,
                                  f'SDSS spectrum {designation}',
                                  refresh=refresh_cache) as cache:
            if not cache.hit:
                try:
                    spectrum = _fetch_spectrum(row)

                    # Inside the try, so that a fetch which failed is not
                    # remembered as a spectrum that does not exist
                    if spectrum is not None and len(spectrum):
                        cache.save(spectrum)
                    else:
                        cache.save_empty()
                        spectrum = None
                except Exception as e:
                    log(f"  {designation}: {type(e).__name__}: {e}")
                    spectrum = None
            else:
                spectrum = cache.data

        if spectrum is None or not len(spectrum):
            continue

        wavelength = np.asarray(spectrum['wavelength'], dtype=float)
        flux = np.asarray(spectrum['flux'], dtype=float)
        ivar = np.asarray(spectrum['ivar'], dtype=float)

        dead = int(np.sum(ivar <= 0))

        log(f"  {designation}: {len(wavelength)} points from"
            f" {wavelength.min():.0f} to {wavelength.max():.0f} A"
            + (f", {dead} of them with no weight" if dead else ''))

        name = f"sdss_{designation}"

        with plots.figure_saver(os.path.join(basepath, name + '.png'),
                                figsize=(10, 4), show=show) as fig:
            ax = fig.add_subplot(1, 1, 1)

            ax.plot(wavelength, flux, '-', lw=0.6, color='#7d3c98')

            ax.grid(alpha=0.2)
            ax.set_xlabel('Wavelength, A')
            ax.set_ylabel(r'Flux, $10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$')
            ax.set_title(f"{config['target_name']} - SDSS {designation}")

        # The survey's own scale, into the one every spectrum here is on, and
        # its inverse variance as the uncertainty the others all publish. The
        # cache above keeps both as SDSS served them. A pixel of no weight is
        # one the pipeline could not measure, and its error is left undefined
        # rather than infinite.
        with np.errstate(divide='ignore', invalid='ignore'):
            error = np.where(ivar > 0, SDSS_TO_CGS / np.sqrt(ivar), np.nan)

        write_spectrum(Table({
            'wavelength': wavelength,
            'flux': flux * SDSS_TO_CGS,
            'flux_error': error,
        }), basepath, name)

        log(f"    Spectrum plotted in file:{name}.png")
        log(f"    Spectrum written to file:{name}.vot")
        log(f"    Spectrum written to file:{name}.txt")
