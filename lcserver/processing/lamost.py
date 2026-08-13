"""LAMOST spectra acquisition module.

Not a light curve: LAMOST is a spectroscopic survey, and what it offers a
variable star is a handful of epochs, each a spectrum and - for the
medium-resolution survey - a radial velocity. Both are worth having beside the
photometry, so this source fetches every spectrum there is and plots each of
them, with the velocities against time where there is more than one epoch.

The spectra themselves are not in Vizier. Its DR11 tables list the
observations, with a link out per row, so the identifiers come from Vizier and
the data from LAMOST's own server, one request per observation.
"""

import os
import io
import gzip
import requests
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

from astroquery.vizier import Vizier

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import cleanup_paths, cached_votable_query, write_spectrum


# The observation lists: low resolution, and medium resolution
LAMOST_LRS_CATALOGUE = 'V/162/dr11l'
LAMOST_MRS_CATALOGUE = 'V/162/dr11m'

# Matching radius, in arcsec. LAMOST fibres are 3.3 arcsec across.
LAMOST_SR = 3.0

# Where the spectra live, by resolution. The low-resolution path returns a
# medium-resolution observation as an error, and the other way round.
LAMOST_URLS = {
    'low': 'https://www.lamost.org/dr11/v1.0/spectrum/fits/{}',
    'medium': 'https://www.lamost.org/dr11/v1.0/medspectrum/fits/{}',
}

# A well-visited star can have a great many epochs, and each is a request and a
# plot, so only this many are drawn
LAMOST_MAX_SPECTRA = 20

# LAMOST publishes its fluxes in units of 1e-17 erg/s/cm2/A, where every
# spectrum here is written in erg/s/cm2/A outright
LAMOST_TO_CGS = 1e-17


def _fetch_spectrum(obsid, resolution, log):
    """One observation's spectrum, as a table of wavelength, flux and arm.

    The server answers a missing spectrum with a JSON error under a status of
    200, so what came back has to be looked at rather than merely counted.
    """
    res = requests.get(LAMOST_URLS[resolution].format(obsid), timeout=120)

    if res.content[:2] != b'\x1f\x8b' and not res.content.startswith(b'SIMPLE'):
        log(f"  {obsid}: no spectrum returned ({res.content[:80].decode('utf-8', 'replace')})")
        return None

    raw = gzip.decompress(res.content) if res.content[:2] == b'\x1f\x8b' else res.content

    rows = []

    with fits.open(io.BytesIO(raw)) as hdus:
        for hdu in hdus:
            if not hasattr(hdu, 'columns') or not hdu.columns:
                continue

            names = [_.name for _ in hdu.columns]

            if 'WAVELENGTH' not in names or 'FLUX' not in names:
                continue

            # The medium-resolution files hold the coadded arms alongside every
            # single exposure; the coadds are what is worth drawing
            if resolution == 'medium' and not hdu.name.upper().startswith('COADD'):
                continue

            wavelength = np.asarray(hdu.data['WAVELENGTH'][0], dtype=float)
            flux = np.asarray(hdu.data['FLUX'][0], dtype=float)

            arm = hdu.name.upper().replace('COADD', '').strip('_') or 'ALL'

            rows.append(Table({
                'wavelength': wavelength,
                'flux': flux,
                'arm': np.full(len(wavelength), arm, dtype='<U4'),
            }))

    if not rows:
        return None

    from astropy.table import vstack

    return vstack(rows)


def _trim_dead_edges(spectrum, log):
    """Drop the dead pixels each arm begins and ends on.

    An arm is padded with a dozen or two exact zeros, which the published
    inverse variance does mark as worthless. The pixel or two just inside them
    is not marked at all - it carries an ordinary weight - and still comes out
    at minus eight hundred where the spectrum runs at three thousand. So the
    leading and trailing runs of non-positive flux go, both being edge
    artefacts rather than measurements of anything.

    Only the runs at the two ends: a negative in the middle of an arm is the
    noise of a faint object, which is a measurement and stays.
    """
    arms = np.asarray(spectrum['arm'], dtype=str)
    flux = np.asarray(spectrum['flux'], dtype=float)

    keep = np.ones(len(spectrum), dtype=bool)
    dropped = 0

    for arm in dict.fromkeys(arms):
        here = np.where(arms == arm)[0]
        good = np.where(flux[here] > 0)[0]

        if not len(good):
            continue

        # Everything outside the first and last real measurement of this arm
        edges = here[np.concatenate([np.arange(good[0]),
                                     np.arange(good[-1] + 1, len(here))])]
        keep[edges] = False
        dropped += len(edges)

    if dropped:
        log(f"  dropped {dropped} dead pixels from the ends of the arms")

    return spectrum[keep]


def _query(catalogue, ra, dec, sr, basepath, log, name, refresh):
    """The observations of one survey at a position, cached."""
    cache_name = f"lamost_{name}_{ra:.4f}_{dec:.4f}_{sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log,
                              f'LAMOST {name} observations', refresh=refresh) as cache:
        if not cache.hit:
            res = Vizier(columns=['**'], row_limit=-1).query_region(
                SkyCoord(ra, dec, unit='deg'), radius=sr*u.arcsec, catalog=catalogue)

            cat = res[0] if res and len(res) else None

            if cat is not None and len(cat):
                cache.save(cat)
            else:
                cat = None
        else:
            cat = cache.data

    return cat


def _value(row, key):
    """A column as a float, or None where the catalogue has nothing."""
    if key not in row.colnames:
        return None

    value = row[key]

    if value is None or value is np.ma.masked:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if np.isfinite(value) else None


@survey_source(
    name='LAMOST',
    short_name='LAMOST',
    state_acquiring='acquiring LAMOST spectra',
    state_acquired='LAMOST spectra acquired',
    log_file='lamost.log',
    output_files=['lamost.log', 'lamost_*.png', 'lamost_*.vot', 'lamost_*.txt'],
    button_text='Get LAMOST spectra',
    form_fields={
        'lamost_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': LAMOST_SR,
            'required': False,
        },
    },
    help_text='LAMOST DR11 spectra, northern sky, 3700-9100 A',
    # The spectra sit below the photometry, and together
    order=80,
    # Spectra rather than a light curve, so no lc_mode is declared
    spectrum_files='lamost_*.txt',
    # A target may have several - low and medium resolution, and more than one
    # epoch of either - and they are taken in filename order, which puts the
    # low-resolution ones first
    spectrum_palette=['#c0392b', '#e67e22', '#8e44ad', '#d35400', '#7f8c8d'],
    template_layout='complex',
    additional_plots=['lamost_*.png'],
)
def target_lamost(config, basepath=None, verbose=True, show=False):
    """
    Get LAMOST spectra.

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
    cleanup_paths(get_output_files('lamost'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = config.get('lamost_sr', LAMOST_SR)

    log(f"within {sr:.1f} arcsec")

    lrs = _query(LAMOST_LRS_CATALOGUE, ra, dec, sr, basepath, log, 'lrs', refresh_cache)
    mrs = _query(LAMOST_MRS_CATALOGUE, ra, dec, sr, basepath, log, 'mrs', refresh_cache)

    nlrs = len(lrs) if lrs is not None else 0
    nmrs = len(mrs) if mrs is not None else 0

    if not nlrs and not nmrs:
        log("\nWarning: No LAMOST observations at this position")
        return

    log(f"\n{nlrs} low-resolution and {nmrs} medium-resolution observation(s)")

    # What LAMOST made of the star
    if nlrs:
        log("\n---- Low resolution ----\n")

        for row in lrs:
            snr = _value(row, 'snrg')
            log(f"{row['ObsID']}  {row['Obs.Date']}  MJD {row['MJD']}"
                + (f"  S/N(g) = {snr:.0f}" if snr is not None else "")
                + (f"  {row['Class']}" if 'Class' in lrs.colnames else "")
                + (f" {row['subClass']}" if 'subClass' in lrs.colnames else ""))

            z = _value(row, 'z')
            if z is not None:
                # A redshift for a star is really its radial velocity
                log(f"    z = {z:+.6f}, which is {z * 299792.458:+.1f} km/s")

    if nmrs:
        log("\n---- Medium resolution ----\n")
        log(f"  {'obsid':>12s} {'date':>11s} {'band':>5s} {'S/N':>6s}"
            f" {'RV (arms)':>16s} {'RV (LASP)':>16s}")

        for row in mrs:
            snr = _value(row, 'snr')
            parts = [f"  {str(row['ObsID']):>12s} {str(row['Obs.Date']):>11s}"
                     f" {str(row['Band']):>5s} {'-':>6s}" if snr is None else
                     f"  {str(row['ObsID']):>12s} {str(row['Obs.Date']):>11s}"
                     f" {str(row['Band']):>5s} {snr:6.0f}"]

            for key, err in [('RVbr0', 'e_RVbr0'), ('RVlasp0', 'e_RVlasp0')]:
                value, error = _value(row, key), _value(row, err)
                if value is None:
                    parts.append(f"{'-':>16s}")
                elif error is None:
                    parts.append(f"{value:>16.2f}")
                else:
                    parts.append(f"{value:>9.2f} +/-{error:5.2f}")

            log(' '.join(parts))

        log("\n  velocities in km/s, from the two arms together and from LASP")

    # Radial velocity against time, where there is a curve to draw
    epochs = []

    if nmrs:
        for row in mrs:
            mjd, rv = _value(row, 'MJD'), _value(row, 'RVbr0')
            if mjd is not None and rv is not None:
                epochs.append((mjd, rv, _value(row, 'e_RVbr0') or 0.0))

    if len({_[0] for _ in epochs}) > 1:
        epochs = np.array(sorted(epochs))

        with plots.figure_saver(os.path.join(basepath, 'lamost_rv.png'),
                                figsize=(8, 4), show=show) as fig:
            ax = fig.add_subplot(1, 1, 1)

            time = Time(epochs[:, 0], format='mjd')
            ax.errorbar(time.datetime, epochs[:, 1], epochs[:, 2],
                        fmt='o', color='#c0392b')

            ax.grid(alpha=0.2)
            ax.set_ylabel('Radial velocity, km/s')
            ax.set_xlabel('Time')
            ax.set_title(f"{config['target_name']} - LAMOST radial velocities")

        log(f"\nRadial velocities at {len(set(epochs[:, 0]))} epochs"
            f" written to file:lamost_rv.png")

    # The spectra themselves, one request each
    wanted = []

    for cat, resolution, prefix in [(lrs, 'low', 'lrs'), (mrs, 'medium', 'mrs')]:
        if cat is None:
            continue

        # The medium-resolution table has a row per arm, and both arms come in
        # the one file
        for obsid in dict.fromkeys(int(_) for _ in cat['ObsID']):
            wanted.append((obsid, resolution, prefix))

    if len(wanted) > LAMOST_MAX_SPECTRA:
        log(f"\n{len(wanted)} spectra, of which the first {LAMOST_MAX_SPECTRA} are fetched")
        wanted = wanted[:LAMOST_MAX_SPECTRA]
    else:
        log(f"\nFetching {len(wanted)} spectra")

    for obsid, resolution, prefix in wanted:
        cache_name = f"lamost_spec_{prefix}_{obsid}.vot"

        with cached_votable_query(cache_name, basepath, log,
                                  f'LAMOST spectrum {obsid}', refresh=refresh_cache) as cache:
            if not cache.hit:
                try:
                    spectrum = _fetch_spectrum(obsid, resolution, log)
                except Exception as e:
                    log(f"  {obsid}: {e}")
                    spectrum = None

                if spectrum is not None and len(spectrum):
                    cache.save(spectrum)
                else:
                    spectrum = None
            else:
                spectrum = cache.data

        if spectrum is None or not len(spectrum):
            continue

        # After the cache as well as before it, so that what was already
        # fetched is cleaned up too
        spectrum = _trim_dead_edges(spectrum, log)

        if not len(spectrum):
            continue

        name = f"lamost_{prefix}_{obsid}"

        # The survey's own scale, into the one every spectrum here is on. The
        # cache above keeps the flux as LAMOST published it.
        written = Table({
            'wavelength': np.asarray(spectrum['wavelength'], dtype=float),
            'flux': np.asarray(spectrum['flux'], dtype=float) * LAMOST_TO_CGS,
            'arm': np.asarray(spectrum['arm'], dtype=str),
        })

        with plots.figure_saver(os.path.join(basepath, name + '.png'),
                                figsize=(10, 4), show=show) as fig:
            ax = fig.add_subplot(1, 1, 1)

            # The medium-resolution spectra come in two disjoint arms, which
            # are drawn apart rather than joined across the gap between them
            for arm in dict.fromkeys(np.asarray(written['arm'], dtype=str)):
                part = written[np.asarray(written['arm'], dtype=str) == arm]
                ax.plot(part['wavelength'], part['flux'] / LAMOST_TO_CGS,
                        '-', lw=0.7, label=arm if arm != 'ALL' else None)

            if len(set(np.asarray(written['arm'], dtype=str))) > 1:
                ax.legend()

            ax.grid(alpha=0.2)
            ax.set_xlabel('Wavelength, A')
            ax.set_ylabel(r'Flux, $10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$')
            ax.set_title(f"{config['target_name']} - LAMOST {obsid}"
                         f" ({resolution} resolution)")

        # The VOTable was promised in output_files from the beginning and
        # never written; both forms are what every other source leaves
        write_spectrum(written, basepath, name)

        log(f"  {obsid}: {len(spectrum)} points plotted in file:{name}.png,"
            f" written to file:{name}.vot and file:{name}.txt")
