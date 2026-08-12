"""DESI DR1 spectra acquisition module.

Like LAMOST, a spectroscopic survey rather than a photometric one: what it has
for a star is a spectrum from 3600 to 9824 A, a redshift that for a star is
really a radial velocity, and - where the Milky Way Survey looked at it -
atmospheric parameters and abundances.

The spectra come through SPARCL, NOIRLab's retrieval service, which hands over
one merged spectrum per request in a second or two. The alternative is the
public file tree, where the unit of storage is a healpix coadd of some 240 MB
holding several hundred spectra; pulling one row out of it over HTTP ranges
was measured at fifty seconds against SPARCL's two, so SPARCL it is.

The stellar parameters are not in SPARCL, so those come from the Astro Data
Lab table service, which carries the same release as desi_dr1.
"""

import os
import json
import requests
import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import SourceError, cleanup_paths, cached_votable_query


# Optional: the client is a NOIRLab package of its own, and everything else
# here works without it, so its absence is reported rather than raised at
# import time - which would take the whole registry down with it
try:
    from sparcl.client import SparclClient
except ImportError:
    SparclClient = None


DESI_DATASET = 'DESI-DR1'

# Matching radius, in arcsec. DESI fibres are 1.5 arcsec across.
DESI_SR = 3.0

# Each is a request and a plot; DR1 rarely has more than one per star
DESI_MAX_SPECTRA = 10

# Where the stellar parameters are, SPARCL not carrying them
DESI_TAP = 'https://datalab.noirlab.edu/tap/sync'
DESI_MWS_TABLE = 'desi_dr1.mws'
DESI_MWS_COLUMNS = [
    'targetid', 'target_ra', 'target_dec', 'survey', 'program',
    'teff', 'teff_err', 'logg', 'logg_err', 'feh', 'feh_err',
    'rv_adop', 'rv_err', 'vsini', 'snr_med', 'rvs_warn', 'success', 'source_id',
]

# Speed of light in km/s, to read a redshift as a velocity
DESI_C = 299792.458


def _epochs(dateobs):
    """The exposure times behind a coadd, as SPARCL gives them.

    A JSON list of timestamps, one per exposure that went into it, so a coadd
    of two nights says so rather than pretending to a single date.
    """
    if not dateobs:
        return []

    try:
        stamps = json.loads(dateobs)
    except (TypeError, ValueError):
        return [str(dateobs)]

    if not isinstance(stamps, list):
        stamps = [stamps]

    return [str(_)[:19] for _ in stamps]


def _query_tap(query, log):
    """A table from the Data Lab table service, or None."""
    res = requests.get(DESI_TAP, timeout=180, params={
        'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv', 'QUERY': query})

    # A failed query comes back as a VOTable saying so, whatever was asked for
    if res.status_code != 200 or res.text.lstrip().startswith('<'):
        raise SourceError("the table service refused the query: "
                          f"{res.text[:160]}")

    # A list of lines rather than a StringIO: the fast reader wants bytes from
    # a file-like object and refuses one that gives it text
    table = Table.read(res.text.splitlines(), format='csv')

    return table if len(table) else None


def _find(ra, dec, sr, log):
    """The DESI spectra at a position.

    SPARCL takes ranges rather than a cone, so the box is cut down to a circle
    afterwards - it is a box in degrees, and wider in right ascension the
    further from the equator it is.
    """
    client = SparclClient()

    half = sr / 3600.0
    half_ra = half / max(np.cos(np.deg2rad(dec)), 1e-6)

    found = client.find(
        outfields=['sparcl_id', 'ra', 'dec', 'specid', 'redshift', 'spectype'],
        constraints={'data_release': [DESI_DATASET],
                     'ra': [ra - half_ra, ra + half_ra],
                     'dec': [dec - half, dec + half]},
        limit=100)

    records = list(found.records)

    if not records:
        return []

    here = SkyCoord(ra, dec, unit='deg')
    inside = []

    for record in records:
        other = SkyCoord(record['ra'], record['dec'], unit='deg')
        separation = here.separation(other).arcsec

        if separation <= sr:
            inside.append((separation, record))

    inside.sort(key=lambda _: _[0])

    return inside


@survey_source(
    name='DESI',
    short_name='DESI',
    state_acquiring='acquiring DESI spectra',
    state_acquired='DESI spectra acquired',
    log_file='desi.log',
    output_files=['desi.log', 'desi_*.png', 'desi_*.vot', 'desi_*.txt'],
    button_text='Get DESI spectra',
    form_fields={
        'desi_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': DESI_SR,
            'required': False,
        },
    },
    help_text='DESI DR1 spectra, 3600-9800 A, northern sky',
    order=28,
    # Spectra rather than a light curve, so no lc_mode is declared
    spectrum_files='desi_*.txt',
    spectrum_palette=['#16a085', '#27ae60', '#1abc9c', '#2ecc71'],
    template_layout='complex',
    additional_plots=['desi_*.png'],
)
def target_desi(config, basepath=None, verbose=True, show=False):
    """
    Get DESI DR1 spectra.

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
    cleanup_paths(get_output_files('desi'), basepath=basepath)

    if SparclClient is None:
        log("The sparclclient package is not installed, and DESI spectra are "
            "fetched through it.")
        log("  pip install sparclclient")
        raise RuntimeError("sparclclient is not installed")

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = config.get('desi_sr', DESI_SR)

    log(f"within {sr:.1f} arcsec")

    matches = _find(ra, dec, sr, log)

    if not matches:
        log("\nWarning: No DESI spectra at this position")
        return

    log(f"\n{len(matches)} spectr{'um' if len(matches) == 1 else 'a'} found")

    if len(matches) > DESI_MAX_SPECTRA:
        log(f"  of which the nearest {DESI_MAX_SPECTRA} are fetched")
        matches = matches[:DESI_MAX_SPECTRA]

    client = SparclClient()
    targetids = []

    for separation, record in matches:
        uuid = record['sparcl_id']
        cache_name = f"desi_spec_{uuid}.vot"

        with cached_votable_query(cache_name, basepath, log,
                                  f'DESI spectrum {uuid[:8]}', refresh=refresh_cache) as cache:
            if not cache.hit:
                try:
                    res = client.retrieve(
                        uuid_list=[uuid], dataset_list=[DESI_DATASET],
                        include=['sparcl_id', 'ra', 'dec', 'targetid', 'specid',
                                 'flux', 'wavelength', 'ivar', 'redshift',
                                 'redshift_err', 'redshift_warning', 'spectype',
                                 'survey', 'instrument', 'dateobs', 'exptime',
                                 'specprimary'])
                    got = res.records[0] if res.records else None
                except Exception as e:
                    log(f"  {uuid[:8]}: {e}")
                    got = None

                if got is None:
                    spectrum = None
                else:
                    spectrum = Table({
                        'wavelength': np.asarray(got['wavelength'], dtype=float),
                        'flux': np.asarray(got['flux'], dtype=float),
                        'ivar': np.asarray(got['ivar'], dtype=float),
                    })

                    # A VOTable keeps no arbitrary metadata, so what describes
                    # the spectrum rides along as columns of one value
                    for key in ['targetid', 'redshift', 'redshift_err',
                                'redshift_warning', 'specprimary', 'exptime']:
                        value = got.get(key)
                        if value is not None:
                            spectrum[key] = np.full(len(spectrum), value)

                    for key in ['spectype', 'survey', 'instrument', 'dateobs']:
                        value = got.get(key)
                        if value is not None:
                            spectrum[key] = np.full(len(spectrum), str(value))

                    cache.save(spectrum)
            else:
                spectrum = cache.data

        if spectrum is None or not len(spectrum):
            continue

        first = spectrum[0]

        def field(name, default=None):
            return first[name] if name in spectrum.colnames else default

        targetid = field('targetid')
        redshift = field('redshift')

        log(f"\n---- {targetid} ----\n")
        log(f"{separation:.2f} arcsec away")

        epochs = _epochs(field('dateobs'))

        log(f"{field('instrument', 'DESI')} / {field('survey', '?')}"
            + (f"  {field('exptime'):.0f} s" if field('exptime') is not None else ""))

        if len(epochs) == 1:
            log(f"Observed {epochs[0]}")
        elif epochs:
            log(f"Coadded from {len(epochs)} exposures, {epochs[0]} to {epochs[-1]}")
        log(f"Classified as {field('spectype', '?')}"
            + ("" if field('specprimary') in (None, 0) else ", the primary spectrum of this object"))

        if redshift is not None:
            warning = field('redshift_warning')
            error = field('redshift_err')
            log(f"z = {redshift:.6f}"
                + (f" +/- {error:.6f}" if error is not None else "")
                + f", which for a star is {redshift * DESI_C:+.1f} km/s"
                + ("" if not warning else f" (warning flag {int(warning)})"))

        wavelength = np.asarray(spectrum['wavelength'], dtype=float)
        flux = np.asarray(spectrum['flux'], dtype=float)

        log(f"{len(wavelength)} points from {wavelength.min():.0f}"
            f" to {wavelength.max():.0f} A")

        name = f"desi_{targetid}"

        with plots.figure_saver(os.path.join(basepath, name + '.png'),
                                figsize=(10, 4), show=show) as fig:
            ax = fig.add_subplot(1, 1, 1)

            ax.plot(wavelength, flux, '-', lw=0.6, color='#16a085')

            ax.grid(alpha=0.2)
            ax.set_xlabel('Wavelength, A')
            ax.set_ylabel(r'Flux, $10^{-17}$ erg s$^{-1}$ cm$^{-2}$ A$^{-1}$')
            ax.set_title(f"{config['target_name']} - DESI {targetid}")

        # The VOTable was promised in output_files from the beginning and
        # never written; both forms are what every other source leaves
        table = spectrum[['wavelength', 'flux', 'ivar']]
        table.write(os.path.join(basepath, name + '.vot'),
                    format='votable', overwrite=True)
        table.write(os.path.join(basepath, name + '.txt'),
                    format='ascii.commented_header', overwrite=True)

        log(f"Spectrum plotted in file:{name}.png, written to "
            f"file:{name}.vot and file:{name}.txt")

        if targetid is not None:
            targetids.append(int(targetid))

    # What the Milky Way Survey made of the star, which SPARCL does not carry
    if targetids:
        cache_name = f"desi_mws_{ra:.4f}_{dec:.4f}_{sr:.1f}.vot"

        with cached_votable_query(cache_name, basepath, log, 'DESI stellar parameters',
                                  refresh=refresh_cache) as cache:
            if not cache.hit:
                ids = ', '.join(str(_) for _ in sorted(set(targetids)))
                mws = _query_tap(
                    f"SELECT {', '.join(DESI_MWS_COLUMNS)} FROM {DESI_MWS_TABLE}"
                    f" WHERE targetid IN ({ids})", log)

                if mws is not None and len(mws):
                    cache.save(mws)
                else:
                    mws = None
            else:
                mws = cache.data

        if mws is None or not len(mws):
            log("\nNot in the Milky Way Survey, so no stellar parameters")
        else:
            log("\n---- Stellar parameters (Milky Way Survey) ----\n")

            for row in mws:
                log(f"{row['targetid']}  {row['survey']}/{row['program']}")

                for label, key, err, fmt in [
                        ('Teff', 'teff', 'teff_err', '.0f'),
                        ('log g', 'logg', 'logg_err', '.2f'),
                        ('[Fe/H]', 'feh', 'feh_err', '.2f'),
                        ('RV', 'rv_adop', 'rv_err', '+.2f'),
                        ('v sin i', 'vsini', None, '.1f')]:
                    value = row[key] if key in mws.colnames else None

                    if value is None or not np.isfinite(value):
                        continue

                    spread = row[err] if err and err in mws.colnames else None
                    log(f"    {label:8s} = {value:{fmt}}"
                        + (f" +/- {spread:{fmt.lstrip('+')}}"
                           if spread is not None and np.isfinite(spread) else "")
                        + (' km/s' if key == 'rv_adop' else '')
                        + (' K' if key == 'teff' else ''))

                if 'rvs_warn' in mws.colnames and row['rvs_warn']:
                    log(f"    velocity carries warning flag {int(row['rvs_warn'])}")
