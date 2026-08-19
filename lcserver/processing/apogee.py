"""APOGEE spectra acquisition module.

Not a light curve: APOGEE observed in the near infrared, at H band, where the
dust of the Galactic plane is a tenth of the obstacle it is in the optical.
That is the whole point of it here - the plane is where BGDS, OGLE and the
rest of the disk surveys work, and it is where optical spectroscopy stops.

A star it looked at gets one combined spectrum per telescope and field it was
observed from, and behind each of those sit the individual visits, which the
survey used to measure a radial velocity apiece. Those velocities are a time
series in their own right and are plotted where there is more than one.

The observation list comes from Vizier, which carries the DR17 summary
catalogue, and the spectra from the SDSS science archive, one file each.
"""

import os
import io

import requests
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy import units as u

from astroquery.vizier import Vizier

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    break_at_gaps, plot_with_errors, write_spectrum)


# The DR17 summary catalogue, one row per star per field it was observed in
APOGEE_CATALOGUE = 'III/286/catalog'

# Matching radius, in arcsec. The positions are 2MASS ones and the fibres are
# two arcseconds across.
APOGEE_SR = 3.0

# Where the combined spectra live. The field goes into a path element and may
# hold a plus sign, which is why it is quoted rather than pasted in.
APOGEE_URL = ('https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/'
              'stars/{telescope}/{field}/{filename}')

APOGEE_TIMEOUT = 180

# A star observed from several fields is a spectrum and a plot per field
APOGEE_MAX_SPECTRA = 10

# APOGEE publishes its fluxes in units of 1e-17 erg/s/cm2/A, where every
# spectrum here is written in erg/s/cm2/A outright
APOGEE_TO_CGS = 1e-17

# What to ask Vizier for. The whole catalogue is some two hundred columns of
# abundances, of which the ones worth a line of the log are named here.
APOGEE_COLUMNS = [
    'APOGEE', 'Tel', 'Field', 'File', 'RAJ2000', 'DEJ2000',
    'Nvis', 'SNR', 'HRV', 'e_HRV', 's_HRV', 'Teff', 'e_Teff',
    'logg', 'e_logg', '[M/H]', 'e_[M/H]', '[a/M]', 'e_[a/M]',
    'Vsini', 'Vmicro', 'SFlag', 'AFlag', 'Hmag',
]

# The parameters, as they are logged: the column, its error, a label, and how
# to print it
APOGEE_PARAMETERS = [
    ('Teff', 'e_Teff', 'Teff', '.0f', ' K'),
    ('logg', 'e_logg', 'log g', '.2f', ''),
    ('[M/H]', 'e_[M/H]', '[M/H]', '.2f', ''),
    ('[a/M]', 'e_[a/M]', '[alpha/M]', '.2f', ''),
    ('Vsini', None, 'v sin i', '.1f', ' km/s'),
    ('Vmicro', None, 'v_micro', '.2f', ' km/s'),
]


def _value(row, key):
    """One field of a row, or None where it is absent, masked or not a number."""
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
    """One field of a row as a word, or None where it says nothing."""
    if key not in row.colnames:
        return None

    value = row[key]

    if value is np.ma.masked:
        return None

    return str(value).strip() or None


def _download(row):
    """One star's combined spectrum and its visits, from the archive.

    Returns the pair as tables: the spectrum on the survey's own log-linear
    wavelength grid, in vacuum Angstrom and the fluxes as published, and one
    row per visit carrying the velocity measured from it.
    """
    url = APOGEE_URL.format(telescope=_text(row, 'Tel'),
                            field=requests.utils.quote(_text(row, 'Field') or '', safe=''),
                            filename=_text(row, 'File'))

    try:
        res = requests.get(url, timeout=APOGEE_TIMEOUT)
        res.raise_for_status()
    except requests.RequestException as e:
        raise SourceError(f"could not fetch {_text(row, 'File')} - "
                          f"{type(e).__name__}: {e}")

    with fits.open(io.BytesIO(res.content)) as hdus:
        header = hdus[0].header

        flux = np.atleast_2d(np.asarray(hdus[1].data, dtype=float))
        error = np.atleast_2d(np.asarray(hdus[2].data, dtype=float))

        # A star seen more than once has two combinations of its visits at the
        # head of the file, followed by the visits themselves; one seen once
        # has only the single spectrum. Either way the first row is the one to
        # draw - for the repeat observations it is the pixel-weighted combination,
        # which is what the survey recommends.
        wavelength = 10 ** (header['CRVAL1']
                            + header['CDELT1'] * np.arange(flux.shape[1]))

        spectrum = Table({
            'wavelength': wavelength,
            'flux': flux[0],
            'flux_error': error[0],
        })

        # The velocity of each visit, which the summary catalogue gives only
        # the average of
        visits = None

        if len(hdus) > 9 and getattr(hdus[9], 'data', None) is not None:
            table = hdus[9].data

            if len(table):
                visits = Table({
                    # A full Julian date, where everything else here is an MJD
                    'mjd': np.asarray(table['jd'], dtype=float) - 2400000.5,
                    'vhelio': np.asarray(table['vhelio'], dtype=float),
                    'vhelio_error': np.asarray(table['vrelerr'], dtype=float),
                    'snr': np.asarray(table['snr'], dtype=float),
                })

    return spectrum, visits


@survey_source(
    name='APOGEE',
    short_name='APOGEE',
    state_acquiring='acquiring APOGEE spectra',
    state_acquired='APOGEE spectra acquired',
    log_file='apogee.log',
    output_files=['apogee.log', 'apogee_*.png', 'apogee_*.vot', 'apogee_*.txt'],
    button_text='Get APOGEE spectra',
    form_fields={
        'apogee_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': APOGEE_SR,
            'required': False,
        },
    },
    help_text='APOGEE DR17 near-infrared spectra, H band, 1.51-1.70 um',
    order=84,
    spectrum_files='apogee_*.txt',
    spectrum_palette=['#7e5109', '#9c640c', '#b9770e', '#d68910', '#f0b27a'],
    template_layout='complex',
    additional_plots=['apogee_*.png'],
)
def target_apogee(config, basepath=None, verbose=True, show=False):
    """Acquire APOGEE spectra."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('apogee'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = float(config.get('apogee_sr', APOGEE_SR))

    log(f"within {sr:.1f} arcsec")

    cache_name = f"apogee_{ra:.4f}_{dec:.4f}_{sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'APOGEE observations',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            try:
                found = Vizier(columns=APOGEE_COLUMNS, row_limit=-1).query_region(
                    f"{ra} {dec}", radius=sr * u.arcsec, catalog=APOGEE_CATALOGUE)
            except Exception as e:
                raise SourceError("could not query Vizier for APOGEE - "
                                  f"{type(e).__name__}: {e}")

            found = found[0] if found and len(found) else None

            if found is None or not len(found):
                cache.save_empty()
                log("\nWarning: No APOGEE observations at this position - the "
                    "survey put a fibre on two-thirds of a million stars, "
                    "which is not everything in its footprint")
                return

            cache.save(found)

        found = cache.data

    # Nothing here, and cached as nothing - the helper has said so already
    if found is None:
        return

    log(f"\n{len(found)} APOGEE observation(s)")

    log("\n---- APOGEE derived parameters ----\n")

    for row in found:
        name = _text(row, 'APOGEE')
        nvis, snr = _value(row, 'Nvis'), _value(row, 'SNR')

        log(f"{name}  {_text(row, 'Tel')} field {_text(row, 'Field')}"
            + (f"  {int(nvis)} visit(s)" if nvis is not None else '')
            + (f"  S/N = {snr:.0f}" if snr is not None else ''))

        rv, rv_err, scatter = (_value(row, 'HRV'), _value(row, 'e_HRV'),
                               _value(row, 's_HRV'))

        if rv is not None:
            log(f"    RV = {rv:+.2f}"
                + (f" +/- {rv_err:.2f}" if rv_err is not None else '')
                + " km/s"
                + (f", scattering {scatter:.2f} km/s between visits"
                   if scatter is not None and nvis and nvis > 1 else ''))

        for key, err, label, fmt, unit in APOGEE_PARAMETERS:
            value = _value(row, key)

            if value is None:
                continue

            spread = _value(row, err) if err else None

            log(f"    {label:10s} = {value:{fmt}}"
                + (f" +/- {spread:{fmt}}" if spread is not None else '')
                + unit)

        # The pipeline says where it does not trust itself, and a spectrum
        # with a flag set is still worth having as long as the flag is seen
        for key, what in [('SFlag', 'the star'), ('AFlag', 'the fit')]:
            flag = _text(row, key)

            if flag and flag not in ('0', '--'):
                log(f"    warning flag on {what}: {flag}")

    wanted = list(found)

    if len(wanted) > APOGEE_MAX_SPECTRA:
        log(f"\n{len(wanted)} spectra, of which the first {APOGEE_MAX_SPECTRA} are fetched")
        wanted = wanted[:APOGEE_MAX_SPECTRA]
    else:
        log(f"\nFetching {len(wanted)} spectr{'um' if len(wanted) == 1 else 'a'}")

    for row in wanted:
        star = _text(row, 'APOGEE')
        telescope = _text(row, 'Tel')
        field = _text(row, 'Field')

        # The field belongs in the name as much as the star and the telescope
        # do. A star in the overlap of several fields was observed in each of
        # them and the survey combined each field's visits separately, so one
        # star can be three rows here - of four, five and seven visits, from
        # three different files. Named without the field they share a cache
        # entry and a set of output files, and the second and third quietly
        # become copies of the first.
        stem = f"apogee_{star}_{telescope}_{field}"

        # The spectrum and the visit velocities come out of the one file, so
        # it is downloaded at most once however many caches are being filled
        downloaded = {}

        def fetched(what):
            if 'data' not in downloaded:
                downloaded['data'] = _download(row)

            return downloaded['data'][what]

        with cached_votable_query(f"{stem}_spec.vot", basepath, log,
                                  f'APOGEE spectrum {star}',
                                  refresh=refresh_cache) as cache:
            if not cache.hit:
                try:
                    spectrum = fetched(0)

                    # Inside the try, so that a fetch which failed is not
                    # remembered as a spectrum that does not exist
                    if spectrum is not None and len(spectrum):
                        cache.save(spectrum)
                    else:
                        cache.save_empty()
                        spectrum = None
                except Exception as e:
                    log(f"  {star} ({field}): {type(e).__name__}: {e}")
                    spectrum = None
            else:
                spectrum = cache.data

        with cached_votable_query(f"{stem}_visits.vot", basepath, log,
                                  f'APOGEE visits of {star}',
                                  refresh=refresh_cache) as cache:
            if not cache.hit:
                try:
                    visits = fetched(1)

                    if visits is not None and len(visits):
                        cache.save(visits)
                    else:
                        cache.save_empty()
                        visits = None
                except Exception:
                    # Already reported above, where the spectrum was wanted
                    visits = None
            else:
                visits = cache.data

        if spectrum is not None and len(spectrum):
            wavelength = np.asarray(spectrum['wavelength'], dtype=float)
            flux = np.asarray(spectrum['flux'], dtype=float)
            error = np.asarray(spectrum['flux_error'], dtype=float)

            # A pixel the pipeline could not measure is left at exactly zero
            # with no error, which is an absence rather than a measurement of
            # nothing, and is dropped rather than drawn as a hole in the star
            good = np.isfinite(flux) & np.isfinite(error) & (error > 0)

            log(f"  {star} ({field}): {int(np.sum(good))} of {len(wavelength)}"
                f" points from {wavelength.min():.0f} to"
                f" {wavelength.max():.0f} A")

            if np.any(good):
                name = f"{stem}_spec"

                with plots.figure_saver(os.path.join(basepath, name + '.png'),
                                        figsize=(10, 4), show=show) as fig:
                    ax = fig.add_subplot(1, 1, 1)

                    # The spectrograph has three detectors with gaps between
                    # them, and the pipeline leaves those pixels empty. Joining
                    # the runs either side would draw a straight line across
                    # sixty Angstrom of nothing, so the line is broken there.
                    ax.plot(*break_at_gaps(wavelength[good], flux[good]),
                            '-', lw=0.6, color='#9c640c')

                    ax.grid(alpha=0.2)
                    ax.set_xlabel('Wavelength, A')
                    ax.set_ylabel(r'Flux, $10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$')
                    ax.set_title(f"{config['target_name']} - APOGEE {star}"
                                 f" ({telescope}, {field})")

                # The survey's own scale, into the one every spectrum here is
                # on. The cache above keeps the flux as APOGEE published it.
                write_spectrum(Table({
                    'wavelength': wavelength[good],
                    'flux': flux[good] * APOGEE_TO_CGS,
                    'flux_error': error[good] * APOGEE_TO_CGS,
                }), basepath, name)

                log(f"    Spectrum plotted in file:{name}.png")
                log(f"    Spectrum written to file:{name}.vot")
                log(f"    Spectrum written to file:{name}.txt")

        # The velocities of the individual visits, where the star was seen
        # more than once. This is the one thing APOGEE offers that is a time
        # series, and a binary shows up in it plainly.
        if visits is not None and len(visits) > 1:
            mjd = np.asarray(visits['mjd'], dtype=float)
            rv = np.asarray(visits['vhelio'], dtype=float)
            rv_err = np.asarray(visits['vhelio_error'], dtype=float)

            good = np.isfinite(mjd) & np.isfinite(rv)

            if int(np.sum(good)) > 1:
                name = f"{stem}_rv"

                with plots.figure_saver(os.path.join(basepath, name + '.png'),
                                        figsize=(8, 4), show=show) as fig:
                    ax = fig.add_subplot(1, 1, 1)

                    time = Time(mjd[good], format='mjd')
                    plot_with_errors(ax, time.datetime, rv[good],
                                     np.where(np.isfinite(rv_err[good]),
                                              rv_err[good], 0.0),
                                     marker='o', color='#7e5109')

                    ax.grid(alpha=0.2)
                    ax.set_ylabel('Heliocentric radial velocity, km/s')
                    ax.set_xlabel('Time')
                    ax.set_title(f"{config['target_name']} - APOGEE {star}"
                                 f" visit velocities ({field})")

                spread = float(np.nanmax(rv[good]) - np.nanmin(rv[good]))

                log(f"    {int(np.sum(good))} visit velocities spanning"
                    f" {spread:.2f} km/s")
                log(f"    Velocities plotted in file:{name}.png")
