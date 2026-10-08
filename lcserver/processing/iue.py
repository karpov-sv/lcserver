"""IUE ultraviolet spectra acquisition module.

Not a light curve: the International Ultraviolet Explorer took some hundred
thousand spectra between 1978 and 1996, from 1150 to 3350 A, and for most stars
it observed they are the only ultraviolet spectra there are. What they bring a
target here is the part of its spectral energy distribution blueward of the
Balmer jump, which no ground-based survey reaches.

The spectra come from INES, the final IUE archive, which re-extracted every
NEWSIPS image and publishes each as a single flux calibrated table - including
the high resolution echelle images, which it also rebins onto the low
resolution grid. Those rebinned ones are taken alongside the low resolution
spectra proper, being on the same scale and the same sampling; the echelle
spectra at full resolution are not, a 0.2 A spectrum adding nothing to an SED
that 6 A does not.
"""

import io
import os
import re

import numpy as np
import requests

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files, KIND_SPECTROSCOPY
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    write_spectrum)


INES_SSAP_URL = 'https://sdc.cab.inta-csic.es/ines/jsp/ssap.jsp'

# Where a file is fetched from, by its name. Built here rather than taken from
# the search answer, which links to whatever scheme it was asked by but keeps
# the port it was asked on - asked over https, it links to http on 443.
INES_DOWNLOAD_URL = 'https://sdc.cab.inta-csic.es/ines/jsp/SingleDownload.jsp'

# Matching radius, in arcsec. The positions INES gives are those of the
# pointing, quoted to 0.4 arcsec, and the large aperture is a 10 x 20 arcsec
# slot - the spectra of one star land up to a few arcsec apart.
IUE_SR = 10.0

# The archive's own matching is coarse below some tens of arcsec - asked for
# two, it returns everything within five - so the radius is applied again here
IUE_MIN_QUERY_SR = 30.0

# A star used as a calibrator was observed hundreds of times over, which is a
# file and a line in the log for each, so only this many are fetched
IUE_MAX_SPECTRA = 30

# What the last two letters of an INES file name say: how it was extracted,
# and through which aperture. The high resolution spectra at their own
# sampling (H) are left out - see the module docstring.
IUE_PRODUCTS = {
    'L': 'low dispersion',
    'R': 'high dispersion, rebinned to low',
}
IUE_APERTURES = {
    'L': 'large aperture',
    'S': 'small aperture',
}

# The order they are fetched in, where there are more than IUE_MAX_SPECTRA: the
# large aperture first, as the only one that is photometric - the 3 arcsec
# small one loses a varying part of the light, a third of it for the star this
# was written against - and the low dispersion spectra before the rebinned
# echelle ones, which are calibrated less well.
IUE_PREFERENCE = ['LL', 'RL', 'LS', 'RS']

# Which pixels are dropped, by the NEWSIPS quality flags - negative sums of
# these bits. Those whose flux is wrong rather than merely less certain: a
# saturated pixel reads low by an unknown amount, and an IUE spectrum of a
# bright star is saturated across the peak of the camera's response. What is
# kept is the warning track near the edge of the extraction (512), the pixels
# far below the first ITF level (128), which are faint rather than wrong, and
# a gap in the background (4).
#
#   16384  not photometrically corrected     256  positively extrapolated ITF
#    8192  missing minor frame                 64  cosmic ray / bright spot
#    4096  reseau                              32  cosmic ray (low dispersion)
#    2048  permanent ITF artifact              16  microphonics (LWR)
#    1024  saturated                            8  potential DMU corruption
#                                               2  uncalibrated
IUE_BAD_FLAGS = 16384 | 8192 | 4096 | 2048 | 1024 | 256 | 64 | 32 | 16 | 8 | 2

# What is kept of each camera, in Angstrom. Their files run further, but the
# ends are where the sensitivity falls away: below 2000 A the long wavelength
# cameras are noise where the short one is at its best, and above 3200 A they
# rise or fall to nothing with the calibration rather than with the star. The
# two halves meet at 1975 A, as they are usually joined.
IUE_RANGES = {
    'SWP': (1150.0, 1975.0),
    'LWP': (1975.0, 3200.0),
    'LWR': (1975.0, 3200.0),
}

# Per camera, as the overview plot draws them
IUE_CAMERA_COLORS = {
    'SWP': '#6c3483',
    'LWP': '#2471a3',
    'LWR': '#17a589',
}


def _ines_rows(content):
    """The INES answer, as a list of dicts of strings keyed by field ID.

    Read as text rather than as a VOTable: it declares a field with an empty
    arraysize, which astropy refuses, and a long integer holding 'N/A', which
    it then reads into the wrong columns. The table itself is regular - one
    TD per FIELD, in order - so nothing more is needed to read it.
    """
    text = content.decode('utf-8', errors='replace')

    fields = []
    for tag in re.findall(r'<FIELD\b(.*?)>', text, re.S):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag))
        fields.append(attrs.get('ID') or attrs.get('name'))

    rows = []
    for tr in re.findall(r'<TR>(.*?)</TR>', text, re.S):
        cells = [re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', _, flags=re.S).strip()
                 for _ in re.findall(r'<TD>(.*?)</TD>', tr, re.S)]

        if len(cells) != len(fields):
            raise SourceError(f"unrecognised INES answer - {len(cells)} values "
                              f"for {len(fields)} fields")

        rows.append(dict(zip(fields, cells)))

    return rows


def _search(ra, dec, sr):
    """The INES spectra within sr arcsec of a position, nearest first.

    One row per file rather than per image: an echelle image is published
    both at its own resolution and rebinned, and a low dispersion one through
    each aperture that was exposed. Only the files kept by IUE_PRODUCTS are
    returned, as the plain columns the cache can hold.
    """
    res = requests.get(INES_SSAP_URL, timeout=120, params={
        'REQUEST': 'queryData',
        'POS': f'{ra:.6f},{dec:.6f}',
        'SIZE': f'{max(sr, IUE_MIN_QUERY_SR) / 3600:.6f}',
    })
    res.raise_for_status()

    rows = []
    for row in _ines_rows(res.content):
        url = row['Spectrum']
        m = re.search(r'filename=(([A-Z]{3})(\d+)([A-Z])([A-Z]))\.FITS', url, re.I)
        if not m:
            continue

        name, camera, _, product, aperture = (_.upper() for _ in m.groups())
        if product not in IUE_PRODUCTS or aperture not in IUE_APERTURES:
            continue

        rows.append({
            'name': name,
            'camera': camera,
            'product': product,
            'aperture': aperture,
            'object': row['Object'],
            'ra': float(row['RA']),
            'dec': float(row['DEC']),
            'mjd': float(row['TimeObs']),
            'exptime': float(row['TimeExtent']),
        })

    if not rows:
        return None

    table = Table(rows=rows, names=list(rows[0].keys()))

    table['dist'] = SkyCoord(table['ra'], table['dec'], unit='deg').separation(
        SkyCoord(ra, dec, unit='deg')).arcsec
    table = table[table['dist'] <= sr]
    table.sort('dist')

    return table if len(table) else None


def _fetch_spectrum(name):
    """One INES spectrum, as published - quality flags and all."""
    res = requests.get(INES_DOWNLOAD_URL, timeout=120,
                       params={'filename': f'{name}.FITS'})
    res.raise_for_status()

    # A file it cannot serve comes back as an HTML page saying so, with a 200
    if not res.content.startswith(b'SIMPLE'):
        raise SourceError("INES returned no FITS file")

    with fits.open(io.BytesIO(res.content)) as hdus:
        data = hdus[1].data

        return Table({
            'wavelength': np.asarray(data['WAVELENGTH'], dtype=float),
            'flux': np.asarray(data['FLUX'], dtype=float),
            'sigma': np.asarray(data['SIGMA'], dtype=float),
            'quality': np.asarray(data['QUALITY'], dtype=np.int32),
        })


def _describe(row):
    """A file of _search, as the log names it."""
    date = Time(float(row['mjd']), format='mjd').datetime.strftime('%Y-%m-%d')

    return (f"{row['name']}  {row['camera']}, {IUE_PRODUCTS[str(row['product'])]}, "
            f"{IUE_APERTURES[str(row['aperture'])]}, {date}, "
            f"{float(row['exptime']):.0f} s, {float(row['dist']):.1f} arcsec")


@survey_source(
    name='IUE',
    short_name='IUE',
    state_acquiring='acquiring IUE spectra',
    state_acquired='IUE spectra acquired',
    log_file='iue.log',
    output_files=['iue.log', 'iue.png', 'iue_*.vot', 'iue_*.txt'],
    button_text='Get IUE spectra',
    form_fields={
        'iue_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': IUE_SR,
            'required': False,
        },
    },
    help_text='IUE ultraviolet spectra from INES, 1150-3350 A',
    # The bluest of the spectra
    order=78,
    about=(
        "The International Ultraviolet Explorer, which took spectra from "
        "1150 to 3350 A between 1978 and 1996, as re-extracted and served "
        "by INES, the final IUE archive."),
    about_links=[
        ('INES', 'https://sdc.cab.inta-csic.es/ines/'),
        ('IUE at MAST', 'https://archive.stsci.edu/iue/'),
    ],
    acknowledgement="Based on INES data from the IUE satellite.",
    kind=KIND_SPECTROSCOPY,
    spectrum_files='iue_*.txt',
    # Worth having, but not to be read against the SED by default: the small
    # aperture is not photometric, and its spectra sit a third to a half below
    # the others; the rebinned echelle spectra of the long wavelength cameras
    # come out a fifth too faint in the 2175 A bump and a fifth too bright
    # beyond it, against TD1, where the low dispersion ones agree to 5%
    spectrum_hidden=['iue_*S.txt', 'iue_*R?.txt'],
    spectrum_palette=['#6c3483', '#2471a3', '#17a589', '#8e44ad', '#2e86c1',
                      '#48c9b0'],
    template_layout='complex',
    additional_plots=['iue.png'],
)
def target_iue(config, basepath=None, verbose=True, show=False):
    """Acquire IUE spectra."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('iue'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    sr = float(config.get('iue_sr') or IUE_SR)

    cache_name = f"iue_{ra:.4f}_{dec:.4f}_{sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'INES',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            log(f"within {sr:.1f} arcsec")
            try:
                found = _search(ra, dec, sr)
            except Exception as e:
                raise SourceError("could not query INES - "
                                  f"{type(e).__name__}: {e}")

            if found is None:
                cache.save_empty()
                log("\nWarning: No IUE spectra at this position - IUE observed "
                    "only what it was pointed at")
                return

            cache.save(found)

        found = cache.data

    # Nothing here, and cached as nothing - the helper has said so already
    if found is None:
        return

    log(f"\n{len(found)} IUE spectr{'um' if len(found) == 1 else 'a'} of "
        f"{', '.join(sorted(set(str(_) for _ in found['object'])))}\n")

    for row in found:
        log(f"  {_describe(row)}")

    wanted = sorted(found, key=lambda row: (
        IUE_PREFERENCE.index(f"{row['product']}{row['aperture']}"),
        float(row['mjd'])))

    if len(wanted) > IUE_MAX_SPECTRA:
        log(f"\n{len(wanted)} spectra, of which {IUE_MAX_SPECTRA} are fetched - "
            f"the large aperture and the low dispersion first")
        wanted = wanted[:IUE_MAX_SPECTRA]
    else:
        log(f"\nFetching {len(wanted)} spectr{'um' if len(wanted) == 1 else 'a'}")

    if any(row['aperture'] == 'S' or row['product'] == 'R' for row in wanted):
        log("Only the low dispersion spectra through the large aperture are "
            "shown in the viewer by default. The small aperture is smaller than "
            "the image of the star and loses a varying part of its light, and "
            "the echelle spectra rebinned to low dispersion are calibrated less "
            "well")

    published = []

    for row in wanted:
        name = str(row['name'])
        spec_cache = f"iue_spec_{name}.vot"

        with cached_votable_query(spec_cache, basepath, log,
                                  f'IUE spectrum {name}',
                                  refresh=refresh_cache) as cache:
            if not cache.hit:
                try:
                    spectrum = _fetch_spectrum(name)

                    # Inside the try, so that a fetch which failed is not
                    # remembered as a spectrum that does not exist
                    if spectrum is not None and len(spectrum):
                        cache.save(spectrum)
                    else:
                        cache.save_empty()
                        spectrum = None
                except Exception as e:
                    log(f"  {name}: {type(e).__name__}: {e}")
                    spectrum = None
            else:
                spectrum = cache.data

        if spectrum is None or not len(spectrum):
            continue

        wavelength = np.asarray(spectrum['wavelength'], dtype=float)
        flux = np.asarray(spectrum['flux'], dtype=float)
        sigma = np.asarray(spectrum['sigma'], dtype=float)
        quality = np.asarray(spectrum['quality'], dtype=np.int64)

        # The flags are stored negated
        bad = ((-quality) & IUE_BAD_FLAGS) != 0
        saturated = ((-quality) & 1024) != 0
        lo, hi = IUE_RANGES.get(str(row['camera']), (-np.inf, np.inf))
        inside = (wavelength >= lo) & (wavelength <= hi)
        good = inside & ~bad & np.isfinite(wavelength) & np.isfinite(flux)

        log(f"  {name}: {np.sum(inside)} points from {lo:.0f} to {hi:.0f} A, "
            f"{np.sum(inside & ~good)} dropped on quality"
            + (f" ({np.sum(inside & saturated)} saturated)"
               if np.any(inside & saturated) else ''))

        if not np.any(good):
            continue

        stem = f"iue_{name}"

        write_spectrum(Table({
            'wavelength': wavelength[good],
            'flux': flux[good],
            # A pixel the extraction could not weigh is left undefined rather
            # than quoted as exact
            'flux_error': np.where(sigma[good] > 0, sigma[good], np.nan),
        }), basepath, stem)

        published.append((row, wavelength[good], flux[good]))

        log(f"    Spectrum written to file:{stem}.vot")
        log(f"    Spectrum written to file:{stem}.txt")

    if not published:
        log("\nWarning: None of the IUE spectra could be fetched")
        return

    with plots.figure_saver(os.path.join(basepath, 'iue.png'),
                            figsize=(10, 5), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        for row, wavelength, flux in published:
            small = row['aperture'] == 'S'

            # Broken where pixels were dropped, rather than drawn straight
            # across what was not measured
            gaps = np.where(np.diff(wavelength) > 1.5 * np.median(np.diff(wavelength)))[0] + 1
            wavelength = np.insert(wavelength, gaps, np.nan)
            flux = np.insert(flux, gaps, np.nan)

            ax.plot(wavelength, flux, '-', lw=0.6,
                    color=IUE_CAMERA_COLORS.get(str(row['camera']), '#7f8c8d'),
                    alpha=0.35 if small else 0.8,
                    ls=':' if small else '-')

        # Drawn once each, as the legend has three cameras and not a line per
        # spectrum
        for camera in sorted(set(str(row['camera']) for row, _, _ in published)):
            ax.plot([], [], '-', color=IUE_CAMERA_COLORS.get(camera, '#7f8c8d'),
                    label=camera)
        if any(row['aperture'] == 'S' for row, _, _ in published):
            ax.plot([], [], ':', color='#7f8c8d', label='small aperture')

        positive = np.concatenate([_[2][_[2] > 0] for _ in published])
        if len(positive):
            ax.set_yscale('log')
            ax.set_ylim(np.percentile(positive, 0.5) / 2,
                        np.percentile(positive, 99.5) * 2)

        ax.legend(loc='best', fontsize='small')
        ax.grid(alpha=0.2)
        ax.set_xlabel('Wavelength, A')
        ax.set_ylabel(r'Flux, erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$')
        ax.set_title(f"{config['target_name']} - IUE")

    log(f"\nSpectra plotted in file:iue.png")
