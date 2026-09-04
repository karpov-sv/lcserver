"""Reading the BOSZ library as the archive hands it out.

BOSZ is published one file per model, and organised by rather more than the
three axes a fit here varies: metallicity, alpha enhancement, carbon abundance,
microturbulence and instrumental broadening, which between them make a hundred
times more spectra than are wanted. What is wanted is one composition at one
broadening - and the one the cube already here was built from turns out to be
the solar-abundance, two-kilometre-per-second slice, which convolves back to
its stored fluxes exactly once multiplied by four pi.
https://archive.stsci.edu/hlsp/bosz

Three things about the files.

  * They carry no wavelength. Every model at a given broadening is resampled
    onto one grid, published beside them as a single file, so the spectra go in
    as they are and nothing has to be resampled or reconciled.

  * The two columns are the flux and the continuum it is measured against. The
    first is what a cube holds; the second is not read.

  * The flux is the Eddington flux, so it is four pi short of the flux leaving
    the surface. The factor is not guessed - it is what makes every band of the
    installed cube come back as one.

The library reaches thirty-two microns, which for the temperatures it covers is
further than any other grid here goes. Its infrared falls as lambda to the
minus four, and at these temperatures that is where a real model should be
rather than a sign of a tail stitched on.
"""

import os
import re
import gzip
import glob

import numpy as np

from ..processing.utils import SourceError
from . import passbands, store


# Eddington flux to the flux leaving the surface
EDDINGTON = 4 * np.pi

# bosz2024_mp_t5000_g+4.5_m+0.00_a+0.00_c+0.00_v2_r2000_resam.txt.gz
MODEL = re.compile(
    r'bosz\d+_(?P<atmos>[a-z]+)_t(?P<teff>[\d.]+)_g(?P<logg>[+-][\d.]+)'
    r'_m(?P<feh>[+-][\d.]+)_a(?P<alpha>[+-][\d.]+)_c(?P<carbon>[+-][\d.]+)'
    r'_v(?P<micro>\d+)_r(?P<broad>\w+)_')


def read_wavelengths(path):
    """The grid every model at one broadening is resampled onto, in Angstrom."""
    found = sorted(glob.glob(os.path.join(path, '*wave*.txt')))
    if not found:
        raise SourceError(f'no wavelength grid in {path} - it is published '
                          f'beside the models, not inside them')

    return np.loadtxt(found[0])


def find_models(path):
    """Every model file under the download, with what its name says it is."""
    out = []
    for name in sorted(glob.glob(os.path.join(path, '*', '*.txt.gz'))
                       + glob.glob(os.path.join(path, '*.txt.gz'))):
        found = MODEL.match(os.path.basename(name))
        if found:
            out.append((name, found.groupdict()))

    return out


def read_flux(path, count):
    """The flux column of one model, as a surface flux in erg/s/cm2/um.

    Split rather than parsed a line at a time: the files are thirty thousand
    rows each and there are six and a half thousand of them, and the difference
    between the two is most of an afternoon.
    """
    with gzip.open(path, 'rt') as handle:
        fields = handle.read().split()

    flux = np.array(fields[0::2], dtype=float)
    if len(flux) != count:
        raise SourceError(f'{os.path.basename(path)} has {len(flux)} points '
                          f'and the wavelength grid has {count}')

    return flux * EDDINGTON * 1e4


def ingest(path, cube_path, spectra_path, name, label=None, description=None,
           verbose=None):
    """Read a BOSZ download and write the two files a grid here is."""
    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    wave_aa = read_wavelengths(path)
    models = find_models(path)
    if not models:
        raise SourceError(f'no model files under {path}')

    bands = passbands.filter_set()

    # A download should be one composition and one broadening; if it is not,
    # say so rather than average two of them together
    for key in ('alpha', 'carbon', 'micro', 'broad'):
        values = sorted({_[1][key] for _ in models})
        if len(values) > 1:
            raise SourceError(f'the download has {len(values)} values of '
                              f'{key} in it ({", ".join(values)}) - a grid '
                              f'here varies temperature, gravity and '
                              f'metallicity and nothing else')

    fixed = models[0][1]
    log(f"{len(models)} models at alpha {fixed['alpha']}, carbon "
        f"{fixed['carbon']}, microturbulence {fixed['micro']} km/s, "
        f"broadening R = {fixed['broad']}")
    log(f'{len(wave_aa)} wavelengths, {wave_aa.min():.0f} to '
        f'{wave_aa.max() * 1e-4:.0f} um, {len(bands)} passbands')

    teff, logg, feh = [], [], []
    fluxes, spectra = [], []

    for n, (each, about) in enumerate(models):
        flux_um = read_flux(each, len(wave_aa))

        teff.append(float(about['teff']))
        logg.append(float(about['logg']))
        feh.append(float(about['feh']))
        fluxes.append(passbands.convolve(wave_aa, flux_um, bands))
        spectra.append(flux_um)

        if not (n + 1) % 250:
            log(f'  {n + 1} of {len(models)}')

    fluxes = np.array(fluxes)
    covered = np.isfinite(fluxes).any(axis=0)

    log(f'\n  {len(fluxes)} models, {len(np.unique(teff))} temperatures '
        f'{min(teff):.0f} to {max(teff):.0f} K, '
        f'{len(np.unique(logg))} gravities, '
        f'{len(np.unique(feh))} metallicities')
    log(f'  {covered.sum()} of {len(bands)} passbands reached')

    store.write(cube_path, spectra_path, name=name,
                teff=teff, logg=logg, feh=feh, fluxes=fluxes, bands=bands,
                wave_um=wave_aa * 1e-4, spectra=spectra,
                label=label, description=description,
                source=f"BOSZ, a{fixed['alpha']} c{fixed['carbon']} "
                       f"v{fixed['micro']} r{fixed['broad']}",
                reference='https://archive.stsci.edu/hlsp/bosz')

    return len(fluxes)
