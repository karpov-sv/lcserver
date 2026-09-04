"""Reading a model atlas out of the layout STScI distributes them in.

The CDBS grids - the 1993 Kurucz atlas, the 2004 Castelli & Kurucz one, and
others in the same reference-atlas collection - are all laid out the same way: a
directory per metallicity, a FITS table per temperature within it, and in each
table a wavelength column and one flux column per gravity, ``g00`` through
``g50``, in FLAM at the stellar surface. That last part is the quantity our
cubes hold, so the whole conversion is a wavelength unit.
https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs

A gravity a temperature does not reach is written as a column of zeros rather
than left out, so the grid's raggedness is read off the data instead of a
table: about three fifths of the rectangle exists.

Two things make these worth reading.

  * They are what the cubes already here were built from - convolving them
    reproduces astroARIADNE's stored fluxes to four places - but astroARIADNE
    stopped at 12000 K, which is less than half of either atlas. Both run to
    50000 K.

  * They carry the spectra, at about two hundred resolving elements per e-fold
    in the ultraviolet and three hundred in the optical. Coarse beside TLUSTY,
    enough for photometry and for a continuum, and it is what the atlas has.

Past about ten microns the tabulated flux is a Rayleigh-Jeans tail rather than
a model - measurably so, lambda to the minus four to three decimal places -
which both atlases say of themselves, so the grid is written believing itself
only to 8.5 microns.
"""

import os
import re
import glob

import numpy as np

from ..processing.utils import SourceError
from . import passbands, store


# Where the models stop being models. The atlas tabulates to 160 microns and
# says of itself that it covers 1000 Angstrom to 10; what is between is a
# Rayleigh-Jeans continuation of the last real point.
REACH_UM = 8.5

# A directory is a metallicity, named for it after whatever the atlas calls
# itself: km05 is -0.5 in the 1993 grid, ckp02 is +0.2 in the 2004 one
METALLICITY = re.compile(r'^[a-z]+([mp])(\d+)$')

SIGNS = {'m': -1, 'p': +1}

# What the atlases are called, where the directory does not say it plainly
KNOWN = {
    'k93models': ('kurucz', 'Kurucz 1993',
                  'plane-parallel and in LTE, so a cross-check on the hot end '
                  'rather than the model to reach for'),
    'ck04models': ('ck04', 'Castelli & Kurucz',
                   'ATLAS9 with the 2004 opacities, plane-parallel and in LTE'),
}


def read_metallicity(name):
    """The [M/H] a directory of the atlas is for, from its name."""
    match = METALLICITY.match(name)
    if not match:
        return None

    return SIGNS[match.group(1)] * int(match.group(2)) / 10


def atlas_name(path):
    """What to call the grid, and how to describe it, from its directory."""
    stem = os.path.basename(os.path.normpath(path))

    return KNOWN.get(stem, (stem, stem, None))


def read_temperature(path):
    """The temperature a file is for, from its name."""
    try:
        return float(os.path.basename(path).split('_')[1].split('.')[0])
    except (IndexError, ValueError):
        return None


def read_models(path):
    """Every model in the atlas, as (teff, logg, feh, wavelength, flux).

    The flux comes back in erg/s/cm2/um, which is what the cubes are in; the
    atlas gives it per Angstrom at the surface, so that is the whole conversion.
    """
    from astropy.io import fits

    directories = sorted(_ for _ in glob.glob(os.path.join(path, '*'))
                         if os.path.isdir(_)
                         and read_metallicity(os.path.basename(_)) is not None)
    if not directories:
        raise SourceError(f'no metallicity directories in {path}')

    for directory in directories:
        feh = read_metallicity(os.path.basename(directory))
        if feh is None:
            continue

        for name in sorted(glob.glob(os.path.join(directory, '*.fits'))):
            teff = read_temperature(name)
            if teff is None:
                continue

            with fits.open(name) as opened:
                data = opened[1].data
                wave = np.asarray(data['WAVELENGTH'], dtype=float)

                for column in data.columns.names[1:]:
                    flux = np.asarray(data[column], dtype=float)

                    # A gravity this temperature does not reach is a column of
                    # zeros, which is the atlas saying there is no model rather
                    # than that the star is dark
                    if not flux.any():
                        continue

                    yield (teff, int(column[1:]) / 10, feh, wave, flux * 1e4)


def ingest(path, cube_path, spectra_path, name, label=None, description=None,
           verbose=None):
    """Read the atlas and write the two files a grid here is."""
    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    bands = passbands.filter_set()
    log(f'reading {os.path.basename(os.path.normpath(path))}, '
        f'{len(bands)} passbands')

    teff, logg, feh = [], [], []
    fluxes, spectra, axis = [], [], None

    for t, g, z, wave, flux in read_models(path):
        if axis is None:
            axis = wave
        elif len(wave) != len(axis) or not np.allclose(wave, axis):
            raise SourceError('the atlas does not sample every model the same '
                              'way, and this reads it as though it did')

        teff.append(t)
        logg.append(g)
        feh.append(z)
        fluxes.append(passbands.convolve(wave, flux, bands))
        spectra.append(flux)

        if not len(fluxes) % 500:
            log(f'  {len(fluxes)} models')

    if not fluxes:
        raise SourceError(f'no models read from {path}')

    fluxes = np.array(fluxes)
    covered = np.isfinite(fluxes).any(axis=0)

    log(f'\n  {len(fluxes)} models, {len(np.unique(teff))} temperatures '
        f'{min(teff):.0f} to {max(teff):.0f} K, '
        f'{len(np.unique(feh))} metallicities')
    log(f'  {len(axis)} wavelengths, {axis.min():.1f} to {axis.max() * 1e-4:.0f} um, '
        f'believed to {REACH_UM} um')
    log(f'  {covered.sum()} of {len(bands)} passbands reached')

    store.write(cube_path, spectra_path, name=name,
                teff=teff, logg=logg, feh=feh, fluxes=fluxes, bands=bands,
                wave_um=axis * 1e-4, spectra=spectra,
                label=label, description=description, reach_um=REACH_UM,
                source=f'STScI CDBS, {os.path.basename(os.path.normpath(path))}',
                reference='https://www.stsci.edu/hst/instrumentation/'
                          'reference-data-for-calibration-and-tools/'
                          'astronomical-catalogs')

    return len(fluxes)
