"""Reading Koester's white-dwarf models as the SVO service hands them out.

One file per model, a couple of comment lines naming the temperature and the
gravity, then a wavelength in Angstrom and a flux in erg/cm2/s/A. That flux is
already the one our cubes hold - convolving it reproduces the stored fluxes of
the cube here to four places - so the whole conversion is a wavelength unit.
http://svo2.cab.inta-csic.es/theory/newov2/

These are DA atmospheres, hydrogen-line white dwarfs: 5000 to 80000 K at
gravities from log g 6.5 to 9.5, which is a complete rectangle of 1066 models
with nothing missing, and one composition, so nothing varies but the two.

What the grid adds is the spectra, which the cube here has none of. They are
finely sampled where a white dwarf is interesting - about five thousand
resolving elements per e-fold through the optical, where the Balmer lines of a
DA are the whole of what there is to see - and they stop at three microns,
where a Rayleigh-Jeans tail takes over for the few bands just beyond.

Every model is sampled differently, so they are put on one axis to be stored;
the cube is convolved from each model's own sampling and nothing that is fitted
passes through the resampling.
"""

import os
import re
import glob

import numpy as np

from ..processing.utils import SourceError
from . import passbands, store


# The files stop at three microns, and a little past that are four bands worth
# having on a white dwarf - WISE W1 and W2, IRAC 3.6 and 4.5 - so each model is
# carried to ten with a Rayleigh-Jeans tail. Ten because a filter is convolved
# over its whole width and W1's runs to six and a half microns although it is
# named for three and a half; what is believed is the shorter REACH_UM, which
# is where the reddest of those bands has its pivot. That is not a guess: the models
# are measurably in that regime where they end, going as lambda to the minus
# 3.9 through their last half micron, so the continuation is good to a per cent
# where a band needs it and the grid is believed that far.
#
# It is also a correction. The cube this replaces has W2 brighter than W1 on a
# white dwarf, which no Rayleigh-Jeans tail does; whatever produced its
# near-infrared, it was not these models.
REACH_UM = 5.0
EXTEND_TO_UM = 10.0
RAYLEIGH_JEANS = -4.0

# The wavelengths the spectra are kept on. Five thousand per e-fold holds the
# optical sampling the models have, which is the point of keeping them at all.
SPECTRA_RANGE_UM = (0.085, 3.05)
SPECTRA_RESOLUTION = 5000

# The header says what the model is, and is believed over the file name
HEADER = re.compile(r'^#\s*(teff|logg)\s*=\s*([\d.eE+-]+)', re.IGNORECASE)


def read_model(path):
    """One model, as (teff, logg, wavelength in Angstrom, flux per micron)."""
    teff = logg = None

    with open(path) as handle:
        for line in handle:
            if not line.startswith('#'):
                break

            found = HEADER.match(line)
            if found:
                if found.group(1).lower() == 'teff':
                    teff = float(found.group(2))
                else:
                    logg = float(found.group(2))

    if teff is None or logg is None:
        return None

    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        return None

    wave = data[:, 0].astype(float)
    order = np.argsort(wave)
    wave, flux = wave[order], data[order, 1].astype(float) * 1e4

    return (teff, logg) + extend(wave, flux)


def extend(wave_aa, flux_um, to_um=EXTEND_TO_UM):
    """The model carried into the Rayleigh-Jeans tail it is already in.

    Anchored on the reddest point the model has, which for these is three
    microns and long past the peak of anything in the grid.
    """
    if wave_aa[-1] * 1e-4 >= to_um or not flux_um[-1] > 0:
        return wave_aa, flux_um

    tail = np.geomspace(wave_aa[-1] * 1.001, to_um * 1e4, 60)

    return (np.concatenate([wave_aa, tail]),
            np.concatenate([flux_um,
                            flux_um[-1] * (tail / wave_aa[-1]) ** RAYLEIGH_JEANS]))


def spectra_axis():
    """The common wavelength the spectra are resampled onto, in microns."""
    return passbands.log_axis(*SPECTRA_RANGE_UM, SPECTRA_RESOLUTION)


def ingest(path, cube_path, spectra_path, name, label=None, description=None,
           verbose=None):
    """Read a directory of Koester models and write the two files a grid is."""
    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    files = sorted(glob.glob(os.path.join(path, '*.txt'))
                   + glob.glob(os.path.join(path, '*.dat')))
    if not files:
        raise SourceError(f'no model files in {path}')

    bands = passbands.filter_set()
    axis = spectra_axis()

    log(f'{len(files)} files in {os.path.basename(os.path.normpath(path))}, '
        f'{len(bands)} passbands, spectra on {len(axis)} wavelengths')

    teff, logg, feh = [], [], []
    fluxes, spectra = [], []

    for n, each in enumerate(files):
        read = read_model(each)
        if read is None:
            log(f'  {os.path.basename(each)}: no temperature or gravity in it')
            continue

        t, g, wave_aa, flux_um = read

        teff.append(t)
        logg.append(g)
        feh.append(0.0)
        fluxes.append(passbands.convolve(wave_aa, flux_um, bands))
        spectra.append(np.interp(axis, wave_aa * 1e-4, flux_um,
                                 left=0.0, right=0.0))

        if not (n + 1) % 100:
            log(f'  {n + 1} of {len(files)}')

    if not fluxes:
        raise SourceError(f'no models read from {path}')

    fluxes = np.array(fluxes)
    covered = np.isfinite(fluxes).any(axis=0)

    log(f'\n  {len(fluxes)} models, {len(np.unique(teff))} temperatures '
        f'{min(teff):.0f} to {max(teff):.0f} K, '
        f'{len(np.unique(logg))} gravities {min(logg):.1f} to {max(logg):.1f}')
    log(f'  {covered.sum()} of {len(bands)} passbands reached; the models end '
        f'at 3 um and are carried to {EXTEND_TO_UM} um as lambda^'
        f'{RAYLEIGH_JEANS:.0f}, which is what they already go as')

    store.write(cube_path, spectra_path, name=name,
                teff=teff, logg=logg, feh=feh, fluxes=fluxes, bands=bands,
                wave_um=axis, spectra=spectra,
                label=label, description=description, reach_um=REACH_UM,
                source=f'Koester via SVO, {os.path.basename(os.path.normpath(path))}',
                reference='http://svo2.cab.inta-csic.es/theory/newov2/')

    return len(fluxes)
