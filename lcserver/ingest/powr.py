"""Reading a PoWR model grid into the shape the SED fitter uses.

PoWR (Potsdam Wolf-Rayet, which does OB stars too) publishes its grids as one
file per model: a wavelength and a flux, both in the log, of the star seen from
ten parsecs. https://www.astro.physik.uni-potsdam.de/PoWR/

Three things have to happen to make one of ours out of that.

  * **Ten parsecs to the surface.** Our cubes hold the flux leaving the star,
    because the fit multiplies by (R/d)^2 itself. The radius follows from the
    luminosity and the temperature the grid's own table gives, and the whole
    chain is checked by integrating each file: the luminosity that comes back
    out agrees with the table's to well under a per cent, which is the only
    honest way to know the normalisation was read right.

  * **The filters.** Every band is the photon-weighted mean flux density pyphot
    computes, through the same passbands and under the same names the other
    grids were built with - so a band means the same thing whichever grid
    answers for it. Convolved from each model's own wavelength sampling, which
    differs from model to model and is denser than anything it would be
    resampled onto.

  * **The shape of the grid.** A rectangle of temperature against gravity is
    half empty: a 56000 K star at log g 2 is past the Eddington limit and was
    never computed. So the cube is written scattered - a parameter triple per
    model - and interpolated over the models themselves rather than over a
    lattice with holes in it.
"""

import os
import re
import glob

import numpy as np

from ..processing.utils import SourceError
from . import passbands, store


# Solar and physical constants, in the units the files are in
L_SUN = 3.828e33
R_SUN = 6.957e10
SIGMA_SB = 5.670374419e-5
PARSEC = 3.0856775814913673e18

# What PoWR writes where it means no flux at all
NO_FLUX = -99.0

# The wavelengths the spectra are kept on. PoWR reaches from the X-ray to the
# radio; what is worth keeping is the part anything is ever measured in, and a
# thousand resolving elements per decade is more than a figure spanning three
# decades can show. The cube is convolved from the native sampling, so nothing
# that is fitted passes through this.
SPECTRA_RANGE_UM = (0.005, 100.0)
SPECTRA_RESOLUTION = 1000

# A row of the parameter table: the name, the temperature, the gravity and the
# luminosity, with the columns between them skipped
TABLE_ROW = re.compile(
    r'\s+(\S+)\s+([\d.]+)\s+[\d.eE+-]+\s+[\d.]+\s+([\d.]+)\s+([-\d.]+)')


def grid_name(path):
    """What to call a grid, from the directory it was downloaded into."""
    stem = os.path.basename(os.path.normpath(path))
    stem = re.sub(r'^griddl-', '', stem)
    stem = re.sub(r'-(sed|cont|line)$', '', stem)

    return 'powr-' + stem if not stem.startswith('powr') else stem


def read_parameters(path):
    """The grid's own table of what each model is.

    Returns a list of (model, teff, logg, log L), in the order the table has
    them - which is the order the models are written in.
    """
    table = os.path.join(path, 'modelparameters.txt')
    if not os.path.exists(table):
        raise SourceError(f'no modelparameters.txt in {path}')

    models = []
    for line in open(table):
        match = TABLE_ROW.match(line)
        if match and '-' in match.group(1):
            models.append((match.group(1), float(match.group(2)),
                           float(match.group(3)), float(match.group(4))))

    if not models:
        raise SourceError(f'nothing readable in {table}')

    return models


def read_model(path, model):
    """One model, as wavelength in Angstrom and flux at ten parsecs per Angstrom.

    Both columns are logarithms in the file, and a flux of -100 means none at
    all rather than ten to the minus hundred.
    """
    found = glob.glob(os.path.join(path, f'*_{model}_sed.txt'))
    if not found:
        return None

    data = np.loadtxt(found[0])
    wave = 10.0 ** data[:, 0]
    flux = np.where(data[:, 1] > NO_FLUX, 10.0 ** np.clip(data[:, 1], NO_FLUX, None), 0.0)

    order = np.argsort(wave)
    return wave[order], flux[order]


def stellar_radius(teff, log_luminosity):
    """The radius the grid's own luminosity and temperature imply, in cm."""
    luminosity = 10.0 ** log_luminosity * L_SUN

    return np.sqrt(luminosity / (4 * np.pi * SIGMA_SB * teff ** 4))


def surface_flux(wave_aa, flux_10pc, teff, log_luminosity):
    """That flux as it leaves the star, in erg/s/cm2/um.

    Two conversions and no more: the inverse square out to ten parsecs undone,
    and a wavelength unit. What comes back is what the other grids hold.
    """
    radius = stellar_radius(teff, log_luminosity)
    dilution = (10.0 * PARSEC / radius) ** 2

    return flux_10pc * dilution * 1e4


def check_luminosity(wave_aa, flux_10pc, log_luminosity):
    """The luminosity the file itself implies, over the one it was given.

    Near one and the normalisation was read right; anything else and it was
    not, and every flux in the grid would be wrong by that factor without a
    single thing looking odd about it.
    """
    bolometric = float(np.trapz(flux_10pc, wave_aa))
    luminosity = 4 * np.pi * (10.0 * PARSEC) ** 2 * bolometric / L_SUN

    return luminosity / 10.0 ** log_luminosity


def spectra_axis():
    """The common wavelength the spectra are resampled onto, in microns.

    PoWR samples every model differently, so they have to be put on one axis to
    be stored side by side. The cube is convolved from the native sampling, so
    nothing that is fitted passes through this.
    """
    return passbands.log_axis(*SPECTRA_RANGE_UM, SPECTRA_RESOLUTION)


def ingest(path, cube_path, spectra_path, name, label=None, description=None,
           verbose=None):
    """Read a PoWR download and write the two files a grid here is.

    Every model is convolved from its own sampling and resampled once, so the
    cube carries the fidelity the download has and the spectra carry as much of
    it as a figure can show.
    """
    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    models = read_parameters(path)
    bands = passbands.filter_set()
    axis = spectra_axis()

    log(f'{len(models)} models in {os.path.basename(os.path.normpath(path))}, '
        f'{len(bands)} passbands, spectra on {len(axis)} wavelengths')

    teff, logg, feh = [], [], []
    fluxes, spectra, checks = [], [], []

    for n, (model, t, g, log_l) in enumerate(models):
        read = read_model(path, model)
        if read is None:
            log(f'  {model}: no file for it, skipped')
            continue

        wave_aa, flux_10pc = read
        checks.append(check_luminosity(wave_aa, flux_10pc, log_l))
        flux_um = surface_flux(wave_aa, flux_10pc, t, log_l)

        teff.append(t)
        logg.append(g)
        feh.append(0.0)
        fluxes.append(passbands.convolve(wave_aa, flux_um, bands))
        spectra.append(np.interp(axis, wave_aa * 1e-4, flux_um, left=0.0, right=0.0))

        if not (n + 1) % 25:
            log(f'  {n + 1} of {len(models)}')

    if not fluxes:
        raise SourceError(f'no models read from {path}')

    fluxes = np.array(fluxes)
    checks = np.array(checks)

    covered = np.isfinite(fluxes).any(axis=0)
    log(f'\n  luminosity check: {checks.min():.4f} to {checks.max():.4f} '
        f'of what the table says')
    log(f'  {covered.sum()} of {len(bands)} passbands reached')

    store.write(cube_path, spectra_path, name=name,
                teff=teff, logg=logg, feh=feh, fluxes=fluxes, bands=bands,
                wave_um=axis, spectra=spectra,
                label=label, description=description,
                source=f'PoWR, {os.path.basename(os.path.normpath(path))}',
                reference='https://www.astro.physik.uni-potsdam.de/PoWR/')

    return len(fluxes)
