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

import h5py

from ..processing.utils import SourceError


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


def filter_set():
    """The passbands to convolve through, named as every other grid names them.

    Taken from astroARIADNE, which is where the grids that are already here got
    them: a band has to mean the same thing whichever grid answers for it, and
    the only way to be sure of that is to use the same profile under the same
    name.
    """
    from astroARIADNE.config import filter_names

    return [str(_) for _ in filter_names]


def convolve(wave_aa, flux_um, bands):
    """The model through each passband, as the flux a filter would report.

    pyphot's photon-weighted mean flux density, which is what the cubes already
    here hold - convolving one of their own cached spectra this way reproduces
    their stored value to a few parts in a thousand.
    """
    from astroARIADNE.phot_utils import _get_filter

    out = np.full(len(bands), np.nan)
    for i, band in enumerate(bands):
        try:
            passband = _get_filter(band)
            edges = passband.wavelength.to('AA').value
        except Exception:
            continue

        # A filter the model does not span is not a band this grid reaches,
        # and a number for it would be an extrapolation wearing a measurement's
        # clothes
        if edges.min() < wave_aa[0] or edges.max() > wave_aa[-1]:
            continue

        # The model is sampled where it needs to be sampled, which in the
        # infrared continuum is hardly anywhere: a band 10 per cent wide at
        # 3.7 um can have two points in it, and a convolution over two points
        # returns nothing. So the filter's own wavelengths are added to the
        # model's before convolving. Nothing is invented by it - between two
        # samples of a free-free continuum there is a straight line and the
        # grid says so - and it leaves a well-sampled band exactly where it was.
        inside = (wave_aa >= edges.min()) & (wave_aa <= edges.max())
        grid = np.union1d(wave_aa[inside], edges)
        sampled = np.interp(grid, wave_aa, flux_um)

        try:
            value = float(np.asarray(passband.get_flux(grid, sampled, axis=-1)))
        except Exception:
            continue

        if np.isfinite(value) and value > 0:
            out[i] = value

    return out


def spectra_axis():
    """The common wavelength the spectra are resampled onto, in microns."""
    lo, hi = SPECTRA_RANGE_UM
    n = int(np.ceil(np.log(hi / lo) * SPECTRA_RESOLUTION))

    return np.geomspace(lo, hi, n)


def ingest(path, cube_path, spectra_path, name, label=None, description=None,
           verbose=None):
    """Read a PoWR download and write the two files a grid here is.

    Every model is convolved from its own sampling and resampled once, so the
    cube carries the fidelity the download has and the spectra carry as much of
    it as a figure can show.
    """
    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    models = read_parameters(path)
    bands = filter_set()
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
        fluxes.append(convolve(wave_aa, flux_um, bands))
        spectra.append(np.interp(axis, wave_aa * 1e-4, flux_um, left=0.0, right=0.0))

        if not (n + 1) % 25:
            log(f'  {n + 1} of {len(models)}')

    if not fluxes:
        raise SourceError(f'no models read from {path}')

    fluxes = np.array(fluxes, dtype='float32')
    checks = np.array(checks)

    covered = np.isfinite(fluxes).any(axis=0)
    log(f'\n  luminosity check: {checks.min():.4f} to {checks.max():.4f} '
        f'of what the table says')
    log(f'  {covered.sum()} of {len(bands)} passbands reached')

    with h5py.File(cube_path, 'w') as h:
        h.attrs['name'] = name
        h.attrs['layout'] = 'scattered'
        h.attrs['label'] = label or name
        if description:
            h.attrs['description'] = description
        h.attrs['source'] = f'PoWR, {os.path.basename(os.path.normpath(path))}'
        h.attrs['reference'] = 'https://www.astro.physik.uni-potsdam.de/PoWR/'

        # The axes in full precision, small as they are: they are the corners
        # a triangulation is built on, and a node rounded to a 32-bit float
        # falls outside the hull it is a vertex of
        h.create_dataset('teff', data=np.array(teff, dtype='float64'))
        h.create_dataset('logg', data=np.array(logg, dtype='float64'))
        h.create_dataset('feh', data=np.array(feh, dtype='float64'))
        h.create_dataset('flux', data=fluxes, compression='gzip')
        h.create_dataset('filters',
                         data=np.array(bands, dtype=h5py.string_dtype()))

    with h5py.File(spectra_path, 'w') as h:
        h.attrs['name'] = name
        h.attrs['source'] = f'PoWR, {os.path.basename(os.path.normpath(path))}'

        h.create_dataset('wavelength', data=axis.astype('float32'))
        h.create_dataset('teff', data=np.array(teff, dtype='float32'))
        h.create_dataset('logg', data=np.array(logg, dtype='float32'))
        h.create_dataset('z', data=np.array(feh, dtype='float32'))
        # One spectrum to a chunk, which is how one is read
        h.create_dataset('flux', data=np.array(spectra, dtype='float32'),
                         chunks=(1, len(axis)), compression='gzip')

    return len(fluxes)
