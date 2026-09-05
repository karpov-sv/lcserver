"""Turning a grid's spectra into the bins Gaia's spectra are compared in.

Gaia publishes a spectrum for two hundred million sources, and a fit here can
use one the way it uses a magnitude - as long as something can say what a model
would look like through the same bins. That is what this writes:
``<name>.xp.h5``, a cube of the same shape as the grid's own, holding a flux per
bin per model instead of a flux per filter per model.

Once it exists nothing about an XP bin is special. It is interpolated at the
fitted parameters exactly as a filter is, by the same code, out of a file of the
same layout - so a bin can be fitted, left out, drawn, or compared with, and
none of that needs a spectrum at run time or a node standing in for the fit.

The bins are not filters and are not convolved like them. They are boxcars
twice the width of the instrument's line spread function, and the model goes
through what the data went through before it is binned: integrated onto a fine
axis - never sampled, or a line half a bin deep could fall between two points of
it - and smeared by that function, so that a bin edge landing on a Balmer jump
gets the flux the spectrograph would have put across it.
"""

import os

import numpy as np

import h5py

from ..processing.utils import SourceError
from . import store


def bin_spectrum(wave_um, flux, bands):
    """One model through the bins, as the flux Gaia would have reported."""
    from ..processing import sedfit

    axis_nm, smoothed = sedfit.at_xp_resolution(wave_um, flux)
    if axis_nm is None:
        return None

    return np.array([np.mean(smoothed[(axis_nm >= b['lo_nm'])
                                      & (axis_nm < b['hi_nm'])])
                     for b in bands])


def build(spectra_path, out_path, name, label=None, verbose=None):
    """Every model of one grid through the bins, as a cube beside its own."""
    from ..processing import sedfit

    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    bands = sedfit.xp_bands()

    with h5py.File(spectra_path, 'r') as h:
        wave_um = np.asarray(h['wavelength'][:], dtype=float)
        teff = np.asarray(h['teff'][:], dtype=float)
        logg = np.asarray(h['logg'][:], dtype=float)
        feh = np.asarray(h['z'][:], dtype=float)
        count = h['flux'].shape[0]

        covered = (wave_um[0] * 1e3 <= sedfit.XP_RANGE_NM[0]
                   and wave_um[-1] * 1e3 >= sedfit.XP_RANGE_NM[1])
        if not covered:
            raise SourceError(
                f'{os.path.basename(spectra_path)} runs '
                f'{wave_um[0] * 1e3:.0f} to {wave_um[-1] * 1e3:.0f} nm and the '
                f'bins need {sedfit.XP_RANGE_NM[0]:.0f} to '
                f'{sedfit.XP_RANGE_NM[1]:.0f}')

        log(f'{count} models, {len(bands)} bins of {sedfit.XP_BIN_NM:.0f} nm '
            f'from {bands[0]["lo_nm"]:.0f} to {bands[-1]["hi_nm"]:.0f} nm')

        fluxes = []
        for n in range(count):
            # One model at a time: the spectra are chunked one to a row, and a
            # grid of these is most of a gigabyte read whole
            binned = bin_spectrum(wave_um, np.asarray(h['flux'][n], dtype=float),
                                  bands)
            if binned is None:
                raise SourceError(f'model {n} does not span the bins')

            fluxes.append(binned)

            if not (n + 1) % 1000:
                log(f'  {n + 1} of {count}')

    fluxes = np.array(fluxes)
    usable = np.isfinite(fluxes).all(axis=1) & (fluxes > 0).all(axis=1)

    log(f'  {int(usable.sum())} of {count} models have every bin')
    if not usable.any():
        raise SourceError(f'no model of {name} covers the bins')

    store.write(out_path, None, name=f'{name} XP bins',
                teff=teff[usable], logg=logg[usable], feh=feh[usable],
                fluxes=fluxes[usable], bands=[b['band'] for b in bands],
                label=label or name,
                description=f'Gaia XP in {sedfit.XP_BIN_NM:.0f} nm bins, '
                            f'from the {name} spectra',
                source=os.path.basename(spectra_path))

    return int(usable.sum())


def compare(cube_path, spectra_path, name, verbose=None):
    """What the bins say against what the grid's own filters say.

    Not a check that reproduces a number - nothing here was built from a cube
    of bins before - but a check that they are the same star and the same
    order. A bin is thirty nanometres and a filter is a hundred and more, at a
    different mean wavelength on a sloping continuum, so a fifth either way is
    ordinary. Larger differences are ordinary too where a line is: on a WC
    grid a bin between the carbon lines is a fifth of the r band that
    contains them, which is the emission and not a mistake. What this catches
    is a factor of ten - a unit, a normalisation, a wavelength read as
    Angstrom that was microns.
    """
    from ..processing import sedfit

    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    bins = sedfit.Grid(cube_path, name=f'{name}-xp')
    grid = sedfit.load_grid(name)

    at = (grid.teff[len(grid.teff) // 2], float(np.median(grid.logg)),
          float(np.median(grid.feh)))

    log(f'\n  at {at[0]:.0f} K, {grid.axis_logg} {at[1]:.2f},'
        f' [Fe/H] {at[2]:+.2f}:')
    for band, near in (('SDSS_g', 'GAIA_XP_455'), ('GROUND_JOHNSON_V', 'GAIA_XP_545'),
                       ('SDSS_r', 'GAIA_XP_635'), ('SDSS_i', 'GAIA_XP_755')):
        if band not in grid.covers or near not in bins.covers:
            continue

        one = grid.flux(*at, np.array([grid.column[band]]))[0]
        other = bins.flux(*at, np.array([bins.column[near]]))[0]
        if np.isfinite(one) and np.isfinite(other) and one > 0:
            log(f'    {near} over {band}: {other / one:.3f}')
