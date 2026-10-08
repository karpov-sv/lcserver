"""Turning a grid's spectra into the bins IUE spectra are fitted in.

The ultraviolet counterpart of xp.py: ``<name>.iue.h5``, a cube of the same
shape as the grid's own, holding a flux per bin per model instead of a flux per
filter per model. Once it exists an IUE bin is interpolated at the fitted
parameters exactly as a filter is.

The bins are boxcars a hundred Angstrom wide, and the model is integrated over
each rather than sampled - never sampled, or a line half a bin deep could fall
between two points of it. It is not smeared first, as it is for Gaia: the
spectrograph's six Angstrom are a sixteenth of a bin, and smearing by them moves
flux across a bin edge by an amount nobody could measure.
"""

import os

import numpy as np

import h5py

from ..processing.utils import SourceError
from . import store


def bin_spectrum(wave_um, flux, bands):
    """One model through the bins, as the mean flux IUE would have reported."""
    from ..processing import sedfit

    edges = np.array([bands[0]['lo_aa']] + [b['hi_aa'] for b in bands])

    return sedfit.rebin(np.asarray(wave_um, dtype=float) * 1e4, flux, edges)


def build(spectra_path, out_path, name, label=None, verbose=None):
    """Every model of one grid through the bins, as a cube beside its own."""
    from ..processing import sedfit

    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    bands = sedfit.iue_bands()

    with h5py.File(spectra_path, 'r') as h:
        wave_um = np.asarray(h['wavelength'][:], dtype=float)
        teff = np.asarray(h['teff'][:], dtype=float)
        logg = np.asarray(h['logg'][:], dtype=float)
        feh = np.asarray(h['z'][:], dtype=float)
        count = h['flux'].shape[0]

        covered = (wave_um[0] * 1e4 <= sedfit.IUE_RANGE_AA[0]
                   and wave_um[-1] * 1e4 >= sedfit.IUE_RANGE_AA[1])
        if not covered:
            raise SourceError(
                f'{os.path.basename(spectra_path)} runs '
                f'{wave_um[0] * 1e4:.0f} to {wave_um[-1] * 1e4:.0f} A and the '
                f'bins need {sedfit.IUE_RANGE_AA[0]:.0f} to '
                f'{sedfit.IUE_RANGE_AA[1]:.0f}')

        # Only what the bins need is read of each model: a grid's spectra run
        # to the far infrared, and the ultraviolet is a small part of them
        near = np.where((wave_um * 1e4 >= sedfit.IUE_RANGE_AA[0] - 50)
                        & (wave_um * 1e4 <= sedfit.IUE_RANGE_AA[1] + 50))[0]
        lo, hi = max(near[0] - 1, 0), min(near[-1] + 2, len(wave_um))
        wave_um = wave_um[lo:hi]

        log(f'{count} models, {len(bands)} bins of {sedfit.IUE_BIN_AA:.0f} A '
            f'from {bands[0]["lo_aa"]:.0f} to {bands[-1]["hi_aa"]:.0f} A, '
            f'{hi - lo} points of each spectrum')

        fluxes = []
        for n in range(count):
            # One model at a time: the spectra are chunked one to a row
            fluxes.append(bin_spectrum(
                wave_um, np.asarray(h['flux'][n, lo:hi], dtype=float), bands))

            if not (n + 1) % 1000:
                log(f'  {n + 1} of {count}')

    fluxes = np.array(fluxes)
    usable = np.isfinite(fluxes).all(axis=1) & (fluxes > 0).all(axis=1)

    log(f'  {int(usable.sum())} of {count} models have every bin')
    if not usable.any():
        raise SourceError(f'no model of {name} covers the bins')

    store.write(out_path, None, name=f'{name} IUE bins',
                teff=teff[usable], logg=logg[usable], feh=feh[usable],
                fluxes=fluxes[usable], bands=[b['band'] for b in bands],
                label=label or name,
                description=f'IUE in {sedfit.IUE_BIN_AA:.0f} A bins, '
                            f'from the {name} spectra',
                source=os.path.basename(spectra_path))

    return int(usable.sum())


def compare(cube_path, name, verbose=None):
    """What the bins say against GALEX NUV, where the grid has it.

    A check that the bins are the same star and the same order as the grid's
    own filters, not a check that reproduces a number: the NUV band is a
    thousand Angstrom wide and the bin beside its pivot a tenth of that, so a
    fifth either way is ordinary. What this catches is a factor of ten - a
    unit, or a wavelength read as Angstrom that was microns.
    """
    from ..processing import sedfit

    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    bins = sedfit.Grid(cube_path, name=f'{name}-iue')
    grid = sedfit.load_grid(name)

    at = (grid.teff[len(grid.teff) // 2], float(np.median(grid.logg)),
          float(np.median(grid.feh)))

    log(f'\n  at {at[0]:.0f} K, {grid.axis_logg} {at[1]:.2f},'
        f' [Fe/H] {at[2]:+.2f}:')
    for band, near in (('GALEX_NUV', 'IUE_2300'), ('GROUND_JOHNSON_U', 'IUE_3100')):
        if band not in grid.covers or near not in bins.covers:
            continue

        one = grid.flux(*at, np.array([grid.column[band]]))[0]
        other = bins.flux(*at, np.array([bins.column[near]]))[0]
        if np.isfinite(one) and np.isfinite(other) and one > 0:
            log(f'    {near} over {band}: {other / one:.3f}')
