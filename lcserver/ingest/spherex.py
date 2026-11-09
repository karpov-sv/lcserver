"""Turning a grid's spectra into the bins SPHEREx spectra are fitted in.

The near-infrared counterpart of xp.py and iue.py: ``<name>.spherex.h5``, a
cube of the same shape as the grid's own, holding a flux per bin per model
instead of a flux per filter per model. Once it exists a SPHEREx bin is
interpolated at the fitted parameters exactly as a filter is.

Each model goes through what the exposures did: the filter's own width, which
over the four blue detectors is half a bin, and then the bin. The two together
are one weight per wavelength - see sedfit.spherex_response - integrated over
the model as it is, never sampled.

Several grids' spectra stop before five microns - the PHOENIX family at 4.63,
Koester's at three - and those get the bins they reach and no others. A fit
that asks for the rest then skips the grid and says why, rather than a cube
holding a number nobody computed.
"""

import os

import numpy as np

import h5py

from ..processing.utils import SourceError
from . import store


def reached(wave_um, bands):
    """The bins a spectrum covers, response and all."""
    from ..processing import sedfit

    out = []
    for band in bands:
        sigma = (band['wave_um']
                 / float(sedfit.spherex_resolution(band['wave_um'])[0]) / 2.3548)
        if (wave_um[0] <= band['lo_um'] - 4 * sigma
                and wave_um[-1] >= band['hi_um'] + 4 * sigma):
            out.append(band)

    return out


def responses(wave_um, bands):
    """Each bin's weights on this wavelength axis, normalised to integrate to one."""
    from ..processing import sedfit

    out = []
    for band in bands:
        weight = sedfit.spherex_response(wave_um, band)
        out.append(weight / np.trapz(weight, wave_um))

    return np.array(out)


def bin_spectrum(wave_um, flux, weights):
    """One model through the bins, as the mean flux SPHEREx would have reported."""
    return np.trapz(weights * np.asarray(flux, dtype=float)[None, :],
                        wave_um, axis=1)


def build(spectra_path, out_path, name, label=None, verbose=None):
    """Every model of one grid through the bins, as a cube beside its own."""
    from ..processing import sedfit

    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    with h5py.File(spectra_path, 'r') as h:
        wave_um = np.asarray(h['wavelength'][:], dtype=float)
        teff = np.asarray(h['teff'][:], dtype=float)
        logg = np.asarray(h['logg'][:], dtype=float)
        feh = np.asarray(h['z'][:], dtype=float)
        count = h['flux'].shape[0]

        bands = reached(wave_um, sedfit.spherex_bands())
        if not bands:
            raise SourceError(
                f'{os.path.basename(spectra_path)} runs {wave_um[0]:.2f} to '
                f'{wave_um[-1]:.2f} um and reaches none of the bins')

        # Only what the bins need is read of each model
        lo_um, hi_um = bands[0]['lo_um'] * 0.9, bands[-1]['hi_um'] * 1.1
        near = np.where((wave_um >= lo_um) & (wave_um <= hi_um))[0]
        lo, hi = max(near[0] - 1, 0), min(near[-1] + 2, len(wave_um))
        wave_um = wave_um[lo:hi]

        weights = responses(wave_um, bands)

        log(f'{count} models, {len(bands)} of {sedfit.SPHEREX_NBINS} bins from '
            f'{bands[0]["lo_um"]:.3f} to {bands[-1]["hi_um"]:.3f} um, '
            f'{hi - lo} points of each spectrum')

        fluxes = []
        for n in range(count):
            # One model at a time: the spectra are chunked one to a row
            fluxes.append(bin_spectrum(
                wave_um, np.asarray(h['flux'][n, lo:hi], dtype=float), weights))

            if not (n + 1) % 1000:
                log(f'  {n + 1} of {count}')

    fluxes = np.array(fluxes)
    usable = np.isfinite(fluxes).all(axis=1) & (fluxes > 0).all(axis=1)

    log(f'  {int(usable.sum())} of {count} models have every bin')
    if not usable.any():
        raise SourceError(f'no model of {name} covers the bins')

    store.write(out_path, None, name=f'{name} SPHEREx bins',
                teff=teff[usable], logg=logg[usable], feh=feh[usable],
                fluxes=fluxes[usable], bands=[b['band'] for b in bands],
                label=label or name,
                description=f'SPHEREx in {len(bands)} bins of R = '
                            f'{sedfit.SPHEREX_NBINS / np.log(sedfit.SPHEREX_RANGE_UM[1] / sedfit.SPHEREX_RANGE_UM[0]):.0f}, '
                            f'from the {name} spectra',
                source=os.path.basename(spectra_path))

    return int(usable.sum())


def compare(cube_path, name, verbose=None):
    """What the bins say against the broad bands beside them.

    A check that the bins are the same star and the same order as the grid's
    own filters, not one that reproduces a number: a bin a twentieth of a
    micron wide against a band of a quarter is ordinarily a few per cent off on
    a smooth photosphere. What this catches is a factor of ten - a unit, or a
    wavelength read as Angstrom that was microns.
    """
    from ..processing import sedfit

    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    bins = sedfit.Grid(cube_path, name=f'{name}-spherex')
    grid = sedfit.load_grid(name)

    at = (grid.teff[len(grid.teff) // 2], float(np.median(grid.logg)),
          float(np.median(grid.feh)))

    log(f'\n  at {at[0]:.0f} K, {grid.axis_logg} {at[1]:.2f},'
        f' [Fe/H] {at[2]:+.2f}:')
    for band in ('2MASS_J', '2MASS_H', '2MASS_Ks', 'WISE_RSR_W1', 'WISE_RSR_W2'):
        if band not in grid.covers:
            continue

        pivot = sedfit.pivot_aa(band) * 1e-4
        near = min((b for b in sedfit.spherex_bands() if b['band'] in bins.covers),
                   key=lambda b: abs(np.log(b['wave_um'] / pivot)), default=None)
        if near is None or abs(np.log(near['wave_um'] / pivot)) > 0.05:
            continue

        one = grid.flux(*at, np.array([grid.column[band]]))[0]
        other = bins.flux(*at, np.array([bins.column[near['band']]]))[0]
        if np.isfinite(one) and np.isfinite(other) and one > 0:
            log(f"    {near['band']} over {band}: {other / one:.3f}")
