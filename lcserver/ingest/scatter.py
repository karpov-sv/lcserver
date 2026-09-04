"""Rewriting a grid that is stored as a lattice as the models it actually has.

A cube on a lattice has a flux for every combination of the three axes, and for
a real grid most of those combinations were never computed - a star of a given
gravity runs out of temperatures at both ends, and past the Eddington limit
there is nothing to compute. Those are stored as nothing, and a box whose eight
corners are not all models cannot be interpolated in, so the fit refuses a band
of parameter space one cell wide all the way around the region that exists.

Written as the models themselves and triangulated over, that band comes back.
On BT-Settl it is a quarter of the reachable space, and where both layouts
answer they agree to about half a per cent - which is the difference between
interpolating over a box and over a simplex, and is smaller than the difference
between any two grids.

Nothing is recomputed here and nothing is downloaded: the fluxes written out
are the ones read in, with the empty nodes dropped.
"""

import os
import shutil

import numpy as np

import h5py


def convert(path, out=None, verbose=None):
    """One grid from a lattice to its models, in place or beside itself."""
    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    with h5py.File(path, 'r') as h:
        if str(h.attrs.get('layout', 'lattice')) == 'scattered':
            log(f'{os.path.basename(path)}: already the models themselves')
            return None

        attrs = dict(h.attrs)
        logg = np.asarray(h['logg'][:], dtype=float)
        teff = np.asarray(h['teff'][:], dtype=float)
        feh = np.asarray(h['feh'][:], dtype=float)
        cube = np.asarray(h['flux'][:])
        bands = h['filters'][:]

    alive = np.isfinite(cube).any(axis=3)
    i, j, k = np.where(alive)
    if not len(i):
        raise ValueError(f'{path} has no models in it')

    log(f'{os.path.basename(path)}: {alive.sum()} models of {alive.size} nodes'
        f' ({100 * alive.sum() / alive.size:.0f}%)')

    # A lattice with every node computed has nothing to recover, and rewriting
    # it would only swap interpolating over a box for interpolating over a
    # simplex - a change with a cost and no gain
    if alive.all():
        log('  every node is a model, so there is nothing a lattice refuses')
        return None

    # Written beside itself and moved over at the end, so an interruption
    # leaves the grid that was there rather than half of one
    target = out or path
    working = target + '.new'

    with h5py.File(working, 'w') as f:
        for key, value in attrs.items():
            f.attrs[key] = value
        f.attrs['layout'] = 'scattered'
        f.attrs['scattered_from'] = 'a lattice of ' + ' x '.join(
            str(_) for _ in (len(logg), len(teff), len(feh)))

        f.create_dataset('teff', data=teff[j])
        f.create_dataset('logg', data=logg[i])
        f.create_dataset('feh', data=feh[k])
        f.create_dataset('flux', data=cube[i, j, k, :], compression='gzip')
        f.create_dataset('filters', data=bands)

    shutil.move(working, target)
    return target


def compare(before, after, samples=4000, seed=0, verbose=None):
    """What the rewriting changed, over the box the grid is defined in.

    Two numbers matter and both are printed. How much more of the space can be
    reached, which is the point of doing it; and how far the two layouts differ
    where both can answer, which is the price - interpolating over a simplex is
    not interpolating over a box, and the models are the same either way.
    """
    from ..processing import sedfit

    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    old = sedfit.Grid(before, name='before')
    new = sedfit.Grid(after, name='after')

    band = next((b for b in ('GROUND_JOHNSON_V', 'PS1_r', '2MASS_J')
                 if b in old.covers and b in new.covers), None)
    if band is None:
        return

    limits = old.limits
    rng = np.random.default_rng(seed)
    points = np.column_stack([rng.uniform(*limits['teff'], samples),
                              rng.uniform(*limits['logg'], samples),
                              rng.uniform(*limits['feh'], samples)])

    was = np.array([old.flux(*p, np.array([old.column[band]]))[0] for p in points])
    now = np.array([new.flux(*p, np.array([new.column[band]]))[0] for p in points])

    both = np.isfinite(was) & np.isfinite(now)
    ratio = now[both] / was[both]

    log(f'  of {samples} points in the box, the lattice answered '
        f'{np.isfinite(was).sum()} and the models {np.isfinite(now).sum()}')
    log(f'    gained {int((np.isfinite(now) & ~np.isfinite(was)).sum())}, '
        f'lost {int((np.isfinite(was) & ~np.isfinite(now)).sum())}')
    log(f'    where both answer, {band} agrees to '
        f'{np.percentile(np.abs(ratio - 1), 95) * 100:.2f} per cent for 95 of them')
