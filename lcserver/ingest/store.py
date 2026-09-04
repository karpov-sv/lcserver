"""Writing out the two files a model grid is here.

``<name>.h5`` is what the fit interpolates: a flux per passband per model.
Usually that is written as the models themselves rather than as a lattice,
because a rectangle of temperature against gravity is half empty for a hot star
and a lattice with holes in it refuses the band of parameter space around the
region that exists. A grid that did compute every combination keeps its
lattice, since interpolating over a box is better than over a simplex and there
is nothing to recover.

``<name>.spectra.h5`` is what a figure draws: the spectra those fluxes were
summed out of, one to a chunk so that reading one reads one.

Both carry in their attributes what a reader should be told about the grid,
because the directory they sit in is the register of what is installed and
nothing about a grid is written down anywhere else.
"""

import os

import numpy as np

import h5py


def write(cube_path, spectra_path, name, teff, logg, feh, fluxes, bands,
          wave_um=None, spectra=None, label=None, description=None,
          reach_um=None, source=None, reference=None):
    """One grid, as the pair of files the application reads."""
    lattice = as_lattice(teff, logg, feh, fluxes)

    with h5py.File(cube_path, 'w') as h:
        h.attrs['name'] = name
        h.attrs['layout'] = 'lattice' if lattice else 'scattered'
        h.attrs['label'] = label or name
        if description:
            h.attrs['description'] = description
        if reach_um:
            h.attrs['reach_um'] = float(reach_um)
        if source:
            h.attrs['source'] = source
        if reference:
            h.attrs['reference'] = reference

        # The axes in full precision, small as they are: they are the corners a
        # triangulation is built on, and a node rounded to a 32-bit float falls
        # outside the hull it is a vertex of
        axes, block = lattice or ((teff, logg, feh), fluxes)
        h.create_dataset('teff', data=np.asarray(axes[0], dtype='float64'))
        h.create_dataset('logg', data=np.asarray(axes[1], dtype='float64'))
        h.create_dataset('feh', data=np.asarray(axes[2], dtype='float64'))
        h.create_dataset('flux', data=np.asarray(block, dtype='float32'),
                         compression='gzip')
        h.create_dataset('filters',
                         data=np.array(bands, dtype=h5py.string_dtype()))

    if wave_um is None or spectra is None:
        return

    write_spectra(spectra_path, name, teff, logg, feh, wave_um, spectra,
                  source=source)


def write_spectra(path, name, teff, logg, feh, wave_um, spectra, source=None):
    """The spectra of a grid, on their own.

    A grid's spectra need not be the same models as its cube: what a group
    publishes at full wavelength is often a subset of what it computed, and the
    subset is worth having on its own terms. Whoever reads them back finds the
    node nearest what is asked and refuses it if it is not near.
    """
    with h5py.File(path, 'w') as h:
        h.attrs['name'] = name
        if source:
            h.attrs['source'] = source

        h.create_dataset('wavelength', data=np.asarray(wave_um, dtype='float64'))
        h.create_dataset('teff', data=np.asarray(teff, dtype='float64'))
        h.create_dataset('logg', data=np.asarray(logg, dtype='float64'))
        h.create_dataset('z', data=np.asarray(feh, dtype='float64'))
        h.create_dataset('flux', data=np.asarray(spectra, dtype='float32'),
                         chunks=(1, len(wave_um)), compression='gzip')


def as_lattice(teff, logg, feh, fluxes):
    """The models as a lattice, if they are one, and otherwise nothing.

    A grid that computed every combination of its axes loses something by being
    written as scattered points: a box has eight corners to interpolate between
    and a simplex has four, and where there are no holes the box is the better
    of the two. So the shape is read off the models rather than assumed.
    """
    axes = [np.unique(np.asarray(_, dtype=float)) for _ in (teff, logg, feh)]
    shape = [len(_) for _ in axes]

    if int(np.prod(shape)) != len(np.asarray(fluxes)):
        return None

    index = [{v: n for n, v in enumerate(a)} for a in axes]
    cube = np.full((shape[1], shape[0], shape[2],
                    np.asarray(fluxes).shape[1]), np.nan, dtype=float)
    seen = np.zeros(shape[1] * shape[0] * shape[2], dtype=bool)

    for t, g, f, row in zip(teff, logg, feh, fluxes):
        i, j, k = index[1][g], index[0][t], index[2][f]
        flat = (i * shape[0] + j) * shape[2] + k
        if seen[flat]:
            return None

        seen[flat] = True
        cube[i, j, k] = row

    return (axes, cube) if seen.all() else None


def compare(cube_path, against, at=None, bands=None, verbose=None):
    """A cube just written against the one it replaces, band by band.

    Several of these grids are read from the very files astroARIADNE built its
    cubes out of, which is a claim with a number attached: the ratios should be
    ones. They are printed rather than asserted, since a grid that is genuinely
    a different computation will differ and that is not a failure.
    """
    from ..processing import sedfit

    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    bands = bands or ['GROUND_JOHNSON_U', 'GROUND_JOHNSON_V', 'PS1_g',
                      '2MASS_J', '2MASS_Ks', 'WISE_RSR_W1']

    new = sedfit.Grid(cube_path, name='new')
    old = sedfit.Grid(against, name='old')

    log(f"\n  against {os.path.basename(against)}:")
    log(f"    {'Teff':>7}{'logg':>6}{'[Z]':>6}  "
        + ''.join(f'{b.split("_")[-1]:>9}' for b in bands))

    for teff, logg, feh in at or ((20000., 4.0, 0.0), (30000., 4.0, 0.0),
                                 (45000., 4.0, 0.0), (30000., 3.0, -0.3)):
        columns = np.array([new.column[b] for b in bands])
        mine = new.flux(teff, logg, feh, columns)
        theirs = old.flux(teff, logg, feh, np.array([old.column[b] for b in bands]))

        ratios = ''.join(
            f'{a / b:9.4f}' if np.isfinite(a) and np.isfinite(b) and b else f'{"-":>9}'
            for a, b in zip(mine, theirs))
        log(f'    {teff:7.0f}{logg:6.2f}{feh:+6.1f}  {ratios}')
