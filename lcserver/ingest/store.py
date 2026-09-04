"""Writing out the two files a model grid is here.

``<name>.h5`` is what the fit interpolates: a flux per passband per model,
scattered rather than on a lattice, because a rectangle of temperature against
gravity is half empty for a hot star and a lattice with holes in it refuses
half the models that do exist.

``<name>.spectra.h5`` is what a figure draws: the spectra those fluxes were
summed out of, one to a chunk so that reading one reads one.

Both carry in their attributes what a reader should be told about the grid,
because the directory they sit in is the register of what is installed and
nothing about a grid is written down anywhere else.
"""

import numpy as np

import h5py


def write(cube_path, spectra_path, name, teff, logg, feh, fluxes, bands,
          wave_um=None, spectra=None, label=None, description=None,
          reach_um=None, source=None, reference=None):
    """One grid, as the pair of files the application reads."""
    with h5py.File(cube_path, 'w') as h:
        h.attrs['name'] = name
        h.attrs['layout'] = 'scattered'
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
        h.create_dataset('teff', data=np.asarray(teff, dtype='float64'))
        h.create_dataset('logg', data=np.asarray(logg, dtype='float64'))
        h.create_dataset('feh', data=np.asarray(feh, dtype='float64'))
        h.create_dataset('flux', data=np.asarray(fluxes, dtype='float32'),
                         compression='gzip')
        h.create_dataset('filters',
                         data=np.array(bands, dtype=h5py.string_dtype()))

    if wave_um is None or spectra is None:
        return

    with h5py.File(spectra_path, 'w') as h:
        h.attrs['name'] = name
        if source:
            h.attrs['source'] = source

        h.create_dataset('wavelength', data=np.asarray(wave_um, dtype='float64'))
        h.create_dataset('teff', data=np.asarray(teff, dtype='float64'))
        h.create_dataset('logg', data=np.asarray(logg, dtype='float64'))
        h.create_dataset('z', data=np.asarray(feh, dtype='float64'))
        h.create_dataset('flux', data=np.asarray(spectra, dtype='float32'),
                         chunks=(1, len(wave_um)), compression='gzip')
