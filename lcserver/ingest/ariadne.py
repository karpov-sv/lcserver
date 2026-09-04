"""Laying astroARIADNE's grids out the way this application reads them.

astroARIADNE ships one cube per grid, named for the file rather than for the
grid, and every spectrum of every grid in a single cache of some gigabytes.
That is a fine shape for a package that ships them and a poor one for a
directory somebody adds to: appending a grid means rewriting three gigabytes,
and what is known about a grid is known only to whatever reads it.

Here each grid becomes ``<name>.h5`` and, where there are any, an optional
``<name>.spectra.h5`` beside it, both carrying in their attributes what a
reader should be told. Nothing is recomputed - the cube is copied as it is and
the spectra dataset by dataset, so the one-spectrum-to-a-chunk compression
comes across untouched and the split costs seconds rather than an afternoon.
"""

import os
import shutil

import h5py


def split(entry, target, cache=None, force=False, verbose=None):
    """One grid, from astroARIADNE's layout into ours.

    ``entry`` is a row of ``sedfit.grid_registry``, which is what knows where
    the cube is and what was known about the grid before the file could say it
    for itself.
    """
    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    name = entry['name']
    cube(entry, target, force, log)
    spectra(name, target, cache, force, log)


def cube(entry, target, force, log):
    """The flux-per-filter block, copied and told what it is."""
    name = entry['name']
    out = os.path.join(target, f'{name}.h5')

    if os.path.exists(out) and not force:
        log(f'{name:12s} cube is already there, left alone')
        return

    shutil.copy2(entry['path'], out)

    # What the file could not say for itself, said now: a grid built here
    # carries its own, and a copied one carries what we knew of it
    with h5py.File(out, 'r+') as h:
        h.attrs['name'] = name
        if entry['label']:
            h.attrs['label'] = entry['label']
        if entry['description']:
            h.attrs['description'] = entry['description']
        if entry['reach_um']:
            h.attrs['reach_um'] = float(entry['reach_um'])
        h.attrs['source'] = 'astroARIADNE, ' + os.path.basename(entry['path'])

    log(f'{name:12s} cube {os.path.getsize(out) / 1e6:8.1f} MB')


def spectra(name, target, cache, force, log):
    """The spectra of one grid, lifted out of the single cache."""
    out = os.path.join(target, f'{name}.spectra.h5')

    if os.path.exists(out) and not force:
        log(f'{"":12s} spectra are already there, left alone')
        return

    if not cache:
        return

    with h5py.File(cache, 'r') as h:
        if name not in h:
            log(f'{"":12s} no spectra for it in the cache')
            return

        group = h[name]

        # Dataset by dataset rather than array by array, so the chunking and
        # the compression come across as they are - one spectrum to a chunk,
        # which is how one is read
        with h5py.File(out, 'w') as f:
            for key in group:
                h.copy(group[key], f, name=key)

            f.attrs['name'] = name
            f.attrs['source'] = 'astroARIADNE, ' + os.path.basename(cache)

    log(f'{"":12s} spectra {os.path.getsize(out) / 1e6:5.0f} MB')
