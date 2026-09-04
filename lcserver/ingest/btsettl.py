"""Reading BT-Settl's medium-resolution spectra out of the grid file for them.

BT-Settl is the widest grid here and until now the one drawn shortest: its
spectra came from astroARIADNE's cache, which stops at 4.63 microns, so the
default grid had no line under W3 or W4. The medium-resolution grid published
for pystellibs carries them from a nanometre to a millimetre.
https://github.com/mfouesneau/pystellibs

The file is one array of every spectrum with the wavelength as its last row,
and a table of what each model is beside it. The flux is already the one our
cubes hold, which is not assumed - convolving it reproduces the stored fluxes
of the cube here to four places, in every band out to W3.

Only the spectra are written. The grid covers 8132 of the cube's 14183 models -
it starts at 2600 K where the cube starts at 400, and carries seven
metallicities against eleven - so writing a cube from it would cost the cool
and the metal-poor ends of the grid whose whole reach is the point of it. The
two files are allowed to hold different models; a fit outside these is drawn
per band, as it was before.
"""

import os

import numpy as np

from ..processing.utils import SourceError
from . import store


def ingest(path, spectra_path, name, verbose=None):
    """Read the grid file and write the spectra of it."""
    from astropy.io import fits

    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    with fits.open(path, memmap=True) as opened:
        block = opened[0].data
        table = opened[1].data

        if block is None or table is None or len(block) != len(table) + 1:
            raise SourceError(f'{os.path.basename(path)} is not the shape this '
                              f'reads - one row per model and the wavelength last')

        wave_aa = np.asarray(block[-1], dtype=float)
        teff = np.asarray(table['Teff'], dtype=float)
        logg = np.asarray(table['logg'], dtype=float)
        feh = np.asarray(table['logZ'], dtype=float)

        log(f'{len(table)} models, {len(wave_aa)} wavelengths, '
            f'{wave_aa.min() * 1e-4:.4f} to {wave_aa.max() * 1e-4:.0f} um')
        log(f'  {len(np.unique(teff))} temperatures {teff.min():.0f} to '
            f'{teff.max():.0f} K, {len(np.unique(logg))} gravities, '
            f'{len(np.unique(feh))} metallicities')

        # Row by row rather than whole: the file is eight hundred megabytes of
        # double precision and what is written is a third of that in single
        spectra = np.empty((len(table), len(wave_aa)), dtype='float32')
        for n in range(len(table)):
            spectra[n] = np.asarray(block[n], dtype=float) * 1e4

            if not (n + 1) % 2000:
                log(f'  {n + 1} of {len(table)}')

    store.write_spectra(spectra_path, name, teff, logg, feh, wave_aa * 1e-4,
                        spectra,
                        source=f'BT-Settl medium resolution, '
                               f'{os.path.basename(path)}')

    return len(table)
