"""Reading BT-Settl out of the medium-resolution grid file for it.

BT-Settl is the widest grid here and was for a while the one drawn shortest:
its spectra came from astroARIADNE's cache, which stops at 4.63 microns, so
there was no line under W3 or W4. The medium-resolution grid published for
pystellibs carries them from a nanometre to a millimetre.
https://github.com/mfouesneau/pystellibs

The file is one array of every spectrum with the wavelength as its last row,
and a table of what each model is beside it, and its header names which
BT-Settl it is: the AGSS2009 abundances. The flux is already the one our cubes
hold, which is not assumed - convolving it reproduces the stored fluxes of the
cube that was here before to five and six places through the optical and the
near infrared.

Where the two part company is the far infrared, and there it is this one that
is right. astroARIADNE's spectra stop at 4.63 microns, so everything its cube
said past there was a Rayleigh-Jeans continuation of the last real point;
these models are computed out to a millimetre, and a photosphere sits above
that tail. Herschel and IRAS come out one to eight per cent brighter as a
result, always in the same direction.

Both files are written from it, which costs something and is worth saying
plainly. The grid is 8132 models where astroARIADNE's cube was 14183: it
starts at 2600 K rather than 400, and carries seven metallicities rather than
eleven, dropping -4 to -3 at one end and +0.3 at the other. Every model it
does have is one that cube had too. What is bought is a grid that can say
where it came from - the cube astroARIADNE shipped carries no attribution at
all, and a grid that cannot be credited cannot be put on a page crediting the
rest. The brown dwarf and halo extremes it gives up are not what this
application is asked about.

Nothing is padded, so no reach is declared. Past 20 microns a warm model is
Rayleigh-Jeans to three places, which is what a photosphere there should be;
a 2600 K one measurably is not, the molecular bands still being in it. A
continuation would have looked the same at the first and could not have
produced the second.
"""

import os

import numpy as np

from ..processing.utils import SourceError
from . import passbands, store


def ingest(path, cube_path, spectra_path, name, label=None, description=None,
           verbose=None):
    """Read the grid file and write the two files a grid here is."""
    from astropy.io import fits

    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    bands = passbands.filter_set()

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
            f'{wave_aa.min() * 1e-4:.4f} to {wave_aa.max() * 1e-4:.0f} um, '
            f'{len(bands)} passbands')
        log(f'  {len(np.unique(teff))} temperatures {teff.min():.0f} to '
            f'{teff.max():.0f} K, {len(np.unique(logg))} gravities, '
            f'{len(np.unique(feh))} metallicities')

        # Row by row rather than whole: the file is eight hundred megabytes of
        # double precision and what is written is a third of that in single
        spectra = np.empty((len(table), len(wave_aa)), dtype='float32')
        fluxes = np.empty((len(table), len(bands)))

        for n in range(len(table)):
            flux = np.asarray(block[n], dtype=float) * 1e4
            spectra[n] = flux
            fluxes[n] = passbands.convolve(wave_aa, flux, bands)

            if not (n + 1) % 500:
                log(f'  {n + 1} of {len(table)}')

    covered = np.isfinite(fluxes).any(axis=0)
    log(f'\n  {covered.sum()} of {len(bands)} passbands reached')

    store.write(cube_path, spectra_path, name=name,
                teff=teff, logg=logg, feh=feh, fluxes=fluxes, bands=bands,
                wave_um=wave_aa * 1e-4, spectra=spectra,
                label=label, description=description,
                source=f'BT-Settl AGSS2009 via pystellibs, '
                       f'{os.path.basename(path)}',
                reference='https://svo2.cab.inta-csic.es/theory/newov2/'
                          'index.php?models=bt-settl-agss')

    return len(table)
