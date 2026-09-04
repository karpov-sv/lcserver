"""Reading a TLUSTY grid out of the merged file it is published as.

TLUSTY's OB-star grids come as one large ASCII file in the format Cloudy reads:
a header saying what the parameters are and how many models and frequencies
there are, then the parameter triple of every model, then the frequencies they
all share, then every model's flux in turn.
https://tlusty.oca.eu/tlusty/Tlusty2002/tlusty-cloudy.html

Two things make this worth reading rather than the original per-model files.

  * It is what the cube already here was built from. Convolving it through the
    passbands returns astroARIADNE's stored TLUSTY fluxes to five figures, in
    every band and at every temperature checked - so nothing about the fit
    changes, and the originals would only reproduce a number we have.

  * It carries the spectra, which we have none of for this grid. Nineteen
    thousand frequencies at a uniform resolving power of about 1800, from the
    soft X-ray to 300 microns: enough to resolve the Balmer lines, matched to
    what the viewer draws beside it, and far enough into the infrared that
    nothing has to be extrapolated under W3 and W4.

Every model shares one frequency grid, so the spectra are stored on it as they
are. Nothing is resampled and nothing is lost.
"""

import os

import numpy as np

from ..processing.utils import SourceError
from . import passbands, store


# The speed of light in Angstrom per second, for turning a flux per hertz into
# a flux per wavelength
C_AA = 2.99792458e18

# What the header calls the axes, against what we call them
AXES = {'teff': 'teff', 'log(g)': 'logg', 'log(z)': 'feh'}


class Tokens:
    """Whitespace-separated fields of a file, taken as many as are wanted.

    The file is a third of a gigabyte of text and its records do not line up
    with its lines, so it is read as a stream of fields rather than parsed
    whole - one model at a time is a hundred and sixty kilobytes, and the whole
    of it at once is a couple of gigabytes of Python strings.
    """

    def __init__(self, handle):
        self.handle = handle
        self.buffer = []

    def take(self, count):
        while len(self.buffer) < count:
            line = self.handle.readline()
            if not line:
                raise SourceError('the grid file ends in the middle of a model')
            self.buffer.extend(line.split())

        out, self.buffer = self.buffer[:count], self.buffer[count:]
        return out

    def numbers(self, count):
        return np.array([float(_) for _ in self.take(count)])


def read_header(tokens):
    """What the file says it is, before any of it is read."""
    _, _, npar = tokens.take(3)
    names = [_.lower() for _ in tokens.take(int(npar))]
    models, frequencies = (int(_) for _ in tokens.take(2))
    abscissa, abscissa_scale = tokens.take(2)
    ordinate, ordinate_scale = tokens.take(2)

    unknown = [_ for _ in names if _ not in AXES]
    if unknown:
        raise SourceError(f"the grid varies {', '.join(unknown)}, which this "
                          f"does not know what to do with")

    if abscissa.lower() != 'nu':
        raise SourceError(f'the grid is tabulated against {abscissa}, not nu')

    return {'axes': [AXES[_] for _ in names], 'models': models,
            'frequencies': frequencies,
            'abscissa_scale': float(abscissa_scale),
            'ordinate': ordinate, 'ordinate_scale': float(ordinate_scale)}


def surface_flux(flux_nu, wave_aa, scale):
    """The tabulated ordinate as a surface flux in erg/s/cm2/um.

    TLUSTY publishes the Eddington flux, which the header's own scale factor of
    four pi turns into the flux leaving the surface - the same quantity every
    other cube here holds. The rest is a change of variable from per hertz to
    per wavelength, and a wavelength unit.
    """
    return flux_nu * scale * C_AA / wave_aa ** 2 * 1e4


def ingest(path, cube_path, spectra_path, name, label=None, description=None,
           verbose=None):
    """Read the merged file and write the two files a grid here is."""
    log = verbose if callable(verbose) else (print if verbose else lambda *a: None)

    bands = passbands.filter_set()

    with open(path) as handle:
        tokens = Tokens(handle)
        head = read_header(tokens)
        axes, count = head['axes'], head['models']

        log(f"{count} models over {', '.join(axes)}, "
            f"{head['frequencies']} frequencies, {len(bands)} passbands")
        log(f"ordinate {head['ordinate']} x {head['ordinate_scale']:.6g}")

        parameters = tokens.numbers(count * len(axes)).reshape(count, len(axes))
        frequency = tokens.numbers(head['frequencies'])

        # Ascending in wavelength, which is how everything downstream wants it
        wave_aa = C_AA / (frequency * head['abscissa_scale'])
        order = np.argsort(wave_aa)
        wave_aa = wave_aa[order]

        values = {axis: parameters[:, n] for n, axis in enumerate(axes)}
        teff = values['teff']
        logg = values.get('logg', np.zeros(count))
        feh = values.get('feh', np.zeros(count))

        fluxes, spectra = [], []
        for n in range(count):
            flux_um = surface_flux(tokens.numbers(head['frequencies'])[order],
                                   wave_aa, head['ordinate_scale'])

            fluxes.append(passbands.convolve(wave_aa, flux_um, bands))
            spectra.append(flux_um)

            if not (n + 1) % 100:
                log(f'  {n + 1} of {count}')

    fluxes = np.array(fluxes)
    covered = np.isfinite(fluxes).any(axis=0)

    log(f'\n  {wave_aa.min():.1f} to {wave_aa.max() * 1e-4:.0f} um, '
        f'{covered.sum()} of {len(bands)} passbands reached')

    store.write(cube_path, spectra_path, name=name,
                teff=teff, logg=logg, feh=feh, fluxes=fluxes, bands=bands,
                wave_um=wave_aa * 1e-4, spectra=spectra,
                label=label, description=description,
                source=f'TLUSTY, {os.path.basename(path)}',
                reference='https://tlusty.oca.eu/')

    return count
