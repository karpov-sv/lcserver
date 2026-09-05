"""Putting a model spectrum through the filters, the way every grid here was.

A band has to mean the same thing whichever grid answers for it, or the
temperature two grids disagree about is an artefact of how their cubes were
built rather than of the atmospheres in them. So there is one function that
does it, over the passbands ``processing.filters`` defines, and every ingest
calls it.

The check that it is the right function is that it reproduces the grids that
are already here: convolving one of their own cached spectra this way returns
their stored value to a few parts in a thousand, and against TLUSTY - whose
cube was built from the very file we read - to five figures.
"""

import numpy as np


def filter_set():
    """The passbands to convolve through, named as every other grid names them."""
    from ..processing.filters import FILTER_NAMES

    return list(FILTER_NAMES)


def convolve(wave_aa, flux_um, bands):
    """The model through each passband, as the flux a filter would report.

    pyphot's photon-weighted mean flux density. A band the model does not span
    comes back as nothing rather than as an extrapolation wearing a
    measurement's clothes.
    """
    from ..processing.filters import get_filter

    out = np.full(len(bands), np.nan)
    for i, band in enumerate(bands):
        try:
            passband = get_filter(band)
            edges = passband.wavelength.to('AA').value
        except Exception:
            continue

        if edges.min() < wave_aa[0] or edges.max() > wave_aa[-1]:
            continue

        # The model is sampled where it needs to be sampled, which in an
        # infrared continuum is hardly anywhere: a band ten per cent wide at
        # 3.7 um can have two points in it, and a convolution over two points
        # returns nothing. So the filter's own wavelengths are added to the
        # model's first. Nothing is invented by it - between two samples of a
        # smooth continuum the grid says there is a straight line - and a band
        # that was already well sampled comes out exactly where it was.
        inside = (wave_aa >= edges.min()) & (wave_aa <= edges.max())
        grid = np.union1d(wave_aa[inside], edges)
        sampled = np.interp(grid, wave_aa, flux_um)

        try:
            value = float(np.asarray(passband.get_flux(grid, sampled, axis=-1)))
        except Exception:
            continue

        if np.isfinite(value) and value > 0:
            out[i] = value

    return out


def log_axis(lo_um, hi_um, resolution):
    """A wavelength axis of so many resolving elements per e-fold, in microns.

    For a grid whose models are each sampled differently and have to be put on
    one axis to be stored. A grid that samples every model the same way needs
    none of this - its own axis is kept.
    """
    n = int(np.ceil(np.log(hi_um / lo_um) * resolution))

    return np.geomspace(lo_um, hi_um, n)
