"""What a band name means here: its passband, and which system its magnitudes are on.

The names are the ones every cube in the grid directory is keyed by, so this is
the one place that decides what a band is. They came from astroARIADNE, which
is where this application's grids came from too, and are kept here rather than
imported: what a column in a file we wrote means should not depend on somebody
else's package still being installed, and the grids outlived the need for the
rest of it.

Most of the curves are pyphot's own. Forty are not: the mid- and far-infrared
bands (AKARI, IRAS, Spitzer MIPS and IRS), the SPHEREx super-bands and three
J-PLUS medium bands, which pyphot's bundled library has never carried. Those
are kept beside this file, in ``filters.h5``.

Where this came from
--------------------

The list of names and the collection of curves are both astroARIADNE's, taken
from v1.5.0-53-g5063ea9 of its v2 branch:

    astroARIADNE - https://github.com/jvines/astroARIADNE
    MIT Licence, Copyright (c) 2019 Jose Vines

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

The transmission curves are not that project's work either, and it does not
claim them: it collected them from the **SVO Filter Profile Service**, which is
what each filter's ``svo_id`` in the file beside this one names, and which asks
to be cited as Rodrigo & Solano (2020) and Rodrigo, Solano & Bayo (2012).
http://svo2.cab.inta-csic.es/theory/fps/

The same notice is written into the attributes of ``filters.h5``, so that the
data carries where it came from wherever the file goes.
"""

import os

import numpy as np


# Every band a grid here may hold a flux for, blue to red and then the
# mid-infrared extension, in the order the cubes were built in. A cube says
# which columns it has, so the order is not load-bearing; it is kept so that
# rebuilding a grid produces the file it produced before.
FILTER_NAMES = (
    'GALEX_FUV', 'GALEX_NUV', 'STROMGREN_u', 'SkyMapper_u', 'SDSS_u',
    'GROUND_JOHNSON_U', 'SkyMapper_v', 'STROMGREN_v', 'TYCHO_B_MvB',
    'GROUND_JOHNSON_B', 'STROMGREN_b', 'SDSS_g', 'PS1_g', 'SkyMapper_g',
    'GaiaDR2v2_BP', 'TYCHO_V_MvB', 'STROMGREN_y', 'GROUND_JOHNSON_V',
    'SkyMapper_r', 'SDSS_r', 'PS1_r', 'PS1_w', 'KEPLER_Kp', 'GaiaDR2v2_G',
    'GROUND_COUSINS_R', 'NGTS_I', 'SDSS_i', 'PS1_i', 'SkyMapper_i',
    'GaiaDR2v2_RP', 'GROUND_COUSINS_I', 'TESS', 'PS1_z', 'SDSS_z',
    'SkyMapper_z', 'PS1_y', '2MASS_J', '2MASS_H', '2MASS_Ks', 'WISE_RSR_W1',
    'SPITZER_IRAC_36', 'SPITZER_IRAC_45', 'WISE_RSR_W2', 'WISE_RSR_W3',
    'WISE_RSR_W4', 'HERSCHEL_PACS_BLUE', 'HERSCHEL_PACS_GREEN', 'HERSCHEL_PACS_RED',
    'SPITZER_IRAC_58', 'SPITZER_IRAC_80', 'AKARI_IRC_S9W', 'IRAS_12',
    'AKARI_IRC_L18W', 'IRAS_25', 'SPITZER_MIPS_24', 'IRAS_60', 'SPITZER_MIPS_70',
    'IRAS_100', 'SPITZER_MIPS_160', 'HERSCHEL_SPIRE_PSW', 'HERSCHEL_SPIRE_PMW',
    'HERSCHEL_SPIRE_PLW', 'AKARI_FIS_N60', 'AKARI_FIS_WIDE_S', 'AKARI_FIS_WIDE_L',
    'AKARI_FIS_N160', 'SPHEREX_1050', 'SPHEREX_1130', 'SPHEREX_1400',
    'SPHEREX_1850', 'SPHEREX_2350', 'SPHEREX_2750', 'SPHEREX_3050',
    'SPHEREX_3350', 'SPHEREX_3700', 'SPHEREX_4050', 'SPHEREX_4300',
    'SPHEREX_4650', 'SPHEREX_4900', 'JPLUS_J0515', 'JPLUS_J0660', 'JPLUS_J0861',
    'IRS_5500', 'IRS_6500', 'IRS_8000', 'IRS_10000', 'IRS_12000', 'IRS_13000',
    'IRS_15000', 'IRS_18000', 'IRS_23000', 'IRS_30000', 'IRS_35000',
)

# The curves pyphot does not carry, as this application keeps them
CUSTOM_FILTERS = os.path.join(os.path.dirname(__file__), 'filters.h5')

# Which surveys report AB magnitudes rather than Vega. It is a property of the
# survey and not of the filter, so there is nothing in a passband to read it
# off - it has to be written down, and written down it can go stale: SkyMapper
# was taken as Vega until mid-2026, which made its magnitudes 20 to 60 per cent
# faint in flux, the redder the band the worse.
AB_PREFIXES = ('PS1_', 'SDSS_', 'GALEX_', 'SkyMapper_')


def is_ab(band):
    """Whether this band's magnitudes are AB rather than Vega."""
    return any(prefix in str(band) for prefix in AB_PREFIXES)


_custom = None


def _custom_library():
    """The vendored curves, as pyphot filters, read once."""
    global _custom

    if _custom is not None:
        return _custom

    import h5py
    import astropy.units as u
    import pyphot

    _custom = {}
    if not os.path.isfile(CUSTOM_FILTERS):
        return _custom

    with h5py.File(CUSTOM_FILTERS, 'r') as h:
        for name in h:
            entry = h[name]
            _custom[name] = pyphot.Filter(
                np.asarray(entry['wavelength'][:], dtype=float) * u.AA,
                np.asarray(entry['transmission'][:], dtype=float),
                name=name,
                dtype=str(entry.attrs.get('dtype', 'photon')),
                unit=str(entry.attrs.get('unit', 'Angstrom')))

    return _custom


def get_filter(band):
    """One band's passband: ours where we have it, and pyphot's otherwise.

    Ours first, and not only for the bands pyphot lacks - it carries a Herschel
    SPIRE PLW of its own, and the one here is the curve every cube was
    convolved through.
    """
    import pyphot

    custom = _custom_library()
    if band in custom:
        return custom[band]

    return pyphot.get_library()[band]


def pivot_aa(band):
    """The pivot wavelength, in Angstrom.

    The one that makes <f_lambda> = <f_nu> c / lambda^2 hold for a photon
    counter, which is the convention every flux in a cube is on.
    """
    return float(get_filter(band).lpivot.to('AA').value)


def width_aa(band):
    """How wide the band is, in Angstrom, for drawing it as a range."""
    return float(get_filter(band).width.to('AA').value)


def vega_zero_jy(band):
    """The band's Vega zero point in Jansky, for a magnitude that is not AB."""
    return float(get_filter(band).Vega_zero_Jy.value)
