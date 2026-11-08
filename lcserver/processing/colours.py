"""What a colour of the light curve viewer would do were the star a blackbody,
or were it reddened: the references its colour diagrams are read against.

Each series of the viewer is a survey in one band, named as the survey names
it. Here it is given the passband it was measured through - or the one it was
calibrated onto, ZTF's g and r being on the Pan-STARRS scale - and the system
its magnitudes are on, so that a blackbody can be put through it and the
extinction curve averaged over it. Where a survey's band has no curve in the
library, the nearest standard one stands in, and is said to; where nothing
does - unfiltered CCDs, narrow bands, the broad bands of KELT or WASP - there
is no reference, and the viewer says so rather than drawing a wrong one.
"""

import functools

import numpy as np

from . import filters


# Renamed in numpy 2
_integrate = getattr(np, 'trapezoid', None) or np.trapz

# Blackbody temperatures the tracks are tabulated at, in K
COLOUR_TEMPERATURES = np.geomspace(2000.0, 50000.0, 80)

# Wavelengths the blackbody is computed at, in Angstrom - past both ends of
# every band a light curve here is measured in
_WAVE = np.geomspace(900.0, 400000.0, 6000)

# The extinction law, as the SED fit has it
COLOUR_RV = 3.1

# Fitzpatrick (1999) is defined to 6 microns; beyond, the extinction of WISE
# W3 and W4 is taken as none, which against the optical is near enough
_EXTINCTION_MAX_AA = 60000.0

# ATLAS's two bands, which no library curve here has, as top hats over the
# ranges Tonry et al. (2018, PASP 130, 064505) give for them
ATLAS_BANDS = {
    'ATLAS_c': (4200.0, 6500.0),
    'ATLAS_o': (5600.0, 8200.0),
}

# Bands whose magnitudes are AB, beyond those filters.is_ab() knows of
_AB_EXTRA = ('ZTF_', 'ATLAS_')

# The passband of each band of each survey, as (curve, approximate): the
# curve its magnitudes are on, and whether that is only the nearest standard
# band to the one actually used. A band not listed has none.
_JOHNSON = {'U': 'GROUND_JOHNSON_U', 'B': 'GROUND_JOHNSON_B', 'V': 'GROUND_JOHNSON_V',
            'R': 'GROUND_COUSINS_R', 'Rc': 'GROUND_COUSINS_R',
            'I': 'GROUND_COUSINS_I', 'Ic': 'GROUND_COUSINS_I'}
_PS1 = {b: f'PS1_{b}' for b in 'grizy'}

LC_PASSBANDS = {
    'ztf': {'g': ('PS1_g', False), 'r': ('PS1_r', False), 'i': ('ZTF_i', False)},
    'ptf': {'g': ('SDSS_g', True)},
    'bgds': {b: (_JOHNSON[b], False) for b in 'UBV'}
            | {b: (f'SDSS_{b}', False) for b in 'riz'},
    'atlas': {'c': ('ATLAS_c', True), 'o': ('ATLAS_o', True)},
    'asas': {'V': (_JOHNSON['V'], False), 'g': ('SDSS_g', True)},
    'asas3': {'V': (_JOHNSON['V'], False)},
    'kws': {b: (_JOHNSON[b], False) for b in ('B', 'V', 'Ic')},
    'aavso': {b: (_JOHNSON[b], False) for b in 'UBVRI'}
             | {f'S{b.upper()}': (f'SDSS_{b}', False) for b in 'griz'}
             | {b: (f'2MASS_{b}', True) for b in 'JH'} | {'K': ('2MASS_Ks', True)},
    'omc': {'V': (_JOHNSON['V'], False)},
    'ogle': {'V': (_JOHNSON['V'], False), 'I': (_JOHNSON['I'], False)},
    'vista': {'J': ('2MASS_J', True), 'H': ('2MASS_H', True), 'Ks': ('2MASS_Ks', True)},
    'nsc': {b: (f'PS1_{b}', True) for b in 'griz'},
    'skymapper': {b: (f'SkyMapper_{b}', False) for b in 'uvgriz'},
    'pgir': {'J': ('2MASS_J', True)},
    'dasch': {'phot': (_JOHNSON['B'], True)},
    'applause': {'RP': ('Gaia_RP', True)},
    'mmt9': {'V': (_JOHNSON['V'], True)},
    'fram': {b: (_JOHNSON[b], False) for b in 'BVRI'} | {'z': ('SDSS_z', True)},
    'wise': {f'W{k}': (f'WISE_RSR_W{k}', False) for k in range(1, 5)},
    'ps1': {b: (_PS1[b], False) for b in 'grizy'},
    'gaia': {b: (f'Gaia_{b}', False) for b in ('G', 'BP', 'RP')},
}

# What a band brought onto another scale through an assumed colour is on,
# by the first word of its label: 'g (conv.)' is on the common g scale
_DERIVED = {'g': ('PS1_g', False), 'r': ('PS1_r', False),
            'V': (_JOHNSON['V'], False), 'R': (_JOHNSON['R'], False)}


def lc_passband(source_id, label, kind=None):
    """The passband a series' magnitudes are on, as (curve, approximate), or None."""
    band = LC_PASSBANDS.get(source_id, {}).get(label)
    if band is None and kind == 'derived' and label:
        band = _DERIVED.get(label.split()[0])
    return band


def is_ab(name):
    return filters.is_ab(name) or name.startswith(_AB_EXTRA)


@functools.lru_cache(maxsize=None)
def _passband(name):
    """The curve as (wavelength in Angstrom, photon-counting weight), on _WAVE."""
    if name in ATLAS_BANDS:
        lo, hi = ATLAS_BANDS[name]
        trans = ((_WAVE >= lo) & (_WAVE <= hi)).astype(float)
    else:
        f = filters.get_filter(name)
        wave = np.asarray(f.wavelength.to('AA').value, dtype=float)
        trans = np.interp(_WAVE, wave, np.asarray(f.transmit, dtype=float), left=0, right=0)
    return trans * _WAVE


@functools.lru_cache(maxsize=None)
def _zero_flam(name):
    """The flux per Angstrom of a star of magnitude zero, on the band's system."""
    weight = _passband(name)
    if is_ab(name):
        # 3631 Jy as f_lambda, averaged over the band as a photon counter does
        flam = 3631e-23 * 2.99792458e18 / _WAVE ** 2
    else:
        import pyphot
        vega = pyphot.Vega()
        flam = np.interp(_WAVE, vega.wavelength.to('AA').value,
                         vega.flux.to('erg / (s cm2 AA)').value, left=0, right=0)
    return _integrate(flam * weight, _WAVE) / _integrate(weight, _WAVE)


def _planck(temperature):
    """A blackbody of unit solid angle, in erg/s/cm2/A."""
    wl = _WAVE * 1e-8
    x = 1.4387769 / (wl * temperature)
    with np.errstate(over='ignore'):
        return 1.191042972e-5 / wl ** 5 / np.expm1(x) * 1e-8


@functools.lru_cache(maxsize=None)
def band_track(name):
    """A blackbody through one band: its magnitude at each of
    COLOUR_TEMPERATURES, for a unit solid angle, and the extinction in the
    band per magnitude of A_V, which for a broad band depends a little on
    the temperature of what is reddened.
    """
    import extinction

    weight = _passband(name)
    zero = _zero_flam(name)
    norm = _integrate(weight, _WAVE)

    red = _WAVE <= _EXTINCTION_MAX_AA
    attenuation = np.zeros_like(_WAVE)
    attenuation[red] = extinction.fitzpatrick99(_WAVE[red], 1.0, COLOUR_RV)
    dimmed = 10 ** (-0.4 * attenuation)

    mags, ext = [], []
    for temperature in COLOUR_TEMPERATURES:
        flam = _planck(temperature)
        clear = _integrate(flam * weight, _WAVE) / norm
        reddened = _integrate(flam * dimmed * weight, _WAVE) / norm
        mags.append(-2.5 * np.log10(clear / zero))
        ext.append(-2.5 * np.log10(reddened / clear))

    return np.array(mags), np.array(ext)


def colour_references(series):
    """The references for the series of a colour diagram.

    `series` are (source_id, band label, kind) each. Returns the temperatures
    and, for each series, its passband with its system and whether it is only
    approximate, the blackbody's magnitude at each temperature and the
    extinction per A_V - or a passband of None, where there is none.
    """
    out = []
    for source_id, label, kind in series:
        band = lc_passband(source_id, label, kind)
        if band is None:
            out.append({'passband': None})
            continue
        name, approximate = band
        mags, ext = band_track(name)
        out.append({
            'passband': name,
            'system': 'AB' if is_ab(name) else 'Vega',
            'approximate': approximate,
            'mag': [round(float(_), 4) for _ in mags],
            'ext': [round(float(_), 4) for _ in ext],
        })
    return {'temperatures': [round(float(_), 1) for _ in COLOUR_TEMPERATURES], 'bands': out}
