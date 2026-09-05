"""What the model grids say, in the quantities a reader recognises them by.

A cube holds a surface flux per filter on a lattice of (Teff, log g, [Fe/H]),
which is what a fit needs and nothing a person reads. What a person reads is a
colour, a bolometric correction, a track of one parameter against another - all
of them the same numbers put through the arithmetic below, and all of them
cheap enough to do for a whole grid at once: a cube is a few megabytes and the
conversion is a division.

Two normalisations make the rest possible.

The magnitudes here are those of a star of one solar radius seen from ten
parsecs. A grid says nothing about radius - three parameters go in and none of
them is one - so an absolute magnitude cannot be had from it alone. What can is
exact: a star of radius R is 5 log10(R/Rsun) brighter than this, so one number
subtracted afterwards moves every point of a plot to whatever radius is wanted,
and a colour does not move at all.

Extinction is the same shape. The fit reddens a band by 10^(-0.4 Av k_b), with
k_b the attenuation at the band's pivot for Av = 1, so in magnitudes a band
moves by exactly Av k_b - linear, with a slope that depends on the band and not
on the model. Both quantities are returned beside the values they apply to, so
that a page can move a plot in radius or in reddening without asking again.
"""

import functools

import numpy as np

import h5py

from .filters import (FILTER_NAMES, has_zero_point, is_ab, pivot_aa,
                      vega_zero_jy)
from .sedfit import (AB_ZERO_JY, C_AA, PARSEC, R_SUN, attenuation,
                     grid_registry)
from .utils import SourceError


# The irradiance a bolometric magnitude of zero means, erg/s/cm2. IAU 2015 B2,
# which fixes the scale absolutely rather than through an adopted solar value
BOLOMETRIC_ZERO = 2.518021002e-5

# erg/s/cm2/K^4, for the bolometric flux a model's effective temperature is
# the definition of
STEFAN_BOLTZMANN = 5.670374419e-5

# How far the star of these magnitudes is, and how big
DISTANCE_PC = 10.0
RADIUS_SUN = 1.0

# The dilution those two amount to, applied to every surface flux here
DILUTION = (RADIUS_SUN * R_SUN / (DISTANCE_PC * PARSEC)) ** 2


@functools.lru_cache(maxsize=1)
def bands():
    """Every band a cube may hold, blue to red, with what converts it.

    The pivot wavelength and zero point are the ones the fit puts a measured
    magnitude through, so a synthetic magnitude here and an observed one there
    mean the same thing. `ext` is the band's attenuation at Av = 1: the slope
    a page reddens with, and the same number the likelihood is evaluated with.

    A band is `usable` when a magnitude can be formed in it at all, which
    filters.has_zero_point decides and two submillimetre bands fail: their
    curves lie past the end of the reference Vega spectrum, so there is no
    zero point to read a magnitude against. A grid may hold a perfectly good
    flux there - BT-Settl does, for thirteen thousand models - and no
    magnitude can be made of it.
    """
    out = []

    for band in FILTER_NAMES:
        try:
            pivot = pivot_aa(band)
        except Exception:
            # A band whose curve is not installed is one nothing can be said
            # about, rather than one to fail the whole page over
            continue

        zero = AB_ZERO_JY if is_ab(band) else vega_zero_jy(band)

        out.append({
            'band': band,
            'wavelength': pivot,
            'system': 'AB' if is_ab(band) else 'Vega',
            'zero_jy': zero,
            'usable': has_zero_point(band),
            'ext': float(attenuation(np.array([pivot * 1e-4]), 1.0)[0]),
        })

    return sorted(out, key=lambda b: b['wavelength'])


def _rows(path):
    """One cube as a row per model, whichever way the file is laid out.

    A scattered grid is already rows. A lattice is a block over three axes and
    is spread out here, which costs the memory of the block and makes every
    plot below one expression rather than two.
    """
    with h5py.File(path, 'r') as h:
        layout = str(h.attrs.get('layout', 'lattice'))
        logg = np.asarray(h['logg'][:], dtype=float)
        teff = np.asarray(h['teff'][:], dtype=float)
        feh = np.asarray(h['feh'][:], dtype=float)
        flux = np.asarray(h['flux'][:], dtype=float)
        names = [b.decode() if isinstance(b, bytes) else str(b)
                 for b in h['filters'][:]]

    if layout != 'scattered':
        # The axis order the cube was written in, and the one the interpolator
        # reads it back in
        gg, tt, zz = np.meshgrid(logg, teff, feh, indexing='ij')
        logg, teff, feh = gg.ravel(), tt.ravel(), zz.ravel()
        flux = flux.reshape(-1, flux.shape[-1])

    return layout, teff, logg, feh, flux, names


@functools.lru_cache(maxsize=6)
def photometry(name):
    """Every model of one grid as magnitudes, at one solar radius and ten parsecs.

    Bands the grid does not reach come back as NaN, which is what they are:
    the cube carries no flux there and no magnitude can be invented for it.
    """
    entry = grid_registry().get(str(name).lower())
    if not entry:
        raise SourceError(f'no grid called {name}')

    layout, teff, logg, feh, flux, names = _rows(entry['path'])

    table = {b['band']: b for b in bands()}
    keep = [i for i, b in enumerate(names) if b in table]
    names = [names[i] for i in keep]
    flux = flux[:, keep]

    pivot = np.array([table[b]['wavelength'] for b in names])
    zero = np.array([table[b]['zero_jy'] if table[b]['usable'] else 0.0
                     for b in names])

    # Surface f_lambda per micron, to f_lambda per Angstrom at ten parsecs,
    # to f_nu in Jansky, to a magnitude on the band's own zero point
    fnu = flux * 1e-4 * DILUTION * pivot ** 2 / C_AA / 1e-23

    with np.errstate(divide='ignore', invalid='ignore'):
        mag = np.where((fnu > 0) & (zero > 0), -2.5 * np.log10(fnu / zero),
                       np.nan)

    # The effective temperature is the definition of the bolometric flux, so
    # this needs no integration and no cube column
    mbol = -2.5 * np.log10(STEFAN_BOLTZMANN * teff ** 4 * DILUTION
                           / BOLOMETRIC_ZERO)

    return {
        'name': entry['name'],
        'layout': layout,
        'axis_logg': entry.get('axis_logg') or 'logg',
        'teff': teff, 'logg': logg, 'feh': feh,
        'mag': mag.astype(np.float32),
        'mbol': mbol,
        'bands': names,
        'column': {b: i for i, b in enumerate(names)},
        'covers': [b for i, b in enumerate(names)
                   if np.isfinite(mag[:, i]).any()],
    }


# What an axis may be asked for, and what it means. A page names one of these
# and the arithmetic is here rather than in the page.
AXES = {
    'teff': 'effective temperature, K',
    'logg': 'surface gravity, log g',
    'feh': 'metallicity, [Fe/H]',
    'mbol': 'bolometric magnitude at 1 Rsun, 10 pc',
    'mag': 'absolute magnitude at 1 Rsun, 10 pc',
    'color': 'colour',
    'bc': 'bolometric correction',
}


def parse_axis(spec):
    """An axis as (kind, arguments), from the short form a page sends.

    'teff', 'mbol', 'mag:PS1_g', 'color:PS1_g-PS1_r', 'bc:GROUND_JOHNSON_V'.
    """
    spec = str(spec or '').strip()
    kind, _, rest = spec.partition(':')

    if kind not in AXES:
        raise SourceError(f'no axis called {spec}')

    if kind in ('teff', 'logg', 'feh', 'mbol'):
        return kind, ()

    if kind == 'color':
        first, _, second = rest.partition('-')
        if not first or not second:
            raise SourceError(f'{spec} is not a pair of bands')
        return kind, (first.strip(), second.strip())

    if not rest:
        raise SourceError(f'{spec} names no band')

    return kind, (rest.strip(),)


def axis(table, spec):
    """One axis of a plot: its values, and how reddening and radius move them.

    `slope` is what a magnitude of Av adds, and `radius` whether 5 log10(R/Rsun)
    should be taken off for a star that is not one solar radius. A colour has
    a slope and no radius term, a temperature has neither, and a bolometric
    correction has both - it is a magnitude subtracted from one that reddening
    and radius do not touch, so it carries their sign reversed.
    """
    kind, args = parse_axis(spec)

    if kind in ('teff', 'logg', 'feh', 'mbol'):
        return {'values': table[kind], 'slope': 0.0, 'radius': kind == 'mbol',
                'label': _label(kind, args, table)}

    known = {b['band']: b for b in bands()}

    for band in args:
        if band not in table['column']:
            raise SourceError(f'{table["name"]} has no {band}')

    columns = [table['column'][b] for b in args]

    if kind == 'mag':
        values = table['mag'][:, columns[0]]
        slope = known[args[0]]['ext']
        radius = True
    elif kind == 'color':
        values = table['mag'][:, columns[0]] - table['mag'][:, columns[1]]
        slope = known[args[0]]['ext'] - known[args[1]]['ext']
        radius = False
    else:
        # BC = Mbol - Mband, and Mbol is neither reddened nor a function of
        # the band, so everything the magnitude does the correction undoes
        values = table['mbol'] - table['mag'][:, columns[0]]
        slope = -known[args[0]]['ext']
        radius = False

    return {'values': np.asarray(values, dtype=float), 'slope': float(slope),
            'radius': bool(radius), 'label': _label(kind, args, table)}


def _label(kind, args, table):
    """How the axis is written on the plot."""
    if kind == 'teff':
        return 'Teff, K'
    if kind == 'logg':
        return 'log Rt' if table['axis_logg'] != 'logg' else 'log g'
    if kind == 'feh':
        return '[Fe/H]'
    if kind == 'mbol':
        return 'M_bol'
    if kind == 'mag':
        return 'M(%s)' % args[0]
    if kind == 'color':
        return '%s - %s' % args
    return 'BC(%s)' % args[0]


def coverage():
    """Which of the bands each offered grid actually predicts.

    Not a detail: a fit on a grid that stops at three microns simply drops the
    points beyond it, and the difference between a grid covering ninety-three
    bands and one covering fifty-nine is the difference between a mid-infrared
    excess measured and one never looked for.
    """
    from .sedfit import offered_grids

    names = [b['band'] for b in bands()]
    usable = {b['band']: b['usable'] for b in bands()}
    grids = []

    for entry in offered_grids():
        try:
            table = photometry(entry['name'])
        except Exception:
            continue

        covers = set(table['covers'])

        # Three states rather than two: a grid may predict a band, may have no
        # flux for it, or may have one that no magnitude can be made of
        grids.append({
            'name': entry['name'],
            'label': entry['label'],
            'covered': [2 if b in covers else (0 if usable[b] else 1)
                        for b in names],
            'count': len(covers),
        })

    return {'bands': [dict(b) for b in bands()], 'grids': grids}


def support(name):
    """How the models of one grid are laid out, and where they are not.

    A scattered grid is interpolated over a triangulation of the models it
    has, and a hull is convex where a set of computed models is not: past the
    temperatures a given gravity was computed at, a simplex carries on across
    a gap. Grid._compactness marks those, and the fraction of them is worth
    knowing before trusting a fit that landed there.
    """
    from .sedfit import load_grid

    table = photometry(name)
    grid = load_grid(name)

    bridging = None
    if getattr(grid, '_compact', None) is not None:
        bridging = float(1.0 - np.mean(grid._compact))

    # What the grid is actually a function of. A scattered grid says so itself,
    # having been triangulated in those directions and no others; a lattice is
    # asked, since one of its axes may hold a single value - Koester's does,
    # and reporting it as varying in metallicity would be a plain untruth
    varies = getattr(grid, '_varies', None)
    if not varies:
        varies = [p for p in ('logg', 'teff', 'feh')
                  if np.ptp(np.unique(table[p])) > 0]

    return {
        'name': table['name'],
        'layout': table['layout'],
        'models': int(len(table['teff'])),
        'axis_logg': table['axis_logg'],
        'varies': list(varies),
        'bridging': bridging,
        'covers': len(table['covers']),
        'teff': [float(table['teff'].min()), float(table['teff'].max())],
        'logg': [float(table['logg'].min()), float(table['logg'].max())],
        'feh': [float(table['feh'].min()), float(table['feh'].max())],
    }


# How many points a spectrum is reduced to before it is sent to a browser.
# The models run to twenty thousand samples over six decades of wavelength and
# a plot has a thousand pixels; the reduction is an integration over each bin
# rather than a stride, so a line that falls between two samples still counts.
SPECTRUM_BINS = 2000


@functools.lru_cache(maxsize=6)
def spectrum_nodes(name):
    """The parameters of every spectrum one grid has, or None if it has none.

    A grid's spectra are a coarser set than its cube - what is published at
    full wavelength is usually a subset of what was computed - so this is read
    from the spectra file itself rather than assumed from the cube.
    """
    from .sedfit import spectra_source

    source = spectra_source(name)
    if source is None:
        return None

    path, group = source

    with h5py.File(path, 'r') as h:
        if group is not None and group not in h:
            return None

        node = h[group] if group is not None else h

        return {
            'teff': np.asarray(node['teff'][:], dtype=float),
            'logg': np.asarray(node['logg'][:], dtype=float),
            'feh': np.asarray(node['z'][:], dtype=float),
        }


def spectra(name, indices, bins=SPECTRUM_BINS):
    """Those spectra, binned onto one logarithmic wavelength grid.

    Surface flux as the file holds it, erg/s/cm2/um, and not diluted or
    reddened: what is wanted here is the shape a family of models has, and
    every normalisation a reader might want is a scalar the page can apply
    without asking again.
    """
    from .sedfit import spectra_source, rebin

    source = spectra_source(name)
    if source is None:
        raise SourceError(f'{name} has no spectra')

    path, group = source
    out = []

    with h5py.File(path, 'r') as h:
        node = h[group] if group is not None else h

        wave = np.asarray(node['wavelength'][:], dtype=float)
        teff = np.asarray(node['teff'][:], dtype=float)
        logg = np.asarray(node['logg'][:], dtype=float)
        feh = np.asarray(node['feh' if 'feh' in node else 'z'][:], dtype=float)

        good = wave > 0
        wave = wave[good]
        edges = np.geomspace(wave.min(), wave.max(), bins + 1)
        centres = np.sqrt(edges[1:] * edges[:-1])

        for i in indices:
            i = int(i)
            flux = np.asarray(node['flux'][i], dtype=float)[good]
            out.append({
                'teff': float(teff[i]), 'logg': float(logg[i]),
                'feh': float(feh[i]),
                'flux': rebin(wave, flux, edges),
            })

    return centres, out


def spectrum_family(name, along='teff', at=None, count=8):
    """Which nodes to draw when one parameter is stepped and the others held.

    The models are at whatever parameters they were computed at, so 'hold log g
    at 4.5' means the nearest gravity the grid has rather than that number. The
    nearest is found first, and the family is what sits at it - which is how a
    row of temperatures at one gravity comes out as a sequence rather than as a
    scatter of whatever happened to be closest.
    """
    nodes = spectrum_nodes(name)
    if nodes is None:
        raise SourceError(f'{name} has no spectra')

    at = dict(at or {})
    held = [p for p in ('teff', 'logg', 'feh') if p != along]

    mask = np.ones(len(nodes[along]), dtype=bool)
    fixed = {}

    for parameter in held:
        values = np.unique(nodes[parameter][mask])
        if not len(values):
            continue

        want = at.get(parameter)
        pick = (values[int(np.argmin(np.abs(values - float(want))))]
                if want is not None else values[len(values) // 2])

        fixed[parameter] = float(pick)
        mask &= nodes[parameter] == pick

    index = np.flatnonzero(mask)
    if not len(index):
        return [], fixed

    # Evenly through what is there, rather than the first few: a family of
    # eight is meant to span the range the grid covers
    order = index[np.argsort(nodes[along][index])]
    if len(order) > count:
        order = order[np.linspace(0, len(order) - 1, count).round().astype(int)]

    return [int(i) for i in order], fixed
