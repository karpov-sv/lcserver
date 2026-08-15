"""Hipparcos epoch photometry acquisition module.

Acquires the individual transits Hipparcos recorded between 1989 and 1993 -
the earliest photometry available here, by a decade.

The Epoch Photometry Annexe is one compressed file of 127 MB, but it is
compressed in blocks and comes with an index of where each star begins, so a
single star costs a few kilobytes rather than the whole archive.
"""

import os
import zlib
import requests
import numpy as np

from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

from astroquery.vizier import Vizier

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query, log_bands,
                    log_conversion, plot_with_errors,
                    assumed_color, v_to_g, V_TO_G_FORMULA)


HIP_ANNEXE_URL = 'https://cdsarc.cds.unistra.fr/ftp/I/239/epophot/hep.gz'
HIP_INDEX_URL = HIP_ANNEXE_URL + '.idx'

# The new reduction, which is what a position should be matched against
HIP_CATALOGUE = 'I/311/hip2'

# Matching radius, in arcsec. Hipparcos positions are good to milliarcseconds,
# so this only has to cover proper motion since 1991.
HIP_SR = 10.0

# Transits are timed in HJD - 2440000
HIP_MJD0 = 2440000 - 2400000.5

# A transit was set aside by the reduction if any of these bits is raised.
# The Hipparcos catalogue defines them in a printed volume rather than in
# anything CDS carries, so the mask was recovered by asking which one makes
# the surviving count match the number of accepted transits each star's own
# header quotes. It does, exactly, for all 28 stars it was tried on.
HIP_UNUSED_MASK = 440

# Slack past the next star's offset, to be sure of catching the tail of the
# compression block this star ends in
HIP_READ_SLACK = 4096


def _find_star(ra, dec, sr, log):
    """The Hipparcos star at a position, if it has one."""
    res = Vizier(columns=['**'], row_limit=10).query_region(
        SkyCoord(ra, dec, unit='deg'), radius=sr*u.arcsec, catalog=HIP_CATALOGUE)

    if not res or not len(res[0]):
        return None

    table = res[0]
    if '_r' in table.colnames:
        table = table[np.argsort(np.asarray(table['_r'], dtype=float))]

    row = table[0]
    if len(table) > 1:
        log(f"{len(table)} Hipparcos stars within {sr:.0f} arcsec, using the closest")

    return {
        'hip': int(row['HIP']),
        'hpmag': float(row['Hpmag']) if 'Hpmag' in table.colnames else np.nan,
        'ntr': int(row['Ntr']) if 'Ntr' in table.colnames else None,
    }


def _read_index(log):
    """Where each star begins in the annexe, as HIP number to byte offset."""
    log("Fetching the index of the photometry annexe")

    res = requests.get(HIP_INDEX_URL, timeout=300)
    res.raise_for_status()

    offsets = {}
    for line in res.text.splitlines():
        if '=' not in line:
            continue
        hip, offset = line.split('=', 1)
        try:
            offsets[int(hip)] = int(offset)
        except ValueError:
            continue

    return offsets


def _fetch_record(hip, offsets, log):
    """One star's block of the annexe, fetched by byte range and unpacked.

    Returns the header line and the transit rows. The offsets are into the
    compressed file, and a star's block is bounded by wherever the next star
    starts, so only those bytes are asked for.
    """
    if hip not in offsets:
        return None, None

    start = offsets[hip]
    later = [_ for _ in offsets.values() if _ > start]
    end = min(later) if later else start + (1 << 20)

    res = requests.get(HIP_ANNEXE_URL, timeout=300,
                       headers={'Range': f'bytes={start-1}-{end + HIP_READ_SLACK}'})
    res.raise_for_status()

    log(f"Read {len(res.content)} bytes of the annexe for HIP {hip}")

    # The offsets point into a gzip stream that restarts often enough for a
    # block to be unpacked on its own
    text = zlib.decompressobj(zlib.MAX_WBITS | 16).decompress(res.content)
    lines = text.decode('latin-1').splitlines()

    if not lines:
        return None, None

    header, rows = lines[0], []

    # A blank line separates one star from the next
    for line in lines[1:]:
        if not line.strip():
            break

        parts = line.split('|')
        if len(parts) < 4:
            break

        try:
            rows.append((float(parts[0]), float(parts[1]),
                         float(parts[2]), int(parts[3])))
        except ValueError:
            break

    return header, rows


@survey_source(
    name='Hipparcos',
    short_name='Hipparcos',
    state_acquiring='acquiring Hipparcos lightcurve',
    state_acquired='Hipparcos lightcurve acquired',
    log_file='hipparcos.log',
    output_files=['hipparcos.log', 'hipparcos_lc.png', 'hipparcos.vot', 'hipparcos.txt'],
    button_text='Get Hipparcos lightcurve',
    form_fields={
        'hipparcos_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': HIP_SR,
            'required': False,
        },
        'hipparcos_all': {
            'type': 'choice',
            'label': 'Transits',
            'choices': [('accepted', 'Accepted by the reduction'),
                        ('all', 'All, including those set aside')],
            'initial': 'accepted',
            'required': False,
        },
    },
    help_text='Hipparcos epoch photometry, Hp band, 1989-1993, brighter than V~12',
    order=26,
    # Lightcurve metadata
    votable_file='hipparcos.vot',
    lc_bands=[
        surveys.band('Hp', 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value='Hp', color='#7f7f7f',
                     note='Hipparcos broad band, wider than V'),
        surveys.band('g (conv.)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     filter_column='filter', filter_value='Hp', color='#c7c7c7',
                     note='Hp taken as V - no Hp - V term is applied - and put '
                          'on the common g scale using an assumed g - r',
                     combined=True),
    ],
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color='#7f7f7f',
    lc_mode='magnitude',
    lc_short=False,
    # Template metadata. No cutout: Hipparcos published no imaging.
    template_layout='simple',
)
def target_hipparcos(config, basepath=None, verbose=True, show=False):
    """
    Get Hipparcos epoch photometry.

    Parameters
    ----------
    config : dict
        Configuration dictionary with target coordinates
    basepath : str, optional
        Base path for output files
    verbose : bool or callable, optional
        Verbose logging mode or log function
    show : bool, optional
        Show plots interactively
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('hipparcos'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    hip_sr = config.get('hipparcos_sr', HIP_SR)

    cache_name = f"hipparcos_{ra:.4f}_{dec:.4f}_{hip_sr:.1f}.vot"

    with cached_votable_query(cache_name, basepath, log, 'Hipparcos',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            log(f"within {hip_sr:.0f} arcsec")

            try:
                star = _find_star(ra, dec, hip_sr, log)

                if star is None:
                    log("Warning: No Hipparcos star at this position - the catalogue "
                        "reaches only to about twelfth magnitude")
                    return

                log(f"HIP {star['hip']} at Hp = {star['hpmag']:.2f}"
                    + (f", {star['ntr']} transits" if star['ntr'] else ''))

                offsets = _read_index(log)
                header, rows = _fetch_record(star['hip'], offsets, log)

                if not rows:
                    log(f"HIP {star['hip']} has no epoch photometry in the annexe")
                    return

                data = np.array(rows)

                # What the star's own header says about how many transits the
                # reduction kept, carried as a column rather than in the table
                # metadata, which a VOTable does not preserve - the check below
                # would otherwise quietly vanish on every cached run
                try:
                    fields = header.split('|')
                    nused = int(fields[4].split()[0])
                except (ValueError, IndexError):
                    nused = -1

                hipp = Table({
                    'mjd': data[:, 0] + HIP_MJD0,
                    'mag': data[:, 1],
                    'magerr': data[:, 2],
                    'filter': np.full(len(data), 'Hp', dtype='<U2'),
                    'flags': data[:, 3].astype(int),
                    'hip': np.full(len(data), star['hip']),
                    'nused': np.full(len(data), nused),
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise SourceError("could not download the data - "
                                  f"{type(e).__name__}: {e}")

            if not len(hipp):
                log("Warning: No Hipparcos data points found")
                return

            cache.save(hipp)

        hipp = cache.data

    log(f"{len(hipp)} transits")

    # Transits the reduction set aside, which it counts in the star's header
    flags = np.asarray(hipp['flags'], dtype=int)
    accepted = (flags & HIP_UNUSED_MASK) == 0

    nused = int(hipp['nused'][0]) if 'nused' in hipp.colnames and len(hipp) else -1
    if nused >= 0:
        log(f"  {int(np.sum(accepted))} accepted, and the star's header says {nused}"
            + ("" if int(np.sum(accepted)) == nused else " - which does not agree"))

    if str(config.get('hipparcos_all', 'accepted')) == 'all':
        log("  keeping every transit, including those the reduction set aside")
    else:
        hipp = hipp[accepted]

    idx = np.isfinite(hipp['mjd']) & np.isfinite(hipp['mag']) & np.isfinite(hipp['magerr'])
    idx &= hipp['magerr'] > 0

    log(f"{int(np.sum(idx))} data points after filtering")

    hipp = hipp[idx]
    hipp.sort('mjd')

    if not len(hipp):
        log("Warning: No valid Hipparcos data points")
        return

    log_conversion(
        log, 'Hipparcos',
        'no conversion applied - the band is published as measured',
        {'colour term': ('none', 'Hp is a broad band of its own, wider than V'),
         'timestamps': 'HJD - 2440000 in the annexe, converted to MJD'},
        npoints=len(hipp),
    )

    # Onto the common g scale, by treating Hp as V and converting from there.
    # The Hipparcos zero point was set so that Hp = V at B - V = 0, which is
    # what makes this defensible at all; away from that colour Hp and V part
    # company, and the published relation between them is a table interpolated
    # in colours we do not have (Bessell 2000), not something to apply here.
    # So the Hp - V term is simply not taken out, and this band is the roughest
    # on the combined curve.
    g_minus_r, g_minus_r_origin = assumed_color(config, 'g_minus_r')
    hipp['mag_g'] = v_to_g(np.asarray(hipp['mag'], dtype=float), g_minus_r)

    log_conversion(
        log, 'Hipparcos',
        'V = Hp   (no colour term),  then  ' + V_TO_G_FORMULA,
        {'(g - r)': (g_minus_r, g_minus_r_origin),
         'Hp - V': ('not corrected for',
                    'zero point set so that Hp = V at B - V = 0')},
        npoints=len(hipp),
        note='the Hp - V term is left in, so a red star lands further off the '
             'common scale than the other converted bands do',
    )

    log_bands(log, 'Hipparcos', [
        {'label': 'Hp', 'kind': 'native', 'npoints': len(hipp),
         'note': 'individual transits, as reported'},
        {'label': 'g (conv.)', 'kind': 'derived', 'npoints': len(hipp),
         'note': 'Hp taken as V and converted, without an Hp - V term'},
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'hipparcos_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        time = Time(np.asarray(hipp['mjd'], dtype=float), format='mjd')
        plot_with_errors(ax, time.datetime, hipp['mag'], hipp['magerr'],
                         color='#7f7f7f')

        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        ax.set_ylabel('Hp')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - Hipparcos")

    log("Hipparcos lightcurve plot saved to file:hipparcos_lc.png")

    hipp.write(os.path.join(basepath, 'hipparcos.vot'), format='votable', overwrite=True)
    hipp.write(os.path.join(basepath, 'hipparcos.txt'), format='ascii.commented_header', overwrite=True)
    log("Hipparcos data written to file:hipparcos.vot")
    log("Hipparcos data written to file:hipparcos.txt")
