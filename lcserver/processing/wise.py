"""WISE / NEOWISE epoch photometry acquisition module.

Acquires the single-exposure infrared photometry that IRSA publishes for the
WISE mission and its NEOWISE reactivation.
"""

import os
import numpy as np

from astropy.table import Table, vstack
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

from astroquery.ipac.irsa import Irsa

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import cleanup_paths, cached_votable_query, log_bands, log_conversion


# The two IRSA tables that between them cover the whole mission, and the bands
# each of them carries. The cryogenic survey measured all four bands for a few
# months in 2010; the reactivation has been revisiting the sky in W1 and W2
# ever since, with the magnitudes under plainer column names.
WISE_PHASES = [
    {
        'catalog': 'allwise_p3as_mep',
        'name': 'AllWISE Multiepoch',
        'phase': 'cryo',
        'bands': ['W1', 'W2', 'W3', 'W4'],
        'mag': '{b}mpro_ep',
        'err': '{b}sigmpro_ep',
    },
    {
        'catalog': 'neowiser_p1bs_psd',
        'name': 'NEOWISE-R Single Exposure',
        'phase': 'neowise',
        'bands': ['W1', 'W2'],
        'mag': '{b}mpro',
        'err': '{b}sigmpro',
    },
]

# Default matching radius, in arcsec. WISE resolves some 6 arcsec, so a wider
# cone would start collecting the neighbours rather than the star.
WISE_SR = 3.0


def _flag_char(column, i):
    """The i-th character of a per-band flag string, for every row.

    The flags carry one character per band, but only as many as the table has:
    four for the cryogenic survey, two for the reactivation. Reading them per
    band rather than comparing the whole string keeps W1 when it is only W2
    that is contaminated - and comparing against '0000' would reject every
    NEOWISE row outright, its flags being two characters long.
    """
    values = np.asarray(column).astype(str)
    return np.array([v[i] if len(v) > i else '0' for v in values])


def _frame_is_usable(table):
    """Frame-level quality, for the columns the table happens to carry."""
    ok = np.ones(len(table), dtype=bool)

    for name in ('qual_frame', 'qi_fact', 'saa_sep'):
        if name in table.colnames:
            ok &= np.asarray(table[name], dtype=float) > 0

    return ok


@survey_source(
    name='WISE',
    short_name='WISE',
    state_acquiring='acquiring WISE lightcurve',
    state_acquired='WISE lightcurve acquired',
    log_file='wise.log',
    output_files=['wise.log', 'wise_lc.png', 'wise.vot', 'wise.txt'],
    button_text='Get WISE lightcurve',
    form_fields={
        'wise_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': WISE_SR,
            'required': False,
        }
    },
    help_text='WISE and NEOWISE infrared epoch photometry',
    order=70,
    # Lightcurve metadata
    votable_file='wise.vot',
    lc_bands=[
        surveys.band(band, 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value=band, color=color,
                     note='WISE single-exposure photometry, as reported')
        for band, color in [('W1', '#d62728'), ('W2', '#8c564b'),
                            ('W3', '#9467bd'), ('W4', '#7f7f7f')]
    ],
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color='#d62728',
    lc_mode='magnitude',
    lc_short=False,
    # Template metadata
    template_layout='with_cutout',
    show_cutout=True,
    cutout_hips='CDS/P/allWISE/color',
    cutout_fov=0.05,
)
def target_wise(config, basepath=None, verbose=True, show=False):
    """
    Get WISE / NEOWISE epoch photometry.

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
    cleanup_paths(get_output_files('wise'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')
    wise_sr = config.get('wise_sr', WISE_SR)
    coords = SkyCoord(ra, dec, unit='deg')

    rows = []

    for phase in WISE_PHASES:
        cache_name = f"wise_{phase['phase']}_{ra:.4f}_{dec:.4f}_{wise_sr:.1f}.vot"

        log(f"\n---- {phase['name']} ----\n")

        with cached_votable_query(cache_name, basepath, log, phase['name'],
                                  refresh=refresh_cache) as cache:
            if not cache.hit:
                log(f"within {wise_sr:.1f} arcsec")
                try:
                    data = Irsa.query_region(coords, catalog=phase['catalog'],
                                             spatial='Cone', radius=wise_sr*u.arcsec)
                except:
                    import traceback
                    traceback.print_exc()
                    log(f"Error: could not query {phase['name']}")
                    data = None

                if data is not None and len(data):
                    cache.save(data)
                else:
                    data = None
            else:
                data = cache.data

        if data is None or not len(data):
            log("Warning: No data")
            continue

        mjd = np.asarray(data['mjd'], dtype=float)
        frame = _frame_is_usable(data) & np.isfinite(mjd)

        log(f"{len(data)} exposures, {int(np.sum(~frame))} dropped on frame quality")

        for i, band in enumerate(phase['bands']):
            magcol = phase['mag'].format(b=band.lower())
            errcol = phase['err'].format(b=band.lower())

            if magcol not in data.colnames or errcol not in data.colnames:
                continue

            mag = np.asarray(data[magcol], dtype=float)
            err = np.asarray(data[errcol], dtype=float)

            # A missing uncertainty means an upper limit rather than a detection
            idx = frame & np.isfinite(mag) & np.isfinite(err)

            # Contamination and moon glare are flagged per band
            if 'cc_flags' in data.colnames:
                idx &= _flag_char(data['cc_flags'], i) == '0'
            if 'moon_masked' in data.colnames:
                idx &= _flag_char(data['moon_masked'], i) == '0'

            log(f"  {band}: {int(np.sum(idx))} of {int(np.sum(frame))} usable")

            if not np.any(idx):
                continue

            rows.append(Table({
                'mjd': mjd[idx],
                'mag': mag[idx],
                'magerr': err[idx],
                'filter': np.full(int(np.sum(idx)), band, dtype='<U2'),
                'phase': np.full(int(np.sum(idx)), phase['phase'], dtype='<U8'),
            }))

    if not rows:
        log("\nWarning: No usable WISE photometry")
        return

    wise = vstack(rows)
    wise.sort('mjd')

    log_conversion(
        log, 'WISE',
        'no conversion applied - each band is published as measured',
        {'colour term': ('none', 'profile-fit magnitudes on the WISE Vega scale'),
         'phases': ('cryogenic and reactivation combined',
                    'kept apart in the phase column, as the two reductions '
                    'differ slightly')},
        npoints=len(wise),
    )

    log_bands(log, 'WISE', [
        {'label': band, 'kind': 'native',
         'npoints': int(np.sum(wise['filter'] == band)),
         'note': 'single-exposure photometry, as reported'}
        for band in ('W1', 'W2', 'W3', 'W4')
        if np.any(wise['filter'] == band)
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'wise_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        time = Time(np.asarray(wise['mjd'], dtype=float), format='mjd')

        for band, color in [('W1', '#d62728'), ('W2', '#8c564b'),
                            ('W3', '#9467bd'), ('W4', '#7f7f7f')]:
            idx = np.asarray(wise['filter']).astype(str) == band
            if not np.any(idx):
                continue

            ax.errorbar(time[idx].datetime, wise['mag'][idx], wise['magerr'][idx],
                        fmt='.', color=color, alpha=0.5, label=band)

        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        ax.legend()
        ax.set_ylabel('WISE magnitude')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - WISE")

    log("WISE lightcurve plot saved to file:wise_lc.png")

    wise.write(os.path.join(basepath, 'wise.vot'), format='votable', overwrite=True)
    wise.write(os.path.join(basepath, 'wise.txt'), format='ascii.commented_header', overwrite=True)
    log("WISE data written to file:wise.vot")
    log("WISE data written to file:wise.txt")
