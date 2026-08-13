"""Kepler lightcurve acquisition module.

Covers both phases of the mission. Kepler stared at one field in Cygnus from
2009 to 2013 and divides its data into quarters; after the second reaction
wheel failed it flew on along the ecliptic as K2 until 2018, in campaigns of
some eighty days. It is one telescope and one photometric system throughout, so
it is one source here, and a target that was watched in both phases gets both.

Two things differ from TESS and are easy to get wrong. Timestamps are BKJD,
counted from 2454833, where TESS uses BTJD from 2457000 - and lightkurve offers
a .btjd on every Time object regardless of the mission, so reading it here would
silently shift everything by 2167 days. And the official K2 products carry
AUTHOR = 'Kepler' in their headers, so what to call a product has to come from
the search rather than from the file.
"""

import os
import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

import lightkurve as lk

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import (cleanup_paths, drop_mast_downloads, mast_download_dir,
                    mission_quality_mask, log_quality, quality_field,
                    quality_level,
                    log_bands, log_conversion,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


# Days between BKJD and BJD, as the mission counts time in both phases
KEPLER_BJD0 = 2454833

# Anything shorter than this is the short cadence, in seconds. The long cadence
# is just under 1800 s and the short one 60 s, so the boundary is not delicate.
KEPLER_SHORT_CADENCE = 120

# The two phases of the mission, in the order they were flown. Each names its
# observing segments differently, and each has its own pipelines - the mission's
# own reduction first, then the ones that recovered what it did not target, or
# corrected the drift K2 flew with.
KEPLER_PHASES = [
    {
        'mission': 'Kepler',
        'segment_name': 'Quarter',
        'prefix': 'Q',
        'authors': ['Kepler', 'KBONUS-BKG'],
        'color': '#2980b9',
    },
    {
        'mission': 'K2',
        'segment_name': 'Campaign',
        'prefix': 'C',
        'authors': ['K2', 'K2SFF', 'EVEREST', 'K2VARCAT'],
        'color': '#8e44ad',
    },
]


def _segments(res):
    """Which observing segment each product belongs to.

    The archive spells this as 'Kepler Quarter 02' or 'K2 Campaign 01' in the
    mission column. A product stitched from the whole mission leaves the number
    off, and gets -1 here.
    """
    out = []

    for mission in res.table['mission']:
        last = str(mission).strip().split()[-1] if str(mission).strip() else ''
        out.append(int(last) if last.isdigit() else -1)

    return np.array(out)


def _acquire_phase(config, basepath, log, show, phase, sr, cadence, wanted_author):
    """Fetch one phase of the mission, a file per observing segment.

    Returns the number of segments written and of points in them.
    """
    mission = phase['mission']
    segment_name = phase['segment_name']
    authors = phase['authors']

    log(f"\n==== {mission} ====\n")

    # A pipeline asked for by name belongs to one phase or the other, so the
    # phase that does not have it has nothing to do
    if wanted_author != 'auto':
        if wanted_author not in authors:
            log(f"{wanted_author} is not a {mission} pipeline, skipping this phase")
            return 0, 0

        authors = [wanted_author]

    res = lk.search_lightcurve(
        SkyCoord(config.get('target_ra'), config.get('target_dec'), unit='deg'),
        radius=sr*u.arcsec, mission=mission)

    if not len(res):
        log(f"Warning: No {mission} data found")
        return 0, 0

    log(f"{len(res)} data products found")

    exptime = np.asarray(res.table['exptime'], dtype=float)
    is_short = exptime < KEPLER_SHORT_CADENCE

    if cadence == 'long':
        wanted = ~is_short
    elif cadence == 'short':
        wanted = is_short
    else:
        wanted = np.ones(len(res), dtype=bool)

    if not np.sum(wanted):
        log(f"Warning: No {cadence} cadence data among the {mission} products found")
        return 0, 0

    segments = _segments(res)
    authorcol = np.asarray(res.table['author'], dtype=str)

    # A product with no segment of its own is stitched from the whole phase, so
    # it does not belong to any one file here
    stitched = wanted & (segments < 0)
    if np.sum(stitched):
        log(f"Skipping {int(np.sum(stitched))} product(s) covering the whole "
            f"mission at once: {', '.join(sorted(set(authorcol[stitched])))}")

    nwritten = 0
    npoints = 0

    for segment in sorted(set(segments[wanted & (segments >= 0)].tolist())):
        insegment = wanted & (segments == segment)

        # The archive quotes the same mission-wide t_min and t_max on every
        # product, so the dates are taken from the data itself below
        log(f"\n{segment_name} {segment}:")

        for prod in res.table[insegment]:
            log(f"    {prod['author']:10s} {prod['exptime']:.0f} s exp")

        # One representative lightcurve per segment, from the most preferred
        # pipeline that has one
        for author in authors:
            idx = insegment & (authorcol == author)

            if not np.sum(idx):
                continue

            row = res[idx][0]
            lc = row.download(download_dir=mast_download_dir(basepath, 'kepler'))

            if not lc:
                continue

            exp = float(np.asarray(row.table['exptime'])[0])

            # The segment carries a letter for the phase it belongs to, as the
            # two number their segments from one independently
            token = f"{phase['prefix']}{segment:02d}"
            lcname = f"kepler_lc_{token}_{author}_{exp:.0f}.png"

            with plots.figure_saver(os.path.join(basepath, lcname),
                                    figsize=(8, 4), show=show) as fig:
                ax = fig.add_subplot(1, 1, 1)

                # bkjd, not btjd - lightkurve would hand over either
                time = lc.time.bkjd
                flux = lc.normalize().flux
                # Not every pipeline reports one - K2SFF publishes eight
                # columns and no quality among them
                if 'quality' in lc.colnames:
                    keep = mission_quality_mask(lc['quality'], 'Kepler',
                                                quality_level(config, 'kepler'))
                    # What the mask costs, rather than what it marks: the
                    # pipeline may have emptied those cadences already
                    had = np.isfinite(flux)
                    flux[~keep] = np.nan
                    log_quality(log, lc['quality'], had & ~keep, 'Kepler')

                ax.axhline(1, ls='--', color='gray', alpha=0.3)
                ax.plot(time, flux, drawstyle='steps', lw=1, color=phase['color'])

                ax.grid(alpha=0.2)

                ax.set_ylabel('Normalized ' + lc.meta.get('FLUX_ORIGIN', 'flux'))
                ax.set_xlabel(f'Time - {KEPLER_BJD0}, BKJD days')
                ax.set_title(f"{config['target_name']} - {mission} "
                             f"{segment_name} {segment} - {author} - {exp:.0f} s")

            # Time itself cannot be serialized, so it goes as two columns
            lc1 = lc.to_table()
            lc1['mjd'] = lc1['time'].mjd
            lc1['bkjd'] = lc1['time'].bkjd
            lc1.remove_column('time')

            mjd = np.asarray(lc1['mjd'], dtype=float)
            tmin = Time(np.nanmin(mjd), format='mjd')
            tmax = Time(np.nanmax(mjd), format='mjd')
            log(f"    {tmin.datetime.strftime('%Y-%m-%d')}"
                f" - {tmax.datetime.strftime('%Y-%m-%d')}, {len(lc1)} points")

            votname = os.path.splitext(lcname)[0] + '.vot'
            txtname = os.path.splitext(lcname)[0] + '.txt'
            lc1.write(os.path.join(basepath, votname), format='votable', overwrite=True)
            lc1.write(os.path.join(basepath, txtname), format='ascii.commented_header', overwrite=True)
            log(f"    {segment_name} lightcurve written to file:{votname}")
            log(f"    {segment_name} lightcurve written to file:{txtname}")

            nwritten += 1
            npoints += len(lc1)
            break

    return nwritten, npoints


@survey_source(
    name='Kepler',
    short_name='Kepler',
    state_acquiring='acquiring Kepler lightcurves',
    state_acquired='Kepler lightcurves acquired',
    log_file='kepler.log',
    output_files=['kepler.log', 'kepler_lc_*.vot', 'kepler_lc_*.txt', 'kepler_lc_*.png'],
    button_text='Get Kepler lightcurves',
    form_fields={
        'kepler_quality': quality_field({
            QUALITY_STANDARD: 'Drop every cadence the mission flagged',
            QUALITY_RELAXED: 'Drop only what the mission calls unusable',
            QUALITY_PUBLISHED: 'None - every cadence as published',
        }),
        'kepler_sr': {
            'type': 'float',
            'label': 'Search radius, arcsec',
            'initial': 10.0,
            'required': False,
        },
        'kepler_author': {
            'type': 'choice',
            'label': 'Pipeline',
            'choices': [('auto', 'Best available'),
                        ('Kepler', 'Kepler (official)'),
                        ('KBONUS-BKG', 'KBONUS-BKG'),
                        ('K2', 'K2 (official)'),
                        ('K2SFF', 'K2SFF - drift corrected'),
                        ('EVEREST', 'EVEREST - drift corrected'),
                        ('K2VARCAT', 'K2VARCAT')],
            'initial': 'auto',
            'required': False,
        },
        'kepler_cadence': {
            'type': 'choice',
            'label': 'Cadence',
            'choices': [('long', 'Long, 30 min'),
                        ('short', 'Short, 1 min'),
                        ('any', 'Whatever is available')],
            'initial': 'long',
            'required': False,
        },
    },
    help_text='NASA Kepler and its K2 extended mission, 2009-2018',
    order=31,
    # Lightcurve metadata
    votable_file='kepler_lc_*.vot',
    lc_flux_column='flux',
    lc_err_column='flux_err',
    lc_quality_column='quality',
    lc_color='#2980b9',
    lc_mode='flux',
    lc_short=False,
    lc_segment_name='Quarter',
    # The phase a segment belongs to, by the letter its files carry. Listed in
    # the order flown, which is the order the viewer shows them in.
    lc_segment_prefixes={'Q': 'Kepler Quarter', 'C': 'K2 Campaign'},
    lc_color_palette=['#2980b9', '#27ae60', '#16a085', '#2c3e50', '#7f8c8d'],
    # Template metadata. No cutout: the mission published no imaging service to
    # take one from, and its 4 arcsec pixels are not resolved by anything else.
    template_layout='complex',
    additional_plots=['kepler_lc_*.png'],
)
def target_kepler(config, basepath=None, verbose=True, show=False):
    """
    Get Kepler lightcurves, from both the original mission and K2.

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

    # lightkurve skips whatever it has already downloaded, for both phases at
    # once as they share this source's directory. The flag is read rather than
    # consumed - see the other sources.
    if config.get('refresh_cache', False):
        drop_mast_downloads(basepath, 'kepler', log)

    # Cleanup stale plots
    cleanup_paths(get_output_files('kepler'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    sr = config.get('kepler_sr', 10.0)
    cadence = str(config.get('kepler_cadence', 'long'))

    # Which reduction to take. The drift-corrected K2 pipelines are worth
    # asking for by name: the official one leaves the six-hour thruster cycle
    # in, which the others remove.
    wanted_author = str(config.get('kepler_author', 'auto'))

    log(f"Requesting Kepler data for {config['target_name']} within {sr:.1f} arcsec")

    nwritten = 0
    npoints = 0
    covered = []

    for phase in KEPLER_PHASES:
        n, points = _acquire_phase(config, basepath, log, show, phase, sr,
                                   cadence, wanted_author)

        nwritten += n
        npoints += points

        if n:
            covered.append(f"{n} {phase['segment_name'].lower()}(s) of {phase['mission']}")

    if not nwritten:
        log("\nWarning: No Kepler lightcurves could be downloaded")
        return

    log("")

    log_conversion(
        log, 'Kepler',
        'no conversion applied - fluxes are kept as measured',
        {'normalization': 'divided by the median, at display time only',
         'timestamps': f'BKJD (BJD - {KEPLER_BJD0}) in the archive, MJD stored alongside'},
        npoints=npoints,
    )

    log_bands(log, 'Kepler', [
        {'label': 'flux', 'kind': 'native', 'npoints': npoints,
         'note': ', '.join(covered) + ', white light'},
    ])
