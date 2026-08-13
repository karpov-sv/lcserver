"""Kamogata Wide-field Survey lightcurve acquisition module.

Acquires KWS (Kamogata Wide-field Survey) optical lightcurves.
"""

import os
import requests
import numpy as np

from astropy.table import Table
from astropy.time import Time

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    clip_noisy_points, quality_field, quality_level,
                    log_bands, log_conversion, CLIP_RATIO_BY_LEVEL,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


# Largest separation between a V and an Ic measurement still counted as
# simultaneous, in days. KWS steps through its filters in seconds, so anything
# from the same visit falls well inside this.
KWS_COLOR_DT = 0.01


@survey_source(
    name='Kamogata Wide-field Survey',
    short_name='KWS',
    state_acquiring='acquiring KWS lightcurve',
    state_acquired='KWS lightcurve acquired',
    log_file='kws.log',
    output_files=['kws.log', 'kws_lc.png', 'kws_color_mag.png', 'kws.vot', 'kws.txt'],
    button_text='Get KWS lightcurve',
    form_fields={
        'kws_quality': quality_field({
            QUALITY_STANDARD: 'Drop the frames a band measured worst',
            QUALITY_RELAXED: 'Drop only the very worst frames',
            QUALITY_PUBLISHED: 'None - every measurement as published',
        }),
    },
    help_text='Kamogata Wide-field Survey',
    order=23,
    # Lightcurve metadata
    votable_file='kws.vot',
    lc_bands=[
        # KWS observes in V, Ic and B, each on its own scale and none converted
        surveys.band(fn, 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value=fn, color=color,
                     note='as reported by KWS')
        for fn, color in [('V', '#e377c2'), ('Ic', '#8c564b'), ('B', '#1f77b4')]
    ],
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color='#e377c2',
    lc_mode='magnitude',
    lc_short=False,
    # Template metadata
    template_layout='with_cutout',
    requires_coordinates=False,
    # No cutout - KWS is queried by name, so there are no coordinates to cut on
    show_cutout=False,
    show_color_mag=True,
    color_mag_file='kws_color_mag.png',
)
def target_kws(config, basepath=None, verbose=True, show=False):
    """Acquire Kamogata Wide-field Survey lightcurve."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('kws'), basepath=basepath)

    # KWS uses target name, not coordinates
    target_name = config.get('target_name')
    if not target_name:
        log("Error: target_name not found in config")
        raise RuntimeError("Target name required for KWS query")

    # Create cache name based on target name
    # Sanitize the name for use in filename
    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in target_name)
    safe_name = safe_name.replace(' ', '_')
    cache_name = f"kws_{safe_name}.vot"

    with cached_votable_query(cache_name, basepath, log, 'Kamogata Wide-field Survey', refresh=refresh_cache) as cache:
        if not cache.hit:
            log(f"for {target_name}")

            # Query KWS database
            # Note: KWS does not support coordinates, only object names resolved via SIMBAD
            try:
                res = requests.post(
                    "http://kws.cetus-net.org/~maehara/VSdata.py",
                    {
                        "object": target_name,  # KWS resolves object name via SIMBAD
                        "resolver": "simbad",
                        "p_band": "All",
                        "plot": "0",
                        "obs_ys": "",
                        "obs_ms": "",
                        "obs_ds": "",
                        "obs_ye": "",
                        "obs_me": "",
                        "obs_de": "",
                        "submit": "Send query"
                    },
                    timeout=30
                )
                res.raise_for_status()
            except requests.RequestException as e:
                raise SourceError("could not query KWS - "
                                  f"{type(e).__name__}: {e}")

            # Parse response
            # KWS returns HTML with embedded table
            try:
                content = res.content.decode('utf-8', errors='ignore')

                # Extract table from HTML
                start_marker = '<table>'
                end_marker = '</table>'

                start_idx = content.find(start_marker)
                if start_idx == -1:
                    log("Warning: No table found in KWS response")
                    log("Object might not be in KWS database or name not resolved")
                    return

                end_idx = content.find(end_marker, start_idx)
                if end_idx == -1:
                    raise SourceError("malformed KWS response - no table end")

                end_idx += len(end_marker)

                table_html = content[start_idx:end_idx]

                # Parse HTML table
                # Expected columns: name, time, mag, magerr, filter, frame
                kws = Table.read(
                    table_html,
                    format='html',
                    names=['name', 'time', 'mag', 'magerr', 'filter', 'frame'],
                    data_start=1
                )

                if not len(kws):
                    log("Warning: No KWS data points found")
                    return

                # Convert time to MJD
                kws['mjd'] = Time(kws['time']).mjd

                cache.save(kws)

                log(f"Found {len(kws)} KWS data points")

            except SourceError:
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise SourceError("could not parse the KWS response - "
                                  f"{type(e).__name__}: {e}")

        kws = cache.data

    # Filter out bad data
    kws = kws[np.isfinite(kws['mag'])]
    kws = kws[np.isfinite(kws['magerr'])]
    kws = kws[kws['magerr'] > 0]
    kws = kws[kws['magerr'] < 1.0]  # Filter out large errors

    log(f"{len(kws)} data points after filtering")

    if not len(kws):
        log("Warning: No valid KWS data points after filtering")
        return

    # The survey quotes a hundredth of a magnitude for most of what it does and
    # a tenth for the nights it should not have kept, and the difference is
    # where the points a magnitude and a half off the star sit. Band by band,
    # as V and Ic are not measured to the same precision, and against what the
    # band achieved at that brightness rather than against its median, so that
    # a Mira at minimum is not mistaken for a run of bad nights.
    quality = quality_level(config, 'kws')
    clip = (clip_noisy_points(kws['mag'], kws['magerr'], kws['filter'],
                              log=log, group_name='band',
                              ratio=CLIP_RATIO_BY_LEVEL[quality])
            if quality != QUALITY_PUBLISHED
            else np.zeros(len(kws), dtype=bool))

    if np.any(clip):
        kws = kws[~clip]
        log(f"{len(kws)} data points left")

    # Add time column for plotting
    kws['time_obj'] = Time(kws['mjd'], format='mjd')

    # Sort by time
    kws.sort('mjd')

    # Kept for compatibility, but only the V measurements belong in it - it
    # used to hold every band's magnitudes under a V label
    kws['mag_V'] = np.nan
    idx_V = kws['filter'] == 'V'
    kws['mag_V'][idx_V] = kws['mag'][idx_V]

    kws_filters = [str(_) for _ in np.unique(kws['filter'])]

    log_conversion(
        log, 'KWS',
        'no conversion applied - each band is published as measured',
        {'colour term': ('none', 'KWS reports standard V, Ic and B magnitudes'),
         'bands present': ', '.join(kws_filters)},
        npoints=len(kws),
    )

    log_bands(log, 'KWS', [
        {'label': fn, 'kind': 'native',
         'npoints': int(np.sum(kws['filter'] == fn)),
         'note': 'as reported by KWS'}
        for fn in kws_filters
    ])

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'kws_lc.png'), figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        # Plot by filter if multiple filters present
        unique_filters = np.unique(kws['filter'])

        for filt in unique_filters:
            idx = kws['filter'] == filt
            if np.sum(idx):
                label = f'{filt}' if filt else 'unfiltered'
                ax.errorbar(
                    kws['time_obj'][idx].datetime,
                    kws['mag'][idx],
                    kws['magerr'][idx],
                    fmt='.',
                    label=label
                )

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        if len(unique_filters) > 1:
            ax.legend()

        ax.set_ylabel('Magnitude')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - Kamogata Wide-field Survey")

    log("KWS lightcurve plot saved to file:kws_lc.png")

    # Colour-magnitude diagram, from the V and Ic measurements taken together.
    # KWS cycles through its filters within seconds, so the two bands pair up
    # almost exactly; the window is wide enough to absorb the cycle without
    # ever reaching into another night.
    kws_color_dt = config.get('kws_color_dt', KWS_COLOR_DT)

    tV, mV, eV = [kws[_][kws['filter'] == 'V'] for _ in ('mjd', 'mag', 'magerr')]
    tI, mI, eI = [kws[_][kws['filter'] == 'Ic'] for _ in ('mjd', 'mag', 'magerr')]

    iiV, iiI = [], []
    if len(tV) and len(tI):
        tI_arr = np.asarray(tI, dtype=float)
        for i, t1 in enumerate(np.asarray(tV, dtype=float)):
            dist = np.abs(tI_arr - t1)
            j = np.argmin(dist)
            if dist[j] < kws_color_dt:
                iiV.append(i)
                iiI.append(j)

    if len(iiV):
        color = np.asarray(mV)[iiV] - np.asarray(mI)[iiI]
        color_err = np.hypot(np.asarray(eV)[iiV], np.asarray(eI)[iiI])
        magV, magI = np.asarray(mV)[iiV], np.asarray(mI)[iiI]
        errV, errI = np.asarray(eV)[iiV], np.asarray(eI)[iiI]

        with plots.figure_saver(os.path.join(basepath, 'kws_color_mag.png'),
                                figsize=(10, 5), show=show) as fig:
            ax = fig.add_subplot(1, 2, 1)
            ax.errorbar(color, magV, xerr=color_err, yerr=errV,
                        fmt='.', color='#e377c2', alpha=0.5)
            ax.grid(alpha=0.3)
            ax.set_xlabel('V - Ic')
            ax.set_ylabel('V')
            ax.invert_yaxis()

            ax = fig.add_subplot(1, 2, 2, sharex=ax)
            ax.errorbar(color, magI, xerr=color_err, yerr=errI,
                        fmt='.', color='#8c564b', alpha=0.5)
            ax.grid(alpha=0.3)
            ax.set_xlabel('V - Ic')
            ax.set_ylabel('Ic')
            ax.invert_yaxis()

            fig.suptitle(f"{config['target_name']} - KWS")

        log("KWS color-magnitude diagram saved to file:kws_color_mag.png")

        color_mean, color_std = float(np.mean(color)), float(np.std(color))
        log(f"\n---- KWS colour ----\n")
        log(f"{len(iiV)} quasi-simultaneous V and Ic measurements "
            f"within {kws_color_dt*24*60:.0f} min")
        log(f"(V - Ic) = {color_mean:.3f} +/- {color_std:.3f}")

        config['V_minus_Ic'] = color_mean
    else:
        log("No quasi-simultaneous V and Ic measurements, "
            "skipping the colour-magnitude diagram")

    # Save data
    # Remove time_obj column (not serializable to VOTable)
    kws_save = kws[[_ for _ in kws.columns if _ != 'time_obj']]

    kws_save.write(
        os.path.join(basepath, 'kws.vot'),
        format='votable', overwrite=True
    )
    kws_save.write(
        os.path.join(basepath, 'kws.txt'),
        format='ascii.commented_header', overwrite=True
    )
    log("KWS data written to file:kws.vot")
    log("KWS data written to file:kws.txt")
