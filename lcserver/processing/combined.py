"""Combined lightcurve plotting module.

Creates multi-survey combined lightcurve plots.
"""

import os
import numpy as np

from astropy.table import Table
from astropy.time import Time

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import cleanup_paths, plot_with_errors


@survey_source(
    name='Combined Lightcurve',
    short_name='Combined',
    state_acquiring='acquiring combined lightcurve',
    state_acquired='combined lightcurve acquired',
    log_file='combined.log',
    output_files=['combined.log', 'combined_lc.png', 'combined_short_lc.png'],
    button_text='Get combined lightcurve',
    button_class='btn-success',
    help_text='Multi-survey combined plot',
    order=100,
    # Template metadata
    template_layout='custom',
    declination_min=-30,
    main_plot='combined_short_lc.png',
    additional_plots=['combined_lc.png'],
)
def target_combined(config, basepath=None, verbose=True, show=False):
    """
    Get combined lightcurve from all available surveys.

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

    # Cleanup stale plots
    cleanup_paths(get_output_files('combined'), basepath=basepath)

    from .. import surveys

    # Read once each and kept: a source reaching g by two routes appears twice
    # in the series, and both figures draw from the same tables
    tables = {}

    def read_table(filename):
        if filename not in tables:
            fullname = os.path.join(basepath, filename)
            tables[filename] = Table.read(fullname) if os.path.exists(fullname) else None

        return tables[filename]

    for short, lcname in [[True, 'combined_short_lc.png'], [False, 'combined_lc.png']]:
        log(f"\n---- Plotting {'short ' if short else ''}lightcurve ----\n")

        with plots.figure_saver(os.path.join(basepath, lcname), figsize=(12, 4), show=show) as fig:
            ax = fig.add_subplot(1, 1, 1)

            nplotted = 0
            derived = False

            for entry in surveys.get_combined_series(short=short):
                data = read_table(entry['filename'])
                if data is None:
                    continue

                # A target acquired before this band existed has a table
                # without the column, and is left out rather than failing the
                # whole plot - re-running the source fills it in
                if entry['mag'] not in data.colnames or entry['err'] not in data.colnames:
                    log(f"{entry['label']}: no column {entry['mag']} in "
                        f"file:{entry['filename']} - re-run the source to add it")
                    continue

                # Filled rather than asarray: a VOTable column comes back
                # masked, and the fill value under the mask is a number like
                # any other once the mask is dropped
                mag = np.ma.filled(np.ma.asarray(data[entry['mag']], dtype=float),
                                   np.nan)
                idx = np.isfinite(mag)

                # Bands sharing a table with others take only their own rows
                if entry['filter_column'] and entry['filter_value'] is not None:
                    if entry['filter_column'] in data.colnames:
                        idx &= np.asarray(data[entry['filter_column']]) == entry['filter_value']

                if not np.sum(idx):
                    continue

                data = data[idx]
                x = Time(data['mjd'], format='mjd').datetime

                log(f"{entry['label']}: {len(data)} points from "
                    f"file:{entry['filename']}")

                plot_with_errors(ax, x, data[entry['mag']],
                                 data[entry['err']], color=entry['color'],
                                 label=entry['label'], ms=1)

                nplotted += 1
                derived |= entry['kind'] == surveys.BAND_DERIVED

            if not nplotted:
                log("Nothing to plot - no source has data on the common g scale")

            ax.invert_yaxis()
            ax.grid(alpha=0.2)

            # Two columns and a translucent frame: with every source on one
            # axis the legend is long, and it has nowhere to sit that is not
            # over somebody's data
            if nplotted:
                ax.legend(ncol=2 if nplotted > 4 else 1, fontsize='small',
                          framealpha=0.8)

            ax.set_ylabel('g magnitude')
            ax.set_xlabel('Time')
            ax.set_title(f"{config['target_name']}")

            # Said on the figure and not only in the log, as the figure is what
            # gets looked at, and a converted amplitude is not a measured one
            if derived:
                ax.text(0.995, 0.03,
                        'converted bands assume a constant colour',
                        transform=ax.transAxes, ha='right', va='bottom',
                        fontsize='x-small', alpha=0.6)

    log(f"\nCombined lightcurves written to file:combined_short_lc.png and "
        f"file:combined_lc.png")
