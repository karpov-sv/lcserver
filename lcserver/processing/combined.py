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
from .utils import cleanup_paths


@survey_source(
    name='Combined Lightcurve',
    short_name='Combined',
    state_acquiring='acquiring combined lightcurve',
    state_acquired='combined lightcurve acquired',
    log_file='combined.log',
    output_files=['combined.log', 'combined_lc.png', 'combined_short_lc.png', 'combined_color_mag.png'],
    button_text='Get combined lightcurve',
    button_class='btn-success',
    help_text='Multi-survey combined plot',
    order=100,
    # Template metadata
    template_layout='custom',
    declination_min=-30,
    main_plot='combined_short_lc.png',
    additional_plots=['combined_lc.png'],
    show_color_mag=True,
    color_mag_file='combined_color_mag.png',
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

    # Get rules from registry instead of hardcoded dict
    from .. import surveys
    combined_lc_rules = surveys.get_combined_lc_rules()

    for short,lcname in [[True, 'combined_short_lc.png'], [False, 'combined_lc.png']]:
        log(f"\n---- Plotting {'short ' if short else ''}lightcurve ----\n")

        with plots.figure_saver(os.path.join(basepath, lcname), figsize=(12, 4), show=show) as fig:
            ax = fig.add_subplot(1, 1, 1)

            for name,rules in combined_lc_rules.items():
                if short and not rules.get('short'):
                    continue

                fullname = os.path.join(basepath, rules['filename'])
                if os.path.exists(fullname):
                    log(f"Reading {rules['name']} data from file:{rules['filename']}")
                    data = Table.read(fullname)

                    data['time'] = Time(data['mjd'], format='mjd')

                    x = data['time'].datetime
                    y = data[rules.get('mag', 'mag_g')]
                    dy = data[rules.get('err', 'magerr')]

                    if 'filter' in rules and rules['filter'] in data.colnames:
                        fnames = data[rules['filter']]
                    else:
                        fnames = np.repeat(rules.get('filter') or '', len(data))

                    for fn in np.unique(fnames):
                        idx = fnames == fn

                        label = rules['name']
                        if fn:
                            label += ' ' + fn

                        ax.errorbar(x[idx], y[idx], dy[idx], fmt='.', label=label)

            ax.invert_yaxis()
            ax.grid(alpha=0.2)

            ax.legend()
            ax.set_ylabel('g magnitude')
            ax.set_xlabel('Time')
            ax.set_title(f"{config['target_name']}")
