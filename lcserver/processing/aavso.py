"""AAVSO International Database lightcurve acquisition module.

Acquires visual and instrumental photometry contributed by AAVSO observers.

The database is read through the official API at apps.aavso.org, which wants a
key: there is no anonymous route to the photometry. The search page hands out
the first page of a result and answers the second with an AWS WAF CAPTCHA. The
key goes in AAVSO_TOKEN; with none set this step says so and finishes with
nothing.

The download endpoint returns the whole light curve as a zipped CSV in one
request, which takes about a second whatever the size of it. There is also a
listing endpoint that pages ten observations at a time, and it is not used:
the throttle is one bucket for the whole account - both endpoints refuse with
the same Retry-After to the second - so paging eight hundred observations
spends eighty requests of the same allowance the download spends one of, and
gets less back for them, the listing carrying neither the airmass nor whether
the observer transformed.

Throttled, then, there is nothing to fall back to, and the step says how long
the archive asked to be left alone for.

The listing is asked one thing all the same. The download answers 500 for a
star it holds nothing on, rather than with an empty file, so a 500 is
ambiguous between an archive that has no photometry here and one that is
broken. The listing answers nought for the first and is the only thing that
can tell them apart, so it settles that question and no other.
"""

import os
import csv
import io
import json
import zipfile
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from astropy.table import Table
from astropy.time import Time

from django.conf import settings

# STDPipe
from stdpipe import plots

from .. import surveys
from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    clip_noisy_points, quality_field, quality_level,
                    log_bands, log_conversion, plot_with_errors,
                    assumed_color, v_to_g, b_to_g, rc_to_g,
                    V_TO_G_FORMULA, B_TO_G_FORMULA, RC_TO_G_FORMULA,
                    CLIP_RATIO_BY_LEVEL,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


AAVSO_LIST_URL = 'https://apps.aavso.org/v2/api/observations/photometry/'
AAVSO_DOWNLOAD_URL = AAVSO_LIST_URL + 'download/'

# The bands, as the AAVSO's own file format abbreviates them. The download
# spells most of them out and gives a few as the abbreviation itself, so both
# spellings are read and anything unrecognised is kept as it arrived rather
# than dropped.
AAVSO_BAND_NAMES = {
    'Visual': 'Vis', 'Unknown': 'Unk', 'Johnson U': 'U', 'Johnson V': 'V',
    'Johnson B': 'B', 'Cousins R': 'R', 'Cousins I': 'I',
    'Orange (Liller)': 'O',
    'Unfiltered with V Zeropoint': 'CV', 'Unfiltered with R Zeropoint': 'CR',
    'Johnson R': 'RJ', 'Johnson I': 'IJ',
    'Halpha': 'HA', 'Halpha-continuum': 'HAC',
    'J NIR 1.2micron': 'J', 'H NIR 1.6micron': 'H', 'K NIR 2.2micron': 'K',
    'Sloan u': 'SU', 'Sloan g': 'SG', 'Sloan r': 'SR', 'Sloan i': 'SI',
    'Sloan z': 'SZ',
    'Stromgren u': 'STU', 'Stromgren v': 'STV', 'Stromgren b': 'STB',
    'Stromgren y': 'STY', 'Stromgren Hbw': 'STHBW', 'Stromgren Hbn': 'STHBN',
    'Tri-Color Blue': 'TB', 'Tri-Color Green': 'TG', 'Tri-Color Red': 'TR',
    'Optec Wing A': 'MA', 'Optec Wing B': 'MB', 'Optec Wing C': 'MI',
    'PanSTARRS Z-short': 'ZS', 'PanSTARRS Y': 'Y',
    # The plain colour filters an observer names without saying which
    # realisation. No abbreviation of their own, so they keep the name.
    'Blue': 'Blue', 'Green': 'Green', 'Red': 'Red', 'Yellow': 'Yellow',
}

# Every band the viewer offers, in the order above and each named once
AAVSO_BAND_CODES = list(dict.fromkeys(AAVSO_BAND_NAMES.values()))

# What the measurement was made with, for the log
AAVSO_OBSTYPES = {'Visual': 'visual', 'CCD': 'CCD', 'PEP': 'PEP',
                  'DSLR': 'DSLR'}

# The colours the viewer draws the common bands in. Anything not here falls
# back to the source's own colour, which is what the rare filters want - there
# are forty of them, and a palette that named each would say nothing.
AAVSO_COLORS = {
    'V': '#2ca02c', 'B': '#1f77b4', 'R': '#d62728', 'I': '#8c564b',
    'CV': '#98df8a', 'CR': '#ff9896', 'Vis': '#7f7f7f', 'U': '#9467bd',
    'SG': '#17becf', 'SR': '#e377c2', 'SI': '#bcbd22',
    'TB': '#aec7e8', 'TG': '#c5b0d5', 'TR': '#ffbb78',
}

# Largest quoted uncertainty still kept, in magnitudes
AAVSO_MAX_ERR = 1.0


class AavsoThrottled(SourceError):
    """The archive has said to come back later, and in how many seconds.

    A SourceError, because that is what it is from the run's point of view:
    the step could not get its data, for a reason belonging to the archive
    rather than to the code, and the reason is the whole of the story.
    """

    def __init__(self, retry_after):
        self.retry_after = retry_after
        wait = (f" - it asks for another {retry_after // 60} minutes"
                if retry_after else "")
        super().__init__("the AAVSO is throttling this key" + wait)


class AavsoServerError(SourceError):
    """The archive answered with something other than the data.

    Carries the status, because one of them has to be told apart: the
    download endpoint answers 500 for a star it holds nothing on.
    """

    def __init__(self, code, reason):
        self.code = code
        super().__init__(f"could not query the AAVSO - HTTP {code} {reason}")


def _aavso_get(url, token, timeout=90):
    """One request, as bytes, with the AAVSO's refusals told apart."""
    request = urllib.request.Request(url, headers={
        'Authorization': f'Token {token}',
        'Accept': '*/*',
    })

    try:
        with urllib.request.urlopen(request, timeout=timeout) as res:
            return res.read()
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise SourceError("the AAVSO refused the key in AAVSO_TOKEN - "
                              "it answered 401")
        if e.code == 429:
            # Its own words for how long, where it gave them
            try:
                retry = int(e.headers.get('Retry-After') or 0)
            except (TypeError, ValueError):
                retry = 0
            raise AavsoThrottled(retry)
        raise AavsoServerError(e.code, e.reason)


def _aavso_short_band(value):
    """A band as the AAVSO file format abbreviates it, however it arrived."""
    value = str(value or '').strip()

    return AAVSO_BAND_NAMES.get(value, value)


def _is_true(value):
    """Whether a flag arrived set, spelt however the archive spelt it."""
    return str(value).strip().lower() in ('true', '1', 'yes')


def _aavso_row(jd, mag, err, band, obstype, obscode, airmass, transformed):
    """One observation, in the columns the rest of this reads."""
    def number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    return {
        'mjd': number(jd) - 2400000.5,
        'mag': number(mag),
        # Visual estimates carry no uncertainty, and neither does much of what
        # was submitted before they were asked for. NaN rather than dropped: a
        # measurement without an error bar is still a measurement, and
        # inventing one would put a number on the plot nobody measured.
        'magerr': number(err),
        'filter': _aavso_short_band(band),
        'obstype': AAVSO_OBSTYPES.get(str(obstype or '').strip(), 'unknown'),
        'obscode': str(obscode or ''),
        'airmass': number(airmass),
        'transformed': _is_true(transformed),
    }


def _aavso_count(name, token):
    """How many observations the archive says it holds under a name."""
    url = AAVSO_LIST_URL + '?' + urllib.parse.urlencode({'target': name})

    return json.loads(_aavso_get(url, token)).get('count', 0)


def _aavso_bulk(name, token, log):
    """The whole light curve in one request, from the download endpoint.

    Returns the observations as rows, or raises AavsoThrottled where the
    account's allowance has been spent.
    """
    url = AAVSO_DOWNLOAD_URL + '?' + urllib.parse.urlencode({
        'target': name,
        'output_format': 'csv',
    })

    # Generous: the request itself is quick, but a century of a bright star is
    # a large file, and the archive assembles it before it starts sending
    try:
        content = _aavso_get(url, token, timeout=300)
    except AavsoServerError as e:
        if e.code < 500:
            raise

        # The download endpoint answers 500 for a star it has nothing on,
        # where the listing answers nought quite happily. Which of the two
        # this is decides whether the step failed or simply found nothing, and
        # only the listing can say, so it is asked - but only here, once the
        # cheap route has already gone wrong.
        if _aavso_count(name, token):
            raise

        log("The AAVSO download failed, and the listing says the archive has "
            "nothing under this name - taking that as the answer")
        return []

    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
        members = [n for n in archive.namelist() if n.lower().endswith('.csv')]

        if not members:
            raise SourceError("the AAVSO download held no CSV - it contained "
                              + (', '.join(archive.namelist()) or 'nothing'))

        text = archive.read(members[0]).decode('utf8', errors='replace')
    except zipfile.BadZipFile:
        raise SourceError("the AAVSO download was not a readable zip archive")

    rows, nlimits = [], 0

    for row in csv.DictReader(io.StringIO(text)):
        # An upper limit rather than a measurement - the observer saw nothing
        # and reported how faint they would have seen it. Not a magnitude, and
        # drawn as one it would look like an outburst upside down.
        if _is_true(row.get('fainterthan')):
            nlimits += 1
            continue

        rows.append(_aavso_row(
            row.get('jd'), row.get('mag'), row.get('uncertainty'),
            row.get('band'), row.get('type'), row.get('observer'),
            row.get('airmass'), row.get('transformed')))

    log(f"Downloaded {len(rows) + nlimits} observations in one request")

    if nlimits:
        log(f"Dropped {nlimits} fainter-than reports, which are limits and "
            "not measurements")

    return rows


def _aavso_plain(table):
    """The table with its masks resolved into ordinary values.

    A VOTable read back from the cache returns masked columns, and the
    observations with no quoted uncertainty come back masked rather than NaN.
    numpy then hides them from everything asked about them - np.isfinite over
    a masked column reports nothing missing, and the boolean masks built from
    it are themselves masked - so the cached run and the fresh one would
    disagree about the same data. Resolved here, once, whichever route the
    table arrived by.
    """
    for name in ('mjd', 'mag', 'magerr', 'airmass'):
        if name in table.colnames:
            table[name] = np.ma.filled(
                np.ma.asarray(table[name], dtype=float), np.nan)

    for name in ('filter', 'obstype', 'obscode'):
        if name in table.colnames:
            table[name] = np.ma.filled(np.ma.asarray(table[name]), '')

    return table


def _aavso_table(rows):
    """The observations as a table, with the columns in a settled order."""
    columns = ['mjd', 'mag', 'magerr', 'filter', 'obstype', 'obscode',
               'airmass', 'transformed']

    return Table({name: np.array([row[name] for row in rows])
                  for name in columns})


@survey_source(
    name='AAVSO International Database',
    short_name='AAVSO',
    state_acquiring='acquiring AAVSO lightcurve',
    state_acquired='AAVSO lightcurve acquired',
    log_file='aavso.log',
    output_files=['aavso.log', 'aavso_lc.png', 'aavso.vot', 'aavso.txt'],
    button_text='Get AAVSO lightcurve',
    form_fields={
        'aavso_quality': quality_field({
            QUALITY_STANDARD: 'Drop the frames a band measured worst',
            QUALITY_RELAXED: 'Drop only the very worst frames',
            QUALITY_PUBLISHED: 'None - every observation as submitted',
        }),
    },
    help_text='Photometry contributed by AAVSO observers, queried by name',
    order=24,
    about=(
        "The AAVSO International Database, a century of visual and "
        "instrumental photometry of variable stars contributed by observers "
        "worldwide, and still the longest continuous record most bright "
        "variables have. Queried by star name rather than by position, and "
        "read through the AAVSO API, which needs a key."),
    about_links=[
        ('AAVSO', 'https://www.aavso.org/'),
        ('Search the database',
         'https://apps.aavso.org/v2/data/search/photometry/'),
        ('API documentation', 'https://docs.aavso.org/'),
    ],
    # The wording the archive itself ships in the README of every download,
    # rather than one recalled from the literature
    acknowledgement=(
        "Observations from the AAVSO International Database, contributed by "
        "observers worldwide. American Association of Variable Star Observers "
        "(AAVSO). AAVSO International Database. https://www.aavso.org/"),
    # Lightcurve metadata
    votable_file='aavso.vot',
    lc_bands=[
        # Every filter the database distinguishes, as it reports it. A band
        # with nothing in it is not drawn, so naming them all costs a target
        # nothing and means a rare filter is not silently dropped.
        surveys.band(label, 'mag', 'magerr', surveys.BAND_NATIVE,
                     filter_column='filter', filter_value=label,
                     color=AAVSO_COLORS.get(label),
                     note='as submitted to the AAVSO',
                     # Sloan g is the scale the combined curve is drawn on, so
                     # what an observer measured through a g filter joins it
                     # as measured and needs no conversion to get there
                     combined=(label == 'SG'))
        for label in AAVSO_BAND_CODES
    ] + [
        # The common g scale. V is the route the combined curve takes, and CV
        # the other one worth taking: clear frames measured against V-band
        # comparison stars sit on the V scale already, and an observer works
        # either through a V filter or through none, so the two rarely
        # describe the same night twice.
        surveys.band('g (conv.)', 'mag_g', 'magerr', surveys.BAND_DERIVED,
                     filter_column='filter', filter_value='V', color='#98df8a',
                     note='V put on the common g scale using an assumed g - r',
                     combined=True),
        surveys.band('g (from CV)', 'mag_g_from_CV', 'magerr',
                     surveys.BAND_DERIVED,
                     filter_column='filter', filter_value='CV', color='#c7e9c0',
                     note='clear frames on the V scale, moved onto g using an '
                          'assumed g - r',
                     combined=True),
        # Offered for inspection but kept off the combined curve: B and R are
        # usually taken alongside the V of the same night, and drawing them
        # too would show one star three times from one archive.
        surveys.band('g (from B)', 'mag_g_from_B', 'magerr',
                     surveys.BAND_DERIVED,
                     filter_column='filter', filter_value='B', color='#aec7e8',
                     note='B put on the common g scale using an assumed g - r'),
        surveys.band('g (from Rc)', 'mag_g_from_R', 'magerr',
                     surveys.BAND_DERIVED,
                     filter_column='filter', filter_value='R', color='#ff9896',
                     note='Cousins R put on the common g scale using an '
                          'assumed g - r'),
        surveys.band('g (from Vis)', 'mag_g_from_Vis', 'magerr',
                     surveys.BAND_DERIVED,
                     filter_column='filter', filter_value='Vis', color='#c7c7c7',
                     note='visual estimates treated as V and moved onto g; a '
                          'tenth of a magnitude at best, and shown for the '
                          'shape of the curve rather than its level'),
    ],
    lc_mag_column='mag',
    lc_err_column='magerr',
    lc_filter_column='filter',
    lc_color='#2ca02c',
    lc_mode='magnitude',
    lc_short=True,
    # Template metadata
    template_layout='simple',
    # Queried by name, so there are no coordinates to cut an image on
    requires_coordinates=False,
)
def target_aavso(config, basepath=None, verbose=True, show=False):
    """Acquire AAVSO International Database lightcurve."""
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('aavso'), basepath=basepath)

    token = getattr(settings, 'AAVSO_TOKEN', '')

    if not token:
        # An installation without a key, rather than a star without data. Said
        # once and finished cleanly: the step has nothing to show either way,
        # and a run of every source should not report a failure over a setting.
        log("No AAVSO_TOKEN is set, and the database cannot be read without "
            "one. Get a key from the AAVSO account settings page and put it "
            "in the environment as AAVSO_TOKEN.")
        return

    # The AAVSO knows the star under its variable-star designation, which is
    # what VSX calls it and what the info step has already looked up. The name
    # the target was created with is the fallback, and is often the same one
    # typed less carefully - "af and" resolves as readily as "AF And".
    name = config.get('vsx_name') or config.get('target_name')

    if not name:
        log("Error: neither vsx_name nor target_name is in the config")
        raise RuntimeError("Target name required for AAVSO query")

    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_'
                        for c in name).replace(' ', '_')
    cache_name = f"aavso_{safe_name}.vot"

    with cached_votable_query(cache_name, basepath, log,
                              'AAVSO International Database',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            log(f"for {name}")

            rows = _aavso_bulk(name, token, log)

            if not rows:
                cache.save_empty()
                log(f"The AAVSO has no measurements under the name {name}")
                return

            cache.save(_aavso_table(rows))

        aavso = cache.data

    # Nothing here, and cached as nothing - the helper has said so already
    if aavso is None:
        return

    aavso = _aavso_plain(aavso)

    log(f"{len(aavso)} observations, "
        f"{int(np.sum(~np.isfinite(aavso['magerr'])))} of them without a "
        "quoted uncertainty")

    # Filter out bad data. An absent uncertainty is not a bad one, so the
    # bounds are applied only where there is a number to apply them to.
    good = np.isfinite(aavso['mag'])
    err = aavso['magerr']
    good &= ~np.isfinite(err) | ((err > 0) & (err < AAVSO_MAX_ERR))
    aavso = aavso[good]

    log(f"{len(aavso)} data points after filtering")

    if not len(aavso):
        log("Warning: No valid AAVSO data points after filtering")
        return

    # The archive is many observers rather than one instrument, and what one
    # of them calls a tenth of a magnitude another calls a hundredth, so the
    # clipping is done band by band against what that band achieved at that
    # brightness. Only where an uncertainty was quoted: the rest cannot be
    # judged this way, and dropping them for that would throw out the visual
    # record entirely.
    quality = quality_level(config, 'aavso')
    clip = np.zeros(len(aavso), dtype=bool)

    if quality != QUALITY_PUBLISHED:
        known = np.isfinite(aavso['magerr'])

        if np.sum(known):
            clip[known] = clip_noisy_points(
                aavso['mag'][known], aavso['magerr'][known],
                aavso['filter'][known], log=log, group_name='band',
                ratio=CLIP_RATIO_BY_LEVEL[quality])

    if np.any(clip):
        aavso = aavso[~clip]
        log(f"{len(aavso)} data points left")

    # Sort by time
    aavso.sort('mjd')

    aavso_filters = [str(_) for _ in np.unique(aavso['filter'])]

    log_conversion(
        log, 'AAVSO',
        'no conversion applied - each band is published as submitted',
        {'colour term': ('none', 'observers submit standard magnitudes, '
                                 'transformed or not as they say'),
         'bands present': ', '.join(aavso_filters)},
        npoints=len(aavso),
    )

    # The common g scale. Every route is through the same assumed colour, and
    # each band gets its own column so that the viewer can offer them apart.
    g_minus_r, g_minus_r_origin = assumed_color(config, 'g_minus_r')

    routes = [
        ('V', 'mag_g', v_to_g, V_TO_G_FORMULA, None),
        ('CV', 'mag_g_from_CV', v_to_g, V_TO_G_FORMULA,
         'clear frames measured against V-band comparison stars, so taken as '
         'V and converted as one'),
        ('B', 'mag_g_from_B', b_to_g, B_TO_G_FORMULA,
         'offered for inspection only - the combined light curve takes the V '
         'route, and B is usually the same night seen through another filter'),
        ('R', 'mag_g_from_R', rc_to_g, RC_TO_G_FORMULA,
         'offered for inspection only, for the same reason as B'),
        ('Vis', 'mag_g_from_Vis', v_to_g, V_TO_G_FORMULA,
         'a visual estimate is a tenth of a magnitude at best, and is worth '
         'the shape of the curve rather than its level'),
    ]

    counts = {}

    for band, column, convert, formula, note in routes:
        idx = aavso['filter'] == band
        aavso[column] = np.nan
        aavso[column][idx] = convert(aavso['mag'][idx], g_minus_r)
        counts[band] = int(np.sum(idx))

        if counts[band]:
            log_conversion(
                log, 'AAVSO', formula,
                {'(g - r)': (g_minus_r, g_minus_r_origin)},
                npoints=counts[band],
                note=note or 'the colour is assumed constant over the whole '
                             'light curve',
            )

    log_bands(log, 'AAVSO', [
        {'label': fn, 'kind': 'native',
         'npoints': int(np.sum(aavso['filter'] == fn)),
         'note': 'as submitted to the AAVSO'}
        for fn in aavso_filters
    ] + [
        {'label': label, 'kind': 'derived', 'npoints': counts[band],
         'note': f'{band} on the common g scale'}
        for band, label in [('V', 'g (conv.)'), ('CV', 'g (from CV)'),
                            ('B', 'g (from B)'), ('R', 'g (from Rc)'),
                            ('Vis', 'g (from Vis)')]
        if counts[band]
    ])

    # What each observer contributed, which is what this archive is
    obscodes = np.unique(aavso['obscode'])
    obstypes = [f"{np.sum(aavso['obstype'] == t)} {t}"
                for t in np.unique(aavso['obstype'])]
    log(f"\n{len(obscodes)} observer{'s' if len(obscodes) != 1 else ''} "
        "contributed these points: " + ', '.join(obstypes))

    # Add time column for plotting
    aavso['time_obj'] = Time(aavso['mjd'], format='mjd')

    # Plot lightcurve
    with plots.figure_saver(os.path.join(basepath, 'aavso_lc.png'),
                            figsize=(12, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        for filt in aavso_filters:
            idx = aavso['filter'] == filt
            if np.sum(idx):
                plot_with_errors(
                    ax,
                    aavso['time_obj'][idx].datetime,
                    aavso['mag'][idx],
                    aavso['magerr'][idx],
                    color=AAVSO_COLORS.get(filt),
                    label=filt
                )

        ax.invert_yaxis()
        ax.grid(alpha=0.2)

        if len(aavso_filters) > 1:
            ax.legend()

        ax.set_ylabel('Magnitude')
        ax.set_xlabel('Time')
        ax.set_title(f"{config['target_name']} - AAVSO")

    log("AAVSO lightcurve plot saved to file:aavso_lc.png")

    # Save data
    # Remove time_obj column (not serializable to VOTable)
    aavso_save = aavso[[_ for _ in aavso.columns if _ != 'time_obj']]

    aavso_save.write(
        os.path.join(basepath, 'aavso.vot'),
        format='votable', overwrite=True
    )
    aavso_save.write(
        os.path.join(basepath, 'aavso.txt'),
        format='ascii.commented_header', overwrite=True
    )
    log("AAVSO data written to file:aavso.vot")
    log("AAVSO data written to file:aavso.txt")
