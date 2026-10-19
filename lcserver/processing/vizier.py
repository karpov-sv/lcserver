"""Every VizieR catalogue with an entry at the target's position, and the entries.

One request answers it: a cone search naming no catalogue searches all of
them, and returns a table for every catalogue that has anything there - some
fifty for an ordinary star, two hundred for Vega, in ten to thirty seconds.
Its entries come with it, up to ROWS of each, in the columns VizieR shows by
default. Most tables have one; the rest are epochs of a time series - Gaia's
epoch photometry, its XP spectrum sampled point by point - and the observation
logs of the archives, which for a bright star run to thousands and are what
the limit is for.

Each catalogue says which paper it comes from, as a bibcode, which ties it to
the references SIMBAD lists exactly: the tables published with the papers
about the object are the ones worth reading first. Tables without coordinates
are not searched, so a paper's table of its own measurements may not be here.

The reply is kept as it came, as the cache and as the file the page reads:
every table has columns of its own, which no single table could hold. Read
with ElementTree rather than astropy, which reads only the first of the
tables, and does not keep the metadata of a resource.
"""

import os
import shutil
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

import requests

from .utils import cached_votable_query
from . import references as references_


VIZIER_VOTABLE = 'https://vizier.cds.unistra.fr/viz-bin/votable'
VIZIER_QUERY = 'https://vizier.cds.unistra.fr/viz-bin/VizieR-4'

# Cone radius, in arcsec, as the SIMBAD query has it
SR = 3.0

# The most entries of any one table. Enough for every ordinary catalogue in
# full; an archive's observation log of a bright star is cut short, and its
# link on the page leads to the rest.
ROWS = 50

# Connect, then read. Vega's reply is three megabytes and half a minute.
TIMEOUT = (15, 180)

# How many of the catalogues the info log names
LISTED = 5

# The file the catalogue list reads: VizieR's reply as it came
CATALOGUES_FILE = 'catalogues.vot'


def _info(resource, name):
    """The value of a named INFO of a resource, or ''."""
    for info in resource.findall('{*}INFO'):
        if info.get('name') == name:
            return info.get('value', '')

    return ''


def _text(element):
    """The text of an element, whitespace collapsed, or ''."""
    return ' '.join((element.text or '').split()) if element is not None else ''


def query(ra, dec, sr=SR):
    """VizieR's reply to a search of every catalogue within `sr` arcsec, as bytes."""
    res = requests.post(VIZIER_VOTABLE, data={
        '-c': f"{ra:.6f} {dec:+.6f}", '-c.rs': sr,
        # The entries of each table with their distance, and the description
        # of the catalogue it is in: what astroquery asks for, and without
        # which the tables come back merged into one and the catalogue names
        # are lost
        '-out.max': ROWS, '-out.add': '_r', '-out.meta': 'huUD',
        '-out.form': 'mini', '-oc.form': 'd'}, timeout=TIMEOUT)
    res.raise_for_status()

    return res.content


def parse(content):
    """The catalogues in a reply, in VizieR's order, and the number of errors
    it reported - catalogues it could not search.

    Each a dict with its tables, and each table with its columns and entries,
    the values as the text VizieR formatted them in.
    """
    root = ET.fromstring(content)
    result = []

    for resource in root.findall('.//{*}RESOURCE'):
        bibcode = _info(resource, 'cites')
        name = resource.get('name', '')
        tables = []

        for table in resource.findall('{*}TABLE'):
            fields = [{'name': _.get('name', ''), 'unit': _.get('unit', ''),
                       'description': _text(_.find('{*}DESCRIPTION'))}
                      for _ in table.findall('{*}FIELD')]
            rows = [[(_.text or '').strip() for _ in tr.findall('{*}TD')]
                    for tr in table.findall('{*}DATA/{*}TABLEDATA/{*}TR')]

            if not rows:
                continue

            names = [_['name'] for _ in fields]
            distances = []

            if '_r' in names:
                i = names.index('_r')
                fields[i].update(name='r', unit='arcsec')

                for row in rows:
                    try:
                        distances.append(float(row[i]))
                    except (ValueError, IndexError):
                        pass

            tables.append({
                'table': table.get('name', ''),
                # VizieR's '* output (out of 526,280,881 stars)' of a big survey
                'description': _text(table.find('{*}DESCRIPTION')).removeprefix('* '),
                'distance': min(distances) if distances else None,
                'fields': fields,
                'rows': rows,
                'truncated': len(rows) >= ROWS,
            })

        if tables:
            distances = [_['distance'] for _ in tables if _['distance'] is not None]
            result.append({
                'catalogue': name,
                'title': _text(resource.find('{*}DESCRIPTION')),
                'creator': _info(resource, 'creator'),
                'year': _info(resource, 'original_date'),
                'bibcode': bibcode[8:] if bibcode.startswith('bibcode:') else '',
                'url': _info(resource, 'reference_url'),
                # Published with a paper, rather than a survey or a compilation
                'journal': name.startswith('J/'),
                'distance': min(distances) if distances else None,
                'tables': tables,
            })

    errors = sum(_.get('name') == 'Error' for _ in root.findall('.//{*}INFO'))

    return result, errors


def link(table, ra, dec, sr=SR):
    """VizieR's page of a table's entries at the target."""
    return VIZIER_QUERY + '?' + urlencode({
        '-source': table, '-c': f"{ra:.6f} {dec:+.6f}", '-c.rs': sr,
        '-out.add': '_r', '-sort': '_r'}, safe='/')


def catalogues(cats, refs, ra, dec):
    """The catalogues ranked for the templates, with links to their entries.

    Those from the papers SIMBAD links to the object come first, as the
    references rank them, and the rest in VizieR's own order - the big
    surveys, then the journal tables, newest first.
    """
    refs = {_['bibcode']: _ for _ in refs or []}

    for order, cat in enumerate(cats):
        cat['reference'] = refs.get(cat['bibcode'])
        cat['score'] = cat['reference']['score'] if cat['reference'] else None
        cat['order'] = order

        for table in cat['tables']:
            table['link'] = link(table['table'], ra, dec)

    return sorted(cats, key=lambda _: (_['score'] is None, -(_['score'] or 0), _['order']))


def acquire(ra, dec, basepath, log, refresh=False):
    """Fetch the catalogues at the target and write them for the page to read.

    Returns them, or None where there are none or VizieR could not be asked.
    """
    with cached_votable_query(f"vizier_rows_{ra:.4f}_{dec:.4f}.vot", basepath, log,
                              'VizieR', refresh=refresh, raw=True) as cache:
        if not cache.hit:
            try:
                content = query(ra, dec)
                cats, errors = parse(content)
            except Exception as e:
                log(f"Error: could not search VizieR: {e}")
                return None

            # Some catalogues sit in databases VizieR fails to reach, the same
            # ones every time; what it found in the rest is still an answer
            if errors:
                log(f"VizieR reported {errors} errors, so some catalogues may be missing")

            # Kept whether or not anything was found, as the reply says which
            cache.save(content)
        else:
            cats, errors = parse(cache.data)

    if not cats:
        log(f"No VizieR catalogue has an entry within {SR:.0f} arcsec")
        return None

    shutil.copyfile(cache.path, os.path.join(basepath, CATALOGUES_FILE))

    cats = catalogues(cats, references_.load(basepath), ra, dec)
    cited = [_ for _ in cats if _['reference']]
    entries = sum(len(t['rows']) for c in cats for t in c['tables'])

    log(f"Catalogues: [{len(cats)}](page:target_catalogues) with an entry within "
        f"{SR:.0f} arcsec, {len(cited)} of them from the papers SIMBAD links to it; "
        f"{entries} entries in all")

    for cat in cited[:LISTED]:
        # Italic, so any star in the title is escaped from ending it early
        title = cat['title'].replace('*', '\\*')
        log(f"  [{cat['catalogue']}]({cat['tables'][0]['link']}): *{title}*")

    return cats


def load(basepath, ra, dec):
    """The catalogues written for a target, papers on it first, and the number
    of errors VizieR reported finding them; or None and 0."""
    path = os.path.join(basepath, CATALOGUES_FILE)

    if not os.path.exists(path):
        return None, 0

    with open(path, 'rb') as f:
        cats, errors = parse(f.read())

    return catalogues(cats, references_.load(basepath), ra, dec), errors
