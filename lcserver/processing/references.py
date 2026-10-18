"""The literature on a target, as SIMBAD has indexed it.

SIMBAD links every object to the papers that mention it, and records for each
link how often the paper names the object and where - title, abstract, a
table, or only in passing. That is enough to tell a paper about the object from
one that lists it among a million others, which is what makes a reference list
of thousands - Vega has nearly three thousand - worth reading at all.

Read from the TAP service as JSON rather than through astroquery. The name a
paper gives the object comes back in a column declared as ASCII that holds
Greek letters ('alpha Lyr' written as one), and astropy's VOTable parser
refuses the whole reply over it.

ADS adds what SIMBAD does not keep: how often each paper has been cited, and
where to read it for free. It wants a key, which goes in ADS_TOKEN; without one
the list is SIMBAD's alone.
"""

import os
import re
from urllib.parse import quote

import numpy as np
import requests

from astropy.table import Table, MaskedColumn

from django.conf import settings

from .utils import cached_votable_query


SIMBAD_TAP = 'https://simbad.cds.unistra.fr/simbad/sim-tap/sync'
ADS_BIGQUERY = 'https://api.adsabs.harvard.edu/v1/search/bigquery'

# The most bibcodes ADS takes in one request
ADS_CHUNK = 2000

ADS_ABSTRACT = 'https://ui.adsabs.harvard.edu/abs/'

# Connect, then read. Vega's list with its abstracts is 2.7 MB and takes
# SIMBAD several seconds to assemble.
TIMEOUT = (15, 120)

# Where the paper names the object, as SIMBAD's ref_flag bits, in the order
# its own pages list them. Recorded fully only since 2008, and mostly for the
# major journals, so an older paper with none of them may still be about it.
REF_FLAGS = [
    ('T', 1, 'title'),
    ('K', 4, 'keywords'),
    ('A', 2, 'abstract'),
    ('S', 8, 'a subtitle'),
    ('O', 256, 'an \\object tag'),
    ('X', 128, 'the text'),
    ('C', 16, 'a caption'),
    ('D', 32, 'a table'),
    ('F', 64, 'a figure'),
]

# A paper naming the object in any of these is about it rather than mentioning it
FOCUS_FLAGS = 1 | 2 | 4

# How many of the papers the info log names, most relevant first
KEY_PAPERS = 5

# How many authors to name before 'et al.'
AUTHORS_SHOWN = 3

# The file the target page and the reference list read
REFERENCES_FILE = 'references.vot'

COLUMNS = {
    # name: (type, filler for a null)
    'bibcode': (str, ''), 'year': (int, 0), 'journal': (str, ''),
    'volume': (int, 0), 'page': (int, 0), 'title': (str, ''),
    'doi': (str, ''), 'abstract': (str, ''), 'nbobject': (int, 0),
    'obj_freq': (int, 0), 'ref_flag': (int, 0), 'ref_raw_id': (str, ''),
    'authors': (str, ''), 'nauthors': (int, 0),
    'citations': (int, -1), 'arxiv': (str, ''), 'data': (str, ''),
}

# Where ADS sends a reader for the data of a paper in one archive: a redirect
# to that archive's own page for the paper, which needs no key and no request
# of ours to know the address of
ADS_GATEWAY = 'https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/{archive}'

# Archives ADS lists for a paper that hold the objects it names rather than
# data from it, which the list of references is already made from
ADS_NOT_DATA = {'SIMBAD', 'NED'}


def _tap(query):
    """Rows of a SIMBAD TAP query, as dicts."""
    res = requests.post(SIMBAD_TAP, data={
        'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'json',
        'MAXREC': 1000000, 'query': query}, timeout=TIMEOUT)
    res.raise_for_status()

    reply = res.json()
    names = [_['name'] for _ in reply['metadata']]

    return [dict(zip(names, row)) for row in reply['data']]


def _table(rows):
    """Rows as a table, nulls filled, so that it writes to a VOTable unmasked."""
    return Table({
        name: np.array([kind(row.get(name)) if row.get(name) is not None
                        else filler for row in rows],
                       dtype=str if kind is str else int)
        for name, (kind, filler) in COLUMNS.items()})


def _filled(table):
    """A table read back from a VOTable, with its masked values filled.

    An empty string goes into a VOTable as itself and comes out as a masked one.
    """
    if table is None:
        return None

    table = table.copy()

    for name in table.colnames:
        if isinstance(table[name], MaskedColumn):
            filler = COLUMNS.get(name, (str, ''))[1]
            table[name] = table[name].filled(filler)

    return table


def query_simbad(main_id):
    """Every reference SIMBAD links to an object, with its first authors."""
    # Quoted for ADQL, though a SIMBAD name with an apostrophe is a rare thing
    oid = main_id.replace("'", "''")
    join = ("FROM has_ref AS h JOIN ref AS r ON r.oidbib = h.oidbibref "
            f"JOIN basic AS b ON b.oid = h.oidref WHERE b.main_id = '{oid}'")

    rows = _tap(
        'SELECT r.bibcode, r."year", r.journal, r.volume, r.page, r.title, '
        'r.doi, r.abstract, r.nbobject, h.obj_freq, h.ref_flag, h.ref_raw_id '
        + join)

    if not rows:
        return None

    authors = {}
    for row in _tap('SELECT r.bibcode, a.name, a.pos FROM author AS a '
                    'JOIN has_ref AS h ON h.oidbibref = a.oidbibref '
                    'JOIN ref AS r ON r.oidbib = a.oidbibref '
                    f'JOIN basic AS b ON b.oid = h.oidref WHERE b.main_id = \'{oid}\' '
                    f'AND a.pos <= {AUTHORS_SHOWN}'):
        authors.setdefault(row['bibcode'], []).append((row['pos'], row['name']))

    counts = {row['bibcode']: row['n'] for row in _tap(
        'SELECT r.bibcode, count(*) AS n FROM author AS a '
        'JOIN ref AS r ON r.oidbib = a.oidbibref '
        'JOIN has_ref AS h ON h.oidbibref = a.oidbibref '
        f'JOIN basic AS b ON b.oid = h.oidref WHERE b.main_id = \'{oid}\' '
        'GROUP BY r.bibcode')}

    for row in rows:
        row['authors'] = '; '.join(name for _, name in
                                   sorted(authors.get(row['bibcode'], [])))
        row['nauthors'] = counts.get(row['bibcode'], 0)

    return _table(rows)


def query_ads(bibcodes, token):
    """Citation counts, arXiv identifiers and data archives from ADS, by bibcode.

    The archives as ADS names them with how many datasets each holds -
    'MAST:1|ESO:8' - which is all a link to them needs: see ADS_GATEWAY.

    In bigquery requests of up to ADS_CHUNK bibcodes, which ADS allows a
    hundred of a day, against five thousand of the ordinary search. One per
    target, then, where the ordinary search would want one per paper, and the
    cache keeps it to one per target until it is refreshed.
    """
    result = {}

    for start in range(0, len(bibcodes), ADS_CHUNK):
        chunk = bibcodes[start:start + ADS_CHUNK]

        res = requests.post(
            ADS_BIGQUERY,
            params={'q': '*:*', 'fl': 'bibcode,citation_count,identifier,data',
                    'rows': len(chunk)},
            headers={'Authorization': f'Bearer {token}',
                     'Content-Type': 'big-query/csv'},
            data='bibcode\n' + '\n'.join(chunk), timeout=TIMEOUT)
        res.raise_for_status()

        for doc in res.json()['response']['docs']:
            arxiv = [_[6:] for _ in doc.get('identifier', [])
                     if _.startswith('arXiv:')]
            result[doc['bibcode']] = (doc.get('citation_count', 0),
                                      arxiv[0] if arxiv else '',
                                      '|'.join(doc.get('data', [])))

    return result


def acquire(main_id, safe_name, basepath, log, refresh=False):
    """Fetch the references to an object and write them for the pages to read.

    Returns the table, or None where SIMBAD has none or could not be asked.
    """
    with cached_votable_query(f"simbad_refs_{safe_name}.vot", basepath, log,
                              'SIMBAD references', refresh=refresh) as cache:
        if not cache.hit:
            try:
                refs = query_simbad(main_id)
            except Exception as e:
                log(f"Error: could not get the references from SIMBAD: {e}")
                return None

            if refs is not None:
                cache.save(refs)
            else:
                cache.save_empty()
        else:
            refs = _filled(cache.data)

    if refs is None or not len(refs):
        log("SIMBAD lists no references")
        return None

    token = getattr(settings, 'ADS_TOKEN', '')

    if token:
        with cached_votable_query(f"ads_refs_{safe_name}.vot", basepath, log,
                                  'ADS', refresh=refresh) as cache:
            # Written before the archives were asked for, and so without them
            if cache.hit and cache.data is not None and 'data' not in cache.data.colnames:
                log("The cached ADS reply has no data archives, asking again")
                cache.invalidate()

            if not cache.hit:
                try:
                    ads = query_ads([str(_) for _ in refs['bibcode']], token)
                    cache.save(Table({
                        'bibcode': np.array(list(ads.keys()), dtype=str),
                        'citations': np.array([_[0] for _ in ads.values()], dtype=int),
                        'arxiv': np.array([_[1] for _ in ads.values()], dtype=str),
                        'data': np.array([_[2] for _ in ads.values()], dtype=str),
                    }))
                except Exception as e:
                    log(f"Error: could not reach ADS: {e}")

            ads = cache.data

        if ads is not None and len(ads):
            ads = {str(row['bibcode']): row for row in _filled(ads)}
            known = [ads.get(str(_)) for _ in refs['bibcode']]

            # Whole columns rather than element by element: a column of empty
            # strings is one character wide, and would cut every identifier
            refs['citations'] = [int(row['citations']) if row is not None else -1
                                 for row in known]
            refs['arxiv'] = np.array([str(row['arxiv']) if row is not None else ''
                                      for row in known], dtype=str)
            refs['data'] = np.array([str(row['data']) if row is not None else ''
                                     for row in known], dtype=str)

    refs.write(os.path.join(basepath, REFERENCES_FILE), format='votable',
               overwrite=True)

    # The rest are a link away, on a page of their own; the log can name that
    # page by its route and leave the address of it to whatever renders it
    focused = np.sum((refs['ref_flag'] & FOCUS_FLAGS) > 0)
    log(f"\nReferences: [{len(refs)}](page:target_references), {focused} of them "
        "naming it in the title, abstract or keywords")

    log("Key papers:")
    for row in references(refs)[:KEY_PAPERS]:
        letters = ''.join(letter for letter, _ in row['flags'])
        # Italic, so any star in the title is escaped from ending it early
        title = row['title'].replace('*', '\\*')
        log(f"  [{row['citation']}]({ADS_ABSTRACT}{quote(row['bibcode'], safe='')}): "
            f"*{title}* [{letters},{row['obj_freq']}]")

    return refs


def relevance(refs):
    """How much each paper is about the object, highest first.

    Naming it in the title, abstract or keywords counts most, then how many
    times it is named, against how many other objects the paper is about - a
    survey catalogue names everything in it once. Citations, where ADS has
    supplied them, break the ties between papers equally about it.
    """
    focus = (np.asarray(refs['ref_flag']) & FOCUS_FLAGS) > 0
    freq = np.asarray(refs['obj_freq'])
    nobj = np.maximum(np.asarray(refs['nbobject']), 1)
    cites = np.maximum(np.asarray(refs['citations']), 0)

    return 3*focus + np.log2(1 + freq) - 0.5*np.log10(nobj) + 0.5*np.log10(1 + cites)


def plain(text):
    """A title or abstract without the TeX SIMBAD keeps them in.

    Only the font switches and the grouping they come in - '{\\em Gaia}' - which
    is most of it. Mathematics is left as written.
    """
    text = re.sub(r'\\(em|it|bf|rm|sc|tt)\b\s*', '', text)
    text = re.sub(r'(?<!\\)[{}]', '', text)

    return ' '.join(text.split())


def datasets(bibcode, data):
    """The archives holding data of a paper, as (archive, count, link) tuples."""
    result = []

    for entry in filter(None, data.split('|')):
        archive, _, count = entry.partition(':')

        if archive not in ADS_NOT_DATA:
            result.append((archive, int(count) if count.isdigit() else 0,
                           ADS_GATEWAY.format(bibcode=quote(bibcode, safe=''),
                                              archive=quote(archive, safe=''))))

    return result


def flags(ref_flag):
    """Where a paper names the object, as SIMBAD's letters and their meanings."""
    return [(letter, meaning) for letter, bit, meaning in REF_FLAGS
            if ref_flag & bit]


def references(refs):
    """The references as dicts for the templates, most relevant first."""
    score = relevance(refs)
    result = []

    for i in np.argsort(-score, kind='stable'):
        row = {name: refs[name][i].item() for name in refs.colnames}

        # A list written before a column was added has it at its filler
        for name, (_, filler) in COLUMNS.items():
            row.setdefault(name, filler)

        authors = row['authors'].split('; ') if row['authors'] else []
        first = authors[0].split()[0].title() if authors else ''

        row['ref_raw_id'] = ', '.join(row['ref_raw_id'].split(';'))
        row['title'] = plain(row['title'])
        row['abstract'] = plain(row['abstract'])
        row['score'] = float(score[i])
        row['flags'] = flags(row['ref_flag'])
        row['focus'] = bool(row['ref_flag'] & FOCUS_FLAGS)
        row['author_list'] = ', '.join(authors) + (
            ' et al.' if row['nauthors'] > len(authors) else '')
        row['citation'] = (f"{first}{' et al.' if row['nauthors'] > 1 else ''} "
                           f"{row['year']}").strip()
        row['datasets'] = datasets(row['bibcode'], row['data'])
        result.append(row)

    return result


def load(basepath):
    """The references written for a target, most relevant first, or None."""
    path = os.path.join(basepath, REFERENCES_FILE)

    if not os.path.exists(path):
        return None

    return references(_filled(Table.read(path)))
