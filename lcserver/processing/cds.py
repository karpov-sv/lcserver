"""SIMBAD and VizieR, through whichever of their servers is answering.

Both are run by CDS in Strasbourg and mirrored elsewhere, and when Strasbourg
is down every step here that reads a catalogue goes down with it - the info
step first of all, which every other step needs. So every query to either goes
through here, tries the servers in turn, and returns what the first to answer
said.

A mirror is not always the whole of the original. IDIA's VizieR lacks some of
the largest catalogues - Pan-STARRS, Gaia DR2 and EDR3, ATLAS-RefCat2 - and
says so only in an INFO of an error inside an otherwise ordinary empty table,
which astroquery reads as no rows. That is caught here and taken as the
server failing, so that a catalogue a mirror does not hold is never mistaken
for one with nothing at the position and cached as such.

What failed is remembered, so that the dozen queries of one step - and the
steps of the targets after it, in the same worker - do not each wait out the
same timeouts in turn. A server that cannot be reached at all is passed over
for every query for DEAD_FOR seconds. One that can be reached but fails a
query - refuses it, answers 5xx, or accepts it and then says nothing - is
passed over for that query alone, for FAILED_FOR seconds. A query every server
is being passed over for fails at once, rather than waiting through all of them
again.
"""

import re
import time
from urllib.parse import urlparse

import numpy as np
import requests

from astropy.coordinates import SkyCoord
from astropy import units as u

from .utils import SourceError, _TimeoutSession


# In the order they are tried, with the scheme each answers on. CfA is a full
# mirror, but has been seen to answer 502 while Strasbourg is down; IDIA lacks
# some of the largest catalogues, so it goes last.
#
# Not NAO's (http://vizier.nao.ac.jp), which answers over http only, and has
# been seen to answer the small catalogues IDIA also has and hang on the large
# ones it lacks - adding nothing but the wait. Nor China's
# (http://vizier.china-vo.org), which has been seen to return the right rows
# flagged as an error - its own statistics table being broken - and time out
# on the large catalogues.
VIZIER_SERVERS = ['https://vizier.cds.unistra.fr',
                  'https://vizier.cfa.harvard.edu',
                  'https://vizier.idia.ac.za']
SIMBAD_SERVERS = ['https://simbad.cds.unistra.fr', 'https://simbad.harvard.edu']

# Connect, then read. The connection timeout is what a dead server costs, so
# it is short; the read timeout is what one that accepts and then hangs costs,
# and a cone search of one catalogue takes a second or two on any server that
# is going to answer it at all.
TIMEOUT = (10, 30)

# How long, in seconds, a server that could not be reached is passed over for
# everything, and one that failed a query is passed over for that query
DEAD_FOR = 300
FAILED_FOR = 1800

# The columns stdpipe asks for of every catalogue, whatever else is wanted
BASE_COLUMNS = ['*', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000']

# server: when it could not be reached; (server, query): when it failed that
_dead = {}
_failed = {}


class Refused(Exception):
    """A server answered, but not with an answer - a mirror without the
    catalogue, or a query it would not run. A reason to pass it over for that
    query, not for the next, which may be one it can answer."""


def _host(server):
    """A server's host name, which is how astroquery's SIMBAD and the log
    name it."""
    return urlparse(server).netloc


def _unreachable(e):
    """Whether an exception is a server not reachable at all, rather than one
    failing this query.

    Not every ConnectionError of requests: a server that drops the connection
    halfway through a query it cannot answer raises one too. Only a connection
    never made - timed out, refused, or to a name that does not resolve - is
    looked for, along the chain, as pyvo wraps what requests raised in its own
    and urllib3 keeps the reason apart.
    """
    from urllib3.exceptions import NewConnectionError, ConnectTimeoutError

    while e is not None:
        if isinstance(e, (requests.ConnectTimeout, NewConnectionError,
                          ConnectTimeoutError)):
            return True

        # pyvo's own reason is a string, urllib3's an exception
        e = next((_ for _ in (getattr(e, 'cause', None), getattr(e, 'reason', None),
                              e.__cause__, e.__context__)
                  if isinstance(_, BaseException)), None)

    return False


def _skipped(server, key, now):
    """Why a server is passed over for a query, or None if it is not."""
    if now - _dead.get(server, 0) < DEAD_FOR:
        return "not reachable"

    if key is not None and now - _failed.get((server, key), 0) < FAILED_FOR:
        return "failed it recently"


def ask(servers, call, what, log=None, key=None):
    """What `call(server)` returns for the first of `servers` to answer.

    `key` names the query - the catalogue, usually - for remembering which
    server failed it; without one, only servers not reachable at all are
    remembered. Raises SourceError if none answers, so that a failure never
    reaches a cache. An answer from any but the first server is logged, as the
    data then came from a mirror.
    """
    errors = []

    for server in servers:
        now = time.time()
        reason = _skipped(server, key, now)

        if reason:
            errors.append(f"{_host(server)}: {reason}")
            continue

        try:
            result = call(server)
        except Exception as e:
            if _unreachable(e):
                _dead[server] = now
                errors.append(f"{_host(server)}: not reachable")
            else:
                if key is not None:
                    _failed[(server, key)] = now
                errors.append(f"{_host(server)}: " + (
                    str(e) if isinstance(e, Refused)
                    else f"{type(e).__name__}: {str(e)[:200]}"))
        else:
            _dead.pop(server, None)

            if log and server != servers[0]:
                log(f"  {what} from the mirror at {_host(server)}, as "
                    f"{_host(servers[0])} is not answering")

            return result

    raise SourceError(f"no {what} server answered - " + '; '.join(errors))


def _info(content, name, value=None):
    """The INFO elements of a VOTable with a given name, and value if given,
    whatever the order of their attributes and whether or not they are closed
    on themselves - as (value, text) pairs."""
    out = []

    for m in re.finditer(rb'<INFO\b([^>]*?)(/>|>(.*?)</INFO>)', content, re.S):
        attrs = dict(re.findall(rb'(\w+)="([^"]*)"', m.group(1)))

        if attrs.get(b'name') == name and (value is None or attrs.get(b'value') == value):
            out.append((attrs.get(b'value', b'').decode(errors='replace'),
                        (m.group(3) or b'').decode(errors='replace')))

    return out


def _refusal(content):
    """Why a VizieR or TAP reply says it failed, or None if it does not.

    Either way of saying it counts: a QUERY_STATUS of ERROR, or INFOs named
    Error without one - which is how a mirror answers for a catalogue it does
    not know, and how CfA has been seen to answer for ones it lacks the data
    of. Unless the reply is also an OVERFLOW: a result cut at the row limit
    holds the rows asked for, and IDIA reports the cut as an error as well.
    """
    if _info(content, b'QUERY_STATUS', b'OVERFLOW'):
        return None

    status = _info(content, b'QUERY_STATUS', b'ERROR')
    errors = [value for value, _ in _info(content, b'Error')]

    if status or errors:
        text = ' '.join(dict.fromkeys(errors + [text for _, text in status]))
        return ' '.join(text.split())[:300] or 'ERROR'


# VizieR

def _vizier(columns, row_limit, column_filters):
    from astroquery.vizier import Vizier

    return Vizier(columns=list(columns), row_limit=row_limit,
                  column_filters=column_filters or {})


def _vizier_tables(vizier, server, payload):
    """The tables a server returns for a query astroquery has written, or an
    exception saying it returned none.

    Sent here rather than by astroquery, which puts https:// before any server
    it is given, and so cannot reach one that answers over http alone.
    """
    from astroquery.exceptions import TableParseError

    response = requests.post(f"{server}/viz-bin/votable", data=payload,
                             timeout=TIMEOUT)
    response.raise_for_status()

    reason = _refusal(response.content)
    if reason:
        raise Refused(reason)

    tables = vizier._parse_result(response)

    if isinstance(tables, str):
        raise TableParseError("VizieR returned something that is not a VOTable")

    return tables


def vizier_region(ra, dec, sr, catalog, columns=('**',), row_limit=-1,
                  column_filters=None, log=None):
    """The rows of one VizieR table within `sr` arcsec of a position.

    None where there are none; SourceError where no server could be asked.
    """
    coord = SkyCoord(ra, dec, unit='deg')

    vizier = _vizier(columns, row_limit, column_filters)
    payload = vizier.query_region_async(coord, radius=sr*u.arcsec, catalog=catalog,
                                        get_query_payload=True)

    def call(server):
        return _vizier_tables(vizier, server, payload)

    tables = ask(VIZIER_SERVERS, call, 'VizieR', log=log, key=catalog)

    return tables[0] if len(tables) and len(tables[0]) else None


def vizier_catalog(catalog, columns=('**',), row_limit=-1, log=None):
    """The whole of one VizieR table, for those small enough to take whole."""
    vizier = _vizier(columns, row_limit, None)
    payload = vizier.get_catalogs_async(catalog, get_query_payload=True)

    def call(server):
        tables = _vizier_tables(vizier, server, payload)

        if not len(tables):
            raise Refused(f"no table {catalog}")

        return tables[0]

    return ask(VIZIER_SERVERS, call, 'VizieR', log=log, key=catalog)


def vizier_cat(ra, dec, sr, catalog, extra=(), get_distance=False,
               augment_bands=True, log=None):
    """stdpipe's get_cat_vizier, the same columns and bands, with `sr` in
    arcsec - and SourceError rather than None when VizieR could not be asked,
    which stdpipe does not tell apart from finding nothing.
    """
    from stdpipe import catalogs, astrometry

    known = catalogs.catalogs.get(catalog)

    if known:
        columns = BASE_COLUMNS + list(extra) + known.get('extra', [])
        catalog_id = known['vizier']
    else:
        columns = BASE_COLUMNS + list(extra)
        catalog_id = catalog

    cat = vizier_region(ra, dec, sr, catalog_id, columns=columns, log=log)

    if cat is None:
        return None

    cat.meta['vizier_id'] = catalog_id
    cat.meta['name'] = known['name'] if known else catalog

    if ('_RAJ2000' in cat.colnames and '_DEJ2000' in cat.colnames
        and 'RAJ2000' not in cat.colnames):
        cat.rename_columns(['_RAJ2000', '_DEJ2000'], ['RAJ2000', 'DEJ2000'])

    if get_distance and 'RAJ2000' in cat.colnames and 'DEJ2000' in cat.colnames:
        # In degrees, as stdpipe has it
        cat['_r'] = astrometry.spherical_distance(ra, dec, cat['RAJ2000'],
                                                  cat['DEJ2000'])

    if augment_bands:
        catalogs.augment_cat_bands(cat, catalog=catalog if known else None)

    return cat


def vizier_post(path, data, timeout=TIMEOUT, log=None):
    """The reply of a VizieR service at `path` - 'viz-bin/votable' - to a POST,
    and the server that gave it."""
    def call(server):
        res = requests.post(f"{server}/{path}", data=data, timeout=timeout)
        res.raise_for_status()

        return res.content, server

    return ask(VIZIER_SERVERS, call, 'VizieR', log=log, key=path)


# SIMBAD

def _simbad(server, fields=()):
    from astroquery.simbad import Simbad

    sim = Simbad()
    sim.server = _host(server)
    # As irsa_client does, and for the same reason: pyvo is handed a session
    # that carries no timeout, and a server accepting connections and saying
    # nothing would otherwise hold the step until Celery kills it
    sim._session = _TimeoutSession(TIMEOUT)

    if fields:
        sim.add_votable_fields(*fields)

    return sim


def simbad_region(coord, radius, fields=(), log=None):
    """SIMBAD's objects within `radius` of a position, or None."""
    res = ask(SIMBAD_SERVERS,
              lambda server: _simbad(server, fields).query_region(coord, radius=radius),
              # Keyed, as pyvo does not say whether a server could be reached
              'SIMBAD', log=log, key='query_region')

    return res if res is not None and len(res) else None


def simbad_object(name, log=None):
    """SIMBAD's entry for a name, or None if it does not know it."""
    res = ask(SIMBAD_SERVERS,
              lambda server: _simbad(server).query_object(name),
              'SIMBAD', log=log, key='query_object')

    return res if res is not None and len(res) else None


def simbad_tap(query, fmt='json', timeout=TIMEOUT, log=None):
    """The reply of SIMBAD's TAP service to an ADQL query, as bytes."""
    def call(server):
        res = requests.post(f"{server}/simbad/sim-tap/sync", data={
            'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': fmt,
            'MAXREC': 1000000, 'query': query}, timeout=timeout)

        reason = _refusal(res.content)
        if reason:
            raise Refused(reason)

        res.raise_for_status()

        return res.content

    return ask(SIMBAD_SERVERS, call, 'SIMBAD', log=log)


def resolve(name, log=None):
    """A name or coordinates as a position, or None.

    stdpipe's resolver first, which knows coordinates, SIMBAD, TNS and Sesame;
    then SIMBAD on its mirrors, as stdpipe asks it only through Sesame, whose
    every server has been seen down along with Strasbourg.
    """
    from stdpipe import resolve as resolve_

    target = resolve_.resolve(name)

    if target is None:
        try:
            res = simbad_object(name, log=log)
        except SourceError:
            res = None

        if res is not None and np.isfinite(float(res['ra'][0])):
            target = SkyCoord(float(res['ra'][0]), float(res['dec'][0]), unit='deg')

    return target
