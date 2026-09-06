"""Where the data comes from.

A page of prose about other people's archives and other people's models, and
the one page here that is not built from anything computed. It is built from
the registry all the same: every entry on it is the `about`, `about_links` and
`acknowledgement` a source declares in its own `@survey_source`, so a source
added tomorrow appears here without anyone remembering to come and write it in,
and a source removed stops being described. A hand-written list would be wrong
within a month.

An archive asking to be acknowledged has its wording quoted on its own card
and nowhere else. A second copy gathered at the foot of the page was one copy
too many: the text is already in front of whoever is reading about that
archive, and a list of the same paragraphs invites citing an archive the
result never rested on.

The model grids are the same idea over a different registry: the directory of
cubes rather than the source registry, and each grid described by the
attributes its own file carries. A grid that says too little to be credited
properly is left off, which is a thing to fix in the file rather than here.
"""

from django.template.response import TemplateResponse

from . import surveys
from .processing.sedfit import described_grids
from .processing.utils import SourceError


def about(request):
    """The archives this server fetches from, and the models it fits them with."""
    groups = surveys.get_about_entries()

    # The other half of what is not ours: the observations came from the
    # archives above, and what is fitted to them came from the groups who
    # computed these. Built the same way - from what each grid file says about
    # itself - and silently empty if no grid directory is configured, which is
    # the right answer for a server that cannot fit an SED at all.
    try:
        grids = described_grids()
    except SourceError:
        grids = []

    context = {
        'groups': groups,
        # Counted here rather than in the template, where {{ }} cannot add up
        'nsources': sum(len(_['entries']) for _ in groups),
        'grids': grids,
    }

    return TemplateResponse(request, 'about.html', context)
