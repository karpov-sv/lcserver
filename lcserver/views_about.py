"""Where the data comes from.

A page of prose about other people's archives, and the one page here that is
not built from anything computed. It is built from the registry all the same:
every entry on it is the `about`, `about_links` and `acknowledgement` a source
declares in its own `@survey_source`, so a source added tomorrow appears here
without anyone remembering to come and write it in, and a source removed stops
being described. A hand-written list would be wrong within a month.

An archive asking to be acknowledged has its wording quoted on its own card
and nowhere else. A second copy gathered at the foot of the page was one copy
too many: the text is already in front of whoever is reading about that
archive, and a list of the same paragraphs invites citing an archive the
result never rested on.
"""

from django.template.response import TemplateResponse

from . import surveys


def about(request):
    """The archives this server fetches from."""
    groups = surveys.get_about_entries()

    context = {
        'groups': groups,
        # Counted here rather than in the template, where {{ }} cannot add up
        'nsources': sum(len(_['entries']) for _ in groups),
    }

    return TemplateResponse(request, 'about.html', context)
