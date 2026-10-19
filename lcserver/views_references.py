from django.http import Http404
from django.template.response import TemplateResponse
from django.shortcuts import get_object_or_404

from . import models
from .processing import references, vizier


def target_references(request, id):
    """Every paper SIMBAD links to the target, most relevant first."""
    target = get_object_or_404(models.Target, id=id)

    if not target.can_view(request.user):
        raise Http404

    refs = references.load(target.path()) or []

    return TemplateResponse(request, 'references.html', context={
        'target': target,
        'target_id': id,
        'references': refs,
        # Citations are ADS's, and there only where a key was set when the
        # list was fetched - without them the column would be all blanks
        'has_citations': any(_['citations'] >= 0 for _ in refs),
        'ref_flags': references.REF_FLAGS,
    })


def target_catalogues(request, id):
    """Every VizieR catalogue with an entry at the target, those of the papers
    about it first."""
    target = get_object_or_404(models.Target, id=id)

    if not target.can_view(request.user):
        raise Http404

    ra, dec = target.config.get('target_ra'), target.config.get('target_dec')
    cats, errors = vizier.load(target.path(), ra, dec) if ra is not None else (None, 0)

    return TemplateResponse(request, 'catalogues.html', context={
        'target': target,
        'target_id': id,
        'catalogues': cats or [],
        'cited': sum(1 for _ in cats or [] if _['reference']),
        'errors': errors,
        'sr': vizier.SR,
        'max_rows': vizier.ROWS,
    })
