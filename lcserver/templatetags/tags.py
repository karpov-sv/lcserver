from django import template
from django.template.defaultfilters import stringfilter
from django.utils.safestring import mark_safe
from django.utils.html import escape
from django.urls import reverse
from django.conf import settings

import os
import uuid
import re

from functools import partial

from astropy.io import fits

register = template.Library()


def target_file_link(m, target=None):
    name = m.group(2)
    url = reverse('target_files', kwargs={'id':target.id, 'path':name})

    return r"<a href='" + url + "'>" + name + r"</a>"


def target_cache_link(m, target=None):
    name = m.group(2)
    url = reverse('target_files', kwargs={'id':target.id, 'path':os.path.join('cache', name)})

    return r"<a href='" + url + "'>" + name + r"</a>"


@register.simple_tag
def target_file_contents(target, filename, highlight=False):
    path = os.path.join(target.path(), filename)

    contents = ""

    try:
        with open(path, "r") as f:
            contents = f.read()
    except:
        pass

    contents = escape(contents)

    if highlight:
        # Highlight some patterns in the text
        contents = re.sub(r"^(----\s+(.+)+\s+----)$",
                          r"<span class='text-primary'>\1</span>",
                          contents, flags=re.MULTILINE)

        contents = re.sub(r"^(RuntimeError:)(.*)$",
                          r"\1<span class='text-danger fw-bold'>\2</span>",
                          contents, flags=re.MULTILINE)

        # Both a Python exception name from a traceback and the bare Error: the
        # sources write themselves - hence the star rather than a plus
        contents = re.sub(r"^(\S*Error:)(.*)$",
                          r"\1<span class='text-danger'>\2</span>",
                          contents, flags=re.MULTILINE)

        contents = re.sub(r"^(Warning:)(.*)$",
                          r"\1<span class='text-warning'>\2</span>",
                          contents, flags=re.MULTILINE)

        # A plus is in the character class because survey identifiers carry
        # one: an APOGEE spectrum is named for its 2MASS designation, and half
        # of those are a declination that is not negative. Without it the name
        # stops at the plus, fails to reach an extension, and the whole link
        # goes unrecognised rather than merely being cut short.
        contents = re.sub(r"\b(file:([\w.+-]+\.\w+))\b",
                          partial(target_file_link, target=target),
                          contents, flags=re.MULTILINE)

        contents = re.sub(r"\b(cache:([\w.+-]+\.\w+))\b",
                          partial(target_cache_link, target=target),
                          contents, flags=re.MULTILINE)

        # Last, so that it only ever sees the text: the links put in above are
        # relative, and carry no scheme for this to match inside. The text is
        # escaped by now, so an ampersand in a query string arrives as &amp; -
        # which is what an href should hold anyway - and the quotes and angle
        # brackets ending the match cannot occur inside a URL any more. The
        # last character may not be punctuation, so that a URL at the end of a
        # sentence, or in brackets, does not swallow what closes it.
        contents = re.sub(r"""(https?://[^\s<>"']+[^\s<>"'.,;:!?)\]}])""",
                          r"<a href='\1' target='_blank' rel='noopener'>\1</a>",
                          contents, flags=re.MULTILINE)

    return mark_safe(contents)


@register.simple_tag
def target_fits_header(target, filename):
    path = os.path.join(target.path(), filename)

    contents = ""

    try:
        header = fits.getheader(path)
        contents = header.tostring('\n')
    except:
        contents = "Cannot get FITS header from " + filename

    return contents


@register.simple_tag
def make_uuid():
    return str(uuid.uuid1())


# Code borrowed from https://stackoverflow.com/a/3715794
@register.tag('make_list')
def make_list(parser, token):
    bits = token.contents.split()
    if len(bits) >= 4 and bits[-2] == "as":
        varname = bits[-1]
        items = bits[1:-2]
        return MakeListNode(items, varname)
    else:
        raise template.TemplateSyntaxError("%r expected format is 'item [item ...] as varname'" % bits[0])


class MakeListNode(template.Node):
    def __init__(self, items, varname):
        self.items = items
        self.varname = varname

    def render(self, context):
        items = map(template.Variable, self.items)
        context[self.varname] = [ i.resolve(context) for i in items ]
        return ""


@register.simple_tag
def free_disk_space():
    s = os.statvfs(settings.TARGETS_PATH)
    return s.f_bavail*s.f_frsize
