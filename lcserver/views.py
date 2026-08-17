from django.http import HttpResponse, FileResponse, HttpResponseRedirect, JsonResponse, Http404
from django.template.response import TemplateResponse
from django.views.decorators.cache import cache_page
from django.views.decorators.vary import vary_on_cookie
from django.views.decorators.http import require_POST
from django.db.models import Q
from django.urls import reverse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.shortcuts import get_object_or_404

import os, io, glob, shutil, time, re

import mimetypes
import magic

import numpy as np
from astropy.table import Table
from astropy.time import Time
from astropy.io import fits

# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from . import forms
from . import models
from . import celery
from . import celery_tasks
from . import surveys

def index(request):
    context = {}

    context['form_new_target'] = forms.TargetNewForm()

    return TemplateResponse(request, 'index.html', context=context)


def sanitize_path(path):
    """Confine a user-supplied path to the folder it will be resolved against.

    Rejecting absolute paths is not enough on its own: Django's <path:path>
    converter hands '..' straight through, and a client that does not
    normalize the URL for itself - curl --path-as-is, or percent-encoded
    separators - delivers it to the view intact.
    """
    if not path or os.path.isabs(path):
        return ''

    # normpath collapses the '..' segments first, so anything still climbing
    # afterwards was really trying to leave the folder
    path = os.path.normpath(path)

    if path == os.pardir or path.startswith(os.pardir + os.sep) or os.path.isabs(path):
        return ''

    return path


def make_breadcrumb(path, base="Files"):
    """Create breadcrumb navigation from path"""
    parts = []

    if path:
        components = path.split(os.sep)
        accumulated = ""

        for component in components:
            if accumulated:
                accumulated = os.path.join(accumulated, component)
            else:
                accumulated = component

            parts.append({'path': accumulated, 'name': component})

    return [{'path': '.', 'name': base}] + parts


def list_files(request, path='', base=settings.TARGETS_PATH):
    """Browse files in a directory with support for viewing different file types"""
    context = {}

    path = sanitize_path(path)
    fullpath = os.path.join(base, path)

    context['path'] = path
    context['breadcrumb'] = make_breadcrumb(path, base="Files")

    if os.path.isfile(fullpath):
        # Display a file
        context['mime'] = magic.from_file(filename=fullpath, mime=True)
        context['magic_info'] = magic.from_file(filename=fullpath)
        context['stat'] = os.stat(fullpath)
        context['size'] = context['stat'].st_size
        context['time'] = Time(context['stat'].st_mtime, format='unix')

        context['mode'] = 'download'

        # VOTable/Parquet files
        if path.endswith('.vot') or path.endswith('.parquet'):
            try:
                context['table'] = Table.read(fullpath)
                context['mode'] = 'table'
            except:
                pass

        # FITS files
        elif 'fits' in context['mime'] or 'FITS' in context['magic_info'] or os.path.splitext(path)[1].lower().startswith('.fit'):
            context['mode'] = 'fits'

            try:
                hdus = fits.open(fullpath)
                context['fitsfile'] = hdus
            except:
                import traceback
                traceback.print_exc()
                pass

        # Text files
        elif 'text' in context['mime']:
            try:
                with open(fullpath, 'r') as f:
                    context['contents'] = f.read()
                context['mode'] = 'text'
            except:
                pass

        # Image files
        elif 'image' in context['mime']:
            context['mode'] = 'image'

        return TemplateResponse(request, 'files.html', context=context)

    elif os.path.isdir(fullpath):
        # List files in directory
        files = []

        for entry in os.scandir(fullpath):
            # Check for broken symlinks
            if not os.path.exists(os.path.join(fullpath, entry.name)):
                continue

            stat = entry.stat()

            elem = {
                'path': os.path.join(path, entry.name),
                'name': entry.name,
                'stat': stat,
                'size': stat.st_size,
                'time': Time(stat.st_mtime, format='unix'),
                'mime': mimetypes.guess_type(entry.name)[0],
                'is_dir': entry.is_dir(),
            }

            if elem['is_dir']:
                elem['type'] = 'dir'
            elif elem['mime'] and 'fits' in elem['mime']:
                elem['type'] = 'fits'
            elif os.path.splitext(entry.name)[1].lower().startswith('.fit'):
                elem['type'] = 'fits'
            elif elem['mime'] and 'image' in elem['mime']:
                elem['type'] = 'image'
            elif elem['mime'] and 'text' in elem['mime']:
                elem['type'] = 'text'
            else:
                elem['type'] = 'file'

            files.append(elem)

        files = sorted(files, key=lambda _: _.get('name'))

        # Add parent directory link if not at root
        if len(context['breadcrumb']) > 1:
            files = [{'path': os.path.dirname(path), 'name': '..', 'is_dir': True, 'type':'up'}] + files

        context['files'] = files
        context['mode'] = 'list'

        return TemplateResponse(request, 'files.html', context=context)

    return HttpResponse("Path not found", status=404)


def preview(request, path, width=None, minwidth=256, maxwidth=1024, base=settings.TARGETS_PATH):
    """Generate preview image for FITS files"""
    path = sanitize_path(path)
    fullpath = os.path.join(base, path)

    if not os.path.isfile(fullpath):
        return HttpResponse("File not found", status=404)

    # Try to open as FITS
    try:
        with fits.open(fullpath) as hdus:
            # Find first image HDU
            image = None
            for hdu in hdus:
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    image = hdu.data
                    break

            if image is None:
                return HttpResponse("No image data found in FITS", status=404)

            # Create figure
            fig = Figure(figsize=(8, 8))
            ax = fig.add_subplot(111)

            # Display image with auto-scaling
            from matplotlib.colors import Normalize
            import numpy as np

            # Compute percentile scaling
            vmin = np.percentile(image[np.isfinite(image)], 1)
            vmax = np.percentile(image[np.isfinite(image)], 99)

            ax.imshow(image, origin='lower', cmap='gray',
                     norm=Normalize(vmin=vmin, vmax=vmax))
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(os.path.basename(path))

            # Render to response
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)

            return HttpResponse(buf.getvalue(), content_type='image/png')

    except:
        import traceback
        traceback.print_exc()
        return HttpResponse("Error generating preview", status=500)


def filename_prefix(name):
    """Turn a target name into something that may lead a filename.

    Names arrive as the user typed them - 'RA=10.0 Dec=20.0', 'Gaia DR3 12345'
    - so everything outside a conservative set becomes an underscore, and runs
    of underscores collapse, to keep the result readable in a download folder.
    """
    prefix = re.sub(r'[^\w+.-]+', '_', name, flags=re.UNICODE).strip('_.')

    return prefix


def download(request, path, attachment=True, base=settings.TARGETS_PATH, prefix=None):
    path = sanitize_path(path)

    fullpath = os.path.join(base, path)

    if os.path.isfile(fullpath):
        # The files are stored under names that only say what they are, not
        # what they are of - so the target name leads the one the user gets
        filename = os.path.basename(path)
        if prefix and not filename.startswith(prefix + '_'):
            filename = prefix + '_' + filename

        return FileResponse(open(os.path.abspath(fullpath), 'rb'),
                            as_attachment=attachment, filename=filename)
    else:
        return HttpResponse("No such file", status=404)


def target_download(request, id=None, path='', **kwargs):
    target = get_object_or_404(models.Target, id=id)

    # These serve a target's own files, so they answer to the same rule as the
    # target page - which hides itself from everyone else, while these used to
    # hand the files to anyone who knew the URL
    if not target.can_view(request.user):
        raise Http404

    return download(request, path, base=target.path(),
                    prefix=filename_prefix(target.name), **kwargs)


# vary_on_cookie keeps the cache from answering one user out of another's
# entry: cache_page is consulted before the view runs, so without it a preview
# fetched by the owner would be served to whoever asked for the same URL next
@cache_page(15 * 60)
@vary_on_cookie
def target_preview(request, id=None, path='', **kwargs):
    target = get_object_or_404(models.Target, id=id)

    if not target.can_view(request.user):
        raise Http404

    return preview(request, path, base=target.path(), **kwargs)


@login_required
def target_files(request, id, path=''):
    """Browse files within a target folder using the generic file browser"""
    target = models.Target.objects.get(id=id)

    # Permission check: owner or staff
    if not target.can_view(request.user):
        return HttpResponse("You don't have permission to view this target", status=403)

    # Call generic list_files with target base path
    response = list_files(request, path=path, base=target.path())

    # Customize context for target-specific rendering
    if hasattr(response, 'context_data'):
        context = response.context_data
        context['target'] = target
        context['target_id'] = id
        context['is_target_browser'] = True
        context['user_may_delete'] = target.can_edit(request.user)

    return response


@login_required
@require_POST
def target_file_delete(request, id, path=''):
    """Delete one file or directory from a target folder."""
    target = get_object_or_404(models.Target, id=id)

    if not target.can_edit(request.user):
        return HttpResponse("You don't have permission to modify this target", status=403)

    if target.celery_id is not None:
        messages.warning(request, f"Target {id} is running, not touching its files")
        return HttpResponseRedirect(reverse('target_files', kwargs={'id': id}))

    base = target.path()
    path = sanitize_path(path)
    fullpath = os.path.join(base, path)

    # sanitize_path() drops absolute paths, this catches the rest - a symlink or
    # a .. that resolves outside the target folder
    if not path or not os.path.realpath(fullpath).startswith(os.path.realpath(base) + os.sep):
        messages.error(request, "Refusing to delete outside the target folder")
        return HttpResponseRedirect(reverse('target_files', kwargs={'id': id}))

    if not os.path.exists(fullpath):
        messages.warning(request, f"{path} does not exist")
    elif os.path.isdir(fullpath):
        shutil.rmtree(fullpath, ignore_errors=True)
        messages.success(request, f"Deleted directory {path}")
    else:
        os.unlink(fullpath)
        messages.success(request, f"Deleted {path}")

    # Back to the directory the file was in, rather than the root
    parent = os.path.dirname(path)
    return HttpResponseRedirect(
        reverse('target_files', kwargs={'id': id, 'path': parent}) if parent
        else reverse('target_files', kwargs={'id': id}))


def target_cache_entries(target):
    """Cache contents of a target, annotated with the source each entry serves."""
    entries = []
    cachepath = os.path.join(target.path(), 'cache')

    for fullpath in sorted(glob.glob(os.path.join(cachepath, '*'))):
        name = os.path.basename(fullpath)

        if os.path.isdir(fullpath):
            size = sum(os.path.getsize(os.path.join(root, f))
                       for root, _, files in os.walk(fullpath) for f in files)
        else:
            size = os.path.getsize(fullpath)

        source_id = surveys.cache_source_for(name)
        source = surveys.SURVEY_SOURCES.get(source_id) if source_id else None

        entries.append({
            'name': name,
            'path': os.path.join('cache', name),
            'size': size,
            'modified': Time(os.path.getmtime(fullpath), format='unix').datetime,
            'source_id': source_id,
            'source_name': source['short_name'] if source else 'Unknown',
            'is_dir': os.path.isdir(fullpath),
        })

    return entries


def targets(request, id=None):
    context = {}

    if id:
        target = get_object_or_404(models.Target, id=id)
        path = target.path()

        # The listing hides other people's targets through accessible_to(), and
        # every other endpoint asks can_view(), but this page used to show
        # itself to anyone who had the id
        if not target.can_view(request.user):
            raise Http404

        # Permissions
        context['user_may_submit'] = target.can_edit(request.user)

        # Clear the link to queued target if it was revoked
        if target.celery_id:
            task = celery.app.AsyncResult(target.celery_id)
            if task.state == 'REVOKED' or task.state == 'FAILURE':
                target.celery_id = None
                target.state = 'failed' # Should we do it?
                target.complete()
                target.save()

        # Prevent target operations if it is still running
        if target.celery_id is not None and request.method == 'POST':
            messages.warning(request, f"Task for target {id} is already running")
            return HttpResponseRedirect(request.path_info)

        all_forms = {}

        # Auto-generate forms from registry
        for source_id in surveys.SURVEY_SOURCES.keys():
            form_class = forms.get_survey_form(source_id)
            # Skip sources without forms (lightcurve-only sources)
            if form_class is None:
                continue
            # Special case for info form which includes name and title
            if source_id == 'info':
                params = target.config.copy()
                params.update({'name': target.name, 'title': target.title})
                all_forms[f'target_{source_id}'] = form_class(request.POST or None, initial=params)
            else:
                all_forms[f'target_{source_id}'] = form_class(request.POST or None, initial=target.config)

        for name,form in all_forms.items():
            context['form_'+name] = form

        # Also provide forms indexed by source_id for template loop
        survey_forms = {}
        for source_id in surveys.SURVEY_SOURCES.keys():
            form_name = f'target_{source_id}'
            if form_name in all_forms:
                survey_forms[source_id] = all_forms[form_name]
        context['survey_forms'] = survey_forms

        # Cache management, handled before the survey forms as these carry no
        # form of their own
        if request.method == 'POST' and request.POST.get('action') in ('delete_cache', 'clear_cache'):
            action = request.POST.get('action')

            if not target.can_edit(request.user):
                messages.error(request, "You don't have permission to modify this target")
            elif action == 'clear_cache':
                cachepath = os.path.join(path, 'cache')
                nentries = len(target_cache_entries(target))
                if os.path.exists(cachepath):
                    shutil.rmtree(cachepath, ignore_errors=True)
                messages.success(request, f"Cleared {nentries} cache entries")
            else:
                name = os.path.basename(request.POST.get('name', ''))
                entry = os.path.join(path, 'cache', name)
                # basename() above already confines this to the cache folder
                if name and os.path.exists(entry):
                    if os.path.isdir(entry):
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        os.unlink(entry)
                    messages.success(request, f"Dropped cached {name}")
                else:
                    messages.warning(request, f"No cached {name}")

            return HttpResponseRedirect(request.path_info)

        # Form actions
        if request.method == 'POST':
            # Handle forms
            form_type = request.POST.get('form_type')
            form = all_forms.get(form_type)
            if form and form.is_valid():
                if form.has_changed():
                    for name,value in form.cleaned_data.items():
                        # we do not want these to go to target.config
                        ignored_fields = [
                            'form_type',
                            'name', 'title',
                        ]
                        if name not in ignored_fields:
                            if name in form.changed_data or name not in target.config:
                                # update only changed or new fields
                                target.config[name] = value

                    target.save()

                # Handle actions
                action = request.POST.get('action')

                if action == 'delete_target':
                    if target.can_delete(request.user):
                        target.delete()
                        messages.success(request, f"Target {str(id )} is deleted")
                        return HttpResponseRedirect(reverse('targets'))
                    else:
                        messages.error(request, f"Cannot delete target {str(id)} belonging to {target.user.username}")
                        return HttpResponseRedirect(request.path_info)

                elif action == 'cleanup_target':
                    target.celery_id = celery_tasks.task_cleanup.delay(target.id).id
                    target.config = {} # should we reset the config on cleanup?..
                    target.source_states = {}
                    target.state = 'cleaning'
                    target.save()
                    messages.success(request, f"Started cleanup for target {target.id}")

                elif action == 'target_everything':
                    # Use run_target_steps for proper chain management
                    steps = surveys.get_survey_ids_for_everything()
                    celery_tasks.run_target_steps(target, steps)
                    messages.success(request, f"Started doing everything for target {target.id}")

                # Check if it's a survey source action
                elif action and action.startswith('target_'):
                    source_id = action.replace('target_', '')
                    survey_config = surveys.get_survey_source(source_id)

                    if survey_config:
                        # Special handling for 'info' action which updates name/title
                        if source_id == 'info':
                            if 'name' in form.changed_data and form.cleaned_data.get('name'):
                                target.name = form.cleaned_data.get('name')
                            if 'title' in form.changed_data:
                                target.title = form.cleaned_data.get('title')
                            target.save()

                        # Get task function and start it. The id is settled and
                        # recorded before the task is published, not after: the
                        # task asks on entry whether it still has a celery_id,
                        # to see whether it has been cancelled, and a worker
                        # can reach that question before a save that comes
                        # afterwards - reading none, and doing nothing at all.
                        signature = celery_tasks.get_survey_task(source_id).subtask(
                            args=[target.id])
                        target.celery_id = signature.freeze().id
                        target.state = survey_config['state_acquiring']
                        target.source_states = dict(target.source_states or {},
                                                    **{source_id: 'pending'})
                        target.save()

                        signature.apply_async()

                        messages.success(request,
                            f"Started getting {survey_config['short_name']} data for target {target.id}")
                    else:
                        messages.error(request, f"Unknown survey source: {source_id}")


                return HttpResponseRedirect(request.path_info)

        # Display target
        context['target'] = target
        context['survey_sources'] = surveys.get_all_survey_sources()

        # A chain is running, so every step in it will become active in turn.
        # Sections render their log placeholders upfront, which lets the state
        # poller update them in place instead of reloading on each transition.
        context['running_chain'] = bool(target.celery_id and target.celery_chain_ids)

        context['files'] = [os.path.split(_)[1] for _ in glob.glob(os.path.join(path, '*'))]

        # Cached replies from the surveys, so that they can be dropped one by one
        # Only offer the spectral viewer where there is something to view
        from .views_spectrum import has_spectra
        context['has_spectra'] = has_spectra(path)

        context['cache_entries'] = target_cache_entries(target)
        context['cache_size'] = sum(_['size'] for _ in context['cache_entries'])

        # Additional info

        return TemplateResponse(request, 'target.html', context=context)
    else:
        # List targets. The base queryset is already restricted to what the user
        # is allowed to see, so neither a forged 'show_all' nor a request method
        # that skips the filtering below can widen it.
        targets = models.Target.accessible_to(request.user)
        targets = targets.order_by('-created')

        # Filter form uses GET method
        filter_form = forms.TargetsFilterForm(
            request.GET,
            show_all=request.user.is_staff if request.user.is_authenticated else False,
        )
        context['form_filter'] = filter_form

        # New target form uses POST method
        new_target_form = forms.TargetNewForm(
            request.POST if request.method == 'POST' and request.POST.get('form_type') == 'new_target' else None
        )
        context['form_new_target'] = new_target_form

        # Handle GET filtering
        if request.method == 'GET':
            if filter_form.is_valid():
                # Narrow down to own targets unless show_all is checked. Users
                # who may not see others' targets are already limited by the
                # accessible_to() base queryset above.
                show_all = filter_form.cleaned_data.get('show_all')
                if not show_all and request.user.is_authenticated:
                    targets = targets.filter(user=request.user)

                # Text search filter
                query = filter_form.cleaned_data.get('query')
                if query:
                    targets = targets.filter(
                        Q(name__icontains=query) |
                        Q(title__icontains=query) |
                        Q(user__username__icontains=query) |
                        Q(user__first_name__icontains=query) |
                        Q(user__last_name__icontains=query)
                    )

        # Handle POST for new target creation
        if request.method == 'POST':
            if new_target_form.is_valid():
                target = models.Target(
                    title=new_target_form.cleaned_data.get('title'),
                    name=new_target_form.cleaned_data.get('name')
                )
                target.user = request.user
                target.state = 'created'
                target.save()  # to populate target.id
                messages.success(request, f"New target {target.id} created")

                # Let's immediately start collecting basic info for it
                target.celery_id = celery_tasks.task_info.delay(target.id).id
                target.state = 'acquiring info'
                target.save()
                messages.success(request, f"Started info collection for target {target.id}")

                return HttpResponseRedirect(reverse('targets', kwargs={'id': target.id}))

        context['targets'] = targets

    return TemplateResponse(request, 'targets.html', context=context)


def targets_actions(request):
    """Handle bulk operations on targets (cleanup, delete)."""
    form = forms.TargetsActionsForm(request.POST)

    if request.method == 'POST':
        if form.is_valid():
            target_ids = form.cleaned_data['targets']
            action = request.POST.get('action')

            for id in target_ids:
                target = get_object_or_404(models.Target, id=id)

                # Permission check: only owner or staff can perform actions
                if not target.can_edit(request.user):
                    messages.error(request, f"Cannot perform action on target {id} belonging to {target.user.username}")
                    continue

                if action == 'cleanup':
                    # Clear cache and output files
                    from . import processing
                    from .surveys import get_all_output_files

                    cleanup_files = get_all_output_files(cache=True)
                    processing.cleanup_paths(cleanup_files, basepath=target.path())

                    # Clear configuration
                    target.config = {}
                    target.state = 'created'
                    target.save()

                    messages.success(request, f"Cleaned up target {id}")

                elif action == 'delete':
                    if target.can_delete(request.user):
                        target.delete()
                        messages.success(request, f"Target {id} is deleted")
                    else:
                        messages.error(request, f"Cannot delete target {id} belonging to {target.user.username}")

            return HttpResponseRedirect(form.cleaned_data['referer'])

    return HttpResponseRedirect(reverse('targets'))


# How long after a log stops growing it is still worth sending. The page asks
# every three seconds, so this covers the lines a step writes between the last
# poll it was running for and the moment it finished - which otherwise waited
# for the reload at the end of the whole run.
LIVE_LOG_SECONDS = 30


def live_logs(target):
    """The log files worth sending to a page that is watching this run.

    Read from source_states rather than from the target's own state: that field
    can only name one step, and the sources no longer take turns - four of them
    run at once, and the last one to write the state is not the only one with
    something to say. A step that has just finished is included too, so that the
    end of its log arrives rather than waiting for the reload.
    """
    now = time.time()
    files = []

    for source_id, state in (target.source_states or {}).items():
        name = (surveys.SURVEY_SOURCES.get(source_id) or {}).get('log_file')

        if not name or state == 'pending':
            continue

        try:
            recent = now - os.path.getmtime(os.path.join(target.path(), name))
        except OSError:
            continue

        if state == 'running' or recent < LIVE_LOG_SECONDS:
            files.append(name)

    return files


def target_state(request, id):
    """AJAX endpoint to get current target state."""
    target = get_object_or_404(models.Target, id=id)

    # Permission check
    if not target.can_view(request.user):
        return JsonResponse({'error': 'Permission denied'}, status=403)

    # Refresh from database to get latest state
    target.refresh_from_db()

    result = {
        'state': target.state,
        'id': target.id,
        'celery_id': target.celery_id,
        # How each source stands, so the sections can show it without waiting
        # for the reload at the end of a run
        'source_states': surveys.get_source_states(target),
    }

    # While running, also return the freshly-rendered logs of whatever is
    # writing, so the page can update them in place without a full reload.
    if target.celery_id:
        from .templatetags.tags import target_file_contents
        result['logs'] = {name: target_file_contents(target, name, highlight=True)
                          for name in live_logs(target)}

    return JsonResponse(result)


@login_required
def profile(request):
    """User profile page with account info and statistics."""
    # Get user's target statistics
    user_targets = models.Target.objects.filter(user=request.user)

    target_count = user_targets.count()
    completed_count = user_targets.filter(
        state__in=['info acquired', 'combined acquired', 'ZTF acquired',
                   'ASAS acquired', 'TESS acquired', 'DASCH acquired',
                   'APPLAUSE acquired', 'PTF acquired', 'CSS acquired',
                   'KWS acquired', 'MMT9 acquired']
    ).count()
    # Contains rather than equals: a chain that ran to the end reports how
    # many of its steps failed, rather than the bare 'failed' of a single step
    failed_count = user_targets.filter(state__contains='failed').count()

    context = {
        'target_count': target_count,
        'completed_count': completed_count,
        'failed_count': failed_count,
    }

    return TemplateResponse(request, 'profile.html', context=context)
