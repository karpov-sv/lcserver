from django.http import HttpResponse, FileResponse, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.views.decorators.cache import cache_page
from django.db.models import Q
from django.urls import reverse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.conf import settings

import os, io, glob
import shutil
import json

import mimetypes
import magic

import numpy as np
from astropy.table import Table
from astropy.time import Time
from astropy.io import fits

# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from celery import chain

from . import forms
from . import models
from . import celery
from . import celery_tasks

def index(request):
    context = {}

    context['form_new_target'] = forms.TargetNewForm()

    return TemplateResponse(request, 'index.html', context=context)


def sanitize_path(path):
    # Prevent escaping from parent folder
    if not path or os.path.isabs(path):
        path = ''

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


def download(request, path, attachment=True, base=settings.TARGETS_PATH):
    path = sanitize_path(path)

    fullpath = os.path.join(base, path)

    if os.path.isfile(fullpath):
        return FileResponse(open(os.path.abspath(fullpath), 'rb'), as_attachment=attachment)
    else:
        return HttpResponse("No such file", status=404)


def target_download(request, id=None, path='', **kwargs):
    target = models.Target.objects.get(id=id)

    return download(request, path, base=target.path(), **kwargs)


@cache_page(15 * 60)
def target_preview(request, id=None, path='', **kwargs):
    target = models.Target.objects.get(id=id)

    return preview(request, path, base=target.path(), **kwargs)


@login_required
def target_files(request, id, path=''):
    """Browse files within a target folder using the generic file browser"""
    target = models.Target.objects.get(id=id)

    # Permission check: owner or staff
    if not request.user.is_authenticated or not (
        request.user.is_staff or request.user == target.user
    ):
        return HttpResponse("You don't have permission to view this target", status=403)

    # Call generic list_files with target base path
    response = list_files(request, path=path, base=target.path())

    # Customize context for target-specific rendering
    if hasattr(response, 'context_data'):
        context = response.context_data
        context['target'] = target
        context['target_id'] = id
        context['is_target_browser'] = True

    return response


def targets(request, id=None):
    context = {}

    if id:
        target = models.Target.objects.get(id=id)
        path = target.path()

        # Permissions
        if request.user.is_authenticated and (request.user.is_staff or request.user == target.user):
            context['user_may_submit'] = True
        else:
            context['user_may_submit'] = False

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

        params = target.config.copy()
        params.update({'name':target.name, 'title':target.title})
        all_forms['target_info'] = forms.TargetInfoForm(request.POST or None, initial = params)

        all_forms['target_ztf'] = forms.TargetZTFForm(request.POST or None, initial = target.config)
        all_forms['target_asas'] = forms.TargetASASForm(request.POST or None, initial = target.config)
        all_forms['target_tess'] = forms.TargetTESSForm(request.POST or None, initial = target.config)
        all_forms['target_dasch'] = forms.TargetDASCHForm(request.POST or None, initial = target.config)
        all_forms['target_applause'] = forms.TargetAPPLAUSEForm(request.POST or None, initial = target.config)
        all_forms['target_combined'] = forms.TargetCombinedForm(request.POST or None, initial = target.config)

        for name,form in all_forms.items():
            context['form_'+name] = form

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
                    if request.user.is_staff or request.user == target.user:
                        target.delete()
                        messages.success(request, f"Target {str(id )} is deleted")
                        return HttpResponseRedirect(reverse('targets'))
                    else:
                        messages.error(request, f"Cannot delete target {str(id)} belonging to {target.user.username}")
                        return HttpResponseRedirect(request.path_info)

                if action == 'cleanup_target':
                    target.celery_id = celery_tasks.task_cleanup.delay(target.id).id
                    target.config = {} # should we reset the config on cleanup?..
                    target.state = 'cleaning'
                    target.save()
                    messages.success(request, f"Started cleanup for target {target.id}")

                if action == 'target_info':
                    # Only accept non-empty new values
                    if 'name' in form.changed_data and form.cleaned_data.get('name'):
                        target.name = form.cleaned_data.get('name')
                    if 'title' in form.changed_data:
                        target.title = form.cleaned_data.get('title')

                    target.save()

                    target.celery_id = celery_tasks.task_info.delay(target.id).id
                    target.state = 'acquiring info'
                    target.save()
                    messages.success(request, f"Started info collection for target {target.id}")

                if action == 'target_ztf':
                    target.celery_id = celery_tasks.task_ztf.delay(target.id).id
                    target.state = 'acquiring ZTF lightcurve'
                    target.save()
                    messages.success(request, f"Started getting ZTF lightcurve for target {target.id}")

                if action == 'target_asas':
                    target.celery_id = celery_tasks.task_asas.delay(target.id).id
                    target.state = 'acquiring ASAS-SN lightcurve'
                    target.save()
                    messages.success(request, f"Started getting ASAS-SN lightcurve for target {target.id}")

                if action == 'target_tess':
                    target.celery_id = celery_tasks.task_tess.delay(target.id).id
                    target.state = 'acquiring TESS lightcurves'
                    target.save()
                    messages.success(request, f"Started getting TESS lightcurves for target {target.id}")

                if action == 'target_dasch':
                    target.celery_id = celery_tasks.task_dasch.delay(target.id).id
                    target.state = 'acquiring DASCH lightcurve'
                    target.save()
                    messages.success(request, f"Started getting DASCH lightcurve for target {target.id}")

                if action == 'target_applause':
                    target.celery_id = celery_tasks.task_applause.delay(target.id).id
                    target.state = 'acquiring APPLAUSE lightcurve'
                    target.save()
                    messages.success(request, f"Started getting APPLAUSE lightcurve for target {target.id}")

                if action == 'target_combined':
                    target.celery_id = celery_tasks.task_combined.delay(target.id).id
                    target.state = 'acquiring combined lightcurve'
                    target.save()
                    messages.success(request, f"Started getting combined lightcurve for target {target.id}")

                if action == 'target_everything':
                    target.celery_id = chain(
                        celery_tasks.task_info.subtask(args=[target.id], immutable=True),
                        celery_tasks.task_ztf.subtask(args=[target.id], immutable=True),
                        celery_tasks.task_asas.subtask(args=[target.id], immutable=True),
                        celery_tasks.task_tess.subtask(args=[target.id], immutable=True),
                        celery_tasks.task_dasch.subtask(args=[target.id], immutable=True),
                        celery_tasks.task_applause.subtask(args=[target.id], immutable=True),
                        celery_tasks.task_combined.subtask(args=[target.id], immutable=True),
                    ).apply_async().id

                    target.state = 'acquiring all possible data'
                    target.save()
                    messages.success(request, f"Started doing everything for target {target.id}")


                return HttpResponseRedirect(request.path_info)

        # Display target
        context['target'] = target

        context['files'] = [os.path.split(_)[1] for _ in glob.glob(os.path.join(path, '*'))]

        # Additional info

        return TemplateResponse(request, 'target.html', context=context)
    else:
        # List all targets
        targets = models.Target.objects.all()
        targets = targets.order_by('-created')

        all_forms = {}

        all_forms['new_target'] = forms.TargetNewForm(request.POST or None)
        all_forms['filter'] = forms.TargetsFilterForm(request.POST or None)

        for name,form in all_forms.items():
            context['form_'+name] = form

        if request.method == 'POST':
            # Handle forms
            form_type = request.POST.get('form_type')
            form = all_forms.get(form_type)

            if form and form.is_valid():
                if form_type == 'new_target':
                    target = models.Target(title=form.cleaned_data.get('title'), name=form.cleaned_data.get('name'))
                    target.user = request.user
                    target.state = 'created'
                    target.save() # to populate target.id
                    messages.success(request, f"New target {target.id} created")

                    # Let's immediately start collecting basic info for it
                    target.celery_id = celery_tasks.task_info.delay(target.id).id
                    target.state = 'acquiring info'
                    target.save()
                    messages.success(request, f"Started info collection for target {target.id}")

                    return HttpResponseRedirect(reverse('targets', kwargs={'id': target.id}))

                elif form_type == 'filter':
                    if form.is_valid():
                        query = form.cleaned_data.get('query')
                        if query:
                            targets = targets.filter(Q(name__icontains = query) |
                                                 Q(title__icontains = query) |
                                                 Q(user__username__icontains = query) |
                                                 Q(user__first_name__icontains = query) |
                                                 Q(user__last_name__icontains = query)
                                                 )

        context['targets'] = targets

    return TemplateResponse(request, 'targets.html', context=context)
