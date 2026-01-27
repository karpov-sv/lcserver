from django.http import JsonResponse, HttpResponseRedirect
from django.template.response import TemplateResponse

from django.contrib.auth.decorators import login_required

from django.contrib import messages

from . import models
from . import celery
from .celery_tasks import kill_task_processes


def find_target_by_chain_id(chain_id):
    """Find a Target that has the given chain_id in its celery_chain_ids list.
    SQLite doesn't support JSONField __contains lookup, so we filter in Python."""
    for target in models.Target.objects.exclude(celery_chain_ids=[]):
        if chain_id in target.celery_chain_ids:
            return target
    return None


def revoke_task_chain(target):
    """
    Revoke all tasks in a chain cooperatively (thread pool safe).
    Also clears the target's celery_id to signal cancellation.
    """
    ids_to_revoke = []

    # Add the main celery_id
    if target.celery_id:
        ids_to_revoke.append(target.celery_id)

    # Add all chain task IDs
    if target.celery_chain_ids:
        ids_to_revoke.extend(target.celery_chain_ids)

    # Revoke all tasks - use SIGTERM first to allow cleanup
    for task_id in ids_to_revoke:
        celery.app.control.revoke(task_id, terminate=False, signal='SIGTERM')

    # Kill any external processes spawned by the target
    kill_task_processes(target)

    # Clear target state
    target.celery_id = None
    target.celery_chain_ids = []
    target.celery_pid = None
    target.state = 'cancelled'
    target.save()

    return len(ids_to_revoke)

@login_required
def view_queue(request, id=None):
    context = {}

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'terminatealltasks':
            if request.user.is_staff:
                # Terminate all tasks with proper chain revocation
                count = 0
                for target in models.Target.objects.filter(celery_id__isnull=False):
                    count += revoke_task_chain(target)
                messages.success(request, f"Terminated {count} queued tasks")

            return HttpResponseRedirect(request.path_info)

        if action == 'cleanuplinkedtasks':
            if request.user.is_staff:
                for target in models.Target.objects.filter(celery_id__isnull=False):
                    target.celery_id = None
                    target.celery_chain_ids = []
                    target.celery_pid = None
                    target.state = 'failed'
                    target.save()

            return HttpResponseRedirect(request.path_info)

        if action == 'terminatetask' and id:
            if request.user.is_staff or True:
                # Find Django target and revoke entire chain
                target = models.Target.objects.filter(celery_id=id).first()
                if not target:
                    # Try to find by chain ID
                    target = find_target_by_chain_id(id)

                if target:
                    count = revoke_task_chain(target)
                    messages.success(request, f"Terminated task chain ({count} subtasks)")
                else:
                    # Fallback: revoke just this ID
                    celery.app.control.revoke(id, terminate=False, signal='SIGTERM')
                    messages.success(request, f"Terminated task {id}")

            return HttpResponseRedirect(request.path_info)

        if action == 'cleanuplinkedtask' and id:
            if request.user.is_staff or True:
                for target in models.Target.objects.filter(celery_id=id):
                    target.celery_id = None
                    target.celery_chain_ids = []
                    target.celery_pid = None
                    target.state = 'failed'
                    target.save()

            return HttpResponseRedirect(request.path_info)

    if id:
        ctask = celery.app.AsyncResult(id)
        context['ctask'] = ctask

        # Find linked Django target
        target = models.Target.objects.filter(celery_id=id).first()
        if not target:
            target = find_target_by_chain_id(id)
        context['target'] = target

        # Show chain position if part of a chain
        if target and target.celery_chain_ids and id in target.celery_chain_ids:
            context['chain_position'] = target.celery_chain_ids.index(id) + 1
            context['chain_total'] = len(target.celery_chain_ids)

    else:
        queue = []

        inspect = celery.app.control.inspect(timeout=0.1)
        for res,state in [(inspect.active(), 'active'), (inspect.reserved(), 'pending'), (inspect.scheduled(), 'scheduled')]:
            if res:
                for wtasks in res.values():
                    for ctask in wtasks:
                        if 'name' in ctask:
                            ctask['shortname'] = ctask['name'].split('.')[-1]

                        ctask['state'] = state

                        # Find linked Django target and add chain info
                        target = models.Target.objects.filter(celery_id=ctask['id']).first()
                        if not target:
                            target = find_target_by_chain_id(ctask['id'])

                        if target:
                            ctask['target_id'] = target.id
                            ctask['target_name'] = target.name
                            if target.celery_chain_ids and ctask['id'] in target.celery_chain_ids:
                                ctask['chain_position'] = target.celery_chain_ids.index(ctask['id']) + 1
                                ctask['chain_total'] = len(target.celery_chain_ids)

                        queue.append(ctask)

        context['queue'] = queue

    return TemplateResponse(request, 'queue.html', context=context)


def get_queue(request, id):
    ctask = celery.app.AsyncResult(id)

    return JsonResponse({'state': ctask.state, 'id': ctask.id})
