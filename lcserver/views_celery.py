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


def is_ajax(request):
    return request.headers.get('x-requested-with') == 'XMLHttpRequest'


def build_queue(user=None):
    """Collect active/pending/scheduled Celery tasks, annotated with linked targets."""
    queue = []

    inspect = celery.app.control.inspect(timeout=0.5)
    for res,state in [(inspect.active(), 'active'), (inspect.reserved(), 'pending'), (inspect.scheduled(), 'scheduled')]:
        if res:
            for wtasks in res.values():
                for ctask in wtasks:
                    if 'name' in ctask:
                        ctask['shortname'] = ctask['name'].split('.')[-1]

                    ctask['state'] = state

                    # Find linked target and add chain info
                    target = models.Target.objects.filter(celery_id=ctask['id']).first()
                    if not target:
                        target = find_target_by_chain_id(ctask['id'])

                    if target:
                        ctask['target_id'] = target.id
                        ctask['target_name'] = target.name
                        if target.celery_chain_ids and ctask['id'] in target.celery_chain_ids:
                            ctask['chain_position'] = target.celery_chain_ids.index(ctask['id']) + 1
                            ctask['chain_total'] = len(target.celery_chain_ids)

                    if user is not None:
                        ctask['can_manage'] = target.can_edit(user) if target else user.is_staff

                    queue.append(ctask)

    # Stable ordering so that entries do not jump around between refreshes
    order = {'active': 0, 'pending': 1, 'scheduled': 2}
    queue.sort(key=lambda _: (order.get(_['state'], 3), _.get('time_start') or 0, _['id']))

    return queue


@login_required
def view_queue(request, id=None):
    context = {}

    if request.method == 'POST':
        action = request.POST.get('action')
        ok = False
        message = "Unknown action"

        if action == 'terminatealltasks':
            if request.user.is_staff:
                ntargets = 0
                for target in models.Target.objects.filter(celery_id__isnull=False):
                    revoke_task_chain(target)
                    ntargets += 1
                ok = True
                message = f"Terminated {ntargets} running tasks"
            else:
                message = "Only staff may terminate all tasks"

        elif action == 'cleanuplinkedtasks':
            if request.user.is_staff:
                ntargets = 0
                for target in models.Target.objects.filter(celery_id__isnull=False):
                    target.celery_id = None
                    target.celery_chain_ids = []
                    target.celery_pid = None
                    target.state = 'failed'
                    target.save()
                    ntargets += 1
                ok = True
                message = f"Cleaned up {ntargets} linked targets"
            else:
                message = "Only staff may cleanup all tasks"

        elif action == 'terminatetask' and id:
            # Find linked target and revoke the entire chain
            target = models.Target.objects.filter(celery_id=id).first()
            if not target:
                target = find_target_by_chain_id(id)

            if target:
                if target.can_edit(request.user):
                    count = revoke_task_chain(target)
                    ok = True
                    message = f"Terminated task chain ({count} subtasks)"
                else:
                    message = f"Cannot terminate task for target {target.id} belonging to {target.user.username}"
            elif request.user.is_staff:
                # Fallback: revoke just this ID
                celery.app.control.revoke(id, terminate=False, signal='SIGTERM')
                ok = True
                message = f"Terminated task {id}"
            else:
                message = "Task not found"

        elif action == 'cleanuplinkedtask' and id:
            ntargets = 0
            denied = None
            for target in models.Target.objects.filter(celery_id=id):
                if target.can_edit(request.user):
                    target.celery_id = None
                    target.celery_chain_ids = []
                    target.celery_pid = None
                    target.state = 'failed'
                    target.save()
                    ntargets += 1
                else:
                    denied = target

            if denied and not ntargets:
                message = f"Cannot cleanup target {denied.id} belonging to {denied.user.username}"
            else:
                ok = True
                message = f"Cleaned up {ntargets} linked targets"

        if is_ajax(request):
            return JsonResponse({'ok': ok, 'message': message}, status=200 if ok else 403)

        if ok:
            messages.success(request, message)
        else:
            messages.error(request, message)

        return HttpResponseRedirect(request.path_info)

    if id:
        ctask = celery.app.AsyncResult(id)
        context['ctask'] = ctask

        # Find linked target
        target = models.Target.objects.filter(celery_id=id).first()
        if not target:
            target = find_target_by_chain_id(id)
        context['target'] = target

        context['can_manage'] = target.can_edit(request.user) if target else request.user.is_staff

        # Show chain position if part of a chain
        if target and target.celery_chain_ids and id in target.celery_chain_ids:
            context['chain_position'] = target.celery_chain_ids.index(id) + 1
            context['chain_total'] = len(target.celery_chain_ids)

    else:
        context['queue'] = build_queue(request.user)

    return TemplateResponse(request, 'queue.html', context=context)


@login_required
def queue_list(request):
    """HTML fragment with current queue contents, for AJAX refreshing."""
    return TemplateResponse(request, 'queue_list.html', context={'queue': build_queue(request.user)})


@login_required
def get_queue(request, id):
    ctask = celery.app.AsyncResult(id)

    result = {'id': ctask.id, 'state': ctask.state, 'ready': ctask.ready()}

    # Find linked target
    target = models.Target.objects.filter(celery_id=id).first()
    if not target:
        target = find_target_by_chain_id(id)

    if target and target.can_view(request.user):
        result['target_id'] = target.id
        result['target_state'] = target.state
        result['target_running'] = target.celery_id is not None
        if target.celery_chain_ids and id in target.celery_chain_ids:
            result['chain_position'] = target.celery_chain_ids.index(id) + 1
            result['chain_total'] = len(target.celery_chain_ids)

    return JsonResponse(result)
