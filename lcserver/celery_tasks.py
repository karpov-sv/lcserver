# Django + Celery imports
from celery import shared_task

import os, glob, shutil, copy

from functools import partial

import numpy as np

from . import models
from . import processing
from . import surveys


# Thread pool-safe hook for killing external processes
def kill_task_processes(target):
    """
    No-op in thread pool mode.
    Killing by process group would terminate the entire worker process.
    """
    return


class TaskProcessContext:
    """
    Context manager for task execution in thread pool mode.
    Handles: cancellation check, celery_pid cleanup, and finalization.

    The target it hands out is a working copy. Nothing the task does to it is
    written back wholesale - on the way out the config is compared against how
    it was found, and only the difference is applied to a freshly read row.
    Two sources acquired at the same time therefore compose, instead of the
    slower one overwriting the faster one's results with what it read minutes
    earlier.
    """
    def __init__(self, celery_task, target_id, finalize=True, source_id=None):
        self.celery_task = celery_task
        self.target_id = target_id
        self.finalize = finalize
        self.source_id = source_id
        self.target = None
        self.basepath = None
        self.cancelled = False
        self.config_before = {}

    def __enter__(self):
        self.target = models.Target.objects.get(id=self.target_id)

        # Check if target was cancelled before starting
        if not self.target.celery_id:
            if getattr(self.celery_task.request, 'chain', None) is not None:
                self.celery_task.request.chain = None
            self.cancelled = True
            return self

        self.basepath = self.target.path()

        # How the config looked before the task touched it, to tell afterwards
        # which keys are this task's own doing
        self.config_before = copy.deepcopy(self.target.config)

        fields = []

        # In thread pool mode, celery_pid refers to the whole worker process.
        # Clear any stale PID values to avoid accidental process-group kills.
        if self.target.celery_pid is not None:
            self.target.celery_pid = None
            fields.append('celery_pid')

        if self.source_id:
            self.target.source_states[self.source_id] = 'running'
            fields.append('source_states')

        if fields:
            source_id = self.source_id

            def mark(fresh):
                if 'celery_pid' in fields:
                    fresh.celery_pid = None
                if source_id:
                    fresh.source_states[source_id] = 'running'

            models.Target.update_atomic(self.target_id, mark, fields)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.target:
            return False

        # What this task did to the config, as a set of keys to write and a
        # set to drop - rather than the whole dictionary as it read it
        changed = {k: v for k, v in self.target.config.items()
                   if k not in self.config_before or self.config_before[k] != v}
        removed = [k for k in self.config_before if k not in self.target.config]

        state = self.target.state
        source_state = self.target.source_states.get(self.source_id) if self.source_id else None
        finalize = self.finalize
        source_id = self.source_id

        def merge(fresh):
            fresh.celery_pid = None

            for key, value in changed.items():
                fresh.config[key] = value

            for key in removed:
                fresh.config.pop(key, None)

            if source_id and source_state:
                fresh.source_states[source_id] = source_state

            # Cancelled while this task was running: the run is already over
            # and has said so, so its state is not overwritten here. What the
            # step managed to do is still recorded above.
            if fresh.celery_id is None:
                return

            fresh.state = state

            # A step that failed no longer stops the steps after it. The
            # sources are independent of each other - one survey being down
            # says nothing about the next - so the failure is recorded, in the
            # log and in source_states, and the chain carries on. Only
            # cancellation breaks a chain, which task_break_if_cancelled
            # checks for between the steps.
            if finalize:
                fresh.celery_id = None
                fresh.celery_chain_ids = []
                fresh.complete()
                # A refresh was asked for this run only. Dropped here for a
                # single step, and in task_finalize for a whole chain.
                fresh.config.pop('refresh_cache', None)

        models.Target.update_atomic(
            self.target_id, merge,
            ['config', 'source_states', 'state', 'celery_pid',
             'celery_id', 'celery_chain_ids', 'completed'])

        return False  # Don't suppress exceptions


def fix_config(config):
    """
    Fix non-serializable Numpy types in config
    """
    for key in config.keys():
        if type(config[key]) == np.float32:
            config[key] = float(config[key])


# Steps the rest of a chain cannot do without. The info step resolves the
# coordinates every other source queries by; a survey being down concerns only
# itself, and the chain carries on past it.
GATING_STEPS = {'info'}


def create_survey_task(source_id, survey_config):
    """Factory function to create a Celery task for a survey source."""
    processing_func_name = survey_config['processing_function']

    @shared_task(bind=True, acks_late=True, reject_on_worker_lost=True, name=f'lcserver.celery_tasks.task_{source_id}')
    def survey_task(self, id, finalize=True):
        with TaskProcessContext(self, id, finalize=finalize,
                                source_id=source_id) as ctx:
            if ctx.cancelled:
                return

            target = ctx.target
            config = target.config
            config['target_name'] = target.name

            log = partial(processing.print_to_file,
                         logname=os.path.join(ctx.basepath, survey_config['log_file']))
            log(clear=True)

            try:
                # Get processing function by name
                processing_func = getattr(processing, processing_func_name)
                processing_func(config, basepath=ctx.basepath, verbose=log)
                target.state = survey_config['state_acquired']
                target.source_states[source_id] = 'done'
            except:
                import traceback
                log("\nError!\n", traceback.format_exc())
                target.state = 'failed'
                target.source_states[source_id] = 'failed'

            fix_config(config)

    return survey_task


# Auto-register all survey tasks
_survey_tasks = {}
for source_id, config in surveys.SURVEY_SOURCES.items():
    # Skip sources without processing functions (lightcurve-only sources)
    if config.get('processing_function') is None:
        continue
    _survey_tasks[source_id] = create_survey_task(source_id, config)

# Create named references for backward compatibility
task_info = _survey_tasks['info']
task_ztf = _survey_tasks['ztf']
task_asas = _survey_tasks['asas']
task_css = _survey_tasks['css']
task_kws = _survey_tasks['kws']
task_ptf = _survey_tasks['ptf']
task_tess = _survey_tasks['tess']
task_dasch = _survey_tasks['dasch']
task_applause = _survey_tasks['applause']
task_mmt9 = _survey_tasks['mmt9']
task_combined = _survey_tasks['combined']


def get_survey_task(source_id):
    """Get Celery task for a survey source."""
    return _survey_tasks.get(source_id)


@shared_task(bind=True)
def task_finalize(self, id, steps=None):
    target = models.Target.objects.get(id=id)

    # The refresh applied to the chain that has just finished, and every step
    # in it saw the flag. Clearing it here rather than in the first step is
    # what lets a chain refresh all of its sources.
    target.config.pop('refresh_cache', None)

    # A chain runs every step it was given, so it can end with some of them
    # having failed. The state says how many, as the last step to run would
    # otherwise be the only thing reported; which ones is in source_states,
    # and each has its error in its own log.
    # Only the steps this run was asked for: a source that failed a week ago
    # and was not asked for again is not this run's business
    asked = steps if steps is not None else list(target.source_states)
    failed = [_ for _ in asked if target.source_states.get(_) == 'failed']

    if failed:
        # The field holds 50 characters, so only the count goes in it
        target.state = f"completed, {len(failed)} failed"

    target.celery_id = None
    target.celery_chain_ids = []
    target.complete()
    target.forget_unfinished_sources()
    target.save()


@shared_task(bind=True)
def task_set_state(self, id, state):
    from django.db import transaction

    with transaction.atomic():
        target = models.Target.objects.select_for_update().get(id=id)
        target.state = state
        target.save()

    # Force commit and close connection to ensure visibility
    from django.db import connection
    connection.close()


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_break_if_cancelled(self, id):
    """Stop a chain that has been cancelled, between one step and the next.

    A step that merely failed does not get here - the sources are independent,
    so the chain carries on and task_finalize reports what failed. Only an
    outside hand clearing celery_id, which is what revoke_task_chain does,
    ends the run early.
    """
    target = models.Target.objects.get(id=id)

    if not target.celery_id:
        print(f"Target {id} was cancelled, breaking the chain")
        # Clear chain to prevent further execution
        self.request.chain = None
        raise RuntimeError("Task chain cancelled")


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_break_if_step_failed(self, id, source_id):
    """Stop a chain whose gating step failed.

    Most steps are independent and a failure in one says nothing about the
    next, but the info step resolves the coordinates every other source
    queries by - so if it fails there is nothing for them to look up, and
    running them would turn one failure into nineteen.
    """
    target = models.Target.objects.get(id=id)

    if target.source_states.get(source_id) != 'failed':
        return

    print(f"Target {id} could not do {source_id}, breaking the chain")
    self.request.chain = None

    # task_finalize is further down the chain and will not get to run, so the
    # end-of-run bookkeeping happens here instead. The state is left as the
    # failed step set it.
    target.celery_id = None
    target.celery_chain_ids = []
    target.complete()
    target.config.pop('refresh_cache', None)
    target.forget_unfinished_sources()
    target.save()

    raise RuntimeError(f"Cannot continue past a failed {source_id} step")


@shared_task(bind=True)
def task_cleanup(self, id, finalize=True):
    target = models.Target.objects.get(id=id)
    basepath = target.path()

    for path in glob.glob(os.path.join(basepath, '*')):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)

    if finalize:
        # End processing
        target.state = 'cleaned'
        target.source_states = {}
        target.celery_id = None
        target.celery_chain_ids = []
        target.complete()

    target.save()


# Note: Individual task definitions (task_info, task_ztf, etc.) are now
# auto-generated by create_survey_task() factory function above.
# Named references are preserved for backward compatibility.


def _collect_ids(result):
    """Every task id in a frozen canvas, so that cancelling can revoke them all.

    AsyncResult.as_list() would do it, but only for a straight chain: it walks
    parents, and a chord's parent is a GroupResult, which does not implement
    it. So the walk happens here, over both the parent links and the members
    of any group along the way.
    """
    ids = []
    seen = set()

    def walk(node):
        if node is None or id(node) in seen:
            return

        seen.add(id(node))

        for member in getattr(node, 'results', None) or []:
            walk(member)

        task_id = getattr(node, 'id', None)
        if task_id and task_id not in ids:
            ids.append(task_id)

        walk(getattr(node, 'parent', None))

    walk(result)

    return ids


# Higher-level interface for running multiple processing steps for the target
def run_target_steps(target, steps):
    """
    Build and execute a chain of processing steps for a target.

    Pattern:

        providers... -> group(everything else) -> combined -> finalize

    The sources do not depend on one another, so they are acquired together
    rather than in turn - most of their time is spent waiting on a survey to
    answer. Three things are kept in order around them:

    - the info step first, and the chain stops if it fails: it resolves the
      coordinates every source queries by;
    - any source declaring provides_config, because the others convert their
      photometry with what it writes. ZTF measures the colour that five of
      them use, and alongside them they would race it and silently fall back
      to the catalogue value the info step derived;
    - the combined plot last, as it reads what all of them wrote.

    A step that fails does not end the run: the sources are independent, so
    the rest carry on and task_finalize reports how many failed. Only
    cancellation, or a gating step failing, stops it early.

    How many sources actually run at once is the worker's concurrency, not the
    size of the group - see worker_concurrency in celery.py.
    """
    from celery import chain, chord, group

    steps = [_ for _ in steps if surveys.get_survey_source(_)]

    # Sources whose results the others read, in the order the registry gives
    prologue = [_ for _ in steps
                if _ in GATING_STEPS
                or surveys.get_survey_source(_).get('provides_config')]

    # Reads every source's output, so it cannot be one of them
    epilogue = [_ for _ in steps if _ == 'combined']

    parallel = [_ for _ in steps if _ not in prologue and _ not in epilogue]

    if not steps:
        return None

    print(f"Will run {len(steps)} steps for target {target.id}: "
          f"{'+'.join(prologue)} then {len(parallel)} at once"
          + (f" then {'+'.join(epilogue)}" if epilogue else ""))

    def sequential(step):
        """A step that runs on its own, announcing itself as it goes."""
        survey_config = surveys.get_survey_source(step)
        out = [
            # The cancellation check comes before the state is announced, so
            # that a cancelled run does not leave 'acquiring ...' as its last
            # word. There is no trailing state step: the task sets its own
            # state, acquired or failed, and a step here would overwrite a
            # failure with a claim of success.
            task_break_if_cancelled.subtask(args=[target.id], immutable=True),
            task_set_state.subtask(
                args=[target.id, survey_config['state_acquiring']], immutable=True),
            get_survey_task(step).subtask(args=[target.id, False], immutable=True),
        ]

        # The one kind of failure the rest of a chain cannot survive
        if step in GATING_STEPS:
            out.append(task_break_if_step_failed.subtask(
                args=[target.id, step], immutable=True))

        return out

    todo = []

    for step in prologue:
        todo += sequential(step)

    tail = []

    for step in epilogue:
        tail += sequential(step)

    tail.append(task_finalize.subtask(args=[target.id, list(steps)], immutable=True))

    if parallel:
        todo.append(task_break_if_cancelled.subtask(args=[target.id], immutable=True))
        # One announcement for the lot: which of them is doing what is in
        # source_states, which each of them writes its own key of
        todo.append(task_set_state.subtask(
            args=[target.id, 'acquiring lightcurves'], immutable=True))

        # The group members never raise - they record their own failures - so
        # the callback runs whatever happens to any of them
        todo.append(chord(
            group([get_survey_task(_).subtask(args=[target.id, False], immutable=True)
                   for _ in parallel]),
            chain(tail)))
    else:
        todo += tail

    task_chain = chain(todo)

    # Freezing settles every id up front, which is what apply_async would go on
    # to use. They are recorded before the run is published rather than after:
    # the first thing the chain does is ask whether it has been cancelled, by
    # looking for the very celery_id being assigned here, and a worker can pick
    # the task up before a save that happens afterwards - reading no id, and
    # stopping the run before it began.
    res = task_chain.freeze()

    # Every id the run consists of, so that cancelling revokes all of them
    target.celery_chain_ids = list(reversed(_collect_ids(res)))
    target.celery_id = res.id
    target.state = 'running'

    # What the run means to attempt. A group is queued all at once but only
    # worker_concurrency of it runs, so most of these sit waiting - which is
    # worth showing rather than leaving the sections blank until their turn.
    target.source_states = dict(target.source_states or {},
                                **{_: 'pending' for _ in steps})
    target.save()

    return task_chain.apply_async()
