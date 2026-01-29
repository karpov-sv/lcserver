# Django + Celery imports
from celery import shared_task

import os, glob, shutil

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
    """
    def __init__(self, celery_task, target_id, finalize=True):
        self.celery_task = celery_task
        self.target_id = target_id
        self.finalize = finalize
        self.target = None
        self.basepath = None
        self.cancelled = False

    def __enter__(self):
        self.target = models.Target.objects.get(id=self.target_id)

        # Check if target was cancelled before starting
        if not self.target.celery_id:
            if getattr(self.celery_task.request, 'chain', None) is not None:
                self.celery_task.request.chain = None
            self.cancelled = True
            return self

        self.basepath = self.target.path()

        # In thread pool mode, celery_pid refers to the whole worker process.
        # Clear any stale PID values to avoid accidental process-group kills.
        if self.target.celery_pid is not None:
            self.target.celery_pid = None
            self.target.save(update_fields=['celery_pid'])

        return self

    def _cleanup_pid(self):
        """Clear PID from database."""
        if self.target:
            try:
                self.target.celery_pid = None
                self.target.save(update_fields=['celery_pid'])
            except:
                pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_pid()

        # Handle finalization based on target state
        if self.target:
            # Check if target failed (state ends with '_failed' or equals 'failed')
            if self.target.state and ('failed' in self.target.state.lower()):
                # Always break chain on error
                self.target.celery_id = None
            elif self.finalize:
                # Success with finalize=True: mark complete
                self.target.celery_id = None
                self.target.celery_chain_ids = []
                self.target.complete()
            # else: Success with finalize=False: leave celery_id (continue chain)

            # Save target (always)
            self.target.save()

        return False  # Don't suppress exceptions


def fix_config(config):
    """
    Fix non-serializable Numpy types in config
    """
    for key in config.keys():
        if type(config[key]) == np.float32:
            config[key] = float(config[key])


def create_survey_task(source_id, survey_config):
    """Factory function to create a Celery task for a survey source."""
    processing_func_name = survey_config['processing_function']

    @shared_task(bind=True, acks_late=True, reject_on_worker_lost=True, name=f'lcserver.celery_tasks.task_{source_id}')
    def survey_task(self, id, finalize=True):
        with TaskProcessContext(self, id, finalize=finalize) as ctx:
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
            except:
                import traceback
                log("\nError!\n", traceback.format_exc())
                target.state = 'failed'

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
task_tess = _survey_tasks['tess']
task_dasch = _survey_tasks['dasch']
task_applause = _survey_tasks['applause']
task_mmt9 = _survey_tasks['mmt9']
task_combined = _survey_tasks['combined']


def get_survey_task(source_id):
    """Get Celery task for a survey source."""
    return _survey_tasks.get(source_id)


@shared_task(bind=True)
def task_finalize(self, id):
    target = models.Target.objects.get(id=id)

    # Determine if processing succeeded or failed
    target.celery_id = None
    target.celery_chain_ids = []
    target.complete()
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
def task_break_if_failed(self, id):
    target = models.Target.objects.get(id=id)

    if not target.celery_id:
        print("Breaking the chain!!!")
        # Clear chain to prevent further execution
        self.request.chain = None
        raise RuntimeError("Task chain cancelled")


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
        target.celery_id = None
        target.celery_chain_ids = []
        target.complete()

    target.save()


# Note: Individual task definitions (task_info, task_ztf, etc.) are now
# auto-generated by create_survey_task() factory function above.
# Named references are preserved for backward compatibility.


# Higher-level interface for running multiple processing steps for the target
def run_target_steps(target, steps):
    """
    Build and execute a chain of processing steps for a target.

    Pattern (from stdweb):
    - For each step: set_state -> task -> break_if_failed -> set_state
    - Add finalize at end
    - Freeze chain to get all task IDs
    - Store in target.celery_chain_ids (reversed)
    - Apply chain
    """
    from celery import chain

    todo = []

    for step in steps:
        survey_config = surveys.get_survey_source(step)
        if not survey_config:
            print(f"Unknown step: {step}")
            continue

        print(f"Will run {step} step for target {target.id}")

        # Get task function
        task_func = get_survey_task(step)

        # Add to chain: set_state -> task -> break -> set_state
        todo.append(task_set_state.subtask(
            args=[target.id, survey_config['state_acquiring']], immutable=True))
        todo.append(task_func.subtask(
            args=[target.id, False], immutable=True))
        todo.append(task_break_if_failed.subtask(
            args=[target.id], immutable=True))
        todo.append(task_set_state.subtask(
            args=[target.id, survey_config['state_acquired']], immutable=True))

    if todo:
        # Add finalize at the end
        todo.append(task_finalize.subtask(args=[target.id], immutable=True))

        # Create the chain and freeze it to get task IDs before applying
        task_chain = chain(todo)
        res = task_chain.freeze()

        # Extract all task IDs from the frozen chain (reversed order)
        target.celery_chain_ids = list(reversed(res.as_list()))

        # Apply the chain
        result = task_chain.apply_async()
        target.celery_id = result.id
        target.state = 'running'
        target.save()

        return result

    return None
