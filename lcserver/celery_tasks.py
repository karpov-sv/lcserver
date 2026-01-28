# Django + Celery imports
from celery import shared_task

import os, glob, shutil

from functools import partial

import numpy as np

from . import models
from . import processing


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
    target = models.Target.objects.get(id=id)
    target.state = state
    target.save()


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


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_info(self, id, finalize=True):
    with TaskProcessContext(self, id, finalize=finalize) as ctx:
        if ctx.cancelled:
            return

        target = ctx.target
        basepath = ctx.basepath
        config = target.config
        config['target_name'] = target.name

        log = partial(processing.print_to_file, logname=os.path.join(basepath, 'info.log'))
        log(clear=True)

        # Start processing
        try:
            processing.target_info(config, basepath=basepath, verbose=log)
            target.state = 'info acquired'
        except:
            import traceback
            log("\nError!\n", traceback.format_exc())
            target.state = 'failed'

        fix_config(config)
        # Context manager handles finalize and save


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_ztf(self, id, finalize=True):
    with TaskProcessContext(self, id, finalize=finalize) as ctx:
        if ctx.cancelled:
            return

        target = ctx.target
        basepath = ctx.basepath
        config = target.config
        config['target_name'] = target.name

        log = partial(processing.print_to_file, logname=os.path.join(basepath, 'ztf.log'))
        log(clear=True)

        # Start processing
        try:
            processing.target_ztf(config, basepath=basepath, verbose=log)
            target.state = 'ZTF lightcurve acquired'
        except:
            import traceback
            log("\nError!\n", traceback.format_exc())
            target.state = 'failed'

        fix_config(config)
        # Context manager handles finalize and save


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_asas(self, id, finalize=True):
    with TaskProcessContext(self, id, finalize=finalize) as ctx:
        if ctx.cancelled:
            return

        target = ctx.target
        basepath = ctx.basepath
        config = target.config
        config['target_name'] = target.name

        log = partial(processing.print_to_file, logname=os.path.join(basepath, 'asas.log'))
        log(clear=True)

        # Start processing
        try:
            processing.target_asas(config, basepath=basepath, verbose=log)
            target.state = 'ASAS-SN lightcurve acquired'
        except:
            import traceback
            log("\nError!\n", traceback.format_exc())
            target.state = 'failed'

        fix_config(config)
        # Context manager handles finalize and save


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_tess(self, id, finalize=True):
    with TaskProcessContext(self, id, finalize=finalize) as ctx:
        if ctx.cancelled:
            return

        target = ctx.target
        basepath = ctx.basepath
        config = target.config
        config['target_name'] = target.name

        log = partial(processing.print_to_file, logname=os.path.join(basepath, 'tess.log'))
        log(clear=True)

        # Start processing
        try:
            processing.target_tess(config, basepath=basepath, verbose=log)
            target.state = 'TESS lightcurves acquired'
        except:
            import traceback
            log("\nError!\n", traceback.format_exc())
            target.state = 'failed'

        fix_config(config)
        # Context manager handles finalize and save


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_dasch(self, id, finalize=True):
    with TaskProcessContext(self, id, finalize=finalize) as ctx:
        if ctx.cancelled:
            return

        target = ctx.target
        basepath = ctx.basepath
        config = target.config
        config['target_name'] = target.name

        log = partial(processing.print_to_file, logname=os.path.join(basepath, 'dasch.log'))
        log(clear=True)

        # Start processing
        try:
            processing.target_dasch(config, basepath=basepath, verbose=log)
            target.state = 'DASCH lightcurve acquired'
        except:
            import traceback
            log("\nError!\n", traceback.format_exc())
            target.state = 'failed'

        fix_config(config)
        # Context manager handles finalize and save


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_applause(self, id, finalize=True):
    with TaskProcessContext(self, id, finalize=finalize) as ctx:
        if ctx.cancelled:
            return

        target = ctx.target
        basepath = ctx.basepath
        config = target.config
        config['target_name'] = target.name

        log = partial(processing.print_to_file, logname=os.path.join(basepath, 'applause.log'))
        log(clear=True)

        # Start processing
        try:
            processing.target_applause(config, basepath=basepath, verbose=log)
            target.state = 'APPLAUSE lightcurve acquired'
        except:
            import traceback
            log("\nError!\n", traceback.format_exc())
            target.state = 'failed'

        fix_config(config)
        # Context manager handles finalize and save


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_mmt9(self, id, finalize=True):
    with TaskProcessContext(self, id, finalize=finalize) as ctx:
        if ctx.cancelled:
            return

        target = ctx.target
        basepath = ctx.basepath
        config = target.config
        config['target_name'] = target.name

        log = partial(processing.print_to_file, logname=os.path.join(basepath, 'mmt9.log'))
        log(clear=True)

        # Start processing
        try:
            processing.target_mmt9(config, basepath=basepath, verbose=log)
            target.state = 'Mini-MegaTORTORA lightcurve acquired'
        except:
            import traceback
            log("\nError!\n", traceback.format_exc())
            target.state = 'failed'

        fix_config(config)
        # Context manager handles finalize and save


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True)
def task_combined(self, id, finalize=True):
    with TaskProcessContext(self, id, finalize=finalize) as ctx:
        if ctx.cancelled:
            return

        target = ctx.target
        basepath = ctx.basepath
        config = target.config
        config['target_name'] = target.name

        log = partial(processing.print_to_file, logname=os.path.join(basepath, 'combined.log'))
        log(clear=True)

        # Start processing
        try:
            processing.target_combined(config, basepath=basepath, verbose=log)
            target.state = 'Combined lightcurve acquired'
        except:
            import traceback
            log("\nError!\n", traceback.format_exc())
            target.state = 'failed'

        fix_config(config)
        # Context manager handles finalize and save
