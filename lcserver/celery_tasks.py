# Django + Celery imports
from celery import shared_task, chain

import os, glob, shutil
import signal
import time

from functools import partial

import numpy as np

from . import models
from . import processing


# Process group management for killing external processes
def kill_task_processes(target):
    """Kill all processes associated with a target via process group."""
    if target.celery_pid:
        try:
            # Kill entire process group
            os.killpg(os.getpgid(target.celery_pid), signal.SIGTERM)
            time.sleep(0.5)
            os.killpg(os.getpgid(target.celery_pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


class TaskProcessContext:
    """
    Context manager for task execution with process group management.
    Handles: cancellation check, process group setup, signal handlers, cleanup, finalization.
    """
    def __init__(self, celery_task, target_id, finalize=True):
        self.celery_task = celery_task
        self.target_id = target_id
        self.finalize = finalize
        self.target = None
        self.basepath = None
        self.cancelled = False
        self._old_sigterm = None

    def __enter__(self):
        self.target = models.Target.objects.get(id=self.target_id)

        # Check if target was cancelled before starting
        if not self.target.celery_id:
            self.celery_task.request.chain = None
            self.cancelled = True
            return self

        self.basepath = self.target.path()

        # Store PID in database
        self.target.celery_pid = os.getpid()
        self.target.save(update_fields=['celery_pid'])

        # Try to become process group leader so children can be killed together
        try:
            os.setpgrp()
        except OSError:
            pass  # Already a group leader

        # Set up signal handler for graceful termination
        self._old_sigterm = signal.signal(signal.SIGTERM, self._sigterm_handler)

        return self

    def _sigterm_handler(self, signum, frame):
        """Handle SIGTERM - clean up and kill process group."""
        self._cleanup_pid()
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except:
            pass
        raise SystemExit(1)

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

        # Restore old signal handler
        if self._old_sigterm is not None:
            signal.signal(signal.SIGTERM, self._old_sigterm)

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
