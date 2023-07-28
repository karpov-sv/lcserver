# Django + Celery imports
from celery import shared_task

import os, glob, shutil

from functools import partial

import numpy as np

from . import models
from . import processing


def fix_config(config):
    """
    Fix non-serializable Numpy types in config
    """
    for key in config.keys():
        if type(config[key]) == np.float32:
            config[key] = float(config[key])


@shared_task(bind=True)
def task_cleanup(self, id):
    target = models.Target.objects.get(id=id)
    basepath = target.path()

    for path in glob.glob(os.path.join(basepath, '*')):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)

    # End processing
    target.state = 'cleaned'
    target.celery_id = None
    target.complete()
    target.save()


@shared_task(bind=True)
def task_info(self, id):
    target = models.Target.objects.get(id=id)
    basepath = target.path()

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
        target.celery_id = None

    # End processing
    target.celery_id = None
    fix_config(config)
    target.complete()
    target.save()

@shared_task(bind=True)
def task_ztf(self, id):
    target = models.Target.objects.get(id=id)
    basepath = target.path()

    config = target.config
    config['target_name'] = target.name

    log = partial(processing.print_to_file, logname=os.path.join(basepath, 'ztf.log'))
    log(clear=True)

    # Start processing
    try:
        processing.target_ztf(config, basepath=basepath, verbose=log)
        target.state = 'ztf lightcurve acquired'
    except:
        import traceback
        log("\nError!\n", traceback.format_exc())

        target.state = 'failed'
        target.celery_id = None

    # End processing
    target.celery_id = None
    fix_config(config)
    target.complete()
    target.save()
