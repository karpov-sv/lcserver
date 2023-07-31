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

        # We should raise it in order to break the chain (as the rest does depend on this stage)
        raise RuntimeError('failed')

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
        target.state = 'ZTF lightcurve acquired'
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
def task_asas(self, id):
    target = models.Target.objects.get(id=id)
    basepath = target.path()

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
        target.celery_id = None

    # End processing
    target.celery_id = None
    fix_config(config)
    target.complete()
    target.save()


@shared_task(bind=True)
def task_tess(self, id):
    target = models.Target.objects.get(id=id)
    basepath = target.path()

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
        target.celery_id = None

    # End processing
    target.celery_id = None
    fix_config(config)
    target.complete()
    target.save()


@shared_task(bind=True)
def task_dasch(self, id):
    target = models.Target.objects.get(id=id)
    basepath = target.path()

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
        target.celery_id = None

    # End processing
    target.celery_id = None
    fix_config(config)
    target.complete()
    target.save()


@shared_task(bind=True)
def task_applause(self, id):
    target = models.Target.objects.get(id=id)
    basepath = target.path()

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
        target.celery_id = None

    # End processing
    target.celery_id = None
    fix_config(config)
    target.complete()
    target.save()


@shared_task(bind=True)
def task_combined(self, id):
    target = models.Target.objects.get(id=id)
    basepath = target.path()

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
        target.celery_id = None

    # End processing
    target.celery_id = None
    fix_config(config)
    target.complete()
    target.save()
