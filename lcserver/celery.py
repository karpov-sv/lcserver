import os

from celery import Celery
from decouple import config

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lcserver.settings')

app = Celery('lcserver')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Additional configuration for better task management
app.conf.update(
    # Re-queue tasks if worker is lost (crash, kill, etc.)
    task_reject_on_worker_lost=True,
    # Store task state for chain tracking
    task_track_started=True,
    #
    worker_prefetch_multiplier=1,
    # How many sources are acquired at once. Left unset this follows the
    # machine's CPU count, which was ten here and is a rude number of
    # simultaneous queries to point at a survey - several of ours share CDS.
    # The work is nearly all waiting on a reply, so a small number costs
    # little. Overridden by -c on the command line.
    worker_concurrency=config('CELERY_CONCURRENCY', default=4, cast=int),
    broker_transport_options={
        "visibility_timeout": 3600,
    },
    # Use SIGTERM for graceful shutdown (allows cleanup handlers to run)
    worker_term_signal='SIGTERM',

)

# Load task modules from all registered Django apps.
app.autodiscover_tasks(related_name='celery_tasks')


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
