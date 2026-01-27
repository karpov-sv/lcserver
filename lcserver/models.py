from django.db import models
from django.db.models.signals import pre_delete, post_save
from django.dispatch import receiver
from django.utils.timezone import now
from django.contrib.auth.models import User
from django.conf import settings

import os, shutil
import datetime


class Target(models.Model):
    # path = models.CharField(max_length=250, blank=False, unique=True, editable=False) # Base dir where task processing will be performed
    name = models.CharField(max_length=250, blank=False) # Target name
    title = models.CharField(max_length=250, blank=True) # Optional title or comment

    state = models.CharField(max_length=50, blank=False, default='initial') # State of the task

    celery_id = models.CharField(max_length=50, blank=True, null=True, default=None, editable=False) # Celery task ID, when running
    celery_chain_ids = models.JSONField(default=list, blank=True, editable=False) # List of all task IDs in the chain
    celery_pid = models.IntegerField(blank=True, null=True, default=None, editable=False) # Process ID of running task

    user =  models.ForeignKey(User, on_delete=models.CASCADE)

    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True) # Updated on every .save()
    completed = models.DateTimeField(default=now, editable=False) # Manually updated on finishing the processing

    config = models.JSONField(default=dict, blank=True) #

    def path(self):
        return os.path.join(settings.TARGETS_PATH, str(self.id))

    def complete(self):
        self.completed = now()

@receiver(pre_delete, sender=Target)
def delete_target_hook(sender, instance, using, **kwargs):
    path = instance.path()

    # Cleanup the data on filesystem related to this model
    if os.path.exists(path):
        shutil.rmtree(path)


@receiver(post_save, sender=Target)
def save_target_hook(sender, instance, created, raw, using, **kwargs):
    if created:
        path = instance.path()

        # Make target folder
        if not os.path.exists(path):
            os.makedirs(path)
