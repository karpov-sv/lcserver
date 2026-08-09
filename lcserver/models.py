from django.db import models, transaction
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

    # How each source last fared, as {source_id: 'running'|'done'|'failed'}.
    # The state field above can only describe one step, so it says nothing
    # useful once the sources no longer take turns; this keeps a place for
    # each of them, and each writes only its own key.
    source_states = models.JSONField(default=dict, blank=True, editable=False)

    def path(self):
        return os.path.join(settings.TARGETS_PATH, str(self.id))

    def complete(self):
        self.completed = now()

    @classmethod
    def update_atomic(cls, id, apply, fields):
        """Apply a change to a freshly read row, writing only the named fields.

        A task holds its Target for as long as it runs, which for a slow survey
        is minutes. Saving that copy at the end writes back every column as it
        was when the task started, undoing anything that happened meanwhile -
        a cancellation, or another source's results. So the change is made to a
        row read here, within the transaction, and `fields` alone are written.

        The lock is real on a backend that has one. SQLite has no
        SELECT ... FOR UPDATE and Django ignores the call there, so the
        serialisation comes instead from OPTIONS['transaction_mode'] =
        'IMMEDIATE' in the settings, which takes the write lock as the
        transaction opens rather than at its first write.
        """
        with transaction.atomic():
            target = cls.objects.select_for_update().get(id=id)
            apply(target)
            # modified is auto_now, and auto_now fields are only touched when
            # they are named in update_fields
            target.save(update_fields=sorted(set(fields) | {'modified'}))

            return target

    # --- Access control -----------------------------------------------------
    # All target access checks funnel through these helpers so that the rules
    # live in exactly one place. The matching queryset filter is
    # Target.accessible_to() below.

    def can_view(self, user):
        """Whether `user` may read this target."""
        if not user.is_authenticated:
            return False
        return user.is_staff or user == self.user

    def can_edit(self, user):
        """Whether `user` may modify / submit / run this target."""
        if not user.is_authenticated:
            return False
        return user.is_staff or user == self.user

    def can_delete(self, user):
        """Whether `user` may delete this target."""
        if not user.is_authenticated:
            return False
        return user.is_staff or user == self.user

    @staticmethod
    def accessible_to(user, queryset=None):
        """Queryset of targets `user` may view. Mirrors can_view() at the DB level."""
        if queryset is None:
            queryset = Target.objects.all()
        if not user.is_authenticated:
            return queryset.none()
        if user.is_staff:
            return queryset
        return queryset.filter(user=user)

    def __str__(self):
        return f"{self.id}: {self.user.username} : {self.name}"


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
