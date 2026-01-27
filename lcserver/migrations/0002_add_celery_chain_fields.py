# Generated migration for celery chain tracking

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('lcserver', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='target',
            name='celery_chain_ids',
            field=models.JSONField(blank=True, default=list, editable=False),
        ),
        migrations.AddField(
            model_name='target',
            name='celery_pid',
            field=models.IntegerField(blank=True, default=None, editable=False, null=True),
        ),
    ]
