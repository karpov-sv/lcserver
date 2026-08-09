from django.core.management.base import BaseCommand
from django.conf import settings
from django.contrib.auth.models import User

import os
import shutil
from datetime import timedelta
from django.utils import timezone

from lcserver import models
from lcserver import celery


class Command(BaseCommand):
    help = 'Maintenance tasks: cleanup old targets, fix stale tasks, check system'

    def add_arguments(self, parser):
        # Cleanup old targets
        parser.add_argument(
            '--cleanup-old',
            action='store_true',
            dest='cleanup_old',
            help='Clean up targets older than specified days'
        )

        parser.add_argument(
            '--days',
            type=int,
            default=30,
            dest='days',
            help='Number of days for cleanup (default: 30)'
        )

        # Fix stale celery IDs
        parser.add_argument(
            '--fix-stale',
            action='store_true',
            dest='fix_stale',
            help='Fix targets with stale celery_id (tasks no longer running)'
        )

        # Cleanup orphaned directories
        parser.add_argument(
            '--cleanup-orphans',
            action='store_true',
            dest='cleanup_orphans',
            help='Remove directories without corresponding database records'
        )

        # System check
        parser.add_argument(
            '--check',
            action='store_true',
            dest='check_system',
            help='Check system status (Redis, Celery, disk space)'
        )

        # Statistics
        parser.add_argument(
            '--stats',
            action='store_true',
            dest='show_stats',
            help='Show statistics about targets'
        )

        # Dry run
        parser.add_argument(
            '--dry-run',
            action='store_true',
            dest='dry_run',
            help='Dry run: show what would be done without making changes'
        )

    def handle(self, *args, **options):
        # Change to project root
        os.chdir(settings.BASE_DIR)

        if options['cleanup_old']:
            self._cleanup_old_targets(options['days'], options['dry_run'])

        elif options['fix_stale']:
            self._fix_stale_tasks(options['dry_run'])

        elif options['cleanup_orphans']:
            self._cleanup_orphaned_directories(options['dry_run'])

        elif options['check_system']:
            self._check_system()

        elif options['show_stats']:
            self._show_statistics()

        else:
            # Default: show stats
            self._show_statistics()

    def _cleanup_old_targets(self, days, dry_run):
        """Clean up old targets."""
        cutoff_date = timezone.now() - timedelta(days=days)

        targets = models.Target.objects.filter(
            modified__lt=cutoff_date,
            celery_id__isnull=True  # Don't delete running tasks
        )

        self.stdout.write(
            f"Finding targets older than {days} days (before {cutoff_date.strftime('%Y-%m-%d')})"
        )

        count = targets.count()
        if count == 0:
            self.stdout.write("No targets to clean up")
            return

        self.stdout.write(f"Found {count} target(s) to clean up")

        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN - no changes will be made"))
            for target in targets[:10]:  # Show first 10
                self.stdout.write(
                    f"  Would delete: {target.id} - {target.name} (modified: {target.modified})"
                )
            if count > 10:
                self.stdout.write(f"  ... and {count - 10} more")
        else:
            # Delete targets
            for target in targets:
                self.stdout.write(
                    f"Deleting: {target.id} - {target.name} (modified: {target.modified})"
                )
                target.delete()

            self.stdout.write(self.style.SUCCESS(f"Deleted {count} target(s)"))

    def _fix_stale_tasks(self, dry_run):
        """Fix targets with stale celery_id."""
        # Get all targets with celery_id set
        targets = models.Target.objects.exclude(celery_id__isnull=True)

        self.stdout.write(f"Checking {targets.count()} target(s) with active celery_id")

        stale_targets = []

        for target in targets:
            # Check if task still exists in Celery
            from celery.result import AsyncResult
            result = AsyncResult(target.celery_id, app=celery.app)

            # Check if task is in terminal state
            if result.state in ['SUCCESS', 'FAILURE', 'REVOKED', 'REJECTED']:
                stale_targets.append((target, result.state))

        if not stale_targets:
            self.stdout.write(self.style.SUCCESS("No stale tasks found"))
            return

        self.stdout.write(f"Found {len(stale_targets)} stale task(s)")

        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN - no changes will be made"))
            for target, state in stale_targets:
                self.stdout.write(
                    f"  Would fix: {target.id} - {target.name} (celery_id: {target.celery_id}, state: {state})"
                )
        else:
            for target, state in stale_targets:
                self.stdout.write(
                    f"Fixing: {target.id} - {target.name} (celery_id: {target.celery_id}, state: {state})"
                )
                target.celery_id = None
                target.celery_chain_ids = []
                target.celery_pid = None
                if 'FAILURE' in state or 'REVOKED' in state:
                    target.state = 'failed'
                target.save()

            self.stdout.write(self.style.SUCCESS(f"Fixed {len(stale_targets)} target(s)"))

    def _cleanup_orphaned_directories(self, dry_run):
        """Remove directories without corresponding database records."""
        targets_path = settings.TARGETS_PATH

        if not os.path.exists(targets_path):
            self.stdout.write("Targets directory does not exist")
            return

        # Get all target IDs from database
        db_ids = set(models.Target.objects.values_list('id', flat=True))

        # Get all directories
        orphaned = []
        for item in os.listdir(targets_path):
            item_path = os.path.join(targets_path, item)
            if os.path.isdir(item_path):
                try:
                    dir_id = int(item)
                    if dir_id not in db_ids:
                        orphaned.append((item, item_path))
                except ValueError:
                    # Not a numeric directory
                    pass

        if not orphaned:
            self.stdout.write(self.style.SUCCESS("No orphaned directories found"))
            return

        self.stdout.write(f"Found {len(orphaned)} orphaned director(ies)")

        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN - no changes will be made"))
            for dir_name, dir_path in orphaned:
                size = self._get_dir_size(dir_path)
                self.stdout.write(f"  Would delete: {dir_name} ({self._format_size(size)})")
        else:
            total_size = 0
            for dir_name, dir_path in orphaned:
                size = self._get_dir_size(dir_path)
                total_size += size
                self.stdout.write(f"Deleting: {dir_name} ({self._format_size(size)})")
                shutil.rmtree(dir_path)

            self.stdout.write(
                self.style.SUCCESS(f"Deleted {len(orphaned)} director(ies), freed {self._format_size(total_size)}")
            )

    def _check_system(self):
        """Check system status."""
        self.stdout.write(self.style.HTTP_INFO("System Status Check"))
        self.stdout.write("=" * 60)

        # Check Redis
        self.stdout.write("\n1. Redis Connection:")
        try:
            from redis import Redis
            r = Redis.from_url(settings.CELERY_BROKER_URL)
            r.ping()
            self.stdout.write(self.style.SUCCESS("   ✓ Redis is running"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   ✗ Redis error: {e}"))

        # Check Celery workers
        self.stdout.write("\n2. Celery Workers:")
        try:
            inspect = celery.app.control.inspect(timeout=1.0)
            active = inspect.active()
            if active:
                worker_count = len(active)
                self.stdout.write(self.style.SUCCESS(f"   ✓ {worker_count} worker(s) active"))
                for worker_name, tasks in active.items():
                    self.stdout.write(f"     - {worker_name}: {len(tasks)} task(s)")
            else:
                self.stdout.write(self.style.WARNING("   ⚠ No active workers found"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   ✗ Celery error: {e}"))

        # Check disk space
        self.stdout.write("\n3. Disk Space:")
        import shutil
        try:
            usage = shutil.disk_usage(settings.BASE_DIR)
            total = usage.total
            used = usage.used
            free = usage.free
            percent = (used / total) * 100

            self.stdout.write(f"   Total: {self._format_size(total)}")
            self.stdout.write(f"   Used:  {self._format_size(used)} ({percent:.1f}%)")
            self.stdout.write(f"   Free:  {self._format_size(free)}")

            if percent > 90:
                self.stdout.write(self.style.ERROR("   ✗ Disk space is critically low!"))
            elif percent > 75:
                self.stdout.write(self.style.WARNING("   ⚠ Disk space is getting low"))
            else:
                self.stdout.write(self.style.SUCCESS("   ✓ Disk space is adequate"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   ✗ Error checking disk: {e}"))

        # Check targets directory
        self.stdout.write("\n4. Targets Directory:")
        targets_path = settings.TARGETS_PATH
        if os.path.exists(targets_path):
            dir_count = len([d for d in os.listdir(targets_path) if os.path.isdir(os.path.join(targets_path, d))])
            total_size = self._get_dir_size(targets_path)
            self.stdout.write(f"   Path: {targets_path}")
            self.stdout.write(f"   Directories: {dir_count}")
            self.stdout.write(f"   Total size: {self._format_size(total_size)}")
        else:
            self.stdout.write(self.style.WARNING(f"   ⚠ Directory does not exist: {targets_path}"))

        # Check templates
        self.stdout.write("\n5. Templates:")
        spilling = self._find_spilling_comments()

        if spilling:
            self.stdout.write(self.style.ERROR(
                f"   ✗ {len(spilling)} template comment(s) spilling onto the page"))
            for path, number, line in spilling:
                self.stdout.write(f"     - {path}:{number}: {line}")
        else:
            self.stdout.write(self.style.SUCCESS("   ✓ No template comments spilling onto the page"))

        self.stdout.write("")

    def _find_spilling_comments(self):
        """Template comments that carry on past the line they start on.

        Django's {# #} is a single-line comment: it ends at the newline whether
        or not anything closed it, so the rest of a comment written across two
        lines is rendered into the page for everyone to read. Nothing warns
        about it - not the template engine, not the tests - and it has reached
        users twice, so it is checked for here.
        """
        found = []

        for directory in [os.path.join(settings.BASE_DIR, 'lcserver', 'templates')]:
            for root, _, files in os.walk(directory):
                for name in sorted(files):
                    if not name.endswith('.html'):
                        continue

                    path = os.path.join(root, name)

                    with open(path, encoding='utf-8', errors='replace') as f:
                        for number, line in enumerate(f, 1):
                            if line.count('{#') > line.count('#}'):
                                found.append((os.path.relpath(path, settings.BASE_DIR),
                                              number, line.strip()[:70]))

        return found

    def _show_statistics(self):
        """Show statistics about targets."""
        self.stdout.write(self.style.HTTP_INFO("Target Statistics"))
        self.stdout.write("=" * 60)

        # Total targets
        total = models.Target.objects.count()
        self.stdout.write(f"\nTotal targets: {total}")

        # By state
        self.stdout.write("\nBy state:")
        from django.db.models import Count
        states = models.Target.objects.values('state').annotate(count=Count('state')).order_by('-count')
        for item in states:
            self.stdout.write(f"  {item['state']:<30} {item['count']:>5}")

        # By user
        self.stdout.write("\nBy user:")
        users = models.Target.objects.values('user__username').annotate(count=Count('user')).order_by('-count')
        for item in users[:10]:  # Top 10 users
            self.stdout.write(f"  {item['user__username']:<30} {item['count']:>5}")

        # Running tasks
        running = models.Target.objects.exclude(celery_id__isnull=True).count()
        if running > 0:
            self.stdout.write(f"\nRunning tasks: {running}")

        # Recent activity
        from datetime import timedelta
        from django.utils import timezone

        recent_date = timezone.now() - timedelta(days=7)
        recent = models.Target.objects.filter(modified__gte=recent_date).count()
        self.stdout.write(f"\nActive in last 7 days: {recent}")

        # Disk usage
        targets_path = settings.TARGETS_PATH
        if os.path.exists(targets_path):
            total_size = self._get_dir_size(targets_path)
            avg_size = total_size / total if total > 0 else 0
            self.stdout.write(f"\nTotal disk usage: {self._format_size(total_size)}")
            self.stdout.write(f"Average per target: {self._format_size(avg_size)}")

        self.stdout.write("")

    def _get_dir_size(self, path):
        """Get directory size recursively."""
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
        return total

    def _format_size(self, size):
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
