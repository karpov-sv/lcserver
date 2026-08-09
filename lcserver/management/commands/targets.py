from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

import os

from lcserver import models


class Command(BaseCommand):
    help = 'Manage targets: list, show details, delete'

    def add_arguments(self, parser):
        # List targets
        parser.add_argument(
            '-l', '--list',
            action='store_true',
            dest='do_list',
            help='List all targets'
        )

        # Sort field
        parser.add_argument(
            '-s', '--sort',
            type=str,
            dest='sort_field',
            default='modified',
            choices=['id', 'name', 'created', 'modified', 'completed', 'state', 'user'],
            help='Sort field for listing (default: modified)'
        )

        # Show details
        parser.add_argument(
            '-i', '--info',
            action='store_true',
            dest='show_info',
            help='Show detailed information about target(s)'
        )

        # Delete targets
        parser.add_argument(
            '-d', '--delete',
            action='store_true',
            dest='do_delete',
            help='Delete target(s)'
        )

        # Cleanup (delete files but keep record)
        parser.add_argument(
            '--cleanup',
            action='store_true',
            dest='do_cleanup',
            help='Clean up target files (keep database record)'
        )

        # Filter by user
        parser.add_argument(
            '-u', '--user',
            type=str,
            dest='username',
            help='Filter by username'
        )

        # Filter by state
        parser.add_argument(
            '--state',
            type=str,
            dest='state',
            help='Filter by state'
        )

        # Limit results
        parser.add_argument(
            '-n', '--limit',
            type=int,
            dest='limit',
            help='Limit number of results'
        )

        # Target IDs/names
        parser.add_argument(
            'targets',
            nargs='*',
            type=str,
            help='Target ID(s) or name(s) to operate on'
        )

    def handle(self, *args, **options):
        # Change to project root
        os.chdir(settings.BASE_DIR)

        if options['do_list']:
            self._list_targets(options)

        elif options['show_info']:
            if not options['targets']:
                raise CommandError("Please specify target ID(s) or name(s)")
            for identifier in options['targets']:
                self._show_info(identifier)

        elif options['do_delete']:
            if not options['targets']:
                raise CommandError("Please specify target ID(s) to delete")
            for identifier in options['targets']:
                self._delete_target(identifier)

        elif options['do_cleanup']:
            if not options['targets']:
                raise CommandError("Please specify target ID(s) to clean up")
            for identifier in options['targets']:
                self._cleanup_target(identifier)

        else:
            # Default: list targets
            self._list_targets(options)

    def _list_targets(self, options):
        """List all targets."""
        # Build query
        targets = models.Target.objects.all()

        # Apply filters
        if options['username']:
            targets = targets.filter(user__username=options['username'])

        if options['state']:
            targets = targets.filter(state__icontains=options['state'])

        # Apply sorting
        sort_field = options['sort_field']
        if sort_field == 'user':
            sort_field = 'user__username'
        targets = targets.order_by(f'-{sort_field}')

        # Apply limit
        if options['limit']:
            targets = targets[:options['limit']]

        # Display header
        self.stdout.write(
            f"{'ID':<6} {'Created':<19} {'Modified':<19} {'User':<15} {'State':<25} {'Name'}"
        )
        self.stdout.write('-' * 120)

        # Display targets
        count = 0
        for target in targets:
            created = target.created.strftime('%Y-%m-%d %H:%M:%S')
            modified = target.modified.strftime('%Y-%m-%d %H:%M:%S')
            user = target.user.username[:14]
            state = target.state[:24]
            name = target.name[:40]

            # Color code by state
            if 'failed' in target.state.lower():
                style = self.style.ERROR
            elif 'acquired' in target.state.lower() or 'done' in target.state.lower():
                style = self.style.SUCCESS
            elif target.celery_id:
                style = self.style.HTTP_INFO
            else:
                style = lambda x: x

            self.stdout.write(
                style(f"{target.id:<6} {created} {modified} {user:<15} {state:<25} {name}")
            )
            count += 1

        self.stdout.write(f"\nTotal: {count} target(s)")

    def _show_info(self, identifier):
        """Show detailed information about a target."""
        target = self._get_target(identifier)
        if not target:
            self.stdout.write(self.style.ERROR(f"Target not found: {identifier}"))
            return

        self.stdout.write(self.style.HTTP_INFO(f"\n{'='*60}"))
        self.stdout.write(self.style.HTTP_INFO(f"Target {target.id}"))
        self.stdout.write(self.style.HTTP_INFO(f"{'='*60}"))
        self.stdout.write(f"Name:       {target.name}")
        self.stdout.write(f"Title:      {target.title or '(none)'}")
        self.stdout.write(f"User:       {target.user.username}")
        self.stdout.write(f"State:      {target.state}")
        self.stdout.write(f"Created:    {target.created}")
        self.stdout.write(f"Modified:   {target.modified}")
        self.stdout.write(f"Completed:  {target.completed}")
        self.stdout.write(f"Path:       {target.path()}")

        if target.celery_id:
            self.stdout.write(f"Celery ID:  {target.celery_id}")
            if target.celery_chain_ids:
                self.stdout.write(f"Chain IDs:  {len(target.celery_chain_ids)} task(s)")
            if target.celery_pid:
                self.stdout.write(f"Process ID: {target.celery_pid}")

        # Show config keys
        if target.config:
            self.stdout.write(f"\nConfig keys: {', '.join(target.config.keys())}")

        # List files
        basepath = target.path()
        if os.path.exists(basepath):
            files = sorted(os.listdir(basepath))
            if files:
                self.stdout.write(f"\nFiles ({len(files)}):")
                total_size = 0
                for f in files:
                    fpath = os.path.join(basepath, f)
                    size = os.path.getsize(fpath)
                    total_size += size
                    size_str = self._format_size(size)
                    self.stdout.write(f"  {f:<40} {size_str:>10}")
                self.stdout.write(f"Total size: {self._format_size(total_size)}")
            else:
                self.stdout.write("\nNo files found")
        else:
            self.stdout.write("\nDirectory does not exist")

        self.stdout.write("")

    def _delete_target(self, identifier):
        """Delete a target."""
        target = self._get_target(identifier)
        if not target:
            self.stdout.write(self.style.ERROR(f"Target not found: {identifier}"))
            return

        # Store ID before deletion
        target_id = target.id
        target_name = target.name

        # Confirm deletion
        self.stdout.write(
            self.style.WARNING(f"Deleting target {target_id}: {target_name}")
        )

        target.delete()
        self.stdout.write(self.style.SUCCESS(f"Target {target_id} deleted"))

    def _cleanup_target(self, identifier):
        """Clean up target files."""
        import shutil

        target = self._get_target(identifier)
        if not target:
            self.stdout.write(self.style.ERROR(f"Target not found: {identifier}"))
            return

        basepath = target.path()
        if os.path.exists(basepath):
            # Remove all files
            for item in os.listdir(basepath):
                item_path = os.path.join(basepath, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)

            self.stdout.write(self.style.SUCCESS(f"Cleaned up target {target.id}"))
            target.state = 'cleaned'
            target.save()
        else:
            self.stdout.write(self.style.WARNING(f"No files found for target {target.id}"))

    def _get_target(self, identifier):
        """Get target by ID or name."""
        try:
            # Try as ID first
            target_id = int(identifier)
            return models.Target.objects.get(id=target_id)
        except (ValueError, models.Target.DoesNotExist):
            # Try as name (get first match)
            return models.Target.objects.filter(name=identifier).first()

    def _format_size(self, size):
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
