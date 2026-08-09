from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth.models import User
from django.conf import settings

import os
import sys
from functools import partial

from lcserver import models
from lcserver import processing
from lcserver import celery_tasks
from lcserver import surveys


class Command(BaseCommand):
    help = 'Test target processing steps synchronously (for debugging)'

    def add_arguments(self, parser):
        # Target selection
        parser.add_argument(
            'target',
            type=str,
            help='Target ID or name to process'
        )

        # Processing steps
        # Only include sources with processing functions
        available_steps = [
            source_id for source_id, config in surveys.SURVEY_SOURCES.items()
            if config.get('processing_function') is not None
        ]
        parser.add_argument(
            '-s', '--step',
            action='append',
            dest='steps',
            choices=available_steps + ['all'],
            help='Processing step(s) to run. Can be specified multiple times.'
        )

        # Create new target
        parser.add_argument(
            '-n', '--new',
            action='store_true',
            dest='create_new',
            help='Create a new target with the given name'
        )

        # User for new target
        parser.add_argument(
            '-u', '--user',
            type=str,
            dest='username',
            help='Username for new target (defaults to first staff user)'
        )

        # Verbose output
        parser.add_argument(
            '--verbose',
            action='store_true',
            dest='verbose',
            help='Verbose output (print all processing messages)'
        )

        # Show plots
        parser.add_argument(
            '--show',
            action='store_true',
            dest='show',
            help='Show plots interactively (for debugging)'
        )

        # Continue on error
        parser.add_argument(
            '--continue-on-error',
            action='store_true',
            dest='continue_on_error',
            help='Continue processing even if a step fails'
        )

        # Debug mode
        parser.add_argument(
            '--debug',
            action='store_true',
            dest='debug',
            help='Debug mode: do not catch exceptions (for use with debugger)'
        )

        # Post-mortem debugging
        parser.add_argument(
            '--pdb',
            action='store_true',
            dest='use_pdb',
            help='Drop into pdb debugger on exception'
        )

    def handle(self, *args, **options):
        # Change to project root
        os.chdir(settings.BASE_DIR)

        # Get or create target
        if options['create_new']:
            target = self._create_target(options['target'], options['username'])
        else:
            target = self._get_target(options['target'])

        if not target:
            raise CommandError(f"Target not found: {options['target']}")

        self.stdout.write(self.style.SUCCESS(f"\nTarget: {target.id} - {target.name}"))
        self.stdout.write(f"State: {target.state}")
        self.stdout.write(f"Path: {target.path()}\n")

        # Determine steps to run
        steps = self._get_steps(options['steps'])

        if not steps:
            self.stdout.write(self.style.ERROR("No steps specified. Use -s/--step to specify steps."))
            # Show available steps (exclude lightcurve-only sources)
            available_steps = [
                source_id for source_id, config in surveys.SURVEY_SOURCES.items()
                if config.get('processing_function') is not None
            ]
            self.stdout.write(f"Available steps: {', '.join(sorted(available_steps, key=lambda k: surveys.SURVEY_SOURCES[k]['order']))}, all")
            return

        self.stdout.write(f"Steps to run: {', '.join(steps)}\n")

        # Run each step
        success_count = 0
        fail_count = 0

        for step in steps:
            self.stdout.write(self.style.HTTP_INFO(f"\n{'='*60}"))
            self.stdout.write(self.style.HTTP_INFO(f"Running step: {step}"))
            self.stdout.write(self.style.HTTP_INFO(f"{'='*60}\n"))

            # Debug mode: don't catch exceptions
            if options['debug']:
                self._run_step(target, step, options['verbose'], options['show'], debug=True)
                self.stdout.write(self.style.SUCCESS(f"\n✓ Step '{step}' completed successfully"))
                success_count += 1
                continue

            # Normal mode: catch exceptions
            try:
                self._run_step(target, step, options['verbose'], options['show'], debug=False)
                self.stdout.write(self.style.SUCCESS(f"\n✓ Step '{step}' completed successfully"))
                success_count += 1

            except Exception as e:
                if options['use_pdb']:
                    # Drop into post-mortem debugger
                    import pdb
                    import traceback
                    import sys

                    self.stdout.write(self.style.ERROR(f"\n✗ Step '{step}' failed: {e}"))
                    traceback.print_exc()
                    self.stdout.write(self.style.WARNING("\nEntering post-mortem debugger..."))
                    pdb.post_mortem(sys.exc_info()[2])

                    # After debugging, continue or stop based on flag
                    if not options['continue_on_error']:
                        raise CommandError(f"Processing stopped due to error in step '{step}'")
                else:
                    self.stdout.write(self.style.ERROR(f"\n✗ Step '{step}' failed: {e}"))

                fail_count += 1

                if not options['continue_on_error'] and not options['use_pdb']:
                    raise CommandError(f"Processing stopped due to error in step '{step}'")

        # Summary
        self.stdout.write(self.style.HTTP_INFO(f"\n{'='*60}"))
        self.stdout.write(self.style.HTTP_INFO("SUMMARY"))
        self.stdout.write(self.style.HTTP_INFO(f"{'='*60}"))
        self.stdout.write(f"Total steps: {len(steps)}")
        self.stdout.write(self.style.SUCCESS(f"Succeeded: {success_count}"))
        if fail_count > 0:
            self.stdout.write(self.style.ERROR(f"Failed: {fail_count}"))
        self.stdout.write(f"\nFinal state: {target.state}")
        self.stdout.write(f"Output directory: {target.path()}\n")

    def _get_target(self, identifier):
        """Get target by ID or name."""
        try:
            # Try as ID first
            target_id = int(identifier)
            return models.Target.objects.get(id=target_id)
        except ValueError:
            # Try as name
            targets = models.Target.objects.filter(name=identifier)
            if targets.count() == 1:
                return targets.first()
            elif targets.count() > 1:
                self.stdout.write(self.style.WARNING(f"Multiple targets found with name '{identifier}':"))
                for t in targets:
                    self.stdout.write(f"  {t.id} - {t.name} (user: {t.user.username}, state: {t.state})")
                raise CommandError("Please specify target by ID")
            else:
                return None
        except models.Target.DoesNotExist:
            return None

    def _create_target(self, name, username):
        """Create a new target."""
        if username:
            user = User.objects.filter(username=username).first()
            if not user:
                raise CommandError(f"User not found: {username}")
        else:
            user = User.objects.filter(is_staff=True).order_by('id').first()
            if not user:
                raise CommandError("No staff user found. Please specify --user")

        target = models.Target(name=name, user=user, state='initial')
        target.save()

        self.stdout.write(self.style.SUCCESS(f"Created new target: {target.id}"))
        return target

    def _get_steps(self, steps):
        """Determine which steps to run."""
        if not steps:
            return []

        if 'all' in steps:
            return surveys.get_survey_ids_for_everything()

        return steps

    def _run_step(self, target, step, verbose, show, debug=False):
        """Run a single processing step synchronously."""
        basepath = target.path()
        config = target.config
        config['target_name'] = target.name

        # Setup logging function
        logname = os.path.join(basepath, f'{step}.log')

        if verbose:
            # Print to both stdout and file
            def log(*args, clear=False, **kwargs):
                message = ' '.join(str(arg) for arg in args)
                self.stdout.write(message)
                processing.print_to_file(message, logname=logname, clear=clear)
        else:
            # Just to file
            log = partial(processing.print_to_file, logname=logname)

        log(clear=True)
        log(f"Starting step: {step}")
        log(f"Target: {target.name}")
        log(f"Path: {basepath}\n")

        # Debug mode: no exception handling
        if debug:
            survey_config = surveys.get_survey_source(step)
            if survey_config:
                # Skip sources without processing functions (lightcurve-only sources)
                if survey_config.get('processing_function') is None:
                    raise ValueError(f"Source '{step}' has no processing function (lightcurve-only source)")

                processing_func = getattr(processing, survey_config['processing_function'])
                processing_func(config, basepath=basepath, verbose=log, show=show)
                target.state = survey_config['state_acquired']
            else:
                raise ValueError(f"Unknown step: {step}")

            log(f"\nStep '{step}' completed successfully")

            # Save in debug mode too
            celery_tasks.fix_config(config)
            target.save()

            # Show output files
            if os.path.exists(basepath):
                files = os.listdir(basepath)
                if files:
                    log(f"\nGenerated files in {basepath}:")
                    for f in sorted(files):
                        fpath = os.path.join(basepath, f)
                        size = os.path.getsize(fpath)
                        log(f"  {f} ({size} bytes)")

            return

        # Normal mode: with exception handling
        try:
            survey_config = surveys.get_survey_source(step)
            if survey_config:
                # Skip sources without processing functions (lightcurve-only sources)
                if survey_config.get('processing_function') is None:
                    raise ValueError(f"Source '{step}' has no processing function (lightcurve-only source)")

                processing_func = getattr(processing, survey_config['processing_function'])
                processing_func(config, basepath=basepath, verbose=log, show=show)
                target.state = survey_config['state_acquired']
            else:
                raise ValueError(f"Unknown step: {step}")

            log(f"\nStep '{step}' completed successfully")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            log(f"\nError in step '{step}':\n{error_msg}")
            target.state = 'failed'
            raise

        finally:
            # Fix numpy types and save
            celery_tasks.fix_config(config)
            target.save()

            # Show output files
            if os.path.exists(basepath):
                files = os.listdir(basepath)
                if files:
                    log(f"\nGenerated files in {basepath}:")
                    for f in sorted(files):
                        fpath = os.path.join(basepath, f)
                        size = os.path.getsize(fpath)
                        log(f"  {f} ({size} bytes)")
