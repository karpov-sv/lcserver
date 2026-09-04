"""The model grids the SED fitter interpolates, as files on disk.

A grid is a file, and the directory is the registry: the fitter reads what is
there rather than a table in the source, so adding one is putting one there.
This command is what puts them there. What each kind of download actually means
is in ``lcserver.ingest``; here is only what to do it to, and what to say about
it afterwards.
"""

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

import os

from lcserver.processing import sedfit
from lcserver.ingest import ariadne


class Command(BaseCommand):
    help = 'Inspect the model grids, and split astroARIADNE\'s into per-grid files'

    def add_arguments(self, parser):
        parser.add_argument(
            '--list', action='store_true', dest='show',
            help='What the grid directory holds, and what is known of each')

        parser.add_argument(
            '--split', action='store_true', dest='split',
            help="Write one cube and one spectrum file per grid, from "
                 "astroARIADNE's directory and its single spectra cache")

        parser.add_argument(
            '--to', dest='to', default=None,
            help='Where to write them (default: the SEDFIT_GRIDS setting)')

        parser.add_argument(
            '--only', dest='only', nargs='+', default=None,
            help='Just these grids, by name')

        parser.add_argument(
            '--ingest-powr', dest='powr', default=None,
            help='A PoWR grid download to convert - the directory of _sed.txt '
                 'files and their modelparameters.txt')

        parser.add_argument(
            '--name', dest='name', default=None,
            help='What to call it (default: from the download directory)')

        parser.add_argument(
            '--label', dest='label', default=None,
            help='What to call it on the fitting form')

        parser.add_argument(
            '--description', dest='description', default=None,
            help='A few words about it, shown beside the label')

        parser.add_argument(
            '--force', action='store_true', dest='force',
            help='Write over files that are already there')

    def handle(self, *args, **options):
        if options['powr']:
            return self.ingest_powr(options)

        if options['show'] or not options['split']:
            return self.show()

        self.split(options)

    # ---------------------------------------------------------------- ingest

    def ingest_powr(self, options):
        """Turn a PoWR download into a cube and a spectrum file.

        The download is one file per model, each a wavelength and a flux in the
        log, at ten parsecs. What comes out is what every other grid here is:
        surface flux per filter to fit against, and the spectrum it came from
        to draw.
        """
        from lcserver.ingest import powr

        target = options['to'] or getattr(settings, 'SEDFIT_GRIDS', '')
        if not target:
            raise CommandError('nowhere to write to: pass --to, or set '
                               'SEDFIT_GRIDS to where the grids should live')

        source = options['powr']
        if not os.path.isdir(source):
            raise CommandError(f'{source} is not a directory')

        name = options['name'] or powr.grid_name(source)
        cube = os.path.join(target, f'{name}.h5')
        spectra = os.path.join(target, f'{name}.spectra.h5')

        for path in (cube, spectra):
            if os.path.exists(path) and not options['force']:
                raise CommandError(f'{path} is already there - --force to '
                                   f'write over it')

        os.makedirs(target, exist_ok=True)

        powr.ingest(source, cube, spectra, name=name,
                    label=options['label'], description=options['description'],
                    verbose=self.stdout.write)

        self.stdout.write(self.style.SUCCESS(
            f'\n{name}: {os.path.getsize(cube) / 1e6:.1f} MB of cube and '
            f'{os.path.getsize(spectra) / 1e6:.1f} MB of spectra in {target}'))

    # ------------------------------------------------------------------ list

    def show(self):
        registry = sedfit.grid_registry()
        if not registry:
            raise CommandError(f'no grids in {sedfit.grids_dir()}')

        cache = sedfit.spectra_cache_path()
        self.stdout.write(f'{len(registry)} grid(s) in {sedfit.grids_dir()}')
        if cache:
            self.stdout.write(f"astroARIADNE's spectra cache: {cache}"
                              f" ({os.path.getsize(cache) / 1e9:.2f} GB)")
        self.stdout.write('')

        offered = {g['name'] for g in sedfit.offered_grids()}
        header = f"{'grid':12s}{'Teff, K':>16}  {'spectra':<9}{'reach':>7}  label"
        self.stdout.write(header)

        for name in sorted(registry):
            entry = registry[name]
            span = (f"{entry.get('teff_lo', 0):.0f}-{entry.get('teff_hi', 0):.0f}"
                    if entry.get('teff_lo') else '')
            spectra = ('beside' if entry['spectra']
                       else 'cache' if sedfit.has_spectra(name)
                       else '-')
            reach = f"{entry['reach_um']:.1f} um" if entry['reach_um'] else ''

            self.stdout.write(
                f"{name:12s}{span:>16}  {spectra:<9}{reach:>7}  "
                f"{entry['label'] or ''}"
                + ('' if name in offered else '   (not offered - says nothing'
                                              ' about itself)'))

    # ----------------------------------------------------------------- split

    def split(self, options):
        target = options['to'] or getattr(settings, 'SEDFIT_GRIDS', '')
        if not target:
            raise CommandError('nowhere to write to: pass --to, or set '
                               'SEDFIT_GRIDS to where the grids should live')

        source = sedfit.grids_dir()
        if os.path.abspath(target) == os.path.abspath(source):
            raise CommandError('--to is where the grids are read from; write '
                               'the split somewhere of its own')

        registry = sedfit.grid_registry()
        wanted = options['only'] or sorted(registry)
        unknown = [_ for _ in wanted if _ not in registry]
        if unknown:
            raise CommandError(f"no grid called {', '.join(unknown)}")

        os.makedirs(target, exist_ok=True)
        cache = sedfit.spectra_cache_path()

        self.stdout.write(f'{source}\n  -> {target}\n')

        for name in wanted:
            ariadne.split(registry[name], target, cache=cache,
                          force=options['force'], verbose=self.stdout.write)

        self.stdout.write(self.style.SUCCESS(
            f'\n{len(wanted)} grid(s) written. Point SEDFIT_GRIDS at '
            f'{target} to read them.'))
