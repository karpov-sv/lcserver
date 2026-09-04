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
from lcserver.ingest import ariadne, store


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
            '--ingest-tlusty', dest='tlusty', default=None,
            help="A TLUSTY grid in the merged ASCII form Cloudy reads - the "
                 "one file, not the per-model originals")

        parser.add_argument(
            '--ingest-cdbs', dest='cdbs', default=None,
            help='An atlas in the layout STScI distributes them in - a '
                 'k93models or ck04models directory, a subdirectory per '
                 'metallicity and one FITS per temperature')

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

        if options['tlusty']:
            return self.ingest_tlusty(options)

        if options['cdbs']:
            return self.ingest_cdbs(options)

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

        source = options['powr']
        if not os.path.isdir(source):
            raise CommandError(f'{source} is not a directory')

        target, name = self.where(options, default=powr.grid_name(source))
        cube, spectra = self.paths(target, name, options['force'])

        powr.ingest(source, cube, spectra, name=name,
                    label=options['label'], description=options['description'],
                    verbose=self.stdout.write)

        self.written(name, cube, spectra, target)

    def ingest_tlusty(self, options):
        """Turn a merged TLUSTY file into a cube and a spectrum file.

        The cube it produces should be the one that is already installed - that
        file is what astroARIADNE built it from - so the two are compared band
        by band afterwards and the ratios printed. Anything but ones means the
        reading is wrong somewhere.
        """
        from lcserver.ingest import tlusty

        source = options['tlusty']
        if not os.path.isfile(source):
            raise CommandError(f'{source} is not a file')

        target, name = self.where(options, default='tlusty')
        cube, spectra = self.paths(target, name, options['force'])

        installed = (sedfit.grid_registry().get(name) or {}).get('path')

        tlusty.ingest(source, cube, spectra, name=name,
                      label=options['label'] or 'TLUSTY',
                      description=options['description'] or 'the hot-star models',
                      verbose=self.stdout.write)

        if installed and os.path.abspath(installed) != os.path.abspath(cube):
            store.compare(cube, installed, verbose=self.stdout.write)

        self.written(name, cube, spectra, target)

    def ingest_cdbs(self, options):
        """Turn an atlas in STScI's layout into a cube and a spectrum file.

        The cubes already installed for these were built from the same atlases
        and stop at 12000 K, which is less than half of either; the comparison
        afterwards is over the part they share.
        """
        from lcserver.ingest import cdbs

        source = options['cdbs']
        if not os.path.isdir(source):
            raise CommandError(f'{source} is not a directory')

        known, label, description = cdbs.atlas_name(source)
        target, name = self.where(options, default=known)
        cube, spectra = self.paths(target, name, options['force'])

        installed = (sedfit.grid_registry().get(name) or {}).get('path')

        cdbs.ingest(source, cube, spectra, name=name,
                    label=options['label'] or label,
                    description=options['description'] or description,
                    verbose=self.stdout.write)

        if installed and os.path.abspath(installed) != os.path.abspath(cube):
            store.compare(cube, installed, verbose=self.stdout.write,
                          at=((5000., 4.0, 0.0), (8000., 4.0, 0.0),
                              (10000., 4.0, 0.0), (10000., 3.0, -0.5)))

        self.written(name, cube, spectra, target)

    # ------------------------------------------------------- where it all goes

    def where(self, options, default=None):
        """The directory to write into, and what to call the grid."""
        target = options['to'] or getattr(settings, 'SEDFIT_GRIDS', '')
        if not target:
            raise CommandError('nowhere to write to: pass --to, or set '
                               'SEDFIT_GRIDS to where the grids should live')

        os.makedirs(target, exist_ok=True)
        return target, options['name'] or default

    def paths(self, target, name, force):
        """Where the two files go, refusing to write over what is there."""
        cube = os.path.join(target, f'{name}.h5')
        spectra = os.path.join(target, f'{name}.spectra.h5')

        for path in (cube, spectra):
            if os.path.exists(path) and not force:
                raise CommandError(f'{path} is already there - --force to '
                                   f'write over it')

        return cube, spectra

    def written(self, name, cube, spectra, target):
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
