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
            '--ingest-btsettl', dest='btsettl', default=None,
            help="BT-Settl's medium-resolution grid file - its spectra only, "
                 'the cube here already being the wider grid')

        parser.add_argument(
            '--ingest-bosz', dest='bosz', default=None,
            help='A BOSZ download - a directory per metallicity of one file '
                 'per model, and the wavelength grid beside them')

        parser.add_argument(
            '--ingest-koester', dest='koester', default=None,
            help="Koester's white-dwarf models as SVO hands them out - the "
                 'directory of one file per model')

        parser.add_argument(
            '--scatter', dest='scatter', nargs='*', default=None,
            metavar='GRID',
            help='Rewrite grids stored as a lattice as the models they have, '
                 'which recovers the band of parameter space a lattice with '
                 'holes refuses. All of them if none are named.')

        parser.add_argument(
            '--ingest-cdbs', dest='cdbs', default=None,
            help='An atlas in the layout STScI distributes them in - a '
                 'k93models or ck04models directory, a subdirectory per '
                 'metallicity and one FITS per temperature')

        parser.add_argument(
            '--xp', dest='xp', nargs='*', default=None, metavar='GRID',
            help="Write each grid's Gaia XP bins beside it, from its spectra, "
                 'so that the spectrum can be fitted the way photometry is. '
                 'All of them that have spectra if none are named.')

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

        if options['btsettl']:
            return self.ingest_btsettl(options)

        if options['bosz']:
            return self.ingest_bosz(options)

        if options['koester']:
            return self.ingest_koester(options)

        if options['scatter'] is not None:
            return self.scatter(options)

        if options['xp'] is not None:
            return self.xp(options)

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

    def ingest_btsettl(self, options):
        """Write BT-Settl's spectra beside the cube that is already here.

        Only the spectra: the grid file covers rather more than half of the
        cube's models, and a cube from it would lose the cool and metal-poor
        ends that are the reason for having this grid at all.
        """
        from lcserver.ingest import btsettl

        source = options['btsettl']
        if not os.path.isfile(source):
            raise CommandError(f'{source} is not a file')

        target, name = self.where(options, default='btsettl')
        spectra = os.path.join(target, f'{name}.spectra.h5')

        if os.path.exists(spectra) and not options['force']:
            raise CommandError(f'{spectra} is already there - --force to '
                               f'write over it')

        btsettl.ingest(source, spectra, name=name, verbose=self.stdout.write)

        self.stdout.write(self.style.SUCCESS(
            f'\n{name}: {os.path.getsize(spectra) / 1e6:.1f} MB of spectra '
            f'in {target}'))

    def ingest_bosz(self, options):
        """Turn a BOSZ download into a cube and a spectrum file.

        The cube already installed is the same composition of the same library,
        so the two are compared afterwards; what this adds is the spectra, out
        to thirty-two microns.
        """
        from lcserver.ingest import bosz

        source = options['bosz']
        if not os.path.isdir(source):
            raise CommandError(f'{source} is not a directory')

        target, name = self.where(options, default='bosz')
        cube, spectra = self.paths(target, name, options['force'])

        installed = (sedfit.grid_registry().get(name) or {}).get('path')

        bosz.ingest(source, cube, spectra, name=name,
                    label=options['label'] or 'BOSZ',
                    description=options['description']
                    or 'ATLAS9 and MARCS, and the only grid here with spectra '
                       'past five microns for a warm star',
                    verbose=self.stdout.write)

        if installed and os.path.abspath(installed) != os.path.abspath(cube):
            store.compare(cube, installed, verbose=self.stdout.write,
                          at=((4000., 4.5, 0.0), (6000., 4.5, 0.0),
                              (10000., 4.0, 0.0), (6000., 2.5, -1.0)))

        self.written(name, cube, spectra, target)

    def ingest_koester(self, options):
        """Turn a directory of Koester models into a cube and its spectra.

        The cube already installed was built from the same models, so the two
        are compared afterwards; what this adds is the spectra, which it has
        none of.
        """
        from lcserver.ingest import koester

        source = options['koester']
        if not os.path.isdir(source):
            raise CommandError(f'{source} is not a directory')

        target, name = self.where(options, default='koester')
        cube, spectra = self.paths(target, name, options['force'])

        installed = (sedfit.grid_registry().get(name) or {}).get('path')

        koester.ingest(source, cube, spectra, name=name,
                       label=options['label'] or 'Koester',
                       description=options['description'] or 'white dwarfs',
                       verbose=self.stdout.write)

        if installed and os.path.abspath(installed) != os.path.abspath(cube):
            store.compare(cube, installed, verbose=self.stdout.write,
                          bands=['GALEX_NUV', 'GROUND_JOHNSON_U', 'PS1_g',
                                 'GROUND_JOHNSON_V', '2MASS_J'],
                          at=((10000., 8.0, 0.0), (20000., 8.0, 0.0),
                              (40000., 7.5, 0.0), (60000., 9.0, 0.0)))

        self.written(name, cube, spectra, target)

    # --------------------------------------------------------------- scatter

    def scatter(self, options):
        """Rewrite lattices as the models they hold.

        Nothing is downloaded and nothing recomputed - the fluxes written out
        are the ones read in, with the empty nodes dropped - so what this
        changes is only how much of the grid can be reached. Both numbers are
        printed per grid: what it gained, and how far the two ways of
        interpolating differ where they overlap.
        """
        import shutil
        import tempfile

        from lcserver.ingest import scatter

        registry = sedfit.grid_registry()
        wanted = options['scatter'] or sorted(registry)
        unknown = [_ for _ in wanted if _ not in registry]
        if unknown:
            raise CommandError(f"no grid called {', '.join(unknown)}")

        for name in wanted:
            path = registry[name]['path']

            # The grid as it was, kept until the comparison has been made
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                before = tmp.name
            shutil.copy2(path, before)

            try:
                if scatter.convert(path, verbose=self.stdout.write):
                    scatter.compare(before, path, verbose=self.stdout.write)
            finally:
                os.unlink(before)

    def xp(self, options):
        """Write the Gaia XP bins of each grid that has spectra to bin.

        This is what lets an XP spectrum be fitted rather than only looked at:
        with the bins in a cube the fit interpolates them at its own parameters,
        exactly as it does a filter, and nothing has to stand in for the fit.
        A grid with no spectra has nothing to bin and is skipped, which is said
        rather than passed over.
        """
        from lcserver.ingest import xp

        registry = sedfit.grid_registry()
        wanted = options['xp'] or sorted(registry)
        unknown = [_ for _ in wanted if _ not in registry]
        if unknown:
            raise CommandError(f"no grid called {', '.join(unknown)}")

        target = options['to'] or os.path.dirname(
            registry[wanted[0]]['path'])

        for name in wanted:
            entry = registry[name]
            if not entry.get('spectra'):
                self.stdout.write(f'{name}: no spectra to bin')
                continue

            out = os.path.join(target, f'{name}.xp.h5')
            if os.path.exists(out) and not options['force']:
                self.stdout.write(f'{name}: {out} is already there '
                                  f'- --force to write over it')
                continue

            self.stdout.write(f'\n{name}:')
            try:
                count = xp.build(entry['spectra'], out, name,
                                 label=entry.get('label'),
                                 verbose=self.stdout.write)
            except Exception as e:
                self.stdout.write(f'  failed: {type(e).__name__}: {e}')
                continue

            self.stdout.write(f'  {count} models written to {out}'
                              f' ({os.path.getsize(out) / 1e6:.1f} MB)')
            try:
                xp.compare(out, entry['spectra'], name,
                           verbose=self.stdout.write)
            except Exception as e:
                self.stdout.write(f'  could not compare: {type(e).__name__}: {e}')

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
        header = f"{'grid':16s}{'Teff, K':>16}  {'spectra':<9}{'reach':>7}  label"
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
                f"{name:16s}{span:>16}  {spectra:<9}{reach:>7}  "
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
