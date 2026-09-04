"""The model grids the SED fitter interpolates, as files on disk.

A grid is a file, and the directory is the registry: the fitter reads what is
there rather than a table in the source, so adding one is putting one there.
This command is what puts them there - for now by splitting astroARIADNE's own
layout, which keeps every spectrum of every grid in one file of some gigabytes,
into a cube and a spectrum file per grid.
"""

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

import os
import shutil

import h5py

from lcserver.processing import sedfit


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
            '--force', action='store_true', dest='force',
            help='Write over files that are already there')

    def handle(self, *args, **options):
        if options['show'] or not options['split']:
            return self.show()

        self.split(options)

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
            self.one(registry[name], target, cache, options['force'])

        self.stdout.write(self.style.SUCCESS(
            f'\n{len(wanted)} grid(s) written. Point SEDFIT_GRIDS at '
            f'{target} to read them.'))

    def one(self, entry, target, cache, force):
        name = entry['name']
        cube = os.path.join(target, f'{name}.h5')

        if os.path.exists(cube) and not force:
            self.stdout.write(f'{name:12s} cube is already there, left alone')
        else:
            shutil.copy2(entry['path'], cube)
            # What the file could not say for itself, said now: a grid built
            # here carries its own, and a copied one carries what we knew of it
            with h5py.File(cube, 'r+') as h:
                h.attrs['name'] = name
                if entry['label']:
                    h.attrs['label'] = entry['label']
                if entry['description']:
                    h.attrs['description'] = entry['description']
                if entry['reach_um']:
                    h.attrs['reach_um'] = float(entry['reach_um'])
                h.attrs['source'] = 'astroARIADNE, ' + os.path.basename(entry['path'])

            self.stdout.write(f'{name:12s} cube {os.path.getsize(cube) / 1e6:8.1f} MB')

        self.spectra(name, target, cache, force)

    def spectra(self, name, target, cache, force):
        out = os.path.join(target, f'{name}.spectra.h5')

        if os.path.exists(out) and not force:
            self.stdout.write(f'{"":12s} spectra are already there, left alone')
            return

        if not cache:
            return

        with h5py.File(cache, 'r') as h:
            if name not in h:
                self.stdout.write(f'{"":12s} no spectra for it in the cache')
                return

            group = h[name]
            # Dataset by dataset rather than array by array, so the chunking
            # and the compression come across as they are - one spectrum to a
            # chunk, which is how one is read
            with h5py.File(out, 'w') as f:
                for key in group:
                    h.copy(group[key], f, name=key)

                f.attrs['name'] = name
                f.attrs['source'] = 'astroARIADNE, ' + os.path.basename(cache)

        self.stdout.write(f'{"":12s} spectra {os.path.getsize(out) / 1e6:5.0f} MB')
