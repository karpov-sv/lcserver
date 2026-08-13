"""SPHEREx spectrophotometry acquisition module.

Not a light curve, and not quite a spectrum either as the other spectroscopic
sources here mean it. SPHEREx carries a linear variable filter over each of its
six detectors, so a single exposure measures a position at one wavelength - the
one its pixel happens to sit at. The spectrum is assembled out of the several
hundred exposures that have crossed the target since launch, each contributing
a point somewhere between 0.75 and 5 microns.

The time axis is real but sparse in a particular way: the survey sweeps the
whole sky twice a year, so the exposures arrive in visit windows a fortnight
long and six months apart, and within a window every exposure is at a different
wavelength. A per-wavelength light curve therefore has about three points in
it, which is why this is registered as a spectrum and not as photometry.

IRSA publishes only the Level 2 spectral images - there is no catalogue of
extracted spectra to query - so the measurements are made here. The obvious way
is IRSA's own cutout service, but it cannot select FITS extensions, and every
cutout therefore carries the file's 121-plane PSF cube: 4.9 MB per exposure of
which a few kB is the data. Three hundred exposures is then 1.7 GB and ten
minutes. So the images are read over HTTP byte ranges instead, taking the few
rows of IMAGE, VARIANCE and FLAGS that the source falls on, which is some
300 kB per exposure and a minute and a half for the same target.

To know where those rows are without reading the whole file, one exposure of
each detector and pipeline version is walked extension by extension, and what
that says about the shape of the files is reused for the rest of the group -
the header lengths, the wavelength grid and the PSF. Only the lengths, though,
and not the positions: the first header of a file is a block longer in some
exposures than in others, within one detector and version, so where the data
starts is taken from each file itself. See _survey_file for what reusing it
instead does, which is to measure a patch of empty sky and call it the target.

The photometry works in the units the images come in - microns, and uJy, which
is what an aperture sum of MJy/sr naturally is - and the two tables that leave
this module are converted at the last moment to what every spectrum here is
written in: Angstrom, and erg/s/cm2/A. That last is not a change of units but
of variable, uJy being a flux per unit frequency, so it changes the shape of
the spectrum and not only its height.
"""

import os
import re
import collections

import numpy as np
import requests

from concurrent.futures import ThreadPoolExecutor

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from astropy.table import Table
from astropy.time import Time

# STDPipe
from stdpipe import plots

from ..surveys import survey_source, get_output_files
from .utils import (SourceError, cleanup_paths, cached_votable_query,
                    quality_field, quality_level, flambda_from_fnu,
                    write_spectrum,
                    QUALITY_STANDARD, QUALITY_RELAXED, QUALITY_PUBLISHED)


# Where the spectral images are listed. SIA2 rather than TAP: the collection is
# the whole of the query, and the reply carries the access URLs directly.
SPHEREX_SIA = 'https://irsa.ipac.caltech.edu/SIA'
SPHEREX_COLLECTION = 'spherex_qr2'

# The radius the image list is asked for, in degrees. Not a matching radius -
# an exposure either covers the position or it does not - so this is only large
# enough to survive rounding, and small enough not to drag in the neighbours.
SPHEREX_SIA_RADIUS = 0.002

# Aperture radius, in detector pixels. The pixels are 6.2 arcsec and the PSF is
# about one of them across, so this is a 9.3 arcsec aperture - the same as the
# published quick-look tooling uses, and wide enough that the aperture
# correction below is a few per cent rather than a rescaling.
SPHEREX_APERTURE = 1.5

# Half-size of the stamp cut around the source, in detector pixels. It sets
# what the preview image can cover and how much sky the background annulus has
# to work with, and each row of it is bytes fetched, so it is kept to what
# those two need.
SPHEREX_STAMP = 7

# How far outside the aperture the background annulus begins, in pixels. The
# PSF is about a pixel across, so this is several times its width - far enough
# out that the wings of the target are not being subtracted from itself.
SPHEREX_ANNULUS_GAP = 1.5

# The preview grid: half-size and pixel, both in arcsec. The stamps arrive at
# every roll angle, so the largest circle that every one of them covers has the
# radius of the inscribed circle of the stamp - anything beyond that would be
# filled by some exposures and not others.
SPHEREX_PREVIEW_HALF = 30.0
SPHEREX_PREVIEW_PIXEL = 2.0

# The three wavelength ranges the preview is made of, in microns, reddest
# first. They divide the range roughly by detector pair, which is also roughly
# equal in log wavelength.
SPHEREX_CHANNELS = [('R', 3.0, 5.1), ('G', 1.6, 3.0), ('B', 0.7, 1.6)]

# How many exposures are read at once. IRSA starts refusing above a dozen or
# so, and a refusal costs more than the concurrency gains.
SPHEREX_WORKERS = 8

# How much is read when reaching for a header: enough for the primary and the
# IMAGE one together, which is the pair that has to arrive in one request. A
# header longer than this - the PSF extension carries some five hundred cards -
# is read again with the window doubled.
SPHEREX_HEAD = 49152

# A cap, so that a position at an ecliptic pole - where the survey has some
# eight hundred exposures and will have far more - cannot run indefinitely.
# The newest are dropped rather than the oldest, so the spectrum keeps its
# wavelength coverage instead of losing one end of it.
SPHEREX_MAX_IMAGES = 600

# What the pixel flags mean is written into the FLAGS header, as MP_<name>
# keywords holding bit numbers, so the two below are named rather than the
# whole list: everything documented counts against a pixel except these, which
# say something about the pixel without saying it is wrong.
SPHEREX_FLAGS_KEPT = {'MP_FULLSAMPLE', 'MP_SOURCE'}

# The flags that make a pixel not a measurement at all, whatever the filtering
SPHEREX_FLAGS_FATAL = {'MP_NONFUNC', 'MP_MISSING_DATA', 'MP_REFERENCE'}

# What the bits are called, for saying why exposures were dropped. The masking
# itself reads the bits out of each file's own FLAGS header, so a bit this does
# not name is still masked - it is only reported as its number. A bright star
# is the usual reason a target loses most of its exposures, and it is worth
# being told that in words rather than as a count.
SPHEREX_FLAG_NAMES = {
    0: 'transient', 1: 'overflow - the target is too bright for SPHEREx',
    2: 'onboard error', 4: 'phantom',
    5: 'reference', 6: 'dead', 7: 'dichroic', 9: 'data lost', 10: 'hot',
    11: 'cold', 12: 'full sample', 14: 'phantom uncorrected',
    15: 'nonlinear', 17: 'persistence', 19: 'outlier', 21: 'known source',
    # Added by the 2026 pipeline versions, and masked from the moment they
    # appeared without this having to be told about them
    22: 'ghost', 23: 'ghost (focal plane)', 24: 'ghost (external)',
    26: 'blooming', 27: 'snowball', 28: 'halo of a bright star',
}

# What counts as a detection, when saying how much of a spectrum is one
SPHEREX_MIN_SNR = 3.0

# Bumped whenever a change here makes the numbers different, since it goes
# into the cache name. What the other sources cache is the archive's own reply,
# which our code cannot invalidate by changing; what this one caches is
# photometry of our own making, so a fix that is not reflected in the cached
# file would go unnoticed on every target already acquired.
SPHEREX_EXTRACTION = 2

# The resolving power the spectrum is binned to. The filter's own is between
# 35 and 130 depending on the detector, so this is coarser than the coarsest
# of them - a bin is then always wider than the measurement it holds, and the
# binning never claims a resolution the instrument did not have.
SPHEREX_RESOLUTION = 30.0

# The survey's wavelengths are quoted in microns everywhere, including in the
# files themselves, and every spectrum here is written in Angstrom
SPHEREX_TO_ANGSTROM = 1e4

# One FITS block
BLOCK = 2880


def _detector_and_version(url):
    """Which detector took an exposure, and what processed it.

    Both are in the file name - '..._0329_3D6_spx_l2b-v20-2025-253.fits' is
    detector 6 out of version v20-2025-253 - and together they fix the layout
    of the file, the wavelength grid and the PSF.
    """
    detector = re.search(r'D(\d)_spx_', url)
    version = re.search(r'_spx_(.+)\.fits', url)

    return (detector.group(1) if detector else '?',
            version.group(1) if version else '?')


class _Ranges:
    """Byte ranges out of one file, over a session that is kept open."""

    def __init__(self, url, session=None):
        self.url = url
        self.session = session or requests.Session()
        self.nbytes = 0

    def get(self, start, end):
        res = self.session.get(self.url, timeout=120,
                               headers={'Range': f'bytes={start}-{end}'})

        if res.status_code not in (200, 206):
            raise SourceError(f"the archive answered {res.status_code} for a "
                              f"byte range of {os.path.basename(self.url)}")

        self.nbytes += len(res.content)

        return res.content


def _header_at(ranges, offset):
    """The FITS header starting at an offset, and where its data begins.

    The window is grown rather than guessed at: the PSF extension carries some
    five hundred cards, which is more than the 32 kB a first read covers.
    """
    want = SPHEREX_HEAD

    while True:
        buf = ranges.get(offset, offset + want - 1)
        parsed = _parse_header(buf, 0)

        if parsed is not None:
            header, length = parsed
            return header, offset + length, buf

        if len(buf) < want:
            raise SourceError("a FITS header ran past the end of the file")

        want *= 2


def _parse_header(buf, offset):
    """One FITS header out of a buffer, and how long it turned out to be.

    None where the header does not end inside what was read, which is the
    caller's cue to read more of it.
    """
    end = buf.find(b'END' + b' ' * 77, offset)

    if end < 0:
        return None

    length = ((end - offset + 80 + BLOCK - 1) // BLOCK) * BLOCK

    if offset + length > len(buf):
        return None

    return (fits.Header.fromstring(
        buf[offset:offset + length].decode('ascii', 'ignore')), length)


def _data_size(header):
    """How many bytes an extension's data occupies, padded to whole blocks."""
    if not header.get('NAXIS'):
        return 0

    count = 1
    for axis in range(1, header['NAXIS'] + 1):
        count *= header[f'NAXIS{axis}']

    size = count * abs(header['BITPIX']) // 8

    return ((size + BLOCK - 1) // BLOCK) * BLOCK


def _wavelength_grid(header, buf, offset_in_buf):
    """The WCS-WAVE table: wavelength and bandwidth over a grid of pixels.

    A lookup table rather than a WCS proper, the filter being linear in one
    direction and slightly curved in the other.

    The grid is not the same shape on every detector - most are sampled nine by
    nine, but D3 and D4 carry fourteen rows, five of them packed into the
    hundred and forty pixels where their response steps - so its shape is taken
    from the header rather than assumed. Reading a 14-row table as a 9-row one
    puts the whole spectrum at a twentieth of its wavelength.
    """
    def count(key):
        return int(re.match(r'(\d+)', str(header[key])).group(1))

    nx, ny = count('TFORM1'), count('TFORM2')

    # TDIM is in FITS order, fastest axis first, which numpy reads backwards
    shape = tuple(reversed([int(_) for _ in
                            re.findall(r'\d+', str(header['TDIM3']))]))

    dtype = np.dtype([('X', '>i4', (nx,)), ('Y', '>i4', (ny,)),
                      ('V', '>f4', shape)])

    record = np.frombuffer(
        buf[offset_in_buf:offset_in_buf + dtype.itemsize], dtype=dtype)[0]

    # The columns are 1-based pixel coordinates, and VALUES is indexed
    # [y, x, (wavelength, bandwidth)]
    return (record['X'].astype(float) - 1, record['Y'].astype(float) - 1,
            np.array(record['V'], dtype=float))


# What has been learned about each detector and pipeline version, kept for as
# long as the worker lives. None of it depends on the target - it is the shape
# of the files themselves - and a year of survey is some seventy versions, each
# of which would otherwise be walked again for every target.
_LAYOUTS = {}


def _survey_file(url, log, key=None):
    """What every file of one detector and one pipeline version has in common.

    Walked once per group: how long each extension's header is, what the flag
    bits mean, the wavelength grid, and where the PSF cube sits. The rest of
    the group is then read without walking again - and so is every later target
    that meets the same version, which is what the key is for.

    Lengths rather than positions, for everything but the PSF. The primary and
    IMAGE headers together are 23040 bytes in some exposures and 25920 in
    others - one FITS block, a few cards of per-exposure provenance - and that
    happens within a single detector and version, not between them. Taking the
    IMAGE data offset from another file of the group therefore reads a third of
    a row early in a fifth of cases, which lands 720 pixels away along the row:
    clean sky, no target, and a measurement of nothing that looked like a
    measurement of zero. Only this first header varies; the FLAGS and VARIANCE
    headers were the same length in all forty exposures checked across every
    group, so those are kept as lengths and added to what each file itself
    says.
    """
    if key is not None and key in _LAYOUTS:
        # Nothing was read this time, so the caller is not charged for it
        return dict(_LAYOUTS[key], bytes=0)

    ranges = _Ranges(url)

    _, primary_end, _ = _header_at(ranges, 0)
    image_header, image_data, _ = _header_at(ranges, primary_end)

    layout = {'IMAGE': (image_header, image_data)}
    offset = image_data + _data_size(image_header)

    flag_bits, grid, psf = {}, None, None

    for _ in range(10):
        header, data, buf = _header_at(ranges, offset)

        # The PSF extension is the one without a name of its own
        name = header.get('EXTNAME', 'PSF')
        layout[name] = (header, data)

        if name == 'FLAGS':
            flag_bits = {card: int(header[card]) for card in header
                         if card.startswith('MP_')}

        if name == 'WCS-WAVE':
            grid = _wavelength_grid(header, buf, data - offset)
            break

        if name == 'PSF':
            psf = (data, header.get('OVERSAMP', 10),
                   [(header.get(f'XCTR_{_ + 1}'), header.get(f'YCTR_{_ + 1}'))
                    for _ in range(header.get('NAXIS3', 0))])

        offset = data + _data_size(header)

    if grid is None or 'VARIANCE' not in layout or 'FLAGS' not in layout:
        raise SourceError(f"{os.path.basename(url)} is not laid out like a "
                          "SPHEREx spectral image")

    plane = _data_size(image_header)

    found = {
        # The file this was learned from. The PSF is the one thing still read
        # at an absolute offset, so it has to be read out of this same file
        # rather than whichever one a later target happens to start with.
        'url': url,
        'flags_header': layout['FLAGS'][1] - (layout['IMAGE'][1] + plane),
        'variance_header': layout['VARIANCE'][1] - (layout['FLAGS'][1] + plane),
        'flag_bits': flag_bits,
        'grid': grid,
        'psf': psf,
    }

    if key is not None:
        _LAYOUTS[key] = found

    return dict(found, bytes=ranges.nbytes)


def _bad_flag_mask(flag_bits, quality):
    """Which flag bits condemn a pixel, at this level of filtering.

    Read off the header rather than listed here: the file says what each bit
    means, so a pipeline that adds one is covered without this having to know
    about it.
    """
    if quality == QUALITY_PUBLISHED:
        return 0

    wanted = (SPHEREX_FLAGS_FATAL if quality == QUALITY_RELAXED
              else set(flag_bits) - SPHEREX_FLAGS_KEPT)

    mask = 0
    for name, bit in flag_bits.items():
        if name in wanted:
            mask |= 1 << bit

    return mask


def _interpolate_grid(grid, plane, x, y):
    """A value off the 9x9 wavelength grid at a pixel position, bilinearly."""
    gx, gy, values = grid

    fx = np.interp(x, gx, np.arange(len(gx)))
    fy = np.interp(y, gy, np.arange(len(gy)))

    i = min(int(np.floor(fy)), len(gy) - 2)
    j = min(int(np.floor(fx)), len(gx) - 2)
    dy, dx = fy - i, fx - j

    v = values[:, :, plane]

    return float((1 - dy) * ((1 - dx) * v[i, j] + dx * v[i, j + 1])
                 + dy * ((1 - dx) * v[i + 1, j] + dx * v[i + 1, j + 1]))


def _psf_index(psf, x, y):
    """Which of the PSFs in the cube belongs at a detector position."""
    _, _, centres = psf

    known = [(n, cx, cy) for n, (cx, cy) in enumerate(centres)
             if cx is not None and cy is not None]

    if not known:
        return 0

    return min(known, key=lambda _: (_[1] - x) ** 2 + (_[2] - y) ** 2)[0]


def _enclosed_energy(ranges, psf, index, radius):
    """What fraction of a point source the aperture holds.

    The files carry their own PSF, normalised to one and oversampled tenfold,
    so the aperture correction is measured rather than assumed. This is the
    step the published quick-look tooling replaces with an empirical rescaling
    fitted against the mission's own fitting pipeline.

    Measured over the detector pixels the aperture actually sums, not over a
    circle of the same radius: the aperture is a handful of whole pixels either
    way, and at six arcseconds a pixel the difference between the two is
    several per cent of the flux.
    """
    offset, oversample, _ = psf
    size = 101 * 101 * 4

    start = offset + index * size
    image = np.frombuffer(ranges.get(start, start + size - 1),
                          dtype='>f4').reshape(101, 101).astype(float)

    # The PSF onto the detector's own grid, the source at the centre of its
    # pixel - which is where it is on average, over exposures that dither
    step = int(oversample)
    half = image.shape[0] // 2
    reach = int(np.ceil(radius))

    total = 0.0

    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if np.hypot(dx, dy) > radius:
                continue

            y0 = half + dy * step - step // 2
            x0 = half + dx * step - step // 2

            total += float(np.nansum(image[max(y0, 0):y0 + step,
                                           max(x0, 0):x0 + step]))

    return total


def _background(pixels):
    """The sky under an annulus, and how much it scatters.

    The median, which is not quite the sky. At six arcseconds a pixel SPHEREx
    is confusion limited - there is something faint in most pixels of most
    annuli - so the median sits a little above the sky rather than on it, and
    every aperture comes out short by around two per cent of the sky it stands
    on. Measured against blank sky that is some 35 uJy, and against a crowded
    field several times more.

    On anything bright this is far inside the calibration: the Cloverleaf comes
    out within three per cent of its 2MASS photometry across J, H and Ks. On a
    target near the per-exposure limit it is the whole measurement, and is why
    a faint one can arrive slightly negative.

    Estimating the mode instead - the usual remedy for a skewed sky - was tried
    and is worse: it moved a blank position by 700 uJy in a single exposure
    where it moved a bright target by 11, which is noise bought at the price of
    a bias one can at least state.
    """
    median = float(np.median(pixels))
    scatter = float(np.median(np.abs(pixels - median))) * 1.4826

    return median, scatter


def _extract(url, ra, dec, group, quality, aperture):
    """One exposure's measurement of the target, and its stamp.

    Four byte ranges: the front of the file for the astrometry and the time,
    and then the rows of IMAGE, VARIANCE and FLAGS that the stamp falls on -
    some 300 kB out of a 70 MB file.
    """
    ranges = _Ranges(url)

    # The primary header and the IMAGE one that follows it, in a single read -
    # the two together run to some 26 kB, and this is the only place the
    # astrometry and the time of an exposure are written
    _, primary_end, head = _header_at(ranges, 0)
    parsed = _parse_header(head, primary_end)

    if parsed is None:
        raise SourceError(f"the IMAGE header of {os.path.basename(url)} did "
                          "not fit in the first read")

    header, image_header_length = parsed

    # Where this file's own data begins and how big a plane of it is, taken
    # from the file rather than from the group's first one - see _survey_file
    # for what goes wrong when the two differ. Only the lengths of the two
    # headers further in are the group's, those being the same in every
    # exposure checked.
    ny, nx = header['NAXIS2'], header['NAXIS1']
    plane = _data_size(header)

    image_offset = primary_end + image_header_length
    flags_offset = image_offset + plane + group['flags_header']
    variance_offset = flags_offset + plane + group['variance_header']

    wcs = WCS(header)
    x, y = wcs.all_world2pix(ra, dec, 0)
    x, y = float(x), float(y)

    if not np.isfinite(x) or not np.isfinite(y):
        return None

    xi, yi = int(round(x)), int(round(y))

    # The whole stamp has to be on the detector, both because a partial one
    # would bias the aperture and because the preview stacks them as squares
    if not (SPHEREX_STAMP <= xi < nx - SPHEREX_STAMP
            and SPHEREX_STAMP <= yi < ny - SPHEREX_STAMP):
        return None

    y0, y1 = yi - SPHEREX_STAMP, yi + SPHEREX_STAMP
    x0, x1 = xi - SPHEREX_STAMP, xi + SPHEREX_STAMP

    def rows(base, dtype):
        raw = ranges.get(base + y0 * nx * 4, base + (y1 + 1) * nx * 4 - 1)
        block = np.frombuffer(raw, dtype=dtype).reshape(y1 - y0 + 1, nx)
        return block[:, x0:x1 + 1].astype(float if dtype == '>f4' else int)

    image = rows(image_offset, '>f4')
    variance = rows(variance_offset, '>f4')
    flags = rows(flags_offset, '>i4')

    # A variance is never negative, so a plane of them that is says the offsets
    # have drifted off the extension they were meant for and this exposure is
    # not being measured at all
    finite = np.isfinite(variance)

    if np.any(finite) and np.mean(variance[finite] < 0) > 0.1:
        raise SourceError(f"the VARIANCE plane of {os.path.basename(url)} "
                          "does not read as one")

    condemned = _bad_flag_mask(group['flag_bits'], quality)
    bad = (flags & condemned) != 0

    # The aperture, as a circle about where the source actually falls rather
    # than about the pixel it was rounded to
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    inside = np.hypot(xx - x, yy - y) <= aperture

    good = inside & ~bad & np.isfinite(image) & np.isfinite(variance)

    # A pixel of the aperture lost to a flag is a measurement of something
    # else, so the whole exposure goes rather than a partial sum being scaled
    npixels = int(np.sum(inside))
    complete = int(np.sum(good)) == npixels

    # What the sky is worth here. The images are surface brightness with
    # everything still in them, and most of what SPHEREx sees at five microns
    # is zodiacal light rather than the target - left in, it doubles the flux
    # at two microns and worse at five, and turns every spectrum red. The
    # files carry a modelled ZODI plane, but a median over an annulus removes
    # the residual and the diffuse Galactic emission along with it, and needs
    # no further bytes off the archive.
    distance = np.hypot(xx - x, yy - y)
    annulus = ((distance > aperture + SPHEREX_ANNULUS_GAP)
               & (distance <= SPHEREX_STAMP - 0.5)
               & ~bad & np.isfinite(image))

    background, scatter = _background(image[annulus]) if np.any(annulus) else (0.0, 0.0)

    # Surface brightness in MJy/sr over pixels of a known size
    pixel_area = proj_plane_pixel_area(wcs) * (np.pi / 180.0) ** 2

    flux = float(np.nansum(image[good] - background)) * pixel_area * 1e12

    # The published variance of the aperture, and what the sky around it
    # scatters by - the second being the confusion the first does not know
    # about, and on a faint target the larger of the two
    error = float(np.sqrt(np.nansum(variance[good])
                          + npixels * scatter ** 2)) * pixel_area * 1e12

    world = wcs.all_pix2world(xx.ravel().astype(float),
                              yy.ravel().astype(float), 0)

    # The stamp is kept with its own sky already taken off, so that the
    # preview stacks exposures whose zodiacal foreground differs - and between
    # a detector at one micron and one at five it differs by a great deal
    stamp = np.where(bad, np.nan, image - background)

    return {
        'mjd': float(header.get('MJD-AVG', header.get('MJD-OBS', np.nan))),
        'x': x, 'y': y,
        'background': background * pixel_area * 1e12,
        'wavelength': _interpolate_grid(group['grid'], 0, x, y),
        'bandwidth': _interpolate_grid(group['grid'], 1, x, y),
        'flux': flux, 'flux_err': error,
        'complete': complete,
        'npixels': npixels,
        # What condemned a pixel of the aperture, so that an exposure dropped
        # below can say why. Only the offending bits: every aperture here also
        # carries the flag saying a known source falls on it, which is the
        # target and not a complaint.
        'flags': int(np.bitwise_or.reduce(
            (flags[inside] & condemned).astype(np.int64))),
        'psf_index': _psf_index(group['psf'], x, y) if group['psf'] else 0,
        'stamp': stamp,
        # Offsets from the target, in arcsec, east positive
        'dra': ((world[0] - ra + 180) % 360 - 180) * np.cos(np.deg2rad(dec)) * 3600.0,
        'ddec': (world[1] - dec) * 3600.0,
        'bytes': ranges.nbytes,
    }


def _query_images(ra, dec, basepath, log, refresh):
    """The spectral images covering a position, cached."""
    cache_name = f"spherex_sia_{ra:.4f}_{dec:.4f}.vot"

    with cached_votable_query(cache_name, basepath, log,
                              'SPHEREx image list', refresh=refresh) as cache:
        if not cache.hit:
            res = requests.get(SPHEREX_SIA, timeout=300, params={
                'COLLECTION': SPHEREX_COLLECTION,
                'POS': f'circle {ra} {dec} {SPHEREX_SIA_RADIUS}',
                'RESPONSEFORMAT': 'VOTABLE',
            })

            if res.status_code != 200:
                raise SourceError(f"IRSA answered {res.status_code} to the "
                                  "image search")

            from astropy.io.votable import parse
            from io import BytesIO

            votable = parse(BytesIO(res.content))
            table = votable.get_first_table()

            # The service returns its columns positionally named, so they are
            # put back before anything reads them by name
            data = table.to_table()
            data.rename_columns(data.colnames, [_.name for _ in table.fields])

            if len(data):
                cache.save(data['access_url', 't_min', 'em_min', 'em_max'])
            else:
                cache.data = None
        else:
            data = cache.data

    return cache.data


def _measure(images, ra, dec, quality, aperture, log):
    """Every exposure measured, grouped so that each layout is learned once.

    Three passes, each of them run across the whole set at once rather than a
    group at a time. Walking one file takes half a dozen round trips that
    depend on each other, and there are a couple of dozen groups over a year of
    survey - done in turn that alone was longer than reading every exposure.
    """
    groups = collections.defaultdict(list)

    for url in images:
        groups[_detector_and_version(url)].append(url)

    log(f"\n{len(images)} exposures, in {len(groups)} detector/version groups")

    nbytes = 0

    # What each group's files look like inside
    def survey(item):
        key, urls = item
        try:
            return key, _survey_file(urls[0], log, key=key)
        except Exception as e:
            log(f"  D{key[0]} {key[1]}: {e}")
            return key, None

    with ThreadPoolExecutor(SPHEREX_WORKERS) as pool:
        layouts = dict(pool.map(survey, sorted(groups.items())))

    nbytes += sum(_['bytes'] for _ in layouts.values() if _)

    # Every exposure of every group whose layout was read
    jobs = [(key, url) for key, urls in sorted(groups.items())
            for url in urls if layouts.get(key)]

    def one(job):
        key, url = job
        try:
            row = _extract(url, ra, dec, layouts[key], quality, aperture)
        except Exception as e:
            log(f"  {os.path.basename(url)}: {e}")
            return None

        if row is not None:
            row['group'] = key
            # Which exposure this was. Only the archive's own name for it, but
            # without it a measurement that looks wrong cannot be taken back to
            # the file it came from
            row['exposure'] = os.path.basename(url)[len('level2_'):-len('.fits')]

        return row

    with ThreadPoolExecutor(SPHEREX_WORKERS) as pool:
        results = [_ for _ in pool.map(one, jobs) if _ is not None]

    nbytes += sum(_['bytes'] for _ in results)

    # The PSF is the same in every file of a group, so the aperture correction
    # is measured once for each part of a detector the source actually landed
    # on, rather than once per exposure
    wanted = sorted({(_['group'], _['psf_index']) for _ in results
                     if layouts[_['group']]['psf'] is not None})

    def correction(item):
        key, index = item
        # The PSF offset is absolute, so it is read out of the file the layout
        # was learned from - which memoising across targets makes a different
        # file from this target's first one
        ranges = _Ranges(layouts[key]['url'])
        return item, _enclosed_energy(ranges, layouts[key]['psf'], index,
                                      aperture), ranges.nbytes

    with ThreadPoolExecutor(SPHEREX_WORKERS) as pool:
        measured = list(pool.map(correction, wanted))

    corrections = {item: value for item, value, _ in measured}
    nbytes += sum(read for _, _value, read in measured)

    for row in results:
        detector = row['group'][0]
        row['detector'] = int(detector) if detector.isdigit() else 0
        row['enclosed'] = corrections.get((row['group'], row['psf_index']), 1.0)

    for key, urls in sorted(groups.items()):
        found = sum(1 for _ in results if _['group'] == key)
        log(f"  D{key[0]} {key[1]}: {found} of {len(urls)} measured")

    log(f"\n{len(results)} exposures measured, {nbytes / 1e6:.0f} MB read")

    return results


def _bin_spectrum(measurements, log):
    """The exposures combined into a spectrum, at fixed resolving power.

    This is how a SPHEREx spectrum is actually made. One exposure measures one
    wavelength, at a depth of about eighteenth magnitude, so for most targets a
    single exposure is a marginal detection and the spectrum only appears once
    the several hundred of them are put together. Bins of constant lambda over
    delta lambda, since the filter's own resolving power is roughly constant
    within a detector, and an inverse-variance mean within each.

    The bins run across the visits rather than within them: a wavelength is
    revisited only every six months, so binning per visit would leave three
    points per bin and a spectrum of mostly noise. A target that varies between
    visits therefore shows up as scatter here and as structure in the
    measurements file, which keeps every exposure separately.
    """
    wavelength = np.asarray(measurements['wavelength'], dtype=float)
    flux = np.asarray(measurements['flux'], dtype=float)
    error = np.asarray(measurements['flux_error'], dtype=float)

    lo, hi = wavelength.min(), wavelength.max()

    if not (hi > lo > 0):
        return None

    step = 1.0 + 1.0 / SPHEREX_RESOLUTION
    edges = lo * step ** np.arange(np.ceil(np.log(hi / lo) / np.log(step)) + 1)

    index = np.digitize(wavelength, edges) - 1

    rows = []

    for number in range(len(edges) - 1):
        here = index == number

        if not np.any(here):
            continue

        weight = 1.0 / error[here] ** 2
        mean = float(np.sum(flux[here] * weight) / np.sum(weight))

        rows.append((
            float(np.sum(wavelength[here] * weight) / np.sum(weight)),
            mean,
            float(np.sqrt(1.0 / np.sum(weight))),
            # What the exposures themselves say about the spread, which counts
            # the target's own variability where the formal error does not
            float(np.std(flux[here], ddof=1) / np.sqrt(np.sum(here)))
            if np.sum(here) > 1 else np.nan,
            int(np.sum(here)),
        ))

    if not rows:
        return None

    spectrum = Table(rows=rows, names=['wavelength', 'flux', 'flux_error',
                                       'scatter', 'nexposures'])

    log(f"\n{len(spectrum)} bins at R = {SPHEREX_RESOLUTION:.0f},"
        f" {int(np.median(spectrum['nexposures']))} exposures in each typically")

    snr = spectrum['flux'] / spectrum['flux_error']
    log(f"{int(np.sum(snr > SPHEREX_MIN_SNR))} of them above"
        f" {SPHEREX_MIN_SNR:.0f} sigma, the best at {np.max(snr):.0f}")

    return spectrum


def _preview(results, basepath, name, show, log):
    """A three-colour image of the field, out of the exposures themselves.

    Every stamp arrives at its own roll angle and its own sub-pixel offset, so
    they are put onto a common grid of offsets from the target rather than
    added as arrays. The stack is a median: at six microns a pixel the images
    are full of unflagged transients, and with some tens of exposures over
    every cell the median removes them where an average would keep them.
    """
    half, pixel = SPHEREX_PREVIEW_HALF, SPHEREX_PREVIEW_PIXEL
    size = int(2 * half / pixel) + 1

    planes = []

    for label, lo, hi in SPHEREX_CHANNELS:
        here = [_ for _ in results if lo <= _['wavelength'] < hi]

        if not here:
            log(f"  {label}: nothing between {lo} and {hi} um")
            planes.append(np.full((size, size), np.nan))
            continue

        # Everything that falls on the grid, as one flat list of cell and value
        cells, values = [], []

        for row in here:
            # East to the left, as the sky is drawn
            column = np.round((-row['dra'] + half) / pixel).astype(int)
            line = np.round((row['ddec'] + half) / pixel).astype(int)

            good = (np.isfinite(row['stamp']) & (line >= 0) & (line < size)
                    & (column >= 0) & (column < size))

            cells.append((line[good] * size + column[good]).ravel())
            values.append(row['stamp'][good].ravel())

        cells = np.concatenate(cells)
        values = np.concatenate(values)

        order = np.argsort(cells, kind='stable')
        cells, values = cells[order], values[order]

        plane = np.full(size * size, np.nan)

        # One median per cell, over the runs the sort has put together
        edges = np.flatnonzero(np.diff(cells)) + 1
        for chunk in np.split(np.arange(len(cells)), edges):
            if len(chunk):
                plane[cells[chunk[0]]] = np.median(values[chunk])

        plane = plane.reshape(size, size)

        # Each stamp came in with its own sky already off; this only levels
        # what is left of one channel against another
        planes.append(plane - np.nanmedian(plane))

        log(f"  {label}: {len(here)} exposures, {lo}-{hi} um,"
            f" {100 * np.mean(np.isfinite(plane)):.0f}% of the field covered")

    # Each channel stretched on its own: they differ by more than a colour
    # image can hold, the zodiacal foreground alone rising steeply to the red
    rgb = np.zeros((size, size, 3))

    for index, plane in enumerate(planes):
        if not np.any(np.isfinite(plane)):
            continue

        top = np.nanpercentile(plane, 99.5)

        if not np.isfinite(top) or top <= 0:
            continue

        # asinh, so that the faint neighbours survive alongside the target
        scaled = np.arcsinh(np.nan_to_num(plane, nan=0.0) / (top / 8.0))
        scaled /= max(np.arcsinh(8.0), 1e-6)

        rgb[:, :, index] = np.clip(scaled, 0, 1)

    with plots.figure_saver(os.path.join(basepath, 'spherex_rgb.png'),
                            figsize=(5.5, 5.5), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        ax.imshow(rgb, origin='lower', interpolation='nearest',
                  extent=[half, -half, -half, half])

        ax.set_xlabel('Offset in RA, arcsec')
        ax.set_ylabel('Offset in Dec, arcsec')
        ax.set_title(f"{name} - SPHEREx\n"
                     + ', '.join(f"{c[0]}: {c[1]}-{c[2]} um"
                                 for c in SPHEREX_CHANNELS),
                     fontsize=10)

    log("\nPreview written to file:spherex_rgb.png")


@survey_source(
    name='SPHEREx',
    short_name='SPHEREx',
    state_acquiring='acquiring SPHEREx spectrophotometry',
    state_acquired='SPHEREx spectrophotometry acquired',
    log_file='spherex.log',
    output_files=['spherex.log', 'spherex*.png', 'spherex.vot', 'spherex.txt',
                  'spherex_binned.vot', 'spherex_binned.txt'],
    # What the step is here to bring back is the exposures; the curve is
    # made of them and cannot exist without them, so a run that wrote the
    # one and not the other found data all the same
    data_files=['spherex.txt'],
    button_text='Get SPHEREx spectrum',
    form_fields={
        'spherex_aperture': {
            'type': 'float',
            'label': 'Aperture radius, detector pixels',
            'initial': SPHEREX_APERTURE,
            'required': False,
        },
        'spherex_quality': quality_field({
            QUALITY_STANDARD: 'Any pixel the file flags, anywhere in the aperture',
            QUALITY_RELAXED: 'Dead, missing and reference pixels only',
            QUALITY_PUBLISHED: 'None - every exposure as measured',
        }),
    },
    help_text='SPHEREx QR2 spectrophotometry, all sky, 0.75-5 um',
    order=29,
    # Spectrophotometry rather than a light curve, so no lc_mode is declared.
    #
    # The curve is matched by the wildcard, which is also what names it in the
    # viewer: 'spherex_binned' gives 'SPHEREx binned'. The exposures it was
    # binned from are the file named for the survey alone, and so keep the
    # survey's plain name - they are what SPHEREx measured, where the curve is
    # what this module made of them.
    spectrum_files='spherex_*.txt',
    spectrum_points='spherex.txt',
    spectrum_color='#c0392b',
    template_layout='complex',
    additional_plots=['spherex_rgb.png'],
    main_plot='spherex_spectrum.png',
)
def target_spherex(config, basepath=None, verbose=True, show=False):
    """
    Get SPHEREx spectrophotometry.

    Parameters
    ----------
    config : dict
        Configuration dictionary with target coordinates
    basepath : str, optional
        Base path for output files
    verbose : bool or callable, optional
        Verbose logging mode or log function
    show : bool, optional
        Show plots interactively
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Read, not consumed: a chain must refresh every step it runs, so the flag
    # is cleared once the whole run finishes rather than by the first source
    refresh_cache = bool(config.get('refresh_cache', False))

    # Cleanup stale plots
    cleanup_paths(get_output_files('spherex'), basepath=basepath)

    if 'target_ra' not in config or 'target_dec' not in config:
        raise RuntimeError("Cannot operate without target coordinates")

    ra = config.get('target_ra')
    dec = config.get('target_dec')

    aperture = float(config.get('spherex_aperture') or SPHEREX_APERTURE)
    quality = quality_level(config, 'spherex')

    log(f"aperture radius {aperture:.1f} px"
        f" ({aperture * 6.2:.1f} arcsec), {quality} filtering")

    images = _query_images(ra, dec, basepath, log, refresh_cache)

    if images is None or not len(images):
        log("\nWarning: No SPHEREx exposures at this position")
        return

    urls = [str(_) for _ in images['access_url']]

    if len(urls) > SPHEREX_MAX_IMAGES:
        # Oldest first, so that what is dropped is the end of the newest visit
        # rather than a slice out of the middle of the wavelength coverage
        order = np.argsort(np.asarray(images['t_min'], dtype=float))
        urls = [urls[_] for _ in order[:SPHEREX_MAX_IMAGES]]

        log(f"\n{len(images)} exposures cover this position, of which the "
            f"earliest {SPHEREX_MAX_IMAGES} are measured")

    cache_name = (f"spherex_{ra:.4f}_{dec:.4f}_{aperture:.1f}_{quality}"
                  f"_v{SPHEREX_EXTRACTION}.vot")

    with cached_votable_query(cache_name, basepath, log,
                              'SPHEREx spectrophotometry',
                              refresh=refresh_cache) as cache:
        if not cache.hit:
            results = _measure(urls, ra, dec, quality, aperture, log)

            if not results:
                log("\nWarning: no exposure could be measured")
                return

            # The stamps ride along with the measurements, so that the preview
            # can be redrawn from the cache without reading the archive again.
            # This one is ours rather than the archive's, and its columns are
            # named for this module's own use; what the two published tables
            # carry is named to match the other spectral sources.
            table = Table({
                'exposure': [_['exposure'] for _ in results],
                'mjd': [_['mjd'] for _ in results],
                'wavelength': [_['wavelength'] for _ in results],
                'bandwidth': [_['bandwidth'] for _ in results],
                'flux': [_['flux'] for _ in results],
                'flux_err': [_['flux_err'] for _ in results],
                'background': [_['background'] for _ in results],
                'enclosed': [_['enclosed'] for _ in results],
                'complete': [int(_['complete']) for _ in results],
                'flags': [_['flags'] for _ in results],
                'detector': [_['detector'] for _ in results],
                'x': [_['x'] for _ in results],
                'y': [_['y'] for _ in results],
                'stamp': np.array([_['stamp'].ravel() for _ in results]),
                'dra': np.array([_['dra'].ravel() for _ in results]),
                'ddec': np.array([_['ddec'].ravel() for _ in results]),
            })

            cache.save(table)
        else:
            table = cache.data

    side = 2 * SPHEREX_STAMP + 1

    results = [{
        'mjd': float(row['mjd']),
        'wavelength': float(row['wavelength']),
        'stamp': np.asarray(row['stamp'], dtype=float).reshape(side, side),
        'dra': np.asarray(row['dra'], dtype=float).reshape(side, side),
        'ddec': np.asarray(row['ddec'], dtype=float).reshape(side, side),
    } for row in table]

    # The aperture holds most of a point source, and the PSF in the files says
    # how much, so the flux is put back onto the whole of it
    flux = np.asarray(table['flux'], dtype=float)
    error = np.asarray(table['flux_err'], dtype=float)
    enclosed = np.asarray(table['enclosed'], dtype=float)

    inside = np.where((enclosed > 0.1) & (enclosed <= 1.0), enclosed, 1.0)
    flux, error = flux / inside, error / inside

    log(f"\nAperture correction from the PSF in the files:"
        f" {1 / np.max(inside):.3f} to {1 / np.min(inside):.3f}")

    if 'background' in table.colnames:
        sky = np.asarray(table['background'], dtype=float)
        log(f"Sky taken off, per pixel: {np.median(sky):.0f} uJy typically,"
            f" up to {np.max(sky):.0f} at the red end")

    wavelength = np.asarray(table['wavelength'], dtype=float)
    mjd = np.asarray(table['mjd'], dtype=float)
    complete = np.asarray(table['complete'], dtype=int).astype(bool)

    good = np.isfinite(flux) & np.isfinite(error) & (error > 0) & (wavelength > 0)

    # Only what was not measured properly. There is deliberately no cut on
    # signal to noise: a single SPHEREx exposure reaches about 18th magnitude,
    # so most targets are marginal in any one of them and the spectrum is made
    # by combining, below. Keeping only the exposures that stood above the
    # noise would also keep only the ones the noise pushed upwards, which
    # brightens a faint object rather than measuring it.
    if quality != QUALITY_PUBLISHED:
        good &= complete

    dropped = int(np.sum(~good))

    if dropped:
        log(f"Dropping {dropped} of {len(good)} exposures"
            + ("" if quality == QUALITY_PUBLISHED
               else " - flagged pixels in the aperture"))

        # Which flags, and how often. A target that loses nearly everything to
        # 'overflow' or 'nonlinear' is a star too bright for SPHEREx rather
        # than a target the archive has nothing good on.
        if 'flags' in table.colnames:
            raised = np.asarray(table['flags'], dtype=np.int64)[~good]

            counts = sorted(
                ((int(np.sum((raised >> bit) & 1)), bit) for bit in range(32)),
                reverse=True)

            for count, bit in counts[:5]:
                if count:
                    log(f"    {count:5d}  {SPHEREX_FLAG_NAMES.get(bit, f'bit {bit}')}")

    if not np.any(good):
        log("\nWarning: nothing survived the filtering")
        return

    # The preview is drawn from everything measured: a pixel too faint to be a
    # spectral point still belongs in a picture of the field
    log("\n---- Preview ----\n")
    _preview(results, basepath, config['target_name'], show, log)

    order = np.argsort(wavelength[good])

    # Onto the scale every spectrum here is written on. The photometry above,
    # the cache and the preview all work in the units the images come in -
    # microns, and uJy, which is what an aperture sum of MJy/sr naturally is -
    # and only what leaves this module is converted, at the last moment.
    angstrom = wavelength[good][order] * SPHEREX_TO_ANGSTROM
    lam_flux = flambda_from_fnu(flux[good][order], angstrom)
    lam_error = flambda_from_fnu(error[good][order], angstrom)

    measurements = Table({
        'wavelength': angstrom,
        'flux': lam_flux,
        'flux_error': lam_error,
        'bandwidth': (np.asarray(table['bandwidth'], dtype=float)[good][order]
                      * SPHEREX_TO_ANGSTROM),
        'mjd': mjd[good][order],
        'detector': np.asarray(table['detector'], dtype=int)[good][order],
        'exposure': (np.asarray(table['exposure'], dtype=str)[good][order]
                     if 'exposure' in table.colnames
                     else np.full(int(np.sum(good)), '', dtype=str)),
    })

    # The log talks in microns throughout, that being how the survey's own
    # wavelengths are always quoted, where the files are written in Angstrom
    micron = measurements['wavelength'] / SPHEREX_TO_ANGSTROM

    log("\n---- Measurements ----\n")
    log(f"{len(measurements)} exposures measured the target, from"
        f" {micron.min():.3f} to {micron.max():.3f} um")

    times = Time(measurements['mjd'], format='mjd')
    log(f"observed between {times.min().iso[:10]} and {times.max().iso[:10]}")

    # The visit windows, which is what the time sampling really is: the survey
    # crosses a position for a fortnight and comes back six months later
    epochs = np.sort(np.unique(np.round(measurements['mjd'])))
    breaks = np.flatnonzero(np.diff(epochs) > 30)
    windows = np.split(epochs, breaks + 1)

    log(f"in {len(windows)} visit window(s):")
    for window in windows:
        inside = ((measurements['mjd'] >= window[0] - 1)
                  & (measurements['mjd'] <= window[-1] + 1))
        log(f"  MJD {window[0]:.0f}-{window[-1]:.0f}"
            f"  {int(np.sum(inside))} exposures"
            f"  {micron[inside].min():.2f}-{micron[inside].max():.2f} um")

    significant = int(np.sum(measurements['flux']
                             > SPHEREX_MIN_SNR * measurements['flux_error']))
    log(f"\n{significant} of them are above {SPHEREX_MIN_SNR:.0f} sigma on"
        " their own; the rest are combined below rather than dropped")

    write_spectrum(measurements, basepath, 'spherex')

    log("SPHEREx measurements written to file:spherex.vot")
    log("SPHEREx measurements written to file:spherex.txt")

    spectrum = _bin_spectrum(measurements, log)

    if spectrum is None:
        log("\nWarning: nothing left to bin into a spectrum")
        return

    with plots.figure_saver(os.path.join(basepath, 'spherex_spectrum.png'),
                            figsize=(10, 4), show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)

        # The exposures behind the spectrum, coloured by time, so that a target
        # which changed between visits shows it as separated tracks rather than
        # as a wider scatter about the binned curve
        epoch = measurements['mjd'] - measurements['mjd'].min()

        points = ax.scatter(micron, measurements['flux'],
                            c=epoch, s=6, cmap='rainbow', alpha=0.8, zorder=2)

        fig.colorbar(points, ax=ax, label='Days since the first exposure')

        ax.errorbar(spectrum['wavelength'] / SPHEREX_TO_ANGSTROM,
                    spectrum['flux'],
                    spectrum['flux_error'], fmt='o-', color='#c0392b', lw=1.0,
                    ms=3, ecolor='#c0392b', capsize=2, zorder=4,
                    label=f"binned to R = {SPHEREX_RESOLUTION:.0f}")

        ax.axhline(0, color='#7f8c8d', lw=0.6, zorder=1)

        # The individual exposures scatter far wider than the spectrum does,
        # and on a faint target they would flatten it against the axis
        top = np.nanmax(spectrum['flux'] + spectrum['flux_error'])
        bottom = np.nanmin(spectrum['flux'] - spectrum['flux_error'])
        span = max(top - bottom, abs(top))
        margin = 0.35 * (span if span > 0 else 1.0)
        ax.set_ylim(min(bottom - margin, -margin / 2), top + margin)

        ax.legend()
        ax.grid(alpha=0.2)
        # Microns on the axis, as the survey's wavelengths are always quoted,
        # against the flux the file carries
        ax.set_xlabel('Wavelength, um')
        ax.set_ylabel(r'Flux, erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$')
        ax.set_title(f"{config['target_name']} - SPHEREx")

    write_spectrum(spectrum, basepath, 'spherex_binned')

    log(f"\nSpectrum plotted in file:spherex_spectrum.png")
    log(f"Binned spectrum written to file:spherex_binned.vot")
    log(f"Binned spectrum written to file:spherex_binned.txt")

    peak = spectrum['flux'] / spectrum['flux_error']

    if np.max(peak) < SPHEREX_MIN_SNR:
        log("\nWarning: the target is not detected in any bin - what is "
            "plotted is the noise of the sky at this position")
