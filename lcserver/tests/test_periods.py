"""Tests of the period analysis behind the light curve viewer.

On synthetic light curves with fixed seeds, sampled as the ground samples a
star - a few points a night, on some nights of a season - so that the daily
aliases and the gaps are there to get wrong. Each test is about something the
analysis has got wrong before, or could quietly get wrong: the true period
against its alias, the uncertainty against the actual error, an eclipsing
binary at half its period, the count of cycles across the gaps, and one
window with a falsely small error deciding where a fit starts.

No database: run with `python manage.py test lcserver.tests`.
"""

import numpy as np
from django.test import SimpleTestCase

from lcserver.processing import periods as p


def nightly_times(rng, nights, per_night=4, first=58000.0, span=600):
    """A few points a night, on `nights` nights scattered over `span` days."""
    chosen = np.sort(rng.choice(np.arange(span), nights, replace=False)) + first
    return np.concatenate([n + 0.15 + 0.06 * np.arange(per_night) for n in chosen])


def pulsator(t, period, amplitude=0.2, phase=0.0):
    """A non-sinusoidal light curve - a sinusoid and its first harmonic."""
    x = 2 * np.pi * (t / period + phase)
    return amplitude * (np.sin(x) + 0.35 * np.sin(2 * x + 0.8))


def eclipsing(t, period, primary=0.8, secondary=0.3, width=0.04):
    """Two eclipses of different depth, half a cycle apart, in magnitudes."""
    ph = (t / period) % 1.0
    d1 = np.minimum(np.abs(ph), np.abs(ph - 1.0))
    d2 = np.abs(ph - 0.5)
    return primary * np.exp(-0.5 * (d1 / width) ** 2) + secondary * np.exp(-0.5 * (d2 / width) ** 2)


def two_bands(rng, signal, nights=150, noise=0.02, per_night=4):
    """The same signal seen in two filters, at different amplitudes, as
    `refine_peaks` and the fits take series: (mjd, value, error, group)."""
    out = []
    for band, scale, level in (('g', 1.0, 15.0), ('r', 0.7, 14.6)):
        t = nightly_times(rng, nights, per_night)
        y = level + scale * signal(t) + rng.normal(0, noise, len(t))
        out.append((t, y, np.full(len(t), noise), ('synthetic', band, 'native')))
    return out


def pooled(series):
    """The series centred on their own medians and pooled, as the search takes them."""
    t = np.concatenate([s[0] for s in series])
    y = np.concatenate([s[1] - np.median(s[1]) for s in series])
    dy = np.concatenate([s[2] for s in series])
    return t, y, dy


def span_of(series):
    return max(s[0].max() for s in series) - min(s[0].min() for s in series)


class DetrendTest(SimpleTestCase):

    def test_trend_goes_and_the_period_stays(self):
        """A trend slower than the window comes off; a period well inside it
        is left with its amplitude."""
        rng = np.random.default_rng(1)
        t = nightly_times(rng, 200, per_night=6)
        signal = pulsator(t, 0.7)
        trend = 0.5 * np.sin(2 * np.pi * (t - t[0]) / 300.0)
        y = 15 + signal + trend + rng.normal(0, 0.01, len(t))

        detrended, fitted = p.detrend_series(t, y, 10.0, subtract=True)

        self.assertLess(np.std(fitted - np.median(fitted) - (trend - np.median(trend))), 0.05)
        kept = detrended - np.median(detrended)
        self.assertAlmostEqual(np.std(kept) / np.std(signal), 1.0, delta=0.1)


class PeriodSearchTest(SimpleTestCase):

    def test_true_period_beats_its_daily_alias(self):
        """Nightly sampling puts aliases a cycle a day either side of the
        period; the refit with harmonics picks the true one, to within its
        uncertainty."""
        rng = np.random.default_rng(2)
        period = 0.731
        series = two_bands(rng, lambda t: pulsator(t, period))
        t, y, dy = pooled(series)

        pg = p.find_period(t, y, dy, 0.3, 3.0)
        peaks, _ = p.refine_peaks(series, pg['peaks'], 0, span_of(series))
        best = peaks[0]

        self.assertAlmostEqual(best['period'], period, delta=1e-3)
        self.assertIsNotNone(best['period_err'])
        self.assertLess(abs(best['period'] - period), 4 * best['period_err'])

    def test_period_error_matches_the_scatter(self):
        """Over many light curves, the uncertainty given is what the fitted
        periods actually scatter by - within the precision a few dozen allow."""
        rng = np.random.default_rng(3)
        period = 2.3456
        fitted, errors = [], []
        for _ in range(30):
            series = two_bands(rng, lambda t: pulsator(t, period), nights=100)
            peaks, _ = p.refine_peaks(
                series, [{'frequency': (1 + 1e-4) / period, 'period': period, 'power': 1, 'fap': None}],
                0, span_of(series))
            fitted.append(peaks[0]['period'])
            errors.append(peaks[0]['period_err'])

        ratio = np.mean(errors) / np.std(fitted)
        self.assertGreater(ratio, 0.7)
        self.assertLess(ratio, 1.5)

    def test_harmonics_find_the_period_a_sinusoid_halves(self):
        """An Algol, its secondary eclipse shallow beside the primary, repeats
        nearly twice a cycle: the plain periodogram finds half its period,
        harmonics and a box find the period itself."""
        rng = np.random.default_rng(4)
        period = 1.093
        series = two_bands(rng, lambda t: eclipsing(t, period, secondary=0.1),
                           nights=250, per_night=6)
        t, y, dy = pooled(series)

        plain = p.find_period(t, y, dy, 0.3, 3.0)
        self.assertAlmostEqual(plain['peaks'][0]['period'], period / 2, delta=2e-3)

        three = p.find_period(t, y, dy, 0.3, 3.0, nterms=3)
        self.assertAlmostEqual(three['peaks'][0]['period'], period, delta=2e-3)

        box = p.find_period(t, y, dy, 0.3, 3.0, method='bls', magnitudes=True)
        self.assertAlmostEqual(box['peaks'][0]['period'], period, delta=2e-3)
        # An eclipse is a rise of the magnitude, and its depth comes out positive
        self.assertGreater(box['peaks'][0]['depth'], 0)

    def test_only_a_box_tells_alike_eclipses_apart(self):
        """Two narrow eclipses not far apart in depth: a few harmonics of half
        the period describe them as well as of the period, and only the box,
        which fits each dip on its own, finds the period."""
        rng = np.random.default_rng(4)
        period = 1.093
        series = two_bands(rng, lambda t: eclipsing(t, period, secondary=0.3, width=0.025),
                           nights=250, per_night=6)
        t, y, dy = pooled(series)

        three = p.find_period(t, y, dy, 0.3, 3.0, nterms=3)
        self.assertAlmostEqual(three['peaks'][0]['period'], period / 2, delta=2e-3)

        box = p.find_period(t, y, dy, 0.3, 3.0, method='bls', magnitudes=True)
        self.assertAlmostEqual(box['peaks'][0]['period'], period, delta=2e-3)

    def test_spectral_window_shows_the_day(self):
        """For nightly sampling the window rises again at a cycle a day."""
        rng = np.random.default_rng(5)
        t = nightly_times(rng, 200)
        window = p.spectral_window(t, 3.0, 20000)
        offsets, power = np.array(window['offsets']), np.array(window['power'])

        near_day = power[np.abs(offsets - 1.0) < 0.01].max()
        between = power[(offsets > 0.3) & (offsets < 0.7)].max()
        self.assertGreater(near_day, 0.5)
        self.assertLess(between, 0.5 * near_day)


class ShapeTest(SimpleTestCase):

    def test_fourier_parameters_of_a_known_shape(self):
        """A sinusoid with its first harmonic at 0.35 and a phase of 0.8 reads
        as R21 0.35 and phi21 0.8, in either band, whatever its time origin."""
        rng = np.random.default_rng(12)
        period = 0.731
        series = two_bands(rng, lambda t: pulsator(t, period, phase=0.37), nights=200)

        coeffs = p.group_coefficients(series, 0, 1.0 / period)
        for group, c in coeffs.items():
            shape = p.fourier_parameters(c, p.PERIOD_HARMONICS)
            self.assertAlmostEqual(shape['R21'], 0.35, delta=0.03)
            self.assertAlmostEqual(shape['phi21'], 0.8, delta=0.08)

        g = p.fourier_parameters(coeffs[('synthetic', 'g', 'native')], p.PERIOD_HARMONICS)
        r = p.fourier_parameters(coeffs[('synthetic', 'r', 'native')], p.PERIOD_HARMONICS)
        self.assertAlmostEqual(r['amplitude'] / g['amplitude'], 0.7, delta=0.05)


class PrewhiteningTest(SimpleTestCase):

    def test_two_close_frequencies_together(self):
        """Two periods fitted together come out where they are, each within
        its uncertainty, with the amplitude ratio of the bands right."""
        rng = np.random.default_rng(6)
        p1, p2 = 0.7312, 1.9134

        def signal(t):
            return pulsator(t, p1, 0.3) + 0.12 * np.cos(2 * np.pi * t / p2)

        series = two_bands(rng, signal, nights=250)
        freqs, errors, coeffs = p.fit_frequencies(series, 0, [1.00002 / p1, 0.99997 / p2])

        for f, e, true in zip(freqs, errors, (1 / p1, 1 / p2)):
            self.assertLess(abs(f - true), 4 * e)

        # The second filter was made at 0.7 of the first
        phase = np.arange(200) / 200.0
        amplitude = {g[1]: np.ptp(p.harmonics(phase, 1.0, 3) @ c[:6]) for g, c in coeffs.items()}
        self.assertAlmostEqual(amplitude['r'] / amplitude['g'], 0.7, delta=0.05)


class RedNoiseTest(SimpleTestCase):

    def test_white_is_one_and_red_is_more(self):
        rng = np.random.default_rng(7)
        t = nightly_times(rng, 200, per_night=8)

        white = rng.normal(0, 1, len(t))
        self.assertLess(p.red_noise_factor(t, white, 0.3), 1.3)

        # An offset a night, which no number of points within the night
        # averages away. A night's eight points fall in two bins of four, so
        # the factor is sqrt(4 * (4 + 1/4) / (4 + 1)), about 1.8
        nightly = white + np.repeat(rng.normal(0, 2, 200), 8)
        self.assertAlmostEqual(p.red_noise_factor(t, nightly, 0.3), 1.84, delta=0.25)


class PeriodChangeTest(SimpleTestCase):
    """A period that drifts by enough to slip the phase by cycles over the
    data - the case where a fit started from a constant period loses count."""

    period = 3.03
    pdot = -2.0e-5          # days per day: two cycles of drift over the span

    def phase(self, t):
        """Cycles of a period changing linearly, counted from MJD 59000."""
        f0, fdot = 1.0 / self.period, -self.pdot / self.period ** 2
        x = t - 59000.0
        return f0 * x + 0.5 * fdot * x ** 2

    def light_curve(self, seed=8):
        rng = np.random.default_rng(seed)
        out = []
        for band, scale in (('g', 1.0), ('r', 0.7)):
            t = nightly_times(rng, 300, per_night=2, first=57600.0, span=2800)
            x = 2 * np.pi * self.phase(t)
            y = 15 + scale * 0.3 * (np.sin(x) + 0.3 * np.sin(2 * x + 1)) + rng.normal(0, 0.03, len(t))
            out.append((t, y, np.full(len(t), 0.03), ('synthetic', band, 'native')))
        return out

    def test_start_survives_a_window_with_a_falsely_small_error(self):
        """A window of a few nights can find a period well off the rest and
        claim to know it ten times better. The start fitted to the windows'
        periods must not follow it - nor clip away the windows at the ends,
        which carry the change - and the fit from it reaches the true drift."""
        series = self.light_curve()
        tmin = min(s[0].min() for s in series)
        tmax = max(s[0].max() for s in series)

        local = p.local_periods(series, 0, 1.0 / self.period, 150)
        lt, lf, le = (np.array([w[k] for w in local]) for k in (0, 4, 5))
        self.assertGreater(len(lt), 8)

        # The impostor, in the middle, at a period 5% off, with a tenth of the error
        middle = int(np.argmin(np.abs(lt - np.median(lt))))
        lf[middle] *= 0.95
        le[middle] = 0.1 * np.median(le)

        model = p.PhaseModel(tmin, tmax, order=1)
        start = model.from_frequencies(lt, lf, le)
        c, cov, chi2, dof, _, _ = p.fit_phase_model(series, 0, model, start)

        f_mid = c[0]
        fdot = c[1] / model.half
        pdot = -fdot / f_mid ** 2
        pdot_err = np.sqrt(cov[1, 1]) / model.half / f_mid ** 2
        self.assertLess(abs(pdot - self.pdot), 4 * pdot_err)
        self.assertLess(chi2 / dof, 1.5)

    def test_cycles_counted_across_the_gaps(self):
        """Phases known only modulo one, in windows a season apart, linked
        one window at a time come out the true count of cycles."""
        rng = np.random.default_rng(9)
        t = np.sort(np.concatenate([season + rng.uniform(0, 150, 4)
                                    for season in np.arange(57600, 60400, 365.25)]))
        true = self.phase(t)
        measured = true % 1.0 + rng.normal(0, 0.02, len(t))

        model = p.PhaseModel(t.min(), t.max(), order=2)
        coeffs = p.connect_phases(model, t, measured, np.full(len(t), 0.02), 1.0 / self.period)

        fitted = model.phase_basis(t) @ coeffs
        # The constant of the phase is the harmonics' business: compare differences
        offset = np.median(fitted - true)
        self.assertLess(np.max(np.abs(fitted - offset - true)), 0.1)

    def test_oc_recovers_a_shift(self):
        """A window shifted by a known part of a cycle reads as that shift."""
        rng = np.random.default_rng(10)
        period, shift = 1.7, 0.12
        t = nightly_times(rng, 200, per_night=3, span=400)
        late = t > t.min() + 200
        phase = t / period - np.where(late, shift, 0.0)
        y = 15 + 0.3 * np.sin(2 * np.pi * phase) + rng.normal(0, 0.02, len(t))
        series = [(t, y, np.full(len(t), 0.02), ('synthetic', 'g', 'native'))]

        # The template: a sinusoid of the reference phase, as harmonics have it
        template = np.zeros(6)
        template[0] = 0.3
        windows = p.measure_oc(series, 0, lambda x: np.asarray(x) / period,
                               {('synthetic', 'g', 'native'): template}, 100)

        for mid, _, _, _, s, err, _, _ in windows:
            expected = shift if mid > t.min() + 200 else 0.0
            self.assertLess(abs(s - expected), max(4 * err, 0.02))


class EclipseTimingTest(SimpleTestCase):
    """A detached eclipsing binary seen as the space missions see it - every
    ten minutes for a month, in sectors years apart - with its period
    changing, and its secondary eclipse off half a cycle as an eccentric
    orbit puts it."""

    period = 1.093
    pdot = 3.0e-9           # days per day: some ten minutes of O-C over the span
    delta = 0.004           # of a cycle, the secondary after half of one
    noise = 0.002

    def phase(self, t):
        f0, fdot = 1.0 / self.period, -self.pdot / self.period ** 2
        x = t - 59000.0
        return f0 * x + 0.5 * fdot * x ** 2

    def light_curve(self, seed=11):
        rng = np.random.default_rng(seed)
        out = []
        for start in (57100.0, 58100.0, 59500.0, 60260.0):
            t = start + np.arange(0, 27, 10 / 1440.0)
            ph = self.phase(t) % 1.0
            d1 = np.minimum(ph, 1.0 - ph)
            d2 = np.abs(ph - 0.5 - self.delta)
            flux = (1.0 - 0.6 * np.exp(-0.5 * (d1 / 0.02) ** 2)
                    - 0.15 * np.exp(-0.5 * (d2 / 0.02) ** 2))
            flux = flux + rng.normal(0, self.noise, len(t))
            out.append((t, flux, np.full(len(t), self.noise), ('synthetic', None, 'native')))
        return out

    def test_harmonics_follow_the_shape(self):
        """Narrow eclipses need many harmonics, a sinusoid one."""
        series = self.light_curve()
        tmin = min(s[0].min() for s in series)
        tmax = max(s[0].max() for s in series)
        # The true frequency, a straight line in the time from the middle
        model = p.PhaseModel(tmin, tmax, order=1)
        fdot = -self.pdot / self.period ** 2
        coeffs = np.array([1.0 / self.period + fdot * (model.tref - 59000.0), fdot * model.half])
        nharm, _ = p.choose_harmonics(series, 0, model, coeffs)
        self.assertGreaterEqual(nharm, 12)

        rng = np.random.default_rng(12)
        sine = [(t, 1 + 0.05 * np.sin(2 * np.pi * self.phase(t)) + rng.normal(0, self.noise, len(t)), dy, g)
                for t, _, dy, g in series]
        nharm, _ = p.choose_harmonics(sine, 0, model, coeffs)
        self.assertLessEqual(nharm, 2)

    def test_minima_give_the_period_change_and_the_secondary(self):
        """The minima, timed one by one from a reference period a little off,
        give the rate the period changes at and the offset of the secondaries
        to within their uncertainties, and those uncertainties match the
        scatter of the minima."""
        series = self.light_curve()
        tmin = min(s[0].min() for s in series)
        tmax = max(s[0].max() for s in series)
        reference = self.period * (1 + 2e-6)
        epoch = 59000.0

        profile = p.eclipse_profile(series, 1.0 / reference, epoch)
        self.assertLess(abs(profile['primary'][0] - 0.0) % 1.0, 0.02)
        self.assertIsNotNone(profile['secondary'])

        minima = p.eclipse_minima(series, 1.0 / reference, epoch, profile)
        mt, me, ms, mi = (np.array([m[k] for m in minima]) for k in (0, 1, 2, 8))
        ms = ms.astype(bool)
        # Every eclipse of the four sectors, but those at their edges
        self.assertGreater((~ms).sum(), 85)
        self.assertGreater(ms.sum(), 85)

        model = p.PhaseModel(tmin, tmax, order=1)
        const, coeffs, cov, chi2, dof, counted, keep, errors = p.fit_minima(
            model, mt, me, ms, 1.0 / reference, classes=mi)

        f_mid = coeffs[0]
        pdot = -coeffs[1] / model.half / f_mid ** 2
        # The constant and delta come first in the covariance
        pdot_err = np.sqrt(cov[3, 3]) / model.half / f_mid ** 2
        self.assertLess(abs(pdot - self.pdot), 4 * pdot_err)
        self.assertLess(abs(const[1] - self.delta), 4 * np.sqrt(cov[1, 1]))

        # The cycles counted are the true ones
        true = np.round(self.phase(mt) - np.where(ms, 0.5 + self.delta, 0.0))
        fitted = np.round(counted - np.where(ms, 0.5 + const[1], 0.0))
        self.assertEqual(len(set(fitted - true)), 1)

        self.assertGreater(chi2 / dof, 0.7)
        self.assertLess(chi2 / dof, 1.3)


class TimeResolvedTest(SimpleTestCase):

    def test_peak_follows_a_drifting_period(self):
        """The periodogram in a sliding window finds the period of its own
        stretch of the light curve, not the mean of all."""
        rng = np.random.default_rng(11)
        t = nightly_times(rng, 500, per_night=4, span=1500)
        # The period running from 2.0 to 2.2 days over the data
        frac = (t - t.min()) / np.ptp(t)
        period = 2.0 + 0.2 * frac
        phase = np.cumsum(np.r_[0, np.diff(t) / period[1:]])
        y = 0.3 * np.sin(2 * np.pi * phase) + rng.normal(0, 0.03, len(t))

        result = p.dynamic_power([(t, y, np.full(len(t), 0.03))], 1.8, 2.5, 200.0, 100.0)
        periods = np.array(result['periods'])

        for centre, row in zip(result['mjd'], result['power']):
            if row is None:
                continue
            found = periods[int(np.nanargmax(np.array(row, dtype=float)))]
            expected = 2.0 + 0.2 * (centre - t.min()) / np.ptp(t)
            self.assertAlmostEqual(found, expected, delta=0.03)
