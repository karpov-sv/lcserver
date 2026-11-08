"""Tests of the variability indices and the structure function.

On synthetic light curves with fixed seeds: white noise, which must read as
no variation at all, and a damped random walk, whose timescale and amplitude
the structure function is there to recover.

No database: run with `python manage.py test lcserver.tests`.
"""

import numpy as np
from django.test import SimpleTestCase

from lcserver.processing import variability as v


def random_walk(t, tau, sigma, rng):
    """A damped random walk of timescale `tau` and rms `sigma` at sorted `t`."""
    y = np.empty(len(t))
    y[0] = rng.normal(0, sigma)
    for k in range(1, len(t)):
        a = np.exp(-(t[k] - t[k - 1]) / tau)
        y[k] = a * y[k - 1] + rng.normal(0, sigma * np.sqrt(1 - a * a))
    return y


class IndicesTest(SimpleTestCase):
    def test_noise_is_not_variable(self):
        rng = np.random.default_rng(1)
        t = np.sort(rng.uniform(0, 1000, 800))
        dy = np.full(len(t), 0.05)
        out = v.variability_indices(t, 15 + rng.normal(0, 0.05, len(t)), dy)
        self.assertAlmostEqual(out['chi2'], 1.0, delta=0.15)
        self.assertGreater(out['log10_p'], -3)
        self.assertFalse(out['rms_detected'])
        self.assertAlmostEqual(out['eta'], 2.0, delta=0.25)
        self.assertAlmostEqual(out['stetson_k'], 0.80, delta=0.05)

    def test_slow_variation(self):
        rng = np.random.default_rng(2)
        t = np.sort(rng.uniform(0, 3000, 1000))
        y = 15 + random_walk(t, 100, 0.2, rng) + rng.normal(0, 0.02, len(t))
        out = v.variability_indices(t, y, np.full(len(t), 0.02))
        self.assertTrue(out['rms_detected'])
        self.assertAlmostEqual(out['rms'], 0.2, delta=0.06)
        self.assertLess(out['eta'], 0.5)
        # Underflowing the exact tail, and still a number
        self.assertTrue(np.isfinite(out['log10_p']))
        self.assertLess(out['log10_p'], -100)

    def test_trend_per_year(self):
        rng = np.random.default_rng(3)
        t = np.sort(rng.uniform(0, 2 * v.YEAR_DAYS, 300))
        y = 15 + 0.1 * t / v.YEAR_DAYS + rng.normal(0, 0.01, len(t))
        out = v.variability_indices(t, y, np.full(len(t), 0.01))
        self.assertAlmostEqual(out['slope'], 0.1, delta=3 * out['slope_err'])


class StructureFunctionTest(SimpleTestCase):
    def test_pairs_sampled_evenly(self):
        """Beyond the cap, pairs are drawn with every pair as likely - their
        lags spread as those of all the pairs do."""
        rng = np.random.default_rng(4)
        t = np.sort(np.concatenate([rng.uniform(0, 10, 3000), rng.uniform(100, 101, 300)]))
        i, j = v.lag_pairs(t, 1.0, 50.0, rng, max_pairs=10 ** 8)
        si, sj = v.lag_pairs(t, 1.0, 50.0, rng, max_pairs=20000)
        self.assertEqual(len(si), 20000)
        self.assertTrue(np.all((t[sj] - t[si] >= 1.0) & (t[sj] - t[si] < 50.0)))
        self.assertAlmostEqual(np.median(t[sj] - t[si]), np.median(t[j] - t[i]), delta=0.2)

    def test_noise_is_taken_off(self):
        rng = np.random.default_rng(5)
        t = np.sort(rng.uniform(0, 1000, 1000))
        dy = np.full(len(t), 0.1)
        bins = v.structure_function([(t, rng.normal(0, 0.1, len(t)), dy)])
        self.assertTrue(bins)
        self.assertLessEqual(sum(b['detected'] for b in bins), 1)

    def test_random_walk_timescale(self):
        rng = np.random.default_rng(6)
        t = np.sort(rng.uniform(0, 3000, 1500))
        y = random_walk(t, 30, 0.2, rng) + rng.normal(0, 0.05, len(t))
        bins = v.structure_function([(t, y, np.full(len(t), 0.05))])
        drw = v.fit_drw(bins, 3000)
        self.assertFalse(drw['tau_limit'])
        self.assertLess(abs(np.log(drw['tau'] / 30)), np.log(1.5))
        self.assertAlmostEqual(drw['sf_inf'], 0.2 * np.sqrt(2), delta=0.05)

    def test_few_points_far_above_noise(self):
        """A sparse survey varying far above its noise is detected, not taken
        for an upper limit because its few pairs leave the value uncertain."""
        rng = np.random.default_rng(7)
        t = np.sort(rng.uniform(0, 3000, 40))
        y = rng.normal(0, 0.5, len(t))
        bins = v.structure_function([(t, y, np.full(len(t), 0.01))])
        long = [b for b in bins if b['lag'] > 100]
        self.assertTrue(long)
        self.assertTrue(all(b['detected'] for b in long))

    def test_pairs_within_series_only(self):
        """Two series at different zero points: their offset is no variation."""
        rng = np.random.default_rng(8)
        sets = []
        for zero in (0.0, 1.0):
            t = np.sort(rng.uniform(0, 1000, 300))
            sets.append((t, zero + rng.normal(0, 0.05, len(t)), np.full(len(t), 0.05)))
        bins = v.structure_function(sets)
        self.assertLessEqual(sum(b['detected'] for b in bins), 1)
