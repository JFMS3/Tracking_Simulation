"""Tests for imu.py: the MotionReading / ImuModel / simulate_imu interface, V1 and V2.

See docs/sensor_fusion_spec.md. Expected statistics are derived from the ImuModel
fields used in each test, never from hard-coded literature numbers, so the
implementer is free to choose (and cite) the realistic defaults.

Conventions these tests rely on (the contract):
- `simulate_imu(walk, rng, model, scan_interval_s)` returns one MotionReading per
  walk position; reading t covers the move walk[t-1] -> walk[t]; reading 0 is no motion.
- Headings are in radians, x-axis = 0, counter-clockwise positive. The "true
  walking heading" of interval t is the direction of the chord walk[t-1] -> walk[t];
  while the walker stands still the heading is held from the last move.
- Step counts come from the *true* distance and `true_step_length_m` with the
  fractional step carried over between intervals, so the cumulative count is
  always within one step of the true path length.
- The reported step length is the phone's step-length *model* output
  (true_step_length_m x (1 + per-walk bias) + per-step noise), never derived from
  the true distance of the interval: a real step counter has no sub-step
  information, so distance information comes only from the quantised step count.
  With every error off, step_length_m == true_step_length_m exactly.
- Step-length noise is per step and independent, so for an interval with n
  steps the reported mean step length deviates from the error-free value with
  std step_length_noise_m / sqrt(n).
- The walking-direction error (heading_noise_per_step_rad, heading outliers) is
  per step and does not accumulate: the heading changes summed up to a moving
  interval are off by that interval's direction error only (mean over its steps).
- simulate_imu uses only the `rng` it is given (never global random / np.random).
- Every error parameter is switched off by `imu_model()` below, which zeroes
  every numeric ImuModel field whose name contains one of ERROR_FIELD_MARKERS
  (and turns the magnetometer off). Any error parameter the implementer adds
  must follow that naming so these tests can isolate one error source at a time.
"""
import math
import random
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from random import Random
from unittest import mock

import numpy as np

from imu import ImuModel, MotionReading, simulate_imu
from network import Network
from scenarios import DECK_APS
from tracking import simulate_walk

LAYOUTS = Path(__file__).resolve().parent.parent / "layouts"

ERROR_FIELD_MARKERS = ("noise", "bias", "probability", "per_s", "std", "error")
MOTION_READING_FIELDS = {"step_count", "step_length_m", "heading_change_rad", "magnetic_heading_rad", "interval_s"}


def imu_model(**overrides) -> ImuModel:
    """An ImuModel with every error source off (a perfect phone), then `overrides` applied."""
    base = ImuModel()
    changes = {}
    for f in fields(ImuModel):
        value = getattr(base, f.name)
        if f.name == "magnetometer_available":
            changes[f.name] = False
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        elif any(marker in f.name for marker in ERROR_FIELD_MARKERS):
            changes[f.name] = type(value)(0)
    changes.update(overrides)
    return replace(base, **changes)


def wrap(angle):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi


def deck_network() -> Network:
    return Network.from_config(LAYOUTS / "accommodation_deck.yaml", access_points=list(DECK_APS))


_DECK_WALKS = {}


def deck_walks(n_walks: int, n_steps: int, first_seed: int = 100):
    """Realistic crew walks on the accommodation deck (shortest routes, doorways, dwells)."""
    key = (n_walks, n_steps, first_seed)
    if key not in _DECK_WALKS:
        network = deck_network()
        _DECK_WALKS[key] = [simulate_walk(network, Random(first_seed + i), n_steps=n_steps) for i in range(n_walks)]
    return _DECK_WALKS[key]


def synthetic_walk(rng: Random, n_steps: int, speed: float = 1.2, dt: float = 1.0):
    """Piecewise-straight walk with turns and dwells (positions need not lie on any deck)."""
    positions = [(0.0, 0.0)]
    heading = rng.uniform(-math.pi, math.pi)
    while len(positions) < n_steps:
        x, y = positions[-1]
        if rng.random() < 0.25:
            positions.extend([(x, y)] * rng.randint(1, 8))
            continue
        heading += rng.uniform(-2.0, 2.0)
        for _ in range(rng.randint(2, 12)):
            x, y = x + speed * dt * math.cos(heading), y + speed * dt * math.sin(heading)
            positions.append((x, y))
    return positions[:n_steps]


def true_distances(walk):
    """Chord length of every interval (index 0 = no motion)."""
    return [0.0] + [math.dist(a, b) for a, b in zip(walk, walk[1:])]


def true_headings(walk):
    """Chord heading of every interval, None where the walker did not move."""
    out = [None]
    for a, b in zip(walk, walk[1:]):
        out.append(math.atan2(b[1] - a[1], b[0] - a[0]) if math.dist(a, b) > 1e-9 else None)
    return out


class ImuInterfaceTests(unittest.TestCase):
    def test_one_reading_per_walk_position_with_valid_types(self) -> None:
        """Expected: len(readings) == len(walk) (also for 0 and 1 positions); every reading is a
        MotionReading with an int step_count >= 0, finite floats, and interval_s == scan_interval_s."""
        self.assertEqual(simulate_imu([], Random(1), ImuModel()), [])
        self.assertEqual(len(simulate_imu([(1.0, 1.0)], Random(1), ImuModel())), 1)
        walk = deck_walks(1, 120)[0]
        for interval in (1.0, 2.0):
            readings = simulate_imu(walk, Random(2), ImuModel(), scan_interval_s=interval)
            self.assertEqual(len(readings), len(walk))
            for r in readings:
                self.assertIsInstance(r, MotionReading)
                self.assertIsInstance(r.step_count, int)
                self.assertNotIsInstance(r.step_count, bool)
                self.assertGreaterEqual(r.step_count, 0)
                self.assertTrue(math.isfinite(r.step_length_m))
                self.assertTrue(math.isfinite(r.heading_change_rad))
                self.assertTrue(math.isfinite(r.distance_m))
                self.assertGreaterEqual(r.distance_m, 0.0)
                self.assertAlmostEqual(r.interval_s, interval)

    def test_reading_zero_is_no_motion(self) -> None:
        """Expected: reading 0 has step_count 0, distance 0 and heading change 0, even with the
        realistic default error model (it covers no time interval)."""
        for seed in range(5):
            first = simulate_imu(deck_walks(5, 30)[seed], Random(seed), ImuModel())[0]
            self.assertEqual(first.step_count, 0)
            self.assertEqual(first.distance_m, 0.0)
            self.assertEqual(first.heading_change_rad, 0.0)

    def test_motion_reading_carries_no_truth(self) -> None:
        """Expected: MotionReading has only the phone-observable fields of the spec and is frozen."""
        self.assertEqual({f.name for f in fields(MotionReading)}, MOTION_READING_FIELDS)
        reading = simulate_imu(deck_walks(1, 10)[0], Random(3), ImuModel())[1]
        with self.assertRaises(FrozenInstanceError):
            reading.step_count = 99  # type: ignore[misc]

    def test_readings_invariant_to_translating_and_rotating_the_walk(self) -> None:
        """Expected: with the magnetometer off, the readings depend only on the walk's shape (step
        distances and relative turns), so a walk translated by (37, -12) m and rotated by 90 degrees
        gives identical readings for the same seed. A reading that leaked absolute position or
        heading would differ."""
        walk = deck_walks(1, 150)[0]
        moved = [(-y + 37.0, x - 12.0) for x, y in walk]
        a = simulate_imu(walk, Random(11), ImuModel())
        b = simulate_imu(moved, Random(11), ImuModel())
        for t, (ra, rb) in enumerate(zip(a, b)):
            self.assertEqual(ra.step_count, rb.step_count, t)
            self.assertAlmostEqual(ra.step_length_m, rb.step_length_m, places=9, msg=t)
            self.assertAlmostEqual(float(wrap(ra.heading_change_rad - rb.heading_change_rad)), 0.0, places=9, msg=t)
            self.assertIsNone(ra.magnetic_heading_rad)
            self.assertIsNone(rb.magnetic_heading_rad)

    def test_deterministic_given_seed(self) -> None:
        """Expected: same walk, model and seed -> identical readings; a different seed -> different."""
        walk = deck_walks(1, 100)[0]
        self.assertEqual(simulate_imu(walk, Random(5), ImuModel()), simulate_imu(walk, Random(5), ImuModel()))
        self.assertNotEqual(simulate_imu(walk, Random(5), ImuModel()), simulate_imu(walk, Random(6), ImuModel()))

    def test_default_model_is_realistic_and_has_no_magnetometer(self) -> None:
        """Expected: defaults include non-zero step-length and gyro noise and a non-zero steel-hull
        magnetometer bias, but the magnetometer is off by default (magnetic_heading_rad None)."""
        model = ImuModel()
        self.assertFalse(model.magnetometer_available)
        self.assertGreater(model.step_length_noise_m, 0.0)
        self.assertGreater(model.gyro_noise_rad_per_sqrt_s, 0.0)
        self.assertGreater(model.magnetometer_bias_std_rad, 0.0)
        self.assertGreater(model.true_step_length_m, 0.0)
        readings = simulate_imu(deck_walks(1, 60)[0], Random(1), model)
        self.assertTrue(all(r.magnetic_heading_rad is None for r in readings))

    def test_invalid_parameters_raise_value_error(self) -> None:
        """Expected: ValueError (from ImuModel construction or from simulate_imu) for non-positive
        step length, negative noise/bias/rate, a probability outside [0, 1], NaN, and a
        non-positive scan interval."""
        walk = deck_walks(1, 20)[0]
        bad = [
            ("true_step_length_m", 0.0), ("true_step_length_m", -0.7), ("true_step_length_m", float("nan")),
            ("step_length_noise_m", -0.1), ("step_miss_probability", -0.1), ("step_miss_probability", 1.5),
            ("false_steps_per_s", -1.0), ("gyro_bias_std_rad_s", -0.01), ("gyro_noise_rad_per_sqrt_s", -0.01),
            ("gyro_noise_rad_per_sqrt_s", float("nan")), ("magnetometer_bias_std_rad", -0.1),
            ("magnetometer_noise_rad", -0.1),
        ]
        for name, value in bad:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                simulate_imu(walk, Random(0), ImuModel(**{name: value}))
        for interval in (0.0, -1.0):
            with self.subTest(scan_interval_s=interval), self.assertRaises(ValueError):
                simulate_imu(walk, Random(0), ImuModel(), scan_interval_s=interval)


class ImuStatisticsTests(unittest.TestCase):
    """V1: simulate_imu statistics match the ImuModel, within statistical tolerance."""

    def test_error_free_step_counts_and_distances_reproduce_path_length(self) -> None:
        """Expected: with every error off, every reported step_length_m equals true_step_length_m
        exactly (the step-length model output, not the true distance / steps of the interval, which
        would leak sub-step truth; this replaces the earlier "step length = true stride" contract),
        and at every epoch the cumulative step_count x true_step_length_m and the cumulative
        distance_m are each within one step length of the true cumulative path length (for two
        different step lengths); standing still gives 0 steps."""
        for step_length in (0.7, 0.6):
            model = imu_model(true_step_length_m=step_length)
            for w, walk in enumerate(deck_walks(4, 300)):
                readings = simulate_imu(walk, Random(w), model)
                self.assertTrue(all(r.step_length_m == step_length for r in readings), (step_length, w))
                true_cum = np.cumsum(true_distances(walk))
                count_cum = np.cumsum([r.step_count for r in readings]) * step_length
                dist_cum = np.cumsum([r.distance_m for r in readings])
                worst_count = float(np.max(np.abs(count_cum - true_cum)))
                worst_dist = float(np.max(np.abs(dist_cum - true_cum)))
                self.assertLessEqual(worst_count, step_length + 1e-9, (step_length, w, worst_count))
                self.assertLessEqual(worst_dist, step_length + 1e-9, (step_length, w, worst_dist))
                for d, r in zip(true_distances(walk), readings):
                    if d == 0.0:
                        self.assertEqual(r.step_count, 0)

    def test_error_free_heading_change_equals_true_turn(self) -> None:
        """Expected: with every error off, the heading changes summed between two consecutive
        moving intervals equal the true change in chord heading (to 1e-9 rad), and every
        stationary interval reports a heading change of exactly 0."""
        model = imu_model()
        for w, walk in enumerate(deck_walks(4, 300) + [synthetic_walk(Random(9), 300)]):
            readings = simulate_imu(walk, Random(w), model)
            headings = true_headings(walk)
            last_move, accumulated = None, 0.0
            for t in range(1, len(walk)):
                accumulated += readings[t].heading_change_rad
                if headings[t] is None:
                    self.assertEqual(readings[t].heading_change_rad, 0.0, (w, t))
                    continue
                if last_move is not None:
                    expected = headings[t] - headings[last_move]
                    self.assertAlmostEqual(float(wrap(accumulated - expected)), 0.0, places=9, msg=(w, t))
                last_move, accumulated = t, 0.0

    def test_step_length_noise_matches_model(self) -> None:
        """Expected: with only step-length noise sigma, step counts equal the error-free counts and
        z = (step_length_m - error-free step_length_m) * sqrt(step_count) has mean 0 (within 4
        standard errors) and std sigma (within 10%)."""
        sigma = 0.12
        noisy_model, clean_model = imu_model(step_length_noise_m=sigma), imu_model()
        z = []
        for w, walk in enumerate(deck_walks(10, 300)):
            noisy = simulate_imu(walk, Random(w), noisy_model)
            clean = simulate_imu(walk, Random(w), clean_model)
            for rn, rc in zip(noisy, clean):
                self.assertEqual(rn.step_count, rc.step_count)
                if rc.step_count > 0:
                    z.append((rn.step_length_m - rc.step_length_m) * math.sqrt(rc.step_count))
        z = np.array(z)
        self.assertGreater(len(z), 1000)
        self.assertLess(abs(z.mean()), 4 * sigma / math.sqrt(len(z)), f"mean {z.mean():.4f}, n={len(z)}")
        self.assertAlmostEqual(z.std() / sigma, 1.0, delta=0.10, msg=f"std {z.std():.4f} vs {sigma}")

    def test_missed_step_rate_matches_model(self) -> None:
        """Expected: with only step_miss_probability p, no interval reports more steps than the
        error-free phone, and the total count is Binomial(N_true, 1 - p): within 4 binomial
        standard deviations (+1 step per walk for the step phase)."""
        p = 0.25
        detected, true_total, walks = 0, 0, deck_walks(10, 300)
        for w, walk in enumerate(walks):
            noisy = simulate_imu(walk, Random(w), imu_model(step_miss_probability=p))
            clean = simulate_imu(walk, Random(w), imu_model())
            for rn, rc in zip(noisy, clean):
                self.assertLessEqual(rn.step_count, rc.step_count)
            detected += sum(r.step_count for r in noisy)
            true_total += sum(r.step_count for r in clean)
        expected = true_total * (1 - p)
        tolerance = 4 * math.sqrt(true_total * p * (1 - p)) + len(walks)
        self.assertLess(abs(detected - expected), tolerance,
                        f"detected {detected} of {true_total} true steps, expected {expected:.0f} +- {tolerance:.0f}")

    def test_false_step_rate_matches_model(self) -> None:
        """Expected: with only false_steps_per_s = lam, no interval reports fewer steps than the
        error-free phone; the extra steps over total time T are Poisson(lam*T) (within 4 std + 1 per
        walk); and false steps also occur while standing still (ship motion)."""
        lam, interval = 0.2, 1.0
        extra, extra_standing, total_time, walks = 0, 0, 0.0, deck_walks(10, 300)
        for w, walk in enumerate(walks):
            noisy = simulate_imu(walk, Random(w), imu_model(false_steps_per_s=lam), scan_interval_s=interval)
            clean = simulate_imu(walk, Random(w), imu_model(), scan_interval_s=interval)
            for d, rn, rc in zip(true_distances(walk), noisy, clean):
                self.assertGreaterEqual(rn.step_count, rc.step_count)
                extra += rn.step_count - rc.step_count
                if d == 0.0:
                    extra_standing += rn.step_count - rc.step_count
            total_time += (len(walk) - 1) * interval
        expected = lam * total_time
        tolerance = 4 * math.sqrt(expected) + len(walks)
        self.assertLess(abs(extra - expected), tolerance, f"{extra} false steps, expected {expected:.0f} +- {tolerance:.0f}")
        self.assertGreater(extra_standing, 0)

    def test_gyro_noise_heading_error_variance_grows_linearly_with_time(self) -> None:
        """Expected (angle random walk): with only gyro noise sigma (rad/sqrt(s)), the accumulated
        heading error after T seconds has variance sigma^2 * T. Fitted slope (through the origin, T
        in seconds, over 600 runs) within 15% of sigma^2 for scan intervals of 1 s and 2 s, and
        var(T=60 s) / var(T=15 s) within [2.5, 6] (linear growth gives 4, a bias would give 16)."""
        sigma = 0.02
        walk = synthetic_walk(Random(4), 61)
        for interval in (1.0, 2.0):
            clean = simulate_imu(walk, Random(0), imu_model(), scan_interval_s=interval)
            clean_turns = np.array([r.heading_change_rad for r in clean[1:]])
            errors = np.empty((600, len(walk) - 1))
            for i in range(errors.shape[0]):
                noisy = simulate_imu(walk, Random(10_000 + i), imu_model(gyro_noise_rad_per_sqrt_s=sigma),
                                     scan_interval_s=interval)
                errors[i] = wrap(np.array([r.heading_change_rad for r in noisy[1:]]) - clean_turns)
            accumulated = np.cumsum(errors, axis=1)
            seconds = interval * np.arange(1, len(walk))
            variance = np.mean(accumulated ** 2, axis=0)
            slope = float(seconds @ variance / (seconds @ seconds))
            self.assertAlmostEqual(slope / sigma ** 2, 1.0, delta=0.15,
                                   msg=f"interval {interval}: slope {slope:.3e} vs sigma^2 {sigma ** 2:.3e}")
            k60, k15 = int(round(60 / interval)) - 1, int(round(15 / interval)) - 1
            ratio = variance[k60] / variance[k15]
            self.assertTrue(2.5 <= ratio <= 6.0, f"interval {interval}: var(60 s)/var(15 s) = {ratio:.2f}")

    def test_gyro_bias_is_a_per_walk_constant_rate(self) -> None:
        """Expected: with only gyro bias std s (rad/s), every interval of a walk carries the same
        heading error b * dt (constant within a walk, to 1e-9 rad), and b across 400 walks has
        mean 0 (within 4 standard errors) and std s (within 15%)."""
        s, interval = 0.005, 2.0
        walk = synthetic_walk(Random(5), 21)
        clean = simulate_imu(walk, Random(0), imu_model(), scan_interval_s=interval)
        clean_turns = np.array([r.heading_change_rad for r in clean[1:]])
        rates = []
        for i in range(400):
            noisy = simulate_imu(walk, Random(20_000 + i), imu_model(gyro_bias_std_rad_s=s), scan_interval_s=interval)
            errors = wrap(np.array([r.heading_change_rad for r in noisy[1:]]) - clean_turns)
            self.assertLess(float(np.ptp(errors)), 1e-9, f"walk {i}: errors not constant {errors[:4]}")
            rates.append(errors[0] / interval)
        rates = np.array(rates)
        self.assertLess(abs(rates.mean()), 4 * s / math.sqrt(len(rates)), f"mean bias {rates.mean():.2e}")
        self.assertAlmostEqual(rates.std() / s, 1.0, delta=0.15, msg=f"bias std {rates.std():.2e} vs {s:.2e}")

    def test_magnetometer_noise_matches_model(self) -> None:
        """Expected: with the magnetometer on, bias 0 and noise sigma_m, every moving reading has a
        magnetic heading whose error against the true chord heading has mean 0 (within 4 standard
        errors) and std sigma_m (within 15%)."""
        sigma_m = 0.05
        model = imu_model(magnetometer_available=True, magnetometer_noise_rad=sigma_m)
        errors = []
        for w, walk in enumerate(deck_walks(6, 300)):
            readings = simulate_imu(walk, Random(w), model)
            for heading, r in zip(true_headings(walk), readings):
                if heading is None:
                    continue
                self.assertIsNotNone(r.magnetic_heading_rad)
                errors.append(float(wrap(r.magnetic_heading_rad - heading)))
        errors = np.array(errors)
        self.assertLess(abs(errors.mean()), 4 * sigma_m / math.sqrt(len(errors)), f"mean {errors.mean():.4f}")
        self.assertAlmostEqual(errors.std() / sigma_m, 1.0, delta=0.15, msg=f"std {errors.std():.4f} vs {sigma_m}")

    def test_magnetometer_bias_is_large_and_varies_in_space(self) -> None:
        """Expected: with only magnetometer bias std B (no noise), the bias is a function of
        position: while standing still the magnetic heading does not change; it varies along a walk
        (spread of errors > 0.2 B in every walk), and its RMS over the deck walks is within
        [0.4 B, 2 B]."""
        B = 0.5
        model = imu_model(magnetometer_available=True, magnetometer_bias_std_rad=B)
        all_errors = []
        for w, walk in enumerate(deck_walks(6, 300)):
            readings = simulate_imu(walk, Random(w), model)
            headings = true_headings(walk)
            held, errors = None, []
            for t in range(1, len(walk)):
                self.assertIsNotNone(readings[t].magnetic_heading_rad, (w, t))
                if headings[t] is not None:
                    held = headings[t]
                elif held is not None:
                    self.assertAlmostEqual(
                        float(wrap(readings[t].magnetic_heading_rad - readings[t - 1].magnetic_heading_rad)), 0.0,
                        places=9, msg=(w, t))
                if held is not None and readings[t].magnetic_heading_rad is not None:
                    errors.append(float(wrap(readings[t].magnetic_heading_rad - held)))
            self.assertGreater(len(errors), 100)
            self.assertGreater(max(errors) - min(errors), 0.2 * B, f"walk {w}: bias does not vary in space")
            all_errors.extend(errors)
        rms = math.sqrt(np.mean(np.square(all_errors)))
        self.assertTrue(0.4 * B <= rms <= 2.0 * B, f"magnetometer error RMS {rms:.3f} rad vs bias std {B}")


def heading_errors(noisy, clean):
    """Per-reading heading-change error of `noisy` against the error-free `clean` readings."""
    return wrap(np.array([rn.heading_change_rad - rc.heading_change_rad for rn, rc in zip(noisy, clean)]))


def outlier_sum_nonzero_probability(n: int, p: float) -> float:
    """P(the n per-step outlier spikes of one interval, each +-A with probability p, do not cancel)."""
    total = 0.0
    for k in range(1, n + 1):
        cancel = math.comb(k, k // 2) / 2 ** k if k % 2 == 0 else 0.0
        total += math.comb(n, k) * p ** k * (1 - p) ** (n - k) * (1.0 - cancel)
    return total


class StepLengthLeakTests(unittest.TestCase):
    """The reported step length carries no sub-step truth: distance comes only from step counts."""

    def test_perfect_imu_distance_is_a_whole_number_of_model_steps(self) -> None:
        """Expected: with every error off, each interval's distance_m is an integer multiple of
        true_step_length_m (0, L, 2L, ...), even though the true chord lengths are not, i.e. the
        phone cannot know the true distance to better than one step."""
        step_length = 0.7
        model = imu_model(true_step_length_m=step_length)
        mismatched = 0
        for w, walk in enumerate(deck_walks(4, 300) + [synthetic_walk(Random(9), 300)]):
            readings = simulate_imu(walk, Random(w), model)
            for t, (d, r) in enumerate(zip(true_distances(walk), readings)):
                multiple = r.distance_m / step_length
                self.assertAlmostEqual(multiple, round(multiple), places=9, msg=(w, t, r.distance_m))
                mismatched += abs(r.distance_m - d) > 0.1
        self.assertGreater(mismatched, 100, "true chords are not whole steps, so the test must see mismatches")

    def test_default_step_lengths_stay_near_the_model_and_ignore_true_distance(self) -> None:
        """Expected: with the realistic defaults, >= 98% of reported step lengths (intervals with
        steps) lie within +-30% of true_step_length_m (the remainder is Gaussian per-step noise on
        single-step intervals), and the step length is uncorrelated with the true chord length of
        the interval (|r| < 0.1); a step length derived from distance / steps (0.2-1.2 m) fails both."""
        model = ImuModel()
        lengths, chords = [], []
        for w, walk in enumerate(deck_walks(10, 300)):
            for d, r in zip(true_distances(walk), simulate_imu(walk, Random(w), model)):
                if r.step_count > 0:
                    lengths.append(r.step_length_m)
                    chords.append(d)
        lengths, chords = np.array(lengths), np.array(chords)
        within = float(np.mean(np.abs(lengths / model.true_step_length_m - 1.0) <= 0.30))
        self.assertGreaterEqual(within, 0.98, f"only {within:.1%} of step lengths within +-30%")
        moving = chords > 0
        r = float(np.corrcoef(lengths[moving], chords[moving])[0, 1])
        self.assertLess(abs(r), 0.1, f"step length correlates with the true chord length (r = {r:.3f})")


class ImuErrorSourceTests(unittest.TestCase):
    """V1 for the additional error sources: each matches its ImuModel field in isolation."""

    def test_false_steps_per_true_step_while_walking(self) -> None:
        """Expected: with only false_step_probability p, no interval reports fewer steps than the
        error-free phone, standing still never gains steps, and the extra steps are
        Binomial(N_true, p) (within 4 binomial standard deviations)."""
        p = 0.2
        extra, true_total = 0, 0
        for w, walk in enumerate(deck_walks(10, 300)):
            noisy = simulate_imu(walk, Random(w), imu_model(false_step_probability=p))
            clean = simulate_imu(walk, Random(w), imu_model())
            for d, rn, rc in zip(true_distances(walk), noisy, clean):
                self.assertGreaterEqual(rn.step_count, rc.step_count)
                if d == 0.0:
                    self.assertEqual(rn.step_count, 0)
                extra += rn.step_count - rc.step_count
                true_total += rc.step_count
        expected = true_total * p
        tolerance = 4 * math.sqrt(true_total * p * (1 - p))
        self.assertLess(abs(extra - expected), tolerance, f"{extra} false steps, expected {expected:.0f} +- {tolerance:.0f}")

    def test_step_length_bias_is_a_per_walk_scale(self) -> None:
        """Expected: with only step_length_bias_std_frac s, step counts equal the error-free ones,
        every reading of a walk reports the same step length L * (1 + b) (to 1e-12), and b across
        400 walks has mean 0 (within 4 standard errors) and std s (within 15%)."""
        s, step_length = 0.05, 0.7
        walk = synthetic_walk(Random(6), 21)
        clean = simulate_imu(walk, Random(0), imu_model(true_step_length_m=step_length))
        biases = []
        for i in range(400):
            noisy = simulate_imu(walk, Random(30_000 + i), imu_model(true_step_length_m=step_length,
                                                                     step_length_bias_std_frac=s))
            self.assertEqual([r.step_count for r in noisy], [r.step_count for r in clean])
            lengths = np.array([r.step_length_m for r in noisy])
            self.assertLess(float(np.ptp(lengths)), 1e-12, f"walk {i}: step length not constant")
            biases.append(lengths[0] / step_length - 1.0)
        biases = np.array(biases)
        self.assertLess(abs(biases.mean()), 4 * s / math.sqrt(len(biases)), f"mean bias {biases.mean():.4f}")
        self.assertAlmostEqual(biases.std() / s, 1.0, delta=0.15, msg=f"bias std {biases.std():.4f} vs {s}")

    def test_gyro_scale_factor_scales_turns_by_a_per_walk_constant(self) -> None:
        """Expected: with only gyro_scale_factor_error_std s, the heading change of every turning
        interval is (1 + e) x the true turn, with e constant within a walk (to 1e-9; U-turns near
        +-pi excluded because their sense is ambiguous), straight and stationary intervals carry no
        error, and e across 400 walks has mean 0 (within 4 standard errors) and std s (within 15%)."""
        s = 0.02
        walk = synthetic_walk(Random(5), 41)
        clean = simulate_imu(walk, Random(0), imu_model())
        turns = np.array([r.heading_change_rad for r in clean])
        turning = (np.abs(turns) > 0.05) & (np.abs(turns) < math.pi - 0.05)
        self.assertGreaterEqual(int(turning.sum()), 3)
        scales = []
        for i in range(400):
            noisy = simulate_imu(walk, Random(40_000 + i), imu_model(gyro_scale_factor_error_std=s))
            errors = heading_errors(noisy, clean)
            self.assertLess(float(np.max(np.abs(errors[turns == 0.0]))), 1e-12, f"walk {i}: error without a turn")
            ratios = errors[turning] / turns[turning]
            self.assertLess(float(np.ptp(ratios)), 1e-9, f"walk {i}: scale factor not constant {ratios}")
            scales.append(ratios[0])
        scales = np.array(scales)
        self.assertLess(abs(scales.mean()), 4 * s / math.sqrt(len(scales)), f"mean scale error {scales.mean():.4f}")
        self.assertAlmostEqual(scales.std() / s, 1.0, delta=0.15, msg=f"scale error std {scales.std():.4f} vs {s}")

    def test_heading_outliers_are_rare_bounded_and_do_not_accumulate(self) -> None:
        """Expected: with only heading_outlier_probability p and outliers of exactly +-A, the heading
        error accumulated up to a moving interval with n steps is k*A/n for an integer |k| <= n (that
        interval's spikes averaged over its steps), so it never exceeds A and does not accumulate;
        stationary intervals carry no error; and the number of moving intervals with a non-zero
        accumulated error matches p (within 4 standard deviations)."""
        p, A = 0.1, 0.5
        model = imu_model(heading_outlier_probability=p, heading_outlier_min_rad=A, heading_outlier_max_rad=A)
        observed, expected, variance = 0, 0.0, 0.0
        for w, walk in enumerate(deck_walks(6, 300)):
            noisy = simulate_imu(walk, Random(w), model)
            clean = simulate_imu(walk, Random(w), imu_model())
            errors = heading_errors(noisy, clean)
            accumulated = np.cumsum(errors)
            for t, heading in enumerate(true_headings(walk)):
                if heading is None:
                    self.assertLess(abs(errors[t]), 1e-12, (w, t))
                    continue
                n = max(clean[t].step_count, 1)
                k = accumulated[t] * n / A
                self.assertAlmostEqual(k, round(k), places=6, msg=(w, t))
                self.assertLessEqual(abs(round(k)), n, (w, t))
                observed += round(k) != 0
                q = outlier_sum_nonzero_probability(n, p)
                expected += q
                variance += q * (1 - q)
        self.assertLess(abs(observed - expected), 4 * math.sqrt(variance),
                        f"{observed} intervals with outliers, expected {expected:.0f}")

    def test_per_step_heading_noise_does_not_accumulate(self) -> None:
        """Expected: with only heading_noise_per_step_rad sigma, the heading error accumulated up to a
        moving interval with n steps is that interval's direction error, so z = error * sqrt(n) has
        mean 0 (within 4 standard errors) and std sigma (within 10%), with the same spread in the
        first and second half of the walks (ratio within [0.85, 1.15]: no random walk), and
        stationary intervals carry no error."""
        sigma = 0.05
        early, late = [], []
        for w, walk in enumerate(deck_walks(6, 300)):
            noisy = simulate_imu(walk, Random(w), imu_model(heading_noise_per_step_rad=sigma))
            clean = simulate_imu(walk, Random(w), imu_model())
            errors = heading_errors(noisy, clean)
            accumulated = np.cumsum(errors)
            for t, heading in enumerate(true_headings(walk)):
                if heading is None:
                    self.assertLess(abs(errors[t]), 1e-12, (w, t))
                    continue
                z = accumulated[t] * math.sqrt(max(clean[t].step_count, 1))
                (early if t < len(walk) // 2 else late).append(z)
        z = np.array(early + late)
        self.assertLess(abs(z.mean()), 4 * sigma / math.sqrt(len(z)), f"mean {z.mean():.4f}")
        self.assertAlmostEqual(z.std() / sigma, 1.0, delta=0.10, msg=f"std {z.std():.4f} vs {sigma}")
        ratio = np.std(late) / np.std(early)
        self.assertTrue(0.85 <= ratio <= 1.15, f"late/early spread {ratio:.2f}: heading error accumulates")

    def test_uses_only_the_given_rng(self) -> None:
        """Expected: with every error source and the magnetometer on, simulate_imu never touches the
        global `random` or `np.random` state: patching their functions to raise changes nothing,
        and reseeding them does not change the readings."""
        model = replace(ImuModel(), magnetometer_available=True, ship_yaw_rate_std_rad_s=0.001,
                        heading_outlier_probability=0.2, false_steps_per_s=0.5)
        walk = deck_walks(1, 120)[0]
        random.seed(1)
        np.random.seed(1)
        reference = simulate_imu(walk, Random(12), model)
        random.seed(2)
        np.random.seed(2)
        self.assertEqual(simulate_imu(walk, Random(12), model), reference)

        def forbidden(*args, **kwargs):
            raise AssertionError("simulate_imu used global random state")

        python_names = ["random", "gauss", "uniform", "randint", "randrange", "choice", "choices", "sample",
                        "shuffle", "expovariate", "normalvariate", "lognormvariate", "getrandbits", "seed"]
        numpy_names = ["random", "rand", "randn", "normal", "uniform", "randint", "choice", "seed", "default_rng",
                       "standard_normal", "poisson", "binomial", "exponential", "shuffle", "permutation"]
        with mock.patch.multiple(random, **{name: forbidden for name in python_names}), \
                mock.patch.multiple(np.random, **{name: forbidden for name in numpy_names}):
            self.assertEqual(simulate_imu(walk, Random(12), model), reference)


class PdrDriftTests(unittest.TestCase):
    """V2: PDR alone drifts, so the IMU is not an oracle (but it is not absurd either)."""

    @staticmethod
    def dead_reckon(walk, readings, horizon: int):
        """Integrate PDR from the true start. The absolute heading is aligned so that the
        integrated heading equals the true heading at the first moving interval (whatever the
        implementation reports before the first move); after that only readings are used."""
        headings = true_headings(walk)
        first = next(t for t in range(1, len(walk)) if headings[t] is not None)
        heading = headings[first] - sum(r.heading_change_rad for r in readings[1:first + 1])
        x, y = walk[0]
        track = [(x, y)]
        for r in readings[1:horizon + 1]:
            heading += r.heading_change_rad
            x, y = x + r.distance_m * math.cos(heading), y + r.distance_m * math.sin(heading)
            track.append((x, y))
        return track

    def test_dead_reckoning_error_after_120_s_exceeds_2_m(self) -> None:
        """Expected: with the realistic default ImuModel (gyro only), dead reckoning from the true
        start with the true initial heading has a mean position error at 120 s across 20 deck
        walks of more than 2 m (not an oracle) and less than 30 m (sane), and the mean error at
        120 s exceeds the mean error at 30 s (drift grows)."""
        walks = deck_walks(20, 121, first_seed=500)
        rng = Random(7)
        err_30, err_120 = [], []
        for walk in walks:
            readings = simulate_imu(walk, rng, ImuModel())
            track = self.dead_reckon(walk, readings, 120)
            err_30.append(math.dist(track[30], walk[30]))
            err_120.append(math.dist(track[120], walk[120]))
        mean_120, mean_30 = float(np.mean(err_120)), float(np.mean(err_30))
        self.assertGreater(mean_120, 2.0, f"PDR error at 120 s {mean_120:.2f} m (at 30 s {mean_30:.2f} m)")
        self.assertLess(mean_120, 30.0, f"PDR error at 120 s {mean_120:.2f} m")
        self.assertGreater(mean_120, mean_30, f"PDR error 30 s {mean_30:.2f} m, 120 s {mean_120:.2f} m")


if __name__ == "__main__":
    unittest.main()
