"""Tests for fusion.py: ParticleTracker unit behaviour plus V3, V5 and V6 of docs/sensor_fusion_spec.md.

Interface under test (fixed by the spec):
    ParticleTracker(localiser, network, rng, n_particles=2000, imu_model=None,
                    use_magnetometer=False, scan_interval_s=1.0)
    .reset(), .update(scan, motion) -> LocalisationEstimate (with a 2x2 covariance), .delay_steps == 0

Internal state these tests assume (keep these names):
    tracker.particles : array-like (n_particles, >=2), columns 0 and 1 are x and y in metres
                        (further columns, e.g. heading, are free)
    tracker.weights   : array-like (n_particles,), non-negative, summing to 1 after every update();
                        a particle killed by a wall has weight 0
Both are read after update() returns; the particle count stays n_particles.

Performance-type checks run on the *tuning* seeds of the spec (walks 67890, scans 78901,
IMU 89012); the held-out seeds are reserved for tests/test_fusion_targets.py.
"""
import math
import random
import unittest
from dataclasses import fields, replace
from pathlib import Path
from random import Random
from unittest import mock

import numpy as np
from shapely.geometry import Point as ShapelyPoint

from environment import Compartment, ShipEnvironment, Wall
from fusion import ParticleTracker
from imu import ImuModel, MotionReading, simulate_imu
from network import AccessPoint, Network, Receiver
from positioning import LocalisationEstimate, WallAwareLocaliser
from scenarios import DECK_APS
from tracking import GridTracker, evaluate_tracking, simulate_walk

LAYOUTS = Path(__file__).resolve().parent.parent / "layouts"

NOISE_FREE = {
    "shadowing_std_db": 0.0,
    "fast_fading_samples": 0,
    "measurement_noise_std_db": 0.0,
    "rssi_quantization_db": 0.0,
    "noise_floor_variation_std_db": 0.0,
    "interference_burst_start_probability": 0.0,
    "interference_burst_end_probability": 1.0,
    "detection_snr_midpoint_db": -100.0,
    "rtt_ranging_noise_m": 0.0,
    "rtt_nlos_noise_per_wall_m": 0.0,
    "rtt_weak_signal_penalty_m": 0.0,
    "rtt_outlier_probability": 0.0,
    "rtt_nlos_outlier_probability": 0.0,
}

TWO_ROOM_APS = [
    AccessPoint("AP-A", "00:00:00:00:00:01", (1.0, 1.0, 1.0), 2412),
    AccessPoint("AP-B", "00:00:00:00:00:02", (7.0, 3.0, 1.0), 2412),
    AccessPoint("AP-C", "00:00:00:00:00:03", (3.0, 3.5, 1.0), 2412),
]

DECK_BOUNDS = (0.0, 0.0, 24.0, 10.0)
TUNING_WALK_SEED, TUNING_SCAN_SEED, TUNING_IMU_SEED = 67890, 78901, 89012
STAND_STILL = MotionReading(step_count=0, step_length_m=0.7, heading_change_rad=0.0)

_ERROR_FIELD_MARKERS = ("noise", "bias", "probability", "per_s", "std", "error")


def perfect_imu_model() -> ImuModel:
    """ImuModel with every error source off (same rule as tests/test_imu.py)."""
    base = ImuModel()
    changes = {}
    for f in fields(ImuModel):
        value = getattr(base, f.name)
        if f.name == "magnetometer_available":
            changes[f.name] = False
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        elif any(marker in f.name for marker in _ERROR_FIELD_MARKERS):
            changes[f.name] = type(value)(0)
    return replace(base, **changes)


def deck_network(noise_free: bool = False, **overrides) -> Network:
    base = Network.from_config(LAYOUTS / "accommodation_deck.yaml", access_points=list(DECK_APS))
    changes = dict(NOISE_FREE) if noise_free else {}
    changes.update(overrides)
    return replace(base, **changes)


def two_room_network(door: bool, noise_free: bool) -> Network:
    """Rooms A (x 0-4) and B (x 4-8), y 0-4, inside a hull. The divider at x = 4 is a solid
    20 dB bulkhead, with a doorway at y 1.5-2.5 if `door`."""
    hull = [
        Wall("Hull S", (0.0, 0.0), (8.0, 0.0)), Wall("Hull E", (8.0, 0.0), (8.0, 4.0)),
        Wall("Hull N", (8.0, 4.0), (0.0, 4.0)), Wall("Hull W", (0.0, 4.0), (0.0, 0.0)),
    ]
    if door:
        divider = [Wall("Divider low", (4.0, 0.0), (4.0, 1.5)), Wall("Divider high", (4.0, 2.5), (4.0, 4.0))]
    else:
        divider = [Wall("Divider", (4.0, 0.0), (4.0, 4.0))]
    environment = ShipEnvironment(
        compartments=[Compartment("A", [(0, 0), (4, 0), (4, 4), (0, 4)]), Compartment("B", [(4, 0), (8, 0), (8, 4), (4, 4)])],
        walls=hull + divider,
    )
    return Network(environment=environment, access_points=list(TWO_ROOM_APS), **(NOISE_FREE if noise_free else {}))


def particle_state(tracker: ParticleTracker):
    return np.asarray(tracker.particles, dtype=float), np.asarray(tracker.weights, dtype=float)


def scan_at(network: Network, xy, rng: Random):
    return network.scan(Receiver("crew", (float(xy[0]), float(xy[1]), 1.0)), rng)


def all_missing(network: Network):
    return {ap: None for ap in network.access_points}


def fmt(stats: dict) -> str:
    return ", ".join(f"{name}: mean {s['mean_err']:.3f} m, P90 {s['p90_err']:.3f} m, room acc {s['compartment_acc']:.2%}, "
                     f"ratio {s['room_changes_ratio']:.2f}, cov95 {s['coverage95']:.2f}" for name, s in stats.items())


class _Deck:
    """Shared realistic-deck fixtures (built once)."""
    _cache = {}

    @classmethod
    def get(cls):
        if not cls._cache:
            network = deck_network()
            cls._cache["network"] = network
            cls._cache["localiser"] = WallAwareLocaliser(network, measurement="hybrid", grid_spacing_m=0.2, robust=True)
            walk_rng = Random(TUNING_WALK_SEED)
            cls._cache["walks"] = [simulate_walk(network, walk_rng, n_steps=150) for _ in range(4)]
        return cls._cache


class ParticleTrackerInterfaceTests(unittest.TestCase):
    def test_constructor_validates_arguments(self) -> None:
        """Expected: ValueError for n_particles <= 0 and for scan_interval_s <= 0."""
        network = two_room_network(door=True, noise_free=True)
        localiser = WallAwareLocaliser(network, measurement="hybrid", grid_spacing_m=0.4)
        for n in (0, -5):
            with self.subTest(n_particles=n), self.assertRaises(ValueError):
                ParticleTracker(localiser, network, Random(1), n_particles=n)
        for interval in (0.0, -1.0):
            with self.subTest(scan_interval_s=interval), self.assertRaises(ValueError):
                ParticleTracker(localiser, network, Random(1), n_particles=100, scan_interval_s=interval)

    def test_is_an_undelayed_tracker_with_constant_particle_count(self) -> None:
        """Expected: delay_steps == 0; particles has shape (n_particles, >=2) and weights
        (n_particles,) after updates."""
        network = two_room_network(door=True, noise_free=True)
        localiser = WallAwareLocaliser(network, measurement="hybrid", grid_spacing_m=0.2)
        tracker = ParticleTracker(localiser, network, Random(1), n_particles=300)
        self.assertEqual(tracker.delay_steps, 0)
        tracker.reset()
        rng = Random(2)
        for t in range(5):
            tracker.update(scan_at(network, (1.0 + 0.5 * t, 2.0), rng), STAND_STILL)
            particles, weights = particle_state(tracker)
            self.assertEqual(particles.ndim, 2)
            self.assertEqual(particles.shape[0], 300)
            self.assertGreaterEqual(particles.shape[1], 2)
            self.assertEqual(weights.shape, (300,))

    def test_first_update_initialises_from_global_wifi_posterior(self) -> None:
        """Expected: after reset(), one noise-free scan anywhere on the deck puts the estimate in
        the right compartment within 1 m (initialisation from the whole Wi-Fi posterior, not a
        local window around a previous estimate)."""
        network = deck_network(noise_free=True)
        localiser = WallAwareLocaliser(network, measurement="hybrid", grid_spacing_m=0.2, robust=True)
        tracker = ParticleTracker(localiser, network, Random(3), n_particles=1000)
        env = network.environment
        for xy in ((20.0, 8.0), (2.0, 1.5), (12.0, 5.0), (6.0, 8.5), (2.0, 1.5)):
            tracker.reset()
            estimate = tracker.update(scan_at(network, xy, Random(4)), STAND_STILL)
            self.assertIsInstance(estimate, LocalisationEstimate)
            self.assertLess(math.dist(estimate.position, xy), 1.0, (xy, estimate.position))
            self.assertEqual(env.compartment_at(estimate.position), env.compartment_at(xy))

    def test_first_scan_is_counted_once(self) -> None:
        """Expected: the first update initialises from the first scan's Wi-Fi posterior and must
        not weight by that scan again. With many particles and the likelihood untempered, the
        estimate's covariance after one update matches the single-likelihood posterior over the
        localiser grid in the estimate's compartment (plus within-cell spread and the tracker's
        cell-quantisation floor, each spacing^2/12 per axis) within 10% in trace. Counting the
        scan twice (a likelihood^2 belief) shrinks it by roughly 40%."""
        deck = _Deck.get()
        network, localiser = deck["network"], deck["localiser"]
        env = network.environment
        spacing = 0.2
        rng = Random(41)
        tracker = ParticleTracker(localiser, network, Random(42), n_particles=50000,
                                  likelihood_exponent=1.0, no_imu_likelihood_exponent=1.0)
        for xy in ((3.0, 8.0), (13.0, 2.0), (20.0, 5.0)):
            network.reset_channel_state()
            scan = scan_at(network, xy, rng)
            tracker.reset()
            estimate = tracker.update(scan, None)
            room = env.compartment_at(estimate.position)
            log_lik = localiser.log_likelihood(scan)
            in_room = localiser.grid_compartments == room
            w = np.exp(log_lik - log_lik[in_room].max()) * in_room
            w /= w.sum()
            mean = w @ localiser.grid_points
            offsets = localiser.grid_points - mean
            expected = (w[:, None] * offsets).T @ offsets + 2 * spacing ** 2 / 12 * np.eye(2)
            actual = np.asarray(estimate.covariance, dtype=float)
            self.assertLess(abs(np.trace(actual) / np.trace(expected) - 1.0), 0.10,
                            (xy, room, np.trace(actual), np.trace(expected)))

    def test_estimates_are_finite_inside_the_deck_with_valid_covariance(self) -> None:
        """Expected: on a realistic walk every update returns an estimate inside a compartment whose
        covariance is a finite, symmetric, positive semi-definite 2x2 matrix with positive trace."""
        deck = _Deck.get()
        network, walk = deck["network"], deck["walks"][0][:60]
        tracker = ParticleTracker(deck["localiser"], network, Random(5), n_particles=1000)
        motions = simulate_imu(walk, Random(TUNING_IMU_SEED), ImuModel())
        rng = Random(TUNING_SCAN_SEED)
        network.reset_channel_state()
        tracker.reset()
        for t, xy in enumerate(walk):
            estimate = tracker.update(scan_at(network, xy, rng), motions[t])
            self.assertIsNotNone(estimate)
            self.assertTrue(np.all(np.isfinite(estimate.position)), t)
            self.assertIsNotNone(network.environment.compartment_at(estimate.position), (t, estimate.position))
            cov = np.asarray(estimate.covariance, dtype=float)
            self.assertEqual(cov.shape, (2, 2))
            self.assertTrue(np.all(np.isfinite(cov)), t)
            np.testing.assert_allclose(cov, cov.T, atol=1e-9)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(cov).min()), -1e-9)
            self.assertGreater(float(np.trace(cov)), 0.0)

    def test_handles_no_scan_at_the_first_epoch(self) -> None:
        """Expected: update(None, None) straight after reset() (no Wi-Fi, no IMU yet) returns a
        finite estimate inside the deck instead of raising."""
        deck = _Deck.get()
        tracker = ParticleTracker(deck["localiser"], deck["network"], Random(6), n_particles=500)
        tracker.reset()
        estimate = tracker.update(None, None)
        self.assertIsNotNone(estimate)
        self.assertTrue(np.all(np.isfinite(estimate.position)))
        self.assertIsNotNone(deck["network"].environment.compartment_at(estimate.position))


class ParticleInvariantTests(unittest.TestCase):
    """V3: weights finite and normalised, no particle move crosses a wall."""

    def assert_weights_valid(self, tracker, n_particles: int, context) -> tuple:
        particles, weights = particle_state(tracker)
        self.assertEqual(particles.shape[0], n_particles, context)
        self.assertEqual(weights.shape, (n_particles,), context)
        self.assertTrue(np.all(np.isfinite(weights)), context)
        self.assertTrue(np.all(np.isfinite(particles[:, :2])), context)
        self.assertGreaterEqual(float(weights.min()), 0.0, context)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=9, msg=context)
        return particles, weights

    def test_weights_finite_normalised_and_particles_stay_on_deck(self) -> None:
        """Expected: on a realistic deck walk that includes a Wi-Fi outage (scan None), epochs
        without IMU (motion None) and all-APs-missing scans, after every update the weights are
        finite, non-negative and sum to 1, and every live particle (weight > 0) is inside the hull
        (leaving it would mean crossing a hull wall)."""
        deck = _Deck.get()
        network, walk = deck["network"], deck["walks"][1][:80]
        n = 1000
        tracker = ParticleTracker(deck["localiser"], network, Random(7), n_particles=n)
        motions = simulate_imu(walk, Random(TUNING_IMU_SEED), ImuModel())
        rng = Random(TUNING_SCAN_SEED)
        x0, y0, x1, y1 = DECK_BOUNDS
        network.reset_channel_state()
        tracker.reset()
        for t, xy in enumerate(walk):
            if 30 <= t < 45:
                scan = None
            elif 60 <= t < 65:
                scan = all_missing(network)
            else:
                scan = scan_at(network, xy, rng)
            motion = None if 50 <= t < 55 else motions[t]
            tracker.update(scan, motion)
            particles, weights = self.assert_weights_valid(tracker, n, t)
            live = particles[weights > 0, :2]
            outside = (live[:, 0] < x0) | (live[:, 0] > x1) | (live[:, 1] < y0) | (live[:, 1] > y1)
            self.assertEqual(int(outside.sum()), 0, f"epoch {t}: {int(outside.sum())} live particles off the deck")

    def test_particles_cannot_cross_a_solid_wall(self) -> None:
        """Expected: two rooms sealed by a solid bulkhead at x = 4. The crew member walks east to
        0.6 m from it and stands there while the PDR keeps reporting 1.4 m/s eastwards (false
        steps). Wi-Fi (noise-free scans, Gaussian model) puts essentially zero posterior mass in
        room B, so any live particle in B could only have got there by moving through the wall.
        At every epoch: weights valid, no live particle has x > 4, the estimate is in room A; at
        the end the estimate is within 1 m of the true position (3.4, 2.0)."""
        truth = two_room_network(door=False, noise_free=True)
        model = two_room_network(door=False, noise_free=False)
        localiser = WallAwareLocaliser(model, measurement="hybrid", grid_spacing_m=0.2)
        n = 500
        tracker = ParticleTracker(localiser, model, Random(8), n_particles=n, imu_model=ImuModel())
        positions = [(0.6 + 0.7 * min(t, 4), 2.0) for t in range(20)]
        motions = [STAND_STILL] + [MotionReading(1 if t <= 4 else 2, 0.7, 0.0) for t in range(1, 20)]
        rng = Random(9)
        tracker.reset()
        estimate = None
        for t, xy in enumerate(positions):
            estimate = tracker.update(scan_at(truth, xy, rng), motions[t])
            particles, weights = self.assert_weights_valid(tracker, n, t)
            in_b = int(np.sum((weights > 0) & (particles[:, 0] > 4.0 + 1e-9)))
            self.assertEqual(in_b, 0, f"epoch {t}: {in_b} live particles crossed the bulkhead into room B")
            self.assertEqual(model.environment.compartment_at(estimate.position), "A", (t, estimate.position))
        self.assertLess(math.dist(estimate.position, positions[-1]), 1.0, estimate.position)

    def test_particles_pass_through_a_doorway(self) -> None:
        """Expected: same two rooms with a doorway at y 1.5-2.5. The crew member walks east along
        y = 2 through the doorway into room B and stops at (6.9, 2.0); with noise-free scans the
        final estimate is in room B within 1 m (the wall constraint must not block doorways)."""
        truth = two_room_network(door=True, noise_free=True)
        model = two_room_network(door=True, noise_free=False)
        localiser = WallAwareLocaliser(model, measurement="hybrid", grid_spacing_m=0.2)
        n = 500
        tracker = ParticleTracker(localiser, model, Random(10), n_particles=n, imu_model=ImuModel())
        positions = [(0.6 + 0.7 * min(t, 9), 2.0) for t in range(15)]
        motions = [STAND_STILL] + [MotionReading(1, 0.7, 0.0) if t <= 9 else STAND_STILL for t in range(1, 15)]
        rng = Random(11)
        tracker.reset()
        for t, xy in enumerate(positions):
            estimate = tracker.update(scan_at(truth, xy, rng), motions[t])
            self.assert_weights_valid(tracker, n, t)
        self.assertEqual(model.environment.compartment_at(estimate.position), "B", estimate.position)
        self.assertLess(math.dist(estimate.position, positions[-1]), 1.0, estimate.position)


    def test_every_accepted_move_has_a_wall_free_route(self) -> None:
        """Expected: on a realistic deck walk (including epochs without IMU), every particle move
        made in the predict step that leaves the particle alive has a wall-free walking route:
        either the straight segment, or two straight legs via a door jamb (a wall end touching no
        other wall, nudged 5 cm into the doorway gap). So a particle only ever changes
        compartment through a doorway, even when its chord clips a jamb."""
        deck = _Deck.get()
        network, walk = deck["network"], deck["walks"][2][:120]
        env = network.environment
        jambs = []
        for i, wall in enumerate(env.walls):
            p1, p2 = np.array(wall.p1, dtype=float), np.array(wall.p2, dtype=float)
            for end, other in ((p1, p2), (p2, p1)):
                touching = any(j != i and env.walls[j].geometry.distance(ShapelyPoint(end)) < 1e-6
                               for j in range(len(env.walls)))
                if not touching:
                    jambs.append(end + 0.05 * (end - other) / np.linalg.norm(end - other))
        jambs = np.array(jambs)
        self.assertGreater(len(jambs), 0)

        tracker = ParticleTracker(deck["localiser"], network, Random(31), n_particles=1000)
        moves = []
        predict = tracker._predict

        def recording_predict(motion):
            before = tracker.particles[:, :2].copy()
            predict(motion)
            alive = tracker.weights > 0
            moves.append((before[alive], tracker.particles[alive, :2].copy()))

        tracker._predict = recording_predict
        motions = simulate_imu(walk, Random(TUNING_IMU_SEED), ImuModel())
        rng = Random(TUNING_SCAN_SEED)
        network.reset_channel_state()
        tracker.reset()
        for t, xy in enumerate(walk):
            tracker.update(scan_at(network, xy, rng), None if 60 <= t < 70 else motions[t])

        a = np.concatenate([m[0] for m in moves])
        b = np.concatenate([m[1] for m in moves])
        straight_blocked = env.crossing_matrix(a, b).any(axis=1)
        violations = 0
        for start, end in zip(a[straight_blocked], b[straight_blocked]):
            starts = np.repeat(start[None, :], len(jambs), axis=0)
            ends = np.repeat(end[None, :], len(jambs), axis=0)
            via_ok = ~(env.crossing_matrix(starts, jambs).any(axis=1) | env.crossing_matrix(jambs, ends).any(axis=1))
            violations += not via_ok.any()
        self.assertGreater(len(a), 100000)
        self.assertEqual(violations, 0, f"{violations} of {len(a)} accepted moves have no wall-free route")

    def test_uses_no_global_random_state(self) -> None:
        """Expected: ParticleTracker draws all randomness from the Random it is given (a numpy
        Generator seeded from it is allowed): with the global `random` functions and numpy's legacy
        global RNG patched to raise, and numpy generators required to be seeded, a deck walk with
        IMU, without IMU, a Wi-Fi outage and all-missing scans runs normally."""
        deck = _Deck.get()
        network, walk = deck["network"], deck["walks"][3][:50]
        motions = simulate_imu(walk, Random(TUNING_IMU_SEED), ImuModel())
        scans = []
        rng = Random(TUNING_SCAN_SEED)
        network.reset_channel_state()
        for t, xy in enumerate(walk):
            scans.append(None if 20 <= t < 25 else all_missing(network) if 30 <= t < 33 else scan_at(network, xy, rng))

        def forbidden(*args, **kwargs):
            raise AssertionError("global random state used")

        real_default_rng = np.random.default_rng

        def seeded_default_rng(seed=None, *args, **kwargs):
            if seed is None:
                raise AssertionError("unseeded numpy Generator")
            return real_default_rng(seed, *args, **kwargs)

        patches = [mock.patch.object(random, name, forbidden) for name in
                   ("random", "uniform", "gauss", "normalvariate", "randint", "randrange", "choice", "choices",
                    "shuffle", "sample", "getrandbits", "seed", "expovariate")]
        patches += [mock.patch.object(np.random, name, forbidden) for name in
                    ("random", "rand", "randn", "normal", "uniform", "choice", "randint", "shuffle",
                     "permutation", "seed", "standard_normal", "random_sample")]
        patches.append(mock.patch.object(np.random, "default_rng", seeded_default_rng))
        for patch in patches:
            patch.start()
        try:
            tracker = ParticleTracker(deck["localiser"], network, Random(32), n_particles=500)
            tracker.reset()
            for t, scan in enumerate(scans):
                estimate = tracker.update(scan, None if 40 <= t < 45 else motions[t])
                self.assertTrue(np.all(np.isfinite(estimate.position)))
        finally:
            for patch in patches:
                patch.stop()


class ParticleIdealCaseTests(unittest.TestCase):
    """V5: null and ideal cases (tuning seeds)."""

    def test_perfect_imu_with_wifi_gives_small_error_after_convergence(self) -> None:
        """Expected: with a perfect IMU (every ImuModel error off, and the PF told so) plus the
        realistic deck Wi-Fi, the PF's mean error over epochs >= 30 of 4 walks x 150 s is < 0.3 m."""
        deck = _Deck.get()
        network, walks = deck["network"], deck["walks"]
        model = perfect_imu_model()
        imu_rng = Random(TUNING_IMU_SEED)
        motions = [simulate_imu(walk, imu_rng, model) for walk in walks]
        tracker = ParticleTracker(deck["localiser"], network, Random(12), n_particles=2000, imu_model=model)
        stats, estimates = evaluate_tracking({"PF perfect IMU": tracker}, network, walks, Random(TUNING_SCAN_SEED), motions=motions)
        errors = [math.dist(est, true) for walk, ests in zip(walks, estimates["PF perfect IMU"])
                  for t, (true, est) in enumerate(zip(walk, ests)) if t >= 30 and est is not None]
        self.assertEqual(len(errors), sum(len(w) - 30 for w in walks))
        self.assertLess(float(np.mean(errors)), 0.3, f"mean error after 30 s {np.mean(errors):.3f} m; whole walk {fmt(stats)}")

    def test_all_aps_missing_scans_keep_belief_within_reach(self) -> None:
        """Expected: when, for 15 s, every scan reports no AP at all (non-detection evidence only,
        which favours far-away rooms), the PF does not teleport: each estimate stays inside a
        compartment and within 3 m of the truth, and live particles stay on the deck."""
        deck = _Deck.get()
        network = deck["network"]
        x0, y0, x1, y1 = DECK_BOUNDS
        for w, walk in enumerate(deck["walks"][:2]):
            walk = walk[:80]
            motions = simulate_imu(walk, Random(TUNING_IMU_SEED + w), ImuModel())
            tracker = ParticleTracker(deck["localiser"], network, Random(13), n_particles=1000)
            rng = Random(TUNING_SCAN_SEED + w)
            network.reset_channel_state()
            tracker.reset()
            for t, xy in enumerate(walk):
                missing = 40 <= t < 55
                estimate = tracker.update(all_missing(network) if missing else scan_at(network, xy, rng), motions[t])
                if not missing:
                    continue
                self.assertIsNotNone(estimate)
                self.assertIsNotNone(network.environment.compartment_at(estimate.position), (w, t, estimate.position))
                error = math.dist(estimate.position, xy)
                self.assertLess(error, 3.0, f"walk {w} epoch {t}: error {error:.2f} m during all-missing scans")
                particles, weights = particle_state(tracker)
                live = particles[weights > 0, :2]
                self.assertTrue(np.all((live[:, 0] >= x0) & (live[:, 0] <= x1) & (live[:, 1] >= y0) & (live[:, 1] <= y1)))

    def test_without_imu_tracks_no_worse_than_1_2_x_grid_tracker(self) -> None:
        """Expected: with motion None at every epoch (no IMU), the PF's mean error on 4 deck walks
        x 150 s is <= 1.2 x the GridTracker's on identical scans."""
        deck = _Deck.get()
        network, walks = deck["network"], deck["walks"]
        trackers = {
            "Grid": GridTracker(deck["localiser"], network),
            "PF no IMU": ParticleTracker(deck["localiser"], network, Random(14), n_particles=2000),
        }
        stats, _ = evaluate_tracking(trackers, network, walks, Random(TUNING_SCAN_SEED), motions=None)
        self.assertLessEqual(stats["PF no IMU"]["mean_err"], 1.2 * stats["Grid"]["mean_err"], fmt(stats))

    def test_recovers_after_being_kidnapped(self) -> None:
        """Expected: a crew member stands in Cabin 1 for 25 s and is then (impossibly) moved to the
        Workshop, with the IMU reporting no steps throughout. Injection from the Wi-Fi posterior
        must let the PF recover: over the last 10 of 30 epochs in the Workshop every estimate is in
        the Workshop and within 1.5 m."""
        deck = _Deck.get()
        network = deck["network"]
        tracker = ParticleTracker(deck["localiser"], network, Random(15), n_particles=1000)
        rng = Random(16)
        network.reset_channel_state()
        tracker.reset()
        for _ in range(25):
            tracker.update(scan_at(network, (2.0, 2.0), rng), STAND_STILL)
        workshop = (21.0, 8.0)
        for t in range(30):
            estimate = tracker.update(scan_at(network, workshop, rng), STAND_STILL)
            if t >= 20:
                self.assertEqual(network.environment.compartment_at(estimate.position), "Workshop", (t, estimate.position))
                self.assertLess(math.dist(estimate.position, workshop), 1.5, (t, estimate.position))


class ParticleDeterminismTests(unittest.TestCase):
    """V6: paired evaluation and reproducibility."""

    @classmethod
    def setUpClass(cls) -> None:
        deck = _Deck.get()
        cls.network, cls.localiser = deck["network"], deck["localiser"]
        cls.walks = [walk[:60] for walk in deck["walks"][:2]]
        imu_rng = Random(TUNING_IMU_SEED)
        cls.motions = [simulate_imu(walk, imu_rng, ImuModel()) for walk in cls.walks]

    def make(self, seed: int) -> ParticleTracker:
        return ParticleTracker(self.localiser, self.network, Random(seed), n_particles=500, imu_model=None)

    def test_identical_trackers_with_equal_seeds_give_identical_results(self) -> None:
        """Expected: two PFs built with equal seeds, fed identical scans and IMU readings by
        evaluate_tracking, give identical stats and estimates; a PF with another seed gives
        different estimates (the rng is really used)."""
        trackers = {"PF A": self.make(21), "PF B": self.make(21), "PF C": self.make(22)}
        stats, estimates = evaluate_tracking(trackers, self.network, self.walks, Random(TUNING_SCAN_SEED), motions=self.motions)
        self.assertEqual(stats["PF A"], stats["PF B"])
        self.assertEqual(estimates["PF A"], estimates["PF B"])
        self.assertNotEqual(estimates["PF A"], estimates["PF C"])

    def test_reset_makes_results_independent_of_walk_order(self) -> None:
        """Expected: after reset(), a PF gives exactly the same estimates for a walk whether or not
        it ran another walk before, and re-running a walk after reset() repeats it exactly."""
        scans = []
        for w, walk in enumerate(self.walks):
            self.network.reset_channel_state()
            rng = Random(100 + w)
            scans.append([scan_at(self.network, xy, rng) for xy in walk])

        def run(tracker, w):
            tracker.reset()
            return [tracker.update(scan, motion).position for scan, motion in zip(scans[w], self.motions[w])]

        used = self.make(23)
        first_a = run(used, 0)
        b_after_a = run(used, 1)
        a_again = run(used, 0)
        b_fresh = run(self.make(23), 1)
        self.assertEqual(b_after_a, b_fresh)
        self.assertEqual(a_again, first_a)


if __name__ == "__main__":
    unittest.main()
