"""Extended tests for WallAwareLocaliser: geometry, grid, likelihood and offset handling."""
import math
import unittest
from dataclasses import replace
from pathlib import Path
from random import Random

import numpy as np

from environment import Compartment, ShipEnvironment, Wall
from network import AccessPoint, Network, Receiver
from positioning import WallAwareLocaliser
from tracking import _segments_cross_walls

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

DECK_APS = [
    AccessPoint("Corridor-W", "30:00:00:00:00:01", (3.0, 5.0, 2.5), 2412),
    AccessPoint("Corridor-MW", "30:00:00:00:00:02", (9.0, 5.0, 2.5), 2437),
    AccessPoint("Corridor-ME", "30:00:00:00:00:03", (15.0, 5.0, 2.5), 2462),
    AccessPoint("Corridor-E", "30:00:00:00:00:04", (21.0, 5.0, 2.5), 2412),
    AccessPoint("Mess", "30:00:00:00:00:05", (12.0, 2.0, 2.5), 2437),
    AccessPoint("ECR", "30:00:00:00:00:06", (15.0, 8.0, 2.5), 2462),
]


def two_room_environment(door: bool = True) -> ShipEnvironment:
    if door:
        divider = [
            Wall("Divider low", (4.0, 0.0), (4.0, 1.5), attenuation_db=20.0),
            Wall("Divider high", (4.0, 2.5), (4.0, 4.0), attenuation_db=20.0),
        ]
    else:
        divider = [Wall("Divider", (4.0, 0.0), (4.0, 4.0), attenuation_db=20.0)]
    return ShipEnvironment(
        compartments=[
            Compartment("A", [(0, 0), (4, 0), (4, 4), (0, 4)]),
            Compartment("B", [(4, 0), (8, 0), (8, 4), (4, 4)]),
        ],
        walls=divider,
    )


def make_network(environment: ShipEnvironment, access_points=None, **overrides) -> Network:
    parameters = {"environment": environment, "access_points": list(access_points or TWO_ROOM_APS), **NOISE_FREE}
    parameters.update(overrides)
    return Network(**parameters)


def deck_network(noise_free: bool = True, **overrides) -> Network:
    base = Network.from_config(LAYOUTS / "accommodation_deck.yaml", access_points=list(DECK_APS))
    changes = dict(NOISE_FREE) if noise_free else {}
    changes.update(overrides)
    return replace(base, **changes)


def _shapely_counts(env: ShipEnvironment, starts, ends) -> np.ndarray:
    return np.array([[len(env.walls_crossed(tuple(s), tuple(e))) for e in ends] for s in starts], dtype=float)


class WallGeometryEquivalenceTests(unittest.TestCase):
    """The vectorised crossing tests must agree with ShipEnvironment.walls_crossed (shapely)."""

    def _check(self, network: Network, starts: np.ndarray, ends: np.ndarray) -> None:
        env = network.environment
        count, loss = WallAwareLocaliser._wall_terms(network, starts, ends)
        expected = _shapely_counts(env, starts, ends)
        mismatches = np.argwhere(count != expected)
        self.assertEqual(
            len(mismatches), 0,
            f"{len(mismatches)} mismatches, e.g. {[(tuple(starts[i]), tuple(ends[j])) for i, j in mismatches[:3]]}",
        )
        expected_loss = np.array(
            [[sum(w.attenuation_db for w in env.walls_crossed(tuple(s), tuple(e))) for e in ends] for s in starts]
        )
        np.testing.assert_allclose(loss, expected_loss)

        # tracking._segments_cross_walls on the same segments (flattened pairwise)
        a = np.repeat(starts, len(ends), axis=0)
        b = np.tile(ends, (len(starts), 1))
        blocked = _segments_cross_walls(network, a, b)
        np.testing.assert_array_equal(blocked, expected.ravel() > 0)

    def test_random_continuous_segments_on_deck(self) -> None:
        network = deck_network()
        rng = Random(11)
        starts = np.array([(rng.uniform(-1, 25), rng.uniform(-1, 11)) for _ in range(35)])
        ends = np.array([(rng.uniform(-1, 25), rng.uniform(-1, 11)) for _ in range(35)])
        self._check(network, starts, ends)

    def test_lattice_segments_hit_degenerate_cases_on_deck(self) -> None:
        # Half-metre lattice: many segments end on walls, pass through wall endpoints
        # / door-gap edges, or run collinear with a wall.
        network = deck_network()
        rng = Random(12)
        lattice = lambda: (rng.randint(0, 48) * 0.5, rng.randint(0, 20) * 0.5)  # noqa: E731
        starts = np.array([lattice() for _ in range(35)])
        ends = np.array([lattice() for _ in range(35)])
        self._check(network, starts, ends)

    def test_named_edge_cases_on_two_room_door_layout(self) -> None:
        network = make_network(two_room_environment(door=True))
        cases = [
            ((3.0, 1.0), (5.0, 1.0), 1),   # clean crossing of the low divider
            ((3.0, 2.0), (5.0, 2.0), 0),   # straight through the doorway
            ((1.0, 1.0), (4.0, 1.0), 0),   # ends exactly on the wall's interior
            ((4.0, 1.0), (6.0, 1.0), 0),   # starts exactly on the wall
            ((3.0, 1.0), (5.0, 2.0), 0),   # passes exactly through the door-gap edge (4, 1.5)
            ((4.0, -1.0), (4.0, 1.0), 0),  # collinear, overlapping the wall
            ((4.0, 0.5), (4.0, 3.0), 0),   # collinear through the door gap
            ((3.0, 0.5), (5.0, 3.5), 0),   # diagonal through the gap (x=4 at y=2)
            ((3.0, 0.0), (5.0, 4.0), 0),   # diagonal through the gap centre
            ((3.0, 3.0), (5.0, 3.5), 1),   # crosses the high divider
            ((3.5, 0.5), (4.5, 3.5), 0),   # x=4 at y=2: gap
            ((2.0, 0.2), (6.0, 3.8), 0),   # x=4 at y=2.0: gap centre
            ((2.0, 0.0), (6.0, 4.0), 0),   # x=4 at y=2
            ((2.0, 3.9), (6.0, 0.1), 0),   # x=4 at y=2
            ((3.9, 0.1), (4.1, 3.9), 0),   # steep, x=4 at y=2.0: gap
            ((3.9, 0.1), (4.1, 0.3), 1),   # short hop across the low divider
            ((4.0, 1.5), (4.0, 2.5), 0),   # exactly spans the door gap, endpoint to endpoint
        ]
        env = network.environment
        for p, q, _ in cases:
            shapely_count = len(env.walls_crossed(p, q))
            count, _ = WallAwareLocaliser._wall_terms(network, np.array([p]), np.array([q]))
            self.assertEqual(count[0, 0], shapely_count, (p, q))
            blocked = _segments_cross_walls(network, np.array([p]), np.array([q]))
            self.assertEqual(bool(blocked[0]), shapely_count > 0, (p, q))
        for p, q, expected in cases:
            self.assertEqual(len(env.walls_crossed(p, q)), expected, (p, q))

    def test_ray_through_t_junction_counts_only_the_wall_it_crosses(self) -> None:
        # (4, 4) on the deck is interior to "S Corridor 2" and the end of "Cabin 1-2 Panel".
        network = deck_network()
        p, q = np.array([[3.0, 3.0]]), np.array([[5.0, 5.0]])
        count, loss = WallAwareLocaliser._wall_terms(network, p, q)
        crossed = network.environment.walls_crossed((3.0, 3.0), (5.0, 5.0))
        self.assertEqual([w.name for w in crossed], ["S Corridor 2"])
        self.assertEqual(count[0, 0], 1)
        self.assertAlmostEqual(loss[0, 0], 20.0)


class GridTests(unittest.TestCase):
    def test_deck_grid_matches_compartment_at_and_spacing(self) -> None:
        network = deck_network()
        for spacing in (0.2, 0.3, 0.7):
            points, names = WallAwareLocaliser._build_grid(network, spacing)
            self.assertGreater(len(points), 0)
            for (x, y), name in zip(points, names):
                self.assertIsNotNone(name)
                self.assertEqual(network.environment.compartment_at((x, y)), name)
            # nearest-neighbour distance is exactly the spacing
            xs = np.unique(np.round(points[:, 0], 9))
            ys = np.unique(np.round(points[:, 1], 9))
            np.testing.assert_allclose(np.diff(xs), spacing, atol=1e-9)
            np.testing.assert_allclose(np.diff(ys), spacing, atol=1e-9)
            # the deck is fully covered by compartments, so every cell centre is kept
            self.assertEqual(len(points), len(xs) * len(ys))
            self.assertTrue(np.all(points[:, 0] > 0) and np.all(points[:, 0] < 24))
            self.assertTrue(np.all(points[:, 1] > 0) and np.all(points[:, 1] < 10))

    def test_grid_skips_points_outside_compartments(self) -> None:
        # Two rooms with a 2 m gap between them that no compartment covers
        env = ShipEnvironment(compartments=[
            Compartment("West", [(0, 0), (3, 0), (3, 3), (0, 3)]),
            Compartment("East", [(5, 0), (8, 0), (8, 3), (5, 3)]),
        ])
        points, names = WallAwareLocaliser._build_grid(make_network(env), 0.25)
        self.assertFalse(np.any((points[:, 0] > 3) & (points[:, 0] < 5)))
        for (x, y), name in zip(points, names):
            self.assertEqual(env.compartment_at((x, y)), name)
        self.assertEqual(set(names), {"West", "East"})

    def test_no_compartments_raises(self) -> None:
        with self.assertRaises(ValueError):
            WallAwareLocaliser(make_network(ShipEnvironment()))


class LikelihoodTests(unittest.TestCase):
    def test_invalid_parameters_raise_value_error(self) -> None:
        network = make_network(two_room_environment())
        bad = [
            {"measurement": "sonar"},
            {"grid_spacing_m": 0.0},
            {"grid_spacing_m": -0.1},
            {"rssi_sigma_db": 0.0},
            {"rssi_sigma_db": -1.0},
            {"device_offset_sigma_db": 0.0},
            {"device_offset_sigma_db": -2.0},
            {"unexplained_miss_probability": 0.0},
            {"unexplained_miss_probability": 1.0},
            {"unexplained_miss_probability": 1.5},
        ]
        for kwargs in bad:
            with self.assertRaises(ValueError, msg=str(kwargs)):
                WallAwareLocaliser(network, **kwargs)

    def test_noise_free_argmax_is_the_true_grid_point(self) -> None:
        network = make_network(two_room_environment(door=True))
        rng = Random(5)
        for measurement in WallAwareLocaliser.MEASUREMENTS:
            localiser = WallAwareLocaliser(network, measurement=measurement, grid_spacing_m=0.25, rssi_sigma_db=1.0)
            for index in range(0, len(localiser.grid_points), 37):
                x, y = localiser.grid_points[index]
                log_lik = localiser.log_likelihood(network.scan(Receiver("p", (x, y, 1.0)), rng))
                self.assertEqual(int(np.argmax(log_lik)), index, (measurement, (x, y)))

    def test_rss_likelihood_falls_off_along_rays_from_the_truth(self) -> None:
        # Noise-free open room, 3 well-spread APs: the log-likelihood is non-increasing along each
        # of 8 grid rays out of the true point for the first metre.
        env = ShipEnvironment(compartments=[Compartment("Room", [(0, 0), (8, 0), (8, 8), (0, 8)])])
        aps = [
            AccessPoint("a", "00:00:00:00:01:01", (1.0, 1.0, 1.0), 2412),
            AccessPoint("b", "00:00:00:00:01:02", (7.0, 1.0, 1.0), 2412),
            AccessPoint("c", "00:00:00:00:01:03", (4.0, 7.0, 1.0), 2412),
        ]
        network = make_network(env, access_points=aps)
        localiser = WallAwareLocaliser(network, grid_spacing_m=0.25, rssi_sigma_db=2.0, use_detections=False)
        index = {(round(x, 6), round(y, 6)): i for i, (x, y) in enumerate(localiser.grid_points)}
        true_xy = (4.125, 3.625)
        log_lik = localiser.log_likelihood(network.scan(Receiver("p", (*true_xy, 1.0)), Random(0)))
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            values = [log_lik[index[(round(true_xy[0] + k * 0.25 * dx, 6), round(true_xy[1] + k * 0.25 * dy, 6))]]
                      for k in range(5)]
            self.assertTrue(all(b < a for a, b in zip(values, values[1:])), (dx, dy, values))

    def test_missed_ap_penalises_loud_points_but_stays_finite(self) -> None:
        # Realistic detection threshold (~-90 dBm) so a miss is informative.
        network = deck_network(noise_free=True, detection_snr_midpoint_db=6.0)
        mess_ap = next(ap for ap in DECK_APS if ap.name == "Mess")
        without_mess = replace(network, access_points=[ap for ap in DECK_APS if ap is not mess_ap])
        localiser = WallAwareLocaliser(network, grid_spacing_m=0.3)
        reference = WallAwareLocaliser(without_mess, grid_spacing_m=0.3)  # same grid, no Mess AP at all

        scan = network.scan(Receiver("p", (12.0, 8.5, 1.0)), Random(1))
        scan[mess_ap] = None  # the Mess AP was not heard
        log_lik = localiser.log_likelihood(scan)
        self.assertTrue(np.all(np.isfinite(log_lik)))

        miss_term = log_lik - reference.log_likelihood(scan)
        self.assertTrue(np.all(np.isfinite(miss_term)))
        p_unexplained = localiser.unexplained_miss_probability
        # P(miss) is between the unexplained-miss floor and 1 at every cell
        self.assertGreaterEqual(miss_term.min(), math.log(p_unexplained) - 1e-9)
        self.assertLessEqual(miss_term.max(), 1e-12)
        names = localiser.grid_compartments
        # loud next to the AP (Mess) -> close to the floor; far away behind bulkheads -> ~no penalty
        self.assertLess(miss_term[names == "Mess"].max(), math.log(2 * p_unexplained))
        self.assertGreater(np.median(miss_term[names == "Workshop"]), math.log(0.5))

    def test_all_terms_finite_far_below_detection_threshold(self) -> None:
        # Heard AP whose expected RSSI is far below threshold at most cells: log_ndtr must not
        # underflow to -inf.
        network = deck_network(noise_free=True, detection_snr_midpoint_db=6.0)
        localiser = WallAwareLocaliser(network, grid_spacing_m=0.3, rssi_sigma_db=0.5)
        scan = {ap: None for ap in DECK_APS}
        scan[DECK_APS[0]] = network.reading(DECK_APS[0], Receiver("p", (3.0, 5.5, 1.0)), Random(0))
        self.assertTrue(np.all(np.isfinite(localiser.log_likelihood(scan))))

    def test_locate_returns_none_when_nothing_heard_in_every_mode(self) -> None:
        network = deck_network()
        for measurement in WallAwareLocaliser.MEASUREMENTS:
            for use_detections in (True, False):
                loc = WallAwareLocaliser(network, measurement=measurement, grid_spacing_m=0.5,
                                         use_detections=use_detections)
                self.assertIsNone(loc.locate({ap: None for ap in DECK_APS}))
                self.assertIsNone(loc.locate({}))

    def test_estimates_inside_a_compartment_with_symmetric_psd_covariance(self) -> None:
        network = deck_network(noise_free=False)
        rng = Random(7)
        localisers = [
            WallAwareLocaliser(network, measurement=m, grid_spacing_m=0.3, robust=r)
            for m in WallAwareLocaliser.MEASUREMENTS for r in (False, True)
        ]
        for _ in range(15):
            room = rng.choice([c.name for c in network.environment.compartments])
            xy = network.environment.random_point_in(room, rng)
            scan = network.scan(Receiver("p", (*xy, 1.0)), rng)
            for loc in localisers:
                est = loc.locate(scan)
                if est is None:
                    continue
                self.assertIsNotNone(network.environment.compartment_at(est.position))
                cov = est.covariance
                self.assertEqual(cov.shape, (2, 2))
                np.testing.assert_allclose(cov, cov.T, atol=1e-12)
                self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), -1e-12)
                self.assertEqual(est.num_aps_used, sum(1 for r in scan.values() if r is not None))

    def test_expected_rssi_matches_mean_rssi_on_deck(self) -> None:
        network = deck_network()
        localiser = WallAwareLocaliser(network, grid_spacing_m=0.5)
        for i in range(0, len(localiser.grid_points), 23):
            x, y = localiser.grid_points[i]
            receiver = Receiver("probe", (x, y, 1.0))
            for j, ap in enumerate(DECK_APS):
                self.assertAlmostEqual(localiser._expected_rssi[i, j], network.mean_rssi(ap, receiver), places=9)

    def test_estimate_stays_inside_a_non_convex_compartment(self) -> None:
        # Regression: for an L-shaped room with one AP at its corner, the ring-shaped posterior
        # covers both arms and its mean used to land in the notch, outside every compartment.
        env = ShipEnvironment(compartments=[Compartment("L", [(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)])])
        ap = AccessPoint("AP", "00:00:00:00:00:01", (0.0, 0.0, 1.0), 2412)
        network = make_network(env, access_points=[ap])
        localiser = WallAwareLocaliser(network, grid_spacing_m=0.1, rssi_sigma_db=1.0)
        est = localiser.locate(network.scan(Receiver("p", (3.9, 0.9, 1.0)), Random(0)))
        self.assertEqual(env.compartment_at(est.position), "L")


class DetectionThresholdTests(unittest.TestCase):
    def test_threshold_is_the_fifty_percent_point_for_default_parameters(self) -> None:
        for beacons in (1, 3):
            network = deck_network(noise_free=False, beacons_per_scan=beacons)
            threshold = WallAwareLocaliser._detection_threshold(network)
            p = network.detection_probability(threshold, network.nominal_noise_floor_dbm)
            self.assertAlmostEqual(p, 0.5, places=6)

    def test_threshold_is_the_fifty_percent_point_for_extreme_midpoints(self) -> None:
        # Regression: the bisection bracket used to be fixed at floor +/- 30 dB.
        for midpoint in (-100.0, 40.0):
            network = deck_network(noise_free=False, detection_snr_midpoint_db=midpoint)
            threshold = WallAwareLocaliser._detection_threshold(network)
            p = network.detection_probability(threshold, network.nominal_noise_floor_dbm)
            self.assertAlmostEqual(p, 0.5, places=3, msg=f"midpoint {midpoint}")


class DeviceOffsetTests(unittest.TestCase):
    def _errors(self, bias_db: float, seed: int):
        network = deck_network(noise_free=True, measurement_noise_std_db=1.0)
        naive = WallAwareLocaliser(network, grid_spacing_m=0.3)
        aware = WallAwareLocaliser(network, grid_spacing_m=0.3, device_offset_sigma_db=6.0)
        rng = Random(seed)
        naive_errors, aware_errors = [], []
        for room in [c.name for c in network.environment.compartments]:
            for _ in range(4):
                xy = network.environment.random_point_in(room, rng)
                scan = network.scan(Receiver("p", (*xy, 1.0), rssi_bias_db=bias_db), rng)
                naive_errors.append(math.dist(xy, naive.locate(scan).position))
                aware_errors.append(math.dist(xy, aware.locate(scan).position))
        return float(np.mean(naive_errors)), float(np.mean(aware_errors))

    def test_offset_aware_beats_naive_with_constant_phone_bias_on_deck(self) -> None:
        for bias in (-8.0, 8.0):
            naive, aware = self._errors(bias, seed=21)
            self.assertLess(aware, naive, f"bias {bias}: aware {aware:.2f} vs naive {naive:.2f}")

    def test_offset_aware_not_much_worse_without_bias(self) -> None:
        naive, aware = self._errors(0.0, seed=22)
        # Losing the absolute RSSI level costs accuracy when there is no bias (observed ~0.85-0.9 m
        # vs ~0.5-0.6 m on this deck); guard against it becoming dramatically worse.
        self.assertLess(aware, 2.0 * naive, f"aware {aware:.2f} vs naive {naive:.2f}")

    def test_offset_is_exactly_absorbed_with_many_aps_in_noise_free_room(self) -> None:
        # With no noise and a flat prior (large sigma), a constant bias is fully explained away,
        # so the argmax is the same point as for an unbiased phone.
        network = make_network(two_room_environment(door=True))
        loc = WallAwareLocaliser(network, grid_spacing_m=0.25, rssi_sigma_db=1.0,
                                 device_offset_sigma_db=1000.0, use_detections=False)
        rng = Random(0)
        for index in range(0, len(loc.grid_points), 41):
            x, y = loc.grid_points[index]
            scan = network.scan(Receiver("p", (x, y, 1.0), rssi_bias_db=-7.0), rng)
            self.assertEqual(int(np.argmax(loc.log_likelihood(scan))), index)


if __name__ == "__main__":
    unittest.main()
