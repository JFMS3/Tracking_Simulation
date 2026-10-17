"""Tests for tracking.py: GridTracker, simulate_walk and evaluate_tracking."""
import math
import unittest
from dataclasses import replace
from pathlib import Path
from random import Random

import numpy as np

from environment import Compartment, ShipEnvironment, Wall
from network import AccessPoint, Network, Receiver
from positioning import LocalisationEstimate, WallAwareLocaliser
from scipy.sparse.csgraph import dijkstra

from tracking import GridTracker, SnapshotAsTracker, _unblocked_pairs, _walk_graph, evaluate_tracking, simulate_walk

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

DECK_APS = [
    AccessPoint("Corridor-W", "30:00:00:00:00:01", (3.0, 5.0, 2.5), 2412),
    AccessPoint("Corridor-MW", "30:00:00:00:00:02", (9.0, 5.0, 2.5), 2437),
    AccessPoint("Corridor-ME", "30:00:00:00:00:03", (15.0, 5.0, 2.5), 2462),
    AccessPoint("Corridor-E", "30:00:00:00:00:04", (21.0, 5.0, 2.5), 2412),
    AccessPoint("Mess", "30:00:00:00:00:05", (12.0, 2.0, 2.5), 2437),
    AccessPoint("ECR", "30:00:00:00:00:06", (15.0, 8.0, 2.5), 2462),
]

_DECK_CACHE = {}


def deck_network(noise_free: bool = False, **overrides) -> Network:
    """A fresh accommodation-deck network (realistic channel unless noise_free)."""
    base = Network.from_config(LAYOUTS / "accommodation_deck.yaml", access_points=list(DECK_APS))
    changes = dict(NOISE_FREE) if noise_free else {}
    changes.update(overrides)
    return replace(base, **changes)


def mass_in(tracker: GridTracker, room: str) -> float:
    return float(tracker.belief[tracker.localiser.grid_compartments == room].sum())


class PredictMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.network = deck_network(noise_free=True)
        cls.localiser = WallAwareLocaliser(cls.network, grid_spacing_m=0.3, robust=True)
        cls.tracker = GridTracker(cls.localiser, cls.network)

    def test_predict_matrix_is_column_stochastic_and_non_negative(self) -> None:
        P = self.tracker._predict_matrix
        n = len(self.localiser.grid_points)
        self.assertEqual(P.shape, (n, n))
        self.assertGreaterEqual(P.data.min(), 0.0)
        np.testing.assert_allclose(np.asarray(P.sum(axis=0)).ravel(), 1.0, atol=1e-12)

    def test_predict_matrix_only_links_cells_within_walking_reach(self) -> None:
        # The motion kernel is on walking distance (shortest wall-free route on the grid),
        # so every linked pair must be joined by such a route no longer than the kernel reach.
        P = self.tracker._predict_matrix.tocoo()
        points = self.localiser.grid_points
        graph = _walk_graph(self.network, points, 0.3)
        off_diagonal = np.flatnonzero(P.row != P.col)
        rng = Random(4)
        reach = 3.0 * (0.5 * 1.4 * 1.0)
        sample = rng.sample(list(off_diagonal), 400)
        sources = np.array([P.col[k] for k in sample])  # predict matrix is T^T: col = from, row = to
        distances = dijkstra(graph, directed=False, indices=sources)
        for i, k in enumerate(sample):
            self.assertLessEqual(distances[i, P.row[k]], reach + 1e-9, (points[P.col[k]], points[P.row[k]]))

    def test_no_link_between_cells_on_either_side_of_a_solid_bulkhead(self) -> None:
        # Cabin 1 (x<4, y<4) and Cabin 2 (x>4) share only a wall; the route between cells just
        # either side of it goes via both doors and the corridor, beyond one step's reach.
        points = self.localiser.grid_points
        P = self.tracker._predict_matrix.tocsc()
        left = np.flatnonzero((points[:, 0] > 3.6) & (points[:, 0] < 4.0) & (points[:, 1] < 3.0))
        right = np.flatnonzero((points[:, 0] > 4.0) & (points[:, 0] < 4.4) & (points[:, 1] < 3.0))
        self.assertTrue(len(left) and len(right))
        self.assertEqual(P[right][:, left].nnz, 0)

    def test_unblocked_pairs_agree_with_shapely(self) -> None:
        points = self.localiser.grid_points[::3]
        pairs = _unblocked_pairs(self.network, points, radius=1.5)
        kept = {(int(i), int(j)) for i, j in pairs}
        env = self.network.environment
        from scipy.spatial import cKDTree
        all_pairs = cKDTree(points).query_pairs(1.5, output_type="ndarray")
        for i, j in all_pairs[::5]:
            clear = not env.walls_crossed(tuple(points[i]), tuple(points[j]))
            self.assertEqual((int(i), int(j)) in kept, clear, (points[i], points[j]))


class GridTrackerBehaviourTests(unittest.TestCase):
    def test_belief_sums_to_one_after_every_update(self) -> None:
        network = deck_network()
        tracker = GridTracker(WallAwareLocaliser(network, measurement="hybrid", grid_spacing_m=0.3, robust=True), network)
        rng = Random(8)
        walk = simulate_walk(network, Random(9), n_steps=40)
        for xy in walk:
            tracker.update(network.scan(Receiver("p", (*xy, 1.0)), rng))
            self.assertAlmostEqual(float(tracker.belief.sum()), 1.0, places=9)
            self.assertGreaterEqual(float(tracker.belief.min()), 0.0)

    def test_belief_leaves_a_cabin_only_through_its_door(self) -> None:
        network = deck_network(noise_free=True)
        localiser = WallAwareLocaliser(network, grid_spacing_m=0.25, use_detections=False)
        tracker = GridTracker(localiser, network, reacquire_probability=0.0)
        in_cabin = localiser.grid_compartments == "Cabin 1"
        tracker.belief = in_cabin / in_cabin.sum()
        uninformative = {ap: None for ap in DECK_APS}

        tracker.update(uninformative)
        # one step: Cabin 1's neighbours other than the corridor are behind walls/panels
        for room in ("Cabin 2", "Cabin 3", "Mess"):
            self.assertLess(mass_in(tracker, room), 1e-200, room)  # only the 1e-300 numerical floor
        self.assertGreater(mass_in(tracker, "Corridor"), 0.0)
        # mass in the corridor is near the door (x 1.5-2.5), not spread along it
        corridor = localiser.grid_compartments == "Corridor"
        far = corridor & (localiser.grid_points[:, 0] > 6.0)
        self.assertLess(float(tracker.belief[far].sum()), 1e-200)

        for _ in range(25):
            tracker.update(uninformative)
        self.assertGreater(mass_in(tracker, "Cabin 2"), 1e-4)  # reached via the corridor and Cabin 2's door
        self.assertAlmostEqual(float(tracker.belief.sum()), 1.0)

    def test_reset_gives_uniform_belief(self) -> None:
        network = deck_network()
        tracker = GridTracker(WallAwareLocaliser(network, grid_spacing_m=0.4), network)
        tracker.update(network.scan(Receiver("p", (2.0, 2.0, 1.0)), Random(1)))
        self.assertGreater(np.ptp(tracker.belief), 0.0)
        tracker.reset()
        n = len(tracker.belief)
        np.testing.assert_allclose(tracker.belief, 1.0 / n)

    def test_reacquire_probability_floors_every_cell(self) -> None:
        network = deck_network(noise_free=True)
        localiser = WallAwareLocaliser(network, grid_spacing_m=0.4, use_detections=False)
        r = 0.01
        tracker = GridTracker(localiser, network, reacquire_probability=r)
        n = len(tracker.belief)
        tracker.belief = np.zeros(n)
        tracker.belief[0] = 1.0  # all mass in one corner cell
        tracker.update({ap: None for ap in DECK_APS})  # no likelihood: belief == prior
        self.assertGreaterEqual(float(tracker.belief.min()), r / n * (1 - 1e-9))
        self.assertAlmostEqual(float(tracker.belief.sum()), 1.0)

    def test_tracker_is_deterministic_given_seeds(self) -> None:
        def run():
            network = deck_network()
            tracker = GridTracker(WallAwareLocaliser(network, measurement="hybrid", grid_spacing_m=0.4, robust=True), network)
            walk = simulate_walk(network, Random(3), n_steps=30)
            rng = Random(4)
            return [tracker.update(network.scan(Receiver("p", (*xy, 1.0)), rng)).position for xy in walk]

        self.assertEqual(run(), run())

    def test_tracker_beats_snapshot_on_walks(self) -> None:
        network = deck_network()
        snapshot = WallAwareLocaliser(network, measurement="hybrid", grid_spacing_m=0.3)
        robust = WallAwareLocaliser(network, measurement="hybrid", grid_spacing_m=0.3, robust=True)
        trackers = {"snapshot": SnapshotAsTracker(snapshot), "tracker": GridTracker(robust, network)}
        walk_rng = Random(67890)
        walks = [simulate_walk(network, walk_rng, n_steps=100) for _ in range(3)]
        stats, _ = evaluate_tracking(trackers, network, walks, Random(78901))
        self.assertLess(stats["tracker"]["mean_err"], 0.9 * stats["snapshot"]["mean_err"], stats)
        self.assertGreaterEqual(stats["tracker"]["compartment_acc"], stats["snapshot"]["compartment_acc"])
        self.assertLess(stats["tracker"]["room_changes_ratio"], stats["snapshot"]["room_changes_ratio"])

    def test_scan_with_no_ap_heard_still_carries_non_detection_evidence(self) -> None:
        # Regression: log_likelihood used to return None for an all-miss scan, so the tracker
        # ignored it. Hearing none of the 4 corridor APs should make the corridor very unlikely.
        network = deck_network(noise_free=True, detection_snr_midpoint_db=6.0)
        localiser = WallAwareLocaliser(network, grid_spacing_m=0.4, robust=True)
        tracker = GridTracker(localiser, network)
        before = mass_in(tracker, "Corridor")
        tracker.update({ap: None for ap in DECK_APS})
        self.assertLess(mass_in(tracker, "Corridor"), 0.2 * before)


class SimulateWalkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.network = deck_network(noise_free=True)
        cls.walks = [simulate_walk(cls.network, Random(seed), n_steps=250) for seed in range(6)]

    def test_deterministic_given_seed(self) -> None:
        self.assertEqual(simulate_walk(self.network, Random(0), n_steps=250), self.walks[0])
        self.assertNotEqual(self.walks[0], self.walks[1])

    def test_length_and_types(self) -> None:
        for walk in self.walks:
            self.assertEqual(len(walk), 250)
            for p in walk:
                self.assertIsInstance(p[0], float)
                self.assertIsInstance(p[1], float)

    def test_positions_inside_compartments(self) -> None:
        env = self.network.environment
        for walk in self.walks:
            self.assertTrue(all(env.compartment_at(p) is not None for p in walk))

    def test_step_length_bounded_by_walking_speed(self) -> None:
        for speed, interval in ((1.2, 1.0), (0.8, 2.0)):
            walk = simulate_walk(self.network, Random(5), n_steps=200, walking_speed_mps=speed, scan_interval_s=interval)
            steps = [math.dist(a, b) for a, b in zip(walk, walk[1:])]
            self.assertLessEqual(max(steps), speed * interval + 1e-6)
            self.assertGreater(max(steps), 0.5 * speed * interval)  # actually walks somewhere

    def test_visits_several_compartments(self) -> None:
        env = self.network.environment
        for walk in self.walks:
            self.assertGreater(len({env.compartment_at(p) for p in walk}), 1)
        self.assertGreater(len({env.compartment_at(p) for w in self.walks for p in w}), 5)

    def test_room_changes_pass_through_the_corridor(self) -> None:
        # Every room on the deck only connects to the corridor, so the walk can never go
        # directly from one non-corridor room to another.
        env = self.network.environment
        for walk in self.walks:
            rooms = [env.compartment_at(p) for p in walk]
            for a, b in zip(rooms, rooms[1:]):
                if a != b:
                    self.assertIn("Corridor", (a, b), (a, b))

    def test_every_true_step_is_within_the_trackers_motion_support(self) -> None:
        # Consecutive walk positions can clip a door jamb in a straight line (the walker turns
        # the corner within one scan), but the tracker's walking-distance kernel must still give
        # every true step non-zero probability. (Previously ~4.6% of steps were impossible.)
        localiser = WallAwareLocaliser(self.network, grid_spacing_m=0.2, robust=True)
        tracker = GridTracker(localiser, self.network, reacquire_probability=0.0)
        P = tracker._predict_matrix.tocsc()
        from scipy.spatial import cKDTree
        tree = cKDTree(localiser.grid_points)
        unsupported = []
        for walk in self.walks:
            _, cells = tree.query(np.array(walk))
            for a, b in zip(cells, cells[1:]):
                if P[b, a] == 0:
                    unsupported.append((walk, a, b))
        self.assertEqual(len(unsupported), 0)


class _ConstantLocaliser:
    """Always returns the same point (or None)."""

    def __init__(self, position):
        self.position = position

    def locate(self, scan):
        return None if self.position is None else LocalisationEstimate(self.position)


class EvaluateTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.network = deck_network()
        cls.walks = [simulate_walk(cls.network, Random(s), n_steps=40) for s in (1, 2)]

    def _make_trackers(self):
        loc_a = WallAwareLocaliser(self.network, measurement="hybrid", grid_spacing_m=0.4, robust=True)
        loc_b = WallAwareLocaliser(self.network, measurement="hybrid", grid_spacing_m=0.4, robust=True)
        return {
            "tracker A": GridTracker(loc_a, self.network),
            "tracker B": GridTracker(loc_b, self.network),
            "snap A": SnapshotAsTracker(loc_a),
            "snap B": SnapshotAsTracker(WallAwareLocaliser(self.network, measurement="hybrid", grid_spacing_m=0.4, robust=True)),
        }

    def test_identical_trackers_see_identical_scans(self) -> None:
        stats, estimates = evaluate_tracking(self._make_trackers(), self.network, self.walks, Random(5))
        self.assertEqual(stats["tracker A"], stats["tracker B"])
        self.assertEqual(stats["snap A"], stats["snap B"])
        self.assertEqual(estimates["tracker A"], estimates["tracker B"])
        self.assertEqual(len(estimates["snap A"]), len(self.walks))
        self.assertEqual([len(w) for w in estimates["snap A"]], [len(w) for w in self.walks])

    def test_deterministic_and_keys_present(self) -> None:
        first, _ = evaluate_tracking(self._make_trackers(), self.network, self.walks, Random(5))
        second, _ = evaluate_tracking(self._make_trackers(), self.network, self.walks, Random(5))
        self.assertEqual(first, second)
        keys = {"errors", "mean_err", "median_err", "p90_err", "compartment_acc", "room_changes_ratio", "coverage95"}
        for s in first.values():
            self.assertEqual(set(s), keys)
            self.assertTrue(0.0 <= s["compartment_acc"] <= 1.0)
            self.assertEqual(s["errors"], sorted(s["errors"]))
            self.assertEqual(len(s["errors"]), sum(len(w) for w in self.walks))

    def test_misses_count_as_wrong_compartment(self) -> None:
        trackers = {
            "never": SnapshotAsTracker(_ConstantLocaliser(None)),
            "corridor": SnapshotAsTracker(_ConstantLocaliser((12.0, 5.0))),
        }
        stats, estimates = evaluate_tracking(trackers, self.network, self.walks, Random(5))
        self.assertEqual(stats["never"]["compartment_acc"], 0.0)
        self.assertEqual(stats["never"]["errors"], [])
        self.assertTrue(math.isnan(stats["never"]["mean_err"]))
        self.assertTrue(all(e is None for w in estimates["never"] for e in w))
        env = self.network.environment
        truth_in_corridor = [env.compartment_at(p) == "Corridor" for w in self.walks for p in w]
        self.assertAlmostEqual(stats["corridor"]["compartment_acc"], sum(truth_in_corridor) / len(truth_in_corridor))
        self.assertEqual(stats["corridor"]["room_changes_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
