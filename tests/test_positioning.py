import math
import unittest
from random import Random

import numpy as np

from environment import Compartment, ShipEnvironment, Wall
from network import AccessPoint, Network, Receiver
from positioning import WallAwareLocaliser
from tracking import GridTracker


def two_room_environment(door: bool = False) -> ShipEnvironment:
    """Rooms A (x 0-4) and B (x 4-8), 4 m deep, split by a 20 dB bulkhead at x = 4.

    With `door`, the bulkhead has a 1 m gap at y 1.5-2.5.
    """
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


ACCESS_POINTS = [
    AccessPoint("AP-A", "00:00:00:00:00:01", (1.0, 1.0, 1.0), 2412),
    AccessPoint("AP-B", "00:00:00:00:00:02", (7.0, 3.0, 1.0), 2412),
    AccessPoint("AP-C", "00:00:00:00:00:03", (3.0, 3.5, 1.0), 2412),
]


def make_network(environment: ShipEnvironment, **overrides) -> Network:
    """Noise-free channel, as in tests/test_network.py."""
    parameters = {
        "environment": environment,
        "access_points": list(ACCESS_POINTS),
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
    parameters.update(overrides)
    return Network(**parameters)


class WallAwareLocaliserTests(unittest.TestCase):
    def test_vectorised_wall_terms_match_walls_crossed(self) -> None:
        network = make_network(two_room_environment(door=True))
        localiser = WallAwareLocaliser(network, grid_spacing_m=0.25)

        for i in range(0, len(localiser.grid_points), 7):
            x, y = localiser.grid_points[i]
            receiver = Receiver("probe", (x, y, 1.0))
            for j, ap in enumerate(network.access_points):
                self.assertAlmostEqual(
                    localiser._expected_rssi[i, j],
                    network.mean_rssi(ap, receiver),
                    places=9,
                )

    def test_noise_free_scan_is_located_in_the_right_room_and_place(self) -> None:
        network = make_network(two_room_environment())
        rng = Random(1)

        for measurement in WallAwareLocaliser.MEASUREMENTS:
            localiser = WallAwareLocaliser(network, measurement=measurement, grid_spacing_m=0.1, rssi_sigma_db=0.5)
            # Room A has two APs inside it, room B only one: a clean scan in B pins
            # the phone to a ring around AP-B, so the posterior mean is less exact there.
            for true_xy, room, tolerance in [((1.7, 2.6), "A", 0.25), ((5.4, 0.9), "B", 0.25), ((7.2, 3.6), "B", 0.6)]:
                estimate = localiser.locate(network.scan(Receiver("probe", (*true_xy, 1.0)), rng))
                self.assertIsNotNone(estimate)
                self.assertEqual(network.environment.compartment_at(estimate.position), room, measurement)
                self.assertLess(math.dist(true_xy, estimate.position), tolerance, measurement)

    def test_returns_none_when_no_ap_is_heard(self) -> None:
        localiser = WallAwareLocaliser(make_network(two_room_environment()))
        self.assertIsNone(localiser.locate({ap: None for ap in ACCESS_POINTS}))

    def test_device_offset_estimation_absorbs_a_constant_phone_bias(self) -> None:
        network = make_network(two_room_environment(), measurement_noise_std_db=1.0)
        naive = WallAwareLocaliser(network, measurement="rss")
        offset_aware = WallAwareLocaliser(network, measurement="rss", device_offset_sigma_db=6.0)
        rng = Random(1)

        naive_errors, aware_errors = [], []
        for room in ("A", "B"):
            for _ in range(40):
                true_xy = network.environment.random_point_in(room, rng)
                scan = network.scan(Receiver("probe", (*true_xy, 1.0), rssi_bias_db=-8.0), rng)
                naive_errors.append(math.dist(true_xy, naive.locate(scan).position))
                aware_errors.append(math.dist(true_xy, offset_aware.locate(scan).position))

        self.assertLess(np.mean(aware_errors), np.mean(naive_errors))

    def test_rejects_unknown_measurement(self) -> None:
        with self.assertRaises(ValueError):
            WallAwareLocaliser(make_network(two_room_environment()), measurement="sonar")


class GridTrackerTests(unittest.TestCase):
    def _mass_in_room(self, tracker: GridTracker, room: str) -> float:
        return float(tracker.belief[tracker.localiser.grid_compartments == room].sum())

    def _start_in_room_a(self, tracker: GridTracker) -> None:
        in_a = tracker.localiser.grid_compartments == "A"
        tracker.belief = in_a / in_a.sum()

    def test_belief_cannot_pass_through_a_solid_bulkhead(self) -> None:
        network = make_network(two_room_environment(door=False))
        localiser = WallAwareLocaliser(network, grid_spacing_m=0.2)
        tracker = GridTracker(localiser, network, reacquire_probability=0.0)
        self._start_in_room_a(tracker)

        no_aps_heard = {ap: None for ap in ACCESS_POINTS}
        localiser.use_detections = False  # pure motion: no evidence either way
        for _ in range(30):
            tracker.update(no_aps_heard)

        # only the 1e-300 numerical floor can be there
        self.assertLess(self._mass_in_room(tracker, "B"), 1e-200)

    def test_belief_spreads_through_a_doorway(self) -> None:
        network = make_network(two_room_environment(door=True))
        localiser = WallAwareLocaliser(network, grid_spacing_m=0.2)
        tracker = GridTracker(localiser, network, reacquire_probability=0.0)
        self._start_in_room_a(tracker)

        no_aps_heard = {ap: None for ap in ACCESS_POINTS}
        localiser.use_detections = False
        for _ in range(30):
            tracker.update(no_aps_heard)

        self.assertGreater(self._mass_in_room(tracker, "B"), 0.05)

    def test_tracker_beats_snapshots_for_a_stationary_person(self) -> None:
        network = make_network(two_room_environment(door=True), measurement_noise_std_db=3.0)
        localiser = WallAwareLocaliser(network, grid_spacing_m=0.2)
        tracker = GridTracker(localiser, network)
        rng = Random(3)
        true_xy = (6.0, 1.0)

        snapshot_errors = []
        for _ in range(20):
            scan = network.scan(Receiver("probe", (*true_xy, 1.0)), rng)
            snapshot_errors.append(math.dist(true_xy, localiser.locate(scan).position))
            estimate = tracker.update(scan)

        self.assertEqual(network.environment.compartment_at(estimate.position), "B")
        self.assertLess(math.dist(true_xy, estimate.position), float(np.mean(snapshot_errors)))
        self.assertAlmostEqual(float(np.sum(tracker.belief)), 1.0)

if __name__ == "__main__":
    unittest.main()
