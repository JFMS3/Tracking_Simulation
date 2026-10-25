"""Integration checks for synthetic FTM observations and algorithm inputs."""

from dataclasses import replace
import math
from random import Random
import unittest

from crew_tracking.ftm import FTMLocaliser, aggregate_ftm_burst, rtt_ns_to_distance_m
from crew_tracking.models import AccessPoint
from simulator.environment import ShipEnvironment
from simulator.network import Network, Receiver


def noiseless_network(**overrides):
    parameters = {
        "environment": ShipEnvironment(),
        "shadowing_std_db": 0.0,
        "fast_fading_samples": 0,
        "measurement_noise_std_db": 0.0,
        "rssi_quantization_db": 0.0,
        "noise_floor_variation_std_db": 0.0,
        "interference_burst_start_probability": 0.0,
        "detection_snr_midpoint_db": -100.0,
        "rtt_ranging_noise_m": 0.0,
        "rtt_weak_signal_penalty_m": 0.0,
        "rtt_outlier_probability": 0.0,
        "rtt_nlos_outlier_probability": 0.0,
        "ftm_success_probability": 1.0,
    }
    parameters.update(overrides)
    return Network(**parameters)


class NetworkFTMTests(unittest.TestCase):
    def setUp(self):
        self.anchors = [
            AccessPoint("SW", "01", (0.0, 0.0, 3.0), 5180),
            AccessPoint("SE", "02", (12.0, 0.0, 5.0), 5180),
            AccessPoint("NW", "03", (0.0, 10.0, 7.0), 2412),
            AccessPoint("NE", "04", (12.0, 10.0, 9.0), 2412),
        ]
        self.receiver = Receiver("phone", (2.0, 3.0, 1.5))

    def test_noiseless_scan_measures_slant_ranges_and_localises_end_to_end(self):
        network = noiseless_network(access_points=self.anchors)
        scan = network.scan(self.receiver, Random(123))

        for ap, reading in scan.items():
            self.assertIsNotNone(reading)
            self.assertIsNotNone(reading.ftm_burst)
            self.assertEqual(len(reading.ftm_burst.samples), network.ftm_burst_size)
            expected_distance = math.dist(ap.position, self.receiver.position)
            for sample in reading.ftm_burst.samples:
                self.assertTrue(sample.successful)
                self.assertAlmostEqual(rtt_ns_to_distance_m(sample.rtt_ns), expected_distance, places=7)
        estimate = FTMLocaliser(receiver_height_m=self.receiver.position[2]).locate(scan)
        self.assertIsNotNone(estimate)
        self.assertLess(math.dist(estimate.position, self.receiver.xy), 1e-6)

    def test_short_ftm_ranges_do_not_inherit_rssi_minimum_distance(self):
        ap = AccessPoint("nearby", "05", (0.0, 0.0, 1.0), 5180)
        receiver = Receiver("phone", (0.03, 0.04, 1.0))
        reading = noiseless_network(access_points=[ap]).scan(receiver, Random(1))[ap]

        self.assertIsNotNone(reading)
        estimate = aggregate_ftm_burst(reading.ftm_burst)
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.distance_m, 0.05, places=7)

    def test_ftm_disabled_does_not_remove_rssi_readings(self):
        network = noiseless_network(access_points=self.anchors)
        scan = network.scan(self.receiver, Random(1), include_ftm=False)

        self.assertEqual(len(scan), len(self.anchors))
        for reading in scan.values():
            self.assertIsNotNone(reading)
            self.assertTrue(math.isfinite(reading.rssi_dbm))
            self.assertIsNone(reading.ftm_burst)
        self.assertIsNone(FTMLocaliser().locate(scan))

    def test_nonresponder_remains_visible_to_rssi_algorithms(self):
        ap = replace(self.anchors[0], ftm_responder=False)
        network = noiseless_network(access_points=[ap])
        reading = network.scan(self.receiver, Random(1))[ap]

        self.assertIsNotNone(reading)
        self.assertTrue(math.isfinite(reading.rssi_dbm))
        self.assertIsNone(reading.ftm_burst)

    def test_failed_ftm_exchange_burst_does_not_remove_rssi_readings(self):
        network = noiseless_network(access_points=self.anchors, ftm_success_probability=0.0)
        scan = network.scan(self.receiver, Random(1))

        for reading in scan.values():
            self.assertIsNotNone(reading)
            self.assertTrue(math.isfinite(reading.rssi_dbm))
            self.assertIsNotNone(reading.ftm_burst)
            self.assertTrue(all(not sample.successful for sample in reading.ftm_burst.samples))
            self.assertIsNone(aggregate_ftm_burst(reading.ftm_burst))
        self.assertIsNone(FTMLocaliser().locate(scan))

    def test_burst_size_and_enable_flag_do_not_change_other_measurements_or_rng(self):
        baseline = Network(ShipEnvironment(), access_points=self.anchors)
        variants = [
            Network(ShipEnvironment(), access_points=self.anchors, ftm_burst_size=size)
            for size in (1, 8, 32)
        ]
        baseline_rng = Random(2026)
        variant_rngs = [Random(2026) for _ in variants]

        for _ in range(20):
            expected_scan = baseline.scan(self.receiver, baseline_rng, include_ftm=False)
            for network, rng in zip(variants, variant_rngs):
                actual_scan = network.scan(self.receiver, rng)
                without_ftm = {
                    ap: replace(reading, ftm_burst=None) if reading is not None else None
                    for ap, reading in actual_scan.items()
                }
                self.assertEqual(without_ftm, expected_scan)
                self.assertEqual(rng.getstate(), baseline_rng.getstate())

    def test_explicit_ftm_rng_is_reproducible_and_independent_of_radio_rng(self):
        first = Network(ShipEnvironment(), access_points=self.anchors)
        second = Network(ShipEnvironment(), access_points=self.anchors)
        first_radio_rng, second_radio_rng = Random(55), Random(55)
        first_ftm_rng, second_ftm_rng = Random(99), Random(99)

        for _ in range(5):
            first_scan = first.scan(self.receiver, first_radio_rng, ftm_rng=first_ftm_rng)
            second_scan = second.scan(self.receiver, second_radio_rng, ftm_rng=second_ftm_rng)
            self.assertEqual(first_scan, second_scan)
        self.assertEqual(first_radio_rng.getstate(), second_radio_rng.getstate())
        self.assertEqual(first_ftm_rng.getstate(), second_ftm_rng.getstate())
        self.assertNotEqual(first_ftm_rng.getstate(), Random(99).getstate())

    def test_invalid_ftm_configuration_is_rejected(self):
        for overrides in (
            {"ftm_burst_size": 0},
            {"ftm_burst_size": -1},
            {"ftm_burst_size": True},
            {"ftm_burst_size": 2.5},
            {"ftm_success_probability": -0.01},
            {"ftm_success_probability": 1.01},
            {"ftm_success_probability": float("nan")},
            {"ftm_success_probability": float("inf")},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    noiseless_network(**overrides)


if __name__ == "__main__":
    unittest.main()
