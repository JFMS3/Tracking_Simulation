"""FTM checks using observations constructed without simulator ground truth."""

import math
import unittest

import numpy as np

from crew_tracking.ftm import (
    FTMConfig,
    FTMExchange,
    FTMLocaliser,
    aggregate_ftm_burst,
    distance_m_to_rtt_ns,
    rtt_ns_to_distance_m,
)
from crew_tracking.models import AccessPoint, FTMBurst, FTMSample, Reading


def burst_from_distances(*distances: float) -> FTMBurst:
    return FTMBurst(tuple(FTMSample(distance_m_to_rtt_ns(d)) for d in distances))


def scan_for_position(position, anchors, *, height=1.0, biases=None, errors=None):
    """Build measured slant ranges directly from surveyed anchor coordinates."""
    biases = biases or {}
    errors = errors or (-0.05, 0.0, 0.05, 0.0, 0.0)
    result = {}
    for ap in anchors:
        distance = math.dist((*position, height), ap.position)
        distances = tuple(distance + biases.get(ap.bssid, 0.0) + e for e in errors)
        result[ap] = Reading(rssi_dbm=-60.0, ftm_burst=burst_from_distances(*distances))
    return result


class FTMTimestampTests(unittest.TestCase):
    def test_four_timestamp_exchange_cancels_unsynchronised_clock_offsets(self):
        # Responder timestamps use a clock with a completely different epoch.
        first = FTMExchange(1_000.0, 5_000_000.0, 5_010_000.0, 11_100.0)
        shifted = FTMExchange(8_001_000.0, 3_000_000.0, 3_010_000.0, 8_011_100.0)

        self.assertAlmostEqual(first.to_sample().rtt_ns, 100.0)
        self.assertEqual(first.to_sample(), shifted.to_sample())
        self.assertAlmostEqual(rtt_ns_to_distance_m(first.to_sample().rtt_ns), 14.9896229)

    def test_responder_turnaround_time_does_not_become_propagation_time(self):
        short = FTMExchange(0.0, 100.0, 1_100.0, 1_020.0)
        long = FTMExchange(0.0, 100.0, 100_100.0, 100_020.0)

        self.assertAlmostEqual(short.to_sample().rtt_ns, 20.0)
        self.assertEqual(short.to_sample(), long.to_sample())

    def test_failed_exchange_is_not_a_successful_range_observation(self):
        sample = FTMExchange(0.0, 100.0, 1_100.0, 1_020.0, successful=False).to_sample()
        self.assertFalse(sample.successful)
        self.assertIsNone(aggregate_ftm_burst(FTMBurst((sample,) * 4)))

    def test_conversion_preserves_distance_across_realistic_scales(self):
        for distance in (-2.0, 0.0, 0.001, 0.1, 1.0, 100.0, 10_000.0):
            with self.subTest(distance=distance):
                self.assertAlmostEqual(rtt_ns_to_distance_m(distance_m_to_rtt_ns(distance)), distance)

    def test_nonfinite_conversion_inputs_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for convert in (rtt_ns_to_distance_m, distance_m_to_rtt_ns):
                with self.subTest(value=value, conversion=convert.__name__):
                    with self.assertRaises(ValueError):
                        convert(value)

    def test_integer_timestamps_preserve_nanoseconds_at_large_clock_epochs(self):
        ap_epoch = 10**18
        station_epoch = 8 * 10**18
        sample = FTMExchange(
            ap_epoch, station_epoch, station_epoch + 10_000, ap_epoch + 10_100
        ).to_sample()
        self.assertEqual(sample.rtt_ns, 100.0)

    def test_malformed_exchange_timestamps_are_not_usable_observations(self):
        for exchange in (
            FTMExchange(float("nan"), 100.0, 1_100.0, 1_020.0),
            FTMExchange(0.0, 100.0, float("inf"), 1_020.0),
            FTMExchange(2_000.0, 100.0, 1_100.0, 1_020.0),
            FTMExchange(0.0, 1_200.0, 1_100.0, 1_020.0),
        ):
            with self.subTest(exchange=exchange):
                self.assertFalse(exchange.to_sample().successful)


class FTMConfigurationTests(unittest.TestCase):
    def test_invalid_estimator_settings_fail_at_construction(self):
        for overrides in (
            {"min_samples": 1},
            {"min_samples": True},
            {"min_samples": 3.5},
            {"min_aps": 2},
            {"min_aps": True},
            {"range_std_floor_m": 0.0},
            {"range_std_floor_m": float("nan")},
            {"outlier_threshold": -1.0},
            {"robust_scale": 0.0},
            {"robust_scale": float("inf")},
            {"max_geometry_condition": 1.0},
            {"max_geometry_condition": float("nan")},
            {"robust": "yes"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    FTMConfig(**overrides)

    def test_invalid_geometry_and_calibration_fail_at_construction(self):
        for overrides in (
            {"receiver_height_m": float("nan")},
            {"range_bias_m": float("inf")},
            {"bias_by_bssid": {"AP": float("nan")}},
            {"bounds": ((0.0, 0.0), (0.0, 5.0))},
            {"bounds": ((2.0, 2.0), (1.0, 1.0))},
            {"bounds": ((float("nan"), 0.0), (5.0, 5.0))},
            {"bounds": ((0.0,), (5.0,))},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    FTMLocaliser(**overrides)


class FTMBurstTests(unittest.TestCase):
    def test_invalid_and_failed_samples_are_excluded_before_aggregation(self):
        valid = burst_from_distances(9.9, 10.0, 10.1).samples
        invalid = (
            FTMSample(None),
            FTMSample(float("nan")),
            FTMSample(float("inf")),
            FTMSample(-float("inf")),
            FTMSample(distance_m_to_rtt_ns(20.0), successful=False),
        )
        estimate = aggregate_ftm_burst(FTMBurst(valid + invalid))

        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.distance_m, 10.0)
        self.assertEqual(estimate.num_samples, 3)
        self.assertEqual(estimate.num_rejected, len(invalid))

    def test_repeated_ranges_with_one_large_error_remain_usable(self):
        estimate = aggregate_ftm_burst(burst_from_distances(10.0, 10.0, 10.0, 10.0, 90.0))

        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.distance_m, 10.0)
        self.assertEqual(estimate.num_samples, 4)
        self.assertEqual(estimate.num_rejected, 1)
        self.assertGreaterEqual(estimate.std_m, FTMConfig().range_std_floor_m)

    def test_minimum_sample_count_is_checked_after_rejection(self):
        self.assertIsNone(aggregate_ftm_burst(burst_from_distances(10.0, 10.0, 90.0)))
        self.assertIsNone(aggregate_ftm_burst(FTMBurst(())))
        self.assertIsNone(aggregate_ftm_burst(burst_from_distances(10.0, 10.0)))

    def test_calibration_bias_is_removed_in_metres(self):
        estimate = aggregate_ftm_burst(burst_from_distances(11.9, 12.0, 12.1), bias_m=2.0)

        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.distance_m, 10.0)

    def test_bias_correction_cannot_produce_a_usable_negative_range(self):
        self.assertIsNone(aggregate_ftm_burst(burst_from_distances(1.0, 1.0, 1.0), bias_m=2.0))

    def test_signed_hardware_measurements_can_be_recovered_by_calibration(self):
        estimate = aggregate_ftm_burst(burst_from_distances(-2.0, -2.0, -2.0), bias_m=-5.0)
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.distance_m, 3.0)

    def test_signed_noise_near_an_anchor_is_averaged_without_positive_truncation(self):
        estimate = aggregate_ftm_burst(burst_from_distances(-0.3, -0.1, 0.1, 0.3, 0.5))
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.distance_m, 0.1)
        self.assertEqual(estimate.num_samples, 5)
        self.assertEqual(estimate.num_rejected, 0)

    def test_nonpositive_aggregate_is_not_a_usable_range(self):
        for distances in ((0.0, 0.0, 0.0), (-0.5, -0.3, -0.1)):
            with self.subTest(distances=distances):
                self.assertIsNone(aggregate_ftm_burst(burst_from_distances(*distances)))

    def test_dispersion_increases_reported_uncertainty(self):
        config = FTMConfig(range_std_floor_m=0.01)
        quiet = aggregate_ftm_burst(burst_from_distances(9.99, 10.0, 10.01, 10.0, 10.0), config=config)
        noisy = aggregate_ftm_burst(burst_from_distances(9.0, 9.5, 10.0, 10.5, 11.0), config=config)

        self.assertIsNotNone(quiet)
        self.assertIsNotNone(noisy)
        self.assertGreater(noisy.std_m, quiet.std_m)


class FTMLocalisationTests(unittest.TestCase):
    def setUp(self):
        self.anchors = [
            AccessPoint("southwest", "01", (0.0, 0.0, 1.0), 5180),
            AccessPoint("southeast", "02", (12.0, 0.0, 3.0), 5180),
            AccessPoint("northwest", "03", (0.0, 10.0, 5.0), 5180),
            AccessPoint("northeast", "04", (12.0, 10.0, 9.0), 5180),
        ]
        self.position = (2.0, 3.0)

    def assert_position_close(self, estimate, position, *, tolerance=1e-5):
        self.assertIsNotNone(estimate)
        self.assertLess(math.dist(estimate.position, position), tolerance)

    def test_exact_slant_ranges_resolve_position_with_different_anchor_heights(self):
        scan = scan_for_position(self.position, self.anchors, height=1.7, errors=(0.0,) * 5)
        estimate = FTMLocaliser(receiver_height_m=1.7).locate(scan)

        self.assert_position_close(estimate, self.position)
        self.assertEqual(estimate.num_aps_used, 4)
        self.assertIsNotNone(estimate.covariance)
        self.assertEqual(estimate.covariance.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(estimate.covariance)))
        self.assertTrue(np.all(np.linalg.eigvalsh(estimate.covariance) >= -1e-12))

    def test_three_noncollinear_anchors_are_sufficient(self):
        scan = scan_for_position(self.position, self.anchors[:3], errors=(0.0,) * 5)
        self.assert_position_close(FTMLocaliser().locate(scan), self.position)

    def test_minimum_anchor_count_uses_only_usable_ftm_bursts(self):
        scan = scan_for_position(self.position, self.anchors[:2])
        scan[self.anchors[2]] = Reading(-50.0, ftm_burst=FTMBurst((FTMSample(None),) * 5))
        scan[self.anchors[3]] = None

        self.assertIsNone(FTMLocaliser().locate(scan))
        self.assertIsNone(FTMLocaliser().locate({}))

    def test_generic_network_latency_is_never_used_as_ftm_range(self):
        scan = {ap: Reading(rssi_dbm=-50.0, rtt_ms=8.0001) for ap in self.anchors}
        self.assertIsNone(FTMLocaliser().locate(scan))

    def test_rssi_is_not_required_when_successful_ftm_bursts_are_available(self):
        scan = scan_for_position(self.position, self.anchors, errors=(0.0,) * 5)
        for reading in scan.values():
            reading.rssi_dbm = float("nan")
        self.assert_position_close(FTMLocaliser().locate(scan), self.position)

    def test_ap_without_ftm_capability_does_not_count_towards_minimum(self):
        unsupported = AccessPoint("unsupported", "03", self.anchors[2].position, 5180, ftm_responder=False)
        scan = scan_for_position(self.position, self.anchors[:2] + [unsupported])
        self.assertIsNone(FTMLocaliser().locate(scan))

    def test_repeated_bssid_does_not_count_as_an_independent_anchor(self):
        anchors = [
            AccessPoint("first", "AA:BB", (0.0, 0.0, 1.0), 5180),
            AccessPoint("second", "CC:DD", (10.0, 0.0, 1.0), 5180),
            AccessPoint("alias", "aa:bb", (0.0, 10.0, 1.0), 5180),
        ]
        scan = scan_for_position(self.position, anchors)
        self.assertIsNone(FTMLocaliser().locate(scan))

    def test_conflicting_positions_for_one_bssid_are_rejected_in_any_scan_order(self):
        conflicting = AccessPoint("conflicting survey", "01", (30.0, 20.0, 1.0), 5180)
        scan = scan_for_position(self.position, self.anchors + [conflicting])
        self.assertIsNone(FTMLocaliser().locate(scan))
        self.assertIsNone(FTMLocaliser().locate(dict(reversed(tuple(scan.items())))))

    def test_collinear_anchors_do_not_report_an_ambiguous_position(self):
        anchors = [AccessPoint(str(i), str(i), (float(i * 5), 0.0, 1.0), 5180) for i in range(4)]
        scan = scan_for_position(self.position, anchors, errors=(0.0,) * 5)
        self.assertIsNone(FTMLocaliser().locate(scan))

    def test_duplicate_anchor_positions_are_degenerate(self):
        anchors = [AccessPoint(str(i), str(i), (0.0, 0.0, 1.0), 5180) for i in range(4)]
        scan = scan_for_position(self.position, anchors, errors=(0.0,) * 5)
        self.assertIsNone(FTMLocaliser().locate(scan))

    def test_burst_outliers_do_not_move_the_solution(self):
        scan = scan_for_position(self.position, self.anchors, errors=(0.0, 0.0, 0.0, 0.0, 40.0))
        self.assert_position_close(FTMLocaliser().locate(scan), self.position)

    def test_global_calibration_recovers_position_from_biased_ranges(self):
        scan = scan_for_position(self.position, self.anchors, biases={ap.bssid: 2.0 for ap in self.anchors})
        self.assert_position_close(FTMLocaliser(range_bias_m=2.0).locate(scan), self.position)

    def test_per_anchor_calibration_recovers_position_from_different_biases(self):
        biases = {"01": 1.0, "02": 2.5, "03": 0.25, "04": 4.0}
        scan = scan_for_position(self.position, self.anchors, biases=biases)
        self.assert_position_close(FTMLocaliser(bias_by_bssid=biases).locate(scan), self.position)

    def test_uncertain_burst_has_less_influence_on_the_position(self):
        last = self.anchors[-1]
        quiet_scan = scan_for_position(
            self.position, self.anchors, errors=(0.0,) * 5, biases={last.bssid: 3.0}
        )
        noisy_scan = dict(quiet_scan)
        measured_distance = math.dist((*self.position, 1.0), last.position) + 3.0
        noisy_scan[last] = Reading(-60.0, ftm_burst=burst_from_distances(
            *(measured_distance + error for error in (-4.0, -2.0, 0.0, 2.0, 4.0))
        ))
        localiser = FTMLocaliser(config=FTMConfig(robust=False))

        quiet_estimate = localiser.locate(quiet_scan)
        noisy_estimate = localiser.locate(noisy_scan)

        self.assertIsNotNone(quiet_estimate)
        self.assertIsNotNone(noisy_estimate)
        quiet_error = math.dist(quiet_estimate.position, self.position)
        noisy_error = math.dist(noisy_estimate.position, self.position)
        self.assertLess(noisy_error, 0.1)
        self.assertLess(noisy_error, quiet_error / 5)

    def test_robust_solver_limits_a_persistently_biased_anchor(self):
        anchors = [
            AccessPoint(str(i), str(i), (
                12 * math.cos(i * math.pi / 4),
                12 * math.sin(i * math.pi / 4),
                2.0,
            ), 5180)
            for i in range(8)
        ]
        scan = scan_for_position(
            self.position, anchors, errors=(0.0,) * 5, biases={anchors[-1].bssid: 8.0}
        )
        ordinary = FTMLocaliser(config=FTMConfig(robust=False)).locate(scan)
        robust = FTMLocaliser(config=FTMConfig(robust=True)).locate(scan)

        self.assertIsNotNone(ordinary)
        self.assertIsNotNone(robust)
        ordinary_error = math.dist(ordinary.position, self.position)
        robust_error = math.dist(robust.position, self.position)
        self.assertLess(robust_error, 0.25)
        self.assertLess(robust_error, ordinary_error / 5)

    def test_bounds_constrain_solution_without_reporting_unconstrained_covariance(self):
        scan = scan_for_position(self.position, self.anchors, errors=(0.0,) * 5)
        estimate = FTMLocaliser(bounds=((3.0, 0.0), (10.0, 10.0))).locate(scan)

        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.position[0], 3.0, places=6)
        self.assertGreaterEqual(estimate.position[1], 0.0)
        self.assertLessEqual(estimate.position[1], 10.0)
        self.assertIsNone(estimate.covariance)


if __name__ == "__main__":
    unittest.main()
