"""Lifecycle checks for timestamped crew-device tracking and recovery."""

import unittest

import numpy as np

from crew_tracking.models import LocalisationEstimate
from crew_tracking.tracking import KalmanConfig, KalmanTracker


def measured(x, y=0.0):
    return LocalisationEstimate((x, y))


class TrackingLifecycleTests(unittest.TestCase):
    def test_devices_have_independent_motion_and_reset(self):
        tracker = KalmanTracker()
        tracker.update("alice", 0, measured(0))
        tracker.update("bob", 0, measured(100))
        tracker.update("alice", 1, measured(1))
        bob = tracker.predict("bob", 1)
        self.assertEqual(bob.position, (100, 0))
        self.assertEqual(bob.velocity, (0, 0))
        tracker.reset("alice")
        self.assertIsNone(tracker.predict("alice", 0))
        self.assertEqual(tracker.update("alice", 0, measured(-50)).status, "initialised")
        self.assertIsNotNone(tracker.predict("bob", 2))
        tracker.reset()
        self.assertIsNone(tracker.predict("bob", 3))

    def test_unknown_missing_measurement_does_not_start_track(self):
        tracker = KalmanTracker()
        self.assertIsNone(tracker.update("alice", 5, None))
        self.assertEqual(tracker.update("alice", 0, measured(1)).status, "initialised")

    def test_duplicate_and_out_of_order_timestamps_do_not_change_state(self):
        tracker, reference = KalmanTracker(), KalmanTracker()
        for instance in (tracker, reference):
            instance.update("alice", 2, measured(0))
        for timestamp in (2, 1, -1, np.nan, np.inf, True):
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(ValueError):
                    tracker.update("alice", timestamp, measured(100))
        actual = tracker.update("alice", 3, measured(1))
        expected = reference.update("alice", 3, measured(1))
        np.testing.assert_allclose(actual.position, expected.position)
        np.testing.assert_allclose(actual.covariance, expected.covariance)

    def test_invalid_covariance_does_not_advance_timestamp(self):
        tracker = KalmanTracker()
        tracker.update("alice", 0, measured(0))
        invalid_matrices = (
            np.ones((3, 3)), [[1, 1], [0, 1]], [[-1, 0], [0, 1]],
            [[np.nan, 0], [0, 1]], [[np.inf, 0], [0, 1]], "invalid",
        )
        for covariance in invalid_matrices:
            with self.subTest(covariance=covariance):
                with self.assertRaises(ValueError):
                    tracker.update("alice", 1, None, measurement_covariance=covariance)
        self.assertEqual(tracker.update("alice", 1, measured(0)).status, "updated")

    def test_bad_device_ids_fail_even_on_unknown_missing_positions(self):
        for device in (None, "", " ", 5):
            with self.subTest(device=device):
                with self.assertRaises(ValueError):
                    KalmanTracker().predict(device, 0)

    def test_invalid_positions_act_as_missing_measurements(self):
        for position in ((np.nan, 0), (0, np.inf), (1,), (1, 2, 3), None):
            with self.subTest(position=position):
                tracker = KalmanTracker()
                invalid = LocalisationEstimate(position)
                self.assertIsNone(tracker.update("alice", 0, invalid))
                tracker.update("alice", 0, measured(0))
                result = tracker.update("alice", 1, invalid)
                self.assertEqual(result.status, "predicted")
                self.assertEqual(result.last_measurement_timestamp_s, 0)

    def test_short_outage_predicts_then_hides_stale_position_and_recovers(self):
        tracker = KalmanTracker()
        first = tracker.update("alice", 0, measured(0))
        short = tracker.predict("alice", 2)
        self.assertEqual(short.status, "predicted")
        self.assertEqual(short.measurement_age_s, 2)
        self.assertGreater(short.position_std_m, first.position_std_m)
        self.assertIsNotNone(tracker.predict("alice", 3))
        self.assertIsNone(tracker.predict("alice", 4))
        corrected = tracker.update("alice", 5, measured(0.1))
        self.assertEqual(corrected.status, "updated")
        self.assertEqual(corrected.measurement_age_s, 0)

    def test_expired_state_is_removed_or_reinitialised_without_predicting_huge_gap(self):
        tracker = KalmanTracker()
        tracker.update("alice", 0, measured(0))
        self.assertIsNone(tracker.update("alice", 10, None))
        self.assertEqual(tracker.update("alice", 0, measured(50)).status, "initialised")
        far_future = tracker.update("alice", 1e100, measured(100))
        self.assertEqual(far_future.status, "reinitialised")
        self.assertEqual(far_future.position, (100, 0))
        self.assertEqual(far_future.velocity, (0, 0))
        self.assertTrue(np.all(np.isfinite(far_future.covariance)))

    def test_display_prediction_does_not_reject_a_delayed_new_capture(self):
        tracker, reference = KalmanTracker(), KalmanTracker()
        for instance in (tracker, reference):
            instance.update("alice", 10, measured(0))
        display = tracker.predict("alice", 10.5)
        display.covariance[:] = 10000
        actual = tracker.update("alice", 10.4, measured(0.3))
        expected = reference.update("alice", 10.4, measured(0.3))
        self.assertEqual(actual.status, "updated")
        np.testing.assert_allclose(actual.position, expected.position)
        np.testing.assert_allclose(actual.covariance, expected.covariance)
        self.assertIsNotNone(tracker.predict("alice", 10.4))
        with self.assertRaises(ValueError):
            tracker.predict("alice", 10.3)

    def test_display_predictions_do_not_break_reacquisition_streak(self):
        tracker = KalmanTracker()
        tracker.update("alice", 0, measured(0))
        tracker.update("alice", 0.1, measured(100))
        tracker.predict("alice", 0.15)
        tracker.update("alice", 0.2, measured(100))
        tracker.predict("alice", 0.25)
        self.assertEqual(tracker.update("alice", 0.3, measured(100)).status, "reinitialised")

    def test_expired_display_prediction_does_not_remove_capture_ordering(self):
        tracker = KalmanTracker()
        tracker.update("alice", 5, measured(0))
        self.assertIsNone(tracker.predict("alice", 100))
        with self.assertRaises(ValueError):
            tracker.update("alice", 4, measured(1))
        self.assertEqual(tracker.update("alice", 6, measured(1)).status, "updated")

    def test_isolated_outlier_does_not_refresh_measurement_time(self):
        tracker = KalmanTracker()
        tracker.update("alice", 0, measured(0))
        rejected = tracker.update("alice", 1, measured(100))
        self.assertEqual(rejected.status, "outlier_rejected")
        self.assertEqual(rejected.position, (0, 0))
        self.assertEqual(rejected.last_measurement_timestamp_s, 0)
        self.assertGreater(rejected.innovation_distance_squared, 16)
        self.assertIsNone(tracker.predict("alice", 4))

    def test_three_consistent_rejections_reacquire_with_zero_velocity(self):
        tracker = KalmanTracker()
        tracker.update("alice", 0, measured(0))
        for time, position in ((1, 100), (2, 100.1)):
            self.assertEqual(
                tracker.update("alice", time, measured(position)).status,
                "outlier_rejected",
            )
        recovered = tracker.update("alice", 3, measured(99.9))
        self.assertEqual(recovered.status, "reinitialised")
        self.assertEqual(recovered.position, (99.9, 0))
        self.assertEqual(recovered.velocity, (0, 0))
        self.assertEqual(recovered.measurement_age_s, 0)

    def test_inconsistent_rejections_and_gaps_break_reacquisition_streak(self):
        tracker = KalmanTracker()
        tracker.update("alice", 0, measured(0))
        tracker.update("alice", 0.1, measured(100))
        tracker.update("alice", 0.2, measured(-100))
        self.assertEqual(tracker.update("alice", 0.3, measured(100)).status, "outlier_rejected")
        tracker.update("alice", 0.4, None)
        self.assertEqual(tracker.update("alice", 0.5, measured(100)).status, "outlier_rejected")
        self.assertEqual(tracker.update("alice", 0.6, measured(100)).status, "outlier_rejected")
        self.assertEqual(tracker.update("alice", 0.7, measured(100)).status, "reinitialised")

    def test_accepted_measurement_breaks_reacquisition_streak(self):
        tracker = KalmanTracker()
        tracker.update("alice", 0, measured(0))
        tracker.update("alice", 0.1, measured(100))
        tracker.update("alice", 0.2, measured(100))
        self.assertEqual(tracker.update("alice", 0.3, measured(0)).status, "updated")
        self.assertEqual(tracker.update("alice", 0.4, measured(100)).status, "outlier_rejected")

    def test_input_and_snapshot_mutation_does_not_change_filter(self):
        tracker = KalmanTracker()
        position = np.array([1.0, 2.0])
        supplied = np.array([[1.0, 0.2], [0.2, 2.0]])
        initial = tracker.update(
            "alice", 0, LocalisationEstimate(position), measurement_covariance=supplied
        )
        original = initial.covariance.copy()
        position[:] = 100
        supplied[:] = 100
        initial.covariance[:] = 100
        predicted = tracker.predict("alice", 1)
        self.assertEqual(predicted.position, (1, 2))
        self.assertGreater(predicted.covariance[0, 0], original[0, 0])
        self.assertLess(predicted.covariance[0, 0], 10)

    def test_source_covariance_is_not_automatically_trusted(self):
        first, second = KalmanTracker(), KalmanTracker()
        unusual = LocalisationEstimate((1, 2), covariance=np.full((2, 2), np.nan))
        a = first.update("alice", 0, unusual)
        b = second.update("alice", 0, measured(1, 2))
        np.testing.assert_allclose(a.covariance, b.covariance)

    def test_reacquisition_retains_current_explicit_covariance(self):
        tracker = KalmanTracker()
        tracker.update("alice", 0, measured(0))
        for timestamp in (0.1, 0.2):
            tracker.update("alice", timestamp, measured(100))
        supplied = np.array([[4, 0.25], [0.25, 1]])
        reacquired = tracker.update("alice", 0.3, measured(100), measurement_covariance=supplied)
        self.assertEqual(reacquired.status, "reinitialised")
        np.testing.assert_allclose(reacquired.covariance[:2, :2], supplied)

    def test_config_validation_rejects_unusable_settings(self):
        invalid = (
            {"measurement_std_m": 0}, {"position_std_floor_m": -1},
            {"acceleration_noise_density_m2_s3": np.inf},
            {"initial_velocity_std_mps": np.nan}, {"innovation_gate": 0},
            {"reacquire_after_rejections": 1}, {"reacquire_after_rejections": 2.5},
            {"reacquire_after_rejections": True},
            {"measurement_std_m": 0.1}, {"reset_after_s": 3},
            {"max_prediction_age_s": -1},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    KalmanConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
