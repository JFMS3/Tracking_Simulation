"""Independent mathematical and behaviour checks for the Kalman tracker."""

import math
from random import Random
import unittest

import numpy as np

from crew_tracking.models import LocalisationEstimate
from crew_tracking.tracking import KalmanConfig, KalmanTracker


def fix(x, y=0.0):
    return LocalisationEstimate((x, y), num_aps_used=4)


def transition_and_noise(dt, density):
    """Independent continuous white-acceleration model in x,y,vx,vy order."""
    transition = np.eye(4)
    transition[0, 2] = transition[1, 3] = dt
    noise = np.zeros((4, 4))
    for axis in (0, 1):
        noise[axis, axis] = density * dt ** 3 / 3
        noise[axis, axis + 2] = noise[axis + 2, axis] = density * dt ** 2 / 2
        noise[axis + 2, axis + 2] = density * dt
    return transition, noise


class KalmanMathematicsTests(unittest.TestCase):
    def config(self, **overrides):
        parameters = dict(
            measurement_std_m=2.0,
            initial_velocity_std_mps=3.0,
            acceleration_noise_density_m2_s3=0.5,
            position_std_floor_m=1e-6,
            innovation_gate=1e6,
            max_prediction_age_s=30.0,
            reset_after_s=60.0,
        )
        parameters.update(overrides)
        return KalmanConfig(**parameters)

    def test_prediction_matches_analytic_covariance_for_actual_elapsed_time(self):
        tracker = KalmanTracker(self.config())
        tracker.update("crew", 10.0, fix(4.0, 7.0))
        result = tracker.predict("crew", 12.0)

        transition, noise = transition_and_noise(2.0, 0.5)
        expected = transition @ np.diag([4.0, 4.0, 9.0, 9.0]) @ transition.T + noise
        np.testing.assert_allclose(result.covariance, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(result.position, (4.0, 7.0))
        self.assertEqual(result.velocity, (0.0, 0.0))
        self.assertEqual(result.measurement_age_s, 2.0)

    def test_update_matches_independent_matrix_solution_with_correlated_xy_noise(self):
        tracker = KalmanTracker(self.config())
        initial = tracker.update("crew", 0.0, fix(1.0, 2.0))
        measurement_noise = np.array([[1.0, 0.3], [0.3, 2.0]])
        transition, noise = transition_and_noise(1.5, 0.5)
        state = transition @ np.array([1.0, 2.0, 0.0, 0.0])
        prior = transition @ initial.covariance @ transition.T + noise
        observation = np.eye(2, 4)
        residual = np.array([2.0, 1.0]) - observation @ state
        innovation_cov = observation @ prior @ observation.T + measurement_noise
        gain = prior @ observation.T @ np.linalg.inv(innovation_cov)
        expected_state = state + gain @ residual
        identity_minus_gain = np.eye(4) - gain @ observation
        expected_cov = identity_minus_gain @ prior @ identity_minus_gain.T + gain @ measurement_noise @ gain.T

        result = tracker.update("crew", 1.5, fix(2.0, 1.0), measurement_covariance=measurement_noise)
        np.testing.assert_allclose((*result.position, *result.velocity), expected_state, atol=1e-12)
        np.testing.assert_allclose(result.covariance, expected_cov, atol=1e-12)
        self.assertAlmostEqual(result.innovation_distance_squared, float(residual @ np.linalg.solve(innovation_cov, residual)))

    def test_prediction_is_independent_of_ui_polling_frequency(self):
        single = KalmanTracker(self.config())
        frequent = KalmanTracker(self.config())
        single.update("crew", 0.0, fix(1.0))
        frequent.update("crew", 0.0, fix(1.0))
        for timestamp in (0.2, 0.7, 1.4, 2.0):
            frequent_result = frequent.predict("crew", timestamp)
        single_result = single.predict("crew", 2.0)
        np.testing.assert_allclose(frequent_result.covariance, single_result.covariance, atol=1e-12)
        self.assertEqual(frequent_result.position, single_result.position)

    def test_continuous_process_noise_is_consistent_across_missing_scan_intervals(self):
        single, frequent = KalmanTracker(self.config()), KalmanTracker(self.config())
        single.update("crew", 0.0, fix(1.0))
        frequent.update("crew", 0.0, fix(1.0))
        for timestamp in (0.2, 0.7, 1.4, 2.0):
            frequent_result = frequent.update("crew", timestamp, None)
        single_result = single.update("crew", 2.0, None)
        np.testing.assert_allclose(frequent_result.covariance, single_result.covariance, atol=1e-12)
        self.assertEqual(frequent_result.position, single_result.position)

    def test_long_irregular_sequence_remains_symmetric_and_positive_semidefinite(self):
        tracker = KalmanTracker()
        rng = Random(29071)
        timestamp = 0.0
        for index in range(500):
            timestamp += rng.uniform(0.1, 1.2)
            measurement = None if index % 7 == 0 else fix(
                0.5 * timestamp + rng.gauss(0.0, 1.5), rng.gauss(0.0, 1.5),
            )
            result = tracker.update("crew", timestamp, measurement)
            if result is not None:
                self.assertTrue(np.isfinite(result.covariance).all())
                np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1e-12)
                self.assertGreaterEqual(np.linalg.eigvalsh(result.covariance).min(), -1e-10)


class KalmanTrackingValidityTests(unittest.TestCase):
    def test_unbiased_noisy_walk_has_lower_rms_error_than_individual_fixes(self):
        # This tests the model in conditions that satisfy its noise assumptions;
        # the separate radio benchmark tests imperfect observation models.
        rng = Random(82019)
        tracker = KalmanTracker()
        raw_errors, filtered_errors = [], []
        for index in range(240):
            timestamp = index * 0.5
            truth = (0.6 * timestamp, 2.0)
            observed = fix(truth[0] + rng.gauss(0, 1.5), truth[1] + rng.gauss(0, 1.5))
            result = tracker.update("walk", timestamp, observed)
            if index >= 20:
                self.assertIsNotNone(result)
                raw_errors.append(math.dist(observed.position, truth))
                filtered_errors.append(math.dist(result.position, truth))
        rms = lambda errors: math.sqrt(sum(error ** 2 for error in errors) / len(errors))
        self.assertLess(rms(filtered_errors), 0.8 * rms(raw_errors))

    def test_stationary_bias_is_not_misrepresented_as_removed(self):
        tracker = KalmanTracker()
        # Ground truth is (0,0); every observation is consistently 2 m wrong.
        for index in range(40):
            result = tracker.update("bias", float(index), fix(2.0))
        self.assertAlmostEqual(result.position[0], 2.0)
        self.assertAlmostEqual(result.position[1], 0.0)
        self.assertGreaterEqual(np.linalg.eigvalsh(result.covariance[:2, :2]).min(), 0.5 ** 2 - 1e-12)

    def test_isolated_jump_is_rejected_without_preventing_next_valid_fix(self):
        tracker = KalmanTracker()
        for index in range(8):
            tracker.update("crew", float(index), fix(0.0))
        bad = tracker.update("crew", 8.0, fix(100.0, -100.0))
        self.assertEqual(bad.status, "outlier_rejected")
        self.assertLess(math.dist(bad.position, (0, 0)), 1e-8)
        self.assertEqual(bad.last_measurement_timestamp_s, 7.0)
        recovered = tracker.update("crew", 9.0, fix(0.1))
        self.assertEqual(recovered.status, "updated")
        self.assertLess(math.dist(recovered.position, (0, 0)), 0.2)

    def test_filter_follows_a_synthetic_room_boundary_transition_with_bounded_delay(self):
        # Two synthetic rooms meet at x=0. This is a motion test, not an assumed
        # doorway in the vessel layout and no room label reaches the tracker.
        rng = Random(3871)
        tracker = KalmanTracker()
        crossed_at = None
        truth_crossing = 10.0
        for index in range(51):
            timestamp = index * 0.5
            truth = (0.7 * (timestamp - truth_crossing), 0.0)
            result = tracker.update("transition", timestamp, fix(
                truth[0] + rng.gauss(0, 0.3), rng.gauss(0, 0.3),
            ))
            if timestamp >= truth_crossing and result.position[0] >= 0 and crossed_at is None:
                crossed_at = timestamp
        self.assertIsNotNone(crossed_at)
        self.assertLessEqual(crossed_at - truth_crossing, 2.0)


if __name__ == "__main__":
    unittest.main()
