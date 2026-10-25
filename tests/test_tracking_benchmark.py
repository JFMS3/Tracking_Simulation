"""Validate paired temporal evaluation and expose the limits of smoothing."""

from contextlib import redirect_stdout
from io import StringIO
import math
from random import Random
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import LineString

from crew_tracking.models import AccessPoint, LocalisationEstimate, Reading
from crew_tracking.tracking import KalmanTracker
from simulator.environment import Compartment, ShipEnvironment
from simulator.evaluation import create_evaluation_session
from simulator.network import Network
from simulator.tracking import (
    JourneyPoint, TrackingFix, evaluate_tracking_fixes,
    generate_journeys, report_tracking_benchmark,
)


class TrackingBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.environment = ShipEnvironment(compartments=[
            Compartment("Room", [(0, 0), (6, 0), (6, 4), (0, 4)]),
        ])
        self.network = Network(
            self.environment,
            access_points=[AccessPoint("AP", "a", (3, 2, 1), 2412)],
        )

    def report(self, localiser, *, profile="matched", seed=19):
        with redirect_stdout(StringIO()) as output:
            result = report_tracking_benchmark(
                localiser, self.network,
                create_evaluation_session(self.network, profile, seed), seed,
            )
        return result, output.getvalue()

    def test_journeys_are_continuous_timestamped_stops_walks_and_turns(self):
        self.environment.compartments.append(
            Compartment("Other room", [(20, 0), (24, 0), (24, 4), (20, 4)])
        )
        points = generate_journeys(self.environment)
        self.assertEqual(points, generate_journeys(self.environment))
        self.assertEqual({point.phase for point in points}, {"stationary", "walking", "turn"})
        devices = {point.device_id for point in points}
        self.assertEqual(len(devices), 2)
        for device in devices:
            route = [point for point in points if point.device_id == device]
            self.assertGreater(route[0].timestamp_s, 0)
            self.assertGreater(route[-1].elapsed_s, 75)
            self.assertEqual(len({point.compartment_name for point in route}), 1)
            polygon = next(comp.geometry for comp in self.environment.compartments
                           if comp.name == route[0].compartment_name)
            intervals = []
            for first, second in zip(route, route[1:]):
                dt = second.timestamp_s - first.timestamp_s
                intervals.append(round(dt, 4))
                self.assertGreater(dt, 0)
                self.assertLessEqual(math.dist(first.truth, second.truth), 0.75 * dt + 1e-10)
                self.assertTrue(polygon.covers(LineString((first.truth, second.truth))))
            self.assertGreater(len(set(intervals)), 1)

    def test_scan_and_localise_once_then_replay_identical_delivered_fixes(self):
        class Recorder:
            def __init__(self):
                self.scans = []
                self.fixes = []

            def locate(self, scan):
                self.scans.append(scan)
                fix = LocalisationEstimate((2.0, 1.0)) if len(self.scans) % 9 else None
                self.fixes.append(fix)
                return fix

        recorder = Recorder()
        result, output = self.report(recorder, profile="deployment")
        radio, outage = result.scenarios
        self.assertEqual(len(recorder.scans), len(radio.samples))
        self.assertEqual(len(radio.samples), len(outage.samples))
        self.assertIn("common fixes", output)
        self.assertIn("stable bias can survive", output)
        for scan in recorder.scans:
            self.assertTrue(all(reading is None or type(reading) is Reading
                                for reading in scan.values()))
        for index, (normal, stressed) in enumerate(zip(radio.samples, outage.samples)):
            self.assertIs(normal.raw, recorder.fixes[index])
            if stressed.forced_outage:
                self.assertIsNone(stressed.raw)
            else:
                self.assertIs(stressed.raw, normal.raw)
        self.assertEqual(radio.paired_count,
                         sum(sample.raw is not None and sample.filtered is not None
                             for sample in radio.samples))
        errors = [math.dist(sample.truth, sample.raw.position)
                  for sample in radio.samples
                  if sample.raw is not None and sample.filtered is not None]
        self.assertAlmostEqual(radio.paired_raw_mean_m, float(np.mean(errors)))
        self.assertAlmostEqual(radio.paired_raw_p90_m, float(np.percentile(errors, 90)))

    def test_tracker_only_receives_identity_capture_time_and_observation(self):
        points = generate_journeys(self.environment)[:6]
        measurements = [LocalisationEstimate((2, 1), covariance=np.eye(2) * 1e-12)
                        for _ in points]
        real_tracker = KalmanTracker()
        with patch("simulator.tracking.KalmanTracker", return_value=real_tracker), \
                patch.object(real_tracker, "update", wraps=real_tracker.update) as update:
            evaluate_tracking_fixes(
                tuple(TrackingFix(point, measurement)
                      for point, measurement in zip(points, measurements)),
                self.environment,
            )
        for call, point, measurement in zip(update.call_args_list, points, measurements):
            self.assertEqual(call.args[:2], (point.device_id, point.timestamp_s))
            self.assertIs(call.args[2], measurement)
            self.assertEqual(len(call.args), 3)
            self.assertEqual(call.kwargs, {})

    def test_empty_observations_do_not_create_tracks_or_hide_missing_denominators(self):
        class Missing:
            def locate(self, scan):
                return None

        result, output = self.report(Missing())
        for scenario in result.scenarios:
            self.assertEqual(scenario.paired_count, 0)
            self.assertIsNone(scenario.paired_raw_mean_m)
            self.assertIsNone(scenario.paired_kalman_p90_m)
            self.assertEqual(scenario.kalman.available_count, 0)
            self.assertEqual(scenario.kalman.compartment_accuracy, 0)
            self.assertEqual(scenario.kalman.prediction_count, 0)
            self.assertEqual(scenario.kalman.recovery_count, 0)
        self.assertIn("0.0%", output)

    def test_short_gap_predicts_long_gap_expires_and_reacquires(self):
        # Perfect observations are an explicit synthetic filter test. They are
        # never substituted into the end-to-end radio benchmark.
        fixes = tuple(TrackingFix(point, LocalisationEstimate(point.truth))
                      for point in generate_journeys(self.environment))
        result = evaluate_tracking_fixes(
            fixes, self.environment, delivery_outages=((20, 22), (42, 55)),
        )
        short = [sample for sample in result.samples if 20 <= sample.elapsed_s < 22]
        long_stale = [sample for sample in result.samples if 46 <= sample.elapsed_s < 55]
        self.assertTrue(short)
        self.assertTrue(all(sample.filtered is not None for sample in short))
        self.assertTrue(long_stale)
        self.assertTrue(all(sample.filtered is None for sample in long_stale))
        self.assertGreater(result.kalman.prediction_count, 0)
        self.assertGreater(result.kalman.stale_count, 0)
        self.assertEqual(result.kalman.recovery_count, 2)
        self.assertLess(result.kalman.recovery_delay_s, 1.0)
        self.assertGreater(result.kalman.availability, result.raw.availability)
        # Smoothing perfect positions introduces motion error: validity does
        # not depend on asserting that filtering must always improve accuracy.
        self.assertAlmostEqual(result.paired_raw_mean_m, 0)
        self.assertGreater(result.paired_kalman_mean_m, 0)

    def test_persistent_bias_remains_and_is_not_hidden_by_zero_jitter(self):
        fixes = tuple(TrackingFix(
            JourneyPoint("crew", 1000 + index * 0.7, index * 0.7,
                         (2, 2), "Room", "stationary", index * 0.7, 0),
            LocalisationEstimate((4, 2)),
        ) for index in range(80))
        result = evaluate_tracking_fixes(fixes, self.environment)
        self.assertAlmostEqual(result.raw.stationary_jitter_m, 0)
        self.assertAlmostEqual(result.kalman.stationary_jitter_m, 0)
        self.assertAlmostEqual(result.raw.stationary_error_m, 2)
        self.assertAlmostEqual(result.kalman.stationary_error_m, 2)

    def test_prediction_only_error_is_scored_during_moving_outage(self):
        fixes = tuple(TrackingFix(
            JourneyPoint("crew", 1000 + index * 0.5, index * 0.5,
                         (0.2 * index, 2), "Room", "walking", index * 0.5),
            LocalisationEstimate((0.2 * index, 2)),
        ) for index in range(30))
        result = evaluate_tracking_fixes(fixes, self.environment, delivery_outages=((8, 10),))
        gap_errors = [math.dist(sample.truth, sample.filtered.position)
                      for sample in result.samples if sample.raw is None and sample.filtered is not None]
        self.assertEqual(result.kalman.prediction_only_count, 4)
        self.assertEqual(result.raw.prediction_only_count, 0)
        self.assertAlmostEqual(result.kalman.prediction_only_mean_m, float(np.mean(gap_errors)))
        self.assertAlmostEqual(result.kalman.prediction_only_p90_m, float(np.percentile(gap_errors, 90)))
        self.assertLess(max(gap_errors), 0.2)

    def test_correlated_errors_and_isolated_outlier_are_distinct_from_white_noise(self):
        rng = Random(8321)
        correlated = 0.0
        fixes = []
        for index in range(100):
            correlated = 0.95 * correlated + rng.gauss(0, 0.05)
            measured = (3.5 + correlated, 2 + rng.gauss(0, 0.05))
            if index == 60:
                measured = (40, -20)
            fixes.append(TrackingFix(
                JourneyPoint("crew", 1000 + index * 0.7, index * 0.7,
                             (2, 2), "Room", "stationary", index * 0.7, 0),
                LocalisationEstimate(measured),
            ))
        result = evaluate_tracking_fixes(fixes, self.environment)
        outlier = result.samples[60]
        self.assertEqual(outlier.filtered.status, "outlier_rejected")
        self.assertLess(math.dist(outlier.filtered.position, outlier.truth), 3)
        self.assertGreater(result.kalman.stationary_error_m, 1)
        self.assertLess(result.kalman.stationary_jitter_m, result.raw.stationary_jitter_m)
        self.assertEqual(result.samples[61].filtered.measurement_age_s, 0)

    def test_reproducible_radio_run_with_identical_fresh_sessions(self):
        class RSSIOnly:
            def locate(self, scan):
                values = [reading.rssi_dbm for reading in scan.values() if reading is not None]
                if not values:
                    return None
                return LocalisationEstimate((3 + 0.05 * (values[0] + 45), 2))

        first, _ = self.report(RSSIOnly(), profile="deployment", seed=918)
        second, _ = self.report(RSSIOnly(), profile="deployment", seed=918)
        for a, b in zip(first.scenarios, second.scenarios):
            self.assertEqual(a.raw, b.raw)
            self.assertEqual(a.kalman, b.kalman)
            self.assertEqual(a.paired_kalman_mean_m, b.paired_kalman_mean_m)
            for sa, sb in zip(a.samples, b.samples):
                self.assertEqual(sa.raw, sb.raw)
                self.assertEqual(sa.filtered is None, sb.filtered is None)
                if sa.filtered is not None:
                    self.assertEqual(sa.filtered.position, sb.filtered.position)


if __name__ == "__main__":
    unittest.main()
