"""Regression checks for measurement-only RSSI and legacy RTT estimators."""

import math
import unittest

import numpy as np

from crew_tracking.fingerprinting import (
    FingerprintCompartmentLocaliser, FingerprintEntry, FingerprintLocaliser,
)
from crew_tracking.models import SPEED_OF_LIGHT, AccessPoint, PathLossModel, Reading
from crew_tracking.positioning import (
    NearestAPLocaliser, RSSTrilaterationLocaliser,
    RTTTrilaterationLocaliser, WeightedCentroidLocaliser,
)


class BaselineLocaliserTests(unittest.TestCase):
    def setUp(self):
        self.aps = [
            AccessPoint(str(index), str(index), (x, y, 1.0), 2412)
            for index, (x, y) in enumerate(((0, 0), (10, 0), (0, 10), (10, 10)))
        ]
        self.target = (2.0, 3.0)
        self.scan = {}
        for ap in self.aps:
            distance = math.dist(ap.xy, self.target)
            self.scan[ap] = Reading(
                rssi_dbm=-40 - 20 * math.log10(distance),
                rtt_ms=1 + 2000 * distance / SPEED_OF_LIGHT,
            )

    def test_calibrated_rssi_and_legacy_rtt_recover_known_geometry(self):
        localisers = (
            RSSTrilaterationLocaliser(PathLossModel(path_loss_exponent=2.0)),
            RTTTrilaterationLocaliser(rtt_offset_ms=1.0),
        )
        for localiser in localisers:
            with self.subTest(localiser=type(localiser).__name__):
                estimate = localiser.locate(self.scan)
                self.assertIsNotNone(estimate)
                np.testing.assert_allclose(estimate.position, self.target, atol=1e-6)
                self.assertEqual(estimate.num_aps_used, 4)

    def test_rtt_uses_timing_even_when_rssi_is_missing(self):
        timing_scan = {
            ap: Reading(float("nan"), rtt_ms=reading.rtt_ms)
            for ap, reading in self.scan.items()
        }
        result = RTTTrilaterationLocaliser(rtt_offset_ms=1.0).locate(timing_scan)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.position, self.target, atol=1e-6)

    def test_missing_and_negative_corrected_rtt_do_not_become_zero_ranges(self):
        localiser = RTTTrilaterationLocaliser(rtt_offset_ms=1.0)
        scan = dict(self.scan)
        scan[self.aps[0]] = Reading(-50)
        scan[self.aps[1]] = Reading(-50, rtt_ms=0.5)
        self.assertIsNone(localiser.locate(scan))

    def test_collinear_anchors_are_not_reported_as_unique_positions(self):
        scan = {}
        for index in range(4):
            ap = AccessPoint(str(index), str(index), (index * 3.0, 0.0, 1.0), 2412)
            distance = math.dist(ap.xy, self.target)
            scan[ap] = Reading(-40 - 20 * math.log10(distance))
        self.assertIsNone(RSSTrilaterationLocaliser().locate(scan))

    def test_unheard_or_nonfinite_rssi_is_ignored(self):
        scan = {
            self.aps[0]: None,
            self.aps[1]: Reading(float("nan")),
            self.aps[2]: Reading(-60),
            self.aps[3]: Reading(float("inf")),
        }
        for localiser in (NearestAPLocaliser(), WeightedCentroidLocaliser()):
            with self.subTest(localiser=type(localiser).__name__):
                estimate = localiser.locate(scan)
                self.assertEqual(estimate.position, self.aps[2].xy)
                self.assertEqual(estimate.num_aps_used, 1)

    def test_explicit_bounds_constrain_range_fit(self):
        estimate = RSSTrilaterationLocaliser(
            PathLossModel(path_loss_exponent=2), bounds=((4, 4), (8, 8))
        ).locate(self.scan)
        self.assertIsNotNone(estimate)
        self.assertTrue(all(4 <= coordinate <= 8 for coordinate in estimate.position))

    def test_fingerprint_matches_reference_and_ignores_unknown_aps(self):
        entries = [FingerprintEntry(
            self.target, "engine room",
            np.array([self.scan[ap].rssi_dbm for ap in self.aps]),
        )]
        localiser = FingerprintLocaliser(entries, self.aps)
        classifier = FingerprintCompartmentLocaliser(entries, self.aps)
        estimate = localiser.locate(self.scan)
        self.assertEqual(estimate.position, self.target)
        self.assertEqual(classifier.classify(self.scan), "engine room")
        unknown = AccessPoint("unknown", "other", (1, 1, 1), 2412)
        self.assertIsNone(localiser.locate({unknown: Reading(-30)}))
        self.assertIsNone(classifier.classify({unknown: Reading(-30)}))

    def test_invalid_fingerprint_shape_is_rejected_before_matching(self):
        entries = [FingerprintEntry(self.target, "room", np.array([-50.0]))]
        with self.assertRaises(ValueError):
            FingerprintLocaliser(entries, self.aps)


if __name__ == "__main__":
    unittest.main()
