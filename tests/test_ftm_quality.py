"""Observable FTM diagnostics, including failures hidden by a robust fit."""

from dataclasses import dataclass
import math
from types import MappingProxyType
import unittest

import numpy as np

from crew_tracking.ftm import FTMConfig, FTMLocaliser, distance_m_to_rtt_ns
from crew_tracking.models import AccessPoint, FTMBurst, FTMQuality, FTMSample, Reading


def burst_for_distance(distance, errors=(0.0,) * 5):
    return FTMBurst(tuple(
        FTMSample(distance_m_to_rtt_ns(distance + error)) for error in errors
    ))


class FTMQualityTests(unittest.TestCase):
    def setUp(self):
        self.anchors = tuple(
            AccessPoint(str(i), str(i), (x, y, 1.0), 5180)
            for i, (x, y) in enumerate(((0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)))
        )
        self.position = (5.0, 5.0)

    def scan(self, *, position=None, errors=(0.0,) * 5, biases=None):
        position = self.position if position is None else position
        biases = {} if biases is None else biases
        return {
            ap: Reading(-60.0, ftm_burst=burst_for_distance(
                math.dist((*position, 1.0), ap.position) + biases.get(ap.bssid, 0.0),
                errors,
            ))
            for ap in self.anchors
        }

    def quality(self, estimate):
        self.assertIsNotNone(estimate)
        self.assertIsInstance(estimate.quality, FTMQuality)
        return estimate.quality

    def test_clean_square_geometry_has_expected_dimensionless_dilution(self):
        estimate = FTMLocaliser().locate(self.scan())
        quality = self.quality(estimate)

        np.testing.assert_allclose(estimate.position, self.position, atol=1e-8)
        self.assertLess(quality.residual_rms_m, 1e-8)
        # Four diagonal unit vectors produce G.T @ G == 2 I.
        self.assertAlmostEqual(quality.geometry_dilution, 1.0)
        self.assertAlmostEqual(quality.range_std_rms_m, FTMConfig().range_std_floor_m)
        self.assertEqual(quality.retained_fraction, 1.0)
        self.assertEqual(quality.min_samples_per_ap, 5)
        self.assertFalse(quality.at_bound)

    def test_height_difference_degrades_horizontal_geometry_without_changing_noise(self):
        scan = {}
        for ap in self.anchors:
            elevated = AccessPoint(ap.name, ap.bssid, (*ap.xy, 11.0), 5180)
            scan[elevated] = Reading(-60.0, ftm_burst=burst_for_distance(
                math.dist((*self.position, 1.0), elevated.position)
            ))
        quality = self.quality(FTMLocaliser().locate(scan))

        # Horizontal derivatives shrink by sqrt(50 / 150).
        self.assertAlmostEqual(quality.geometry_dilution, math.sqrt(3.0))
        self.assertAlmostEqual(quality.range_std_rms_m, FTMConfig().range_std_floor_m)

    def test_persistent_bias_is_visible_in_unclipped_metre_residuals(self):
        localiser = FTMLocaliser()
        clean = self.quality(localiser.locate(self.scan()))
        scan = self.scan(biases={self.anchors[-1].bssid: 8.0})
        estimate = localiser.locate(scan)
        quality = self.quality(estimate)
        residuals = [
            math.dist((*estimate.position, 1.0), ap.position)
            - math.dist((*self.position, 1.0), ap.position)
            - (8.0 if ap == self.anchors[-1] else 0.0)
            for ap in self.anchors
        ]

        self.assertGreater(quality.residual_rms_m, clean.residual_rms_m + 1.0)
        self.assertAlmostEqual(quality.residual_rms_m, math.sqrt(sum(r * r for r in residuals) / 4))
        # Low burst variation must not conceal a mutually inconsistent range.
        self.assertAlmostEqual(quality.range_std_rms_m, clean.range_std_rms_m)

    def test_noisy_bursts_increase_reported_range_dispersion(self):
        localiser = FTMLocaliser()
        quiet = self.quality(localiser.locate(self.scan()))
        noisy = self.quality(localiser.locate(self.scan(errors=(-1.0, -0.5, 0.0, 0.5, 1.0))))

        self.assertGreater(noisy.range_std_rms_m, quiet.range_std_rms_m)
        self.assertAlmostEqual(noisy.retained_fraction, 1.0)

    def test_failed_malformed_and_outlier_exchanges_reduce_retained_fraction(self):
        scan = self.scan(errors=(0.0,) * 4)
        for reading in scan.values():
            reading.ftm_burst = FTMBurst(reading.ftm_burst.samples + (
                FTMSample(None, successful=False),
                FTMSample(float("nan")),
                FTMSample(distance_m_to_rtt_ns(100.0)),
            ))
        # An entirely unusable AP is not one of the accepted range anchors.
        scan[AccessPoint("failed", "failed", (4.0, 4.0, 1.0), 5180)] = Reading(
            -50.0, ftm_burst=FTMBurst((FTMSample(None, successful=False),) * 20)
        )
        quality = self.quality(FTMLocaliser().locate(scan))

        self.assertAlmostEqual(quality.retained_fraction, 4 / 7)
        self.assertEqual(quality.min_samples_per_ap, 4)

    def test_duplicate_radio_is_not_counted_twice_in_sample_diagnostics(self):
        scan = self.scan()
        first = self.anchors[0]
        duplicate = AccessPoint("alias", first.bssid.upper(), first.position, 5180)
        scan[duplicate] = Reading(-60.0, ftm_burst=FTMBurst(
            scan[first].ftm_burst.samples + (FTMSample(None, successful=False),) * 20
        ))
        estimate = FTMLocaliser().locate(scan)
        quality = self.quality(estimate)

        self.assertEqual(estimate.num_aps_used, 4)
        self.assertEqual(quality.retained_fraction, 1.0)
        self.assertEqual(quality.min_samples_per_ap, 5)

    def test_constrained_solution_is_flagged_at_active_bound(self):
        estimate = FTMLocaliser(bounds=((6.0, 0.0), (10.0, 10.0))).locate(self.scan())
        quality = self.quality(estimate)

        self.assertAlmostEqual(estimate.position[0], 6.0, places=6)
        self.assertTrue(quality.at_bound)
        self.assertGreater(quality.residual_rms_m, 0.0)

    def test_diagnostics_accept_read_only_hardware_observations_without_truth(self):
        @dataclass(frozen=True)
        class HardwareObservation:
            ftm_burst: FTMBurst

            def __getattr__(self, name):
                raise AssertionError(f"Not an observable FTM field: {name}")

        scan = MappingProxyType({
            ap: HardwareObservation(reading.ftm_burst) for ap, reading in self.scan().items()
        })
        quality = self.quality(FTMLocaliser().locate(scan))

        self.assertLess(quality.residual_rms_m, 1e-8)
        self.assertAlmostEqual(quality.geometry_dilution, 1.0)


if __name__ == "__main__":
    unittest.main()
