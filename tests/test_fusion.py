"""Checks for the end-to-end fusion localiser."""

import unittest

import numpy as np

from crew_tracking.fusion import FixedWeightFusionLocaliser, FusionConfig, FusionLocaliser
from crew_tracking.models import (
    FingerprintQuality, FTMQuality, LocalisationEstimate,
)


class StubLocaliser:
    def __init__(self, estimate):
        self.estimate = estimate
        self.scans = []

    def locate(self, scan):
        self.scans.append(scan)
        return self.estimate


def estimate(x, y=0.0, aps=1):
    return LocalisationEstimate((x, y), num_aps_used=aps)


class FixedWeightFusionLocaliserTests(unittest.TestCase):
    def make_fusion(self, nearest, power, fingerprint, ftm):
        components = [StubLocaliser(value) for value in (nearest, power, fingerprint, ftm)]
        return FixedWeightFusionLocaliser(*components), components

    def test_original_weights_are_used_when_ftm_is_available(self):
        fusion, components = self.make_fusion(
            estimate(0, aps=1), estimate(10, aps=2),
            estimate(20, aps=3), estimate(30, aps=4),
        )
        scan = {}

        result = fusion.locate(scan)

        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.position, (19.0, 0.0))
        self.assertEqual(result.num_aps_used, 4)
        self.assertTrue(all(component.scans == [scan] for component in components))

    def test_non_ftm_weights_are_used_when_ftm_misses(self):
        fusion, _ = self.make_fusion(estimate(0), estimate(10), estimate(20), None)

        result = fusion.locate({})

        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.position, (15.0, 0.0))

    def test_remaining_weights_are_renormalised_when_a_component_misses(self):
        fusion, _ = self.make_fusion(estimate(0), None, None, estimate(30))

        result = fusion.locate({})

        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.position, (22.5, 0.0))

    def test_no_component_estimate_returns_none(self):
        fusion, _ = self.make_fusion(None, None, None, None)
        self.assertIsNone(fusion.locate({}))

    def test_invalid_weights_are_rejected(self):
        components = [StubLocaliser(None) for _ in range(4)]
        for weights in ((0.0, 0.0, 0.0, 0.0), (0.1, -0.1, 0.5, 0.5)):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    FixedWeightFusionLocaliser(*components, weights_with_ftm=weights)


class QualityAwareFusionTests(unittest.TestCase):
    def make_fusion(self, fingerprint, ftm, *, centroid=None, nearest=None, config=FusionConfig()):
        components = [StubLocaliser(value) for value in (nearest, centroid, fingerprint, ftm)]
        return FusionLocaliser(*components, config=config), components

    def fingerprint(self, position=(0.0, 0.0), *, rms=1.0, spread=0.2, ambiguity=0.3, aps=4):
        return LocalisationEstimate(
            position, num_aps_used=aps,
            quality=FingerprintQuality(rms, spread, ambiguity, aps / 4),
        )

    def ftm(self, position=(2.0, 0.0), *, residual=0.1, geometry=1.0,
            std=0.25, retained=1.0, samples=8, bound=False, aps=4):
        return LocalisationEstimate(
            position, num_aps_used=aps,
            quality=FTMQuality(residual, geometry, std, retained, samples, bound),
        )

    def test_inconsistent_ftm_has_less_influence_than_clean_ftm(self):
        good, _ = self.make_fusion(self.fingerprint(), self.ftm())
        poor, _ = self.make_fusion(self.fingerprint(), self.ftm(residual=4.0))
        good_result, poor_result = good.locate({}), poor.locate({})
        self.assertLess(poor_result.quality.ftm_weight, good_result.quality.ftm_weight)
        self.assertLess(poor_result.position[0], good_result.position[0])

    def test_weak_geometry_failures_short_bursts_and_bounds_reduce_ftm_influence(self):
        base, _ = self.make_fusion(self.fingerprint(), self.ftm())
        base_weight = base.locate({}).quality.ftm_weight
        for overrides in ({"geometry": 10.0}, {"retained": 0.4}, {"samples": 3},
                          {"bound": True}, {"std": 3.0}, {"aps": 3}):
            with self.subTest(overrides=overrides):
                fusion, _ = self.make_fusion(self.fingerprint(), self.ftm(**overrides))
                self.assertLess(fusion.locate({}).quality.ftm_weight, base_weight)

    def test_mismatch_ambiguity_and_sparse_rssi_reduce_fingerprint_influence(self):
        base, _ = self.make_fusion(self.fingerprint(), self.ftm())
        base_weight = base.locate({}).quality.fingerprint_weight
        for overrides in ({"rms": 15.0}, {"spread": 3.0}, {"ambiguity": 4.0}, {"aps": 1}):
            with self.subTest(overrides=overrides):
                fusion, _ = self.make_fusion(self.fingerprint(**overrides), self.ftm())
                self.assertLess(fusion.locate({}).quality.fingerprint_weight, base_weight)

    def test_fallbacks_do_not_vote_alongside_primary_estimators(self):
        fusion, components = self.make_fusion(
            self.fingerprint(), self.ftm(), centroid=estimate(100), nearest=estimate(-100),
        )
        scan = {}
        result = fusion.locate(scan)
        self.assertEqual(components[0].scans, [])
        self.assertEqual(components[1].scans, [])
        self.assertIs(components[2].scans[0], scan)
        self.assertIs(components[3].scans[0], scan)
        self.assertTrue(0 < result.position[0] < 2)
        self.assertEqual(result.num_aps_used, 4)  # Not the sum of overlapping AP sets.
        self.assertIsNone(result.covariance)

    def test_large_disagreement_selects_lower_risk_position_instead_of_midpoint(self):
        fusion, _ = self.make_fusion(self.fingerprint(), self.ftm(position=(30.0, 0.0)))
        result = fusion.locate({})
        self.assertEqual(result.position, (0.0, 0.0))
        self.assertEqual(result.quality.decision, "disagreement_fingerprint")
        self.assertEqual(result.quality.ftm_weight, 0)

    def test_large_disagreement_can_select_ftm_when_fingerprint_is_ambiguous(self):
        fusion, _ = self.make_fusion(
            self.fingerprint(ambiguity=3.0), self.ftm(position=(30.0, 0.0)),
        )
        result = fusion.locate({})
        self.assertEqual(result.position, (30.0, 0.0))
        self.assertEqual(result.quality.decision, "disagreement_ftm")

    def test_primary_method_outage_preserves_available_position(self):
        for fp, ftm, decision in ((self.fingerprint(), None, "fingerprint_only"),
                                  (None, self.ftm(), "ftm_only")):
            with self.subTest(decision=decision):
                fusion, _ = self.make_fusion(fp, ftm)
                result = fusion.locate({})
                self.assertEqual(result.position, (fp or ftm).position)
                self.assertEqual(result.quality.decision, decision)

    def test_only_fallbacks_available_choose_centroid_then_nearest(self):
        fusion, components = self.make_fusion(None, None, centroid=estimate(4), nearest=estimate(8))
        self.assertEqual(fusion.locate({}).position, (4.0, 0.0))
        self.assertEqual(components[0].scans, [])
        fusion, _ = self.make_fusion(None, None, nearest=estimate(8))
        self.assertEqual(fusion.locate({}).quality.decision, "nearest_ap_fallback")
        fusion, _ = self.make_fusion(None, None)
        self.assertIsNone(fusion.locate({}))

    def test_nonfinite_estimate_uses_other_primary(self):
        fusion, _ = self.make_fusion(estimate(float("nan")), self.ftm())
        self.assertEqual(fusion.locate({}).quality.decision, "ftm_only")

    def test_missing_diagnostics_never_imply_perfect_confidence_from_covariance(self):
        fp = estimate(0, aps=4)
        ftm = estimate(2, aps=4)
        fp.covariance = np.zeros((2, 2))
        ftm.covariance = np.eye(2) * 1e6
        fusion, _ = self.make_fusion(fp, ftm)
        result = fusion.locate({})
        self.assertEqual(result.quality.fingerprint_scale_m, 5.0)
        self.assertEqual(result.quality.ftm_scale_m, 5.0)
        self.assertEqual(result.quality.fingerprint_weight, 0.5)

    def test_invalid_diagnostics_are_treated_as_unknown(self):
        fp = self.fingerprint(rms=float("nan"))
        ftm = self.ftm(retained=0.0)
        fusion, _ = self.make_fusion(fp, ftm)
        result = fusion.locate({})
        self.assertEqual(result.quality.fingerprint_scale_m, 5.0)
        self.assertEqual(result.quality.ftm_scale_m, 5.0)

    def test_invalid_configuration_fails_before_localising(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    FusionConfig(ftm_floor_m=value)

    def test_nonintegral_or_nonfinite_counts_cannot_claim_good_quality(self):
        for count in (float("nan"), float("inf"), 3.5, True, 0):
            with self.subTest(count=count):
                fp, ftm = self.fingerprint(), self.ftm(samples=count)
                fp.num_aps_used = count
                fusion, _ = self.make_fusion(fp, ftm)
                result = fusion.locate({})
                self.assertEqual(result.quality.fingerprint_scale_m, 5.0)
                self.assertEqual(result.quality.ftm_scale_m, 5.0)

    def test_extreme_finite_diagnostics_cannot_produce_nan_weights_or_crash(self):
        fusion, _ = self.make_fusion(
            self.fingerprint(spread=1e308, ambiguity=1e308, aps=1),
            self.ftm(residual=1e308, geometry=1e308),
        )
        result = fusion.locate({})
        self.assertTrue(np.isfinite(result.position).all())
        self.assertAlmostEqual(result.quality.fingerprint_weight + result.quality.ftm_weight, 1)


if __name__ == "__main__":
    unittest.main()
