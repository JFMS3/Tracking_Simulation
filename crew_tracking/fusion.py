"""Measurement-only fusion of the project's complementary localisers.

The fusion localiser deliberately knows nothing about the simulator or receiver
ground truth.  It combines estimates produced from one shared scan so it can be
used by both a real integration and the simulator's paired benchmark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
import sys

import numpy as np

from .models import (
    FingerprintQuality, FTMQuality, FusionQuality,
    LocalisationEstimate, Localiser, Scan,
)


class FixedWeightFusionLocaliser:
    """Combine RSSI, fingerprint and FTM estimates using configured weights.

    The defaults preserve the project's original end-to-end locating heuristic.
    When FTM is available it receives 30% of the configured weight.  When it is
    unavailable, fingerprinting and the power centroid receive larger weights.
    If any other component misses, the remaining weights are renormalised.

    These weights are engineering defaults rather than calibrated probabilities,
    so the result intentionally does not claim a statistical covariance.
    """

    def __init__(
        self,
        nearest_ap: Localiser,
        weighted_power: Localiser,
        fingerprint: Localiser,
        ftm: Localiser,
        *,
        weights_with_ftm: tuple[float, float, float, float] = (0.1, 0.2, 0.4, 0.3),
        weights_without_ftm: tuple[float, float, float] = (0.1, 0.3, 0.6),
    ) -> None:
        self.localisers = (nearest_ap, weighted_power, fingerprint, ftm)
        self.weights_with_ftm = self._validate_weights(weights_with_ftm, 4)
        self.weights_without_ftm = self._validate_weights(weights_without_ftm, 3)

    @staticmethod
    def _validate_weights(weights: tuple[float, ...], expected: int) -> np.ndarray:
        values = np.asarray(weights, dtype=float)
        if values.shape != (expected,):
            raise ValueError(f"Expected {expected} fusion weights")
        if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
            raise ValueError("Fusion weights must be finite, non-negative, and not all zero")
        return values.copy()

    @staticmethod
    def _usable(estimate: LocalisationEstimate | None) -> bool:
        return (
            estimate is not None
            and len(estimate.position) == 2
            and all(math.isfinite(value) for value in estimate.position)
        )

    def locate(self, scan: Scan) -> LocalisationEstimate | None:
        estimates = tuple(localiser.locate(scan) for localiser in self.localisers)
        ftm_available = self._usable(estimates[3])

        if ftm_available:
            selected = estimates
            configured_weights = self.weights_with_ftm
        else:
            selected = estimates[:3]
            configured_weights = self.weights_without_ftm

        available = [
            (estimate, float(weight))
            for estimate, weight in zip(selected, configured_weights)
            if self._usable(estimate) and weight > 0
        ]
        if not available:
            return None

        coordinates = np.asarray([estimate.position for estimate, _ in available])
        weights = np.asarray([weight for _, weight in available])
        position = np.average(coordinates, axis=0, weights=weights)

        return LocalisationEstimate(
            position=(float(position[0]), float(position[1])),
            num_aps_used=max(estimate.num_aps_used for estimate, _ in available),
        )


@dataclass(frozen=True)
class FusionConfig:
    """Conservative engineering scales requiring deployment validation.

    These defaults are not trained against simulation truth. They define risk
    scores for weighting, not standard deviations or claimed accuracy bounds.
    Persistent map/device drift and coherent NLOS bias can evade all indicators.
    """

    fingerprint_floor_m: float = 0.75
    ftm_floor_m: float = 1.5
    fingerprint_mismatch_db: float = 6.0
    fingerprint_mismatch_penalty_m: float = 1.0
    unknown_quality_scale_m: float = 5.0
    disagreement_multiplier: float = 3.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


class FusionLocaliser:
    """Quality-aware fingerprint/FTM fusion using observable evidence only.

    RSSI nearest-AP and centroid are correlated with fingerprinting, so they
    contribute only when both primary methods fail. The existing covariance
    matrices are deliberately not interpreted as calibrated uncertainty.
    ``estimate.quality`` explains the weights or disagreement/fallback decision.
    """

    def __init__(
        self,
        nearest_ap: Localiser,
        weighted_power: Localiser,
        fingerprint: Localiser,
        ftm: Localiser,
        *,
        config: FusionConfig = FusionConfig(),
    ) -> None:
        self.nearest_ap = nearest_ap
        self.weighted_power = weighted_power
        self.fingerprint = fingerprint
        self.ftm = ftm
        self.config = config

    @staticmethod
    def _usable(estimate: LocalisationEstimate | None) -> bool:
        if estimate is None:
            return False
        position = np.asarray(estimate.position, dtype=float)
        return position.shape == (2,) and bool(np.isfinite(position).all())

    @staticmethod
    def _valid_count(value, minimum: int) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum

    @staticmethod
    def _finite_scale(value: float) -> float:
        # Overflow from corrupt/extreme but finite diagnostics represents very
        # poor quality, not missing metadata. Keep weighting numerically safe.
        return min(value, sys.float_info.max)

    def _fingerprint_scale(self, estimate: LocalisationEstimate) -> float:
        quality = estimate.quality
        if not isinstance(quality, FingerprintQuality):
            return self.config.unknown_quality_scale_m
        values = (quality.match_rms_db, quality.neighbour_spread_m, quality.ambiguity_spread_m)
        if (not all(math.isfinite(value) and value >= 0 for value in values)
                or not 0 < quality.observed_ap_fraction <= 1
                or not self._valid_count(estimate.num_aps_used, 1)):
            return self.config.unknown_quality_scale_m
        mismatch = (self.config.fingerprint_mismatch_penalty_m
                    * quality.match_rms_db / self.config.fingerprint_mismatch_db)
        scale = math.hypot(
            self.config.fingerprint_floor_m, quality.neighbour_spread_m,
            quality.ambiguity_spread_m, mismatch,
        )
        # A very close match with one AP remains weak evidence of XY position.
        support = quality.observed_ap_fraction * min(estimate.num_aps_used / 3, 1.0)
        return self._finite_scale(scale / math.sqrt(support))

    def _ftm_scale(self, estimate: LocalisationEstimate) -> float:
        quality = estimate.quality
        if not isinstance(quality, FTMQuality):
            return self.config.unknown_quality_scale_m
        if (not all(math.isfinite(value) and value >= 0 for value in (
                quality.residual_rms_m, quality.range_std_rms_m))
                or not math.isfinite(quality.geometry_dilution) or quality.geometry_dilution <= 0
                or not 0 < quality.retained_fraction <= 1
                or not self._valid_count(quality.min_samples_per_ap, 2)
                or not self._valid_count(estimate.num_aps_used, 3)):
            return self.config.unknown_quality_scale_m
        range_scale = math.hypot(quality.residual_rms_m, quality.range_std_rms_m)
        scale = math.hypot(self.config.ftm_floor_m, range_scale * quality.geometry_dilution)
        scale /= math.sqrt(quality.retained_fraction)
        # Three radios can fit biased ranges without exposing the bias. Short
        # bursts provide little evidence of stability; large bursts cannot
        # remove the persistent error floor.
        if estimate.num_aps_used == 3:
            scale *= 1.5
        scale *= math.sqrt(max(1.0, 6 / quality.min_samples_per_ap))
        if quality.at_bound:
            scale *= 2.0
        return self._finite_scale(scale)

    @staticmethod
    def _result(position, estimates, quality: FusionQuality) -> LocalisationEstimate:
        return LocalisationEstimate(
            position=(float(position[0]), float(position[1])),
            # Component anchor sets overlap; their counts must not be added.
            num_aps_used=max(
                estimate.num_aps_used if FusionLocaliser._valid_count(estimate.num_aps_used, 0) else 0
                for estimate in estimates
            ),
            quality=quality,
        )

    def locate(self, scan: Scan) -> LocalisationEstimate | None:
        fingerprint = self.fingerprint.locate(scan)
        ftm = self.ftm.locate(scan)
        fingerprint = fingerprint if self._usable(fingerprint) else None
        ftm = ftm if self._usable(ftm) else None
        if fingerprint is None and ftm is None:
            # Lazy fallback avoids counting the same RSSI evidence three times.
            for name, localiser in (("centroid_fallback", self.weighted_power),
                                    ("nearest_ap_fallback", self.nearest_ap)):
                estimate = localiser.locate(scan)
                if self._usable(estimate):
                    return self._result(estimate.position, [estimate], FusionQuality(name))
            return None
        if ftm is None:
            return self._result(fingerprint.position, [fingerprint], FusionQuality(
                "fingerprint_only", fingerprint_weight=1.0,
                fingerprint_scale_m=self._fingerprint_scale(fingerprint),
            ))
        if fingerprint is None:
            return self._result(ftm.position, [ftm], FusionQuality(
                "ftm_only", ftm_weight=1.0, ftm_scale_m=self._ftm_scale(ftm),
            ))

        fingerprint_scale = self._fingerprint_scale(fingerprint)
        ftm_scale = self._ftm_scale(ftm)
        separation = math.dist(fingerprint.position, ftm.position)
        # Relative inverse-square scores avoid overflow for tiny weights. These
        # scores are not independent Gaussian likelihoods; no fused covariance
        # or improvement guarantee is reported.
        minimum = min(fingerprint_scale, ftm_scale)
        weights = np.square([minimum / fingerprint_scale, minimum / ftm_scale])
        weights /= weights.sum()
        decision = "blended"
        if separation > self.config.disagreement_multiplier * math.hypot(fingerprint_scale, ftm_scale):
            weights = np.array([1.0, 0.0]) if fingerprint_scale <= ftm_scale else np.array([0.0, 1.0])
            decision = "disagreement_fingerprint" if weights[0] else "disagreement_ftm"
        position = np.average([fingerprint.position, ftm.position], axis=0, weights=weights)
        contributors = [estimate for estimate, weight in zip((fingerprint, ftm), weights) if weight > 0]
        return self._result(position, contributors, FusionQuality(
            decision, float(weights[0]), float(weights[1]),
            fingerprint_scale, ftm_scale, separation,
        ))
