"""RSSI and legacy RTT baselines with no dependency on simulated truth.

RSSI ranges invert a calibrated log-distance propagation model. These
baselines fit horizontal ranges; use ``FTMLocaliser`` for FTM ranging and
explicit access-point/receiver height geometry.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from .models import (
    SPEED_OF_LIGHT, AccessPoint, LocalisationEstimate, Localiser,
    PathLossModel, Reading, Scan,
)


Bounds = tuple[tuple[float, float], tuple[float, float]]


def rssi_to_distance(
    rssi_dbm: float,
    reference_rssi_dbm: float,
    reference_distance_m: float,
    path_loss_exponent: float,
) -> float:
    """Invert a calibrated log-distance path-loss model, returning metres."""
    if not all(math.isfinite(v) for v in (
        rssi_dbm, reference_rssi_dbm, reference_distance_m, path_loss_exponent
    )):
        raise ValueError("RSSI and path-loss parameters must be finite")
    if reference_distance_m <= 0 or path_loss_exponent <= 0:
        raise ValueError("Reference distance and path-loss exponent must be positive")
    exponent = (reference_rssi_dbm - rssi_dbm) / (10 * path_loss_exponent)
    return reference_distance_m * (10 ** exponent)


def _valid_rssi(scan: Scan) -> dict[AccessPoint, Reading]:
    return {
        ap: reading
        for ap, reading in scan.items()
        if reading is not None and math.isfinite(reading.rssi_dbm)
        and all(math.isfinite(value) for value in ap.xy)
    }


class NearestAPLocaliser:
    """Use the coordinates of the access point with strongest valid RSSI."""

    def locate(self, scan: Scan) -> LocalisationEstimate | None:
        valid = _valid_rssi(scan)
        if not valid:
            return None
        best_ap = max(valid, key=lambda ap: valid[ap].rssi_dbm)
        return LocalisationEstimate(position=best_ap.xy, num_aps_used=1)


class WeightedCentroidLocaliser:
    """Average AP coordinates using inverse-square range or power weights.

    The returned covariance describes the AP spread, not calibrated position
    uncertainty. Parameters must come from a deployment calibration.
    """

    def __init__(
        self,
        path_loss: PathLossModel = PathLossModel(),
        weight_mode: str = "distance",
    ):
        if weight_mode not in ("distance", "linear_power"):
            raise ValueError(f"Unknown weight_mode: {weight_mode}")
        self.path_loss = path_loss
        self.weight_mode = weight_mode

    def locate(self, scan: Scan) -> LocalisationEstimate | None:
        valid = _valid_rssi(scan)
        if not valid:
            return None

        positions = np.array([ap.xy for ap in valid], dtype=float)
        rssi = np.array([reading.rssi_dbm for reading in valid.values()])
        if self.weight_mode == "linear_power":
            log_weights = rssi * (math.log(10) / 10)
        else:
            model = self.path_loss
            log_distance = math.log(model.reference_distance_m) + (
                model.reference_rssi_dbm - rssi
            ) * (math.log(10) / (10 * model.path_loss_exponent))
            log_weights = -2 * np.maximum(log_distance, math.log(0.1))

        # Normalising in log space avoids overflow/underflow for weak RSSI.
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        centre = np.average(positions, axis=0, weights=weights)
        delta = positions - centre
        covariance = (delta.T * weights) @ delta
        return LocalisationEstimate(
            position=(float(centre[0]), float(centre[1])),
            covariance=covariance,
            num_aps_used=len(valid),
        )


class _TrilaterationLocaliserBase:
    """Shared two-dimensional range-fit implementation for baseline methods."""

    def __init__(
        self,
        *,
        bounds: Bounds | None = None,
        min_aps: int = 3,
        robust: bool = False,
        f_scale_m: float = 1.5,
    ):
        if isinstance(min_aps, bool) or not isinstance(min_aps, int) or min_aps < 3:
            raise ValueError("min_aps must be an integer of at least 3")
        if not math.isfinite(f_scale_m) or f_scale_m <= 0:
            raise ValueError("f_scale_m must be finite and positive")
        lower, upper = bounds if bounds is not None else (
            (-math.inf, -math.inf), (math.inf, math.inf)
        )
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if lower.shape != (2,) or upper.shape != (2,):
            raise ValueError("bounds must contain two lower and two upper coordinates")
        if np.isnan(lower).any() or np.isnan(upper).any() or not np.all(lower < upper):
            raise ValueError("Each lower bound must be strictly below its upper bound")
        self.min_aps = min_aps
        self.robust = robust
        self.f_scale_m = f_scale_m
        self.bounds = (lower.copy(), upper.copy())

    def _distance_estimate(self, reading: Reading) -> float | None:
        raise NotImplementedError

    def locate(self, scan: Scan) -> LocalisationEstimate | None:
        positions = []
        ranges = []
        for ap, reading in scan.items():
            if reading is None or not all(math.isfinite(value) for value in ap.xy):
                continue
            try:
                distance = self._distance_estimate(reading)
            except (ValueError, OverflowError):
                continue
            if distance is not None and math.isfinite(distance) and distance >= 0:
                positions.append(ap.xy)
                ranges.append(distance)
        if len(ranges) < self.min_aps:
            return None

        ap_positions = np.asarray(positions, dtype=float)
        if np.linalg.matrix_rank(ap_positions - ap_positions.mean(axis=0)) < 2:
            return None  # Collinear APs cannot resolve both sides of their line.
        observed_ranges = np.asarray(ranges, dtype=float)

        def residuals(point: np.ndarray) -> np.ndarray:
            return np.linalg.norm(ap_positions - point, axis=1) - observed_ranges

        initial_guess = np.clip(ap_positions.mean(axis=0), *self.bounds)
        try:
            result = least_squares(
                residuals,
                initial_guess,
                bounds=self.bounds,
                loss="soft_l1" if self.robust else "linear",
                f_scale=self.f_scale_m,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return None
        if not result.success or not np.isfinite(result.x).all():
            return None
        return LocalisationEstimate(
            position=(float(result.x[0]), float(result.x[1])),
            covariance=self._covariance_estimate(result, len(ranges)),
            num_aps_used=len(ranges),
        )

    @staticmethod
    def _covariance_estimate(result, n_obs: int) -> np.ndarray | None:
        """Approximate local least-squares uncertainty, assuming valid ranges."""
        jtj = result.jac.T @ result.jac
        if n_obs <= 2 or np.linalg.matrix_rank(jtj) < 2:
            return None
        residual_variance = float(np.sum(result.fun ** 2) / (n_obs - 2))
        try:
            covariance = residual_variance * np.linalg.inv(jtj)
        except np.linalg.LinAlgError:
            return None
        return covariance if np.isfinite(covariance).all() else None


class RSSTrilaterationLocaliser(_TrilaterationLocaliserBase):
    """Fit RSSI-derived ranges with a calibrated log-distance model.

    Unmodelled attenuation increases inferred distance. This baseline treats
    inferred ranges as horizontal distances; height and NLOS bias remain
    model errors.
    """

    def __init__(
        self,
        path_loss: PathLossModel = PathLossModel(),
        *,
        bounds: Bounds | None = None,
        min_aps: int = 3,
        robust: bool = False,
        f_scale_m: float = 1.5,
    ):
        super().__init__(bounds=bounds, min_aps=min_aps, robust=robust, f_scale_m=f_scale_m)
        self.path_loss = path_loss

    def _distance_estimate(self, reading: Reading) -> float:
        model = self.path_loss
        return rssi_to_distance(
            reading.rssi_dbm, model.reference_rssi_dbm,
            model.reference_distance_m, model.path_loss_exponent,
        )


class RTTTrilaterationLocaliser(_TrilaterationLocaliserBase):
    """Fit ranges from legacy aggregate RTT, after a calibrated time offset.

    This is a comparison baseline, not an implementation of FTM. Generic
    network RTT includes device/stack/queueing delays. Even hardware FTM
    suffers positive NLOS and multipath range bias. A constant offset cannot
    correct these effects. Prefer ``FTMLocaliser`` for FTM timestamp bursts.
    """

    def __init__(
        self,
        *,
        rtt_offset_ms: float = 0.0,
        bounds: Bounds | None = None,
        min_aps: int = 3,
        robust: bool = False,
        f_scale_m: float = 1.5,
    ):
        super().__init__(bounds=bounds, min_aps=min_aps, robust=robust, f_scale_m=f_scale_m)
        if not math.isfinite(rtt_offset_ms) or rtt_offset_ms < 0:
            raise ValueError("rtt_offset_ms must be finite and nonnegative")
        self.rtt_offset_ms = rtt_offset_ms

    def _distance_estimate(self, reading: Reading) -> float | None:
        if reading.rtt_ms is None or not math.isfinite(reading.rtt_ms):
            return None
        corrected_rtt_ms = reading.rtt_ms - self.rtt_offset_ms
        if corrected_rtt_ms < 0:
            return None
        return SPEED_OF_LIGHT * corrected_rtt_ms / 2000.0
