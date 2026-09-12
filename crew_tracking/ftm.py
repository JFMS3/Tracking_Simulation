"""Wi-Fi FTM burst processing and robust weighted range positioning.

The timestamp equation follows Ibrahim et al., MobiCom 2018, section 2.3:
https://doi.org/10.1145/3241539.3241555. MAD rejection and the uncertainty floor
are configurable engineering choices, not IEEE protocol requirements. See
docs/algorithm.md for equations, assumptions, calibration, and references.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
from scipy.optimize import least_squares

from .models import (
    Coord, FTMBurst, FTMSample, LocalisationEstimate, Scan, SPEED_OF_LIGHT,
)

_MAD_NORMAL_SCALE = 1.482602218505602


def rtt_ns_to_distance_m(rtt_ns: float) -> float:
    """Convert a finite, signed propagation RTT to measured one-way range."""
    if not math.isfinite(rtt_ns):
        raise ValueError("rtt_ns must be finite")
    return rtt_ns * (SPEED_OF_LIGHT / 2_000_000_000.0)


def distance_m_to_rtt_ns(distance_m: float) -> float:
    """Convert a finite, signed measured range to propagation RTT."""
    if not math.isfinite(distance_m):
        raise ValueError("distance_m must be finite")
    return distance_m * (2_000_000_000.0 / SPEED_OF_LIGHT)


@dataclass(frozen=True)
class FTMExchange:
    """Four FTM timestamps in ns, on two independent clocks.

    t1/t4: AP transmission of FTM / reception of ACK.
    t2/t3: station reception of FTM / transmission of ACK.
    Integer inputs retain precision: same-clock subtraction precedes floating
    conversion. The device adapter must unwrap hardware counters and correct
    clock-rate errors before supplying timestamps. No clock synchronisation
    or software request-latency subtraction is performed here.
    """

    t1_ns: int | float
    t2_ns: int | float
    t3_ns: int | float
    t4_ns: int | float
    successful: bool = True

    def to_sample(self) -> FTMSample:
        if not self.successful:
            return FTMSample(None, successful=False)
        if not all(math.isfinite(t) for t in (
            self.t1_ns, self.t2_ns, self.t3_ns, self.t4_ns,
        )):
            return FTMSample(None, successful=False)
        ap_interval = self.t4_ns - self.t1_ns
        station_interval = self.t3_ns - self.t2_ns
        if ap_interval < 0 or station_interval < 0:
            return FTMSample(None, successful=False)
        return FTMSample(float(ap_interval - station_interval))


@dataclass(frozen=True)
class FTMConfig:
    """Tunable estimator settings; distance quantities are metres."""

    min_samples: int = 3
    min_aps: int = 3
    range_std_floor_m: float = 0.25
    outlier_threshold: float = 3.5
    robust: bool = True
    robust_scale: float = 1.5  # dimensionless, after whitening by range std
    max_geometry_condition: float = 1e6

    def __post_init__(self) -> None:
        for name, minimum in (("min_samples", 2), ("min_aps", 3)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        for name in ("range_std_floor_m", "outlier_threshold", "robust_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.max_geometry_condition) or self.max_geometry_condition <= 1:
            raise ValueError("max_geometry_condition must be finite and greater than 1")
        if not isinstance(self.robust, bool):
            raise ValueError("robust must be a boolean")


@dataclass(frozen=True)
class FTMRangeEstimate:
    distance_m: float
    std_m: float
    num_samples: int
    num_rejected: int


def aggregate_ftm_burst(
    burst: FTMBurst,
    *,
    bias_m: float = 0.0,
    config: FTMConfig = FTMConfig(),
) -> Optional[FTMRangeEstimate]:
    """Calibrate, screen, reject isolated outliers, and average one AP burst.

    Subtract signed calibrated bias before aggregation. Preserve noisy signed
    samples to avoid positive truncation bias near an AP; reject a nonpositive
    final range estimate instead.
    MAD screening uses a floor to handle constant/quantized bursts. Effective
    uncertainty combines retained sample-mean variance with a systematic floor
    which does not shrink with burst length. It cannot account for every NLOS
    bias; no simulator truth is consulted.
    """
    if not math.isfinite(bias_m):
        raise ValueError("bias_m must be finite")
    distances = []
    for sample in burst.samples:
        if not sample.successful or sample.rtt_ns is None or not math.isfinite(sample.rtt_ns):
            continue
        distance = rtt_ns_to_distance_m(sample.rtt_ns) - bias_m
        if math.isfinite(distance):
            distances.append(distance)
    if len(distances) < config.min_samples:
        return None

    values = np.asarray(distances, dtype=float)
    median = float(np.median(values))
    deviations = np.abs(values - median)
    scale = max(_MAD_NORMAL_SCALE * float(np.median(deviations)), config.range_std_floor_m)
    retained = values[deviations <= config.outlier_threshold * scale]
    count = len(retained)
    if count < config.min_samples:
        return None

    variance_of_mean = float(np.var(retained, ddof=1)) / count
    std_m = math.sqrt(variance_of_mean + config.range_std_floor_m ** 2)
    distance_m = float(np.mean(retained))
    if not math.isfinite(distance_m) or distance_m <= 0 or not math.isfinite(std_m):
        return None
    return FTMRangeEstimate(distance_m, std_m, count, len(burst.samples) - count)


class FTMLocaliser:
    """Estimate x/y from FTM slant ranges and a configured receiver height.

    Solve sum rho(((||[x,y,z]-AP_i|| - range_i) / sigma_i)**2).
    Requires three distinct noncollinear surveyed APs and successful bursts.
    An RSSI observation or legacy RTT by itself never counts as FTM data.
    ``range_bias_m`` and per-BSSID offsets are additive measured-range biases
    to subtract, supplied from external calibration (never channel truth).
    """

    def __init__(
        self,
        *,
        receiver_height_m: float = 1.0,
        bounds: Optional[tuple[Coord, Coord]] = None,
        range_bias_m: float = 0.0,
        bias_by_bssid: Optional[Mapping[str, float]] = None,
        config: FTMConfig = FTMConfig(),
    ) -> None:
        if not math.isfinite(receiver_height_m):
            raise ValueError("receiver_height_m must be finite")
        if not math.isfinite(range_bias_m):
            raise ValueError("range_bias_m must be finite")
        self.receiver_height_m = receiver_height_m
        self.range_bias_m = range_bias_m
        self.bias_by_bssid = {key.lower(): value for key, value in (bias_by_bssid or {}).items()}
        if not all(math.isfinite(value) for value in self.bias_by_bssid.values()):
            raise ValueError("per-AP biases must be finite")
        self.config = config
        self.bounds = np.asarray(
            bounds if bounds is not None else ((-np.inf, -np.inf), (np.inf, np.inf)),
            dtype=float,
        ).copy()
        if self.bounds.shape != (2, 2) or np.isnan(self.bounds).any():
            raise ValueError("bounds must be (lower_xy, upper_xy) without NaN")
        if not np.all(self.bounds[0] < self.bounds[1]):
            raise ValueError("each lower bound must be less than its upper bound")

    def locate(self, scan: Scan) -> Optional[LocalisationEstimate]:
        # Conflicting surveys of one physical radio must not silently select
        # whichever dictionary entry happens to come first.
        surveyed_positions = {}
        for ap in scan:
            bssid = ap.bssid.lower()
            previous = surveyed_positions.setdefault(bssid, ap.position)
            if previous != ap.position:
                return None
        positions, distances, uncertainties = [], [], []
        seen_bssids: set[str] = set()
        for ap, reading in scan.items():
            if reading is None or reading.ftm_burst is None or not ap.ftm_responder:
                continue
            bssid = ap.bssid.lower()
            if bssid in seen_bssids:
                continue
            estimate = aggregate_ftm_burst(
                reading.ftm_burst,
                bias_m=self.range_bias_m + self.bias_by_bssid.get(bssid, 0.0),
                config=self.config,
            )
            if estimate is None:
                continue
            seen_bssids.add(bssid)
            positions.append(ap.position)
            distances.append(estimate.distance_m)
            uncertainties.append(estimate.std_m)
        if len(positions) < self.config.min_aps:
            return None

        anchors = np.asarray(positions, dtype=float)
        ranges = np.asarray(distances)
        stds = np.asarray(uncertainties)
        xy = anchors[:, :2]
        centred = xy - xy.mean(axis=0)
        singular_values = np.linalg.svd(centred, compute_uv=False)
        if singular_values[-1] <= 0 or singular_values[0] / singular_values[-1] > self.config.max_geometry_condition:
            return None
        dz_squared = (anchors[:, 2] - self.receiver_height_m) ** 2

        def predicted(point: np.ndarray) -> np.ndarray:
            return np.sqrt(np.sum((point - xy) ** 2, axis=1) + dz_squared)

        def residuals(point: np.ndarray) -> np.ndarray:
            return (predicted(point) - ranges) / stds

        def jacobian(point: np.ndarray) -> np.ndarray:
            # At an anchor the range derivative is undefined; a zero row
            # avoids division by zero and lets the other anchors constrain it.
            denominators = np.maximum(predicted(point), np.finfo(float).eps)
            return (point - xy) / (denominators * stds)[:, None]

        # A squared-range linear solution plus the anchor centre reduce
        # sensitivity to one poor initialisation of the nonlinear objective.
        rhs = np.sum(xy ** 2, axis=1) + dz_squared - ranges ** 2
        linear_start = np.linalg.lstsq(2.0 * centred, rhs - rhs.mean(), rcond=None)[0]
        starts = (linear_start, np.average(xy, axis=0, weights=1.0 / stds ** 2))
        solutions = []
        for start in starts:
            if not np.isfinite(start).all():
                continue
            try:
                result = least_squares(
                    residuals,
                    np.clip(start, *self.bounds),
                    jac=jacobian,
                    bounds=self.bounds,
                    loss="soft_l1" if self.config.robust else "linear",
                    f_scale=self.config.robust_scale,
                    max_nfev=200,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
            if result.success and np.isfinite(result.x).all() and math.isfinite(result.cost):
                solutions.append(result)
        if not solutions:
            return None
        result = min(solutions, key=lambda solution: solution.cost)
        # SciPy's robust Jacobian includes the loss's local curvature. The
        # inverse normal matrix is an approximate local covariance, not a
        # guarantee of calibrated coverage in NLOS or at active bounds.
        singular_values = np.linalg.svd(result.jac, compute_uv=False)
        if singular_values[-1] <= 0 or singular_values[0] / singular_values[-1] > self.config.max_geometry_condition:
            return None
        covariance = None
        if not np.any(result.active_mask):
            try:
                covariance = np.linalg.inv(result.jac.T @ result.jac)
            except np.linalg.LinAlgError:
                return None
            if not np.isfinite(covariance).all():
                covariance = None
        return LocalisationEstimate(
            position=(float(result.x[0]), float(result.x[1])),
            covariance=covariance,
            num_aps_used=len(positions),
        )
