"""Timestamped constant-velocity tracking using observed position estimates.

The defaults are provisional engineering settings, not field-calibrated error
models. This layer consumes ordinary localisation results and capture times;
it has no simulator, map, radio-channel or ground-truth dependencies. It smooths
motion but cannot remove persistent localisation bias or enforce doorways.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .models import Coord, LocalisationEstimate


@dataclass(frozen=True)
class KalmanConfig:
    """Provisional tracking parameters, using metres and seconds.

    ``acceleration_noise_density_m2_s3`` is continuous white acceleration
    spectral density, rather than variance of one constant acceleration per
    scan. ``innovation_gate`` is a *squared* Mahalanobis-distance threshold.
    The position uncertainty floor limits collapse on correlated Wi-Fi scans;
    it does not make the reported covariance a calibrated confidence bound.
    """

    measurement_std_m: float = 1.5
    acceleration_noise_density_m2_s3: float = 1.0
    initial_velocity_std_mps: float = 1.5
    position_std_floor_m: float = 0.5
    max_prediction_age_s: float = 3.0
    reset_after_s: float = 10.0
    innovation_gate: float = 16.0
    reacquire_after_rejections: int = 3

    def __post_init__(self) -> None:
        for name in (
            "measurement_std_m", "acceleration_noise_density_m2_s3",
            "initial_velocity_std_mps", "position_std_floor_m",
            "max_prediction_age_s", "reset_after_s", "innovation_gate",
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
        if self.measurement_std_m < self.position_std_floor_m:
            raise ValueError("measurement_std_m must be at least position_std_floor_m")
        if self.reset_after_s <= self.max_prediction_age_s:
            raise ValueError("reset_after_s must exceed max_prediction_age_s")
        count = self.reacquire_after_rejections
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 2:
            raise ValueError("reacquire_after_rejections must be an integer of at least 2")


@dataclass(frozen=True)
class TrackEstimate:
    """A track snapshot; covariance order is ``[x, y, vx, vy]``.

    The covariance is a detached 4-by-4 array. Predicted and outlier-rejected
    snapshots preserve the time of the last accepted position measurement.
    """

    device_id: str
    position: Coord
    velocity: tuple[float, float]
    covariance: np.ndarray
    timestamp_s: float
    last_measurement_timestamp_s: float
    status: str
    innovation_distance_squared: float | None = None

    @property
    def measurement_age_s(self) -> float:
        return self.timestamp_s - self.last_measurement_timestamp_s

    @property
    def position_std_m(self) -> float:
        """Largest principal-axis position standard deviation in metres."""
        return float(np.sqrt(max(0.0, np.linalg.eigvalsh(self.covariance[:2, :2])[-1])))


@dataclass
class _TrackState:
    vector: np.ndarray
    covariance: np.ndarray
    timestamp_s: float
    last_measurement_timestamp_s: float
    candidate_position: np.ndarray | None = None
    candidate_covariance: np.ndarray | None = None
    candidate_count: int = 0


class KalmanTracker:
    """Independent constant-velocity Kalman tracks for crew-device IDs.

    Call ``update`` once per device with a strictly increasing monotonic
    *capture* timestamp, including scans without a position. Source estimate
    covariance and heuristic quality fields are deliberately ignored. An
    externally calibrated position covariance may instead be supplied through
    ``measurement_covariance``; otherwise the configured constant is used.

    Missing positions permit short predictions, then yield ``None`` until
    observations recover. Tracks expire after ``reset_after_s`` without an
    accepted measurement. Rejected measurements do not refresh that age.
    Several consecutive rejected positions can reinitialise a track when each
    agrees with its predecessor within their combined measurement uncertainty.
    This bounded recovery heuristic can also follow a persistent bad fix; it
    is not evidence that a candidate location is correct.

    ``predict`` returns a non-mutating display snapshot, so a display tick does
    not advance the capture-time watermark or interfere with delayed scans.
    Calls for unknown devices with no usable position create no state. Expiry
    during ``update`` and explicit reset clear ordering for the track.
    Instances are intended to be owned by one consumer, not shared across
    concurrent threads without external synchronisation.
    """

    def __init__(self, config: KalmanConfig = KalmanConfig()) -> None:
        if not isinstance(config, KalmanConfig):
            raise ValueError("config must be a KalmanConfig")
        self.config = config
        self._tracks: dict[str, _TrackState] = {}

    @staticmethod
    def _validate_device_id(device_id: str) -> None:
        if not isinstance(device_id, str) or not device_id.strip():
            raise ValueError("device_id must be a non-empty string")

    def reset(self, device_id: str | None = None) -> None:
        """Forget one device, or all devices when no device ID is supplied."""
        if device_id is None:
            self._tracks.clear()
        else:
            self._validate_device_id(device_id)
            self._tracks.pop(device_id, None)

    def predict(self, device_id: str, timestamp_s: float) -> TrackEstimate | None:
        """Return a display prediction without committing a capture event.

        Display time may equal or exceed the most recent captured scan. A
        later display tick does not prevent accepting a delayed capture newer
        than that scan. Expired predictions return ``None``; state is retained
        until ``update`` processes a capture event or ``reset`` clears it.
        Use ``update(device_id, timestamp_s, None)`` for an actual missing scan.
        """
        self._validate_device_id(device_id)
        timestamp_s = self._validated_timestamp(timestamp_s)
        state = self._tracks.get(device_id)
        if state is None:
            return None
        if timestamp_s < state.timestamp_s:
            raise ValueError("prediction time cannot precede the latest capture")
        if timestamp_s - state.last_measurement_timestamp_s > self.config.max_prediction_age_s:
            return None
        vector, covariance = self._prediction(state, timestamp_s - state.timestamp_s)
        snapshot_state = _TrackState(
            vector, covariance, timestamp_s, state.last_measurement_timestamp_s,
        )
        return self._snapshot(device_id, snapshot_state, "predicted", None)

    @staticmethod
    def _validated_timestamp(timestamp_s: float) -> float:
        if (isinstance(timestamp_s, bool) or not isinstance(timestamp_s, Real)
                or not math.isfinite(timestamp_s) or timestamp_s < 0):
            raise ValueError("timestamp_s must be finite and nonnegative")
        return float(timestamp_s)

    def update(
        self,
        device_id: str,
        timestamp_s: float,
        measurement: LocalisationEstimate | None,
        *,
        measurement_covariance: np.ndarray | None = None,
    ) -> TrackEstimate | None:
        """Assimilate a measured position or advance through a missing scan.

        Invalid/nonfinite coordinates count as a missing measurement. Invalid
        IDs, timestamps or explicitly supplied covariance raise ``ValueError``
        before changing any state. The supplied covariance must be finite,
        symmetric, positive semidefinite and 2-by-2. Its eigenvalues are floored
        to ``position_std_floor_m ** 2`` before use.
        """
        self._validate_device_id(device_id)
        timestamp_s = self._validated_timestamp(timestamp_s)
        state = self._tracks.get(device_id)
        if state is not None and timestamp_s <= state.timestamp_s:
            raise ValueError("device capture timestamps must increase strictly")
        observation_covariance = self._measurement_covariance(measurement_covariance)
        position = self._measurement_position(measurement)

        if state is None:
            if position is None:
                return None
            return self._initialise(
                device_id, timestamp_s, position, observation_covariance, "initialised"
            )

        age = timestamp_s - state.last_measurement_timestamp_s
        if age >= self.config.reset_after_s:
            if position is None:
                del self._tracks[device_id]
                return None
            return self._initialise(
                device_id, timestamp_s, position, observation_covariance, "reinitialised"
            )

        vector, covariance = self._prediction(state, timestamp_s - state.timestamp_s)
        status = "predicted"
        distance_squared = None
        accepted_time = state.last_measurement_timestamp_s
        candidate_position = state.candidate_position
        candidate_covariance = state.candidate_covariance
        candidate_count = state.candidate_count

        if position is not None:
            innovation = position - vector[:2]
            innovation_covariance = covariance[:2, :2] + observation_covariance
            distance_squared = float(
                innovation @ np.linalg.solve(innovation_covariance, innovation)
            )
            if distance_squared <= self.config.innovation_gate:
                # H = [I_2, 0]. Solve instead of explicitly inverting S.
                gain = np.linalg.solve(innovation_covariance, covariance[:2, :]).T
                vector = vector + gain @ innovation
                identity_minus_kh = np.eye(4)
                identity_minus_kh[:, :2] -= gain
                covariance = (
                    identity_minus_kh @ covariance @ identity_minus_kh.T
                    + gain @ observation_covariance @ gain.T
                )
                covariance = self._floor_position_covariance(covariance)
                accepted_time = timestamp_s
                status = "updated"
                candidate_position = candidate_covariance = None
                candidate_count = 0
            else:
                status = "outlier_rejected"
                consistent = False
                if candidate_position is not None and candidate_covariance is not None:
                    difference = position - candidate_position
                    combined_covariance = observation_covariance + candidate_covariance
                    candidate_distance = float(
                        difference @ np.linalg.solve(combined_covariance, difference)
                    )
                    consistent = candidate_distance <= self.config.innovation_gate
                candidate_count = candidate_count + 1 if consistent else 1
                if candidate_count >= self.config.reacquire_after_rejections:
                    return self._initialise(
                        device_id, timestamp_s, position, observation_covariance,
                        "reinitialised", distance_squared,
                    )
                candidate_position = position.copy()
                candidate_covariance = observation_covariance.copy()
        else:
            # A gap breaks the requirement for consecutive rejected positions.
            candidate_position = candidate_covariance = None
            candidate_count = 0

        new_state = _TrackState(
            vector, covariance, timestamp_s, accepted_time,
            candidate_position, candidate_covariance, candidate_count,
        )
        self._tracks[device_id] = new_state
        if timestamp_s - accepted_time > self.config.max_prediction_age_s:
            return None
        return self._snapshot(device_id, new_state, status, distance_squared)

    @staticmethod
    def _measurement_position(measurement: LocalisationEstimate | None) -> np.ndarray | None:
        if measurement is None:
            return None
        try:
            position = np.array(measurement.position, dtype=float, copy=True)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            return None
        return position

    def _measurement_covariance(self, supplied: np.ndarray | None) -> np.ndarray:
        if supplied is None:
            return np.eye(2) * self.config.measurement_std_m ** 2
        try:
            covariance = np.array(supplied, dtype=float, copy=True)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("measurement_covariance must be a finite 2-by-2 matrix") from error
        if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
            raise ValueError("measurement_covariance must be a finite 2-by-2 matrix")
        if not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-10):
            raise ValueError("measurement_covariance must be symmetric")
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if np.min(eigenvalues) < -1e-10:
            raise ValueError("measurement_covariance must be positive semidefinite")
        eigenvalues = np.maximum(eigenvalues, self.config.position_std_floor_m ** 2)
        return (eigenvectors * eigenvalues) @ eigenvectors.T

    def _initialise(
        self, device_id: str, timestamp_s: float, position: np.ndarray,
        observation_covariance: np.ndarray, status: str,
        innovation_distance_squared: float | None = None,
    ) -> TrackEstimate:
        vector = np.zeros(4)
        vector[:2] = position
        covariance = np.zeros((4, 4))
        covariance[:2, :2] = observation_covariance
        covariance[2:, 2:] = np.eye(2) * self.config.initial_velocity_std_mps ** 2
        state = _TrackState(vector, covariance, timestamp_s, timestamp_s)
        self._tracks[device_id] = state
        return self._snapshot(device_id, state, status, innovation_distance_squared)

    def _prediction(self, state: _TrackState, dt: float) -> tuple[np.ndarray, np.ndarray]:
        transition = np.eye(4)
        transition[0, 2] = transition[1, 3] = dt
        process_covariance = np.zeros((4, 4))
        process_covariance[:2, :2] = np.eye(2) * (dt ** 3 / 3)
        process_covariance[:2, 2:] = process_covariance[2:, :2] = np.eye(2) * (dt ** 2 / 2)
        process_covariance[2:, 2:] = np.eye(2) * dt
        process_covariance *= self.config.acceleration_noise_density_m2_s3
        vector = transition @ state.vector
        covariance = transition @ state.covariance @ transition.T + process_covariance
        return vector, self._floor_position_covariance(covariance)

    def _floor_position_covariance(self, covariance: np.ndarray) -> np.ndarray:
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues, eigenvectors = np.linalg.eigh(covariance[:2, :2])
        additions = np.maximum(self.config.position_std_floor_m ** 2 - eigenvalues, 0.0)
        covariance[:2, :2] += (eigenvectors * additions) @ eigenvectors.T
        return (covariance + covariance.T) * 0.5

    @staticmethod
    def _snapshot(
        device_id: str, state: _TrackState, status: str,
        innovation_distance_squared: float | None,
    ) -> TrackEstimate:
        return TrackEstimate(
            device_id=device_id,
            position=(float(state.vector[0]), float(state.vector[1])),
            velocity=(float(state.vector[2]), float(state.vector[3])),
            covariance=state.covariance.copy(),
            timestamp_s=state.timestamp_s,
            last_measurement_timestamp_s=state.last_measurement_timestamp_s,
            status=status,
            innovation_distance_squared=innovation_distance_squared,
        )
