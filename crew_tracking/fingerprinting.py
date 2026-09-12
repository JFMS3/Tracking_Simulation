"""RSSI fingerprint matching against an externally supplied radio map.

Nearest-neighbour signal-space matching follows the RADAR approach:
Bahl and Padmanabhan, IEEE INFOCOM 2000, doi:10.1109/INFCOM.2000.832252.
The inverse-distance weighting below is an explicit heuristic extension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .models import AccessPoint, Coord, LocalisationEstimate, Scan


DEFAULT_FLOOR_DBM = -100.0


@dataclass
class FingerprintEntry:
    """One reference point; RSSI ordering must match the supplied AP order."""

    position: Coord
    compartment: str | None
    rssi_vector: np.ndarray
    n_samples: int = 0


def _signal_vector(
    ap_order: Sequence[AccessPoint],
    scan: Scan,
    floor_dbm: float,
) -> tuple[np.ndarray, int]:
    vector = np.full(len(ap_order), floor_dbm, dtype=float)
    heard = 0
    for index, ap in enumerate(ap_order):
        reading = scan.get(ap)
        if reading is not None and math.isfinite(reading.rssi_dbm):
            vector[index] = reading.rssi_dbm
            heard += 1
    return vector, heard


class _FingerprintBase:
    def __init__(
        self,
        radio_map: Sequence[FingerprintEntry],
        ap_order: Sequence[AccessPoint],
        k: int = 3,
        floor_dbm: float = DEFAULT_FLOOR_DBM,
    ):
        if not radio_map:
            raise ValueError("radio_map must contain at least one calibrated point")
        if not ap_order or len(set(ap_order)) != len(ap_order):
            raise ValueError("ap_order must contain unique access points")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not math.isfinite(floor_dbm):
            raise ValueError("floor_dbm must be finite")
        self.radio_map = tuple(radio_map)
        self.ap_order = tuple(ap_order)
        self.k = min(k, len(radio_map))
        self.floor_dbm = floor_dbm
        self._map_matrix = np.asarray([e.rssi_vector for e in radio_map], dtype=float)
        self._positions = np.asarray([e.position for e in radio_map], dtype=float)
        self._compartments = tuple(e.compartment for e in radio_map)
        if self._map_matrix.shape != (len(radio_map), len(ap_order)):
            raise ValueError("Each RSSI vector must have one value per AP")
        if self._positions.shape != (len(radio_map), 2):
            raise ValueError("Each reference position must have two coordinates")
        if not np.isfinite(self._map_matrix).all() or not np.isfinite(self._positions).all():
            raise ValueError("Radio-map positions and RSSI vectors must be finite")

    def _nearest(self, scan: Scan) -> tuple[np.ndarray, np.ndarray, int] | None:
        live_vector, heard = _signal_vector(self.ap_order, scan, self.floor_dbm)
        if heard == 0:
            return None
        distances = np.linalg.norm(self._map_matrix - live_vector, axis=1)
        # Stable sorting makes ties deterministic in radio-map order.
        nearest = np.argsort(distances, kind="stable")[:self.k]
        return nearest, distances[nearest], heard


class FingerprintLocaliser(_FingerprintBase):
    """Inverse-square weighted mean of k nearest signal-space neighbours.

    Covariance describes reference-point spread, not calibrated uncertainty.
    Missing APs use the configured RSSI floor. Unknown APs are ignored.
    """

    def locate(self, scan: Scan) -> LocalisationEstimate | None:
        matched = self._nearest(scan)
        if matched is None:
            return None
        nearest, distances, heard = matched
        points = self._positions[nearest]
        weights = 1.0 / np.maximum(distances, 1e-3) ** 2
        weights /= weights.sum()
        centre = np.average(points, axis=0, weights=weights)
        delta = points - centre
        return LocalisationEstimate(
            position=(float(centre[0]), float(centre[1])),
            covariance=(delta.T * weights) @ delta,
            num_aps_used=heard,
        )


class FingerprintCompartmentLocaliser(_FingerprintBase):
    """Inverse-distance weighted vote of k nearest compartment labels."""

    def classify(self, scan: Scan) -> str | None:
        matched = self._nearest(scan)
        if matched is None:
            return None
        nearest, distances, _ = matched
        votes: dict[str | None, float] = {}
        for index, distance in zip(nearest, distances):
            compartment = self._compartments[index]
            votes[compartment] = votes.get(compartment, 0.0) + 1.0 / max(distance, 1e-3)
        return max(votes, key=votes.__getitem__)
