from dataclasses import dataclass, field
from random import Random
from typing import List, Optional, Sequence

import numpy as np
from shapely.geometry import Point

from environment import Coord
from network import AccessPoint, Network, Reading, Receiver, Scan
from positioning import LocalisationEstimate



DEFAULT_FLOOR_DBM = -100.0

@dataclass
class FingerprintEntry:
    """A calibrated reference point"""
    position: Coord
    compartment: Optional[str]
    rssi_vector: np.ndarray # one value per AP in the localiser's ap_order
    n_samples: int = 0


def _signal_vector(
    ap_order: Sequence[AccessPoint],
    readings: dict,
    floor_dbm: float = DEFAULT_FLOOR_DBM,
) -> np.ndarray:
    return np.array(
        [readings.get(ap, floor_dbm) if readings.get(ap) is not None else floor_dbm
         for ap in ap_order],
        dtype=float,
    )


def build_radio_map(
    network: Network,
    rng: Random,
    ap_order: Sequence[AccessPoint],
    grid_spacing_m: float = 1.0,
    samples_per_point: int = 30,
    floor_dbm: float = DEFAULT_FLOOR_DBM,
) -> List[FingerprintEntry]:
    "simulate the actual fingerprinting that would be done manually"
    radio_map: List[FingerprintEntry] = []

    for comp in network.environment.compartments:
        minx, miny, maxx, maxy = comp.geometry.bounds
        nx = max(int((maxx - minx) / grid_spacing_m), 1)
        ny = max(int((maxy - miny) / grid_spacing_m), 1)

        for i in range(nx + 1):
            for j in range(ny + 1):
                x = minx + i * grid_spacing_m
                y = miny + j * grid_spacing_m
                if not comp.geometry.covers(Point(x, y)):
                    continue

                accum = np.zeros(len(ap_order))
                valid_counts = np.zeros(len(ap_order))

                for _ in range(samples_per_point):
                    receiver = Receiver("calib", (x, y, 1.0))
                    scan = network.scan(receiver, rng)
                    for idx, ap in enumerate(ap_order):
                        reading: Optional[Reading] = scan.get(ap)
                        if reading is not None:
                            accum[idx] += reading.rssi_dbm
                            valid_counts[idx] += 1

                vector = np.where(valid_counts > 0, accum / np.maximum(valid_counts, 1), floor_dbm)

                radio_map.append(
                    FingerprintEntry(
                        position=(x, y),
                        compartment=comp.name,
                        rssi_vector=vector,
                        n_samples=samples_per_point,
                    )
                )

    return radio_map


class FingerprintLocaliser:
    """Estimates position by first building an RSSI vector for every AP. Signal space distance is measured
    to every calibration point and return average position of k nearest calibration points"""

    def __init__(
        self,
        radio_map: List[FingerprintEntry],
        ap_order: Sequence[AccessPoint],
        k: int = 3,
        floor_dbm: float = DEFAULT_FLOOR_DBM,
    ):
        if not radio_map:
            raise ValueError("radio_map must contain at least one calibrated point")
        self.radio_map = radio_map
        self.ap_order = list(ap_order)
        self.k = k
        self.floor_dbm = floor_dbm
        self._map_matrix = np.stack([e.rssi_vector for e in radio_map])  # (n_points, n_aps)
        self._positions = np.array([e.position for e in radio_map]) # (n_points, 2)

    def locate(self, scan: Scan) -> Optional[LocalisationEstimate]:
        readings = {ap: (reading.rssi_dbm if reading is not None else None) for ap, reading in scan.items()}
        if not any(v is not None for v in readings.values()):
            return None

        live_vector = _signal_vector(self.ap_order, readings, self.floor_dbm)

        distances = np.linalg.norm(self._map_matrix - live_vector, axis=1)
        k = min(self.k, len(distances))
        nearest_idx = np.argpartition(distances, k - 1)[:k]

        nearest_idx = nearest_idx[np.argsort(distances[nearest_idx])]
        nearest_distances = distances[nearest_idx]
        nearest_positions = self._positions[nearest_idx]

        weights = 1.0 / np.maximum(nearest_distances, 1e-3) ** 2
        weights /= weights.sum()

        position = tuple(np.average(nearest_positions, axis=0, weights=weights))
        covariance = self._weighted_spread(nearest_positions, weights, position)

        return LocalisationEstimate(
            position=position,
            covariance=covariance,
            num_aps_used=sum(1 for v in readings.values() if v is not None),
        )

    @staticmethod
    def _weighted_spread(points: np.ndarray, weights: np.ndarray, centre: Coord) -> np.ndarray:
        cx, cy = centre
        cov = np.zeros((2, 2))
        for (x, y), w in zip(points, weights):
            dx, dy = x - cx, y - cy
            cov += w * np.array([[dx * dx, dx * dy], [dx * dy, dy * dy]])
        return cov


class FingerprintCompartmentLocaliser:
    """Classify by majority vote of k nearest point's compartment labels. Bit more robust than above"""

    def __init__(
        self,
        radio_map: List[FingerprintEntry],
        ap_order: Sequence[AccessPoint],
        k: int = 3,
        floor_dbm: float = DEFAULT_FLOOR_DBM,
    ):
        if not radio_map:
            raise ValueError("radio_map must contain at least one calibrated point")
        self.radio_map = radio_map
        self.ap_order = list(ap_order)
        self.k = k
        self.floor_dbm = floor_dbm
        self._map_matrix = np.stack([e.rssi_vector for e in radio_map])

    def classify(self, scan: Scan) -> Optional[str]:
        readings = {ap: (reading.rssi_dbm if reading is not None else None) for ap, reading in scan.items()}
        if not any(v is not None for v in readings.values()):
            return None

        live_vector = _signal_vector(self.ap_order, readings, self.floor_dbm)
        distances = np.linalg.norm(self._map_matrix - live_vector, axis=1)
        k = min(self.k, len(distances))
        nearest_idx = np.argpartition(distances, k - 1)[:k]

        votes: dict[str, float] = {}
        for idx in nearest_idx:
            entry = self.radio_map[idx]
            w = 1.0 / max(distances[idx], 1e-3)
            votes[entry.compartment] = votes.get(entry.compartment, 0.0) + w

        return max(votes, key=votes.get)