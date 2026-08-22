from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from environment import Coord
from network import AccessPoint, Network, Scan


@dataclass
class LocalisationEstimate:
    position: Coord
    covariance: Optional[np.ndarray] = None # 2x2 position covariance
    num_aps_used: int = 0


# abstract class inhereted by each localiser
class Localiser(Protocol):
    def locate(self, scan: Scan) -> Optional[LocalisationEstimate]:
        pass


def rssi_to_distance(
    rssi_dbm: float,
    reference_rssi_dbm: float,
    reference_distance_m: float,
    path_loss_exponent: float,
) -> float:
    exponent = (reference_rssi_dbm - rssi_dbm) / (10 * path_loss_exponent)
    return reference_distance_m * (10 ** exponent)
 
 
class NearestAPLocaliser:
    def locate(self, scan: Scan) -> Optional[LocalisationEstimate]:
        valid = {ap: reading for ap, reading in scan.items() if reading is not None}
        if not valid:
            return None
 
        best_ap = max(valid, key=lambda ap: valid[ap].rssi_dbm)
        return LocalisationEstimate(
            position=best_ap.xy,
            covariance=None,
            num_aps_used=1,
        )
 
 
class WeightedCentroidLocaliser:
    """Weighted average of AP positions, weighted by estimated proximity."""
    def __init__(self, network: Network, weight_mode: str = "distance"):
        if weight_mode not in ("distance", "linear_power"):
            raise ValueError(f"Unknown weight_mode: {weight_mode}")
 
        self.reference_rssi_dbm = network.reference_rssi_dbm
        self.reference_distance_m = network.reference_distance_m
        self.path_loss_exponent = network.path_loss_exponent
        self.weight_mode = weight_mode
 
    def locate(self, scan: Scan) -> Optional[LocalisationEstimate]:
        valid = {ap: reading for ap, reading in scan.items() if reading is not None}
        if not valid:
            return None
 
        weights = {ap: self._weight(reading.rssi_dbm) for ap, reading in valid.items()}
        total_weight = sum(weights.values())
 
        x = sum(ap.xy[0] * w for ap, w in weights.items()) / total_weight
        y = sum(ap.xy[1] * w for ap, w in weights.items()) / total_weight
 
        covariance = self._weighted_spread(weights, total_weight, (x, y))
 
        return LocalisationEstimate(
            position=(x, y),
            covariance=covariance,
            num_aps_used=len(valid),
        )
 
    def _weight(self, rssi_dbm: float) -> float:
        if self.weight_mode == "linear_power":
            return 10 ** (rssi_dbm / 10)

        distance = rssi_to_distance(
            rssi_dbm,
            self.reference_rssi_dbm,
            self.reference_distance_m,
            self.path_loss_exponent,
        )
        return 1.0 / max(distance, 0.1) ** 2

    def _weighted_spread(
        self,
        weights: dict[AccessPoint, float],
        total_weight: float,
        centre: Coord,
    ) -> np.ndarray:

        cx, cy = centre
        cov = np.zeros((2, 2))
        for ap, w in weights.items():
            normalised_w = w / total_weight
            dx = ap.xy[0] - cx
            dy = ap.xy[1] - cy
            cov += normalised_w * np.array([[dx * dx, dx * dy], [dx * dy, dy * dy]])
        return cov
 