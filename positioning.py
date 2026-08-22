from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from environment import Coord
from network import AccessPoint, Network, Scan
from scipy.optimize import least_squares


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


class TrilaterationLocaliser:
    """Estimate position by nonlinear least-squares fit to RSSI-derived ranges."""
    def __init__(self, network: Network, min_aps: int = 3, robust: bool = False, f_scale_m: float = 1.5):
        self.reference_rssi_dbm = network.reference_rssi_dbm
        self.reference_distance_m = network.reference_distance_m
        self.path_loss_exponent = network.path_loss_exponent
        self.min_aps = min_aps
        self.robust = robust
        self.f_scale_m = f_scale_m

        self.bounds = self._environment_bounds(network)

    def _distance_estimate(self, rssi_dbm: float) -> float:
        return rssi_to_distance(
            rssi_dbm,
            self.reference_rssi_dbm,
            self.reference_distance_m,
            self.path_loss_exponent,
        )

    @staticmethod
    def _environment_bounds(network: Network) -> tuple[np.ndarray, np.ndarray]:
        all_bounds = [c.geometry.bounds for c in network.environment.compartments]
        if not all_bounds:
            return np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf])
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)
        return np.array([minx, miny]), np.array([maxx, maxy])


    def locate(self, scan: Scan) -> Optional[LocalisationEstimate]:
        valid = {ap: reading for ap, reading in scan.items() if reading is not None}
        if len(valid) < self.min_aps:
            return None

        aps = list(valid.keys())
        ap_positions = np.array([ap.xy for ap in aps])
        ranges = np.array([self._distance_estimate(valid[ap].rssi_dbm) for ap in aps])

        def residuals(point: np.ndarray) -> np.ndarray:
            return np.linalg.norm(ap_positions - point, axis=1) - ranges

        initial_guess = np.clip(ap_positions.mean(axis=0), *self.bounds) # start with initial positions of aps as guess
        solver_kwargs = {"bounds": self.bounds}
        if self.robust:
            solver_kwargs['loss'] = 'soft_l1'
            solver_kwargs['f_scale'] = self.f_scale_m

        result = least_squares(residuals, initial_guess, **solver_kwargs)

        position = (float(result.x[0]), float(result.x[1]))
        covariance = self._covariance_estimate(result, n_obs=len(aps))

        return LocalisationEstimate(
            position=position,
            covariance=covariance,
            num_aps_used=len(aps),
        )

    def _covariance_estimate(self, result, n_obs: int) -> Optional[np.ndarray]:
        """Linearised (Gauss-Newton) covariance: residual_variance * inv(J'*J)
        Reflects how sensitive the fitted position is to actual range residuals
        """
        n_params = 2  # x, y
        degrees_of_freedom = n_obs - n_params
        if degrees_of_freedom <= 0:
            return None

        residual_variance = float(np.sum(result.fun ** 2) / degrees_of_freedom)
        jtj = result.jac.T @ result.jac
        try:
            return residual_variance * np.linalg.inv(jtj)
        except np.linalg.LinAlgError:
            return None
