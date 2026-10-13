from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import shapely
from environment import Coord
from network import AccessPoint, Network, Scan, Reading, SPEED_OF_LIGHT
from scipy.optimize import least_squares
from scipy.special import log_ndtr


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
    """Average positions of APs we hear, but weight stronger signals higher"""
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
        """Calculates covariance based on how spread the voting APs are"""
        cx, cy = centre
        cov = np.zeros((2, 2))
        for ap, w in weights.items():
            normalised_w = w / total_weight
            dx = ap.xy[0] - cx
            dy = ap.xy[1] - cy
            cov += normalised_w * np.array([[dx * dx, dx * dy], [dx * dy, dy * dy]])
        return cov


class _TrilaterationLocaliserBase:
    """Inherited by RSS and RTT Trilateration localisers"""
    def __init__(
        self,
        network: Network,
        min_aps: int = 3,
        robust: bool = False,
        f_scale_m: float = 1.5,
    ):
        self.min_aps = min_aps
        self.robust = robust
        self.f_scale_m = f_scale_m
        self.bounds = self._environment_bounds(network)
 
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
 
    def _distance_estimate(self, reading: Reading) -> float:
        raise NotImplementedError
 
    def locate(self, scan: Scan) -> Optional[LocalisationEstimate]:
        valid = {ap: reading for ap, reading in scan.items() if reading is not None}
        if len(valid) < self.min_aps:
            return None
        
        aps = list(valid.keys())
        ap_positions = np.array([ap.xy for ap in aps])
        ranges = np.array([self._distance_estimate(valid[ap]) for ap in aps])
 
        def residuals(point: np.ndarray) -> np.ndarray:
            return np.linalg.norm(ap_positions - point, axis=1) - ranges

        initial_guess = np.clip(ap_positions.mean(axis=0), *self.bounds)
        solver_kwargs = {"bounds": self.bounds}
        if self.robust:
            solver_kwargs["loss"] = "soft_l1"
            solver_kwargs["f_scale"] = self.f_scale_m
 
        result = least_squares(residuals, initial_guess, **solver_kwargs)
        position = (float(result.x[0]), float(result.x[1]))
        covariance = self._covariance_estimate(result, n_obs=len(aps))
 
        return LocalisationEstimate(
            position=position,
            covariance=covariance,
            num_aps_used=len(aps),
        )
 
    def _covariance_estimate(self, result, n_obs: int) -> Optional[np.ndarray]:
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
 
 
class RSSTrilaterationLocaliser(_TrilaterationLocaliserBase):
    """Estimate position by nonlinear least-squares fit to RSSI-derived ranges.
 
    Unlike WeightedCentroid, this actually models the geometry: each AP's
    RSSI is converted to a range estimate, and we solve for the (x, y) that
    best satisfies "distance from AP_i should equal range_i" across all APs.
 
    Weakness (see RTTTrilaterationLocaliser below): the RSSI -> distance
    conversion assumes free-space path loss, so it's systematically wrong
    whenever a wall was actually crossed -- the missing dB reads as extra
    distance, not as attenuation.
    """
 
    def __init__(
        self,
        network: Network,
        min_aps: int = 3,
        robust: bool = False,
        f_scale_m: float = 1.5,
    ):
        super().__init__(network, min_aps=min_aps, robust=robust, f_scale_m=f_scale_m)
        self.reference_rssi_dbm = network.reference_rssi_dbm
        self.reference_distance_m = network.reference_distance_m
        self.path_loss_exponent = network.path_loss_exponent
 
    def _distance_estimate(self, reading: Reading) -> float:
        return rssi_to_distance(
            reading.rssi_dbm,
            self.reference_rssi_dbm,
            self.reference_distance_m,
            self.path_loss_exponent,
        )
 
 
class RTTTrilaterationLocaliser(_TrilaterationLocaliserBase):
    """Estimate position by nonlinear least-squares fit to RTT-derived ranges.
 
    RTT (round-trip time, e.g. 802.11mc/WiFi FTM) converts to distance via
    pure physics -- speed of light -- rather than a fitted path-loss model.
    That means it has no equivalent of RSS trilateration's wall-attenuation
    bias: a wall doesn't change how long light takes to travel, only how
    much of it arrives. This is why RTT is the range source real indoor
    positioning systems actually use for trilateration, and RSS is mostly
    reserved for fingerprinting/weighted-average methods instead.
    """
 
    def __init__(
        self,
        network: Network,
        min_aps: int = 3,
        robust: bool = False,
        f_scale_m: float = 1.5,
    ):
        super().__init__(network, min_aps=min_aps, robust=robust, f_scale_m=f_scale_m)
        self.rtt_base_ms = network.rtt_base_ms
 
    def _distance_estimate(self, reading: Reading) -> float:
        time_of_flight_ms = max(reading.rtt_ms - self.rtt_base_ms, 0.0)
        time_of_flight_s = time_of_flight_ms / 1000.0
        return (SPEED_OF_LIGHT * time_of_flight_s) / 2


class WallAwareLocaliser:
    """Maximum-likelihood grid search over the deck that knows where the walls are.

    The trilateration localisers above convert every reading into a range and
    ignore walls, so a 20 dB bulkhead reads as ~8x extra distance. This one
    instead asks, for every candidate point on a grid: given the walls between
    here and each AP, what would we expect to measure? The scan is scored
    against that prediction:

    - "rss": observed RSSI vs the multi-wall path-loss model (log-distance
      loss plus the attenuation of every wall crossed).
    - "rtt": measured range vs geometric distance plus the expected NLOS bias
      for every wall crossed.
    - "hybrid": both terms, treating RSSI and RTT errors as independent.

    Detection is evidence too: an AP that was not heard makes points where it
    should have been loud unlikely, though never impossible (interference
    can hide any AP). Residuals are Gaussian by default, which was slightly
    more accurate for single scans at random points (scenarios.py).
    `robust=True` swaps in a Cauchy-style loss; it is the better choice inside
    GridTracker, where one bad scan would otherwise yank a confident belief
    across the deck (tracking.py).

    Model parameters (path loss, wall losses, RTT bias/noise, noise floor) are
    read from `network`. Pass a separately configured Network to test a
    miscalibrated model against the true channel.
    """

    MEASUREMENTS = ("rss", "rtt", "hybrid")

    def __init__(
        self,
        network: Network,
        measurement: str = "rss",
        grid_spacing_m: float = 0.1,
        receiver_height_m: float = 1.0,
        rssi_sigma_db: float = 3.0,
        robust: bool = False,
        use_detections: bool = True,
        device_offset_sigma_db: Optional[float] = None,
        unexplained_miss_probability: float = 0.05,
    ):
        """`device_offset_sigma_db`: if set, every scan's RSSI may carry an unknown
        constant phone offset with this prior spread; it is estimated jointly with
        position (per grid point) instead of being read as distance."""
        if measurement not in self.MEASUREMENTS:
            raise ValueError(f"Unknown measurement: {measurement!r}. Use one of {self.MEASUREMENTS}")
        if grid_spacing_m <= 0 or rssi_sigma_db <= 0:
            raise ValueError("grid_spacing_m and rssi_sigma_db must be positive")
        if device_offset_sigma_db is not None and device_offset_sigma_db <= 0:
            raise ValueError("device_offset_sigma_db must be positive")
        if not 0.0 < unexplained_miss_probability < 1.0:
            raise ValueError("unexplained_miss_probability must be between 0 and 1")
        if not np.isfinite(receiver_height_m):
            raise ValueError("receiver_height_m must be finite")

        self.measurement = measurement
        self.rssi_sigma_db = rssi_sigma_db
        self.robust = robust
        self.use_detections = use_detections
        self.unexplained_miss_probability = unexplained_miss_probability
        self.device_offset_sigma_db = device_offset_sigma_db
        self.rtt_base_ms = network.rtt_base_ms
        self.access_points = list(network.access_points)
        self.environment = network.environment

        self.grid_points, self.grid_compartments = self._build_grid(network, grid_spacing_m)

        ap_positions = np.array([ap.position for ap in self.access_points])  # (n_aps, 3)
        dx = self.grid_points[:, None, 0] - ap_positions[None, :, 0]
        dy = self.grid_points[:, None, 1] - ap_positions[None, :, 1]
        dz = receiver_height_m - ap_positions[None, :, 2]
        distance = np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 0.1)  # (n_points, n_aps)

        wall_count, wall_loss_db = self._wall_terms(network, self.grid_points, ap_positions[:, :2])

        # Expected measurements at every grid point, for every AP
        self._expected_rssi = (
            network.reference_rssi_dbm
            - 10.0 * network.path_loss_exponent * np.log10(distance / network.reference_distance_m)
            - wall_loss_db
        )
        # Same RTT error model as Network.rtt, including the weak-signal bias
        weak_margin_db = np.maximum(network.rtt_weak_signal_threshold_dbm - self._expected_rssi, 0.0)
        self._expected_range = (
            distance
            + network.rtt_range_bias_m
            + wall_count * network.rtt_nlos_bias_per_wall_m
            + network.rtt_weak_signal_penalty_m * (1.0 - np.exp(-weak_margin_db / 6.0))
        )
        self._range_sigma = np.maximum(
            network.rtt_ranging_noise_m + wall_count * network.rtt_nlos_noise_per_wall_m,
            0.1,
        )
        self._detection_threshold_dbm = self._detection_threshold(network)

    @staticmethod
    def _build_grid(network: Network, spacing: float) -> tuple[np.ndarray, np.ndarray]:
        """Grid cell centres that fall inside a compartment, plus each one's compartment name."""
        compartments = network.environment.compartments
        if not compartments:
            raise ValueError("WallAwareLocaliser needs at least one compartment")
        all_bounds = np.array([c.geometry.bounds for c in compartments])
        minx, miny = all_bounds[:, 0].min(), all_bounds[:, 1].min()
        maxx, maxy = all_bounds[:, 2].max(), all_bounds[:, 3].max()

        xs, ys = np.meshgrid(
            np.arange(minx + spacing / 2, maxx, spacing),
            np.arange(miny + spacing / 2, maxy, spacing),
        )
        xs, ys = xs.ravel(), ys.ravel()
        names = np.full(xs.shape, None, dtype=object)
        for compartment in compartments:
            # first match wins, as in ShipEnvironment.compartment_at (for points, intersects == covers)
            inside = shapely.intersects_xy(compartment.geometry, xs, ys) & (names == None)  # noqa: E711
            names[inside] = compartment.name

        keep = names != None  # noqa: E711
        return np.column_stack([xs[keep], ys[keep]]), names[keep]

    @staticmethod
    def _wall_terms(
        network: Network,
        points: np.ndarray,
        ap_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Wall count and total attenuation for every (point, AP) ray, matching walls_crossed."""
        n_points, n_aps = len(points), len(ap_xy)
        starts = np.repeat(ap_xy, n_points, axis=0)  # AP-major: [ap0 x all points, ap1 x all points, ...]
        ends = np.tile(points, (n_aps, 1))
        crosses = network.environment.crossing_matrix(starts, ends)  # (n_aps * n_points, n_walls)
        attenuation = np.array([wall.attenuation_db for wall in network.environment.walls], dtype=float)
        count = crosses.sum(axis=1).reshape(n_aps, n_points).T
        loss_db = (crosses @ attenuation).reshape(n_aps, n_points).T if len(attenuation) else np.zeros_like(count, dtype=float)
        return count.astype(float), loss_db

    @staticmethod
    def _detection_threshold(network: Network) -> float:
        """RSSI at which an AP is heard in half of all scans, at the nominal noise floor."""
        floor = network.nominal_noise_floor_dbm
        lo, hi = floor - 30.0, floor + 30.0
        # widen the bracket until it contains the 50% point (detection is monotonic in RSSI)
        for _ in range(100):
            if network.detection_probability(lo, floor) < 0.5:
                break
            lo -= 30.0
        for _ in range(100):
            if network.detection_probability(hi, floor) >= 0.5:
                break
            hi += 30.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if network.detection_probability(mid, floor) < 0.5:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def _loss(self, residual: np.ndarray) -> np.ndarray:
        """Negative log-likelihood of a standardised residual (up to a constant)."""
        if self.robust:
            return np.log1p(0.5 * residual ** 2)
        return 0.5 * residual ** 2

    def _measured_range(self, reading: Reading) -> float:
        time_of_flight_s = max(reading.rtt_ms - self.rtt_base_ms, 0.0) / 1000.0
        return SPEED_OF_LIGHT * time_of_flight_s / 2

    def log_likelihood(self, scan: Scan) -> np.ndarray:
        """Log-likelihood of `scan` at every grid point.

        A scan in which no AP was heard still carries evidence (the APs should
        have been loud here but weren't), so this always returns an array;
        `locate` alone declines to fix a position from such a scan.
        """
        log_lik = np.zeros(len(self.grid_points))
        uses_rssi = self.measurement in ("rss", "hybrid")
        rssi_residuals = []  # (n_points,) arrays in dB, one per heard AP
        missed = []          # indices of APs not heard

        for j, ap in enumerate(self.access_points):
            reading = scan.get(ap)
            if reading is None:
                missed.append(j)
                continue

            # When the RSSI value is in the likelihood it already says the AP was
            # loud enough to decode; a separate "was heard" term would count that twice.
            if self.use_detections and not uses_rssi:
                log_lik += log_ndtr((self._expected_rssi[:, j] - self._detection_threshold_dbm) / self.rssi_sigma_db)

            if uses_rssi:
                rssi_residuals.append(reading.rssi_dbm - self._expected_rssi[:, j])

            if self.measurement in ("rtt", "hybrid"):
                sigma = self._range_sigma[:, j]
                residual = (self._measured_range(reading) - self._expected_range[:, j]) / sigma
                log_lik -= self._loss(residual) + np.log(sigma)

        offset = np.zeros(len(self.grid_points))
        if rssi_residuals:
            residuals = np.column_stack(rssi_residuals)
            if self.device_offset_sigma_db is not None:
                # MAP estimate of a shared phone offset at each point (Gaussian prior),
                # i.e. the mean residual shrunk towards zero
                shrink = (self.rssi_sigma_db / self.device_offset_sigma_db) ** 2
                offset = residuals.sum(axis=1) / (residuals.shape[1] + shrink)
                residuals = residuals - offset[:, None]
                log_lik -= 0.5 * (offset / self.device_offset_sigma_db) ** 2
            log_lik -= self._loss(residuals / self.rssi_sigma_db).sum(axis=1)

        if self.use_detections:
            for j in missed:
                expected_rssi = self._expected_rssi[:, j] + offset
                # Interference bursts can hide even a loud AP, so a miss is never
                # treated as impossible: P(miss) >= unexplained_miss_probability.
                log_lik += np.logaddexp(
                    np.log(self.unexplained_miss_probability),
                    np.log1p(-self.unexplained_miss_probability)
                    + log_ndtr((self._detection_threshold_dbm - expected_rssi) / self.rssi_sigma_db),
                )

        return log_lik

    def estimate_from_log_posterior(
        self,
        log_posterior: np.ndarray,
        num_aps_used: int = 0,
    ) -> LocalisationEstimate:
        """Posterior mean within the most likely point's compartment.

        Averaging only inside that compartment stops a two-room ambiguity from
        producing an estimate on the bulkhead between them. If the compartment
        is not convex the mean can still fall outside it; then the nearest grid
        point inside the compartment is used instead.
        """
        best = int(np.argmax(log_posterior))
        room = self.grid_compartments[best]
        in_room = self.grid_compartments == room
        weights = np.exp(log_posterior - log_posterior[best]) * in_room
        weights /= weights.sum()

        position = weights @ self.grid_points
        offsets = self.grid_points - position
        covariance = (weights[:, None] * offsets).T @ offsets

        if self.environment.compartment_at((float(position[0]), float(position[1]))) != room:
            room_points = self.grid_points[in_room]
            position = room_points[np.argmin(np.sum((room_points - position) ** 2, axis=1))]

        return LocalisationEstimate(
            position=(float(position[0]), float(position[1])),
            covariance=covariance,
            num_aps_used=num_aps_used,
        )

    def locate(self, scan: Scan) -> Optional[LocalisationEstimate]:
        heard = sum(1 for ap in self.access_points if scan.get(ap) is not None)
        if heard == 0:
            return None
        return self.estimate_from_log_posterior(self.log_likelihood(scan), num_aps_used=heard)
