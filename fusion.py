"""Sensor fusion trackers: Wi-Fi (RSSI + RTT) likelihood fused with phone PDR.

ParticleTracker is a map-matching particle filter: particles move by the
phone's reported step count / step length / heading change, are killed if
their move crosses a wall, and are weighted by the WallAwareLocaliser
likelihood of each Wi-Fi scan. See docs/sensor_fusion_spec.md.

Design basis (research_notes/Shipboard sensor fusion methods/
fusion_filters_validation.md, "FFV" below; pdr_imu_error_models.md, "PDR";
research_notes/Better shipboard WiFi localisers/bayesian_tracking.md, "BT"):

- PDR as the proposal, Wi-Fi as a per-particle likelihood ("tight" coupling at
  the RSS/RTT level, as De Cock 2021, Raja & Groves 2025; FFV Q1).
- State (x, y, heading, step-length scale). The phone gyro only measures
  heading *changes*, so the absolute heading is a latent per-particle state
  with a uniform prior (Woodman & Harle 2008; FFV Q2). The step-length scale
  absorbs the per-walk bias of the step-length model and, as in De Cock, is
  inherited with (almost) no noise (FFV Q1).
- A small share of particles moves with a heading-free (isotropic) step, so a
  wrong heading hypothesis cannot lock the filter (FFV Q2 inferences), and a
  small share stays put, covering false steps from ship motion (PDR Q5) and
  gross PDR anomalies (Raja & Groves 2025, PDR Q6).
- Walls are hard constraints: a move that crosses a wall is re-proposed with
  fresh noise a few times (Harder et al. 2022's bounded retries, FFV Q1) and
  the particle is killed (weight 0, Woodman & Harle 2008) if every retry
  crosses. If that would kill every particle, the moves are rejected instead
  (particles stay where they were), so the filter cannot die against a
  bulkhead (De Cock's "pushed into walls and trapped" failure, FFV Q1).
  A move whose straight chord clips a door jamb is accepted if a wall-free
  two-leg route via that jamb is at most `corner_slack_m` longer (see
  `_move_blocked`); this is still only passage through a doorway.
- Systematic resampling when ESS < N/2 (Raja & Groves 2025, FFV Q1) with a
  small wall-respecting roughening jitter (FFV Q1 inferences).
- Augmented-MCL style recovery: at each resampling, a fraction of particles
  is redrawn from the Wi-Fi-only posterior of the current scan ("sensor
  resetting"), at a small constant rate plus max(0, 1 - w_fast / w_slow)
  (Thrun et al., Probabilistic Robotics; FFV Q1 "Recovery").
- Initialisation from the whole-deck Wi-Fi posterior of the first scan
  (Woodman & Harle 2008/2009: an RF prior; Raja & Groves list poor, local
  initialisation as a main failure mode; FFV Q1).
- Estimate: weighted mean of the particles in the compartment holding the
  most weight (De Cock's dominant-cluster rule at compartment granularity,
  FFV Q1 "Output"), with their covariance plus a calibrated floor (FFV Q6).
"""
import math
from random import Random
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point

from imu import ImuModel, MotionReading
from network import Network, Scan
from positioning import LocalisationEstimate, WallAwareLocaliser

_X, _Y, _HEADING, _SCALE = range(4)
_STANDING_SIGMA_M = 0.05  # per-interval sway of someone standing (or shuffling) in place


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class ParticleTracker:
    """Map-matching particle filter fusing Wi-Fi likelihood with PDR.

    Tracker protocol (see tracking.evaluate_tracking): reset(), update(scan, motion).
    All randomness comes from `rng`; reset() must make a walk's result
    independent of what ran before (re-derive the internal generator from the
    seed drawn at construction).

    `particles` is an (n_particles, 4) array of x, y (m), absolute walking
    heading (rad, x-axis = 0, CCW +) and step-length scale; `weights` sums to 1.
    `imu_model` describes the phone's PDR errors (None: realistic ImuModel()
    defaults). With `use_magnetometer`, magnetic headings in the readings
    weight particles by heading agreement, trusting the compass as far as
    `imu_model`'s magnetometer noise and bias say.

    The keyword-only parameters after `scan_interval_s` are algorithm settings.
    Their defaults were tuned on the spec's tuning seeds only (walks 67890,
    scans 78901, IMU 89012); see the comments at each.
    """

    delay_steps = 0

    def __init__(
        self,
        localiser: WallAwareLocaliser,
        network: Network,
        rng: Random,
        n_particles: int = 2000,
        imu_model: Optional[ImuModel] = None,
        use_magnetometer: bool = False,
        scan_interval_s: float = 1.0,
        *,
        # Provenance key: [T] tuned on the spec's tuning seeds only (walks Random(67890),
        # scans Random(78901), IMU Random(89012), 6 walks x 240 s, deck / miscalibrated /
        # harsh / 15 s outage / magnetometer runs); [L] literature value, not tuned;
        # [S] structural choice, checked on the tuning seeds.
        walking_speed_mps: float = 1.4,              # [S] same as GridTracker's default
        heading_free_fraction: float = 0.03,         # [T] 0.1 / 0.03 / 0.01 tried; 0.1 kept heading mass ~0.5
        stationary_fraction: float = 0.05,           # [T] 0.05 vs 0.02: no clear difference; covers false steps
        heading_noise_floor_rad: float = math.radians(2.0),   # [L] PDR Q7 default per-step direction noise
        heading_drift_floor_rad: float = math.radians(0.5),   # [S] keeps heading diversity; not swept
        step_noise_floor_m: float = 0.08,            # [L] PDR Q7 default per-step SD; 0.15 (Woodman/De Cock) tried, no gain
        step_scale_std: float = 0.05,                # [L] PDR Q2/Q7 default 5% per-walk step-length bias
        wall_retries: int = 5,                       # [L/T] Harder et al. used 8; 5 vs 10 equal on tuning seeds
        likelihood_exponent: float = 0.6,            # [T] 1.0 / 0.7 / 0.6 / 0.5 tried; 1.0 lost room accuracy (harsh)
        all_missing_exponent: float = 0.2,           # [S] as the test architect's prototype; misses share a cause
        no_imu_likelihood_exponent: float = 1.0,     # [T] 0.6 vs 1.0 with motion None (tests/test_fusion.py tuning walks)
        injection_fraction: float = 0.003,           # [T] 0.01 vs 0.003; FFV Q1 suggests 0.5-1% floor
        injection_alpha_fast: float = 0.1,           # [L] ROS AMCL suggestion (FFV Q1 Gaps, unverified there)
        injection_alpha_slow: float = 0.001,         # [L] ditto
        max_injection_fraction: float = 0.5,         # [S] not swept
        injection_inherit_fraction: float = 1.0,     # [T] 0.8 vs 1.0; uniform headings eroded converged heading
        roughening_m: float = 0.05,                  # [L] FFV Q1 inferences (0.05 m after resampling)
        corner_slack_m: float = 0.5,                 # [T] 0 (off) / 0.3 / 0.5 / 1.0 tried; 0.3-1.0 all similar
        free_walk_stay_fraction: float = 0.5,        # [T] 0 / 0.5 / 0.8 with motion None; BT Q4 "stay-put mass"
        step_quantisation_var: float = 1.0 / 6.0,    # [S] steps^2; difference of two U(0,1) step phases. [T] check: 0 / 1/6 / 1/3, 1/3 worse in outage
        position_floor_std_m: float = 0.0,           # [T] coverage95 already 0.95-0.98 on the deck; no inflation
    ):
        if n_particles <= 0:
            raise ValueError("n_particles must be positive")
        if scan_interval_s <= 0 or walking_speed_mps <= 0:
            raise ValueError("scan_interval_s and walking_speed_mps must be positive")
        for name, value in (("heading_free_fraction", heading_free_fraction),
                            ("stationary_fraction", stationary_fraction),
                            ("injection_fraction", injection_fraction),
                            ("max_injection_fraction", max_injection_fraction),
                            ("injection_inherit_fraction", injection_inherit_fraction),
                            ("free_walk_stay_fraction", free_walk_stay_fraction)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if heading_free_fraction + stationary_fraction > 1.0:
            raise ValueError("heading_free_fraction + stationary_fraction must not exceed 1")
        if (not 0.0 < likelihood_exponent <= 1.0 or not 0.0 < no_imu_likelihood_exponent <= 1.0
                or not 0.0 <= all_missing_exponent <= 1.0):
            raise ValueError("likelihood exponents must be in (0, 1] (all-missing: [0, 1])")
        if not 0.0 <= injection_alpha_slow <= injection_alpha_fast <= 1.0:
            raise ValueError("need 0 <= injection_alpha_slow <= injection_alpha_fast <= 1")
        if wall_retries < 0 or min(heading_noise_floor_rad, heading_drift_floor_rad, step_noise_floor_m, step_scale_std, roughening_m, corner_slack_m, step_quantisation_var, position_floor_std_m) < 0:
            raise ValueError("wall_retries and noise settings must be non-negative")

        self.localiser = localiser
        self.network = network
        self.environment = network.environment
        self.n_particles = int(n_particles)
        self.imu_model = imu_model if imu_model is not None else ImuModel()
        self.use_magnetometer = use_magnetometer
        self.scan_interval_s = scan_interval_s
        self.walking_speed_mps = walking_speed_mps
        self.heading_free_fraction = heading_free_fraction
        self.stationary_fraction = stationary_fraction
        self.heading_noise_floor_rad = heading_noise_floor_rad
        self.heading_drift_floor_rad = heading_drift_floor_rad
        self.step_noise_floor_m = step_noise_floor_m
        self.step_scale_std = step_scale_std
        self.wall_retries = int(wall_retries)
        self.likelihood_exponent = likelihood_exponent
        self.all_missing_exponent = all_missing_exponent
        self.no_imu_likelihood_exponent = no_imu_likelihood_exponent
        self.injection_fraction = injection_fraction
        self.injection_alpha_fast = injection_alpha_fast
        self.injection_alpha_slow = injection_alpha_slow
        self.max_injection_fraction = max_injection_fraction
        self.injection_inherit_fraction = injection_inherit_fraction
        self.roughening_m = roughening_m
        self.corner_slack_m = corner_slack_m
        self.free_walk_stay_fraction = free_walk_stay_fraction
        self.step_quantisation_var = step_quantisation_var
        self.position_floor_std_m = position_floor_std_m

        self._build_cell_lookup()
        self._jambs = self._find_jambs()
        self._seed = rng.getrandbits(64)
        self.reset()

    # ----------------------------------------------------------------- set-up

    def _build_cell_lookup(self) -> None:
        """Dense raster over the grid's bounding box mapping (x, y) to the nearest
        localiser grid point, so a particle's likelihood and compartment are an
        O(1) array lookup instead of a nearest-neighbour query per update."""
        points = self.localiser.grid_points
        if len(points) > 1:
            distances, _ = cKDTree(points).query(points[: min(len(points), 200)], k=2)
            spacing = float(np.median(distances[:, 1]))
        else:
            spacing = 1.0
        origin = points.min(axis=0) - spacing / 2.0
        shape = np.ceil((points.max(axis=0) + spacing / 2.0 - origin) / spacing).astype(int) + 1
        ix, iy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        centres = origin + (np.column_stack([ix.ravel(), iy.ravel()]) + 0.5) * spacing
        _, nearest = cKDTree(points).query(centres)
        self._cell_spacing = spacing
        self._cell_origin = origin
        self._cell_lookup = nearest.reshape(shape[0], shape[1])
        names = sorted(set(self.localiser.grid_compartments))
        self._room_names = names
        index = {name: i for i, name in enumerate(names)}
        self._cell_room = np.array([index[name] for name in self.localiser.grid_compartments])

    def _cells(self, xy: np.ndarray) -> np.ndarray:
        """Localiser grid index of each (x, y)."""
        idx = np.floor((xy - self._cell_origin) / self._cell_spacing).astype(int)
        ix = np.clip(idx[:, 0], 0, self._cell_lookup.shape[0] - 1)
        iy = np.clip(idx[:, 1], 0, self._cell_lookup.shape[1] - 1)
        return self._cell_lookup[ix, iy]

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self.particles = np.zeros((self.n_particles, 4))
        self.particles[:, _SCALE] = 1.0
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
        self._initialised = False
        self._w_fast = None
        self._w_slow = None

    # ------------------------------------------------------------ the filter

    def update(self, scan: Optional[Scan], motion: Optional[MotionReading] = None) -> LocalisationEstimate:
        heard = 0 if scan is None else sum(1 for ap in self.localiser.access_points if scan.get(ap) is not None)
        log_lik = None
        if scan is not None:
            if not heard:
                exponent = self.all_missing_exponent
            elif motion is None:
                exponent = self.no_imu_likelihood_exponent
            else:
                exponent = self.likelihood_exponent
            log_lik = exponent * self.localiser.log_likelihood(scan)

        initialising = not self._initialised
        if initialising:
            self._initialise(log_lik)
        else:
            self._predict(motion)
        if self.use_magnetometer and motion is not None and motion.magnetic_heading_rad is not None:
            self._weight_by_compass(motion.magnetic_heading_rad)

        cells = self._cells(self.particles[:, :2])
        if log_lik is not None:
            # The initialising scan is already in the particles (they were drawn from its
            # posterior); weighting by it again would count it twice (a likelihood^2 belief).
            self._weight_by(log_lik[cells], update_monitor=heard > 0, apply=not initialising)

        estimate = self._estimate(cells, heard)

        ess = 1.0 / float(np.sum(self.weights ** 2))
        if ess < 0.5 * self.n_particles:
            self._resample(log_lik if heard else None)
        return estimate

    def _initialise(self, log_lik: Optional[np.ndarray]) -> None:
        """Sample the whole-deck Wi-Fi posterior (uniform over the grid without a scan)."""
        if log_lik is None:
            log_lik = np.zeros(len(self.localiser.grid_points))
        self.particles = self._sample_from_wifi(log_lik, self.n_particles)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
        self._initialised = True

    def _sample_from_wifi(self, log_lik: np.ndarray, k: int) -> np.ndarray:
        """k fresh particles: grid cells drawn by posterior mass, uniform within the cell,
        heading uniform (unknown), step-length scale from its prior."""
        p = np.exp(log_lik - log_lik.max())
        cumulative = np.cumsum(p)
        cells = np.searchsorted(cumulative, self._rng.random(k) * cumulative[-1], side="right")
        cells = np.minimum(cells, len(p) - 1)
        half = self._cell_spacing / 2.0
        xy = self.localiser.grid_points[cells] + self._rng.uniform(-half, half, (k, 2))
        out = np.empty((k, 4))
        out[:, :2] = xy
        out[:, _HEADING] = self._rng.uniform(-np.pi, np.pi, k)
        out[:, _SCALE] = 1.0 + self._step_scale_prior_std() * self._rng.standard_normal(k)
        return out

    def _step_scale_prior_std(self) -> float:
        """Prior spread of the per-walk step-length scale. Uses the ImuModel's step-length
        bias if it has one, else `step_scale_std` (5%: the PDR notes' default per-walk
        multiplicative bias, Weinberg/Kim models, PDR Q2)."""
        return max(getattr(self.imu_model, "step_length_bias_std_frac", 0.0), self.step_scale_std)

    # --- predict

    def _predict(self, motion: Optional[MotionReading]) -> None:
        n = self.n_particles
        old = self.particles[:, :2].copy()
        if motion is None:
            displacement_fn, heading_fn = self._free_walk_model()
        else:
            displacement_fn, heading_fn = self._pdr_model(motion)

        new_heading = heading_fn(np.arange(n))
        new_xy = old + displacement_fn(np.arange(n), new_heading)
        moved = np.any(new_xy != old, axis=1)
        blocked = np.zeros(n, dtype=bool)
        blocked[moved] = self._move_blocked(old[moved], new_xy[moved])
        for _ in range(self.wall_retries):  # Harder et al. 2022: bounded re-proposal
            if not blocked.any():
                break
            idx = np.flatnonzero(blocked)
            heading = heading_fn(idx)
            proposal = old[idx] + displacement_fn(idx, heading)
            still = self._move_blocked(old[idx], proposal)
            ok = idx[~still]
            new_xy[ok] = proposal[~still]
            new_heading[ok] = heading[~still]
            blocked[ok] = False

        if blocked.any():
            new_xy[blocked] = old[blocked]  # rejected: the particle does not cross
            if np.any(self.weights[~blocked] > 0):
                self.weights = np.where(blocked, 0.0, self.weights)  # killed (Woodman & Harle)
                self.weights /= self.weights.sum()
            # else: every particle was blocked; keep them all where they were (rejection)
        self.particles[:, :2] = new_xy
        self.particles[:, _HEADING] = _wrap(new_heading)

    def _crosses_wall(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """True where the straight segment a[i] -> b[i] crosses a wall."""
        if len(a) == 0 or not self.environment.walls:
            return np.zeros(len(a), dtype=bool)
        return self.environment.crossing_matrix(a, b).any(axis=1)

    def _move_blocked(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """True where no wall-free walking route leads from a[i] to b[i].

        A route is the straight segment, or two straight legs via a door jamb whose
        length exceeds the straight distance by at most `corner_slack_m`. Over one scan
        interval a walker turns round a door jamb (tracking.simulate_walk notes that
        the chord between consecutive true positions can clip a jamb), so the straight
        segment alone would kill exactly the particles that follow the person through a
        door. This mirrors GridTracker's motion kernel on walking distance and Harder et
        al.'s door-transition constraint c_m (FFV Q1). Both legs must be wall-free, so a
        particle still only reaches another compartment through a doorway.
        """
        blocked = self._crosses_wall(a, b)
        if not blocked.any() or len(self._jambs) == 0 or self.corner_slack_m <= 0:
            return blocked
        idx = np.flatnonzero(blocked)
        a_b, b_b = a[idx], b[idx]
        direct = np.linalg.norm(b_b - a_b, axis=1)
        via = (np.linalg.norm(self._jambs[None, :, :] - a_b[:, None, :], axis=2)
               + np.linalg.norm(b_b[:, None, :] - self._jambs[None, :, :], axis=2))
        rows, cols = np.nonzero(via <= direct[:, None] + self.corner_slack_m)
        if len(rows) == 0:
            return blocked
        corner = self._jambs[cols]
        clear = ~(self._crosses_wall(a_b[rows], corner) | self._crosses_wall(corner, b_b[rows]))
        blocked[idx[np.unique(rows[clear])]] = False
        return blocked

    def _find_jambs(self) -> np.ndarray:
        """Door jambs: wall endpoints touching no other wall, nudged 5 cm into the gap."""
        walls = self.environment.walls
        jambs = []
        for i, wall in enumerate(walls):
            p1, p2 = np.array(wall.p1, dtype=float), np.array(wall.p2, dtype=float)
            length = float(np.linalg.norm(p2 - p1))
            if length == 0:
                continue
            for end, other in ((p1, p2), (p2, p1)):
                point = Point(end)
                if any(j != i and walls[j].geometry.distance(point) < 1e-6 for j in range(len(walls))):
                    continue
                jambs.append(end + 0.05 * (end - other) / length)
        return np.array(jambs, dtype=float).reshape(-1, 2)

    def _free_walk_model(self):
        """No IMU reading: BT Q4's no-IMU model, a stay-put mass plus an isotropic Gaussian
        step with sigma = half the walking speed x interval (GridTracker's kernel width).
        Crew spend much of their time standing, so a share of particles only sways
        (`free_walk_stay_fraction`). The heading is forgotten (the turn is unknown)."""
        sigma = 0.5 * self.walking_speed_mps * self.scan_interval_s
        heading_sigma = math.pi / 4
        stay = self._rng.random(self.n_particles) < self.free_walk_stay_fraction

        def heading_fn(idx):
            return self.particles[idx, _HEADING] + heading_sigma * self._rng.standard_normal(len(idx))

        def displacement_fn(idx, heading):
            scale = np.where(stay[idx], _STANDING_SIGMA_M, sigma)
            return scale[:, None] * self._rng.standard_normal((len(idx), 2))

        return displacement_fn, heading_fn

    def _pdr_model(self, motion: MotionReading):
        """Proposal from one PDR reading, with the error terms of the ImuModel.

        Distance = steps x reported step length x the particle's step-length scale, plus
        independent per-step length noise (sqrt(n) x sigma_step; De Cock / Woodman use
        0.15 m filter-side, the PDR notes' default per-step SD is 0.08 m) and the step-count
        errors (missed and extra steps: binomial variance per step; spurious steps from ship
        rocking: Poisson variance rate x dt) and the quantisation of a whole-step counter read
        at scan epochs (variance 1/6 step^2), each worth one step length.

        Heading has two error kinds (PDR Q3, Q7). Accumulating gyro errors (angle random
        walk x sqrt(dt), residual bias x dt, scale factor x the turn) go into the particle's
        heading state as a random walk. The walking direction's own per-step error (2 deg
        per step) does not accumulate, so it perturbs only this interval's move direction
        (std / sqrt(steps)), never the stored heading. Gross heading outliers are covered by
        the heading-free share of particles; false steps while standing by the stationary
        share (see module doc).
        """
        model = self.imu_model
        dt = motion.interval_s if motion.interval_s > 0 else self.scan_interval_s
        steps = max(int(motion.step_count), 0)
        step_m = max(float(motion.step_length_m), 0.0)
        turn = float(motion.heading_change_rad)

        miss = getattr(model, "step_miss_probability", 0.0)
        extra = getattr(model, "false_step_probability", 0.0)
        false_rate = getattr(model, "false_steps_per_s", 0.0)
        step_count_var = steps * (miss * (1 - miss) + extra * (1 - extra)) + false_rate * dt
        if steps > 0:
            # Whole-step counter read at scan epochs: the fractional step carries over, so the
            # interval's distance is off by (f_prev - f_now) x L with f ~ U(0, 1): variance L^2/6.
            step_count_var += self.step_quantisation_var
        step_sigma = max(getattr(model, "step_length_noise_m", 0.0), self.step_noise_floor_m)
        distance_sigma = math.sqrt(steps * step_sigma ** 2 + step_count_var * step_m ** 2)
        drift_sigma = math.sqrt(getattr(model, "gyro_noise_rad_per_sqrt_s", 0.0) ** 2 * dt
                                + ((getattr(model, "gyro_bias_std_rad_s", 0.0)
                                    + getattr(model, "ship_yaw_rate_std_rad_s", 0.0)) * dt) ** 2
                                + (getattr(model, "gyro_scale_factor_error_std", 0.0) * turn) ** 2
                                + self.heading_drift_floor_rad ** 2)
        direction_sigma = (max(getattr(model, "heading_noise_per_step_rad", 0.0), self.heading_noise_floor_rad)
                           / math.sqrt(max(steps, 1)))
        mean_step = steps * step_m
        standing_sigma = _STANDING_SIGMA_M

        # Mixture component per particle, fixed for this interval (retries re-draw noise only)
        u = self._rng.random(self.n_particles)
        heading_free = u < self.heading_free_fraction
        stationary = (u >= self.heading_free_fraction) & (u < self.heading_free_fraction + self.stationary_fraction)
        if steps == 0:
            heading_free[:] = False  # no step: nothing to point anywhere

        def heading_fn(idx):
            h = self.particles[idx, _HEADING] + turn + drift_sigma * self._rng.standard_normal(len(idx))
            free = heading_free[idx]
            h[free] = self._rng.uniform(-np.pi, np.pi, int(free.sum()))
            return h

        def displacement_fn(idx, heading):
            k = len(idx)
            if steps == 0:
                return standing_sigma * self._rng.standard_normal((k, 2))
            distance = np.abs(mean_step * self.particles[idx, _SCALE] + distance_sigma * self._rng.standard_normal(k))
            direction = heading + direction_sigma * self._rng.standard_normal(k)
            d = np.column_stack([distance * np.cos(direction), distance * np.sin(direction)])
            stay = stationary[idx]
            d[stay] = standing_sigma * self._rng.standard_normal((int(stay.sum()), 2))
            return d

        return displacement_fn, heading_fn

    # --- update

    def _weight_by(self, log_lik_particles: np.ndarray, update_monitor: bool, apply: bool = True) -> None:
        """Multiply the weights by the scan likelihood (unless `apply` is False) and update the
        augmented-MCL monitor."""
        log_w = np.log(np.maximum(self.weights, 1e-300)) + log_lik_particles
        log_w[self.weights <= 0] = -np.inf
        top = log_w.max()
        w = np.exp(log_w - top)
        total = w.sum()
        if not np.isfinite(total) or total <= 0:
            return
        if update_monitor:
            # Augmented MCL monitor on the predictive likelihood of the scan, normalised by
            # the best particle so scans with different numbers of APs are comparable.
            live = self.weights > 0
            ratio = float(np.sum(self.weights[live] * np.exp(log_lik_particles[live] - log_lik_particles[live].max())))
            if self._w_fast is None:
                self._w_fast = self._w_slow = ratio
            else:
                self._w_fast += self.injection_alpha_fast * (ratio - self._w_fast)
                self._w_slow += self.injection_alpha_slow * (ratio - self._w_slow)
        if apply:
            self.weights = w / total

    def _weight_by_compass(self, magnetic_heading: float) -> None:
        """Von Mises weighting by heading agreement with the magnetometer, with the spread
        the ImuModel claims (white noise and steel-hull bias added in quadrature)."""
        model = self.imu_model
        sigma = math.hypot(getattr(model, "magnetometer_noise_rad", 0.0), getattr(model, "magnetometer_bias_std_rad", 0.0))
        kappa = 1.0 / max(sigma, math.radians(1.0)) ** 2
        log_w = kappa * (np.cos(self.particles[:, _HEADING] - magnetic_heading) - 1.0)
        live = self.weights > 0
        w = np.where(live, self.weights * np.exp(log_w), 0.0)
        if w.sum() > 0:
            self.weights = w / w.sum()

    def _resample(self, log_lik: Optional[np.ndarray]) -> None:
        """Systematic resampling, roughening jitter (never across a wall), and injection of
        particles from the Wi-Fi-only posterior when the scan heard at least one AP."""
        n = self.n_particles
        positions = (self._rng.random() + np.arange(n)) / n
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        idx = np.minimum(np.searchsorted(cumulative, positions, side="right"), n - 1)
        self.particles = self.particles[idx].copy()

        if self.roughening_m > 0:
            old = self.particles[:, :2].copy()
            new = old + self.roughening_m * self._rng.standard_normal((n, 2))
            bad = self._crosses_wall(old, new)
            new[bad] = old[bad]
            self.particles[:, :2] = new

        if log_lik is not None:
            fraction = self.injection_fraction
            if self._w_fast is not None and self._w_slow > 0:
                fraction += max(0.0, 1.0 - self._w_fast / self._w_slow)
            k = int(round(min(fraction, self.max_injection_fraction) * n))
            if k > 0:
                # Positions from the Wi-Fi posterior; heading and step scale copied from the
                # (resampled) cloud for a share of them, so routine injection does not erode a
                # converged heading, and uniform for the rest, so a wrong one can be replaced.
                slots = self._rng.choice(n, size=k, replace=False)
                fresh = self._sample_from_wifi(log_lik, k)
                inherit = self._rng.random(k) < self.injection_inherit_fraction
                donors = self._rng.integers(0, n, int(inherit.sum()))
                fresh[inherit, _HEADING] = self.particles[donors, _HEADING]
                fresh[inherit, _SCALE] = self.particles[donors, _SCALE]
                self.particles[slots] = fresh
        self.weights = np.full(n, 1.0 / n)

    # --- output

    def _estimate(self, cells: np.ndarray, heard: int) -> LocalisationEstimate:
        rooms = self._cell_room[cells]
        room_mass = np.bincount(rooms, weights=self.weights, minlength=len(self._room_names))
        best = int(np.argmax(room_mass))
        in_room = rooms == best
        w = np.where(in_room, self.weights, 0.0)
        if w.sum() <= 0:
            w = in_room.astype(float)
        w = w / w.sum()
        xy = self.particles[:, :2]
        mean = w @ xy
        offsets = xy - mean
        covariance = (w[:, None] * offsets).T @ offsets
        covariance = 0.5 * (covariance + covariance.T)
        # Floor: quantisation of the likelihood to grid cells (uniform within a cell,
        # variance spacing^2 / 12) plus model error the particle spread cannot see.
        floor = self._cell_spacing ** 2 / 12.0 + self.position_floor_std_m ** 2
        covariance = covariance + floor * np.eye(2)

        position = (float(mean[0]), float(mean[1]))
        if self.environment.compartment_at(position) != self._room_names[best]:
            room_xy = xy[in_room]
            nearest = room_xy[np.argmin(np.sum((room_xy - mean) ** 2, axis=1))]
            position = (float(nearest[0]), float(nearest[1]))
        return LocalisationEstimate(position=position, covariance=covariance, num_aps_used=heard)
