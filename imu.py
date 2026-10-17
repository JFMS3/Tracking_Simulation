"""Simulated smartphone inertial sensing (pedestrian dead reckoning, PDR) between Wi-Fi scans.

A phone carried by a crew member reports, for every scan interval, how many
steps it detected, its own estimate of step length, and how much the walking
heading turned (integrated gyroscope). Optionally it also reports an absolute
magnetometer heading, which is badly biased inside a steel hull.

simulate_imu turns a true walk (one position per scan, as produced by
tracking.simulate_walk) into the MotionReadings a phone would report, with
literature-based error models. See docs/sensor_fusion_spec.md.

Error model (step level, one walk = one phone session):
- Steps: true steps come from the true distance and true_step_length_m with the
  fractional step carried over. Each true step is missed with
  step_miss_probability; false steps are added per true step while walking
  (false_step_probability) and as a Poisson process in time, also while
  standing (false_steps_per_s: ship motion, phone handling).
- Step length: each detected step's length estimate is the phone's step-length
  model output, true_step_length_m x (1 + per-walk bias) + independent per-step
  noise. It never depends on the true distance of the interval, so distance
  information comes only from the quantised step count (a real step counter has
  no sub-step information).
- Heading change: gyro = (1 + per-walk scale-factor error) x true turn
  + (per-walk constant gyro bias + per-walk ship yaw rate) x dt + angle random
  walk; plus a non-accumulating per-step walking-direction error (body sway,
  phone wobble) and rare gross heading outliers, which enter the reported
  change as the difference between consecutive intervals' direction errors.
- Magnetometer (off by default): true heading + a spatially correlated bias
  field (fixed in space, drawn per walk) + white noise.

A constant phone-to-walking-direction misalignment (carry pose) is not
modelled: it cancels in relative heading changes and is absorbed by the
filter's unknown absolute heading. Gyro bias instability (5 deg/h default,
PMC10490716) is not modelled: over a 10 min walk it is < 10% of the residual
constant bias.

Parameter sources (see research_notes/Shipboard sensor fusion methods/
pdr_imu_error_models.md, Q7 table, for the optimistic/default/conservative sets):
- [Kamnik21] Vezocnik/Kamnik et al., Sensors 2021: per-step length SD 5.5-7.4 cm.
- [RS19] Remote Sens. 11(9):1140 (2019); [EL-SLE] Sensors 2022 (PMC9501393):
  Weinberg/Kim walk-level distance error 5.0-6.4%.
- [BH13] Brajdic & Harle, UbiComp 2013; ORA Oxford step-count study; ASTESJ 2020:
  step-count error 1.3-3.5% walking, placement accuracy 96.7-99%.
- [PMC10490716] Smartphone MEMS gyro lab tests (Sensors 2023): ARW 0.65-5 deg/sqrt(h),
  scale factor mostly 0.05-1.5% (up to 2.6%).
- [WH08] Woodman & Harle 2008 (0.6 deg/step, foot mounted); [DL-PDR] Sensors 2023
  (PMC10708641): 1.6-2.2 deg single-step heading error.
- [RG25] Raja & Groves, J. Navigation 2025: PDR anomalies (single bad orientation readings).
"""
import math
from dataclasses import dataclass
from random import Random
from typing import Dict, List, Optional, Sequence, Tuple

from environment import Coord

_MOVING_EPS_M = 1e-9      # chords shorter than this count as standing still
_STEP_PHASE_EPS = 1e-9    # guards floor(distance / step) against round-off at exact step boundaries
_U_TURN_EPS_RAD = 1e-7    # turns this close to +-pi are U-turns of ambiguous sense


@dataclass(frozen=True)
class MotionReading:
    """What the phone's step detector and gyroscope report for one scan interval.

    Reading t covers the interval ending at epoch t (from position t-1 to t);
    reading 0 covers no motion. Contains only what a real phone could know:
    never the true displacement.
    """

    step_count: int
    step_length_m: float          # the phone's per-step length estimate for this interval
    heading_change_rad: float     # gyro-integrated change in walking heading over the interval
    magnetic_heading_rad: Optional[float] = None  # absolute heading (x-axis = 0, CCW +), None if unavailable
    interval_s: float = 1.0

    @property
    def distance_m(self) -> float:
        return self.step_count * self.step_length_m


@dataclass
class ImuModel:
    """Error model for simulated PDR. Defaults are the notes' DEFAULT set (handheld commodity phone).

    Naming rule: every error parameter's name contains "noise", "bias",
    "probability", "per_s", "std" or "error", so a model with all of those set
    to 0 (and the magnetometer off) is a perfect phone.
    """

    # Mean true step length: ~0.70 m (0.6-0.8 m), consistent with Raja & Groves'
    # 0.67 m step markers and 0.94-1.43 m/s walking speed [notes Q2].
    true_step_length_m: float = 0.7
    # Independent per-step length noise: 0.08 m, [Kamnik21] per-step SD 5.5-7.4 cm
    # (notes Q7 default; 0.15 m is the filter-side value of Woodman/De Cock).
    step_length_noise_m: float = 0.08
    # Missed-step probability per true step: 3%, [BH13] 1.3-3.5% count error walking,
    # 96.7-99% placement accuracy (notes Q7 default).
    step_miss_probability: float = 0.03
    # Spurious steps as a Poisson process in time, walking or standing (ship rocking,
    # balance corrections, phone handling): 1 burst/min x 4 steps/burst = 0.067 /s.
    # ASSUMPTION (notes Q1/Q7), anchored only to ~14% error under "false walking" (ORA)
    # and vendor reports of false steps on rocking ship decks (Huawei support).
    false_steps_per_s: float = 4.0 / 60.0
    # Residual constant gyro bias after on-device calibration, drawn per walk:
    # sigma 0.01 deg/s. ASSUMPTION (notes Q3/Q7: inference, not directly sourced).
    gyro_bias_std_rad_s: float = math.radians(0.01)
    # Gyro angle random walk: 1.0 deg/sqrt(h) = 1/60 deg/sqrt(s), [PMC10490716] in-phone
    # Allan variance 0.65-5 deg/sqrt(h); datasheets 0.23-0.42 deg/sqrt(h) (notes Q3/Q7 default).
    gyro_noise_rad_per_sqrt_s: float = math.radians(1.0) / 60.0
    # No reliable absolute heading inside a steel hull: magnetometer not used by
    # default (notes Q4/Q7; De Cock 2021; Afzal 2011).
    magnetometer_available: bool = False
    # Spatially correlated magnetometer heading error: sigma 30 deg (20-45 deg).
    # ASSUMPTION (notes Q4): no ship measurements exist.
    magnetometer_bias_std_rad: float = math.radians(30.0)
    # Magnetometer white noise ~3 deg. ASSUMPTION (notes Q4).
    magnetometer_noise_rad: float = math.radians(3.0)

    # Per-walk multiplicative step-length model bias: sigma 5%, [RS19]/[EL-SLE]
    # Weinberg/Kim 5.0-6.4% walk-level distance error (notes Q2/Q7 default).
    step_length_bias_std_frac: float = 0.05
    # Extra detections per true step while walking: 1.5% (notes Q1/Q7 default,
    # FIELD net count errors [BH13]).
    false_step_probability: float = 0.015
    # Per-walk gyro scale-factor error: sigma 1%, [PMC10490716] mostly 0.05-1.5%
    # (notes Q3/Q7 default). Matters on ships: many 90 deg turns through doors.
    gyro_scale_factor_error_std: float = 0.01
    # Non-accumulating per-step walking-direction error: 2 deg, [DL-PDR] 1.6-2.2 deg,
    # [WH08] 0.6 deg foot mounted (notes Q3/Q7 default).
    heading_noise_per_step_rad: float = math.radians(2.0)
    # Gross single-step heading outliers (phone handling, pocket transitions):
    # 0.5% per step, 20-90 deg, random sense. ASSUMPTION (notes Q6/Q7, after [RG25]).
    heading_outlier_probability: float = 0.005
    heading_outlier_min_rad: float = math.radians(20.0)
    heading_outlier_max_rad: float = math.radians(90.0)
    # Ship yaw rate seen by the gyro, drawn per walk: 0 by default because positions
    # are tracked in the ship frame with the ship's yaw rate subtracted (notes Q5
    # mitigation 1). Without that feed the notes' assumption is 0.1 deg/s for 10%
    # of walks (default) to 0.5 deg/s for 20% of walks (conservative).
    ship_yaw_rate_std_rad_s: float = 0.0
    # Correlation length of the magnetometer heading-error field: 2 m (1-3 m).
    # ASSUMPTION (notes Q4).
    magnetometer_correlation_length_m: float = 2.0
    magnetometer_field_components: int = 64

    def __post_init__(self) -> None:
        positive = {
            "true_step_length_m": self.true_step_length_m,
            "magnetometer_correlation_length_m": self.magnetometer_correlation_length_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")

        non_negative = {
            "step_length_noise_m": self.step_length_noise_m,
            "step_length_bias_std_frac": self.step_length_bias_std_frac,
            "false_steps_per_s": self.false_steps_per_s,
            "gyro_bias_std_rad_s": self.gyro_bias_std_rad_s,
            "gyro_noise_rad_per_sqrt_s": self.gyro_noise_rad_per_sqrt_s,
            "gyro_scale_factor_error_std": self.gyro_scale_factor_error_std,
            "heading_noise_per_step_rad": self.heading_noise_per_step_rad,
            "ship_yaw_rate_std_rad_s": self.ship_yaw_rate_std_rad_s,
            "magnetometer_bias_std_rad": self.magnetometer_bias_std_rad,
            "magnetometer_noise_rad": self.magnetometer_noise_rad,
        }
        for name, value in non_negative.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite number")

        probabilities = {
            "step_miss_probability": self.step_miss_probability,
            "false_step_probability": self.false_step_probability,
            "heading_outlier_probability": self.heading_outlier_probability,
        }
        for name, value in probabilities.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        if not (
            math.isfinite(self.heading_outlier_min_rad)
            and math.isfinite(self.heading_outlier_max_rad)
            and 0.0 <= self.heading_outlier_min_rad <= self.heading_outlier_max_rad <= math.pi
        ):
            raise ValueError("heading outliers need 0 <= heading_outlier_min_rad <= heading_outlier_max_rad <= pi")
        if not isinstance(self.magnetometer_available, bool):
            raise ValueError("magnetometer_available must be a bool")
        if (
            isinstance(self.magnetometer_field_components, bool)
            or not isinstance(self.magnetometer_field_components, int)
            or self.magnetometer_field_components < 1
        ):
            raise ValueError("magnetometer_field_components must be a positive integer")


# Parameter sets of the notes' Q7 table (pdr_imu_error_models.md); sources as in ImuModel.
IMU_PRESETS: Dict[str, ImuModel] = {
    "optimistic": ImuModel(
        step_length_noise_m=0.05,
        step_miss_probability=0.01,
        false_steps_per_s=0.1 * 3.0 / 60.0,          # 0.1 bursts/min x 2-4 steps (assumption)
        gyro_bias_std_rad_s=math.radians(0.003),
        gyro_noise_rad_per_sqrt_s=math.radians(0.4) / 60.0,
        magnetometer_bias_std_rad=math.radians(20.0),
        step_length_bias_std_frac=0.025,
        false_step_probability=0.005,
        gyro_scale_factor_error_std=0.003,
        heading_noise_per_step_rad=math.radians(0.6),
        heading_outlier_probability=0.001,
    ),
    "default": ImuModel(),
    "conservative": ImuModel(
        step_length_noise_m=0.15,
        step_miss_probability=0.08,
        false_steps_per_s=5.0 * 5.0 / 60.0,          # 5 bursts/min (heavy sea) x 2-8 steps (assumption)
        gyro_bias_std_rad_s=math.radians(0.05),
        gyro_noise_rad_per_sqrt_s=math.radians(5.0) / 60.0,
        magnetometer_bias_std_rad=math.radians(45.0),
        step_length_bias_std_frac=0.12,
        false_step_probability=0.05,
        gyro_scale_factor_error_std=0.03,
        heading_noise_per_step_rad=math.radians(5.0),
        heading_outlier_probability=0.02,
        # No ship yaw feed: 0.5 deg/s for 20% of walks, as an RMS-matched Gaussian (assumption).
        ship_yaw_rate_std_rad_s=math.radians(0.5) * math.sqrt(0.2),
    ),
}


def _wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _binomial(rng: Random, n: int, p: float) -> int:
    return sum(1 for _ in range(n) if rng.random() < p)


def _poisson(rng: Random, rate: float, duration: float) -> int:
    """Number of events of a Poisson process with `rate` in `duration` (exponential gaps)."""
    if rate <= 0.0:
        return 0
    count, t = 0, rng.expovariate(rate)
    while t < duration:
        count += 1
        t += rng.expovariate(rate)
    return count


class _MagneticBiasField:
    """Heading error of a magnetometer inside a steel hull: a zero-mean Gaussian random field
    over the deck (random Fourier features, Gaussian covariance with the given correlation
    length), fixed in space."""

    def __init__(self, rng: Random, std_rad: float, correlation_length_m: float, components: int) -> None:
        scale = 1.0 / correlation_length_m
        self.frequencies: List[Tuple[float, float]] = [
            (rng.gauss(0.0, scale), rng.gauss(0.0, scale)) for _ in range(components)
        ]
        self.phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in range(components)]
        self.amplitude = std_rad * math.sqrt(2.0 / components)

    def value(self, position: Coord) -> float:
        x, y = position[0], position[1]
        return self.amplitude * sum(
            math.cos(wx * x + wy * y + phase) for (wx, wy), phase in zip(self.frequencies, self.phases)
        )


def simulate_imu(
    walk: Sequence[Coord],
    rng: Random,
    model: Optional[ImuModel] = None,
    scan_interval_s: float = 1.0,
) -> List[MotionReading]:
    """One MotionReading per walk position (same length as `walk`).

    Reading t covers walk[t-1] -> walk[t]. The true walking heading of an
    interval is the chord direction, held while standing still. All randomness
    comes from `rng`; per-walk errors (step-length bias, gyro bias and scale
    factor, ship yaw rate, magnetometer field) are drawn once per call.
    """
    model = model if model is not None else ImuModel()
    if not math.isfinite(scan_interval_s) or scan_interval_s <= 0:
        raise ValueError("scan_interval_s must be a positive finite number")
    if not walk:
        return []
    dt = scan_interval_s
    step_m = model.true_step_length_m

    # per-walk (per-session) errors
    step_scale = 1.0 + rng.gauss(0.0, model.step_length_bias_std_frac)
    gyro_bias = rng.gauss(0.0, model.gyro_bias_std_rad_s)
    gyro_scale = 1.0 + rng.gauss(0.0, model.gyro_scale_factor_error_std)
    ship_yaw_rate = rng.gauss(0.0, model.ship_yaw_rate_std_rad_s)
    magnetic_field = None
    if model.magnetometer_available:
        magnetic_field = _MagneticBiasField(
            rng, model.magnetometer_bias_std_rad,
            model.magnetometer_correlation_length_m, model.magnetometer_field_components,
        )

    # true chord headings (None while standing still) and the facing direction before the first move
    chord_headings: List[Optional[float]] = [None]
    for a, b in zip(walk, walk[1:]):
        moved = math.dist(a, b) > _MOVING_EPS_M
        chord_headings.append(math.atan2(b[1] - a[1], b[0] - a[0]) if moved else None)
    first_heading = next((h for h in chord_headings if h is not None), None)
    true_heading = first_heading
    if magnetic_field is not None and true_heading is None:
        true_heading = rng.uniform(-math.pi, math.pi)

    def magnetic_heading(t: int, heading: float) -> Optional[float]:
        if magnetic_field is None:
            return None
        bias = magnetic_field.value(walk[t])
        return _wrap(heading + bias + rng.gauss(0.0, model.magnetometer_noise_rad))

    model_step_m = step_m * step_scale  # the phone's step-length model, biased per walk
    readings = [MotionReading(0, model_step_m, 0.0, magnetic_heading(0, true_heading), dt)]
    cumulative_m = 0.0
    steps_so_far = 0
    last_move_heading: Optional[float] = None
    direction_error = 0.0  # walking-direction error of the last moving interval

    for t in range(1, len(walk)):
        distance = math.dist(walk[t - 1], walk[t])
        heading = chord_headings[t]

        # true steps, with the fractional step carried over
        cumulative_m += distance
        steps_total = int(math.floor(cumulative_m / step_m + _STEP_PHASE_EPS))
        true_steps = steps_total - steps_so_far
        steps_so_far = steps_total

        # detected steps: misses remove, false detections add (also while standing)
        detected = true_steps - _binomial(rng, true_steps, model.step_miss_probability)
        detected += _binomial(rng, true_steps, model.false_step_probability)
        detected += _poisson(rng, model.false_steps_per_s, dt)
        if detected > 0:
            noise = sum(rng.gauss(0.0, model.step_length_noise_m) for _ in range(detected)) / detected
        else:
            noise = 0.0
        # never uses the true distance of the interval: no sub-step information
        step_length = max(0.0, model_step_m + noise)

        # gyro: scale factor x true turn + (bias + ship yaw) x dt + angle random walk
        true_turn = 0.0
        if heading is not None:
            if last_move_heading is not None:
                true_turn = _wrap(heading - last_move_heading)
                if abs(true_turn) > math.pi - _U_TURN_EPS_RAD:
                    true_turn = math.pi if rng.random() < 0.5 else -math.pi  # U-turn: left or right
            last_move_heading = heading
            true_heading = heading
        heading_change = (
            gyro_scale * true_turn
            + (gyro_bias + ship_yaw_rate) * dt
            + rng.gauss(0.0, model.gyro_noise_rad_per_sqrt_s * math.sqrt(dt))
        )

        # non-accumulating walking-direction error of this interval (mean over its steps)
        if heading is not None:
            n_steps = max(true_steps, 1)
            total = 0.0
            for _ in range(n_steps):
                total += rng.gauss(0.0, model.heading_noise_per_step_rad)
                if rng.random() < model.heading_outlier_probability:
                    spike = rng.uniform(model.heading_outlier_min_rad, model.heading_outlier_max_rad)
                    total += spike if rng.random() < 0.5 else -spike
            new_error = total / n_steps
            heading_change += new_error - direction_error
            direction_error = new_error

        readings.append(MotionReading(
            step_count=detected,
            step_length_m=step_length,
            heading_change_rad=heading_change,
            magnetic_heading_rad=magnetic_heading(t, true_heading),
            interval_s=dt,
        ))
    return readings
