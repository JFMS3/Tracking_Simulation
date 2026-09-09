from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import ClassVar, List, Mapping, Optional, Tuple

import yaml

from environment import ShipEnvironment, Wall


Coord3D = Tuple[float, float, float]
SPEED_OF_LIGHT = 299_792_458.0
_BOLTZMANN_NOISE_DENSITY_DBM_HZ = -174.0
_EULER_MASCHERONI = 0.5772156649015329


@dataclass(unsafe_hash=True)
class AccessPoint:
    """A Wi-Fi access point: identity, fixed position, and broadcast frequency."""

    name: str
    bssid: str
    position: Coord3D
    frequency_mhz: int

    @property
    def xy(self) -> Tuple[float, float]:
        return self.position[0], self.position[1]


@dataclass
class Receiver:
    name: str
    position: Coord3D
    # A persistent, calibrated device offset. It is deliberately not redrawn
    # on every scan because RSSI bias is a property of the receiver hardware.
    rssi_bias_db: float = 0.0

    @property
    def xy(self) -> Tuple[float, float]:
        return self.position[0], self.position[1]


@dataclass
class Reading:
    """One simulated measurement between an access point and a receiver."""

    # The original fields remain first so existing localisers stay compatible.
    rssi_dbm: float
    link_speed_mbps: float
    tx_mbps: float
    rx_mbps: float
    rtt_ms: float

    # Diagnostics make each stochastic sample explainable and testable.
    noise_floor_dbm: Optional[float] = None
    snr_db: Optional[float] = None
    mean_rssi_dbm: Optional[float] = None
    shadowing_db: Optional[float] = None
    fast_fading_db: Optional[float] = None
    measurement_error_db: Optional[float] = None
    wall_count: int = 0
    is_nlos: bool = False


Scan = dict[AccessPoint, Optional[Reading]]


@dataclass(frozen=True)
class _SpatialShadowField:
    """Random Fourier approximation of an exponential Gaussian field."""

    frequencies: Tuple[Coord3D, ...]
    phases: Tuple[float, ...]

    def value(self, position: Coord3D) -> float:
        x, y, z = position
        total = sum(
            math.cos(wx * x + wy * y + wz * z + phase)
            for (wx, wy, wz), phase in zip(self.frequencies, self.phases)
        )
        return math.sqrt(2.0 / len(self.frequencies)) * total


@dataclass(frozen=True)
class _LinkSample:
    rssi_dbm: Optional[float]
    mean_rssi_dbm: float
    shadowing_db: float
    fast_fading_db: float
    measurement_error_db: float
    noise_floor_dbm: float
    snr_db: float
    walls: Tuple[Wall, ...]


@dataclass
class Network:
    """The ship environment, infrastructure, and stochastic radio channel."""

    environment: ShipEnvironment
    access_points: List[AccessPoint] = field(default_factory=list)
    receivers: List[Receiver] = field(default_factory=list)

    # Median path-loss model.
    reference_rssi_dbm: float = -40.0
    reference_distance_m: float = 1.0
    path_loss_exponent: float = 2.15

    # Large-scale, spatially correlated log-normal shadowing.
    shadowing_std_db: float = 2.0
    shadowing_correlation_distance_m: float = 2.0
    shadowing_components: int = 96
    shadowing_seed: int = 12345

    # Small-scale fading and receiver reporting error. Rayleigh power samples
    # are averaged before conversion to dB, as a real RSSI observation averages
    # energy over multiple symbols/subcarriers.
    fast_fading_samples: int = 8
    measurement_noise_std_db: float = 0.75
    rssi_quantization_db: float = 1.0

    # Receiver noise, shared scan interference, and beacon detection.
    channel_bandwidth_mhz: float = 20.0
    receiver_noise_figure_db: float = 7.0
    background_interference_dbm: Optional[float] = None
    noise_floor_variation_std_db: float = 0.5
    interference_burst_start_probability: float = 0.02
    interference_burst_end_probability: float = 0.35
    interference_burst_mean_rise_db: float = 8.0
    detection_snr_midpoint_db: float = 6.0
    detection_snr_transition_db: float = 1.5
    beacons_per_scan: int = 3

    # Compatibility switch for old configurations. Normal propagation should
    # use attenuation + SNR censoring, not independent per-wall disappearance.
    legacy_wall_dropout: bool = False

    # Wi-Fi FTM/RTT ranging errors, represented first in metres.
    rtt_base_ms: float = 8.0
    rtt_range_bias_m: float = 0.0
    rtt_ranging_noise_m: float = 0.5
    rtt_nlos_bias_per_wall_m: float = 1.0
    rtt_nlos_noise_per_wall_m: float = 0.5
    rtt_weak_signal_threshold_dbm: float = -75.0
    rtt_weak_signal_penalty_m: float = 2.0
    rtt_outlier_probability: float = 0.01
    rtt_nlos_outlier_probability: float = 0.08
    rtt_outlier_mean_m: float = 2.0

    _shadow_fields: dict[str, _SpatialShadowField] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _shadow_value_cache: dict[Tuple[str, Coord3D], float] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _interference_burst_active: dict[int, bool] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    # Wi-Fi 4, 20 MHz, one-stream rates expressed against approximate SNR.
    _RATE_TABLE: ClassVar[Tuple[Tuple[float, float], ...]] = (
        (44.0, 65.0),
        (39.0, 58.5),
        (34.0, 52.0),
        (30.0, 39.0),
        (26.0, 26.0),
        (22.0, 19.5),
        (18.0, 13.0),
        (12.0, 6.5),
    )

    def __post_init__(self) -> None:
        positive = {
            "reference_distance_m": self.reference_distance_m,
            "path_loss_exponent": self.path_loss_exponent,
            "shadowing_correlation_distance_m": self.shadowing_correlation_distance_m,
            "channel_bandwidth_mhz": self.channel_bandwidth_mhz,
            "detection_snr_transition_db": self.detection_snr_transition_db,
            "interference_burst_mean_rise_db": self.interference_burst_mean_rise_db,
            "rtt_outlier_mean_m": self.rtt_outlier_mean_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")

        non_negative = {
            "shadowing_std_db": self.shadowing_std_db,
            "measurement_noise_std_db": self.measurement_noise_std_db,
            "rssi_quantization_db": self.rssi_quantization_db,
            "receiver_noise_figure_db": self.receiver_noise_figure_db,
            "noise_floor_variation_std_db": self.noise_floor_variation_std_db,
            "rtt_ranging_noise_m": self.rtt_ranging_noise_m,
            "rtt_nlos_bias_per_wall_m": self.rtt_nlos_bias_per_wall_m,
            "rtt_nlos_noise_per_wall_m": self.rtt_nlos_noise_per_wall_m,
            "rtt_weak_signal_penalty_m": self.rtt_weak_signal_penalty_m,
        }
        for name, value in non_negative.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite number")

        probabilities = {
            "interference_burst_start_probability": self.interference_burst_start_probability,
            "interference_burst_end_probability": self.interference_burst_end_probability,
            "rtt_outlier_probability": self.rtt_outlier_probability,
            "rtt_nlos_outlier_probability": self.rtt_nlos_outlier_probability,
        }
        for name, value in probabilities.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        if (
            isinstance(self.shadowing_components, bool)
            or not isinstance(self.shadowing_components, int)
            or self.shadowing_components < 1
        ):
            raise ValueError("shadowing_components must be an integer of at least 1")
        if (
            isinstance(self.fast_fading_samples, bool)
            or not isinstance(self.fast_fading_samples, int)
            or self.fast_fading_samples < 0
        ):
            raise ValueError("fast_fading_samples must be a non-negative integer")
        if (
            isinstance(self.beacons_per_scan, bool)
            or not isinstance(self.beacons_per_scan, int)
            or self.beacons_per_scan < 1
        ):
            raise ValueError("beacons_per_scan must be an integer of at least 1")
        if self.background_interference_dbm is not None and not math.isfinite(
            self.background_interference_dbm
        ):
            raise ValueError("background_interference_dbm must be finite or null")

    def walls_between(self, ap: AccessPoint, receiver: Receiver) -> List[Wall]:
        return self.environment.walls_crossed(ap.xy, receiver.xy)

    def _distance(self, ap: AccessPoint, receiver: Receiver) -> float:
        dx = ap.position[0] - receiver.position[0]
        dy = ap.position[1] - receiver.position[1]
        dz = ap.position[2] - receiver.position[2]
        return max(math.sqrt(dx * dx + dy * dy + dz * dz), 0.1)

    def mean_rssi(
        self,
        ap: AccessPoint,
        receiver: Receiver,
        walls: Optional[List[Wall]] = None,
    ) -> float:
        """Return median RSSI before stochastic fading and receiver error."""

        distance = self._distance(ap, receiver)
        crossed_walls = self.walls_between(ap, receiver) if walls is None else walls
        path_loss_db = 10.0 * self.path_loss_exponent * math.log10(
            distance / self.reference_distance_m
        )
        wall_loss_db = sum(wall.attenuation_db for wall in crossed_walls)
        return self.reference_rssi_dbm - path_loss_db - wall_loss_db + receiver.rssi_bias_db

    def _shadow_field(self, ap: AccessPoint) -> _SpatialShadowField:
        key = f"{ap.bssid}|{ap.frequency_mhz}"
        existing = self._shadow_fields.get(key)
        if existing is not None:
            return existing

        digest = hashlib.blake2b(
            f"{self.shadowing_seed}|{key}".encode("utf-8"),
            digest_size=16,
        ).digest()
        field_rng = Random(int.from_bytes(digest, "big"))
        frequencies = []
        phases = []

        # A multivariate Student-t(df=1) spectral distribution corresponds to
        # the exponential (Matern-1/2) covariance used for shadowing models.
        for _ in range(self.shadowing_components):
            denominator = abs(field_rng.gauss(0.0, 1.0))
            while denominator < 1e-8:
                denominator = abs(field_rng.gauss(0.0, 1.0))
            scale = 1.0 / (self.shadowing_correlation_distance_m * denominator)
            frequencies.append(
                (
                    field_rng.gauss(0.0, 1.0) * scale,
                    field_rng.gauss(0.0, 1.0) * scale,
                    field_rng.gauss(0.0, 1.0) * scale,
                )
            )
            phases.append(field_rng.uniform(0.0, 2.0 * math.pi))

        result = _SpatialShadowField(tuple(frequencies), tuple(phases))
        self._shadow_fields[key] = result
        return result

    def spatial_shadowing(self, ap: AccessPoint, receiver: Receiver) -> float:
        """Sample the fixed, spatially correlated shadow field at a position."""

        if self.shadowing_std_db == 0.0:
            return 0.0

        ap_key = f"{ap.bssid}|{ap.frequency_mhz}"
        cache_key = (ap_key, receiver.position)
        existing = self._shadow_value_cache.get(cache_key)
        if existing is not None:
            return self.shadowing_std_db * existing

        field_value = self._shadow_field(ap).value(receiver.position)
        self._shadow_value_cache[cache_key] = field_value
        return self.shadowing_std_db * field_value

    def _fast_fading_db(self, rng: Random) -> float:
        """Return zero-mean-in-dB averaged Rayleigh power fading."""

        count = self.fast_fading_samples
        if count == 0:
            return 0.0

        mean_power_gain = sum(rng.expovariate(1.0) for _ in range(count)) / count
        harmonic = sum(1.0 / k for k in range(1, count))
        expected_log_gain = harmonic - _EULER_MASCHERONI - math.log(count)
        return (10.0 / math.log(10.0)) * (
            math.log(mean_power_gain) - expected_log_gain
        )

    @staticmethod
    def _sum_dbm(*powers_dbm: Optional[float]) -> float:
        linear_mw = sum(
            10.0 ** (power_dbm / 10.0)
            for power_dbm in powers_dbm
            if power_dbm is not None
        )
        return 10.0 * math.log10(linear_mw)

    @property
    def nominal_noise_floor_dbm(self) -> float:
        thermal_dbm = (
            _BOLTZMANN_NOISE_DENSITY_DBM_HZ
            + 10.0 * math.log10(self.channel_bandwidth_mhz * 1_000_000.0)
            + self.receiver_noise_figure_db
        )
        return self._sum_dbm(thermal_dbm, self.background_interference_dbm)

    def _noise_floor_for_scan(self, rng: Random, frequency_mhz: int) -> float:
        burst_active = self._interference_burst_active.get(frequency_mhz, False)
        if burst_active:
            if rng.random() < self.interference_burst_end_probability:
                burst_active = False
        elif rng.random() < self.interference_burst_start_probability:
            burst_active = True
        self._interference_burst_active[frequency_mhz] = burst_active

        noise_floor = self.nominal_noise_floor_dbm
        noise_floor += rng.gauss(0.0, self.noise_floor_variation_std_db)
        if burst_active:
            noise_floor += rng.expovariate(1.0 / self.interference_burst_mean_rise_db)
        return noise_floor

    def detection_probability(self, rssi_dbm: float, noise_floor_dbm: float) -> float:
        """Probability that at least one beacon is decoded in a scan window."""

        snr_db = rssi_dbm - noise_floor_dbm
        x = (snr_db - self.detection_snr_midpoint_db) / self.detection_snr_transition_db
        if x >= 0.0:
            exp_neg_x = math.exp(-min(x, 700.0))
            per_beacon = 1.0 / (1.0 + exp_neg_x)
        else:
            exp_x = math.exp(max(x, -700.0))
            per_beacon = exp_x / (1.0 + exp_x)
        return 1.0 - (1.0 - per_beacon) ** self.beacons_per_scan

    def _sample_link(
        self,
        ap: AccessPoint,
        receiver: Receiver,
        rng: Random,
        noise_floor_dbm: float,
    ) -> _LinkSample:
        walls = tuple(self.walls_between(ap, receiver))
        mean_rssi_dbm = self.mean_rssi(ap, receiver, list(walls))

        if self.legacy_wall_dropout:
            for wall in walls:
                if rng.random() < wall.dropout_prob:
                    return _LinkSample(
                        None,
                        mean_rssi_dbm,
                        0.0,
                        0.0,
                        0.0,
                        noise_floor_dbm,
                        -math.inf,
                        walls,
                    )

        shadowing_db = self.spatial_shadowing(ap, receiver)
        fast_fading_db = self._fast_fading_db(rng)
        measurement_error_db = rng.gauss(0.0, self.measurement_noise_std_db)
        observed_rssi_dbm = (
            mean_rssi_dbm + shadowing_db + fast_fading_db + measurement_error_db
        )
        snr_db = observed_rssi_dbm - noise_floor_dbm

        if rng.random() > self.detection_probability(observed_rssi_dbm, noise_floor_dbm):
            reported_rssi_dbm = None
        elif self.rssi_quantization_db > 0.0:
            step = self.rssi_quantization_db
            reported_rssi_dbm = round(observed_rssi_dbm / step) * step
        else:
            reported_rssi_dbm = observed_rssi_dbm

        return _LinkSample(
            reported_rssi_dbm,
            mean_rssi_dbm,
            shadowing_db,
            fast_fading_db,
            measurement_error_db,
            noise_floor_dbm,
            snr_db,
            walls,
        )

    def rssi(self, ap: AccessPoint, receiver: Receiver, rng: Random) -> Optional[float]:
        """Simulate a reported RSSI, or ``None`` if no beacon is decoded."""

        sample = self._sample_link(
            ap,
            receiver,
            rng,
            self._noise_floor_for_scan(rng, ap.frequency_mhz),
        )
        return sample.rssi_dbm

    def rtt(
        self,
        distance: float,
        rssi: float,
        rng: Random,
        *,
        wall_count: int = 0,
    ) -> float:
        """Simulate an FTM range error and convert the apparent range to RTT."""

        weak_margin_db = max(self.rtt_weak_signal_threshold_dbm - rssi, 0.0)
        weak_signal_bias_m = self.rtt_weak_signal_penalty_m * (
            1.0 - math.exp(-weak_margin_db / 6.0)
        )
        noise_std_m = self.rtt_ranging_noise_m + (
            wall_count * self.rtt_nlos_noise_per_wall_m
        )
        range_error_m = rng.gauss(0.0, noise_std_m)
        range_error_m += self.rtt_range_bias_m
        range_error_m += wall_count * self.rtt_nlos_bias_per_wall_m
        range_error_m += weak_signal_bias_m

        outlier_probability = (
            self.rtt_nlos_outlier_probability if wall_count else self.rtt_outlier_probability
        )
        if rng.random() < outlier_probability:
            range_error_m += rng.expovariate(1.0 / self.rtt_outlier_mean_m)

        apparent_distance = max(distance + range_error_m, 0.05)
        time_of_flight_ms = (2.0 * apparent_distance / SPEED_OF_LIGHT) * 1000.0
        return round(self.rtt_base_ms + time_of_flight_ms, 9)

    def scan(self, receiver: Receiver, rng: Random) -> Scan:
        """Scan all APs under one shared noise/interference realization."""

        noise_floors: dict[int, float] = {}
        for ap in self.access_points:
            if ap.frequency_mhz not in noise_floors:
                noise_floors[ap.frequency_mhz] = self._noise_floor_for_scan(
                    rng,
                    ap.frequency_mhz,
                )
        return {
            ap: self.reading(
                ap,
                receiver,
                rng,
                noise_floor_dbm=noise_floors[ap.frequency_mhz],
            )
            for ap in self.access_points
        }

    def link_speed(self, rssi_dbm: float, noise_floor_dbm: Optional[float] = None) -> float:
        """Map link SNR to an approximate one-stream Wi-Fi 4 PHY rate."""

        floor = self.nominal_noise_floor_dbm if noise_floor_dbm is None else noise_floor_dbm
        snr_db = rssi_dbm - floor
        for threshold, rate in self._RATE_TABLE:
            if snr_db >= threshold:
                return rate
        return 0.0

    def reading(
        self,
        ap: AccessPoint,
        receiver: Receiver,
        rng: Random,
        *,
        noise_floor_dbm: Optional[float] = None,
    ) -> Optional[Reading]:
        """Simulate one internally consistent AP measurement."""

        floor = (
            self._noise_floor_for_scan(rng, ap.frequency_mhz)
            if noise_floor_dbm is None
            else noise_floor_dbm
        )
        sample = self._sample_link(ap, receiver, rng, floor)
        if sample.rssi_dbm is None:
            return None

        link_speed_mbps = self.link_speed(sample.rssi_dbm, sample.noise_floor_dbm)
        distance = self._distance(ap, receiver)
        wall_count = len(sample.walls)

        return Reading(
            rssi_dbm=sample.rssi_dbm,
            link_speed_mbps=link_speed_mbps,
            tx_mbps=link_speed_mbps,
            rx_mbps=link_speed_mbps,
            rtt_ms=self.rtt(distance, sample.rssi_dbm, rng, wall_count=wall_count),
            noise_floor_dbm=round(sample.noise_floor_dbm, 2),
            snr_db=round(sample.snr_db, 2),
            mean_rssi_dbm=round(sample.mean_rssi_dbm, 2),
            shadowing_db=round(sample.shadowing_db, 2),
            fast_fading_db=round(sample.fast_fading_db, 2),
            measurement_error_db=round(sample.measurement_error_db, 2),
            wall_count=wall_count,
            is_nlos=wall_count > 0,
        )

    def reset_channel_state(self) -> None:
        """Reset temporal interference state; the vessel's spatial field stays fixed."""

        self._interference_burst_active.clear()

    @classmethod
    def from_config(
        cls,
        filename: str | Path,
        access_points: Optional[List[AccessPoint]] = None,
        receivers: Optional[List[Receiver]] = None,
    ) -> Network:
        config = yaml.safe_load(Path(filename).read_text(encoding="utf-8")) or {}
        if not isinstance(config, Mapping):
            raise ValueError("The top level of the network configuration must be a mapping")

        environment = ShipEnvironment.from_dict(config)
        propagation = config.get("propagation", {})
        if not isinstance(propagation, Mapping):
            raise ValueError("'propagation' must be a mapping")

        return cls(
            environment=environment,
            access_points=access_points or [],
            receivers=receivers or [],
            **dict(propagation),
        )
