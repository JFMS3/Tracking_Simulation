import math
import yaml
from pathlib import Path
from random import Random
from shapely.geometry import LineString, Point, Polygon
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from environment import ShipEnvironment, Wall

Coord3D = Tuple[float, float, float]


@dataclass(unsafe_hash=True)
class AccessPoint:
    """A WiFi access point: identity, fixed position, and broadcast frequency."""
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

    @property
    def xy(self) -> Tuple[float, float]:
        return self.position[0], self.position[1]


@dataclass
class Reading:
    """One simulated measurement between an AccessPoint and a Receiver."""
    rssi_dbm: float
    link_speed_mbps: float
    tx_mbps: float
    rx_mbps: float
    rtt_ms: float


Scan = dict["AccessPoint", Optional[Reading]]


@dataclass
class Network:
    '''The main layer containing the environment and network infrastructure'''
    environment: ShipEnvironment
    access_points: List[AccessPoint] = field(default_factory=list)
    receivers: List[Receiver] = field(default_factory=list)

    reference_rssi_dbm: float = -40.0
    reference_distance_m: float = 1.0
    path_loss_exponent: float = 2.5
    shadowing_std_db: float = 3.0

    rtt_base_ms: float = 8.0
    rtt_jitter_std_ms: float = 2.0
    rtt_weak_signal_threshold_dbm: float = -75.0
    rtt_weak_signal_penalty_ms: float = 15.0

    # WiFi picks from discrete menu of coding schemes based on how clean the signal is
    # These ones are the WiFi4 PHY rates for a 20MHz channel
    _RATE_TABLE = [
        (-50, 65.0),
        (-55, 58.5),
        (-60, 52.0),
        (-64, 39.0),
        (-68, 26.0),
        (-72, 19.5),
        (-76, 13.0),
        (-82, 6.5),
    ]

    def walls_between(self, ap: AccessPoint, receiver: Receiver) -> List[Wall]:
        return self.environment.walls_crossed(ap.xy, receiver.xy)

    def rssi(self, ap: AccessPoint, receiver: Receiver, rng: Random) -> Optional[float]:
        """Simulate an RSSI reading in dBm, or None if the AP fails to produce one."""
        dx = ap.position[0] - receiver.position[0]
        dy = ap.position[1] - receiver.position[1]
        dz = ap.position[2] - receiver.position[2]
        distance = max(math.sqrt(dx * dx + dy * dy + dz * dz), 0.1)

        reading = self.reference_rssi_dbm - 10 * self.path_loss_exponent * math.log10(
            distance / self.reference_distance_m
        )

        for wall in self.walls_between(ap, receiver):
            if rng.random() < wall.dropout_prob:
                return None
            reading -= wall.attenuation_db

        reading += rng.gauss(0, self.shadowing_std_db)
        return round(reading, 1)


    def scan(self, receiver: Receiver, rng: Random):
        return {ap: self.reading(ap, receiver, rng) for ap in self.access_points}


    def link_speed(self, rssi_dbm: float) -> float:
        """Look up RSSI from rate table"""
        for threshold, rate in self._RATE_TABLE:
            if rssi_dbm >= threshold:
                return rate
        return 0.0


    def rtt(self, rssi_dbm: float, rng: Random) -> float:
        """Round-trip latency in ms: base + jitter, worse on a weak link."""
        latency = rng.gauss(self.rtt_base_ms, self.rtt_jitter_std_ms)
        if rssi_dbm < self.rtt_weak_signal_threshold_dbm:
            latency += self.rtt_weak_signal_penalty_ms
        return round(max(latency, 1.0), 2)


    def reading(self, ap: AccessPoint, receiver: Receiver, rng: Random) -> Optional[Reading]:
        """One simulated measurement, or None if the AP produced no reading at all."""
        rssi_dbm = self.rssi(ap, receiver, rng)

        if rssi_dbm is None:
            return None

        link_speed_mbps = self.link_speed(rssi_dbm)

        return Reading(
            rssi_dbm=rssi_dbm,
            link_speed_mbps=link_speed_mbps,
            tx_mbps=link_speed_mbps,
            rx_mbps=link_speed_mbps,
            rtt_ms=self.rtt(rssi_dbm, rng),
        )


    @classmethod
    def from_config(
        cls,
        filename: str | Path,
        access_points: Optional[List[AccessPoint]] = None,
        receivers: Optional[List[Receiver]] = None,
    ) -> Network:
        config = yaml.safe_load(Path(filename).read_text()) or {}
        environment = ShipEnvironment.from_dict(config)
        propagation = config.get("propagation", {})

        return cls(
            environment=environment,
            access_points=access_points or [],
            receivers=receivers or [],
            **propagation,
        )