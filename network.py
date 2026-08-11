import math
import yaml
from pathlib import Path
from random import Random
from shapely.geometry import LineString, Point, Polygon
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from environment import ShipEnvironment, Wall

Coord3D = Tuple[float, float, float]


@dataclass
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
class Network:
    '''The main layer containing the environment and network infrastructure'''
    environment: ShipEnvironment
    access_points: List[AccessPoint] = field(default_factory=list)
    receivers: List[Receiver] = field(default_factory=list)

    reference_rssi_dbm: float = -40.0
    reference_distance_m: float = 1.0
    path_loss_exponent: float = 2.5
    shadowing_std_db: float = 3.0

    def walls_between(self, ap: AccessPoint, receiver: Receiver) -> List[Wall]:
        return self.environment.walls_crossed(ap.xy, receiver.xy)

    def rssi(self, ap: AccessPoint, receiver: Receiver, rng: Random) -> Optional[float]:
        """Simulate an RSSI reading in dBm, or None if the AP fails to produce one."""
        dx = ap.position[0] - receiver.position[0]
        dy = ap.position[1] - receiver.position[1]
        dz = ap.position[2] - receiver.position[2]
        distance = max(math.sqrt(dx * dx + dy * dy + dz * dz), 0.1)

        reading = self.reference_rssi_dbm - 10*self.path_loss_exponent * math.log10(distance/self.reference_distance_m)

        for wall in self.walls_between(ap, receiver):
            if rng.random() < wall.dropout_prob:
                return None
            reading -= wall.attenuation_db

        reading += rng.gauss(0, self.shadowing_std_db)
        return round(reading, 1)


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