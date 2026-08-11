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

    #def RSSI_reading(self, receiver)


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

    def walls_between(self, ap: AccessPoint, receiver: Receiver) -> List[Wall]:
        return self.environment.walls_crossed(ap.xy, receiver.xy)

    def rssi(self, ap: AccessPoint, receiver: Receiver) -> Optional[float]:
        """Returns None if the AP fails to produce a reading at all."""