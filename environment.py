from shapely.geometry import LineString, Point, Polygon
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Coord = Tuple[float, float]

@dataclass
class Wall:
    name: str
    p1: Coord
    p2: Coord
    material: str = "steel_bulkhead"
    dropout_prob: float = 0.05 # Probability AP fails to produce a reading when this wall is crossed (more for steel than wood)

    @property
    def geometry(self) -> LineString:
        return LineString([self.p1, self.p2])


@dataclass
class Compartment:
    name: str
    boundary: List[Coord]
    deck: str = "main"

    @property
    def geometry(self) -> Polygon:
        return Polygon(self.boundary)



@dataclass
class ShipEnvironment:
    compartments: List[Compartment] = field(default_factory=list)
    walls: List[Wall] = field(default_factory=list)

    def compartment_at(self, point: Coord) -> Optional[str]:
        '''Returns which compartment (if any) contains the given point'''
        pt = Point(point)
        for c in self.compartments:
            if c.geometry.covers(pt):
                return c.name
        return None

    def walls_crossed(self, p1: Coord, p2: Coord) -> List[Wall]:
        '''Returns all walls where the line p1-p2 crosses over'''
        ray = LineString([p1, p2])
        return [w for w in self.walls if ray.crosses(w.geometry)]

    def random_point_in(self, compartment_name: str, rng) -> Coord:
        """Rejection-sample a uniformly random point inside a named compartment."""
        comp = next(c for c in self.compartments if c.name == compartment_name)
        minx, miny, maxx, maxy = comp.geometry.bounds
        for _ in range(1000):
            p = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if comp.geometry.covers(Point(p)):
                return p
        raise RuntimeError(f"Could not sample a point inside {compartment_name}")
