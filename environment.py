from shapely.geometry import LineString, Point, Polygon
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import math
from pathlib import Path
from typing import Any, Mapping

import yaml
from shapely.validation import explain_validity

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

    @classmethod
    def from_config(cls, filename: str | Path) -> "ShipEnvironment":
        """Create an environment from a YAML configuration file."""

        config_path = Path(filename)

        try:
            with config_path.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
        except OSError as exc:
            raise EnvironmentConfigError(
                f"Could not read configuration file: {config_path}"
            ) from exc
        except yaml.YAMLError as exc:
            raise EnvironmentConfigError(
                f"Invalid YAML in configuration file: {config_path}"
            ) from exc

        if config is None:
            config = {}

        if not isinstance(config, Mapping):
            raise EnvironmentConfigError(
                "The top level of the configuration must be a mapping"
            )

        return cls.from_dict(config)

    @classmethod
    def from_dict(
        cls,
        config: Mapping[str, Any],
    ) -> "ShipEnvironment":
        """Create an environment from already-loaded configuration data."""

        compartment_data = config.get("compartments", [])
        wall_data = config.get("walls", [])

        if not isinstance(compartment_data, list):
            raise EnvironmentConfigError(
                "'compartments' must be a list"
            )

        if not isinstance(wall_data, list):
            raise EnvironmentConfigError(
                "'walls' must be a list"
            )

        compartments: List[Compartment] = []
        walls: List[Wall] = []

        compartment_names: set[str] = set()

        for index, item in enumerate(compartment_data):
            location = f"compartments[{index}]"

            if not isinstance(item, Mapping):
                raise EnvironmentConfigError(
                    f"{location} must be a mapping"
                )

            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                raise EnvironmentConfigError(
                    f"{location}.name must be a non-empty string"
                )

            name = name.strip()

            if name in compartment_names:
                raise EnvironmentConfigError(
                    f"Duplicate compartment name: {name}"
                )

            raw_boundary = item.get("boundary")

            if not isinstance(raw_boundary, list) or len(raw_boundary) < 3:
                raise EnvironmentConfigError(
                    f"{location}.boundary must contain at least three points"
                )

            boundary = [
                parse_coord(point, f"{location}.boundary[{point_index}]")
                for point_index, point in enumerate(raw_boundary)
            ]

            deck = item.get("deck", "main")

            if not isinstance(deck, str) or not deck.strip():
                raise EnvironmentConfigError(
                    f"{location}.deck must be a non-empty string"
                )

            compartment = Compartment(
                name=name,
                boundary=boundary,
                deck=deck.strip(),
            )

            polygon = compartment.geometry

            if polygon.is_empty or polygon.area == 0:
                raise EnvironmentConfigError(
                    f"Compartment '{name}' has a zero-area boundary"
                )

            if not polygon.is_valid:
                raise EnvironmentConfigError(
                    f"Compartment '{name}' has an invalid boundary: "
                    f"{explain_validity(polygon)}"
                )

            compartments.append(compartment)
            compartment_names.add(name)

        wall_names: set[str] = set()

        for index, item in enumerate(wall_data):
            location = f"walls[{index}]"

            if not isinstance(item, Mapping):
                raise EnvironmentConfigError(
                    f"{location} must be a mapping"
                )

            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                raise EnvironmentConfigError(
                    f"{location}.name must be a non-empty string"
                )

            name = name.strip()

            if name in wall_names:
                raise EnvironmentConfigError(
                    f"Duplicate wall name: {name}"
                )

            p1 = parse_coord(item.get("p1"), f"{location}.p1")
            p2 = parse_coord(item.get("p2"), f"{location}.p2")

            if p1 == p2:
                raise EnvironmentConfigError(
                    f"Wall '{name}' cannot have identical endpoints"
                )

            material = item.get("material", "steel_bulkhead")

            if not isinstance(material, str) or not material.strip():
                raise EnvironmentConfigError(
                    f"{location}.material must be a non-empty string"
                )

            raw_dropout = item.get("dropout_prob", 0.05)

            try:
                dropout_prob = float(raw_dropout)
            except (TypeError, ValueError) as exc:
                raise EnvironmentConfigError(
                    f"{location}.dropout_prob must be a number"
                ) from exc

            if not 0 <= dropout_prob <= 1:
                raise EnvironmentConfigError(
                    f"{location}.dropout_prob must be between 0 and 1"
                )

            walls.append(
                Wall(
                    name=name,
                    p1=p1,
                    p2=p2,
                    material=material.strip(),
                    dropout_prob=dropout_prob,
                )
            )

            wall_names.add(name)

        return cls(
            compartments=compartments,
            walls=walls,
        )

class EnvironmentConfigError(ValueError):
    """Raised when a ship environment configuration is invalid."""


def parse_coord(value: Any, location: str) -> Coord:
    """Validate and convert a config value into an (x, y) coordinate."""

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise EnvironmentConfigError(
            f"{location} must contain exactly two coordinates"
        )

    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError) as exc:
        raise EnvironmentConfigError(
            f"{location} coordinates must be numbers"
        ) from exc

    if not math.isfinite(x) or not math.isfinite(y):
        raise EnvironmentConfigError(
            f"{location} coordinates must be finite numbers"
        )

    return x, y