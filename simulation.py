import random

from environment import ShipEnvironment

environment = ShipEnvironment.from_config("ship_layout.yaml")

position = (2.0, 2.0)
print(environment.compartment_at(position))
# Engine Room

crossed = environment.walls_crossed(
    p1=(2.0, 2.0),
    p2=(8.0, 2.0),
)

print([wall.name for wall in crossed])
# ['Engine-Machinery Bulkhead']

rng = random.Random(42)

random_position = environment.random_point_in(
    "Machinery Room",
    rng,
)

print(random_position)