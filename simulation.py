import matplotlib.pyplot as plt
from network import Network, AccessPoint, Receiver
from environment import ShipEnvironment
from positioning import NearestAPLocaliser, WeightedCentroidLocaliser
from random import Random
import numpy as np


def display_network(network: Network) -> None:
    environment = network.environment
    fig, ax = plt.subplots()

    # Draw compartments
    for compartment in environment.compartments:
        polygon = compartment.geometry
        x, y = polygon.exterior.xy

        ax.fill(x, y, alpha=0.3, edgecolor="black")

        centre = polygon.centroid
        ax.text(
            centre.x, centre.y, compartment.name, 
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Draw walls
    for wall in environment.walls:
        line = wall.geometry
        x, y = line.xy

        ax.plot(x, y, color="black", linewidth=3)

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw access points
    for ap in network.access_points:
        ax.scatter(*ap.xy, marker="*", s=250, color="crimson", zorder=5)
        ax.annotate(
            ap.name, ap.xy,
            xytext=(0, 8), textcoords="offset points",
            horizontalalignment="center", fontsize=8,
        )

    # Draw receivers
    for receiver in network.receivers:
        ax.scatter(*receiver.xy, marker="o", color="royalblue", zorder=5)
        ax.annotate(
            receiver.name, receiver.xy,
            xytext=(0, 8), textcoords="offset points",
            horizontalalignment="center", fontsize=8,
        )

    plt.show()


rng = Random(12345)
environment = ShipEnvironment.from_config("layouts/simple_layout2.yaml")
ap1 = AccessPoint("Router1", "24:2f:d0:fb:85:b9", (4.5, 1.5, 1.5), 2412)
ap2 = AccessPoint("Router2", "25:3f:d0:fb:85:c4", (1.5, 4.5, 1.0), 2412)
ap3 = AccessPoint("Router3", "26:4f:d0:fb:85:c0", (8.0, 3.0, 0.5), 2412)
re1 = Receiver("Phone-Desk", (3, 5.9, 0.7))
re2 = Receiver("Phone-Engine", (2, 1, 1))

network = Network.from_config("layouts/simple_layout2.yaml", access_points=[ap1, ap2, ap3], receivers=[re1, re2])
nearest_ap_localiser = NearestAPLocaliser()
weighted_distance_localiser = WeightedCentroidLocaliser(network, weight_mode="distance")
weighted_power_localiser = WeightedCentroidLocaliser(network, weight_mode="linear_power")


def get_coordinate_guess(re):
    scan = network.scan(re, rng)
    nearest_ap_location = nearest_ap_localiser.locate(scan).position
    weighted_distance_location = weighted_distance_localiser.locate(scan).position
    weighted_power_location = weighted_power_localiser.locate(scan).position

    coords = [nearest_ap_location, weighted_distance_location, weighted_power_location]
    weighted_coordinate_guess = tuple(np.average(coords, axis=0, weights=[0.1, 0.4, 0.5]))
    return weighted_coordinate_guess


for i in range(10):
    re1_coordinate_guess = get_coordinate_guess(re1)
    re1_compartment_guess = environment.compartment_at(re1_coordinate_guess)
    re2_coordinate_guess = get_coordinate_guess(re2)
    re2_compartment_guess = environment.compartment_at(re2_coordinate_guess)
    print(f"Timestamp {i}: {re1.name} in {re1_compartment_guess} | {re2.name} in {re2_compartment_guess}")


display_network(network)