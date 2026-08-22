import math
from random import Random

import matplotlib.pyplot as plt
import numpy as np

from network import Network, AccessPoint, Receiver
from environment import ShipEnvironment
from positioning import NearestAPLocaliser, WeightedCentroidLocaliser, TrilaterationLocaliser


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


def report_ap_coverage(network: Network, rng: Random, n_trials: int = 300) -> None:
    environment = network.environment
    print("=== AP coverage (avg valid readings per scan, out of "
          f"{len(network.access_points)}) ===")
    for comp in environment.compartments:
        valid_counts = []
        for _ in range(n_trials):
            pos = environment.random_point_in(comp.name, rng)
            receiver = Receiver("probe", (*pos, 1.0))
            scan = network.scan(receiver, rng)
            valid_counts.append(sum(1 for r in scan.values() if r is not None))
        avg = sum(valid_counts) / len(valid_counts)
        print(f"{comp.name:<15} avg valid APs: {avg:.2f}   (min seen: {min(valid_counts)})")
    print()


def report_localiser_stats(
    localisers: dict,
    network: Network,
    rng: Random,
    n_trials_per_compartment: int = 300,
) -> None:
    environment = network.environment
    print("=== Localiser performance across ALL compartments ===")
    header = f"{'Method':<24}{'Mean err':<10}{'Median err':<12}{'P90 err':<10}{'Miss rate':<11}{'Compartment acc':<16}"
    print(header)

    for name, loc in localisers.items():
        errors = []
        misses = 0
        compartment_correct = 0
        total = 0

        for comp in environment.compartments:
            for _ in range(n_trials_per_compartment):
                true_pos = environment.random_point_in(comp.name, rng)
                receiver = Receiver("probe", (*true_pos, 1.0))
                scan = network.scan(receiver, rng)
                estimate = loc.locate(scan)
                total += 1

                if estimate is None:
                    misses += 1
                    continue

                errors.append(math.dist(true_pos, estimate.position))
                if environment.compartment_at(estimate.position) == comp.name:
                    compartment_correct += 1

        mean_err = sum(errors) / len(errors) if errors else float("nan")
        median_err = sorted(errors)[len(errors) // 2] if errors else float("nan")
        p90_err = sorted(errors)[int(len(errors) * 0.9)] if errors else float("nan")
        miss_rate = misses / total
        compartment_acc = compartment_correct / total  # misses count as wrong

        print(f"{name:<24}{mean_err:<10.2f}{median_err:<12.2f}{p90_err:<10.2f}"
              f"{miss_rate:<11.1%}{compartment_acc:<16.1%}")
    print()


def report_compartment_breakdown(
    localisers: dict,
    network: Network,
    rng: Random,
    n_trials_per_compartment: int = 300,
) -> None:
    environment = network.environment
    print("=== Compartment classification accuracy, split by TRUE compartment ===")
    print(f"{'Compartment':<16}" + "".join(f"{name:<24}" for name in localisers))
    for comp in environment.compartments:
        row = f"{comp.name:<16}"
        for name, loc in localisers.items():
            correct = 0
            for _ in range(n_trials_per_compartment):
                true_pos = environment.random_point_in(comp.name, rng)
                receiver = Receiver("probe", (*true_pos, 1.0))
                scan = network.scan(receiver, rng)
                estimate = loc.locate(scan)
                if estimate is not None and environment.compartment_at(estimate.position) == comp.name:
                    correct += 1
            row += f"{correct / n_trials_per_compartment:<24.1%}"
        print(row)
    print()


rng = Random(12345)
environment = ShipEnvironment.from_config("layouts/simple_layout2.yaml")
ap1 = AccessPoint("Router1", "24:2f:d0:fb:85:b9", (4.5, 1.5, 1.5), 2412)
ap2 = AccessPoint("Router2", "25:3f:d0:fb:85:c4", (1.5, 4.5, 1.0), 2412)
ap3 = AccessPoint("Router3", "26:4f:d0:fb:85:c0", (8.0, 3.0, 0.5), 2412)
ap4 = AccessPoint("Router4", "27:A0:d0:fb:85:c4", (5.0, 6.2, 1.0), 2412)
ap5 = AccessPoint("Router5", "28:25:d0:fb:85:c0", (6.8, 4.5, 0.5), 2412)
re1 = Receiver("Phone-Desk", (3, 5.9, 0.7))
re2 = Receiver("Phone-Engine", (2, 1, 1))

network = Network.from_config(
    "layouts/simple_layout2.yaml", access_points=[ap1, ap2, ap3, ap4, ap5], receivers=[re1, re2]
)

nearest_ap_localiser = NearestAPLocaliser()
weighted_distance_localiser = WeightedCentroidLocaliser(network, weight_mode="distance")
weighted_power_localiser = WeightedCentroidLocaliser(network, weight_mode="linear_power")
trilateration_localiser = TrilaterationLocaliser(network)

localisers = {
    "NearestAP": nearest_ap_localiser,
    "WeightedCentroid(dist)": weighted_distance_localiser,
    "WeightedCentroid(power)": weighted_power_localiser,
    "Trilateration": trilateration_localiser,
}


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



report_ap_coverage(network, rng)
report_localiser_stats(localisers, network, rng)
report_compartment_breakdown(localisers, network, rng)

display_network(network)