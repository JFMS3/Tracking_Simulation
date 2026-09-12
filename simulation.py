"""Simulation composition, benchmarks, and plotting; algorithms live in crew_tracking."""

import argparse
import math
from pathlib import Path
from random import Random

import numpy as np

from crew_tracking.models import AccessPoint, PathLossModel
from crew_tracking.ftm import FTMLocaliser
from crew_tracking.positioning import (
    NearestAPLocaliser, WeightedCentroidLocaliser,
    RSSTrilaterationLocaliser, RTTTrilaterationLocaliser,
)
from crew_tracking.fingerprinting import FingerprintEntry, FingerprintLocaliser
from simulator.calibration import build_radio_map
from simulator.network import Network, Receiver
from typing import List, Optional, Sequence

Metric = str

def display_network(network: Network) -> None:
    import matplotlib.pyplot as plt

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

    # Every method must see the same locations and the same stochastic scans;
    # otherwise differences in fading/dropout are confounded with the method.
    trials = []
    for comp in environment.compartments:
        for _ in range(n_trials_per_compartment):
            true_pos = environment.random_point_in(comp.name, rng)
            receiver = Receiver("probe", (*true_pos, 1.0))
            trials.append((comp.name, true_pos, network.scan(receiver, rng)))

    for name, loc in localisers.items():
        errors = []
        misses = 0
        compartment_correct = 0
        total = len(trials)

        for compartment_name, true_pos, scan in trials:
            estimate = loc.locate(scan)

            if estimate is None:
                misses += 1
                continue

            errors.append(math.dist(true_pos, estimate.position))
            if environment.compartment_at(estimate.position) == compartment_name:
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
        trials = []
        for _ in range(n_trials_per_compartment):
            true_pos = environment.random_point_in(comp.name, rng)
            receiver = Receiver("probe", (*true_pos, 1.0))
            trials.append(network.scan(receiver, rng))

        row = f"{comp.name:<16}"
        for name, loc in localisers.items():
            correct = 0
            for scan in trials:
                estimate = loc.locate(scan)
                if estimate is not None and environment.compartment_at(estimate.position) == comp.name:
                    correct += 1
            row += f"{correct / n_trials_per_compartment:<24.1%}"
        print(row)
    print()




 
def _heatmap_values(
    radio_map: List[FingerprintEntry],
    ap_order: Sequence[AccessPoint],
    metric: Metric,
    ap_name: Optional[str],
    coverage_threshold_dbm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    xs = np.array([e.position[0] for e in radio_map])
    ys = np.array([e.position[1] for e in radio_map])
 
    if metric == "best_signal":
        vals = np.array([e.rssi_vector.max() for e in radio_map])
        label = "Strongest RSSI heard (dBm)"
 
    elif metric == "coverage":
        vals = np.array([(e.rssi_vector > coverage_threshold_dbm).sum() for e in radio_map])
        label = f"# APs heard above {coverage_threshold_dbm:.0f} dBm"
 
    elif metric == "ap_signal":
        if ap_name is None:
            raise ValueError("ap_name is required when metric='ap_signal'")
        names = [ap.name for ap in ap_order]
        if ap_name not in names:
            raise ValueError(f"Unknown AP name: {ap_name!r}. Known: {names}")
        idx = names.index(ap_name)
        vals = np.array([e.rssi_vector[idx] for e in radio_map])
        label = f"{ap_name} RSSI (dBm)"
 
    else:
        raise ValueError(f"Unknown metric: {metric!r}")
 
    return xs, ys, vals, label
 
 
def _draw_layout(ax, network: Network, fill_compartments: bool = True) -> None:
    """Draw compartments, walls, APs and receivers onto an existing axis."""
    import matplotlib.patheffects as pe

    environment = network.environment
 
    for compartment in environment.compartments:
        polygon = compartment.geometry
        x, y = polygon.exterior.xy
 
        if fill_compartments:
            ax.fill(x, y, alpha=0.25, edgecolor="black")
        else:
            ax.plot(x, y, color="black", linewidth=1, linestyle="--")
 
        centre = polygon.centroid
        ax.text(
            centre.x, centre.y, compartment.name,
            horizontalalignment="center", verticalalignment="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, edgecolor="none"),
        )
 
    for wall in environment.walls:
        x, y = wall.geometry.xy
        ax.plot(x, y, color="black", linewidth=3)
 
    for ap in network.access_points:
        ax.scatter(*ap.xy, marker="*", s=250, color="crimson", zorder=5, edgecolor="white", linewidth=0.5)
        ax.annotate(
            ap.name, ap.xy,
            xytext=(0, 8), textcoords="offset points",
            horizontalalignment="center", fontsize=8,
            color="white", weight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
        )
 
    for receiver in network.receivers:
        ax.scatter(*receiver.xy, marker="o", color="royalblue", zorder=5, edgecolor="white", linewidth=0.5)
        ax.annotate(
            receiver.name, receiver.xy,
            xytext=(0, 8), textcoords="offset points",
            horizontalalignment="center", fontsize=8,
            color="white", weight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
        )
 
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_aspect("equal")
 
 
def plot_fingerprint_heatmap(
    network: Network,
    radio_map: List[FingerprintEntry],
    ap_order: Sequence[AccessPoint],
    metric: Metric = "best_signal",
    ap_name: Optional[str] = None,
    layout: str = "overlay",
    coverage_threshold_dbm: float = -75.0,
    cmap: str = "viridis",
    show: bool = True,
):
    import matplotlib.pyplot as plt

    xs, ys, vals, label = _heatmap_values(radio_map, ap_order, metric, ap_name, coverage_threshold_dbm)
 
    if layout == "overlay":
        fig, ax = plt.subplots(figsize=(8, 7))
        tpc = ax.tricontourf(xs, ys, vals, levels=20, cmap=cmap)
        fig.colorbar(tpc, ax=ax, label=label)
        _draw_layout(ax, network, fill_compartments=False)
        ax.set_title(f"Fingerprint heatmap — {label}")
 
    elif layout == "side":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
        _draw_layout(ax1, network, fill_compartments=True)
        ax1.set_title("Ship layout")
 
        tpc = ax2.tricontourf(xs, ys, vals, levels=20, cmap=cmap)
        fig.colorbar(tpc, ax=ax2, label=label)
        _draw_layout(ax2, network, fill_compartments=False)
        ax2.set_title(f"Fingerprint heatmap — {label}")
 
    else:
        raise ValueError(f"Unknown layout: {layout!r}. Use 'overlay' or 'side'.")
 
    plt.tight_layout()
    if show:
        plt.show()
    return fig
 



def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true", help="Run benchmarks without plot windows")
    parser.add_argument("--trials", type=_positive_int, default=300,
                        help="Trials per compartment in each benchmark (default: 300)")
    parser.add_argument("--grid-spacing", type=_positive_float, default=0.3,
                        help="Fingerprint survey grid spacing in metres (default: 0.3)")
    parser.add_argument("--calibration-samples", type=_positive_int, default=15,
                        help="Scans per fingerprint reference point (default: 15)")
    args = parser.parse_args(argv)

    radio_map_rng = Random(12345)
    demo_rng = Random(23456)
    coverage_rng = Random(34567)
    performance_rng = Random(45678)
    breakdown_rng = Random(56789)
    layout_path = Path(__file__).resolve().parent / "layouts" / "simple_layout2.yaml"
    ap1 = AccessPoint("Router1", "24:2f:d0:fb:85:b9", (4.5, 1.5, 1.5), 2412)
    ap2 = AccessPoint("Router2", "25:3f:d0:fb:85:c4", (1.5, 4.5, 1.0), 2412)
    ap3 = AccessPoint("Router3", "26:4f:d0:fb:85:c0", (8.0, 3.0, 0.5), 2412)
    ap4 = AccessPoint("Router4", "27:A0:d0:fb:85:c4", (5.0, 6.2, 1.0), 2412)
    ap5 = AccessPoint("Router5", "28:25:d0:fb:85:c0", (6.8, 4.5, 0.5), 2412)
    re1 = Receiver("Phone-Desk", (3, 5.9, 0.7))
    re2 = Receiver("Phone-Engine", (2, 1, 1))

    network = Network.from_config(
        layout_path,
        access_points=[ap1, ap2, ap3, ap4, ap5],
        receivers=[re1, re2],
    )
    environment = network.environment

    # Deployment metadata is passed explicitly: estimators never inspect the
    # simulated environment or receiver's ground-truth coordinates.
    path_loss = PathLossModel(
        reference_rssi_dbm=network.reference_rssi_dbm,
        reference_distance_m=network.reference_distance_m,
        path_loss_exponent=network.path_loss_exponent,
    )
    compartment_bounds = [comp.geometry.bounds for comp in environment.compartments]
    bounds = (
        (min(b[0] for b in compartment_bounds), min(b[1] for b in compartment_bounds)),
        (max(b[2] for b in compartment_bounds), max(b[3] for b in compartment_bounds)),
    ) if compartment_bounds else None

    ap_order = network.access_points
    radio_map = build_radio_map(
        network,
        radio_map_rng,
        ap_order,
        grid_spacing_m=args.grid_spacing,
        samples_per_point=args.calibration_samples,
    )

    nearest_ap_localiser = NearestAPLocaliser()
    weighted_distance_localiser = WeightedCentroidLocaliser(path_loss, weight_mode="distance")
    weighted_power_localiser = WeightedCentroidLocaliser(path_loss, weight_mode="linear_power")
    rss_trilateration_localiser = RSSTrilaterationLocaliser(path_loss, bounds=bounds)
    rtt_trilateration_localiser = RTTTrilaterationLocaliser(
        rtt_offset_ms=network.rtt_base_ms, bounds=bounds,
    )
    # A fixed, configured receiver height is assumed for all FTM estimates.
    # The desk receiver deliberately differs, exposing deployment model error.
    ftm_localiser = FTMLocaliser(receiver_height_m=1.0, bounds=bounds)
    fingerprint_localiser = FingerprintLocaliser(radio_map, ap_order, k=3)

    localisers = {
        "NearestAP": nearest_ap_localiser,
        "WeightedCentroid(dist)": weighted_distance_localiser,
        "WeightedCentroid(power)": weighted_power_localiser,
        "RSS Trilateration": rss_trilateration_localiser,
        "Legacy RTT": rtt_trilateration_localiser,
        "FTM": ftm_localiser,
        "Fingerprint": fingerprint_localiser,
    }

    def get_coordinate_guess(receiver: Receiver) -> Optional[tuple[float, float]]:
        """Demonstrate heuristic fusion; weights are not statistically calibrated."""
        scan = network.scan(receiver, demo_rng)
        nearest_ap_estimate = nearest_ap_localiser.locate(scan)
        weighted_power_estimate = weighted_power_localiser.locate(scan)
        fingerprint_estimate = fingerprint_localiser.locate(scan)
        ftm_estimate = ftm_localiser.locate(scan)

        if ftm_estimate is not None:
            weighted_estimates = [
                (nearest_ap_estimate, 0.1),
                (weighted_power_estimate, 0.2),
                (fingerprint_estimate, 0.4),
                (ftm_estimate, 0.3),
            ]
        else:
            weighted_estimates = [
                (nearest_ap_estimate, 0.1),
                (weighted_power_estimate, 0.3),
                (fingerprint_estimate, 0.6),
            ]

        available = [
            (estimate.position, weight)
            for estimate, weight in weighted_estimates
            if estimate is not None
        ]
        if not available:
            return None

        coords, weights = zip(*available)
        average = np.average(coords, axis=0, weights=weights)
        return float(average[0]), float(average[1])

    for i in range(10):
        re1_coordinate_guess = get_coordinate_guess(re1)
        re1_compartment_guess = (
            environment.compartment_at(re1_coordinate_guess)
            if re1_coordinate_guess is not None
            else None
        ) or "Unknown"
        re2_coordinate_guess = get_coordinate_guess(re2)
        re2_compartment_guess = (
            environment.compartment_at(re2_coordinate_guess)
            if re2_coordinate_guess is not None
            else None
        ) or "Unknown"
        print(
            f"Timestamp {i}: {re1.name} in {re1_compartment_guess} | "
            f"{re2.name} in {re2_compartment_guess}"
        )

    network.reset_channel_state()
    report_ap_coverage(network, coverage_rng, n_trials=args.trials)
    network.reset_channel_state()
    report_localiser_stats(localisers, network, performance_rng,
                           n_trials_per_compartment=args.trials)
    network.reset_channel_state()
    report_compartment_breakdown(localisers, network, breakdown_rng,
                                 n_trials_per_compartment=args.trials)

    if not args.no_plots:
        plot_fingerprint_heatmap(network, radio_map, ap_order, metric="best_signal", layout="overlay")
        plot_fingerprint_heatmap(network, radio_map, ap_order, metric="coverage", layout="side")


if __name__ == "__main__":
    main()
