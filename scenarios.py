"""Benchmark every localiser across several simulated ships and channel conditions.

Each scenario separates three networks:
- truth:  the channel the scans come from (layout preset + channel_overrides)
- model:  what the localisers believe (layout preset + model_overrides). It never
          sees channel_overrides, so localisers are not told a channel is harsh.
- survey: the channel the fingerprint radio map was recorded under. It uses a new
          shadowing seed, i.e. an independent shadow field: a worst case for "the
          ship changed since the survey" (simulation.py surveys the truth's field).

Run with `py scenarios.py`. Tables are printed and saved, with plots, to
results/scenarios/<timestamp>/.
"""
import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
from random import Random
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from environment import ShipEnvironment
from fingerprinting import FingerprintLocaliser, build_radio_map
from network import AccessPoint, Network
from positioning import (
    NearestAPLocaliser,
    RSSTrilaterationLocaliser,
    RTTTrilaterationLocaliser,
    WallAwareLocaliser,
    WeightedCentroidLocaliser,
)
from simulation import RESULTS_DIR, _draw_layout, evaluate_localisers, sample_trials, save_figures, start_saving_results

LAYOUTS_DIR = Path(__file__).resolve().parent / "layouts"

# Chart colours: categorical slots in fixed order, plus chart ink (see dataviz reference palette)
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"

SMALL_SHIP_APS = [
    AccessPoint("Router1", "24:2f:d0:fb:85:b9", (4.5, 1.5, 1.5), 2412),
    AccessPoint("Router2", "25:3f:d0:fb:85:c4", (1.5, 4.5, 1.0), 2412),
    AccessPoint("Router3", "26:4f:d0:fb:85:c0", (8.0, 3.0, 0.5), 2412),
    AccessPoint("Router4", "27:A0:d0:fb:85:c4", (5.0, 6.2, 1.0), 2412),
    AccessPoint("Router5", "28:25:d0:fb:85:c0", (6.8, 4.5, 0.5), 2412),
]

# Ceiling-mounted, mostly in the corridor: most cabins have no AP of their own
DECK_APS = [
    AccessPoint("Corridor-W", "30:00:00:00:00:01", (3.0, 5.0, 2.5), 2412),
    AccessPoint("Corridor-MW", "30:00:00:00:00:02", (9.0, 5.0, 2.5), 2437),
    AccessPoint("Corridor-ME", "30:00:00:00:00:03", (15.0, 5.0, 2.5), 2462),
    AccessPoint("Corridor-E", "30:00:00:00:00:04", (21.0, 5.0, 2.5), 2412),
    AccessPoint("Mess", "30:00:00:00:00:05", (12.0, 2.0, 2.5), 2437),
    AccessPoint("ECR", "30:00:00:00:00:06", (15.0, 8.0, 2.5), 2462),
]

# The localisers' belief is off by a plausible amount: walls 25% weaker than
# reality, a steeper path-loss exponent, and half the real RTT NLOS bias.
MISCALIBRATION = {"path_loss_exponent": 2.6, "rtt_nlos_bias_per_wall_m": 0.5}
MISCALIBRATED_WALL_SCALE = 0.75

# A stress test, not a calibrated preset: doubled shadowing and receiver noise,
# frequent interference bursts, noisier RTT with more NLOS outliers.
HARSH_CHANNEL = {
    "shadowing_std_db": 4.0,
    "measurement_noise_std_db": 1.5,
    "interference_burst_start_probability": 0.1,
    "rtt_ranging_noise_m": 1.0,
    "rtt_nlos_outlier_probability": 0.2,
}

# Commodity-phone RTT at 2.4 GHz / 20 MHz: instrument bias of about -1 to -1.3 m
# (Pixel 4a, Raja & Groves ION 2023) and metre-level spread at 20 MHz
# (Ibrahim et al. 2018; Android's 802.11mc target allows up to 8 m P90 at 20 MHz).
# The localisers keep believing the layout's nominal RTT model.
REALISTIC_2G4_RTT = {
    "rtt_range_bias_m": -1.2,
    "rtt_ranging_noise_m": 1.5,
    "rtt_nlos_outlier_probability": 0.15,
}


@dataclass
class Scenario:
    name: str
    description: str
    layout: str
    access_points: List[AccessPoint]
    channel_overrides: dict = field(default_factory=dict)  # applied to the true channel
    model_overrides: dict = field(default_factory=dict)  # applied to the localisers' belief only
    model_wall_scale: float = 1.0  # localisers believe every wall loses this fraction of its true dB
    phone_rssi_bias_db: float = 0.0  # the tracked phone reads this much above/below the survey phone
    radio_map_spacing_m: float = 0.3
    radio_map_samples: int = 15


SCENARIOS = [
    Scenario(
        "Baseline",
        "simple_layout2, one AP per room, localisers know the channel model",
        "simple_layout2.yaml",
        SMALL_SHIP_APS,
    ),
    Scenario(
        "Miscalibrated",
        "as Baseline, but localisers assume 15 dB walls, n=2.6, 0.5 m RTT bias/wall",
        "simple_layout2.yaml",
        SMALL_SHIP_APS,
        model_overrides=MISCALIBRATION,
        model_wall_scale=MISCALIBRATED_WALL_SCALE,
    ),
    Scenario(
        "Harsh channel",
        "as Baseline, with 4 dB shadowing, noisier RSSI/RTT, more interference and outliers (localisers unaware)",
        "simple_layout2.yaml",
        SMALL_SHIP_APS,
        channel_overrides=HARSH_CHANNEL,
    ),
    Scenario(
        "Phone reads 6 dB low",
        "as Baseline, but the tracked phone's RSSI is 6 dB below the model/survey phone",
        "simple_layout2.yaml",
        SMALL_SHIP_APS,
        phone_rssi_bias_db=-6.0,
    ),
    Scenario(
        "Realistic 2.4GHz RTT",
        "as Baseline, but RTT has -1.2 m bias, 1.5 m noise, 15% NLOS outliers (localisers unaware)",
        "simple_layout2.yaml",
        SMALL_SHIP_APS,
        channel_overrides=REALISTIC_2G4_RTT,
    ),
    Scenario(
        "Room without AP",
        "as Baseline, with the Laundry router (Router4) removed",
        "simple_layout2.yaml",
        [ap for ap in SMALL_SHIP_APS if ap.name != "Router4"],
    ),
    Scenario(
        "Accommodation deck",
        "24x10 m deck, corridor + 10 rooms, doorways, joiner panels, 6 APs (4 in corridor)",
        "accommodation_deck.yaml",
        DECK_APS,
        radio_map_spacing_m=0.5,
        radio_map_samples=10,
    ),
    Scenario(
        "Deck, miscalibrated",
        "Accommodation deck with the Miscalibrated belief",
        "accommodation_deck.yaml",
        DECK_APS,
        model_overrides=MISCALIBRATION,
        model_wall_scale=MISCALIBRATED_WALL_SCALE,
        radio_map_spacing_m=0.5,
        radio_map_samples=10,
    ),
    Scenario(
        "Deck, harsh channel",
        "Accommodation deck with the Harsh channel",
        "accommodation_deck.yaml",
        DECK_APS,
        channel_overrides=HARSH_CHANNEL,
        radio_map_spacing_m=0.5,
        radio_map_samples=10,
    ),
]

# Localisers drawn in the per-scenario error CDFs (max four series)
CDF_LOCALISERS = ["RSS Trilateration", "Fingerprint", "WallAware RSS", "WallAware Hybrid"]


def build_networks(scenario: Scenario) -> tuple[Network, Network, Network]:
    base = Network.from_config(LAYOUTS_DIR / scenario.layout, access_points=list(scenario.access_points))
    truth = replace(base, **scenario.channel_overrides)

    believed_walls = [
        replace(wall, attenuation_db=wall.attenuation_db * scenario.model_wall_scale)
        for wall in base.environment.walls
    ]
    model = replace(
        base,
        environment=ShipEnvironment(compartments=base.environment.compartments, walls=believed_walls),
        **scenario.model_overrides,
    )
    survey = replace(truth, shadowing_seed=truth.shadowing_seed + 1)
    return truth, model, survey


def build_localisers(scenario: Scenario, model: Network, survey: Network) -> dict:
    ap_order = survey.access_points
    radio_map = build_radio_map(
        survey,
        Random(12345),
        ap_order,
        grid_spacing_m=scenario.radio_map_spacing_m,
        samples_per_point=scenario.radio_map_samples,
    )
    return {
        "NearestAP": NearestAPLocaliser(),
        "WeightedCentroid(power)": WeightedCentroidLocaliser(model, weight_mode="linear_power"),
        "RSS Trilateration": RSSTrilaterationLocaliser(model),
        "RTT": RTTTrilaterationLocaliser(model),
        "Fingerprint": FingerprintLocaliser(radio_map, ap_order, k=3),
        "WallAware RSS": WallAwareLocaliser(model, measurement="rss"),
        "WallAware RTT": WallAwareLocaliser(model, measurement="rtt"),
        "WallAware Hybrid": WallAwareLocaliser(model, measurement="hybrid"),
        "WallAware RSS+offset": WallAwareLocaliser(model, measurement="rss", device_offset_sigma_db=6.0),
    }


def compartment_breakdown(truth: Network, localisers: dict, trials: list) -> dict:
    """{localiser: {compartment: accuracy}} from the same trials."""
    environment = truth.environment
    names = [c.name for c in environment.compartments]
    breakdown = {}
    for loc_name, loc in localisers.items():
        correct = dict.fromkeys(names, 0)
        totals = dict.fromkeys(names, 0)
        for compartment_name, _, scan in trials:
            totals[compartment_name] += 1
            estimate = loc.locate(scan)
            if estimate is not None and environment.compartment_at(estimate.position) == compartment_name:
                correct[compartment_name] += 1
        breakdown[loc_name] = {n: correct[n] / totals[n] for n in names}
    return breakdown


def print_scenario(scenario: Scenario, truth: Network, results: dict, breakdown: dict) -> None:
    print(f"##### {scenario.name}: {scenario.description}")
    print(f"      {len(truth.environment.compartments)} compartments, {len(truth.access_points)} APs\n")
    print(f"{'Method':<24}{'Mean err':<10}{'Median err':<12}{'P90 err':<10}{'Miss rate':<11}{'Compartment acc':<16}{'Worst room':<24}")
    for name, stats in results.items():
        worst_room, worst_acc = min(breakdown[name].items(), key=lambda kv: kv[1])
        print(f"{name:<24}{stats['mean_err']:<10.2f}{stats['median_err']:<12.2f}{stats['p90_err']:<10.2f}"
              f"{stats['miss_rate']:<11.1%}{stats['compartment_acc']:<16.1%}{worst_room} ({worst_acc:.0%})")
    print()


def print_summary(all_results: dict) -> None:
    localiser_names = list(next(iter(all_results.values())).keys())
    for metric, label, fmt in (
        ("mean_err", "Mean error (m)", "{:.2f}"),
        ("compartment_acc", "Compartment accuracy", "{:.1%}"),
    ):
        print(f"=== Summary: {label} ===")
        print(f"{'Scenario':<22}" + "".join(f"{n:<14}" for n in _short_names(localiser_names)))
        for scenario_name, results in all_results.items():
            row = "".join(f"{fmt.format(results[n][metric]):<14}" for n in localiser_names)
            print(f"{scenario_name:<22}{row}")
        print()


def _short_names(names: list) -> list:
    return [
        n.replace("WeightedCentroid(power)", "WCentroid").replace("RSS Trilateration", "RSS Tri")
        .replace("WallAware ", "WA ").replace("Fingerprint", "Fingerprnt")
        for n in names
    ]


def save_summary_csv(run_dir: Path, all_results: dict) -> None:
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "localiser", "mean_err_m", "median_err_m", "p90_err_m", "miss_rate", "compartment_acc"])
        for scenario_name, results in all_results.items():
            for name, s in results.items():
                writer.writerow([scenario_name, name, f"{s['mean_err']:.3f}", f"{s['median_err']:.3f}",
                                 f"{s['p90_err']:.3f}", f"{s['miss_rate']:.4f}", f"{s['compartment_acc']:.4f}"])


def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


def plot_summary_heatmap(all_results: dict, metric: str, title: str, cmap: str, fmt: str, vmin, vmax):
    scenario_names = list(all_results.keys())
    localiser_names = list(next(iter(all_results.values())).keys())
    values = np.array([[all_results[s][n][metric] for n in localiser_names] for s in scenario_names])

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(scenario_names) + 2), facecolor=SURFACE)
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(localiser_names)), _short_names(localiser_names))
    ax.set_yticks(range(len(scenario_names)), scenario_names)
    ax.tick_params(length=0)
    threshold = vmin + 0.6 * (vmax - vmin)
    for i in range(len(scenario_names)):
        for j in range(len(localiser_names)):
            dark_cell = (values[i, j] > threshold) == (cmap.endswith("_r") is False)
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if dark_cell else INK)
    ax.set_title(title, loc="left", fontsize=12)
    _style_axis(ax)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout()
    return fig


def plot_error_cdfs(all_results: dict):
    n = len(all_results)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), sharex=True, sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for ax, (scenario_name, results) in zip(axes.ravel(), all_results.items()):
        for colour, name in zip(SERIES_COLOURS, CDF_LOCALISERS):
            errors = np.array(results[name]["errors"])
            if len(errors) == 0:
                continue
            ax.plot(errors, np.arange(1, len(errors) + 1) / len(errors), color=colour, linewidth=2, label=name)
        ax.set_title(scenario_name, loc="left", fontsize=10)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 1)
        ax.grid(True, color=GRID, linewidth=0.6)
        _style_axis(ax)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Position error (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of fixes")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle("Error CDFs (higher and further left is better)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_layouts(networks: dict):
    """One floor plan per distinct (layout, AP set). `networks` maps scenario name -> (Scenario, Network)."""
    unique = {}
    for scenario_name, (scenario, network) in networks.items():
        key = (scenario.layout, tuple(ap.name for ap in network.access_points))
        unique.setdefault(key, (scenario_name, network))
    fig, axes = plt.subplots(1, len(unique), figsize=(6.5 * len(unique), 5), facecolor=SURFACE)
    for ax, (scenario_name, network) in zip(np.atleast_1d(axes), unique.values()):
        _draw_layout(ax, network, fill_compartments=True)
        ax.set_title(scenario_name, loc="left")
        _style_axis(ax)
    fig.tight_layout()
    return fig


def main() -> None:
    run_dir = start_saving_results(RESULTS_DIR / "scenarios")
    all_results = {}
    networks = {}

    for scenario in SCENARIOS:
        truth, model, survey = build_networks(scenario)
        localisers = build_localisers(scenario, model, survey)

        # Fresh, identically seeded scan stream per scenario; every localiser sees the same scans
        truth.reset_channel_state()
        trials = sample_trials(truth, Random(45678), n_trials_per_compartment=200,
                               receiver_rssi_bias_db=scenario.phone_rssi_bias_db)
        results = evaluate_localisers(localisers, truth, trials)
        breakdown = compartment_breakdown(truth, localisers, trials)

        print_scenario(scenario, truth, results, breakdown)
        all_results[scenario.name] = results
        networks[scenario.name] = (scenario, truth)

    print_summary(all_results)
    save_summary_csv(run_dir, all_results)

    figures = {
        "summary_compartment_accuracy": plot_summary_heatmap(
            all_results, "compartment_acc", "Compartment accuracy by scenario", "Blues", "{:.0%}", 0.0, 1.0),
        "summary_mean_error": plot_summary_heatmap(
            all_results, "mean_err", "Mean position error (m) by scenario", "Oranges", "{:.2f}", 0.0, 6.0),
        "error_cdfs": plot_error_cdfs(all_results),
        "layouts": plot_layouts(networks),
    }
    save_figures(run_dir, figures)
    print(f"Saved summary.csv and {len(figures)} figures to {run_dir}")


if __name__ == "__main__":
    main()
