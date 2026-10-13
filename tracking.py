"""Bayesian grid tracking: follow crew members over time instead of fixing each scan alone.

A snapshot localiser treats every scan independently, so one noisy scan can
put someone in the next compartment. GridTracker keeps a belief over the same
grid WallAwareLocaliser uses and, between scans, lets it spread only as far as
a person can walk in that time *without passing through a wall*. Doorways are
gaps in the walls, so the only way between compartments is through a door.
This is the grid/HMM forward filter described in e.g. Viol et al. (IPIN 2012)
and Raja & Groves (J. Navigation 2025), with walls as hard motion constraints
as in Woodman & Harle (UbiComp 2008).

Run `py tracking.py` to simulate crew walking around the accommodation deck
and compare snapshot localisers against the tracker. Results are saved to
results/tracking/<timestamp>/.
"""
import math
from random import Random
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from environment import Coord
from network import Network, Receiver, Scan
from positioning import LocalisationEstimate, WallAwareLocaliser


def _segments_cross_walls(network: Network, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For segment arrays a[i]->b[i], True where the segment crosses any wall (as walls_crossed)."""
    if not network.environment.walls:
        return np.zeros(len(a), dtype=bool)
    return network.environment.crossing_matrix(a, b).any(axis=1)


def _unblocked_pairs(network: Network, points: np.ndarray, radius: float) -> np.ndarray:
    """All (i, j) point pairs within `radius` whose straight line does not cross a wall."""
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return pairs
    blocked = _segments_cross_walls(network, points[pairs[:, 0]], points[pairs[:, 1]])
    return pairs[~blocked]


def _walk_graph(network: Network, points: np.ndarray, spacing: float) -> sparse.csr_matrix:
    """Sparse graph linking neighbouring grid points that are not separated by a wall,
    weighted by distance. Shortest paths in it are walking routes through doorways."""
    pairs = _unblocked_pairs(network, points, radius=1.5 * spacing)
    lengths = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
    graph = sparse.csr_matrix((lengths, (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
    return (graph + graph.T).tocsr()


class GridTracker:
    """Grid Bayes filter (HMM forward pass) with wall-constrained motion.

    Predict: belief diffuses with a Gaussian kernel on *walking distance* (the
    shortest wall-free route on the grid), whose spread is set by walking speed
    x scan interval. Belief therefore moves between compartments only through
    doorways, including turning around a door jamb within one scan interval.
    A small `reacquire_probability` is spread uniformly over the deck so the
    filter can recover if it ever locks onto the wrong compartment.

    Update: multiply by the scan likelihood from a WallAwareLocaliser. Build
    that localiser with `robust=True`: with Gaussian residuals a single
    corrupted scan can outweigh many seconds of consistent evidence.
    """

    def __init__(
        self,
        localiser: WallAwareLocaliser,
        network: Network,
        scan_interval_s: float = 1.0,
        walking_speed_mps: float = 1.4,
        reacquire_probability: float = 1e-3,
    ):
        if scan_interval_s <= 0 or walking_speed_mps <= 0:
            raise ValueError("scan_interval_s and walking_speed_mps must be positive")
        if not 0.0 <= reacquire_probability < 1.0:
            raise ValueError("reacquire_probability must be in [0, 1)")

        self.localiser = localiser
        points = localiser.grid_points
        n = len(points)
        spacing = self._spacing(points)

        sigma = max(0.5 * walking_speed_mps * scan_interval_s, 0.5 * spacing)
        graph = _walk_graph(network, points, spacing)
        rows, cols, dists = [], [], []
        for start in range(0, n, 512):  # chunked so the dense distance block stays small
            sources = np.arange(start, min(start + 512, n))
            d = dijkstra(graph, directed=False, indices=sources, limit=3.0 * sigma)
            r, c = np.nonzero(np.isfinite(d))
            rows.append(sources[r])
            cols.append(c)
            dists.append(d[r, c])
        rows, cols, dists = np.concatenate(rows), np.concatenate(cols), np.concatenate(dists)
        transition = sparse.csr_matrix((np.exp(-dists ** 2 / (2 * sigma ** 2)), (rows, cols)), shape=(n, n))
        row_sums = np.asarray(transition.sum(axis=1)).ravel()
        # predict step is prior = T^T @ belief, with T row-stochastic (from -> to)
        self._predict_matrix = (sparse.diags(1.0 / row_sums) @ transition).T.tocsr()

        self.reacquire_probability = reacquire_probability
        self.belief = np.full(n, 1.0 / n)

    @staticmethod
    def _spacing(points: np.ndarray) -> float:
        distances, _ = cKDTree(points).query(points[:50], k=2)
        return float(np.median(distances[:, 1]))

    def reset(self) -> None:
        self.belief = np.full(len(self.belief), 1.0 / len(self.belief))

    def update(self, scan: Scan) -> LocalisationEstimate:
        prior = self._predict_matrix @ self.belief
        prior = (1.0 - self.reacquire_probability) * prior + self.reacquire_probability / len(prior)

        log_posterior = np.log(np.maximum(prior, 1e-300)) + self.localiser.log_likelihood(scan)

        log_posterior -= log_posterior.max()
        posterior = np.exp(log_posterior)
        self.belief = posterior / posterior.sum()

        heard = sum(1 for ap in self.localiser.access_points if scan.get(ap) is not None)
        return self.localiser.estimate_from_log_posterior(np.log(np.maximum(self.belief, 1e-300)), heard)


class SnapshotAsTracker:
    """Adapts a per-scan localiser to the tracker interface (no memory)."""

    def __init__(self, localiser):
        self.localiser = localiser

    def reset(self) -> None:
        pass

    def update(self, scan: Scan) -> Optional[LocalisationEstimate]:
        return self.localiser.locate(scan)


def simulate_walk(
    network: Network,
    rng: Random,
    n_steps: int,
    scan_interval_s: float = 1.0,
    walking_speed_mps: float = 1.2,
    spacing_m: float = 0.2,
    dwell_steps: tuple[int, int] = (5, 30),
) -> List[Coord]:
    """True positions of one crew member, one per scan.

    Repeatedly picks a random compartment reachable from the current position
    and a random point in it, walks the shortest wall-free route there (through
    doorways) at walking speed, then stands there for a random dwell time.
    Consecutive positions are at most walking_speed x interval apart *along the
    route*; the straight line between them can clip a door jamb.
    """
    if n_steps < 0 or dwell_steps[0] < 0 or dwell_steps[1] < dwell_steps[0]:
        raise ValueError("n_steps must be >= 0 and dwell_steps a valid (min, max) range")
    points, compartments = WallAwareLocaliser._build_grid(network, spacing_m)
    graph = _walk_graph(network, points, spacing_m)

    current = rng.randrange(len(points))
    positions: List[Coord] = []
    step_m = walking_speed_mps * scan_interval_s

    while len(positions) < n_steps:
        for _ in range(rng.randint(*dwell_steps)):
            positions.append(tuple(points[current]))

        distances, predecessors = dijkstra(graph, directed=False, indices=current, return_predecessors=True)
        reachable = np.isfinite(distances)
        names = sorted(set(compartments[reachable]))  # always includes the current compartment
        target_room = rng.choice(names)
        candidates = np.flatnonzero(reachable & (compartments == target_room))
        target = int(candidates[rng.randrange(len(candidates))])
        if target == current:
            positions.append(tuple(points[current]))  # guarantees progress even with zero dwell
            continue

        route = [target]
        while route[-1] != current:
            route.append(int(predecessors[route[-1]]))
        route_points = points[route[::-1]]

        # walk along the route polyline, emitting one position per scan interval
        seg = np.linalg.norm(np.diff(route_points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg)])
        for s in np.arange(step_m, cumulative[-1], step_m):
            k = int(np.searchsorted(cumulative, s) - 1)
            t = (s - cumulative[k]) / seg[k] if seg[k] > 0 else 0.0
            positions.append(tuple(route_points[k] + t * (route_points[k + 1] - route_points[k])))
        current = target

    return [(float(x), float(y)) for x, y in positions[:n_steps]]


def evaluate_tracking(
    trackers: dict,
    network: Network,
    walks: List[List[Coord]],
    rng: Random,
) -> tuple[dict, dict]:
    """Feed identical scan sequences to every tracker. Returns (stats, per-walk estimates)."""
    environment = network.environment
    records = {name: {"errors": [], "correct": 0, "total": 0, "est_changes": 0} for name in trackers}
    estimates = {name: [] for name in trackers}
    true_changes = 0

    for walk in walks:
        network.reset_channel_state()
        for tracker in trackers.values():
            tracker.reset()
        last_est_room = dict.fromkeys(trackers)
        walk_estimates = {name: [] for name in trackers}
        last_true_room = None

        for true_pos in walk:
            scan = network.scan(Receiver("crew", (*true_pos, 1.0)), rng)
            true_room = environment.compartment_at(true_pos)
            if last_true_room is not None and true_room != last_true_room:
                true_changes += 1
            last_true_room = true_room

            for name, tracker in trackers.items():
                est = tracker.update(scan)
                rec = records[name]
                rec["total"] += 1
                if est is None:
                    walk_estimates[name].append(None)
                    continue
                walk_estimates[name].append(est.position)
                rec["errors"].append(math.dist(true_pos, est.position))
                est_room = environment.compartment_at(est.position)
                rec["correct"] += est_room == true_room
                if last_est_room[name] is not None and est_room != last_est_room[name]:
                    rec["est_changes"] += 1
                last_est_room[name] = est_room

        for name in trackers:
            estimates[name].append(walk_estimates[name])

    stats = {}
    for name, rec in records.items():
        errors = sorted(rec["errors"])
        stats[name] = {
            "errors": errors,
            "mean_err": float(np.mean(errors)) if errors else float("nan"),
            "median_err": errors[len(errors) // 2] if errors else float("nan"),
            "p90_err": errors[int(len(errors) * 0.9)] if errors else float("nan"),
            "compartment_acc": rec["correct"] / rec["total"],
            # 1.0 means the estimate changed compartment exactly as often as the person did
            "room_changes_ratio": rec["est_changes"] / max(true_changes, 1),
        }
    return stats, estimates


def main() -> None:
    from fingerprinting import FingerprintLocaliser, build_radio_map
    from scenarios import SCENARIOS, SURFACE, INK, GRID, SERIES_COLOURS, _style_axis, build_networks
    from simulation import RESULTS_DIR, _draw_layout, save_figures, start_saving_results

    run_dir = start_saving_results(RESULTS_DIR / "tracking")
    scan_interval_s = 1.0
    deck_scenarios = [s for s in SCENARIOS if s.layout == "accommodation_deck.yaml"]
    example = None

    for scenario in deck_scenarios:
        truth, model, survey = build_networks(scenario)
        radio_map = build_radio_map(survey, Random(12345), survey.access_points,
                                    grid_spacing_m=scenario.radio_map_spacing_m,
                                    samples_per_point=scenario.radio_map_samples)
        hybrid = WallAwareLocaliser(model, measurement="hybrid", grid_spacing_m=0.2)
        robust_hybrid = WallAwareLocaliser(model, measurement="hybrid", grid_spacing_m=0.2, robust=True)
        robust_rss = WallAwareLocaliser(model, measurement="rss", grid_spacing_m=0.2, robust=True)
        trackers = {
            "Fingerprint (snapshot)": SnapshotAsTracker(FingerprintLocaliser(radio_map, survey.access_points, k=3)),
            "WallAware Hybrid (snapshot)": SnapshotAsTracker(hybrid),
            "WallAware Hybrid robust (snap)": SnapshotAsTracker(robust_hybrid),
            "Tracker RSS": GridTracker(robust_rss, model, scan_interval_s=scan_interval_s),
            "Tracker Hybrid": GridTracker(robust_hybrid, model, scan_interval_s=scan_interval_s),
        }

        walk_rng = Random(67890)
        walks = [simulate_walk(truth, walk_rng, n_steps=300, scan_interval_s=scan_interval_s) for _ in range(8)]
        stats, estimates = evaluate_tracking(trackers, truth, walks, Random(78901))

        print(f"##### {scenario.name}: {scenario.description}")
        print(f"      {len(walks)} walks x {len(walks[0])} scans, one scan every {scan_interval_s:.0f} s\n")
        print(f"{'Method':<32}{'Mean err':<10}{'Median err':<12}{'P90 err':<10}{'Compartment acc':<17}{'Room changes vs truth':<22}")
        for name, s in stats.items():
            print(f"{name:<32}{s['mean_err']:<10.2f}{s['median_err']:<12.2f}{s['p90_err']:<10.2f}"
                  f"{s['compartment_acc']:<17.1%}{s['room_changes_ratio']:<.2f}x")
        print()
        if example is None:
            example = (scenario.name, truth, walks[0], estimates)

    # Example walk: truth vs best snapshot vs tracker
    name, truth, walk, estimates = example
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=SURFACE)
    _draw_layout(ax, truth, fill_compartments=False)
    window = slice(0, 150)
    walk_xy = np.array(walk[window])
    snap = np.array([p for p in estimates["WallAware Hybrid (snapshot)"][0][window] if p is not None])
    track = np.array([p for p in estimates["Tracker Hybrid"][0][window] if p is not None])
    ax.plot(walk_xy[:, 0], walk_xy[:, 1], color=INK, linewidth=2, label="True path")
    ax.scatter(snap[:, 0], snap[:, 1], s=14, color=SERIES_COLOURS[1], label="WallAware Hybrid (snapshot)", zorder=4)
    ax.plot(track[:, 0], track[:, 1], color=SERIES_COLOURS[0], linewidth=2, label="Tracker Hybrid", zorder=5)
    ax.set_title(f"{name}: first {window.stop} s of one walk", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.grid(True, color=GRID, linewidth=0.6)
    _style_axis(ax)
    fig.tight_layout()

    save_figures(run_dir, {"example_walk": fig})
    print(f"Saved example_walk.png to {run_dir}")


if __name__ == "__main__":
    main()
