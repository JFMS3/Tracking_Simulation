"""Held-out benchmark of the sensor-fusion trackers (docs/sensor_fusion_spec.md).

Run with `py fusion_benchmark.py` (about 20 minutes for the 30-walk confirmatory set on 4-6 worker processes; `--jobs 1` runs
serially). Tables (results.txt, summary.csv, per_walk.csv, results.md) and plots are saved to
results/fusion/<timestamp>/.

Seeds (spec, "Seeds: tuning vs held-out validation"), HELD-OUT only:
- walk i: Random(2024 + i), simulated on the accommodation-deck truth network
  (all deck scenarios share the geometry, so the walks are the same in every scenario);
- scans: one stream Random(3024) per scenario, consumed in walk order by evaluate_tracking;
- IMU: one stream Random(4024), re-created for every IMU model (so each motion set is
  drawn the same way), consumed in walk order;
- particles: every ParticleTracker is built with Random(5024).
The bootstrap uses its own stream (BOOTSTRAP_SEED); it only affects confidence intervals.

Pairing: every (scenario, method) pair runs in its own process, but the scan stream does
not depend on the trackers, so every method of a scenario sees identical scans and IMU
readings. This is checked: each run hashes the scans it was fed and the report asserts
the hashes agree within a scenario (spec V6).

Statistics (research notes, fusion_filters_validation.md Q6): walks, not epochs, are the
independent unit. Per walk, the mean error of each method is paired with GridTracker's
on the same walk; the report gives a two-sided Wilcoxon signed-rank test (exact for
small n), Holm-adjusted across the methods of a scenario, a Friedman test across all
methods, and a paired bootstrap 95% CI (10,000 resamples of walks) of the mean-error
difference. Coverage is reported pooled, with a walk-level bootstrap CI and the
reliability at several nominal levels; ANEES with its chi-square bounds for 2 dof.

Metric conventions follow tracking.evaluate_tracking: errors only where an estimate
exists, room accuracy counts a missing estimate as wrong, median/P75/P90 are order
statistics errors[int(q * n)], room-change ratio = estimated / true compartment changes.
P75 is the EvAAL/IPIN score.

"PDR only" is a truth-aided reference, not a tracker: dead reckoning from the true
start position, with the absolute heading aligned to the true heading at the first
moving interval (as tests/test_imu.py V2); afterwards only the phone's readings are used
and walls are ignored.
"""
import argparse
import csv
import functools
import hashlib
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from random import Random
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # figures are only saved
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats as sps  # noqa: E402

from imu import IMU_PRESETS, ImuModel, simulate_imu  # noqa: E402
from positioning import LocalisationEstimate, WallAwareLocaliser  # noqa: E402
from scenarios import (  # noqa: E402
    GRID, INK, INK_SECONDARY, REALISTIC_2G4_RTT, SCENARIOS, SERIES_COLOURS, SURFACE, _style_axis, build_networks,
)
from tracking import FixedLagSmoother, GridTracker, SnapshotAsTracker, evaluate_tracking, simulate_walk  # noqa: E402

# ---------------------------------------------------------------- held-out seeds and sizes

WALK_SEED_BASE = 2024
SCAN_SEED = 3024
IMU_SEED = 4024
PARTICLE_SEED = 5024
BOOTSTRAP_SEED = 6024
TUNING_SEEDS = (67890, 78901, 89012)
assert ({WALK_SEED_BASE + i for i in range(30)} | {SCAN_SEED, IMU_SEED, PARTICLE_SEED, BOOTSTRAP_SEED}) \
    .isdisjoint(TUNING_SEEDS)

N_WALKS = 30  # confirmatory set (spec): walks Random(2024 + i), i in 0..29
MAX_WALKS = 30
FRESH_FROM = 8  # walks i >= 8 (seeds 2032-2053) were never run before the confirmatory benchmark
WALK_STEPS = 300
SCAN_INTERVAL_S = 1.0
GRID_SPACING_M = 0.2
N_PARTICLES = 2000  # ParticleTracker's default
OUTAGE_EPOCHS = range(100, 115)  # 15 s Wi-Fi outage starting at t = 100 s
N_BOOTSTRAP = 10_000
CHI2_2DOF = {0.5: 1.3863, 0.68: 2.2789, 0.9: 4.6052, 0.95: 5.9915}  # -2 ln(1 - level)

DECK, MISCALIBRATED, HARSH = "Accommodation deck", "Deck, miscalibrated", "Deck, harsh channel"
_ERROR_FIELD_MARKERS = ("noise", "bias", "probability", "per_s", "std", "error")

# Method names (keys of the tables)
SNAP = "Snapshot hybrid"
SNAP_MIX = "Snapshot hybrid + mixture"
GRIDT = "Grid (hybrid robust)"
GRID_MIX = "Grid + mixture"
SMOOTH = {lag: f"Smoother lag {lag}" for lag in (2, 5, 10)}
PF = "PF gyro"
PF_MIX = "PF gyro + mixture"
PF_NO_IMU = "PF no IMU (Wi-Fi only)"
PF_PERFECT = "PF perfect IMU"
PF_MAG_NAIVE = "PF + magnetometer (naive)"
PF_MAG_HONEST = "PF + magnetometer (honest bias)"
PF_COMPASS = "PF ideal compass"
PF_CONS = "PF (conservative model)"
PF_CONS_MISSPEC = "PF (told default model)"
PDR = "PDR only (truth-aided ref.)"

MAIN_METHODS = [SNAP, GRIDT, SMOOTH[2], SMOOTH[5], SMOOTH[10], PF, PF_NO_IMU, PDR]


@dataclass(frozen=True)
class BenchScenario:
    name: str
    base: str                      # scenarios.SCENARIOS entry providing (truth, model)
    methods: tuple
    imu: str = "default"           # IMU model of the readings: default | conservative | magnetometer
    truth_overrides: tuple = ()    # extra (key, value) pairs applied to the truth channel only
    outage: bool = False
    cdf_methods: tuple = (SNAP, GRIDT, SMOOTH[5], PF)
    note: str = ""


BENCH_SCENARIOS = [
    BenchScenario(DECK, DECK, tuple(MAIN_METHODS + [PF_PERFECT])),
    BenchScenario(MISCALIBRATED, MISCALIBRATED, tuple(MAIN_METHODS)),
    BenchScenario(HARSH, HARSH, tuple(MAIN_METHODS)),
    BenchScenario("Deck + 15 s outage", DECK, tuple(MAIN_METHODS), outage=True,
                  note="no scans for t in [100, 115) s; outage-window metrics reported separately"),
    BenchScenario("Deck, realistic 2.4 GHz RTT + 1.5 m AP offsets", DECK,
                  (SNAP, SNAP_MIX, GRIDT, SMOOTH[5], PF, PF_MIX, PDR),
                  truth_overrides=tuple(sorted({**REALISTIC_2G4_RTT, "rtt_ap_offset_std_m": 1.5}.items())),
                  cdf_methods=(SNAP, SNAP_MIX, GRIDT, PF),
                  note="truth: scenarios.REALISTIC_2G4_RTT + rtt_ap_offset_std_m=1.5; localisers keep the nominal model"),
    BenchScenario("Deck, conservative IMU", DECK, (GRIDT, PF_CONS, PF_CONS_MISSPEC, PDR), imu="conservative",
                  cdf_methods=(GRIDT, PF_CONS, PF_CONS_MISSPEC, PDR),
                  note="readings from imu.IMU_PRESETS['conservative']"),
    BenchScenario("Deck, magnetometer on", DECK, (GRIDT, PF, PF_MAG_NAIVE, PF_MAG_HONEST, PDR), imu="magnetometer",
                  cdf_methods=(GRIDT, PF, PF_MAG_NAIVE, PF_MAG_HONEST),
                  note="readings: default ImuModel with the steel-hull magnetometer on (30 deg bias field); "
                       "naive PF believes bias 0, honest PF believes the 30 deg bias"),
    BenchScenario("Deck, ideal compass", DECK, (GRIDT, PF, PF_COMPASS, PDR), imu="ideal_compass",
                  cdf_methods=(GRIDT, PF, PF_COMPASS, PDR),
                  note="readings: default ImuModel with an UNBIASED magnetometer (white noise only); "
                       "T6b compares the gyro-only PF with the PF that uses this compass"),
]


# ---------------------------------------------------------------- IMU models

def perfect_imu_model() -> ImuModel:
    """ImuModel with every error source off (same rule as tests/test_imu.py)."""
    base = ImuModel()
    changes = {}
    for f in base.__dataclass_fields__.values():
        value = getattr(base, f.name)
        if f.name == "magnetometer_available":
            changes[f.name] = False
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        elif any(marker in f.name for marker in _ERROR_FIELD_MARKERS):
            changes[f.name] = type(value)(0)
    return replace(base, **changes)


IMU_MODELS = {
    "default": ImuModel,
    "conservative": lambda: IMU_PRESETS["conservative"],
    "magnetometer": lambda: replace(ImuModel(), magnetometer_available=True),
    "perfect": perfect_imu_model,
    "ideal_compass": lambda: replace(ImuModel(), magnetometer_available=True, magnetometer_bias_std_rad=0.0),
}


# ---------------------------------------------------------------- per-process context

@functools.lru_cache(maxsize=None)
def deck_walks(n_walks: int, n_steps: int):
    truth = build_networks(next(s for s in SCENARIOS if s.name == DECK))[0]
    return [simulate_walk(truth, Random(WALK_SEED_BASE + i), n_steps=n_steps, scan_interval_s=SCAN_INTERVAL_S)
            for i in range(n_walks)]


@functools.lru_cache(maxsize=None)
def motions_for(kind: str, n_walks: int, n_steps: int):
    rng = Random(IMU_SEED)
    model = IMU_MODELS[kind]()
    return [simulate_imu(walk, rng, model, scan_interval_s=SCAN_INTERVAL_S) for walk in deck_walks(n_walks, n_steps)]


def scenario_by_name(name: str) -> BenchScenario:
    return next(s for s in BENCH_SCENARIOS if s.name == name)


def networks_for(scenario: BenchScenario):
    truth, model, _ = build_networks(next(s for s in SCENARIOS if s.name == scenario.base))
    if scenario.truth_overrides:
        truth = replace(truth, **dict(scenario.truth_overrides))
    return truth, model


def outage_masks(walks):
    return [[t not in OUTAGE_EPOCHS for t in range(len(walk))] for walk in walks]


# ---------------------------------------------------------------- tracker wrappers

class Instrumented:
    """Delegates to a tracker; records every estimate (incl. flush), the time spent, and a hash
    of the scans it was fed. `motion_override` replaces the harness's motion feed (per walk)
    and `drop_motion` feeds None, so trackers with different IMU inputs still share scans."""

    def __init__(self, inner, motion_override: Optional[list] = None, drop_motion: bool = False):
        self.inner = inner
        self.delay_steps = getattr(inner, "delay_steps", 0)
        self.motion_override = motion_override
        self.drop_motion = drop_motion
        self.records: List[list] = []
        self.seconds = 0.0
        self.epochs = 0
        self._hash = hashlib.sha256()
        self._walk = -1
        self._t = 0

    def reset(self) -> None:
        self._walk += 1
        self._t = 0
        self.records.append([])
        self.inner.reset()

    def update(self, scan, motion):
        if self.drop_motion:
            motion = None
        elif self.motion_override is not None:
            motion = self.motion_override[self._walk][self._t]
        self._t += 1
        self._hash.update(_scan_signature(scan))
        start = time.perf_counter()
        estimate = self.inner.update(scan, motion)
        self.seconds += time.perf_counter() - start
        self.epochs += 1
        self.records[-1].append(estimate)
        return estimate

    def flush(self):
        start = time.perf_counter()
        out = list(self.inner.flush()) if hasattr(self.inner, "flush") else []
        self.seconds += time.perf_counter() - start
        self.records[-1].extend(out)
        return out

    @property
    def scan_hash(self) -> str:
        return self._hash.hexdigest()[:16]


def _scan_signature(scan) -> bytes:
    if scan is None:
        return b"N;"
    parts = [f"{ap.bssid}:{'-' if r is None else f'{r.rssi_dbm!r},{r.rtt_ms!r}'}" for ap, r in scan.items()]
    return (";".join(parts) + "|").encode()


class DeadReckoningReference:
    """PDR-only dead reckoning from the true start (truth-aided reference, see module doc).
    Precomputes each walk's track from its readings; reset() advances to the next walk."""

    delay_steps = 0

    def __init__(self, walks, motions):
        self.tracks = [self._dead_reckon(w, m) for w, m in zip(walks, motions)]
        self._walk = -1
        self._t = 0

    @staticmethod
    def _dead_reckon(walk, readings):
        chords = [None] + [math.atan2(b[1] - a[1], b[0] - a[0]) if math.dist(a, b) > 1e-9 else None
                           for a, b in zip(walk, walk[1:])]
        first = next((t for t in range(1, len(walk)) if chords[t] is not None), None)
        heading = 0.0 if first is None else chords[first] - sum(r.heading_change_rad for r in readings[1:first + 1])
        x, y = walk[0]
        track = [(x, y)]
        for r in readings[1:]:
            heading += r.heading_change_rad
            x, y = x + r.distance_m * math.cos(heading), y + r.distance_m * math.sin(heading)
            track.append((x, y))
        return track

    def reset(self) -> None:
        self._walk += 1
        self._t = 0

    def update(self, scan, motion) -> LocalisationEstimate:
        position = self.tracks[self._walk][self._t]
        self._t += 1
        return LocalisationEstimate(position=position, covariance=None)


# ---------------------------------------------------------------- one (scenario, method) run

def _robust_localiser(model, mixture: bool = False) -> WallAwareLocaliser:
    return WallAwareLocaliser(model, measurement="hybrid", grid_spacing_m=GRID_SPACING_M, robust=True,
                              rtt_nlos_mixture=mixture)


_FUSION_FINGERPRINT = None


def _load_fusion() -> str:
    """Import fusion.py once per process and remember the sha256 of the source it was imported from,
    so a report can prove every particle-filter run used the same fusion.py (it may be edited while
    the benchmark runs)."""
    global _FUSION_FINGERPRINT
    if _FUSION_FINGERPRINT is None:
        path = Path(__file__).resolve().parent / "fusion.py"
        before = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        try:
            import fusion  # noqa: F401
        except Exception as error:  # broken mid-edit: PF runs will fail loudly later
            return f"import-failed({type(error).__name__})"
        after = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        _FUSION_FINGERPRINT = before if before == after else f"changed-during-import({before},{after})"
    return _FUSION_FINGERPRINT


def _particle_tracker(localiser, model, n_particles, imu_model, use_magnetometer=False):
    from fusion import ParticleTracker  # imported lazily (see _load_fusion): fusion.py may be absent
    return ParticleTracker(localiser, model, Random(PARTICLE_SEED), n_particles=n_particles, imu_model=imu_model,
                           use_magnetometer=use_magnetometer, scan_interval_s=SCAN_INTERVAL_S)


def build_method(method: str, scenario: BenchScenario, truth, model, walks, motions, n_walks, n_steps, n_particles):
    """(tracker, Instrumented kwargs) for one method name."""
    imu_model = IMU_MODELS[scenario.imu]()
    if method == SNAP:
        return SnapshotAsTracker(WallAwareLocaliser(model, measurement="hybrid", grid_spacing_m=GRID_SPACING_M)), {}
    if method == SNAP_MIX:
        return SnapshotAsTracker(WallAwareLocaliser(model, measurement="hybrid", grid_spacing_m=GRID_SPACING_M,
                                                    rtt_nlos_mixture=True)), {}
    if method == GRIDT:
        return GridTracker(_robust_localiser(model), model, scan_interval_s=SCAN_INTERVAL_S), {}
    if method == GRID_MIX:
        return GridTracker(_robust_localiser(model, mixture=True), model, scan_interval_s=SCAN_INTERVAL_S), {}
    for lag, name in SMOOTH.items():
        if method == name:
            return FixedLagSmoother(GridTracker(_robust_localiser(model), model, scan_interval_s=SCAN_INTERVAL_S), lag), {}
    if method in (PF, PF_CONS):
        return _particle_tracker(_robust_localiser(model), model, n_particles, imu_model), {}
    if method == PF_MIX:
        return _particle_tracker(_robust_localiser(model, mixture=True), model, n_particles, imu_model), {}
    if method == PF_CONS_MISSPEC:
        return _particle_tracker(_robust_localiser(model), model, n_particles, ImuModel()), {}
    if method == PF_NO_IMU:
        return _particle_tracker(_robust_localiser(model), model, n_particles, ImuModel()), {"drop_motion": True}
    if method == PF_PERFECT:
        perfect = perfect_imu_model()
        return (_particle_tracker(_robust_localiser(model), model, n_particles, perfect),
                {"motion_override": motions_for("perfect", n_walks, n_steps)})
    if method == PF_MAG_NAIVE:
        naive = replace(imu_model, magnetometer_bias_std_rad=0.0)
        return _particle_tracker(_robust_localiser(model), model, n_particles, naive, use_magnetometer=True), {}
    if method == PF_COMPASS:
        return _particle_tracker(_robust_localiser(model), model, n_particles, imu_model, use_magnetometer=True), {}
    if method == PF_MAG_HONEST:
        return _particle_tracker(_robust_localiser(model), model, n_particles, imu_model, use_magnetometer=True), {}
    if method == PDR:
        return DeadReckoningReference(walks, motions), {}
    raise ValueError(f"unknown method {method!r}")


def run_job(scenario_name: str, method: str, n_walks: int, n_steps: int, n_particles: int) -> dict:
    scenario = scenario_by_name(scenario_name)
    truth, model = networks_for(scenario)
    walks = deck_walks(n_walks, n_steps)
    motions = motions_for(scenario.imu, n_walks, n_steps)
    masks = outage_masks(walks) if scenario.outage else None
    tracker, kwargs = build_method(method, scenario, truth, model, walks, motions, n_walks, n_steps, n_particles)
    wrapped = Instrumented(tracker, **kwargs)
    harness_stats, _ = evaluate_tracking({method: wrapped}, truth, walks, Random(SCAN_SEED), motions=motions,
                                         scan_masks=masks)
    estimates = []
    for walk, raw in zip(walks, wrapped.records):
        aligned = (raw[wrapped.delay_steps:] + [None] * len(walk))[:len(walk)]
        estimates.append([None if e is None else
                          (e.position[0], e.position[1], None if e.covariance is None else np.asarray(e.covariance, float))
                          for e in aligned])
    h = harness_stats[method]
    return {
        "scenario": scenario_name, "method": method, "estimates": estimates,
        "ms_per_epoch": 1000.0 * wrapped.seconds / max(wrapped.epochs, 1),
        "scan_hash": wrapped.scan_hash,
        "fusion_sha": _FUSION_FINGERPRINT,
        "harness": {k: h[k] for k in ("mean_err", "median_err", "p90_err", "compartment_acc", "room_changes_ratio",
                                      "coverage95")},
    }


def _run_job_star(args):
    return run_job(*args)


# ---------------------------------------------------------------- metrics and statistics

def score(walks, estimates, environment, epochs: Optional[range] = None) -> dict:
    """Pooled and per-walk metrics, conventions as tracking.evaluate_tracking."""
    errors, per_walk_mean, per_walk_acc, per_walk_cov = [], [], [], []
    correct = total = est_changes = true_changes = 0
    nees_all = []
    covered = {level: 0 for level in CHI2_2DOF}
    with_cov = 0
    for walk, ests in zip(walks, estimates):
        true_rooms = [environment.compartment_at(p) for p in walk]
        if epochs is None:
            true_changes += sum(1 for a, b in zip(true_rooms, true_rooms[1:]) if a != b)
        idx = range(len(walk)) if epochs is None else [t for t in epochs if t < len(walk)]
        w_err, w_correct, w_total, w_cov, w_with = [], 0, 0, 0, 0
        last_room = None
        for t in idx:
            w_total += 1
            est = ests[t]
            if est is None:
                continue
            x, y, cov = est
            e = np.subtract(walk[t], (x, y))
            w_err.append(math.hypot(*e))
            room = environment.compartment_at((x, y))
            w_correct += room == true_rooms[t]
            if epochs is None and last_room is not None and room != last_room:
                est_changes += 1
            last_room = room
            if cov is not None:
                try:
                    nees = float(e @ np.linalg.solve(cov + 1e-9 * np.eye(2), e))
                except np.linalg.LinAlgError:
                    continue
                with_cov += 1
                w_with += 1
                nees_all.append(nees)
                for level, threshold in CHI2_2DOF.items():
                    covered[level] += nees <= threshold
                w_cov += nees <= CHI2_2DOF[0.95]
        errors.extend(w_err)
        correct += w_correct
        total += w_total
        per_walk_mean.append(float(np.mean(w_err)) if w_err else float("nan"))
        per_walk_acc.append(w_correct / w_total if w_total else float("nan"))
        per_walk_cov.append(w_cov / w_with if w_with else float("nan"))
    errors.sort()
    n = len(errors)

    def q(p):
        return errors[int(n * p)] if n else float("nan")

    return {
        "n": n, "mean": float(np.mean(errors)) if n else float("nan"), "median": q(0.5), "p75": q(0.75),
        "p90": q(0.9), "room_acc": correct / total if total else float("nan"),
        "ratio": est_changes / max(true_changes, 1) if epochs is None else float("nan"),
        "coverage": {lvl: covered[lvl] / with_cov if with_cov else float("nan") for lvl in CHI2_2DOF},
        "anees": float(np.mean(nees_all)) if nees_all else float("nan"),
        "n_cov": with_cov,
        "per_walk_mean": per_walk_mean, "per_walk_acc": per_walk_acc, "per_walk_cov": per_walk_cov,
        "errors": errors,
    }


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOTSTRAP):
    values = np.asarray(values, float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        return float("nan"), float("nan")
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(diff: np.ndarray) -> float:
    diff = np.asarray(diff, float)
    if len(diff) < 2 or not np.all(np.isfinite(diff)) or np.all(diff == 0):
        return float("nan")
    try:
        return float(sps.wilcoxon(diff, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    items = sorted(((p, k) for k, p in pvalues.items() if np.isfinite(p)))
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[k] = running
    return {k: adjusted.get(k, float("nan")) for k in pvalues}


def anees_bounds(n_samples: int, dof: int = 2):
    """95% bounds of the average NEES of n independent chi2(dof) samples (treats epochs as independent,
    so they are far too narrow for autocorrelated tracking errors; shown for reference only)."""
    if n_samples <= 0:
        return float("nan"), float("nan")
    return (sps.chi2.ppf(0.025, dof * n_samples) / n_samples, sps.chi2.ppf(0.975, dof * n_samples) / n_samples)


def analyse(results: Dict[str, Dict[str, dict]], walks, n_steps: int) -> dict:
    """{scenario: {"scores": {method: score}, "outage": {...}, "paired": {method: {...}}, "friedman": p}}."""
    rng = np.random.default_rng(Random(BOOTSTRAP_SEED).getrandbits(64))
    analysis = {}
    for scenario in BENCH_SCENARIOS:
        if scenario.name not in results:
            continue
        runs = results[scenario.name]
        truth, _ = networks_for(scenario)
        env = truth.environment
        hashes = {r["scan_hash"] for m, r in runs.items() if m != PDR}
        scores = {m: score(walks, r["estimates"], env) for m, r in runs.items()}
        outage = ({m: score(walks, r["estimates"], env, epochs=OUTAGE_EPOCHS) for m, r in runs.items()}
                  if scenario.outage and n_steps > OUTAGE_EPOCHS.stop else {})
        for m, r in runs.items():  # cross-check against the harness's own pooled stats
            h, s = r["harness"], scores[m]
            if np.isfinite(h["mean_err"]) and abs(h["mean_err"] - s["mean"]) > 1e-9:
                raise AssertionError(f"{scenario.name}/{m}: harness mean {h['mean_err']} != rescored {s['mean']}")
            if abs(h["compartment_acc"] - s["room_acc"]) > 1e-12:
                raise AssertionError(f"{scenario.name}/{m}: harness room acc differs")
        paired, raw_p = {}, {}
        reference = scores.get(GRIDT)
        for m, s in scores.items():
            if reference is None or m == GRIDT:
                continue
            diff = np.array(s["per_walk_mean"]) - np.array(reference["per_walk_mean"])
            lo, hi = bootstrap_mean_ci(diff, rng)
            raw_p[m] = wilcoxon_p(diff)
            wins = int(np.sum(diff < 0))
            paired[m] = {"diff": float(np.mean(diff)), "ci": (lo, hi), "p": raw_p[m], "wins": wins, "n": len(diff),
                         "ratio": s["mean"] / reference["mean"]}
        for m, p in holm(raw_p).items():
            paired[m]["p_holm"] = p
        if outage and GRIDT in outage:
            for m, s in outage.items():
                if m == GRIDT:
                    continue
                diff = np.array(s["per_walk_mean"]) - np.array(outage[GRIDT]["per_walk_mean"])
                s["paired_p"] = wilcoxon_p(diff)
                s["paired_ci"] = bootstrap_mean_ci(diff, rng)
        friedman = float("nan")
        table = [scores[m]["per_walk_mean"] for m in scores]
        if len(table) >= 3 and len(walks) >= 2 and np.all(np.isfinite(table)):
            friedman = float(sps.friedmanchisquare(*table).pvalue)
        cov_ci = {}
        for m, s in scores.items():
            vals = np.array(s["per_walk_cov"])
            if np.all(np.isfinite(vals)):
                cov_ci[m] = bootstrap_mean_ci(vals, rng)
        analysis[scenario.name] = {"scores": scores, "outage": outage, "paired": paired, "friedman": friedman,
                                   "hashes": hashes, "cov_ci": cov_ci,
                                   "ms": {m: r["ms_per_epoch"] for m, r in runs.items()}}
    return analysis


# ---------------------------------------------------------------- reporting

def _f(x, fmt="{:.3f}"):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else fmt.format(x)


def scenario_table_rows(name: str, a: dict) -> List[list]:
    rows = []
    for m, s in a["scores"].items():
        p = a["paired"].get(m)
        cov = s["coverage"][0.95]
        ci = a["cov_ci"].get(m)
        rows.append([
            m, _f(s["mean"]), _f(s["median"]), _f(s["p75"]), _f(s["p90"]), _f(s["room_acc"], "{:.2%}"),
            _f(s["ratio"], "{:.2f}"),
            _f(cov) + ("" if ci is None or not np.isfinite(ci[0]) else f" [{ci[0]:.2f}, {ci[1]:.2f}]"),
            _f(a["ms"][m], "{:.1f}"),
            "ref" if p is None else f"{p['ratio']:.2f}x",
            "" if p is None else f"{p['diff']:+.3f} [{_f(p['ci'][0], '{:+.3f}')}, {_f(p['ci'][1], '{:+.3f}')}]",
            "" if p is None else f"{_f(p['p'], '{:.4f}')} / {_f(p.get('p_holm'), '{:.4f}')} ({p['wins']}/{p['n']})",
        ])
    return rows


HEADER = ["Method", "Mean", "Median", "P75", "P90", "Room acc", "Room-chg ratio", "Cov95 [walk CI]", "ms/epoch",
          "Mean / Grid", "dMean vs Grid [95% CI]", "Wilcoxon p / Holm (walks better)"]


def print_report(analysis: dict, n_walks: int, n_steps: int, n_particles: int, state_note: str) -> List[str]:
    """Prints everything and returns the markdown lines."""
    md = [f"# Fusion benchmark (held-out seeds)", "",
          f"{n_walks} walks x {n_steps} s, walk i = Random({WALK_SEED_BASE}+i), scans Random({SCAN_SEED}), "
          f"IMU Random({IMU_SEED}), particles Random({PARTICLE_SEED}), {n_particles} particles, grid {GRID_SPACING_M} m.",
          f"fusion.py state: {state_note}", "",
          "Errors in metres. Paired statistics: per-walk mean error, method - Grid; bootstrap CI over walks; "
          "Wilcoxon signed-rank two-sided, Holm-adjusted within the scenario; '(k/n)' = walks where the method beat Grid.",
          ""]
    print(md[2])
    print(md[3])
    print()
    for scenario in BENCH_SCENARIOS:
        if scenario.name not in analysis:
            continue
        a = analysis[scenario.name]
        rows = scenario_table_rows(scenario.name, a)
        pairing = "identical scans for all methods" if len(a["hashes"]) == 1 else f"SCANS DIFFER ({a['hashes']})"
        print(f"##### {scenario.name}" + (f" ({scenario.note})" if scenario.note else ""))
        print(f"      pairing check: {pairing}; Friedman p = {_f(a['friedman'], '{:.2e}')}")
        widths = [max(len(str(r[i])) for r in rows + [HEADER]) + 2 for i in range(len(HEADER))]
        print("".join(h.ljust(w) for h, w in zip(HEADER, widths)))
        for r in rows:
            print("".join(str(c).ljust(w) for c, w in zip(r, widths)))
        md += [f"## {scenario.name}", ""]
        if scenario.note:
            md += [f"_{scenario.note}_", ""]
        md += [f"Pairing check: {pairing}. Friedman test across methods (per-walk means): p = {_f(a['friedman'], '{:.2e}')}.", ""]
        md += ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
        md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        md.append("")
        cal_rows = []
        for m, s in a["scores"].items():
            if s["n_cov"] == 0:
                continue
            lo, hi = anees_bounds(s["n_cov"])
            cal_rows.append([m] + [_f(s["coverage"][lvl]) for lvl in CHI2_2DOF] + [_f(s["anees"], "{:.2f}")])
        if cal_rows:
            cal_header = ["Method"] + [f"cov{int(lvl * 100)}" for lvl in CHI2_2DOF] + ["ANEES (2 = consistent)"]
            print("      calibration (empirical coverage at nominal 50/68/90/95%):")
            for r in cal_rows:
                print("        " + "  ".join(str(c) for c in r))
            md += ["Uncertainty calibration (fraction of true positions inside the nominal-level ellipse; "
                   "above nominal = conservative):", "",
                   "| " + " | ".join(cal_header) + " |", "|" + "---|" * len(cal_header)]
            md += ["| " + " | ".join(r) + " |" for r in cal_rows]
            md.append("")
        if a["outage"]:
            print("      outage window t in [100, 115) s only:")
            md += ["Outage window only (t in [100, 115) s; paired vs Grid over walks):", "",
                   "| Method | Mean | P90 | Room acc | dMean vs Grid [95% CI] | Wilcoxon p |", "|---|---|---|---|---|---|"]
            for m, s in a["outage"].items():
                ci = s.get("paired_ci", (float("nan"), float("nan")))
                line = [m, _f(s["mean"]), _f(s["p90"]), _f(s["room_acc"], "{:.2%}"),
                        "" if m == GRIDT else _f(float(np.mean(np.array(s['per_walk_mean'])
                                                               - np.array(a['outage'][GRIDT]['per_walk_mean']))), '{:+.3f}')
                        + f" [{_f(ci[0], '{:+.3f}')}, {_f(ci[1], '{:+.3f}')}]",
                        "" if m == GRIDT else _f(s.get("paired_p"), "{:.4f}")]
                print("        " + "  ".join(line))
                md.append("| " + " | ".join(line) + " |")
            md.append("")
        print()
    return md


def target_checks(analysis: dict, walks) -> List[str]:
    """T1-T7 and V2/V4/V5 of the spec on this held-out run."""
    lines = []

    def get(sc, m):
        return analysis.get(sc, {}).get("scores", {}).get(m)

    def pval(sc, m):
        p = analysis.get(sc, {}).get("paired", {}).get(m)
        return "" if p is None else f" (Wilcoxon p={_f(p['p'], '{:.4f}')}, {p['wins']}/{p['n']} walks better)"

    def verdict(ok):
        return "MET" if ok else "NOT MET"

    for tid, sc in (("T1", DECK), ("T2", MISCALIBRATED), ("T3", HARSH)):
        g, p = get(sc, GRIDT), get(sc, PF)
        if g and p:
            ok = p["mean"] <= 0.85 * g["mean"] and p["room_acc"] >= g["room_acc"]
            extra = ""
            if tid == "T1":
                ok = ok and p["ratio"] <= 1.05
                extra = f", ratio {p['ratio']:.2f} (<= 1.05)"
            lines.append(f"{tid} {verdict(ok)}: PF mean {p['mean']:.3f} vs 0.85 x Grid {0.85 * g['mean']:.3f} "
                         f"(ratio {p['mean'] / g['mean']:.2f}); room acc {p['room_acc']:.2%} vs Grid {g['room_acc']:.2%}"
                         f"{extra}{pval(sc, PF)}")
    out = analysis.get("Deck + 15 s outage", {}).get("outage", {})
    if out.get(PF) and out.get(GRIDT):
        p, g = out[PF], out[GRIDT]
        ok = p["mean"] <= 1.0 and p["room_acc"] >= 0.95 and p["mean"] < g["mean"]
        lines.append(f"T4 {verdict(ok)}: outage-window PF mean {p['mean']:.3f} m (<= 1.0), room acc {p['room_acc']:.2%} "
                     f"(>= 95%), Grid {g['mean']:.3f} m / {g['room_acc']:.2%} (paired p={_f(p.get('paired_p'), '{:.4f}')})")
    g, s = get(DECK, GRIDT), get(DECK, SMOOTH[5])
    if g and s:
        ok = s["mean"] <= 0.85 * g["mean"] and s["room_acc"] >= g["room_acc"] and s["ratio"] <= g["ratio"]
        lines.append(f"T5 {verdict(ok)}: smoother-5 mean {s['mean']:.3f} vs 0.85 x Grid {0.85 * g['mean']:.3f} "
                     f"(ratio {s['mean'] / g['mean']:.2f}); room acc {s['room_acc']:.2%} vs {g['room_acc']:.2%}; "
                     f"room-change ratio {s['ratio']:.2f} vs {g['ratio']:.2f}{pval(DECK, SMOOTH[5])}")
    mg, mn, mh = (get("Deck, magnetometer on", m) for m in (PF, PF_MAG_NAIVE, PF_MAG_HONEST))
    if mg and mn:
        lines.append(f"T6a {verdict(mn['mean'] > mg['mean'])}: naive biased-magnetometer PF {mn['mean']:.3f} > gyro PF "
                     f"{mg['mean']:.3f} on the same readings{pval('Deck, magnetometer on', PF_MAG_NAIVE)}"
                     + ("" if mh is None else f"; honest-bias magnetometer PF {mh['mean']:.3f}"))
    cg, cc = get("Deck, ideal compass", PF), get("Deck, ideal compass", PF_COMPASS)
    if cg and cc:
        lines.append(f"T6b {verdict(cg['mean'] <= 1.15 * cc['mean'])}: gyro PF {cg['mean']:.3f} vs 1.15 x ideal-compass PF "
                     f"{1.15 * cc['mean']:.3f} (ratio {cg['mean'] / cc['mean']:.2f})")
    if cg and cc and len(cg["per_walk_mean"]) > FRESH_FROM + 1:
        g = np.array(cg["per_walk_mean"][FRESH_FROM:])
        c = np.array(cc["per_walk_mean"][FRESH_FROM:])
        rng = np.random.default_rng(Random(BOOTSTRAP_SEED + 1).getrandbits(64))
        idx = rng.integers(0, len(g), size=(N_BOOTSTRAP, len(g)))
        ratios = g[idx].mean(axis=1) / c[idx].mean(axis=1)
        diff_ci = bootstrap_mean_ci(g - c, rng)
        ratio = g.mean() / c.mean()
        lines.append(f"T6b on the {len(g)} fresh walks only (i >= {FRESH_FROM}, seeds {WALK_SEED_BASE + FRESH_FROM}-"
                     f"{WALK_SEED_BASE + FRESH_FROM + len(g) - 1}) {verdict(ratio <= 1.15)}: gyro PF {g.mean():.3f} vs "
                     f"ideal-compass PF {c.mean():.3f}, ratio {ratio:.3f} [walk-bootstrap 95% CI "
                     f"{np.percentile(ratios, 2.5):.3f}, {np.percentile(ratios, 97.5):.3f}]; diff {np.mean(g - c):+.3f} m "
                     f"[{diff_ci[0]:+.3f}, {diff_ci[1]:+.3f}]; Wilcoxon p={_f(wilcoxon_p(g - c), '{:.4f}')} "
                     f"({int(np.sum(g < c))}/{len(g)} walks gyro better); CI upper bound <= 1.15: "
                     f"{np.percentile(ratios, 97.5) <= 1.15}")
    pf, pp = get(DECK, PF), get(DECK, PF_PERFECT)
    if pf and pp:
        lines.append(f"T6c {verdict(pf['mean'] <= 1.15 * pp['mean'])}: gyro PF {pf['mean']:.3f} vs 1.15 x perfect-IMU PF "
                     f"{1.15 * pp['mean']:.3f} (ratio {pf['mean'] / pp['mean']:.2f})")
    rt = "Deck, realistic 2.4 GHz RTT + 1.5 m AP offsets"
    a, b = get(rt, SNAP), get(rt, SNAP_MIX)
    if a and b:
        ok = b["p90"] <= 0.95 * a["p90"] and b["mean"] <= 1.02 * a["mean"]
        lines.append(f"T7 {verdict(ok)}: snapshot mixture P90 {b['p90']:.3f} vs 0.95 x plain {0.95 * a['p90']:.3f} "
                     f"(ratio {b['p90'] / a['p90']:.2f}); mean {b['mean']:.3f} vs 1.02 x plain {1.02 * a['mean']:.3f}")
    for name in (SMOOTH[5], PF):
        s = get(DECK, name)
        if s:
            cov = s["coverage"][0.95]
            ci = analysis[DECK]["cov_ci"].get(name, (float("nan"), float("nan")))
            lines.append(f"V4 {name}: coverage95 {cov:.3f} [{_f(ci[0])}, {_f(ci[1])} walk-bootstrap] -> "
                         f"{verdict(0.85 <= cov <= 0.99)} (target [0.85, 0.99])")
    g, n = get(DECK, GRIDT), get(DECK, PF_NO_IMU)
    if g and n:
        lines.append(f"V5 no-IMU {verdict(n['mean'] <= 1.2 * g['mean'])}: PF no IMU {n['mean']:.3f} vs 1.2 x Grid "
                     f"{1.2 * g['mean']:.3f}")
    for sc, a in analysis.items():
        if len(a["hashes"]) != 1:
            lines.append(f"V6 FAILED in {sc}: methods saw different scans {a['hashes']}")
    lines.append("V6 pairing: " + ("identical scan streams for every method in every scenario"
                                   if all(len(a["hashes"]) == 1 for a in analysis.values()) else "see failures above"))
    return lines


def pdr_error_at(results, walks, t: int) -> Optional[List[float]]:
    runs = results.get(DECK, {})
    if PDR not in runs:
        return None
    return [math.dist(est[t][:2], walk[t]) for walk, est in zip(walks, runs[PDR]["estimates"]) if t < len(walk)]


def save_csvs(run_dir: Path, analysis: dict) -> None:
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "method", "n_estimates", "mean_m", "median_m", "p75_m", "p90_m", "room_acc",
                    "room_change_ratio", "cov50", "cov68", "cov90", "cov95", "anees", "ms_per_epoch",
                    "mean_over_grid", "dmean_vs_grid_m", "dmean_ci_lo", "dmean_ci_hi", "wilcoxon_p", "holm_p",
                    "walks_better", "outage_mean_m", "outage_room_acc"])
        for sc, a in analysis.items():
            for m, s in a["scores"].items():
                p = a["paired"].get(m, {})
                o = a["outage"].get(m, {})
                w.writerow([sc, m, s["n"], s["mean"], s["median"], s["p75"], s["p90"], s["room_acc"], s["ratio"],
                            *(s["coverage"][lvl] for lvl in CHI2_2DOF), s["anees"], a["ms"][m],
                            p.get("ratio", ""), p.get("diff", ""), *(p.get("ci", ("", ""))), p.get("p", ""),
                            p.get("p_holm", ""), p.get("wins", ""), o.get("mean", ""), o.get("room_acc", "")])
    with (run_dir / "per_walk.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "method", "walk", "walk_seed", "mean_m", "room_acc", "cov95"])
        for sc, a in analysis.items():
            for m, s in a["scores"].items():
                for i, (e, acc, c) in enumerate(zip(s["per_walk_mean"], s["per_walk_acc"], s["per_walk_cov"])):
                    w.writerow([sc, m, i, WALK_SEED_BASE + i, e, acc, c])


# ---------------------------------------------------------------- plots

def plot_error_cdfs(analysis: dict):
    names = [s for s in BENCH_SCENARIOS if s.name in analysis]
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.6 * rows), sharex=True, sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for ax, scenario in zip(axes.ravel(), names):
        scores = analysis[scenario.name]["scores"]
        for colour, m in zip(SERIES_COLOURS, [m for m in scenario.cdf_methods if m in scores][:4]):
            errors = np.array(scores[m]["errors"])
            if len(errors):
                ax.plot(errors, np.arange(1, len(errors) + 1) / len(errors), color=colour, linewidth=2, label=m)
        ax.set_title(scenario.name, loc="left", fontsize=9)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.legend(loc="lower right", fontsize=7, frameon=False)
        _style_axis(ax)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Position error (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of epochs")
    fig.suptitle("Error CDFs on held-out walks (higher and further left is better)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_room_accuracy(analysis: dict):
    names = [s.name for s in BENCH_SCENARIOS if s.name in analysis]
    cols = 2
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.0 * cols, 2.9 * rows), facecolor=SURFACE, squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        scores = {m: s for m, s in analysis[name]["scores"].items() if m != PDR}
        methods = list(scores)
        acc = np.array([100 * scores[m]["room_acc"] for m in methods])
        colours = [SERIES_COLOURS[1] if m == GRIDT else SERIES_COLOURS[0] for m in methods]
        ax.barh(range(len(methods)), acc, color=colours)
        ax.set_yticks(range(len(methods)), methods, fontsize=7)
        ax.invert_yaxis()
        lo = max(0.0, math.floor(acc.min()) - 1.0)
        ax.set_xlim(lo, 100.0)
        for i, v in enumerate(acc):
            ax.text(v, i, f" {v:.2f}%", va="center", fontsize=7, color=INK_SECONDARY)
        ax.set_title(name, loc="left", fontsize=9)
        ax.set_xlabel("Room accuracy (%)")
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        _style_axis(ax)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    fig.legend(handles=[Patch(color=SERIES_COLOURS[1], label="Grid (reference)"),
                        Patch(color=SERIES_COLOURS[0], label="Other methods")],
               loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Room (compartment) accuracy per method; PDR-only reference omitted", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_example_walk(results: dict, walks):
    from simulation import _draw_layout
    runs = results[DECK]
    truth, _ = networks_for(scenario_by_name(DECK))
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=SURFACE)
    _draw_layout(ax, truth, fill_compartments=False)
    window = slice(0, 150)
    walk_xy = np.array(walks[0][window])
    ax.plot(walk_xy[:, 0], walk_xy[:, 1], color=INK, linewidth=2, label="True path")
    for colour, m in zip(SERIES_COLOURS, (GRIDT, PF, SMOOTH[5])):
        if m not in runs:
            continue
        pts = np.array([e[:2] for e in runs[m]["estimates"][0][window] if e is not None])
        ax.plot(pts[:, 0], pts[:, 1], color=colour, linewidth=1.6, label=m, zorder=5)
    ax.set_title(f"{DECK}: first {window.stop} s of held-out walk 0 (seed {WALK_SEED_BASE})", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
    ax.grid(True, color=GRID, linewidth=0.6)
    _style_axis(ax)
    fig.tight_layout()
    return fig


def plot_reliability(analysis: dict):
    a = analysis[DECK]
    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor=SURFACE)
    levels = np.array(sorted(CHI2_2DOF))
    ax.plot([0.4, 1.0], [0.4, 1.0], color=INK_SECONDARY, linewidth=1, linestyle="--", label="Perfect calibration")
    for colour, m in zip(SERIES_COLOURS, (GRIDT, SMOOTH[5], PF)):
        s = a["scores"].get(m)
        if s is None or s["n_cov"] == 0:
            continue
        ax.plot(levels, [s["coverage"][lvl] for lvl in levels], marker="o", color=colour, linewidth=2, label=m)
    ax.axhspan(0.85, 0.99, xmin=0.0, xmax=1.0, color=GRID, alpha=0.5, label="V4 band for 95% (0.85-0.99)")
    ax.set_xlabel("Nominal ellipse level")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.01)
    ax.set_title(f"{DECK}: uncertainty calibration", loc="left", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(True, color=GRID, linewidth=0.6)
    _style_axis(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- main

def fusion_state_note() -> str:
    path = Path(__file__).resolve().parent / "fusion.py"
    if not path.exists():
        return "fusion.py missing"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
    return f"fusion.py sha256 {digest}, modified {mtime}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--walks", type=int, default=N_WALKS)
    parser.add_argument("--steps", type=int, default=WALK_STEPS)
    parser.add_argument("--particles", type=int, default=N_PARTICLES)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))  # memory: ~0.5 GB per worker
    parser.add_argument("--scenarios", nargs="*", default=None, help="subset of scenario names")
    args = parser.parse_args()
    if args.walks > MAX_WALKS:
        parser.error(f"the spec defines held-out walk seeds 2024..{2024 + MAX_WALKS - 1} only")

    from simulation import RESULTS_DIR, save_figures, start_saving_results
    run_dir = start_saving_results(RESULTS_DIR / "fusion")
    state = fusion_state_note()
    selected = [s for s in BENCH_SCENARIOS if args.scenarios is None or s.name in args.scenarios]
    jobs = [(s.name, m, args.walks, args.steps, args.particles) for s in selected for m in s.methods]
    print(f"{len(jobs)} runs ({len(selected)} scenarios) on {args.jobs} worker processes; {state}\n")

    started = time.perf_counter()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_load_fusion) as pool:
            outputs = list(pool.map(_run_job_star, jobs))
    else:
        _load_fusion()
        outputs = [run_job(*job) for job in jobs]
    fusion_versions = {o["fusion_sha"] for o in outputs if o["fusion_sha"] is not None}
    if len(fusion_versions) > 1:
        print(f"WARNING: fusion.py changed while the benchmark ran; PF runs used versions {fusion_versions}. "
              "Re-run for a consistent result.\n")
    state = f"{state}; PF runs imported fusion.py version(s) {sorted(fusion_versions)}"
    print(f"All runs finished in {time.perf_counter() - started:.0f} s "
          f"(ms/epoch measured inside worker processes running concurrently)\n")

    results: Dict[str, Dict[str, dict]] = {}
    for out in outputs:
        results.setdefault(out["scenario"], {})[out["method"]] = out
    walks = deck_walks(args.walks, args.steps)
    analysis = analyse(results, walks, args.steps)
    md = print_report(analysis, args.walks, args.steps, args.particles, state)

    print("=== Targets and validation checks (held-out benchmark size) ===")
    checks = target_checks(analysis, walks)
    pdr120 = pdr_error_at(results, walks, 120) if args.steps > 120 else None
    if pdr120:
        checks.append(f"V2 {'MET' if np.mean(pdr120) > 2.0 else 'NOT MET'}: PDR-only error at 120 s, mean over walks "
                      f"{np.mean(pdr120):.2f} m (> 2 m), per walk {', '.join(f'{e:.1f}' for e in pdr120)}")
    for line in checks:
        print(line)
    md += ["## Targets and validation checks", ""] + [f"- {line}" for line in checks] + [""]
    (run_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    save_csvs(run_dir, analysis)

    figures = {"error_cdfs": plot_error_cdfs(analysis), "room_accuracy": plot_room_accuracy(analysis)}
    if DECK in results:
        figures["example_walk"] = plot_example_walk(results, walks)
        figures["calibration_deck"] = plot_reliability(analysis)
    save_figures(run_dir, figures)
    print(f"\nSaved results.md, summary.csv, per_walk.csv and {len(figures)} figures to {run_dir}")


if __name__ == "__main__":
    main()
