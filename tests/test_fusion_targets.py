"""Targets T1-T7 and validation checks V4, V8 of docs/sensor_fusion_spec.md, on HELD-OUT seeds only.

Seeds (spec, "Seeds: tuning vs held-out validation"): walk i uses Random seeded 2024 + i, scans
one stream seeded 3024, IMU one stream seeded 4024 (re-seeded for each IMU model so every motion
set is drawn the same way), particles seeded 5024. V8 below checks that this file uses nothing else.

Sizing (to stay well under 3 minutes on a laptop): N_WALKS = 6 walks of WALK_STEPS = 240 s
(the spec's baseline table used 8 x 300 s), grid 0.2 m, N_PARTICLES = 1500. Baselines are not
hard-coded: GridTracker is run in-test on the same walks, scans and IMU readings (paired).

Configurations:
- "Grid": GridTracker(WallAwareLocaliser(model, "hybrid", 0.2 m, robust=True), model).
- "PF": ParticleTracker on the same localiser and model network, realistic ImuModel() defaults,
  gyro only. The localiser is the robust one for both trackers so the comparison is paired on
  the likelihood as well.
- Walks are generated on the deck truth network (all deck scenarios share the geometry, so the
  walks are identical for every scenario); scans come from each scenario's truth network.
- Expensive runs are cached at module level (functools.lru_cache), so each runs once
  (the GridTracker forward run is repeated once, in smoother_run, so T5 does not need the PF).

If a target is not met after honest effort, do not weaken it: mark the test
@unittest.expectedFailure with the measured numbers in a comment (spec).
"""
import functools
import math
import re
import unittest
from dataclasses import fields, replace
from pathlib import Path
from random import Random

import numpy as np

from fusion import ParticleTracker
from imu import ImuModel, simulate_imu
from positioning import WallAwareLocaliser
from scenarios import REALISTIC_2G4_RTT, SCENARIOS, build_networks
from tracking import FixedLagSmoother, GridTracker, SnapshotAsTracker, evaluate_tracking, simulate_walk

N_WALKS = 6
WALK_STEPS = 240
GRID_SPACING_M = 0.2
N_PARTICLES = 1500
SMOOTHER_LAG = 5
OUTAGE_EPOCHS = range(100, 115)  # 15 s Wi-Fi outage starting at t = 100 s

WALK_SEEDS = tuple(2024 + i for i in range(N_WALKS))
SCAN_SEED = 3024
IMU_SEED = 4024
PARTICLE_SEED = 5024
TUNING_SEEDS = (67890, 78901, 89012)

DECK, MISCALIBRATED, HARSH = "Accommodation deck", "Deck, miscalibrated", "Deck, harsh channel"

_ERROR_FIELD_MARKERS = ("noise", "bias", "probability", "per_s", "std", "error")


def perfect_imu_model() -> ImuModel:
    """ImuModel with every error source off (same rule as tests/test_imu.py)."""
    base = ImuModel()
    changes = {}
    for f in fields(ImuModel):
        value = getattr(base, f.name)
        if f.name == "magnetometer_available":
            changes[f.name] = False
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        elif any(marker in f.name for marker in _ERROR_FIELD_MARKERS):
            changes[f.name] = type(value)(0)
    return replace(base, **changes)


def magnetometer_imu_model() -> ImuModel:
    """Realistic defaults with the (steel-hull biased) magnetometer switched on."""
    return replace(ImuModel(), magnetometer_available=True)


def ideal_compass_imu_model() -> ImuModel:
    """Realistic defaults with an *unbiased* magnetometer: a perfect absolute-heading reference
    (white noise only), i.e. what a compass would give outside a steel hull."""
    return replace(ImuModel(), magnetometer_available=True, magnetometer_bias_std_rad=0.0)


IMU_MODELS = {"realistic": ImuModel, "magnetometer": magnetometer_imu_model, "perfect": perfect_imu_model,
              "ideal_compass": ideal_compass_imu_model}


@functools.lru_cache(maxsize=None)
def networks(name: str):
    """(truth, model, survey) for a deck scenario, as in scenarios.py."""
    return build_networks(next(s for s in SCENARIOS if s.name == name))


@functools.lru_cache(maxsize=None)
def walks():
    truth = networks(DECK)[0]
    return [simulate_walk(truth, Random(seed), n_steps=WALK_STEPS) for seed in WALK_SEEDS]


@functools.lru_cache(maxsize=None)
def motions(kind: str):
    model = IMU_MODELS[kind]()
    rng = Random(IMU_SEED)
    return [simulate_imu(walk, rng, model, scan_interval_s=1.0) for walk in walks()]


@functools.lru_cache(maxsize=None)
def localiser(name: str) -> WallAwareLocaliser:
    return WallAwareLocaliser(networks(name)[1], measurement="hybrid", grid_spacing_m=GRID_SPACING_M, robust=True)


def grid_tracker(name: str) -> GridTracker:
    return GridTracker(localiser(name), networks(name)[1], scan_interval_s=1.0)


def particle_tracker(name: str, imu_model: ImuModel, use_magnetometer: bool = False) -> ParticleTracker:
    return ParticleTracker(localiser(name), networks(name)[1], Random(PARTICLE_SEED), n_particles=N_PARTICLES,
                           imu_model=imu_model, use_magnetometer=use_magnetometer, scan_interval_s=1.0)


@functools.lru_cache(maxsize=None)
def scenario_run(name: str):
    """Grid vs PF on one deck scenario. Returns (stats, estimates) of evaluate_tracking."""
    trackers = {"Grid": grid_tracker(name), "PF": particle_tracker(name, ImuModel())}
    return evaluate_tracking(trackers, networks(name)[0], walks(), Random(SCAN_SEED), motions=motions("realistic"))


@functools.lru_cache(maxsize=None)
def smoother_run():
    """Grid forward filter vs FixedLagSmoother(Grid, 5) on the deck (same scans and IMU readings
    as scenario_run(DECK); kept separate so T5 does not depend on the particle filter)."""
    trackers = {"Grid": grid_tracker(DECK), "Smoother": FixedLagSmoother(grid_tracker(DECK), SMOOTHER_LAG)}
    return evaluate_tracking(trackers, networks(DECK)[0], walks(), Random(SCAN_SEED), motions=motions("realistic"))


@functools.lru_cache(maxsize=None)
def outage_run():
    masks = [[t not in OUTAGE_EPOCHS for t in range(len(walk))] for walk in walks()]
    trackers = {"Grid": grid_tracker(DECK), "PF": particle_tracker(DECK, ImuModel())}
    return evaluate_tracking(trackers, networks(DECK)[0], walks(), Random(SCAN_SEED),
                             motions=motions("realistic"), scan_masks=masks)


@functools.lru_cache(maxsize=None)
def magnetometer_run():
    """Paired on identical readings (magnetometer on): gyro-only PF vs a PF that uses the
    magnetometer as an absolute heading. The magnetometer PF believes the open-space model
    (magnetometer_bias_std_rad = 0), i.e. it trusts the compass as PDR systems do on land; the
    readings carry the steel-hull bias of the realistic defaults."""
    model = magnetometer_imu_model()
    trackers = {
        "PF gyro": particle_tracker(DECK, model, use_magnetometer=False),
        "PF magnetometer": particle_tracker(DECK, replace(model, magnetometer_bias_std_rad=0.0), use_magnetometer=True),
    }
    return evaluate_tracking(trackers, networks(DECK)[0], walks(), Random(SCAN_SEED), motions=motions("magnetometer"))


@functools.lru_cache(maxsize=None)
def ideal_compass_run():
    """Paired on identical readings from a realistic phone whose magnetometer is unbiased: gyro-only
    PF (ignores the compass, must learn absolute heading from Wi-Fi) vs a PF using the compass."""
    model = ideal_compass_imu_model()
    trackers = {
        "PF gyro": particle_tracker(DECK, model, use_magnetometer=False),
        "PF ideal compass": particle_tracker(DECK, model, use_magnetometer=True),
    }
    return evaluate_tracking(trackers, networks(DECK)[0], walks(), Random(SCAN_SEED), motions=motions("ideal_compass"))


@functools.lru_cache(maxsize=None)
def perfect_imu_run():
    model = perfect_imu_model()
    trackers = {"PF perfect IMU": particle_tracker(DECK, model)}
    return evaluate_tracking(trackers, networks(DECK)[0], walks(), Random(SCAN_SEED), motions=motions("perfect"))


@functools.lru_cache(maxsize=None)
def realistic_rtt_run():
    """Hybrid snapshot (Gaussian, as in scenarios.py) with and without the RTT NLOS mixture.
    Truth: the deck with realistic 2.4 GHz RTT (scenarios.REALISTIC_2G4_RTT) plus 1.5 m per-AP
    offsets; the localisers keep the nominal deck model."""
    truth = replace(networks(DECK)[0], **REALISTIC_2G4_RTT, rtt_ap_offset_std_m=1.5)
    model = networks(DECK)[1]
    trackers = {
        "Hybrid": SnapshotAsTracker(WallAwareLocaliser(model, measurement="hybrid", grid_spacing_m=GRID_SPACING_M)),
        "Hybrid + NLOS mixture": SnapshotAsTracker(WallAwareLocaliser(model, measurement="hybrid", grid_spacing_m=GRID_SPACING_M,
                                                                      rtt_nlos_mixture=True)),
    }
    return evaluate_tracking(trackers, truth, walks(), Random(SCAN_SEED))


def fmt(stats: dict) -> str:
    return "; ".join(f"{name}: mean {s['mean_err']:.3f} m, P90 {s['p90_err']:.3f} m, room acc {s['compartment_acc']:.2%}, "
                     f"room-change ratio {s['room_changes_ratio']:.3f}, coverage95 {s['coverage95']:.3f}"
                     for name, s in stats.items())


def outage_metrics(estimates: dict, name: str):
    """(mean error, room accuracy) over the outage epochs only; a missing estimate counts as wrong room."""
    environment = networks(DECK)[0].environment
    errors, correct, total = [], 0, 0
    for walk, walk_estimates in zip(walks(), estimates[name]):
        for t in OUTAGE_EPOCHS:
            total += 1
            estimate = walk_estimates[t]
            if estimate is None:
                continue
            errors.append(math.dist(estimate, walk[t]))
            correct += environment.compartment_at(estimate) == environment.compartment_at(walk[t])
    return (float(np.mean(errors)) if errors else float("nan")), correct / total


class FusionTargetTests(unittest.TestCase):
    """T1-T7 (spec "Targets"). Relative targets are paired: identical scans and IMU readings."""

    def assert_relative_target(self, stats: dict, label: str, check_ratio: bool) -> None:
        grid, pf = stats["Grid"], stats["PF"]
        msg = f"{label}: {fmt(stats)}"
        self.assertLessEqual(pf["mean_err"], 0.85 * grid["mean_err"], msg)
        self.assertGreaterEqual(pf["compartment_acc"], grid["compartment_acc"], msg)
        if check_ratio:
            self.assertLessEqual(pf["room_changes_ratio"], 1.05, msg)

    def test_t1_fusion_beats_wifi_only_tracking_on_the_deck(self) -> None:
        """T1 expected: on the accommodation deck, PF mean error <= 0.85 x Grid, PF room accuracy
        >= Grid, PF room-change ratio <= 1.05."""
        stats, _ = scenario_run(DECK)
        self.assert_relative_target(stats, "T1 deck", check_ratio=True)

    def test_t2_holds_under_model_miscalibration(self) -> None:
        """T2 expected: on "Deck, miscalibrated", PF mean <= 0.85 x Grid and room accuracy >= Grid."""
        stats, _ = scenario_run(MISCALIBRATED)
        self.assert_relative_target(stats, "T2 miscalibrated", check_ratio=False)

    def test_t3_holds_in_the_harsh_channel(self) -> None:
        """T3 expected: on "Deck, harsh channel", PF mean <= 0.85 x Grid and room accuracy >= Grid."""
        stats, _ = scenario_run(HARSH)
        self.assert_relative_target(stats, "T3 harsh", check_ratio=False)

    def test_t4_pdr_bridges_a_15_s_wifi_outage(self) -> None:
        """T4 expected: with no scans for t in [100, 115) s on the deck, over those outage epochs
        only: PF mean error <= 1.0 m, PF room accuracy >= 95%, and PF mean error < Grid's."""
        stats, estimates = outage_run()
        pf_err, pf_acc = outage_metrics(estimates, "PF")
        grid_err, grid_acc = outage_metrics(estimates, "Grid")
        msg = (f"T4 outage epochs: PF mean {pf_err:.3f} m, room acc {pf_acc:.2%}; Grid mean {grid_err:.3f} m, "
               f"room acc {grid_acc:.2%}. Whole walks: {fmt(stats)}")
        self.assertLessEqual(pf_err, 1.0, msg)
        self.assertGreaterEqual(pf_acc, 0.95, msg)
        self.assertLess(pf_err, grid_err, msg)

    def test_t5_fixed_lag_smoothing_helps(self) -> None:
        """T5 expected: FixedLagSmoother(Grid, 5) on the deck: mean <= 0.85 x Grid forward,
        room accuracy >= forward, room-change ratio <= forward."""
        stats, _ = smoother_run()
        grid, smoother = stats["Grid"], stats["Smoother"]
        msg = f"T5 deck: {fmt(stats)}"
        self.assertLessEqual(smoother["mean_err"], 0.85 * grid["mean_err"], msg)
        self.assertGreaterEqual(smoother["compartment_acc"], grid["compartment_acc"], msg)
        self.assertLessEqual(smoother["room_changes_ratio"], grid["room_changes_ratio"], msg)

    def test_t6_magnetometer_is_unsafe_in_a_steel_hull(self) -> None:
        """T6a expected: on identical readings, the PF using the steel-biased magnetometer (believing
        it unbiased) has a larger mean error than the gyro-only PF."""
        stats, _ = magnetometer_run()
        self.assertGreater(stats["PF magnetometer"]["mean_err"], stats["PF gyro"]["mean_err"], f"T6a: {fmt(stats)}")

    def test_t6_gyro_only_heading_is_as_good_as_an_ideal_compass(self) -> None:
        """T6b expected: the gyro-only PF, which must learn absolute heading from Wi-Fi, has mean error
        within 15% of the same PF given an ideal (unbiased) compass, on identical readings. This is
        the well-posed form of the spec's "within 15% of a perfect IMU" clause (see T6c)."""
        stats, _ = ideal_compass_run()
        self.assertLessEqual(stats["PF gyro"]["mean_err"], 1.15 * stats["PF ideal compass"]["mean_err"],
                             f"T6b: {fmt(stats)}")

    @unittest.expectedFailure
    def test_t6_gyro_only_within_15_percent_of_a_perfect_imu(self) -> None:
        """T6c (spec wording, kept as written): gyro-only PF within 15% of a PF given a perfect IMU.
        KNOWN SHORTFALL, recorded rather than weakened: the clause conflates two questions. A
        perfect IMU also removes step-length bias/noise, missed/false steps and per-step heading
        noise, not just the lack of absolute heading. Measured on held-out seeds (2026-10-17):
        PF gyro 0.310 m vs PF perfect IMU 0.173 m (ratio 1.79): realistic phone PDR errors cost
        ~0.14 m of mean error. T6b tests the heading question the clause was meant to capture."""
        mag_stats, _ = magnetometer_run()
        perfect_stats, _ = perfect_imu_run()
        stats = {**mag_stats, **perfect_stats}
        self.assertLessEqual(stats["PF gyro"]["mean_err"], 1.15 * stats["PF perfect IMU"]["mean_err"], f"T6c: {fmt(stats)}")

    def test_t7_rtt_mixture_helps_with_realistic_rtt(self) -> None:
        """T7 expected: with rtt_ap_offset_std_m = 1.5 and realistic 2.4 GHz RTT errors on the deck,
        the hybrid snapshot with rtt_nlos_mixture=True has P90 <= 0.95 x without, and mean error
        <= 1.02 x without."""
        stats, _ = realistic_rtt_run()
        plain, mixture = stats["Hybrid"], stats["Hybrid + NLOS mixture"]
        msg = f"T7: {fmt(stats)}"
        self.assertLessEqual(mixture["p90_err"], 0.95 * plain["p90_err"], msg)
        self.assertLessEqual(mixture["mean_err"], 1.02 * plain["mean_err"], msg)


class FusionValidationTests(unittest.TestCase):
    def test_v4_uncertainty_is_calibrated(self) -> None:
        """V4 expected: on the deck, coverage95 (fraction of true positions inside the estimate's
        95% ellipse) of the PF and of the lag-5 smoother are both within [0.85, 0.99]."""
        # smoother first, so its result is reported even while the PF is missing
        for run, name in ((smoother_run, "Smoother"), (lambda: scenario_run(DECK), "PF")):
            stats, _ = run()
            coverage = stats[name]["coverage95"]
            self.assertTrue(0.85 <= coverage <= 0.99, f"V4 {name} coverage95 {coverage:.3f}. {fmt(stats)}")

    @unittest.expectedFailure
    def test_v4b_uncertainty_is_calibrated_under_model_miscalibration(self) -> None:
        """V4b (added after review, 2026-10-17) expected: PF coverage95 within [0.85, 0.99] on
        "Deck, miscalibrated" too. KNOWN SHORTFALL, recorded rather than fixed: measured 0.76
        held-out (harsh channel 0.87). The overconfidence is model error (wrong wall losses and
        path-loss exponent) that the particle spread cannot see; a covariance floor large enough
        to fix it would push the matched deck above 0.99, i.e. would game V4 rather than fix it.
        Remedies are self-calibration of the propagation model, not covariance inflation."""
        stats, _ = scenario_run(MISCALIBRATED)
        coverage = stats["PF"]["coverage95"]
        self.assertTrue(0.85 <= coverage <= 0.99, f"V4b PF coverage95 {coverage:.3f}. {fmt(stats)}")

    def test_v8_only_held_out_seeds_are_used(self) -> None:
        """V8 expected: the seeds are the spec's held-out ones, disjoint from the tuning seeds, and
        every Random in this file is seeded from those constants only (no numpy global RNG)."""
        self.assertEqual(WALK_SEEDS, tuple(2024 + i for i in range(N_WALKS)))
        self.assertLessEqual(N_WALKS, 8)
        self.assertEqual((SCAN_SEED, IMU_SEED, PARTICLE_SEED), (3024, 4024, 5024))
        self.assertTrue(set(WALK_SEEDS + (SCAN_SEED, IMU_SEED, PARTICLE_SEED)).isdisjoint(TUNING_SEEDS))
        source = Path(__file__).read_text(encoding="utf-8")
        arguments = set(re.findall(r"\bRandom\(([^()]*)\)", source))
        self.assertTrue(arguments)
        self.assertLessEqual(arguments, {"seed", "SCAN_SEED", "IMU_SEED", "PARTICLE_SEED"}, arguments)
        self.assertNotIn("np." + "random", source)
        self.assertNotIn("import " + "random\n", source)


if __name__ == "__main__":
    unittest.main()
