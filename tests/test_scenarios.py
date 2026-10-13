"""Tests for scenarios.py, simulation.py trial/evaluation helpers, and reproducibility."""
import math
import random
import unittest
from dataclasses import fields, replace
from random import Random
from unittest import mock

import numpy as np

from network import Network, Receiver
from positioning import LocalisationEstimate, NearestAPLocaliser, WallAwareLocaliser
from scenarios import SCENARIOS, LAYOUTS_DIR, build_localisers, build_networks, compartment_breakdown
from simulation import evaluate_localisers, sample_trials
from tracking import GridTracker, evaluate_tracking, simulate_walk

NOISE_FREE = {
    "shadowing_std_db": 0.0,
    "fast_fading_samples": 0,
    "measurement_noise_std_db": 0.0,
    "rssi_quantization_db": 0.0,
    "noise_floor_variation_std_db": 0.0,
    "interference_burst_start_probability": 0.0,
    "interference_burst_end_probability": 1.0,
    "detection_snr_midpoint_db": -100.0,
    "rtt_ranging_noise_m": 0.0,
    "rtt_nlos_noise_per_wall_m": 0.0,
    "rtt_weak_signal_penalty_m": 0.0,
    "rtt_outlier_probability": 0.0,
    "rtt_nlos_outlier_probability": 0.0,
}

PUBLIC_FIELDS = [f.name for f in fields(Network) if f.init and f.name not in ("environment", "access_points", "receivers")]


def scenario(name: str):
    return next(s for s in SCENARIOS if s.name == name)


class BuildNetworksTests(unittest.TestCase):
    def test_every_scenario_builds(self) -> None:
        names = [s.name for s in SCENARIOS]
        self.assertEqual(len(names), len(set(names)))
        for s in SCENARIOS:
            truth, model, survey = build_networks(s)
            for net in (truth, model, survey):
                self.assertIsInstance(net, Network)
                self.assertEqual([ap.name for ap in net.access_points], [ap.name for ap in s.access_points])
                self.assertGreater(len(net.environment.compartments), 0)

    def test_truth_matches_layout_plus_channel_overrides_only(self) -> None:
        for s in SCENARIOS:
            base = Network.from_config(LAYOUTS_DIR / s.layout, access_points=list(s.access_points))
            truth, model, survey = build_networks(s)
            for name in PUBLIC_FIELDS:
                expected = s.channel_overrides.get(name, getattr(base, name))
                self.assertEqual(getattr(truth, name), expected, (s.name, name))
            self.assertEqual(
                [(w.name, w.p1, w.p2, w.attenuation_db) for w in truth.environment.walls],
                [(w.name, w.p1, w.p2, w.attenuation_db) for w in base.environment.walls],
                s.name,
            )

    def test_model_is_layout_plus_model_overrides_and_wall_scale_only(self) -> None:
        # The localisers' belief starts from the nominal layout preset: channel_overrides
        # (e.g. the harsh channel) must not leak into it.
        for s in SCENARIOS:
            base = Network.from_config(LAYOUTS_DIR / s.layout, access_points=list(s.access_points))
            truth, model, _ = build_networks(s)
            for name in PUBLIC_FIELDS:
                expected = s.model_overrides.get(name, getattr(base, name))
                self.assertEqual(getattr(model, name), expected, (s.name, name))
            self.assertEqual(len(model.environment.walls), len(truth.environment.walls))
            for believed, real in zip(model.environment.walls, truth.environment.walls):
                self.assertEqual((believed.name, believed.p1, believed.p2), (real.name, real.p1, real.p2))
                self.assertAlmostEqual(believed.attenuation_db, real.attenuation_db * s.model_wall_scale)
            self.assertEqual(
                [c.name for c in model.environment.compartments],
                [c.name for c in truth.environment.compartments],
            )

    def test_miscalibrated_model_does_not_leak_into_truth(self) -> None:
        truth, model, _ = build_networks(scenario("Miscalibrated"))
        self.assertNotEqual(model.path_loss_exponent, truth.path_loss_exponent)
        self.assertIsNot(model.environment, truth.environment)
        real_losses = [w.attenuation_db for w in truth.environment.walls]
        believed_losses = [w.attenuation_db for w in model.environment.walls]
        self.assertTrue(all(b < r for b, r in zip(believed_losses, real_losses) if r > 0))
        # same probe: truth and model disagree on mean RSSI through a wall
        ap = truth.access_points[0]
        receiver = Receiver("p", (9.0, 6.5, 1.0))
        self.assertNotAlmostEqual(truth.mean_rssi(ap, receiver), model.mean_rssi(ap, receiver))

    def test_survey_has_different_shadowing_seed_and_independent_caches(self) -> None:
        truth, model, survey = build_networks(scenario("Baseline"))
        self.assertEqual(survey.shadowing_seed, truth.shadowing_seed + 1)
        self.assertEqual(model.shadowing_seed, truth.shadowing_seed)
        for a, b in ((truth, model), (truth, survey), (model, survey)):
            self.assertIsNot(a._shadow_value_cache, b._shadow_value_cache)
            self.assertIsNot(a._shadow_fields, b._shadow_fields)
            self.assertIsNot(a._interference_burst_active, b._interference_burst_active)

        ap = truth.access_points[0]
        receiver = Receiver("p", (2.0, 2.0, 1.0))
        truth_value = truth.spatial_shadowing(ap, receiver)
        survey_value = survey.spatial_shadowing(ap, receiver)
        self.assertNotAlmostEqual(truth_value, survey_value)
        self.assertEqual(len(model._shadow_value_cache), 0)  # untouched by queries on truth/survey
        # a fresh truth built later reproduces the same field value
        fresh_truth, _, _ = build_networks(scenario("Baseline"))
        self.assertEqual(fresh_truth.spatial_shadowing(ap, receiver), truth_value)

    def test_interference_state_does_not_leak_between_networks(self) -> None:
        s = replace(scenario("Baseline"), channel_overrides={"interference_burst_start_probability": 1.0,
                                                          "interference_burst_end_probability": 0.0})
        truth, model, survey = build_networks(s)
        truth.scan(Receiver("p", (2.0, 2.0, 1.0)), Random(0))
        self.assertTrue(any(truth._interference_burst_active.values()))
        self.assertEqual(len(survey._interference_burst_active), 0)
        self.assertEqual(len(model._interference_burst_active), 0)

    def test_build_localisers_cheaply(self) -> None:
        s = replace(scenario("Baseline"), radio_map_spacing_m=1.5, radio_map_samples=2)
        _, model, survey = build_networks(s)
        localisers = build_localisers(s, model, survey)
        self.assertIn("WallAware Hybrid", localisers)
        self.assertIn("Fingerprint", localisers)
        for loc in localisers.values():
            self.assertTrue(hasattr(loc, "locate"))


class StubLocaliser:
    def __init__(self, position):
        self.position = position

    def locate(self, scan):
        return None if self.position is None else LocalisationEstimate(self.position)


class SimulationHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        truth, _, _ = build_networks(scenario("Baseline"))
        cls.truth = truth
        cls.noise_free = replace(truth, **NOISE_FREE)

    def test_sample_trials_labels_and_counts(self) -> None:
        trials = sample_trials(self.truth, Random(1), n_trials_per_compartment=4)
        env = self.truth.environment
        self.assertEqual(len(trials), 4 * len(env.compartments))
        for name, pos, scan in trials:
            self.assertEqual(env.compartment_at(pos), name)
            self.assertEqual(set(scan), set(self.truth.access_points))

    def test_receiver_bias_shifts_rssi_exactly(self) -> None:
        for bias in (-6.0, 3.5):
            plain = sample_trials(self.noise_free, Random(2), n_trials_per_compartment=3)
            biased = sample_trials(self.noise_free, Random(2), n_trials_per_compartment=3, receiver_rssi_bias_db=bias)
            for (n1, p1, s1), (n2, p2, s2) in zip(plain, biased):
                self.assertEqual((n1, p1), (n2, p2))
                for ap in self.noise_free.access_points:
                    self.assertIsNotNone(s1[ap])
                    self.assertAlmostEqual(s2[ap].rssi_dbm - s1[ap].rssi_dbm, bias, places=9)

    def test_sample_trials_is_deterministic(self) -> None:
        a = sample_trials(self.truth, Random(3), n_trials_per_compartment=3)
        self.truth.reset_channel_state()
        b = sample_trials(self.truth, Random(3), n_trials_per_compartment=3)
        self.truth.reset_channel_state()
        self.assertEqual([(n, p) for n, p, _ in a], [(n, p) for n, p, _ in b])
        self.assertEqual(
            [[None if r is None else r.rssi_dbm for r in s.values()] for _, _, s in a],
            [[None if r is None else r.rssi_dbm for r in s.values()] for _, _, s in b],
        )

    def test_evaluate_localisers_is_paired_and_counts_misses_as_wrong(self) -> None:
        trials = sample_trials(self.truth, Random(4), n_trials_per_compartment=5)
        localisers = {
            "WA 1": WallAwareLocaliser(self.truth, grid_spacing_m=0.3),
            "WA 2": WallAwareLocaliser(self.truth, grid_spacing_m=0.3),
            "never": StubLocaliser(None),
            "engine": StubLocaliser((3.0, 2.0)),
            "nearest": NearestAPLocaliser(),
        }
        results = evaluate_localisers(localisers, self.truth, trials)
        self.assertEqual(results["WA 1"], results["WA 2"])
        keys = {"errors", "mean_err", "median_err", "p90_err", "miss_rate", "compartment_acc"}
        for r in results.values():
            self.assertEqual(set(r), keys)
        self.assertEqual(results["never"]["miss_rate"], 1.0)
        self.assertEqual(results["never"]["compartment_acc"], 0.0)
        self.assertTrue(math.isnan(results["never"]["mean_err"]))
        n_rooms = len(self.truth.environment.compartments)
        self.assertAlmostEqual(results["engine"]["compartment_acc"], 1.0 / n_rooms)
        self.assertEqual(results["engine"]["miss_rate"], 0.0)

        breakdown = compartment_breakdown(self.truth, {"engine": localisers["engine"], "never": localisers["never"]}, trials)
        self.assertEqual(breakdown["engine"]["Engine Room"], 1.0)
        self.assertTrue(all(v == 0.0 for k, v in breakdown["engine"].items() if k != "Engine Room"))
        self.assertTrue(all(v == 0.0 for v in breakdown["never"].values()))

    def test_evaluate_localisers_deterministic_for_fixed_seed(self) -> None:
        def run():
            truth, model, _ = build_networks(scenario("Baseline"))
            trials = sample_trials(truth, Random(45678), n_trials_per_compartment=4)
            return evaluate_localisers({"WA": WallAwareLocaliser(model, measurement="hybrid", grid_spacing_m=0.3)}, truth, trials)

        self.assertEqual(run(), run())


class GlobalRandomnessTests(unittest.TestCase):
    """Nothing may use the global `random` module state or `np.random`."""

    def _pipeline(self):
        truth, model, _ = build_networks(scenario("Accommodation deck"))
        trials = sample_trials(truth, Random(1), n_trials_per_compartment=1)
        results = evaluate_localisers({"WA": WallAwareLocaliser(model, measurement="hybrid", grid_spacing_m=0.4)}, truth, trials)
        walks = [simulate_walk(truth, Random(2), n_steps=25)]
        tracker = GridTracker(WallAwareLocaliser(model, measurement="hybrid", grid_spacing_m=0.4, robust=True), model)
        stats, _ = evaluate_tracking({"T": tracker}, truth, walks, Random(3))
        return results, stats

    def test_results_independent_of_global_seeds(self) -> None:
        random.seed(1)
        np.random.seed(1)
        first = self._pipeline()
        random.seed(999)
        np.random.seed(999)
        second = self._pipeline()
        self.assertEqual(first, second)

    def test_global_random_functions_are_never_called(self) -> None:
        def forbidden(*args, **kwargs):
            raise AssertionError("global random / np.random used")

        global_names = ["random", "uniform", "gauss", "normalvariate", "randint", "randrange", "choice",
                        "choices", "shuffle", "sample", "expovariate", "lognormvariate"]
        np_names = ["rand", "randn", "random", "normal", "uniform", "choice", "randint", "shuffle",
                    "permutation", "default_rng", "exponential", "rayleigh", "standard_normal"]
        patches = [mock.patch.object(random, n, forbidden) for n in global_names]
        patches += [mock.patch.object(np.random, n, forbidden) for n in np_names]
        for p in patches:
            p.start()
        try:
            self._pipeline()
        finally:
            for p in patches:
                p.stop()


if __name__ == "__main__":
    unittest.main()
