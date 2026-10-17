"""Tests for tracking.FixedLagSmoother: exact fixed-lag forward-backward smoothing of a GridTracker.

Interface under test (docs/sensor_fusion_spec.md):
    FixedLagSmoother(tracker: GridTracker, lag_steps: int = 5)
    .delay_steps == lag_steps
    .update(scan, motion) -> None for the first lag_steps epochs, then the smoothed estimate
                             of epoch t - lag_steps given all scans up to epoch t
    .flush() -> the smoothed estimates of the remaining epochs, given all scans of the walk
    .reset()

The reference ("oracle") is the exact HMM that GridTracker implements: belief starts uniform,
each epoch predicts with M = (1 - r) * _predict_matrix + r / n (r = reacquire_probability), then
multiplies by exp(localiser.log_likelihood(scan)) (no factor when scan is None). The smoothed
marginal of epoch s given epochs 0..t is alpha_s * beta_s. Assumed beyond the spec: a smoothed
estimate is `localiser.estimate_from_log_posterior(log(smoothed marginal))`, i.e. the same
estimator GridTracker uses for its filtered belief, so positions and covariances are compared.
"""
import itertools
import math
import unittest
from pathlib import Path
from random import Random

import numpy as np

from environment import Compartment, ShipEnvironment, Wall
from network import AccessPoint, Network, Receiver
from positioning import WallAwareLocaliser
from scenarios import DECK_APS
from tracking import FixedLagSmoother, GridTracker, evaluate_tracking, simulate_walk

LAYOUTS = Path(__file__).resolve().parent.parent / "layouts"


def tiny_network() -> Network:
    """Two 0.8 x 0.8 m rooms joined by a doorway (y 0.4-0.8); with 0.4 m grid spacing: 8 cells."""
    environment = ShipEnvironment(
        compartments=[
            Compartment("A", [(0.0, 0.0), (0.8, 0.0), (0.8, 0.8), (0.0, 0.8)]),
            Compartment("B", [(0.8, 0.0), (1.6, 0.0), (1.6, 0.8), (0.8, 0.8)]),
        ],
        walls=[Wall("Divider", (0.8, 0.0), (0.8, 0.4), attenuation_db=10.0)],
    )
    aps = [
        AccessPoint("AP-A", "00:00:00:00:00:0a", (0.2, 0.6, 1.5), 2412),
        AccessPoint("AP-B", "00:00:00:00:00:0b", (1.4, 0.2, 1.5), 2437),
    ]
    return Network(environment=environment, access_points=aps)


def deck_network() -> Network:
    return Network.from_config(LAYOUTS / "accommodation_deck.yaml", access_points=list(DECK_APS))


def transition(tracker: GridTracker) -> np.ndarray:
    """Dense column-stochastic HMM transition M[to, from] used by GridTracker.update."""
    P = tracker._predict_matrix.toarray()
    n = P.shape[0]
    r = tracker.reacquire_probability
    return (1.0 - r) * P + r / n


def likelihoods(localiser: WallAwareLocaliser, scans) -> list:
    out = []
    for scan in scans:
        if scan is None:
            out.append(np.ones(len(localiser.grid_points)))
        else:
            ll = localiser.log_likelihood(scan)
            out.append(np.exp(ll - ll.max()))
    return out


def exact_smoothed(M: np.ndarray, lik: list, s: int, t: int) -> np.ndarray:
    """p(x_s | scans 0..t) by forward-backward on the dense matrix, recomputed from scratch."""
    n = M.shape[0]
    belief = np.full(n, 1.0 / n)
    alpha = []
    for k in range(t + 1):
        belief = lik[k] * (M @ belief)
        belief /= belief.sum()
        alpha.append(belief)
    beta = np.ones(n)
    for k in range(t - 1, s - 1, -1):
        beta = M.T @ (lik[k + 1] * beta)
        beta /= beta.sum()
    marginal = alpha[s] * beta
    return marginal / marginal.sum()


def enumerated_smoothed(M: np.ndarray, lik: list, s: int, t: int) -> np.ndarray:
    """p(x_s | scans 0..t) by brute force: sum the joint over every state sequence x_0..x_t."""
    n = M.shape[0]
    marginal = np.zeros(n)
    prior0 = M @ np.full(n, 1.0 / n)
    for seq in itertools.product(range(n), repeat=t + 1):
        p = prior0[seq[0]] * lik[0][seq[0]]
        for k in range(1, t + 1):
            p *= M[seq[k], seq[k - 1]] * lik[k][seq[k]]
        marginal[seq[s]] += p
    return marginal / marginal.sum()


def walk_scans(network: Network, positions, rng: Random, missing=()):
    network.reset_channel_state()
    return [None if t in missing else network.scan(Receiver("crew", (*xy, 1.0)), rng) for t, xy in enumerate(positions)]


class SmootherExactnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tiny = tiny_network()
        cls.tiny_localiser = WallAwareLocaliser(cls.tiny, measurement="hybrid", grid_spacing_m=0.4, robust=True)
        cls.tiny_positions = [(0.2, 0.2), (0.5, 0.3), (0.6, 0.6), (1.0, 0.6), (1.3, 0.5), (1.4, 0.2),
                              (1.0, 0.3), (0.7, 0.6), (0.3, 0.5), (0.2, 0.2), (0.5, 0.6), (1.1, 0.6)]
        cls.tiny_scans = walk_scans(cls.tiny, cls.tiny_positions, Random(31), missing={5})

    def tiny_tracker(self) -> GridTracker:
        # slow walker: the motion kernel only links neighbouring cells, so time order matters
        return GridTracker(self.tiny_localiser, self.tiny, walking_speed_mps=0.4, reacquire_probability=0.01)

    def assert_estimate_matches(self, localiser, estimate, marginal, context) -> None:
        expected = localiser.estimate_from_log_posterior(np.log(np.maximum(marginal, 1e-300)))
        self.assertIsNotNone(estimate, context)
        np.testing.assert_allclose(estimate.position, expected.position, atol=1e-6, err_msg=str(context))
        self.assertIsNotNone(estimate.covariance, context)
        np.testing.assert_allclose(estimate.covariance, expected.covariance, atol=1e-6, err_msg=str(context))

    def test_tiny_grid_has_few_cells_and_oracle_matches_enumeration(self) -> None:
        """Expected: the tiny layout has 8 grid cells, and the forward-backward oracle equals brute
        enumeration of all 8^5 state sequences (to 1e-12) for every s <= t = 4, including a step
        without a scan. This validates the oracle used by the other tests."""
        tracker = self.tiny_tracker()
        self.assertEqual(len(self.tiny_localiser.grid_points), 8)
        M = transition(tracker)
        scans = self.tiny_scans[:5]
        scans = scans[:3] + [None] + scans[4:]
        lik = likelihoods(self.tiny_localiser, scans)
        for s in range(5):
            np.testing.assert_allclose(exact_smoothed(M, lik, s, 4), enumerated_smoothed(M, lik, s, 4), atol=1e-12)
        # sanity: the motion model makes smoothing differ from filtering
        self.assertGreater(np.abs(exact_smoothed(M, lik, 1, 4) - exact_smoothed(M, lik, 1, 1)).max(), 1e-3)

    def test_update_returns_exact_lagged_smoothed_estimates_on_tiny_grid(self) -> None:
        """Expected: for lags 1, 2 and 3 on the tiny grid (12 epochs, one without a scan), update()
        returns None for the first `lag` epochs and then, at epoch t, exactly the smoothed estimate
        of epoch t - lag given scans 0..t; flush() then returns exactly `lag` estimates, those of
        the last `lag` epochs given every scan."""
        lik = likelihoods(self.tiny_localiser, self.tiny_scans)
        T = len(self.tiny_scans)
        for lag in (1, 2, 3):
            tracker = self.tiny_tracker()
            M = transition(tracker)
            smoother = FixedLagSmoother(tracker, lag)
            smoother.reset()
            for t, scan in enumerate(self.tiny_scans):
                estimate = smoother.update(scan, None)
                if t < lag:
                    self.assertIsNone(estimate, (lag, t))
                else:
                    self.assert_estimate_matches(self.tiny_localiser, estimate, exact_smoothed(M, lik, t - lag, t), (lag, t))
            remaining = list(smoother.flush())
            self.assertEqual(len(remaining), lag)
            for k, estimate in enumerate(remaining):
                s = T - lag + k
                self.assert_estimate_matches(self.tiny_localiser, estimate, exact_smoothed(M, lik, s, T - 1), (lag, "flush", s))

    def test_exact_on_deck_grid_with_lag_5(self) -> None:
        """Expected: on the accommodation deck (0.5 m grid, realistic channel, 25 epochs of a crew
        walk with a 3-epoch Wi-Fi outage) a lag-5 smoother's estimates equal the oracle's, both
        from update() and from flush()."""
        network = deck_network()
        localiser = WallAwareLocaliser(network, measurement="hybrid", grid_spacing_m=0.5, robust=True)
        tracker = GridTracker(localiser, network)
        positions = simulate_walk(network, Random(41), n_steps=25)
        scans = walk_scans(network, positions, Random(42), missing={10, 11, 12})
        lik = likelihoods(localiser, scans)
        M = transition(tracker)
        lag, T = 5, len(scans)
        smoother = FixedLagSmoother(tracker, lag)
        smoother.reset()
        for t, scan in enumerate(scans):
            estimate = smoother.update(scan, None)
            if t >= lag:
                self.assert_estimate_matches(localiser, estimate, exact_smoothed(M, lik, t - lag, t), t)
        for k, estimate in enumerate(smoother.flush()):
            self.assert_estimate_matches(localiser, estimate, exact_smoothed(M, lik, T - lag + k, T - 1), ("flush", k))

    def test_walk_shorter_than_lag_is_returned_entirely_by_flush(self) -> None:
        """Expected: with lag 5 and only 3 epochs, update() returns None three times and flush()
        returns 3 estimates, each the exact smoothed estimate given all 3 scans."""
        tracker = self.tiny_tracker()
        M = transition(tracker)
        scans = self.tiny_scans[:3]
        lik = likelihoods(self.tiny_localiser, scans)
        smoother = FixedLagSmoother(tracker, 5)
        smoother.reset()
        self.assertEqual([smoother.update(scan, None) for scan in scans], [None, None, None])
        remaining = list(smoother.flush())
        self.assertEqual(len(remaining), 3)
        for s, estimate in enumerate(remaining):
            self.assert_estimate_matches(self.tiny_localiser, estimate, exact_smoothed(M, lik, s, 2), s)


class SmootherBehaviourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.network = deck_network()
        cls.localiser = WallAwareLocaliser(cls.network, measurement="hybrid", grid_spacing_m=0.4, robust=True)
        cls.walks = [simulate_walk(cls.network, Random(s), n_steps=40) for s in (51, 52)]

    def test_delay_steps_equals_lag(self) -> None:
        """Expected: delay_steps == lag_steps for lags 0, 1 and 5."""
        for lag in (0, 1, 5):
            self.assertEqual(FixedLagSmoother(GridTracker(self.localiser, self.network), lag).delay_steps, lag)

    def test_negative_lag_raises(self) -> None:
        """Expected: ValueError for a negative lag."""
        with self.assertRaises(ValueError):
            FixedLagSmoother(GridTracker(self.localiser, self.network), -1)

    def test_lag_zero_is_the_forward_filter(self) -> None:
        """Expected: with lag 0 every update returns the same estimate as an identical GridTracker
        (positions equal to 1e-9) and flush() returns nothing."""
        scans = walk_scans(self.network, self.walks[0], Random(53), missing={7, 8})
        smoother = FixedLagSmoother(GridTracker(self.localiser, self.network), 0)
        forward = GridTracker(self.localiser, self.network)
        smoother.reset()
        forward.reset()
        for t, scan in enumerate(scans):
            a, b = smoother.update(scan, None), forward.update(scan, None)
            np.testing.assert_allclose(a.position, b.position, atol=1e-9, err_msg=str(t))
        self.assertEqual(list(smoother.flush()), [])

    def test_reset_discards_pending_epochs(self) -> None:
        """Expected: a smoother that ran part of another walk, then reset(), gives exactly the same
        estimates (from update and flush) as a fresh smoother; a second flush() returns nothing."""
        first = walk_scans(self.network, self.walks[0][:17], Random(54))
        second = walk_scans(self.network, self.walks[1], Random(55))

        def run(smoother, scans):
            smoother.reset()
            out = [smoother.update(scan, None) for scan in scans]
            out += list(smoother.flush())
            return [None if e is None else e.position for e in out]

        used = FixedLagSmoother(GridTracker(self.localiser, self.network), 4)
        used.reset()
        for scan in first:
            used.update(scan, None)  # no flush: epochs left pending
        self.assertEqual(run(used, second), run(FixedLagSmoother(GridTracker(self.localiser, self.network), 4), second))
        self.assertEqual(list(used.flush()), [])

    def test_evaluate_tracking_aligns_every_epoch(self) -> None:
        """Expected: in evaluate_tracking a lag-5 smoother gets an estimate for every epoch of every
        walk (no None after alignment): error count == total epochs, per-walk estimate lists as
        long as the walks, and the estimates are close to the truth (mean error < 1.5 m)."""
        trackers = {
            "Grid": GridTracker(self.localiser, self.network),
            "Smoother": FixedLagSmoother(GridTracker(self.localiser, self.network), 5),
        }
        stats, estimates = evaluate_tracking(trackers, self.network, self.walks, Random(56))
        total = sum(len(w) for w in self.walks)
        self.assertEqual(len(stats["Smoother"]["errors"]), total)
        self.assertEqual([len(e) for e in estimates["Smoother"]], [len(w) for w in self.walks])
        self.assertTrue(all(p is not None for walk in estimates["Smoother"] for p in walk))
        self.assertLess(stats["Smoother"]["mean_err"], 1.5, stats["Smoother"]["mean_err"])
        self.assertFalse(math.isnan(stats["Smoother"]["coverage95"]))


if __name__ == "__main__":
    unittest.main()
