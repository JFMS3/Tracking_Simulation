"""Tests for the measurement-model additions: RTT NLOS mixture likelihood, per-AP RTT offsets, and V7.

- `WallAwareLocaliser(..., rtt_nlos_mixture: bool = False)`: a heavy-tailed (mixture) RTT
  likelihood for large positive range residuals (NLOS excess delay); off by default, and then
  log_likelihood must be bit-for-bit what it was before the feature.
- `Network.rtt_ap_offset_std_m: float = 0.0`: a fixed per-AP range offset (metres), drawn
  deterministically from (shadowing_seed, bssid) like the shadow field, never from the scan rng.
  Default 0 must reproduce previous scans exactly.

The *_GOLDEN values below were recorded from the code before these features existed
(`sensor-fusion` branch, before the fusion implementation): accommodation deck, DECK_APS, layout
propagation, scans at GOLDEN_POSITIONS drawn in order from Random(2468) on a fresh Network;
log-likelihoods on the 0.2 m grid (6000 cells) at GOLDEN_INDICES.
"""
import math
import unittest
from dataclasses import replace
from pathlib import Path
from random import Random

import numpy as np

from environment import Compartment, ShipEnvironment, Wall
from network import AccessPoint, Network, Reading, Receiver, SPEED_OF_LIGHT
from positioning import WallAwareLocaliser
from scenarios import DECK_APS

LAYOUTS = Path(__file__).resolve().parent.parent / "layouts"

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

GOLDEN_SEED = 2468
GOLDEN_POSITIONS = [(2.0, 2.0), (12.5, 5.0), (20.3, 8.7), (14.0, 1.0)]
GOLDEN_INDICES = [0, 777, 2500, 5999]
# per position, per AP in DECK_APS order: (rssi_dbm, rtt_ms) or None (not heard)
SCAN_GOLDEN = [
    [(-70.0, 8.000021988), (-85.0, 8.000082917), None, None, None, None],
    [(-60.0, 8.00006693), (-51.0, 8.000024485), (-50.0, 8.000019074), (-61.0, 8.000054659), (-70.0, 8.000021373), (-71.0, 8.000043251)],
    [None, None, None, (-48.0, 8.000031578), None, (-80.0, 8.000059741)],
    [(-80.0, 8.000103804), (-72.0, 8.000071437), (-74.0, 8.000028484), None, (-44.0, 8.000023264), (-80.0, 8.000072844)],
]
# (measurement, robust) -> per golden scan: (sum over grid, [values at GOLDEN_INDICES])
LL_GOLDEN = {
    ('rss', False): [
        (-411597.16557063814, [-14.810708690062407, -64.15985249333252, -45.738483099077456, -257.7543320784218]),
        (-1842168.5716820422, [-614.9592760790908, -221.1134395707612, -37.871316224695924, -675.2944099690184]),
        (-809600.6024036884, [-467.7422979472735, -46.908676110408784, -12.81139147635193, -39.26259213770102]),
        (-1247963.6276689982, [-333.957753341894, -39.11714172443275, -145.29370030593398, -454.4658375252003]),
    ],
    ('rss', True): [
        (-71146.73644089139, [-4.144931948014993, -15.129881012767799, -15.489986018785052, -14.46394518347676]),
        (-108849.17908377455, [-23.673350624537964, -19.83117773918994, -9.45177118739462, -23.711076330465477]),
        (-83488.28614618404, [-16.23771239546836, -14.956542063932162, -12.276963608935125, -3.7776706590129705]),
        (-92237.11575180737, [-19.000980547921348, -8.942469662701637, -16.253280060301215, -21.139168049716275]),
    ],
    ('rtt', False): [
        (-663985.2683623247, [-15.114572451361793, -60.95114704781851, -398.4917820041953, -193.44921748652789]),
        (-1088488.387753319, [-378.81638266118404, -36.91502280864907, -323.78495316841764, -321.82084194670864]),
        (-531261.9929631326, [-230.45232830854764, -37.28154195120898, -26.280678709685677, -4.735406109357415]),
        (-1024104.0978416651, [-395.07126991487036, -36.240791731158104, -31.503180504705767, -252.23420223015967]),
    ],
    ('rtt', True): [
        (-162971.63660348675, [-4.226416218716762, -22.219114829918926, -16.90587010668398, -141.0561841199011]),
        (-375597.9912117893, [-191.49432877225132, -20.97697278330658, -17.65982908139055, -166.25976765339513]),
        (-214457.8385351853, [-179.85454734873377, -18.149724981634638, -13.606244654546243, -2.4287346786032926]),
        (-308798.9602393804, [-123.51950573922787, -20.834736745540997, -9.614876317128289, -164.06607186526674]),
    ],
    ('hybrid', False): [
        (-945020.0904843088, [-28.53239082279911, -109.61339185096796, -433.2425155825815, -317.8946048223754]),
        (-2645885.3221786143, [-825.732158535018, -246.85253463409066, -361.56223727305746, -850.7283717392926]),
        (-1155556.6258045984, [-526.7339400514725, -70.52717422855189, -28.104304804901705, -43.99734462541066]),
        (-2033428.8235974424, [-623.5049132785557, -61.18627343671854, -173.70711641702968, -557.3174298813975]),
    ],
    ('hybrid', True): [
        (-103556.02959572419, [-6.978457848106664, -21.85138815250365, -21.408106604777842, -22.211184560803602]),
        (-199675.53303881706, [-47.12417919153245, -29.63222277717692, -27.017568148729083, -43.583963807426215]),
        (-112640.15511914692, [-24.631573539853395, -19.443223212500907, -14.895442882345467, -6.205751715968493]),
        (-162397.17407796648, [-36.9963763089405, -15.605546389370307, -22.778391983819425, -35.822630041020574]),
    ],
}


def deck_network(noise_free: bool = False, **overrides) -> Network:
    base = Network.from_config(LAYOUTS / "accommodation_deck.yaml", access_points=list(DECK_APS))
    changes = dict(NOISE_FREE) if noise_free else {}
    changes.update(overrides)
    return replace(base, **changes) if changes else base


def reading(rssi_dbm: float, rtt_ms: float) -> Reading:
    return Reading(rssi_dbm=rssi_dbm, link_speed_mbps=0.0, tx_mbps=0.0, rx_mbps=0.0, rtt_ms=rtt_ms)


def rtt_ms_for_range(network: Network, range_m: float) -> float:
    return network.rtt_base_ms + 2.0 * range_m / SPEED_OF_LIGHT * 1000.0


def measured_range(network: Network, rtt_ms: float) -> float:
    return (rtt_ms - network.rtt_base_ms) / 1000.0 * SPEED_OF_LIGHT / 2.0


def golden_scans() -> list:
    return [{ap: None if g is None else reading(*g) for ap, g in zip(DECK_APS, row)} for row in SCAN_GOLDEN]


def ring_aps(n: int, radius: float = 30.0) -> list:
    """n APs with distinct BSSIDs on a circle around the origin (open field, no walls)."""
    return [
        AccessPoint(f"Ring-{i}", f"40:00:00:00:{i // 256:02x}:{i % 256:02x}",
                    (radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n), 2.0), 2412)
        for i in range(n)
    ]


def open_field(aps, **overrides) -> Network:
    parameters = {"environment": ShipEnvironment(), "access_points": list(aps), **NOISE_FREE,
                  "rtt_range_bias_m": 0.0}
    parameters.update(overrides)
    return Network(**parameters)


def observed_offsets(network: Network, receivers, n_scans: int = 2, seed: int = 1) -> dict:
    """{bssid: [measured range - true 3D distance, for every receiver and scan]} (noise-free channel)."""
    rng = Random(seed)
    out = {ap.bssid: [] for ap in network.access_points}
    for xy in receivers:
        receiver = Receiver("r", (*xy, 1.0))
        for _ in range(n_scans):
            for ap, r in network.scan(receiver, rng).items():
                if r is not None:
                    out[ap.bssid].append(measured_range(network, r.rtt_ms) - math.dist(ap.position, receiver.position))
    return out


class DefaultBehaviourRegressionTests(unittest.TestCase):
    """V7: with both features at their defaults, results are exactly the previous ones."""

    def test_default_scans_reproduce_recorded_scans(self) -> None:
        """Expected: a fresh deck Network (default rtt_ap_offset_std_m, and explicitly 0.0) gives,
        for Random(2468), exactly the recorded RSSI values and the same RTT values (to 1e-12 ms),
        and the same APs heard: the offset feature must not consume the scan rng."""
        self.assertEqual(Network(environment=ShipEnvironment()).rtt_ap_offset_std_m, 0.0)
        for network in (deck_network(), deck_network(rtt_ap_offset_std_m=0.0)):
            rng = Random(GOLDEN_SEED)
            for p, (xy, row) in enumerate(zip(GOLDEN_POSITIONS, SCAN_GOLDEN)):
                scan = network.scan(Receiver("g", (*xy, 1.0)), rng)
                for ap, golden in zip(DECK_APS, row):
                    if golden is None:
                        self.assertIsNone(scan[ap], (p, ap.name))
                        continue
                    self.assertIsNotNone(scan[ap], (p, ap.name))
                    self.assertEqual(scan[ap].rssi_dbm, golden[0], (p, ap.name))
                    self.assertAlmostEqual(scan[ap].rtt_ms, golden[1], delta=1e-12, msg=(p, ap.name))

    def test_default_log_likelihood_reproduces_recorded_values(self) -> None:
        """Expected: for rss/rtt/hybrid, robust or not, both the default constructor and
        rtt_nlos_mixture=False reproduce the recorded log-likelihoods (sum and sampled cells) to a
        relative 1e-12."""
        network = deck_network()
        scans = golden_scans()
        for (measurement, robust), per_scan in LL_GOLDEN.items():
            localisers = [
                WallAwareLocaliser(network, measurement=measurement, grid_spacing_m=0.2, robust=robust),
                WallAwareLocaliser(network, measurement=measurement, grid_spacing_m=0.2, robust=robust, rtt_nlos_mixture=False),
            ]
            for localiser in localisers:
                self.assertEqual(len(localiser.grid_points), 6000)
                for scan, (total, values) in zip(scans, per_scan):
                    ll = localiser.log_likelihood(scan)
                    np.testing.assert_allclose(ll.sum(), total, rtol=1e-12, err_msg=str((measurement, robust)))
                    np.testing.assert_allclose(ll[GOLDEN_INDICES], values, rtol=1e-12, err_msg=str((measurement, robust)))


class RttNlosMixtureTests(unittest.TestCase):
    def test_mixture_does_not_change_rss_only_likelihood(self) -> None:
        """Expected: measurement='rss' has no RTT term, so rtt_nlos_mixture=True gives identical
        log-likelihoods."""
        network = deck_network()
        plain = WallAwareLocaliser(network, measurement="rss", grid_spacing_m=0.4)
        mixture = WallAwareLocaliser(network, measurement="rss", grid_spacing_m=0.4, rtt_nlos_mixture=True)
        for scan in golden_scans():
            np.testing.assert_array_equal(plain.log_likelihood(scan), mixture.log_likelihood(scan))

    def test_mixture_likelihood_is_finite_even_for_extreme_ranges(self) -> None:
        """Expected: with the mixture on (rtt and hybrid, robust or not) log_likelihood is finite at
        every cell for the recorded scans and for scans whose RTT ranges are 0.05 m or 300 m, and it
        differs from the plain likelihood (the flag does something)."""
        network = deck_network()
        extreme = [
            {ap: reading(-60.0, rtt_ms_for_range(network, 0.05)) for ap in DECK_APS},
            {ap: reading(-60.0, rtt_ms_for_range(network, 300.0)) for ap in DECK_APS},
        ]
        for measurement in ("rtt", "hybrid"):
            for robust in (False, True):
                mixture = WallAwareLocaliser(network, measurement=measurement, grid_spacing_m=0.4, robust=robust, rtt_nlos_mixture=True)
                plain = WallAwareLocaliser(network, measurement=measurement, grid_spacing_m=0.4, robust=robust)
                for scan in golden_scans() + extreme:
                    ll = mixture.log_likelihood(scan)
                    self.assertTrue(np.all(np.isfinite(ll)), (measurement, robust))
                self.assertFalse(np.array_equal(mixture.log_likelihood(extreme[1]), plain.log_likelihood(extreme[1])))

    def test_mixture_has_a_heavier_tail_for_large_positive_range_residuals(self) -> None:
        """Expected: at a cell one wall away from a single AP (rtt only, Gaussian loss, no
        detection term), the log-likelihood drop for an 8 m positive range residual (relative to
        zero residual) is positive but less than half of the Gaussian drop 0.5 * (8 / sigma)^2."""
        environment = ShipEnvironment(
            compartments=[Compartment("A", [(0, 0), (4, 0), (4, 4), (0, 4)]), Compartment("B", [(4, 0), (8, 0), (8, 4), (4, 4)])],
            walls=[Wall("Divider", (4.0, 0.0), (4.0, 4.0))],
        )
        ap = AccessPoint("AP", "00:00:00:00:00:aa", (6.0, 2.0, 1.0), 2412)
        network = Network(environment=environment, access_points=[ap])
        kwargs = dict(measurement="rtt", grid_spacing_m=0.5, use_detections=False, robust=False)
        plain = WallAwareLocaliser(network, **kwargs)
        mixture = WallAwareLocaliser(network, rtt_nlos_mixture=True, **kwargs)
        cell = int(np.argmin(np.sum((plain.grid_points - (2.25, 2.25)) ** 2, axis=1)))
        expected_range = float(plain._expected_range[cell, 0])
        sigma = float(plain._range_sigma[cell, 0])

        def drop(localiser, residual):
            at = lambda r: localiser.log_likelihood({ap: reading(-70.0, rtt_ms_for_range(network, expected_range + r))})[cell]  # noqa: E731
            return float(at(0.0) - at(residual))

        gaussian = drop(plain, 8.0)
        self.assertAlmostEqual(gaussian, 0.5 * (8.0 / sigma) ** 2, places=6)  # setup check (passes today)
        heavy = drop(mixture, 8.0)
        self.assertGreater(heavy, 0.0, f"mixture drop {heavy:.2f}")
        self.assertLess(heavy, 0.5 * gaussian, f"mixture drop {heavy:.2f} nats vs Gaussian {gaussian:.2f} nats")

    def test_one_large_positive_outlier_moves_the_estimate_less_with_the_mixture(self) -> None:
        """Expected: crew member in the Mess at (10, 2), noise-free scan, the localiser's model
        matches the channel. Adding +8 m to the (NLOS) Corridor-MW range moves the plain Gaussian
        snapshot estimate by > 2 m (rtt: 4.2 m, hybrid: 3.9 m when recorded); with the mixture the
        error is < 0.5 x the plain error, and on the clean scan the mixture is no more than 0.5 m
        worse than plain."""
        truth = deck_network(noise_free=True)
        model = deck_network(rtt_weak_signal_penalty_m=0.0)
        position = (10.0, 2.0)
        scan = truth.scan(Receiver("p", (*position, 1.0)), Random(1))
        ap = next(a for a in DECK_APS if a.name == "Corridor-MW")
        self.assertGreaterEqual(scan[ap].wall_count, 1)
        corrupted = dict(scan)
        corrupted[ap] = replace(scan[ap], rtt_ms=scan[ap].rtt_ms + 2 * 8.0 / SPEED_OF_LIGHT * 1000)
        for measurement in ("rtt", "hybrid"):
            plain = WallAwareLocaliser(model, measurement=measurement, grid_spacing_m=0.2)
            mixture = WallAwareLocaliser(model, measurement=measurement, grid_spacing_m=0.2, rtt_nlos_mixture=True)
            err = {name: (math.dist(loc.locate(scan).position, position), math.dist(loc.locate(corrupted).position, position))
                   for name, loc in (("plain", plain), ("mixture", mixture))}
            msg = f"{measurement}: clean/outlier errors {err}"
            self.assertGreater(err["plain"][1], 2.0, msg)  # setup check (passes today)
            self.assertLess(err["mixture"][1], 0.5 * err["plain"][1], msg)
            self.assertLess(err["mixture"][0], err["plain"][0] + 0.5, msg)


class PerApRttOffsetTests(unittest.TestCase):
    RECEIVERS = [(0.0, 0.0), (5.0, -3.0), (-4.0, 6.0)]

    def test_negative_or_non_finite_offset_std_raises(self) -> None:
        """Expected: ValueError for rtt_ap_offset_std_m = -0.1, NaN or inf (validated in __post_init__)."""
        for value in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Network(environment=ShipEnvironment(), rtt_ap_offset_std_m=value)

    def test_offset_is_constant_per_ap_across_scans_and_positions(self) -> None:
        """Expected: in a noise-free open field with rtt_ap_offset_std_m = 1.5, measured range minus
        true distance is the same (to 1 mm) for every scan and receiver position of an AP, and the
        offsets differ between APs."""
        offsets = observed_offsets(open_field(ring_aps(8), rtt_ap_offset_std_m=1.5), self.RECEIVERS)
        per_ap = []
        for bssid, values in offsets.items():
            self.assertEqual(len(values), 2 * len(self.RECEIVERS), bssid)
            self.assertLess(max(values) - min(values), 1e-3, (bssid, values))
            per_ap.append(values[0])
        self.assertGreater(float(np.ptp(per_ap)), 0.1, per_ap)

    def test_offset_spread_matches_configured_std(self) -> None:
        """Expected: over 200 APs the per-AP offsets have mean 0 (within 4 standard errors,
        0.42 m) and sample std 1.5 m within 20%."""
        values = np.array([v[0] for v in observed_offsets(open_field(ring_aps(200), rtt_ap_offset_std_m=1.5), [(0.0, 0.0)], n_scans=1).values()])
        self.assertEqual(len(values), 200)
        self.assertLess(abs(values.mean()), 4 * 1.5 / math.sqrt(200), f"mean {values.mean():.3f} m")
        self.assertAlmostEqual(values.std(ddof=1) / 1.5, 1.0, delta=0.2, msg=f"std {values.std(ddof=1):.3f} m")

    def test_offset_is_deterministic_per_shadowing_seed_and_bssid(self) -> None:
        """Expected: offsets depend only on (shadowing_seed, bssid): a separately built network,
        a reversed AP list, and a subset of the APs give the same offset for each BSSID (1 mm);
        another shadowing_seed gives different offsets."""
        aps = ring_aps(8)
        reference = {b: v[0] for b, v in observed_offsets(open_field(aps, rtt_ap_offset_std_m=1.5), [(0.0, 0.0)], 1).items()}
        variants = {
            "rebuilt": open_field(aps, rtt_ap_offset_std_m=1.5),
            "reversed": open_field(aps[::-1], rtt_ap_offset_std_m=1.5),
            "subset": open_field(aps[2:5], rtt_ap_offset_std_m=1.5),
        }
        for name, network in variants.items():
            for bssid, values in observed_offsets(network, [(0.0, 0.0)], 1, seed=99).items():
                self.assertAlmostEqual(values[0], reference[bssid], delta=1e-3, msg=(name, bssid))
        other = {b: v[0] for b, v in observed_offsets(open_field(aps, rtt_ap_offset_std_m=1.5, shadowing_seed=777), [(0.0, 0.0)], 1).items()}
        self.assertGreater(max(abs(other[b] - reference[b]) for b in reference), 0.1)

    def test_offset_does_not_disturb_the_rest_of_the_realistic_channel(self) -> None:
        """Expected: on the realistic deck, the same scan seed with rtt_ap_offset_std_m 0 and 1.5
        gives identical RSSI and the same APs heard (the offset uses no scan randomness), and the
        range difference per AP is one constant (to 1 mm) across positions and scans, not all zero.
        Readings whose apparent range hits the 0.05 m floor are skipped."""
        positions = [(2.0, 2.0), (12.5, 5.0), (20.3, 8.7), (14.0, 1.0), (6.0, 8.0)]
        scans = {}
        for std in (0.0, 1.5):
            network = deck_network(rtt_ap_offset_std_m=std)
            rng = Random(5)
            scans[std] = [network.scan(Receiver("p", (*xy, 1.0)), rng) for xy in positions for _ in range(3)]
        network = deck_network()
        differences = {ap.name: [] for ap in DECK_APS}
        for plain, offset in zip(scans[0.0], scans[1.5]):
            for ap in DECK_APS:
                self.assertEqual(plain[ap] is None, offset[ap] is None, ap.name)
                if plain[ap] is None:
                    continue
                self.assertEqual(plain[ap].rssi_dbm, offset[ap].rssi_dbm, ap.name)
                a, b = measured_range(network, plain[ap].rtt_ms), measured_range(network, offset[ap].rtt_ms)
                if min(a, b) > 0.1:
                    differences[ap.name].append(b - a)
        constants = []
        for name, diffs in differences.items():
            if diffs:
                self.assertLess(max(diffs) - min(diffs), 1e-3, (name, diffs))
                constants.append(diffs[0])
        self.assertGreaterEqual(len(constants), 4)
        self.assertGreater(max(abs(c) for c in constants), 0.05, constants)


if __name__ == "__main__":
    unittest.main()
