import math
import unittest
from random import Random

from environment import ShipEnvironment, Wall
from network import AccessPoint, Network, Receiver, SPEED_OF_LIGHT


class NetworkNoiseModelTests(unittest.TestCase):
    def make_network(self, **overrides) -> Network:
        parameters = {
            "environment": ShipEnvironment(),
            "shadowing_std_db": 0.0,
            "fast_fading_samples": 0,
            "measurement_noise_std_db": 0.0,
            "rssi_quantization_db": 0.0,
            "noise_floor_variation_std_db": 0.0,
            "interference_burst_start_probability": 0.0,
            "interference_burst_end_probability": 1.0,
            "detection_snr_midpoint_db": -100.0,
            "rtt_ranging_noise_m": 0.0,
            "rtt_weak_signal_penalty_m": 0.0,
            "rtt_outlier_probability": 0.0,
            "rtt_nlos_outlier_probability": 0.0,
        }
        parameters.update(overrides)
        return Network(**parameters)

    def test_log_distance_mean_is_exact_when_noise_is_disabled(self) -> None:
        network = self.make_network(
            reference_rssi_dbm=-40.0,
            reference_distance_m=1.0,
            path_loss_exponent=2.0,
        )
        ap = AccessPoint("AP", "00:00:00:00:00:01", (0.0, 0.0, 1.0), 2412)

        at_reference = Receiver("near", (1.0, 0.0, 1.0))
        ten_times_farther = Receiver("far", (10.0, 0.0, 1.0))

        self.assertAlmostEqual(network.mean_rssi(ap, at_reference), -40.0)
        self.assertAlmostEqual(network.mean_rssi(ap, ten_times_farther), -60.0)

    def test_crossed_wall_adds_its_attenuation(self) -> None:
        wall = Wall("bulkhead", (0.0, -1.0), (0.0, 1.0), attenuation_db=20.0)
        network = self.make_network(environment=ShipEnvironment(walls=[wall]))
        clear_network = self.make_network()
        ap = AccessPoint("AP", "00:00:00:00:00:01", (-1.0, 0.0, 1.0), 2412)
        receiver = Receiver("phone", (1.0, 0.0, 1.0))

        difference = clear_network.mean_rssi(ap, receiver) - network.mean_rssi(ap, receiver)
        self.assertAlmostEqual(difference, 20.0)

    def test_nominal_noise_floor_uses_bandwidth_and_noise_figure(self) -> None:
        network = self.make_network(
            channel_bandwidth_mhz=20.0,
            receiver_noise_figure_db=7.0,
        )

        expected = -174.0 + 10.0 * math.log10(20_000_000.0) + 7.0
        self.assertAlmostEqual(network.nominal_noise_floor_dbm, expected)

    def test_detection_probability_is_monotonic_in_rssi(self) -> None:
        network = self.make_network(
            detection_snr_midpoint_db=6.0,
            detection_snr_transition_db=1.5,
            beacons_per_scan=1,
        )
        floor = -94.0

        weak = network.detection_probability(-95.0, floor)
        midpoint = network.detection_probability(-88.0, floor)
        strong = network.detection_probability(-60.0, floor)

        self.assertLess(weak, midpoint)
        self.assertLess(midpoint, strong)
        self.assertAlmostEqual(midpoint, 0.5)

    def test_shadow_field_is_stable_and_access_order_independent(self) -> None:
        first = self.make_network(shadowing_std_db=2.0, shadowing_seed=4321)
        second = self.make_network(shadowing_std_db=2.0, shadowing_seed=4321)
        ap1 = AccessPoint("AP1", "00:00:00:00:00:01", (0.0, 0.0, 1.0), 2412)
        ap2 = AccessPoint("AP2", "00:00:00:00:00:02", (3.0, 0.0, 1.0), 2412)
        receiver = Receiver("phone", (1.0, 2.0, 1.0))

        first_ap1 = first.spatial_shadowing(ap1, receiver)
        first_ap2 = first.spatial_shadowing(ap2, receiver)
        second_ap2 = second.spatial_shadowing(ap2, receiver)
        second_ap1 = second.spatial_shadowing(ap1, receiver)

        self.assertEqual(first_ap1, first.spatial_shadowing(ap1, receiver))
        self.assertEqual(first_ap1, second_ap1)
        self.assertEqual(first_ap2, second_ap2)

    def test_shadow_field_approximates_exponential_spatial_correlation(self) -> None:
        network = self.make_network(
            shadowing_std_db=1.0,
            shadowing_correlation_distance_m=2.0,
            shadowing_components=96,
            shadowing_seed=2468,
        )
        origin = Receiver("origin", (0.0, 0.0, 1.0))
        one_correlation_length = Receiver("near", (2.0, 0.0, 1.0))
        five_correlation_lengths = Receiver("far", (10.0, 0.0, 1.0))
        origin_values = []
        near_values = []
        far_values = []

        for index in range(600):
            ap = AccessPoint(
                f"AP{index}",
                f"02:00:00:{index // 65536:02x}:{(index // 256) % 256:02x}:{index % 256:02x}",
                (0.0, 0.0, 1.0),
                2412,
            )
            origin_values.append(network.spatial_shadowing(ap, origin))
            near_values.append(network.spatial_shadowing(ap, one_correlation_length))
            far_values.append(network.spatial_shadowing(ap, five_correlation_lengths))

        def correlation(xs, ys) -> float:
            x_mean = sum(xs) / len(xs)
            y_mean = sum(ys) / len(ys)
            covariance = sum(
                (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
            )
            x_energy = sum((x - x_mean) ** 2 for x in xs)
            y_energy = sum((y - y_mean) ** 2 for y in ys)
            return covariance / math.sqrt(x_energy * y_energy)

        near_correlation = correlation(origin_values, near_values)
        far_correlation = correlation(origin_values, far_values)

        self.assertAlmostEqual(near_correlation, math.exp(-1.0), delta=0.12)
        self.assertAlmostEqual(far_correlation, math.exp(-5.0), delta=0.12)

    def test_averaged_rayleigh_fading_is_zero_mean_in_db(self) -> None:
        network = self.make_network(fast_fading_samples=8)
        rng = Random(101)
        samples = [network._fast_fading_db(rng) for _ in range(20_000)]

        self.assertAlmostEqual(sum(samples) / len(samples), 0.0, delta=0.08)

    def test_seeded_scan_sequences_are_reproducible(self) -> None:
        ap = AccessPoint("AP", "00:00:00:00:00:01", (0.0, 0.0, 1.0), 2412)
        receiver = Receiver("phone", (2.0, 0.0, 1.0))
        first = Network(ShipEnvironment(), [ap], [receiver])
        second = Network(ShipEnvironment(), [ap], [receiver])
        first_rng = Random(9876)
        second_rng = Random(9876)

        first_values = [first.scan(receiver, first_rng)[ap] for _ in range(30)]
        second_values = [second.scan(receiver, second_rng)[ap] for _ in range(30)]

        self.assertEqual(first_values, second_values)

    def test_access_points_on_same_channel_share_scan_noise_floor(self) -> None:
        ap1 = AccessPoint("AP1", "00:00:00:00:00:01", (0.0, 0.0, 1.0), 2412)
        ap2 = AccessPoint("AP2", "00:00:00:00:00:02", (1.0, 0.0, 1.0), 2412)
        receiver = Receiver("phone", (0.5, 0.0, 1.0))
        network = self.make_network(
            access_points=[ap1, ap2],
            receivers=[receiver],
            noise_floor_variation_std_db=3.0,
        )

        scan = network.scan(receiver, Random(55))

        self.assertIsNotNone(scan[ap1])
        self.assertIsNotNone(scan[ap2])
        assert scan[ap1] is not None and scan[ap2] is not None
        self.assertEqual(scan[ap1].noise_floor_dbm, scan[ap2].noise_floor_dbm)

    def test_reading_reports_wall_and_noise_diagnostics(self) -> None:
        wall = Wall("bulkhead", (0.0, -1.0), (0.0, 1.0), attenuation_db=20.0)
        network = self.make_network(environment=ShipEnvironment(walls=[wall]))
        ap = AccessPoint("AP", "00:00:00:00:00:01", (-1.0, 0.0, 1.0), 2412)
        receiver = Receiver("phone", (1.0, 0.0, 1.0))

        reading = network.reading(ap, receiver, Random(1), noise_floor_dbm=-120.0)

        self.assertIsNotNone(reading)
        assert reading is not None
        self.assertEqual(reading.wall_count, 1)
        self.assertTrue(reading.is_nlos)
        self.assertIsNotNone(reading.mean_rssi_dbm)
        self.assertIsNotNone(reading.snr_db)

    def test_nlos_rtt_has_more_bias_and_variance_than_los(self) -> None:
        network = self.make_network(
            rtt_ranging_noise_m=0.3,
            rtt_nlos_bias_per_wall_m=1.0,
            rtt_nlos_noise_per_wall_m=0.5,
        )
        los_rng = Random(123)
        nlos_rng = Random(123)

        def range_error(rtt_ms: float) -> float:
            measured_distance = (
                (rtt_ms - network.rtt_base_ms) / 1000.0 * SPEED_OF_LIGHT / 2.0
            )
            return measured_distance - 5.0

        los = [range_error(network.rtt(5.0, -60.0, los_rng)) for _ in range(5_000)]
        nlos = [
            range_error(network.rtt(5.0, -60.0, nlos_rng, wall_count=1))
            for _ in range(5_000)
        ]

        los_mean = sum(los) / len(los)
        nlos_mean = sum(nlos) / len(nlos)
        los_variance = sum((value - los_mean) ** 2 for value in los) / len(los)
        nlos_variance = sum((value - nlos_mean) ** 2 for value in nlos) / len(nlos)

        self.assertGreater(nlos_mean, los_mean + 0.7)
        self.assertGreater(nlos_variance, los_variance)

    def test_invalid_noise_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.make_network(detection_snr_transition_db=0.0)


if __name__ == "__main__":
    unittest.main()
