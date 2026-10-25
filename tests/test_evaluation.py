"""Keep sensitivity scenarios reproducible and hidden truth out of localisers."""

from contextlib import redirect_stdout
from io import StringIO
from random import Random
import unittest

from crew_tracking.ftm import distance_m_to_rtt_ns
from crew_tracking.models import AccessPoint, FTMBurst, FTMSample, Reading
from simulation import report_compartment_breakdown, report_localiser_stats
from simulator.environment import Compartment, ShipEnvironment
from simulator.evaluation import create_evaluation_session, observed_scan
from simulator.network import Network, Receiver, SimulatedReading


class EvaluationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.aps = [
            AccessPoint(f"AP{index}", str(index), (x, y, 1.0), 2412)
            for index, (x, y) in enumerate(((0, 0), (3, 0), (0, 3), (3, 3)))
        ]
        self.network = Network(
            ShipEnvironment(compartments=[
                Compartment("Room", [(0, 0), (3, 0), (3, 3), (0, 3)])
            ]),
            access_points=self.aps,
            fast_fading_samples=0,
            measurement_noise_std_db=0.0,
            noise_floor_variation_std_db=0.0,
            interference_burst_start_probability=0.0,
            detection_snr_midpoint_db=-100.0,
        )
        self.receiver = Receiver("probe", (1.0, 1.0, 1.0))

    def test_copy_retains_measurements_but_removes_every_simulation_diagnostic(self):
        simulated = SimulatedReading(
            -62.0, 13.0, 11.0, 12.0, 8.0000001,
            FTMBurst((FTMSample(24.0), FTMSample(None, successful=False))),
            wall_count=9,
            is_nlos=True,
            mean_rssi_dbm=-61.0,
            shadowing_db=2.0,
        )
        scan = {self.aps[0]: simulated, self.aps[1]: None}
        copied = observed_scan(scan)
        reading = copied[self.aps[0]]
        self.assertIs(type(reading), Reading)
        self.assertEqual(reading.__dict__, {
            key: value for key, value in simulated.__dict__.items()
            if key in Reading.__dataclass_fields__
        })
        self.assertFalse(hasattr(reading, "is_nlos"))
        self.assertFalse(hasattr(reading, "wall_count"))
        self.assertIsNot(reading, simulated)
        self.assertIsNone(copied[self.aps[1]])

    def test_common_ftm_bias_preserves_burst_scatter_and_failed_samples(self):
        original = Reading(-60.0, ftm_burst=FTMBurst((
            FTMSample(20.0), FTMSample(22.0), FTMSample(None, successful=False),
        )))
        observed = observed_scan(
            {self.aps[0]: original},
            rssi_offsets_db={self.aps[0].bssid: -2.5},
            ftm_range_bias_m=0.5,
        )[self.aps[0]]
        self.assertEqual(observed.rssi_dbm, -62.5)
        first, second, failed = observed.ftm_burst.samples
        self.assertAlmostEqual(first.rtt_ns, 20.0 + distance_m_to_rtt_ns(0.5))
        self.assertAlmostEqual(second.rtt_ns - first.rtt_ns, 2.0)
        self.assertEqual(failed, original.ftm_burst.samples[2])
        self.assertEqual(original.rssi_dbm, -60.0)
        self.assertEqual(original.ftm_burst.samples[0].rtt_ns, 20.0)

    def test_deployment_session_does_not_mutate_survey_network_or_receiver(self):
        self.network.scan(self.receiver, Random(3))
        original_fields = dict(self.network._shadow_fields)
        original_cache = dict(self.network._shadow_value_cache)
        original_interference = dict(self.network._interference_burst_active)
        session = create_evaluation_session(self.network, "deployment", 123)
        session.scan(self.receiver, Random(4))
        self.assertIsNot(session.network, self.network)
        self.assertNotEqual(session.network.shadowing_seed, self.network.shadowing_seed)
        self.assertEqual(self.network.shadowing_seed, 12345)
        self.assertEqual(self.network._shadow_fields, original_fields)
        self.assertEqual(self.network._shadow_value_cache, original_cache)
        self.assertEqual(self.network._interference_burst_active, original_interference)
        self.assertEqual(self.receiver.position, (1.0, 1.0, 1.0))
        self.assertEqual(self.receiver.rssi_bias_db, 0.0)

    def test_both_profiles_are_reproducible_and_expose_only_plain_readings(self):
        for profile in ("matched", "deployment"):
            with self.subTest(profile=profile):
                first = create_evaluation_session(self.network, profile, 5)
                second = create_evaluation_session(self.network, profile, 5)
                first_rng, second_rng = Random(19), Random(19)
                for _ in range(30):
                    first_scan = first.scan(self.receiver, first_rng)
                    second_scan = second.scan(self.receiver, second_rng)
                    self.assertEqual(first_scan, second_scan)
                    self.assertTrue(all(
                        reading is None or type(reading) is Reading
                        for reading in first_scan.values()
                    ))

    def test_outage_drops_a_whole_visible_ap_including_its_ftm(self):
        session = create_evaluation_session(self.network, "deployment", 7)
        session.single_ap_outage_probability = 1.0
        scan = session.scan(self.receiver, Random(9))
        self.assertEqual(sum(reading is None for reading in scan.values()), 1)
        self.assertEqual(len(self.network.access_points), 4)

    def test_matched_profile_preserves_original_observed_values(self):
        session = create_evaluation_session(self.network, "matched", 8)
        expected = observed_scan(self.network.scan(self.receiver, Random(22)))
        self.assertEqual(session.scan(self.receiver, Random(22)), expected)

    def test_receiver_reporting_offset_does_not_change_beacon_detection_or_timing(self):
        original = create_evaluation_session(self.network, "matched", 8)
        biased = create_evaluation_session(self.network, "matched", 8)
        biased.receiver_bias_db = 4.0
        # Work near the visibility threshold, where feeding an RSSI reporting
        # offset into the simulated RF channel would change beacon detection.
        original.network.detection_snr_midpoint_db = 45.0
        biased.network.detection_snr_midpoint_db = 45.0
        original_rng, biased_rng = Random(24), Random(24)
        for _ in range(20):
            before = original.scan(self.receiver, original_rng)
            after = biased.scan(self.receiver, biased_rng)
            for ap in before:
                self.assertEqual(before[ap] is None, after[ap] is None)
                if before[ap] is not None:
                    self.assertEqual(after[ap].rssi_dbm, before[ap].rssi_dbm + 4.0)
                    self.assertEqual(after[ap].ftm_burst, before[ap].ftm_burst)

    def test_reports_give_every_method_the_identical_plain_scans(self):
        class Recorder:
            def __init__(self):
                self.scans = []

            def locate(self, scan):
                self.scans.append(scan)
                return None

        for report in (report_localiser_stats, report_compartment_breakdown):
            with self.subTest(report=report.__name__):
                first, second = Recorder(), Recorder()
                session = create_evaluation_session(self.network, "deployment", 4)
                with redirect_stdout(StringIO()):
                    report(
                        {"first": first, "second": second}, self.network, Random(6),
                        n_trials_per_compartment=5, evaluation_session=session,
                    )
                self.assertEqual(len(first.scans), 5)
                for first_scan, second_scan in zip(first.scans, second.scans):
                    self.assertIs(first_scan, second_scan)
                    self.assertTrue(all(
                        reading is None or type(reading) is Reading
                        for reading in first_scan.values()
                    ))

    def test_invalid_profile_or_seed_is_rejected(self):
        with self.assertRaises(ValueError):
            create_evaluation_session(self.network, "unknown")
        for seed in (-1, True, 1.2):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                create_evaluation_session(self.network, seed=seed)


if __name__ == "__main__":
    unittest.main()
