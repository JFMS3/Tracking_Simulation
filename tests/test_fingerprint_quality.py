"""Fingerprint ambiguity and deployment mismatch must remain visible to fusion."""

import unittest

import numpy as np

from crew_tracking.fingerprinting import FingerprintEntry, FingerprintLocaliser
from crew_tracking.models import AccessPoint, Reading


class FingerprintQualityTests(unittest.TestCase):
    def setUp(self):
        self.aps = [AccessPoint(str(i), str(i), (i, 0, 1), 2412) for i in range(4)]
        self.vector = np.array([-45.0, -55.0, -65.0, -75.0])
        self.scan = {ap: Reading(float(rssi)) for ap, rssi in zip(self.aps, self.vector)}

    def test_distant_equally_good_matches_are_detected_even_with_k_one(self):
        radio_map = [FingerprintEntry((0, 0), None, self.vector),
                     FingerprintEntry((10, 0), None, self.vector)]
        result = FingerprintLocaliser(radio_map, self.aps, k=1).locate(self.scan)
        self.assertEqual(result.position, (0.0, 0.0))
        self.assertEqual(result.quality.match_rms_db, 0)
        self.assertEqual(result.quality.neighbour_spread_m, 0)
        self.assertGreater(result.quality.ambiguity_spread_m, 5)

    def test_device_offset_increases_mismatch_without_using_device_truth(self):
        loc = FingerprintLocaliser([FingerprintEntry((0, 0), None, self.vector)], self.aps)
        shifted = {ap: Reading(reading.rssi_dbm + 8) for ap, reading in self.scan.items()}
        self.assertEqual(loc.locate(self.scan).quality.match_rms_db, 0)
        self.assertAlmostEqual(loc.locate(shifted).quality.match_rms_db, 8)

    def test_missing_aps_reduce_observed_fraction_unknown_aps_do_not_inflate_it(self):
        loc = FingerprintLocaliser([FingerprintEntry((0, 0), None, self.vector)], self.aps)
        unknown = AccessPoint("unknown", "new", (5, 5, 1), 2412)
        result = loc.locate({self.aps[0]: self.scan[self.aps[0]], unknown: Reading(-30)})
        self.assertEqual(result.num_aps_used, 1)
        self.assertEqual(result.quality.observed_ap_fraction, 0.25)

    def test_simulation_only_attributes_are_never_read(self):
        class HardwareReading(Reading):
            def __getattribute__(self, name):
                if name in {"wall_count", "is_nlos", "mean_rssi_dbm", "shadowing_db", "snr_db"}:
                    raise AssertionError(f"Unavailable hardware attribute accessed: {name}")
                return super().__getattribute__(name)

        loc = FingerprintLocaliser([FingerprintEntry((0, 0), None, self.vector)], self.aps)
        scan = {ap: HardwareReading(reading.rssi_dbm) for ap, reading in self.scan.items()}
        observed, expected = loc.locate(scan), loc.locate(self.scan)
        self.assertEqual(observed.position, expected.position)
        self.assertEqual(observed.quality, expected.quality)


if __name__ == "__main__":
    unittest.main()
