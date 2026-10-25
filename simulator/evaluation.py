"""Controlled survey-to-use sensitivity scenarios, outside deployment algorithms.

The deployment profile is a reproducible stress test with assumed magnitudes;
it is not a fitted model of errors aboard a real vessel. Only measurement
contracts cross the boundary into a localiser, even for the matched profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from random import Random
from typing import Mapping, Optional

from crew_tracking.ftm import distance_m_to_rtt_ns
from crew_tracking.models import FTMBurst, FTMSample, Reading, Scan

from .network import Network, Receiver


def observed_scan(
    scan: Scan,
    *,
    rssi_offsets_db: Optional[Mapping[str, float]] = None,
    ftm_range_bias_m: float = 0.0,
) -> Scan:
    """Copy observations and discard every simulation-only diagnostic.

    RSSI offsets are reporting perturbations applied after beacon detection;
    they do not simulate changes in transmit power or change link speed.
    A common timing offset represents a persistent device ranging error and
    deliberately leaves within-burst variance unchanged.
    """
    offsets = rssi_offsets_db or {}
    timing_bias_ns = distance_m_to_rtt_ns(ftm_range_bias_m)
    result: Scan = {}
    for ap, reading in scan.items():
        if reading is None:
            result[ap] = None
            continue
        burst = reading.ftm_burst
        if burst is not None:
            burst = FTMBurst(tuple(
                FTMSample(
                    sample.rtt_ns + timing_bias_ns
                    if (sample.successful and sample.rtt_ns is not None
                        and math.isfinite(sample.rtt_ns))
                    else sample.rtt_ns,
                    successful=sample.successful,
                )
                for sample in burst.samples
            ))
        result[ap] = Reading(
            rssi_dbm=reading.rssi_dbm + offsets.get(ap.bssid, 0.0),
            link_speed_mbps=reading.link_speed_mbps,
            tx_mbps=reading.tx_mbps,
            rx_mbps=reading.rx_mbps,
            rtt_ms=reading.rtt_ms,
            ftm_burst=burst,
        )
    return result


@dataclass
class EvaluationSession:
    """A separate simulated session that exposes observations, never truth."""

    network: Network
    profile: str = "matched"
    receiver_bias_db: float = 0.0
    receiver_height_m: Optional[float] = None
    rssi_offsets_db: dict[str, float] = field(default_factory=dict)
    single_ap_outage_probability: float = 0.0
    ftm_range_bias_m: float = 0.0
    _outage_rng: Random = field(default_factory=Random, repr=False)

    def scan(self, receiver: Receiver, rng: Random) -> Scan:
        position = receiver.position
        if self.receiver_height_m is not None:
            position = (position[0], position[1], self.receiver_height_m)
        probe = replace(
            receiver,
            position=position,
        )
        # A device reporting offset is not additional received RF energy. Do
        # not feed this sensitivity offset into Network's detection model.
        offsets = {
            ap.bssid: self.receiver_bias_db + self.rssi_offsets_db.get(ap.bssid, 0.0)
            for ap in self.network.access_points
        }
        scan = observed_scan(
            self.network.scan(probe, rng),
            rssi_offsets_db=offsets,
            ftm_range_bias_m=self.ftm_range_bias_m,
        )
        if (self.single_ap_outage_probability
                and self._outage_rng.random() < self.single_ap_outage_probability):
            visible = [ap for ap, reading in scan.items() if reading is not None]
            if visible:
                scan[self._outage_rng.choice(visible)] = None
        return scan

    def description(self) -> str:
        if self.profile == "matched":
            return (
                "Matched survey/use channel and device; separate scan noise, "
                "same fixed spatial shadow field. Optimistic reference scenario."
            )
        offsets = ", ".join(
            f"{ap.name}={self.rssi_offsets_db[ap.bssid]:+.2f} dB"
            for ap in self.network.access_points
        )
        return (
            "Deployment sensitivity scenario (assumed, not field-calibrated): "
            f"independent shadow field (seed {self.network.shadowing_seed}); "
            "+4 dB persistent receiver reporting offset after detection; 1.3 m receiver height "
            "versus the 1.0 m survey/FTM assumption; per-AP reporting drift "
            "uniform [-3,+3] dB applied after detection "
            f"({offsets}); one visible AP removed on 10% of scans; "
            "+0.5 m common FTM range bias. Stable bias can survive every "
            "quality check; these results do not predict field accuracy."
        )


def create_evaluation_session(
    network: Network,
    profile: str = "matched",
    seed: int = 45678,
) -> EvaluationSession:
    """Clone channel state so evaluation cannot change the calibration model."""
    if profile not in {"matched", "deployment"}:
        raise ValueError("profile must be 'matched' or 'deployment'")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    # Dataclass init=False caches are recreated rather than shared by replace.
    evaluation_network = replace(
        network,
        access_points=list(network.access_points),
        receivers=list(network.receivers),
        shadowing_seed=network.shadowing_seed if profile == "matched" else seed + 100_003,
    )
    if profile == "matched":
        return EvaluationSession(evaluation_network)
    drift_rng = Random(seed + 200_003)
    offsets = {
        ap.bssid: drift_rng.uniform(-3.0, 3.0)
        for ap in sorted(network.access_points, key=lambda ap: ap.bssid)
    }
    return EvaluationSession(
        network=evaluation_network,
        profile=profile,
        receiver_bias_db=4.0,
        receiver_height_m=1.3,
        rssi_offsets_db=offsets,
        single_ap_outage_probability=0.1,
        ftm_range_bias_m=0.5,
        _outage_rng=Random(seed + 300_007),
    )
