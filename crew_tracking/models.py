"""Measurement contracts shared by hardware adapters and localisation algorithms.

This module contains no simulated channel state or ground-truth diagnostics.
Coordinates and distances are in metres; FTM round-trip durations are nanoseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

Coord = tuple[float, float]
Coord3D = tuple[float, float, float]
SPEED_OF_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class AccessPoint:
    """Surveyed anchor; identity/position are immutable because scans key on it.

    ``ftm_responder`` describes configured ranging capability, not reception.
    A real adapter must obtain this from its platform's capability discovery.
    """

    name: str
    bssid: str
    position: Coord3D
    frequency_mhz: int
    ftm_responder: bool = True

    def __post_init__(self) -> None:
        if not self.bssid or not isinstance(self.bssid, str):
            raise ValueError("bssid must be a non-empty string")
        if len(self.position) != 3 or not all(math.isfinite(x) for x in self.position):
            raise ValueError("AP position must contain three finite coordinates")
        object.__setattr__(self, "position", tuple(self.position))
        if not isinstance(self.ftm_responder, bool):
            raise ValueError("ftm_responder must be a boolean")

    @property
    def xy(self) -> Coord:
        return self.position[0], self.position[1]


@dataclass(frozen=True)
class FTMSample:
    """One corrected propagation RTT, after removing station ACK turnaround.

    Failed or malformed device results may be represented without throwing;
    the burst processor screens them. Signed RTTs are preserved until bias
    calibration, since timing noise can produce a negative measured range.
    """

    rtt_ns: Optional[float]
    successful: bool = True


@dataclass(frozen=True)
class FTMBurst:
    """Exchanges from a single AP while the receiver is approximately static."""

    samples: tuple[FTMSample, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(self.samples))


@dataclass
class Reading:
    """Observed radio data, independently consumable by each localiser.

    Legacy ``rtt_ms`` may contain a configured artificial processing offset.
    FTM uses only ``ftm_burst`` and never interprets network ping latency as
    propagation time. RSSI can be NaN for an FTM-only measurement.
    """

    rssi_dbm: float
    link_speed_mbps: float = 0.0
    tx_mbps: float = 0.0
    rx_mbps: float = 0.0
    rtt_ms: Optional[float] = None
    ftm_burst: Optional[FTMBurst] = None


Scan = dict[AccessPoint, Optional[Reading]]


@dataclass
class LocalisationEstimate:
    position: Coord
    covariance: Optional[np.ndarray] = None
    num_aps_used: int = 0


class Localiser(Protocol):
    def locate(self, scan: Scan) -> Optional[LocalisationEstimate]:
        """Return an estimate, or None when observations cannot support one."""
        ...


@dataclass(frozen=True)
class PathLossModel:
    """Calibrated parameters for RSSI algorithms, independent of a simulator."""

    reference_rssi_dbm: float = -40.0
    reference_distance_m: float = 1.0
    path_loss_exponent: float = 2.15

    def __post_init__(self) -> None:
        if not math.isfinite(self.reference_rssi_dbm):
            raise ValueError("reference_rssi_dbm must be finite")
        for name in ("reference_distance_m", "path_loss_exponent"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
