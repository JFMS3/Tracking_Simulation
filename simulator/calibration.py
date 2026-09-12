"""Generate synthetic fingerprint surveys using the simulation channel."""

from __future__ import annotations

import math
from random import Random
from typing import Sequence

import numpy as np
from shapely.geometry import Point

from crew_tracking.fingerprinting import DEFAULT_FLOOR_DBM, FingerprintEntry
from crew_tracking.models import AccessPoint
from .network import Network, Receiver


def build_radio_map(
    network: Network,
    rng: Random,
    ap_order: Sequence[AccessPoint],
    grid_spacing_m: float = 1.0,
    samples_per_point: int = 30,
    floor_dbm: float = DEFAULT_FLOOR_DBM,
    *,
    receiver_height_m: float = 1.0,
) -> list[FingerprintEntry]:
    """Simulate the RSSI survey that a real deployment would collect onsite."""
    if not math.isfinite(grid_spacing_m) or grid_spacing_m <= 0:
        raise ValueError("grid_spacing_m must be finite and positive")
    if (isinstance(samples_per_point, bool) or not isinstance(samples_per_point, int)
            or samples_per_point <= 0):
        raise ValueError("samples_per_point must be a positive integer")
    if not math.isfinite(floor_dbm) or not math.isfinite(receiver_height_m):
        raise ValueError("RSSI floor and receiver height must be finite")
    if not ap_order or len(set(ap_order)) != len(ap_order):
        raise ValueError("ap_order must contain unique access points")

    radio_map: list[FingerprintEntry] = []
    for comp in network.environment.compartments:
        minx, miny, maxx, maxy = comp.geometry.bounds
        nx = max(int((maxx - minx) / grid_spacing_m), 1)
        ny = max(int((maxy - miny) / grid_spacing_m), 1)
        for i in range(nx + 1):
            for j in range(ny + 1):
                x, y = minx + i * grid_spacing_m, miny + j * grid_spacing_m
                if not comp.geometry.covers(Point(x, y)):
                    continue
                accum = np.zeros(len(ap_order))
                valid_counts = np.zeros(len(ap_order))
                receiver = Receiver("calib", (x, y, receiver_height_m))
                for _ in range(samples_per_point):
                    # A fingerprint survey consumes RSSI only. Avoid drawing
                    # unused FTM bursts or changing its historical RNG stream.
                    scan = network.scan(receiver, rng, include_ftm=False)
                    for index, ap in enumerate(ap_order):
                        reading = scan.get(ap)
                        if reading is not None and math.isfinite(reading.rssi_dbm):
                            accum[index] += reading.rssi_dbm
                            valid_counts[index] += 1
                vector = np.full(len(ap_order), floor_dbm, dtype=float)
                np.divide(accum, valid_counts, out=vector, where=valid_counts > 0)
                radio_map.append(FingerprintEntry(
                    position=(x, y), compartment=comp.name,
                    rssi_vector=vector, n_samples=samples_per_point,
                ))
    return radio_map
