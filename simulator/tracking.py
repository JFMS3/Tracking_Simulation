"""Paired temporal benchmarks; truth is used only for journeys and scoring.

The vessel layout has no doorway metadata. Journeys therefore stay inside one
compartment and use different device IDs, rather than inventing wall crossings.
These simulated scans do not reproduce the full temporal correlation of real
Wi-Fi, so a gain here is evidence of implementation behaviour, not field accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import Sequence

import numpy as np
from shapely.geometry import box

from crew_tracking.models import Coord, LocalisationEstimate, Localiser
from crew_tracking.tracking import KalmanTracker, TrackEstimate

from .environment import Compartment, ShipEnvironment
from .evaluation import EvaluationSession
from .network import Network, Receiver


@dataclass(frozen=True)
class JourneyPoint:
    device_id: str
    timestamp_s: float
    elapsed_s: float
    truth: Coord
    compartment_name: str
    phase: str
    phase_age_s: float
    stationary_id: int | None = None


@dataclass(frozen=True)
class TrackingFix:
    """One radio fix cached once before raw/filtered comparisons."""

    journey: JourneyPoint
    measurement: LocalisationEstimate | None


@dataclass(frozen=True)
class TrackingSample:
    device_id: str
    timestamp_s: float
    elapsed_s: float
    truth: Coord
    compartment_name: str
    phase: str
    phase_age_s: float
    stationary_id: int | None
    forced_outage: bool
    raw: LocalisationEstimate | None
    filtered: TrackEstimate | None


@dataclass(frozen=True)
class TrackingMetrics:
    total_count: int
    available_count: int
    compartment_accuracy: float
    stationary_jitter_m: float | None
    stationary_error_m: float | None
    turn_mean_error_m: float | None
    prediction_count: int
    stale_count: int
    recovery_delay_s: float | None
    recovery_count: int
    recovery_total: int
    prediction_only_count: int
    prediction_only_mean_m: float | None
    prediction_only_p90_m: float | None

    @property
    def availability(self) -> float:
        return self.available_count / self.total_count if self.total_count else 0.0


@dataclass(frozen=True)
class TrackingScenarioResult:
    name: str
    samples: tuple[TrackingSample, ...]
    raw: TrackingMetrics
    kalman: TrackingMetrics
    paired_count: int
    paired_raw_mean_m: float | None
    paired_raw_p90_m: float | None
    paired_kalman_mean_m: float | None
    paired_kalman_p90_m: float | None
    paired_stationary_count: int
    paired_turn_count: int


@dataclass(frozen=True)
class TrackingBenchmarkResult:
    profile: str
    seed: int
    scenarios: tuple[TrackingScenarioResult, ...]


@dataclass(frozen=True)
class _Segment:
    start_s: float
    end_s: float
    start: Coord
    end: Coord
    stationary_id: int | None
    follows_turn: bool = False


def _inset_corners(compartment: Compartment) -> tuple[Coord, Coord, Coord, Coord]:
    polygon = compartment.geometry
    centre = polygon.representative_point()
    left, bottom, right, top = polygon.bounds
    half_width, half_height = (right - left) * 0.3, (top - bottom) * 0.3
    for _ in range(60):
        x0, x1 = centre.x - half_width, centre.x + half_width
        y0, y1 = centre.y - half_height, centre.y + half_height
        if polygon.covers(box(x0, y0, x1, y1)):
            return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        half_width *= 0.75
        half_height *= 0.75
    raise ValueError(f"Cannot construct an internal journey for {compartment.name!r}")


def generate_journeys(environment: ShipEnvironment) -> tuple[JourneyPoint, ...]:
    """Deterministic stops and right-angle walks at 0.75 m/s with irregular scans.

    Each route starts at a positive timestamp and gets its own crew/device ID.
    The 60%-of-bounds rectangle is shrunk for nonrectangular compartments.
    This is an exercise path inside surveyed space, not an inferred doorway map.
    """
    if not environment.compartments:
        raise ValueError("Tracking benchmarks require at least one compartment")
    points = []
    for route_index, compartment in enumerate(environment.compartments):
        a, b, c, d = _inset_corners(compartment)
        segments: list[_Segment] = []
        elapsed = 0.0
        stationary_id = 0

        def hold(position: Coord, seconds: float) -> None:
            nonlocal elapsed, stationary_id
            segments.append(_Segment(elapsed, elapsed + seconds, position, position,
                                     stationary_id))
            elapsed += seconds
            stationary_id += 1

        def walk(start: Coord, end: Coord, *, turn: bool = False) -> None:
            nonlocal elapsed
            duration = math.dist(start, end) / 0.75
            segments.append(_Segment(elapsed, elapsed + duration, start, end, None, turn))
            elapsed += duration

        hold(a, 12.0)
        walk(a, b)
        walk(b, c, turn=True)
        hold(c, 8.0)
        walk(c, d)
        walk(d, a, turn=True)
        hold(a, 8.0)
        while elapsed < 65.0:
            walk(a, b)
            walk(b, c, turn=True)
            walk(c, d, turn=True)
            walk(d, a, turn=True)
        hold(a, 12.0)

        scan_time = 0.0
        segment_index = 0
        scan_index = 0
        intervals = (0.55, 0.85, 0.65, 1.0, 0.7)
        while scan_time <= elapsed:
            while (segment_index + 1 < len(segments)
                   and scan_time >= segments[segment_index].end_s):
                segment_index += 1
            segment = segments[segment_index]
            fraction = (scan_time - segment.start_s) / (segment.end_s - segment.start_s)
            position = tuple(
                start + fraction * (end - start)
                for start, end in zip(segment.start, segment.end)
            )
            age = scan_time - segment.start_s
            phase = "stationary" if segment.stationary_id is not None else (
                "turn" if segment.follows_turn and age <= 2.0 else "walking"
            )
            points.append(JourneyPoint(
                f"route-{route_index + 1}", 1000.0 + scan_time, scan_time,
                position, compartment.name, phase, age, segment.stationary_id,
            ))
            scan_time += intervals[scan_index % len(intervals)]
            scan_index += 1
    return tuple(points)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _p90(values: Sequence[float]) -> float | None:
    return float(np.percentile(values, 90)) if len(values) else None


def evaluate_tracking_fixes(
    fixes: Sequence[TrackingFix],
    environment: ShipEnvironment,
    *,
    name: str = "radio",
    delivery_outages: Sequence[tuple[float, float]] = (),
) -> TrackingScenarioResult:
    """Replay identical delivered observations through raw and default trackers.

    Deliberate outages suppress position delivery after localisation. They are
    explicit robustness tests, not additional radio propagation assumptions.
    Neither ground truth nor uncalibrated localiser covariance reaches tracking.
    """
    for start, end in delivery_outages:
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("Outage intervals must have finite 0 <= start < end")
    tracker = KalmanTracker()
    samples = []
    for fix in fixes:
        point = fix.journey
        outage = any(start <= point.elapsed_s < end for start, end in delivery_outages)
        measurement = None if outage else fix.measurement
        # A real service knows identity, time and the delivered coordinate fix.
        # It does not know the true journey phase, position, speed or room.
        filtered = tracker.update(point.device_id, point.timestamp_s, measurement)
        samples.append(TrackingSample(
            point.device_id, point.timestamp_s, point.elapsed_s, point.truth,
            point.compartment_name, point.phase, point.phase_age_s, point.stationary_id,
            outage, measurement, filtered,
        ))
    paired = [sample for sample in samples
              if sample.raw is not None and sample.filtered is not None]
    stationary = [sample for sample in paired
                  if sample.phase == "stationary" and sample.phase_age_s >= 3.0]
    turning = [sample for sample in paired if sample.phase == "turn"]

    def metrics(filtered: bool) -> TrackingMetrics:
        def estimate(sample: TrackingSample):
            return sample.filtered if filtered else sample.raw

        groups: dict[tuple[str, int], list[Coord]] = {}
        for sample in stationary:
            key = (sample.device_id, sample.stationary_id)
            groups.setdefault(key, []).append(estimate(sample).position)
        deviations = []
        for positions in groups.values():
            centre = np.mean(positions, axis=0)
            deviations.extend(float(np.sum((np.asarray(position) - centre) ** 2))
                              for position in positions)
        recoveries = []
        recovery_total = 0
        for device in dict.fromkeys(sample.device_id for sample in samples):
            device_samples = [sample for sample in samples if sample.device_id == device]
            for start, end in delivery_outages:
                if not any(start <= sample.elapsed_s < end for sample in device_samples):
                    continue
                recovery_total += 1
                recovered = next((sample for sample in device_samples
                                  if sample.elapsed_s >= end and estimate(sample) is not None
                                  and (not filtered or sample.filtered.measurement_age_s == 0)),
                                 None)
                if recovered is not None:
                    recoveries.append(recovered.elapsed_s - end)
        available = [sample for sample in samples if estimate(sample) is not None]
        gap_errors = [math.dist(sample.truth, sample.filtered.position)
                      for sample in samples if filtered and sample.raw is None
                      and sample.filtered is not None]
        correct = sum(environment.compartment_at(estimate(sample).position)
                      == sample.compartment_name for sample in available)
        # A stale event is an unavailable track after its device had an estimate.
        seen = set()
        stale = 0
        for sample in samples:
            if estimate(sample) is not None:
                seen.add(sample.device_id)
            elif filtered and sample.device_id in seen:
                stale += 1
        return TrackingMetrics(
            len(samples), len(available), correct / len(samples) if samples else 0.0,
            math.sqrt(float(np.mean(deviations))) if deviations else None,
            _mean([math.dist(sample.truth, estimate(sample).position)
                   for sample in stationary]),
            _mean([math.dist(sample.truth, estimate(sample).position) for sample in turning]),
            sum(sample.filtered.measurement_age_s > 0 for sample in available) if filtered else 0,
            stale, _mean(recoveries), len(recoveries), recovery_total,
            len(gap_errors), _mean(gap_errors), _p90(gap_errors),
        )

    raw_errors = [math.dist(sample.truth, sample.raw.position) for sample in paired]
    kalman_errors = [math.dist(sample.truth, sample.filtered.position) for sample in paired]
    return TrackingScenarioResult(
        name, tuple(samples), metrics(False), metrics(True), len(paired),
        _mean(raw_errors), _p90(raw_errors), _mean(kalman_errors), _p90(kalman_errors),
        len(stationary), len(turning),
    )


def report_tracking_benchmark(
    localiser: Localiser,
    network: Network,
    evaluation_session: EvaluationSession,
    seed: int = 45678,
) -> TrackingBenchmarkResult:
    """Scan each journey point once, cache Fusion fixes, then compare replays."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = Random(seed)
    fixes = []
    previous_device = None
    for point in generate_journeys(network.environment):
        if point.device_id != previous_device:
            # Routes are independent sessions, not simultaneous vessel-wide
            # sampling. Do not carry interference bursts across their clocks.
            evaluation_session.network.reset_channel_state()
            previous_device = point.device_id
        scan = evaluation_session.scan(Receiver(point.device_id, (*point.truth, 1.0)), rng)
        fixes.append(TrackingFix(point, localiser.locate(scan)))
    scenarios = (
        evaluate_tracking_fixes(fixes, network.environment),
        evaluate_tracking_fixes(fixes, network.environment, name="delivery outages",
                                delivery_outages=((20.0, 22.0), (42.0, 55.0))),
    )
    result = TrackingBenchmarkResult(evaluation_session.profile, seed, scenarios)
    print("=== Temporal tracking: identical Fusion fixes, default Kalman parameters ===")
    print("Independent within-compartment journeys: stops, 0.75 m/s walks and right-angle turns.")
    print("No door metadata or inertial input; wall crossing and doorway delay are not validated.")
    print("Outage replay suppresses fix delivery at 20-22 s and 42-55 s; radio inputs stay identical.")
    print("Stationary and turn scores use common returned fixes; stationary scoring starts 3 s after a stop.")

    def number(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "--"

    for scenario in scenarios:
        print(f"\n{scenario.name}: {scenario.paired_count}/{len(scenario.samples)} common fixes; "
              f"stationary n={scenario.paired_stationary_count}, turn n={scenario.paired_turn_count}")
        print(f"{'Method':<18}{'Mean/P90 (m)':<19}{'Returned':<12}{'Room acc':<12}"
              f"{'Stop jitter/error (m)':<25}{'Turn err (m)':<15}{'Predicted':<12}{'Stale':<9}"
              f"{'Gap pred mean/P90 (m)':<25}{'Recovery (s)'}")
        for label, metric, mean, p90 in (
            ("Raw Fusion", scenario.raw, scenario.paired_raw_mean_m, scenario.paired_raw_p90_m),
            ("Fusion + Kalman", scenario.kalman, scenario.paired_kalman_mean_m,
             scenario.paired_kalman_p90_m),
        ):
            error = f"{number(mean)}/{number(p90)}"
            jitter = f"{number(metric.stationary_jitter_m)}/{number(metric.stationary_error_m)}"
            recovery = (f"{number(metric.recovery_delay_s)} "
                        f"({metric.recovery_count}/{metric.recovery_total})"
                        if metric.recovery_total else "--")
            gap = f"{number(metric.prediction_only_mean_m)}/{number(metric.prediction_only_p90_m)}"
            print(f"{label:<18}{error:<19}{metric.availability:<12.1%}"
                  f"{metric.compartment_accuracy:<12.1%}{jitter:<25}"
                  f"{number(metric.turn_mean_error_m):<15}{metric.prediction_count:<12}"
                  f"{metric.stale_count:<9}{gap:<25}{recovery}")
        print(f"Prediction-only error uses {scenario.kalman.prediction_only_count} points "
              "with no raw fix and a returned Kalman prediction.")
    print("\nMean/P90 exclude unpaired points; room accuracy includes unavailable points as errors.")
    print("Jitter is RMS around each stop's estimated centroid, separate from absolute position error.")
    print("Recovery is time after delivery resumes until a fresh fix is accepted, including scan timing.")
    print("Assumed radio/motion errors do not establish field accuracy; stable bias can survive filtering.\n")
    return result
