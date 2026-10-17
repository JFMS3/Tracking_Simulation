# Sensor fusion: specification, targets and test plan

Goal: track crew on a compartmented steel deck by fusing Wi-Fi (RSSI + RTT,
already fused in `WallAwareLocaliser(measurement="hybrid")`) with phone
inertial sensing (pedestrian dead reckoning, PDR), using methods with
published support, and prove it with tests that state expected results up
front.

Read `AI Agents/AGENTS.md` first. Its rules apply, in particular: all
randomness through an explicitly passed `random.Random` (a numpy Generator
seeded from it is fine; never global `random` / `np.random`), paired
comparisons, and **never tune simulator channel parameters to make a method
look good**.

## Components and file ownership

| Component | File | Interface (fixed; do not change without the coordinator) |
|---|---|---|
| PDR / IMU simulation | `imu.py` | `MotionReading`, `ImuModel`, `simulate_imu(walk, rng, model, scan_interval_s) -> List[MotionReading]` |
| Fusion particle filter | `fusion.py` | `ParticleTracker(localiser, network, rng, n_particles, imu_model, use_magnetometer, scan_interval_s)`, `.reset()`, `.update(scan, motion)` |
| Fixed-lag smoother | `tracking.py` | `FixedLagSmoother(tracker, lag_steps)`, `.reset()`, `.update(scan, motion)`, `.flush()`, `.delay_steps` |
| RTT NLOS mixture likelihood | `positioning.py` | new `WallAwareLocaliser(..., rtt_nlos_mixture: bool = False, ...)`; default off keeps current behaviour |
| Per-AP RTT offsets (realism) | `network.py` | new `Network.rtt_ap_offset_std_m: float = 0.0`; a fixed per-AP range offset drawn deterministically from (shadowing_seed, bssid), like the shadow field. Default 0 keeps current behaviour. Validate in `__post_init__`. |
| Evaluation harness | `tracking.py` | `evaluate_tracking(trackers, network, walks, rng, motions=None, scan_masks=None)` (already implemented: Wi-Fi outages via `scan_masks`, motion feed, delayed trackers, `coverage95`) |
| Benchmark | `fusion_benchmark.py` | `py fusion_benchmark.py` prints/saves tables and plots to `results/fusion/<timestamp>/` |

Tracker protocol: `reset()`, `update(scan, motion) -> Optional[LocalisationEstimate]`.
`scan` is `None` during a Wi-Fi outage; `motion` is the `MotionReading` for
the interval ending at this epoch, or `None` (no IMU). Estimates should carry
a 2x2 `covariance` so uncertainty calibration can be validated.

### PDR model requirements (`imu.py`)
- Reading t covers the move from walk position t-1 to t; reading 0 is no motion.
- Step count from true distance / true step length, with missed and false
  steps (ship motion can trigger false steps).
- Step length estimate with per-step noise (and optionally a per-walk bias).
- Gyro heading change = true change + per-walk constant bias x dt + angle random walk.
- Magnetometer (off by default): absolute heading with a large, spatially
  varying bias inside a steel hull.
- Every default value cited in a comment (sources: `research_notes/`).
- MotionReading must never contain the truth.

### Particle filter requirements (`fusion.py`)
- State per particle: x, y, heading (gyro gives only *relative* heading, so the
  absolute heading is a latent state, initially unknown), optionally a step-length scale.
- Predict: move by motion.distance along heading + motion.heading_change, with
  noise from `imu_model`; with no motion, diffuse like a stationary/slow walker.
- Walls are hard constraints: a particle whose move crosses a wall
  (`ShipEnvironment.crossing_matrix`) is killed (weight 0) or rejected. Revision: a move
  is also accepted if a wall-free two-leg route around a door jamb exists that is
  at most `corner_slack_m` longer (walkers turn through doors within one scan
  interval; mirrors GridTracker's walking-distance kernel).
- Update: weight by `localiser.log_likelihood(scan)` looked up at each
  particle's grid cell; skip when scan is None.
- Resample on low effective sample size; inject a small fraction of particles
  from the Wi-Fi-only posterior (recovery from lock-in / kidnapping).
- Initialise from the first scan's Wi-Fi posterior (not a local window).
- Estimate: weighted mean within the dominant compartment, with covariance.

## Seeds: tuning vs held-out validation

Algorithm parameters may be tuned only on **tuning seeds**: walks
`Random(67890)`, scans `Random(78901)`, IMU `Random(89012)`. All targets are
judged on **held-out seeds**: walks `Random(2024 + i)` for i in 0..7, scans
`Random(3024)`, IMU `Random(4024)`, particles `Random(5024)`.

Confirmatory set (added 2026-10-17 after review): 8 walks cannot reach
significance after Holm correction (smallest exact Wilcoxon p = 0.0078), and
T6b was defined after the 8 held-out walks were seen. Final statistics are
therefore reported on **30 walks** `Random(2024 + i)`, i in 0..29; walks
i >= 8 (seeds 2032-2053) had never been run, so T6b is confirmatory only on
those 22 fresh walks.

## Baseline (held-out seeds, 8 walks x 300 s, GridTracker hybrid robust, grid 0.2 m)

| Deck scenario | mean | P90 | room acc | room-change ratio | coverage95 |
|---|---|---|---|---|---|
| Accommodation deck | 0.456 m | 0.83 m | 99.79% | 1.01 | 0.98 |
| Deck, miscalibrated | 0.658 m | 1.38 m | 98.83% | 1.05 | 0.88 |
| Deck, harsh channel | 0.659 m | 1.19 m | 98.88% | 1.09 | 0.91 |
| Deck + 15 s outage at t=100 s | 0.631 m (whole walk) | 0.97 m | 98.54% | 1.01 | 0.95 |

## Targets (expected results)

"PF" = `ParticleTracker` with hybrid Wi-Fi + PDR (gyro only, no magnetometer,
realistic `ImuModel` defaults). "Grid" = `GridTracker` hybrid robust.
Relative targets compare methods on identical scans and IMU readings.

| ID | Claim | Target |
|---|---|---|
| T1 | Fusion beats Wi-Fi-only tracking (deck) | PF mean error <= 0.85 x Grid; room acc >= Grid; room-change ratio <= 1.05 |
| T2 | Holds under model miscalibration | PF mean <= 0.85 x Grid on "Deck, miscalibrated"; room acc >= Grid |
| T3 | Holds in the harsh channel | PF mean <= 0.85 x Grid on "Deck, harsh channel"; room acc >= Grid |
| T4 | PDR bridges Wi-Fi outages | 15 s outage: PF mean error *during the outage* <= 1.0 m, room acc during outage >= 95%, and < Grid's outage error |
| T5 | Fixed-lag smoothing helps | `FixedLagSmoother(Grid, 5)`: mean <= 0.85 x Grid forward; room acc >= forward; ratio <= forward |
| T6 | Magnetometer is unsafe in steel | (a) PF with biased magnetometer (`use_magnetometer=True`) is worse than gyro-only PF; (b) gyro-only PF within 15% of the same PF given an ideal (unbiased) compass; (c) as originally written: gyro-only PF within 15% of PF given a perfect IMU |
| T7 | RTT mixture helps realistic RTT | With `rtt_ap_offset_std_m=1.5` and realistic RTT errors, hybrid snapshot with `rtt_nlos_mixture=True` has P90 <= 0.95 x without, mean not worse than +2% |

T6 revision (2026-10-17, after the held-out run): clause (c) conflated "no
absolute heading" with "all PDR errors removed". It is kept as a recorded
known shortfall (`expectedFailure`, gyro 0.310 m vs perfect IMU 0.173 m),
and (b) was added as the well-posed heading test (gyro 0.310 m vs ideal
compass 0.299 m, ratio 1.04).

Literature basis for the size of the targets (see `research_notes/`): map +
PDR fusion improved mean error 18-25% over map-constrained RF in office
trials; smoothing improved 34-60% over online filtering against weaker
baselines; Gaussian-mixture RTT likelihoods improved PF error by ~0.45 m.
Targets are deliberately below those figures because our baseline is
already strong.

## Validation tests (the simulation and the methods behave as claimed)

| ID | Check |
|---|---|
| V1 | `simulate_imu` statistics match `ImuModel`: step-length error mean/std, miss and false-step rates, gyro heading error variance grows ~linearly with time (angle random walk), within statistical tolerance |
| V2 | PDR alone drifts: dead-reckoned position error after 120 s exceeds 2 m (so the IMU is not an oracle) |
| V3 | PF invariants: weights finite and normalised, every accepted particle move has a wall-free route (straight, or two legs via a door-jamb point within `corner_slack_m`; so compartments change only through doorways), deterministic for fixed seeds, reset() makes walks independent of order |
| V4 | Uncertainty calibration: coverage95 of PF and smoother in [0.85, 0.99] on the deck |
| V5 | Null/ideal cases: perfect IMU + known start => PF error small (< 0.3 m); all-APs-missing scans => belief stays within reachable area; no IMU (motion None) => PF still tracks (no worse than 1.2 x Grid) |
| V6 | Paired evaluation: identical scans/IMU to all trackers; two identical PF instances with equal seeds give identical stats |
| V7 | Regressions: all pre-existing tests keep passing; RTT mixture and per-AP offsets default off reproduce previous results exactly |
| V8 | Held-out discipline: target tests use only held-out seeds |

## Test files

- `tests/test_imu.py`: interface + V1, V2
- `tests/test_fusion.py`: PF unit behaviour + V3, V5, V6
- `tests/test_smoother.py`: smoother correctness (exact forward-backward on a small grid vs brute force) + delay/flush alignment
- `tests/test_measurement_fusion.py`: RTT mixture + per-AP offsets + V7
- `tests/test_fusion_targets.py`: T1-T7, V4, V8 on held-out seeds, sized to run in <= 3 minutes total

If a target is not met after honest effort, do not weaken it: mark the test
`@unittest.expectedFailure` with the measured numbers in a comment and report it.
