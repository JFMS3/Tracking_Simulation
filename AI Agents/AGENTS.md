# Agent instructions (shared by Claude Code and Codex)

Simulation of Wi-Fi RSSI/RTT indoor positioning for tracking crew inside a
compartmented steel ship. It models a realistic stochastic radio channel, then
benchmarks several localisation algorithms against it. README.md documents the
propagation/noise model and the literature behind the parameter values; read it
before changing anything in `network.py`. All paths below are relative to the
repo root.

## Commands

Windows dev environment; use the `py` launcher. Run everything from the repo
root (`Tracking_Simulation/`, the parent of this `AI Agents/` folder), because modules import each other as flat top-level modules (there is
no package or `__init__.py`).

```powershell
py -m pip install -r requirements.txt   # shapely, PyYAML, matplotlib, numpy, scipy
py simulation.py                        # demo + benchmark tables, then opens 2 blocking matplotlib windows
py scenarios.py                         # all localisers x 9 scenarios, saves to results/scenarios/ (several minutes)
py tracking.py                          # crew-walking tracker benchmark, saves to results/tracking/
py fusion_benchmark.py                  # held-out sensor-fusion benchmark (30 walks, ~15 min), saves to results/fusion/
py -m unittest discover tests           # full test suite, ~3-4 min (tests/test_fusion_targets.py is ~3 min of it)
py -m unittest tests.test_network.NetworkNoiseModelTests.test_seeded_scan_sequences_are_reproducible
```

## Architecture

The data flows one way: `environment` → `network` → `positioning` / `fingerprinting` → `simulation`.

- `environment.py`: geometry only. `ShipEnvironment` holds `Compartment`
  polygons and `Wall` line segments (shapely). `walls_crossed(p1, p2)` is the
  basis of every NLOS effect. `from_config`/`from_dict` validate YAML strictly
  and raise `EnvironmentConfigError`.
- `network.py`: the radio channel. `Network` is a large dataclass of
  propagation parameters plus `AccessPoint`s and `Receiver`s. `scan(receiver, rng)`
  returns a `Scan` (`dict[AccessPoint, Optional[Reading]]`); `None` means the AP
  was not heard in that scan. RSSI = log-distance path loss − wall attenuation +
  receiver bias + spatial shadowing + averaged Rayleigh fading + measurement
  noise, followed by SNR-sigmoid detection and quantisation. RTT error is
  modelled in metres and then converted to milliseconds. Optional fixed per-AP
  RTT offsets (`rtt_ap_offset_std_m`, default 0 = off) come from
  `rtt_ap_offset(ap)`, hash-seeded on `(shadowing_seed, bssid)` like the shadow
  field, so they never consume the scan rng.
- `positioning.py`: localisers that follow the `Localiser` protocol
  (`locate(scan) -> Optional[LocalisationEstimate]`): `NearestAPLocaliser`,
  `WeightedCentroidLocaliser` (`"distance"` / `"linear_power"`), and
  `RSSTrilaterationLocaliser` / `RTTTrilaterationLocaliser` (scipy
  `least_squares` bounded to the ship's extent, needs ≥3 APs), and
  `WallAwareLocaliser` (`"rss"` / `"rtt"` / `"hybrid"`): a likelihood grid
  search whose forward model includes every crossed wall, plus non-detection
  evidence and optional per-scan phone-offset estimation. Its model parameters
  come from the `Network` it is given, so pass a differently configured
  Network to simulate miscalibration. `log_likelihood(scan)` exposes the
  per-grid-cell likelihood for the tracker. `rtt_nlos_mixture=True` (default
  off, which reproduces the old likelihood bit for bit) replaces the RTT
  Gaussian/Cauchy term with a Gentner-style two-component mixture
  (`rtt_mixture_*` keywords; tests read `_expected_range` / `_range_sigma`, so
  keep those names).
- `tracking.py`: `GridTracker`, a grid Bayes (HMM forward) filter over the
  `WallAwareLocaliser` grid. Its motion kernel is Gaussian in walking distance
  (Dijkstra on a wall-free neighbour graph), so belief crosses compartments
  only via doorways (gaps in walls). Wall-crossing tests everywhere go through
  `ShipEnvironment.crossing_matrix`, which agrees exactly with `walls_crossed`. Also `simulate_walk` (crew walking shortest wall-free
  routes) and `evaluate_tracking`. `py tracking.py` benchmarks it on the
  accommodation deck. `FixedLagSmoother(tracker, lag_steps)` wraps a
  `GridTracker` with exact fixed-lag forward-backward smoothing: `update`
  returns the estimate of epoch t - lag (None for the first `lag` epochs),
  `flush()` the rest at the end of a walk, and `delay_steps` lets
  `evaluate_tracking` align them. It relies on `GridTracker`'s
  `_predict_matrix`, `reacquire_probability` and `belief`; tests/test_smoother.py
  checks it against a brute-force oracle, so keep those in step if you change
  the tracker's HMM.
- `imu.py`: phone PDR simulator. `simulate_imu(walk, rng, model)` turns a true
  walk into `MotionReading`s (step count, the phone's step-length *model*
  output, gyro heading change, optional magnetometer heading). Distance
  information comes only from quantised step counts; never put truth into a
  reading (tests/test_imu.py guards this). `ImuModel` error fields follow a
  naming rule (contain noise/bias/probability/per_s/std/error) so zeroing them
  gives a perfect phone; defaults cite `research_notes/`, and ship-specific
  ones are labelled assumptions. `IMU_PRESETS` has optimistic/default/conservative.
- `fusion.py`: `ParticleTracker`, a map-matching particle filter fusing the
  WallAwareLocaliser likelihood with PDR (state x, y, latent heading,
  step-length scale; wall-crossing moves killed except two-leg door-jamb
  routes; ESS resampling; augmented-MCL injection). Tuned defaults are marked
  `[T]` and were tuned on tuning seeds only.
- `docs/sensor_fusion_spec.md`: interfaces, targets T1-T7, validation checks
  V1-V8 and the seed discipline (tuning seeds vs held-out seeds; never tune
  on held-out seeds). Documented `expectedFailure`s (T6c, V4b) are recorded
  shortfalls, not tests to "fix" by loosening.
- `scenarios.py`: runs every localiser across several scenarios, each with a
  separate *truth* channel, *model* (localisers' belief) and *survey* network
  (fingerprint map recorded under a different shadowing seed). Saves tables,
  `summary.csv` and plots to `results/scenarios/<timestamp>/`.
- `fingerprinting.py`: `build_radio_map` simulates an offline calibration survey
  on a grid. `FingerprintLocaliser` does weighted k-NN in signal space.
  `FingerprintCompartmentLocaliser` exposes `classify(scan) -> compartment name`
  instead of `locate`, so it can't go in the `localisers` dict.
- `simulation.py`: entry point. Loads `layouts/simple_layout2.yaml`, places 5
  APs and 2 receivers in code, prints a fused per-timestamp compartment guess,
  then prints coverage, localiser benchmark, and per-compartment accuracy
  tables, and plots heatmaps.
- `layouts/*.yaml`: `compartments`, `walls`, and an optional `propagation`
  block. `simple_layout2.yaml` (used by `simulation.py`) and
  `accommodation_deck.yaml` (24x10 m deck with corridor, doorways as wall
  gaps, 8 dB joiner panels) have `propagation`.

## Conventions and gotchas

- **Config → constructor coupling:** `Network.from_config` passes every
  `propagation` key straight through as `Network(**propagation)`. A new `Network`
  field becomes configurable automatically, and an unknown or misspelled YAML key
  raises `TypeError`. Validate new parameters in `Network.__post_init__`
  alongside the existing positive / non-negative / probability checks.
- **Reproducibility is a design requirement.** All randomness flows through an
  explicitly passed `random.Random`; never use the global `random` or
  `np.random`. `simulation.main` uses separate seeded streams for each phase
  (calibration, demo, coverage, performance, breakdown) and calls
  `network.reset_channel_state()` between phases. Benchmarks must feed
  *identical* scans to every localiser (paired comparison); see
  `report_localiser_stats`.
- The shadowing field is deterministic per `(shadowing_seed, bssid, frequency)`
  and cached by receiver position, so it is fixed in space rather than redrawn on
  each scan. Interference-burst state is per frequency and persists across scans
  until it is reset.
- `Reading`'s first five fields (`rssi_dbm, link_speed_mbps, tx_mbps, rx_mbps,
  rtt_ms`) are kept first for backward compatibility. Add new diagnostics as
  optional fields after them.
- `AccessPoint` is hashable (`unsafe_hash=True`) because it is used as a dict
  key in `Scan`, so don't mutate APs after construction.
- `legacy_wall_dropout` / `Wall.dropout_prob` exist only to compare against the
  old model. Normal reception loss comes from attenuation plus SNR detection.
- Fingerprint vectors fill unheard APs with `DEFAULT_FLOOR_DBM = -100`.
- Positions: APs/receivers are 3D `(x, y, z)` in metres; geometry and estimates
  are 2D `(x, y)` (`.xy`). Multi-deck support exists only as a `deck` label.
- Parameter values are provisional literature-based defaults (see README). If
  you change them, justify the change from measurements or citations, never by
  tuning until a localiser looks good.
- Tests disable all noise via `make_network(**overrides)` in
  `tests/test_network.py`; follow that pattern so tests stay deterministic.
- Git remote: `github.com/JFMS3/Tracking_Simulation`. Work happens on feature
  branches merged via PR into `main`.
