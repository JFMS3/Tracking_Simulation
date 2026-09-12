# Crew Tracking

This project provides reusable RSSI, fingerprint, and Wi-Fi FTM positioning
algorithms, plus a separate simulator for a compartmented metallic ship.
FTM processes four-timestamp exchanges and bursts, applies calibrated range
biases, and fits a robust weighted position using surveyed AP heights.

## Structure

```text
crew_tracking/           Algorithms: no simulator, geometry, YAML, or plot imports
  models.py             Measurement contracts, APs, estimates, path-loss parameters
  ftm.py                Timestamp conversion, burst processing, weighted positioning
  positioning.py        Nearest AP, centroids, RSS and legacy RTT baselines
  fingerprinting.py     Radio-map entries and fingerprint estimators
simulator/              Synthetic measurements and survey generation
  environment.py        Ship geometry and YAML layout loading
  network.py            Channel, receiver truth, diagnostics, synthetic FTM bursts
  calibration.py        Simulated RSSI radio-map survey
simulation.py           Composition, reports, plots, command-line entry point
layouts/                Simulation configuration
tests/                  Algorithm, channel, integration, and dependency tests
docs/algorithm.md       Algorithm equations, usage, assumptions, and literature
```

`simulator` imports `crew_tracking`; the reverse is prohibited and tested.
Production algorithms consume observed data and explicit configuration.
They never receive `Network`, `ShipEnvironment`, true receiver positions,
wall counts, or simulated NLOS labels.

## Run

Use Python 3.10 or newer. From this directory:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[simulation]"
python simulation.py
```

For an existing environment with `requirements.txt` installed, the existing
`py simulation.py` command also works. A quick headless run and the test suite:

```powershell
python simulation.py --no-plots --trials 20 --grid-spacing 1 --calibration-samples 3
python -B -m unittest discover -s tests -v
```

Install just the algorithms into another Python application with
`python -m pip install .` (only NumPy and SciPy are runtime requirements).

The script and its layout can also be launched by absolute path from another
working directory. CLI defaults retain the original 300 trials per compartment,
0.3 m calibration spacing, and 15 calibration scans per point.

## FTM algorithm

```python
from crew_tracking.ftm import FTMExchange, FTMLocaliser
from crew_tracking.models import AccessPoint, FTMBurst, Reading

ap = AccessPoint("AP1", "00:11:22:33:44:55", (0.0, 0.0, 2.5), 5180)
# Example timestamps in ns; t1/t4 share the AP clock, t2/t3 the station clock.
sample = FTMExchange(t1_ns=0, t2_ns=1_000_010,
                     t3_ns=1_050_010, t4_ns=50_020).to_sample()
reading = Reading(rssi_dbm=-55.0, ftm_burst=FTMBurst((sample, sample, sample)))
localiser = FTMLocaliser(receiver_height_m=1.0, bounds=((0, 0), (10, 7)))
# Collect actual independent exchanges from >=3 noncollinear surveyed APs.
estimate = localiser.locate({ap: reading})  # None: one AP cannot locate x/y.
```

Real integrations supply measured exchanges, or individual hardware-corrected
propagation RTTs as `FTMSample(rtt_ns=...)`. These APIs do not send Wi-Fi frames.
Do not substitute ping RTT, apply the legacy 8 ms simulation offset, or expand
a platform's one aggregate result into invented independent samples. Device
discovery, ranging requests, stale-result filtering, counter unwrapping and
clock-rate calibration belong in a platform adapter. This repository implements
the measurement-processing algorithm and a synthetic input source.

The benchmark reports `FTM` and `Legacy RTT` separately. FTM requires at least
three valid exchanges from each of three suitable APs by default. Its assumed
receiver height is 1 m; the demonstration phone at 0.7 m intentionally has a
height mismatch. Burst filtering and robust loss reduce isolated errors but do
not remove every persistent NLOS bias. See [algorithm notes](docs/algorithm.md)
for equations, calibration, covariance limitations, and verified references.

## Migration from the original flat modules

- Import estimators from `crew_tracking.positioning` or
  `crew_tracking.fingerprinting`; import shared types from `crew_tracking.models`.
- Replace constructors that took `Network` with explicit `PathLossModel`, bounds,
  and (for legacy RTT only) `rtt_offset_ms`. `simulation.py` demonstrates this.
- Import simulated `Network`/`Receiver` from `simulator.network`, geometry from
  `simulator.environment`, and `build_radio_map` from `simulator.calibration`.
- Channel-only diagnostics now live on `SimulatedReading`, a `Reading` subclass.
  APs are immutable so changing one cannot corrupt a scan's dictionary keys.

The Word engineering guide at the workspace root describes the earlier flat
layout. This README and the algorithm notes describe the current implementation.

## Propagation and noise model

The retained RSSI model is:

```text
RSSI = reference level
       - log-distance path loss
       - sum of crossed-wall losses
       + fixed receiver bias
       + spatially correlated shadowing
       + averaged Rayleigh fast fading
       + measurement error
```

In symbols:

```text
P_r = P_0 - 10 n log10(d / d_0) - sum(L_wall) + b_rx + S(x) + F + e
```

- `P_0`, `d_0`, and `n` define median path loss.
- Each crossed wall contributes deterministic attenuation. Wall attenuation is
  applied before reception is decided; walls do not independently erase an AP.
- `S(x)` is a seeded Gaussian random field with approximately exponential
  spatial covariance `exp(-distance / correlation_distance)`. Random Fourier
  features make it repeatable and independent of grid traversal order.
- `F` is Rayleigh fading generated in linear power. Several independent power
  samples are averaged before converting to dB, avoiding the unrealistically
  deep fades of a single narrow-band ray.
- `e` is independent receiver measurement jitter. The reported value can then
  be quantized, matching devices that expose integer RSSI values.

Thermal noise is calculated from bandwidth and receiver noise figure:

```text
N_thermal = -174 + 10 log10(bandwidth_hz) + noise_figure_db
```

Background interference is combined with thermal noise in linear power. A
two-state process can create scan-wide interference bursts; APs on the same
channel therefore share the same noise-floor event. For each beacon, decoding
follows a smooth SNR sigmoid. If a scan observes `N` beacons, visibility is:

```text
p_seen = 1 - (1 - p_decode) ** N
```

This replaces the old fixed 30% dropout for every crossed wall. The old
behaviour remains available only through `legacy_wall_dropout: true` for result
comparison.

RTT first receives an error in metres and is then converted to round-trip time.
Its model includes Gaussian LOS error, stable device bias, extra variance and
positive bias per crossed wall, a smooth weak-signal penalty, and occasional
positive outliers. This avoids the old discontinuous `+2 m` rule.

FTM draws the same kinds of range errors across a configurable burst
(`ftm_burst_size: 8`, `ftm_success_probability: 0.95` by default). Capability,
RSSI visibility, and individual ranging success are separate conditions. This
is a conditional delivery abstraction, not a model of MAC contention. Stable
device and wall biases survive averaging. The generated timestamp exchanges
remove an arbitrary ACK turnaround and independent clock offset. FTM uses exact
3D geometric distance, without the RSSI model's short-distance clamp.

FTM has a separate deterministic random stream derived from the scan RNG state
without consuming that stream, or an explicit `ftm_rng` supplied by the caller.
Changing burst settings therefore leaves RSSI and legacy RTT inputs unchanged.
RSSI calibration disables FTM generation with `include_ftm=False`.

Each `SimulatedReading` exposes `mean_rssi_dbm`, `shadowing_db`,
`fast_fading_db`, `measurement_error_db`, `noise_floor_dbm`, `snr_db`,
`wall_count`, and `is_nlos` for diagnostics. Algorithms consume only the shared
measurement fields: `rssi_dbm`, `rtt_ms`, and/or `ftm_burst`.

## Parameter basis and calibration

The values in `layouts/simple_layout2.yaml` are defensible starting values, not
universal constants. Measurements aboard ferries at 2.45 GHz found path-loss
exponents from 1.0 to 3.82 and log-normal shadowing standard deviations from
1.21 to 2.56 dB across LOS, obstructed, and NLOS ship spaces. The same study
reported 17-25 dB additional loss from closing watertight doors. Independent
naval-ship measurements found roughly 20 dB attenuation through a
bulkhead/watertight door. The active preset therefore uses `n = 2.15`,
`shadowing_std_db = 2.0`, and `20 dB` steel-wall loss as provisional central
values.

For a real deployment, collect vessel measurements and fit parameters rather
than tuning them until a chosen localiser looks good:

1. Fit reference RSSI, path-loss exponent, and material losses from median RSSI.
2. Fit shadowing sigma and correlation distance from the residual spatial
   covariance.
3. Fit receiver bias and quantization separately for each device model.
4. Fit detection probability against measured SNR and missed beacon scans.
5. Fit RTT bias, variance, and outlier rate separately for LOS and each NLOS
   material class.
6. Validate on held-out compartments using RSSI/SNR distributions, detection
   rate, outage run lengths, and localisation error percentiles.

The benchmark feeds identical simulated scans to every localiser and uses
separate seeded random streams for calibration, demonstration, coverage, and
performance evaluation. This makes algorithm comparisons paired and prevents a
calibration setting from silently changing benchmark noise.

## Research basis

- FTM and positioning algorithm references: [algorithm notes](docs/algorithm.md).
- Shipboard path loss: [Kdouh et al., *Measurements and Path Loss Models for
  Shipboard Environments at 2.4 GHz*, EuMC
  2011](https://doi.org/10.23919/EuMC.2011.6101828).
- Bulkhead loss and non-Ricean ship channels: [Estes et al., *Shipboard Radio
  Frequency Propagation Measurements for Wireless Networks*, MILCOM
  2001](https://experts.boisestate.edu/en/publications/shipboard-radio-frequency-propagation-measurements-for-wireless-n/).
- Exponentially correlated shadowing: [Gudmundson, *Correlation Model for
  Shadow Fading in Mobile Radio Systems*, Electronics Letters
  1991](https://doi.org/10.1049/el:19911328).
- Rayleigh fading in metallic ferry compartments: [Cardoso et al., *Fast Fading
  Characterization for Body Area Networks in Circular Metallic Indoor
  Environments*, IEEE Access
  2020](https://doi.org/10.1109/ACCESS.2020.2977425).
- Indoor propagation and received-level autocorrelation: [ITU-R
  P.1238-13](https://www.itu.int/rec/R-REC-P.1238-13-202509-I/en).
- Wi-Fi RTT hardware, blocker, fluctuation, and outlier errors: [Dong et al.,
  *Error Investigation on Wi-Fi RTT in Commercial Consumer Devices*, Algorithms
  2022](https://doi.org/10.3390/a15120464).
