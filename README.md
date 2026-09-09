# Tracking Simulation

This project simulates Wi-Fi RSSI and RTT positioning inside a compartmented,
metallic ship environment. It compares nearest-AP, weighted-centroid,
trilateration, and fingerprint-based localisers under a configurable stochastic
radio channel.

## Run

```powershell
py -m pip install -r requirements.txt
py simulation.py
```

The layout path is resolved relative to `simulation.py`, so the script may also
be launched from another working directory.

## Propagation and noise model

The model deliberately separates effects that have different physical and
statistical behaviour:

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

Each successful `Reading` exposes `mean_rssi_dbm`, `shadowing_db`,
`fast_fading_db`, `measurement_error_db`, `noise_floor_dbm`, `snr_db`,
`wall_count`, and `is_nlos` for diagnostics. Existing localisers continue to use
the original `rssi_dbm` and `rtt_ms` fields.

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

The benchmark now feeds identical simulated scans to every localiser and uses
separate seeded random streams for calibration, demonstration, coverage, and
performance evaluation. This makes algorithm comparisons paired and prevents a
calibration setting from silently changing benchmark noise.

## Research basis

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
- Wi-Fi RTT hardware, blocker, fluctuation, and outlier errors: [Yu et al.,
  *Error Investigation on Wi-Fi RTT in Commercial Consumer Devices*, Algorithms
  2022](https://doi.org/10.3390/a15120464).
