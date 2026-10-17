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

```powershell
py scenarios.py   # every localiser across 9 ship/channel scenarios -> results/scenarios/
py tracking.py    # simulated crew walks, snapshot vs grid tracker -> results/tracking/
```

## Wall-aware localisation and tracking

`WallAwareLocaliser` (positioning.py) evaluates every point of a grid over the
deck: given the walls between that point and each AP, what RSSI and RTT range
would be expected? It scores the scan against those predictions, also counting
APs that were *not* heard as evidence. It can optionally estimate a constant
per-scan phone RSSI offset. Plain RSS/RTT trilateration ignores walls, so
every bulkhead's 20 dB loss is misread as extra distance; that is why those
localisers fail between compartments.

`GridTracker` (tracking.py) runs the same likelihood as a grid Bayes filter
over time. Between scans, belief spreads by *walking distance* (the shortest
route that does not pass through a wall), so it moves between compartments
only through doorways.

`FixedLagSmoother(tracker, lag_steps=5)` (tracking.py) wraps a `GridTracker`
and returns, at epoch t, the estimate of epoch t - lag_steps given every scan
up to t: exact forward-backward smoothing of the tracker's HMM over the last
`lag_steps` epochs (normalised forward beliefs, backward messages rescaled at
each step, one sparse matrix-vector product per step). `delay_steps` tells
`evaluate_tracking` to align the delayed estimates, and `flush()` returns the
last `lag_steps` estimates at the end of a walk. Lag 0 is the forward filter.
Smoothing improved on causal filtering by 34-60% in indoor trials (De Cock et
al. 2021; Viol et al. IPIN 2012), but no source reports a lag sweep; 5 s covers
a doorway transit. On the deck it costs 5 s of latency for about 20% lower
mean error (docs/sensor_fusion_spec.md, T5).

`WallAwareLocaliser(..., rtt_nlos_mixture=True)` scores each RTT range with a
two-component Gaussian mixture instead of a single Gaussian (or Cauchy) term:
the usual LOS-like core plus, with weight 0.22, a component shifted +1.1 m and
2 m wider. The weight and shift are Gentner et al.'s phone-measured FTM
mixture (PLANS 2020: wide component p = 0.22, mean 1.13 m above the narrow
one; it cut particle-filter error by 0.42-0.60 m); the extra spread follows the
1-3 m proposed for NLOS excess delay and covers metre-level per-AP calibration
residuals (Bullmann et al. 2020). It is applied to LOS and NLOS links alike
(`rtt_mixture_los_weight=0` restricts it to links through a wall) and is off
by default, which keeps the previous likelihood exactly. With realistic
2.4 GHz RTT errors and 1.5 m per-AP offsets it lowers the hybrid snapshot's
P90 error by about 25% (spec T7). **The gain is for single-scan (snapshot)
estimates only: inside the trackers it made things worse** (held-out
benchmark, 8 of 8 walks: GridTracker +11%, ParticleTracker +25% mean error,
and lower coverage95), so keep it off there. Its +1.1 m shift also has the
opposite sign to the realistic preset's -1.2 m instrument bias, so most of
the snapshot gain comes from the wider component, not from bias correction.

Both rely on the model in the `Network` they are given. `scenarios.py` tests
them when that model is wrong (wall losses, path-loss exponent, RTT bias),
when the phone has an RSSI offset, and with 2.4 GHz RTT errors closer to those
reported for commodity phones.

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
positive outliers. This avoids the old discontinuous `+2 m` rule. Optionally
(`rtt_ap_offset_std_m > 0`) each AP also gets a fixed range offset, drawn once
from a zero-mean Gaussian with that standard deviation. Like the shadow field,
it is a deterministic function of `(shadowing_seed, bssid)`, so it does not
depend on AP order or consume the scan random stream; at the default 0 scans
are exactly as before.

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

`rtt_ap_offset_std_m` defaults to 0 (offsets assumed calibrated out, as the
layouts' nominal RTT model does). Uncalibrated commodity FTM is far worse:
Bullmann et al. (Sensors 2020) had to fit per-AP corrections between -10 m and
+0.65 m at 2.4 GHz/20 MHz, Ibrahim et al. a 5.7 m offset for Intel 8260 pairs,
and Horn reports pair offsets of 0.2-5.5 m. Offset correction roughly halved
Bullmann's FTM error. The sensor-fusion target T7 uses 1.5 m, a modest
residual after per-model calibration (it adds to the -1.2 m common device bias
of the realistic 2.4 GHz preset); use several metres to model uncalibrated
APs.

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
- Per-AP FTM offsets and NLOS errors behind metal doors at 2.4 GHz/20 MHz:
  [Bullmann et al., Sensors
  2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7472118/).
- Gaussian-mixture RTT likelihood fitted to phone FTM errors in a particle
  filter: [Gentner, Kuehner, Ulmschneider & Dammann (DLR), IEEE/ION PLANS
  2020](https://elib.dlr.de/134078/1/ION_PLANS_2020.pdf).
- Smoothing vs filtering indoors: [De Cock et al., Sensors
  2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8271586); [Viol et al., IPIN
  2012](https://www.comsys.rwth-aachen.de/fileadmin/papers/2012/2012-11-viol-ipin.pdf).
