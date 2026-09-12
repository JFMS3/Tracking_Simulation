# Localisation algorithms and FTM

The `crew_tracking` package consumes measurements and surveyed AP positions.
It has no dependence on simulated receivers, walls, random channels, calibration
scans, plots, or YAML. Those belong to `simulator` and `simulation.py`.

## FTM timing and units

`FTMExchange` represents one successful hardware exchange using the convention
in Ibrahim et al. (2018), section 2.3:

```text
t1: AP transmits FTM                  t4: AP receives ACK
t2: station receives FTM              t3: station transmits ACK
RTT_ns = (t4_ns - t1_ns) - (t3_ns - t2_ns)
distance_m = 299792458 * RTT_ns / (2 * 10^9)
```

The differences use each device's own clock, cancelling constant clock offsets
and the station's ACK turnaround. Integer differences are calculated before
conversion to float to retain precision at large clock epochs. The integration
adapter must unwrap counters and handle clock-rate error. For sub-nanosecond
hardware timestamps, form relative values before converting units to avoid
rounding large absolute timestamps. See [Ibrahim et al., *Verification: Accuracy
Evaluation of WiFi Fine Time Measurements on an Open Platform*, MobiCom
2018](https://doi.org/10.1145/3241539.3241555),
[author manuscript](https://winlab.rutgers.edu/~gruteser/papers/ftm_mobicom.pdf).

`FTMSample` accepts an already corrected propagation RTT in nanoseconds.
`distance_m_to_rtt_ns` can adapt an individual measured range to this contract.
Signed measurements are preserved because timing noise and hardware bias can
produce negative readings. Generic network/ping latency is not a valid input.
`Reading.rtt_ms` is used only by the separately labelled legacy baseline.

This implementation processes observations; it does not negotiate capabilities
or transmit FTM frames. Platform adapters must validate ranging status, AP
identity, result age, and units before creating a scan. For example Android
reports distance and standard deviation in millimetres, along with successful
measurement counts. A single successful result has an invalid zero standard
deviation. A platform aggregate is not a set of independent raw exchanges:
do not duplicate it to satisfy the burst minimum. A dedicated summary-result
adapter is needed if raw exchanges are unavailable. See [Android
RangingResult](https://developer.android.com/reference/android/net/wifi/rtt/RangingResult).

## Burst processing

For each AP, `aggregate_ftm_burst`:

1. Screens failed samples and nonfinite RTTs.
2. Converts each sample to metres and subtracts the externally calibrated signed
   bias. A positive calibration value means the device overestimates range.
3. Centres on the median and calculates `1.4826022185 * MAD`. This factor converts
   median absolute deviation to a Gaussian-consistent spread estimate.
4. Rejects samples farther than `outlier_threshold * max(MAD_scale, floor)` from
   the median, then checks the retained sample count.
5. Averages retained signed samples and rejects a nonpositive final range.
   Rejecting individual negative samples first would bias short ranges upward.
6. Reports uncertainty `sqrt(sample_variance / n + floor^2)` and retained/rejected
   counts. The floor remains even for a constant or very long burst.

The MAD scaling follows [SciPy's median absolute deviation
documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html).
The rejection rule, retained mean, sample-count threshold, and uncertainty floor
are project engineering choices, not requirements of IEEE FTM. They assume a
mostly stable receiver during a burst. Strong within-burst correlation or several
outlier populations can invalidate the estimated sample-mean variance.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `min_samples` | 3 | Minimum retained exchanges per AP (configurable, at least 2) |
| `min_aps` | 3 | Minimum distinct usable radios (at least 3) |
| `range_std_floor_m` | 0.25 m | Persistent uncertainty floor and minimum screening scale |
| `outlier_threshold` | 3.5 | Median-centred screening multiplier |
| `robust` | true | Use soft-L1 loss instead of ordinary weighted least squares |
| `robust_scale` | 1.5 | Dimensionless loss transition after uncertainty normalization |
| `max_geometry_condition` | 1,000,000 | Maximum singular-value ratio for XY survey/fit Jacobian |

These defaults are provisional. Calibrate offsets and uncertainty from repeated
known-distance observations, separated by AP, device, channel, and deployment
condition. Supply a receiver-wide `range_bias_m` and optional additive offsets
in `bias_by_bssid`; the localiser copies and normalizes the latter's keys.

## Position estimate

For receiver height `z` configured by the deployment and surveyed AP position
`(ax_i, ay_i, az_i)`, the fitted residual is:

```text
predicted_i(x,y) = sqrt((x-ax_i)^2 + (y-ay_i)^2 + (z-az_i)^2)
e_i(x,y) = (predicted_i(x,y) - measured_range_i) / sigma_i
objective = 1/2 * sum rho(e_i^2)
```

Uncertainty normalization implements inverse-variance weighting; less precise
bursts have less influence. The known-height slant-range equation is a direct
geometric derivation, avoiding a square-root conversion of noisy measurements
to horizontal ranges. Weighted nonlinear fitting follows [NIST's weighted
least-squares description](https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd143.htm).

The implementation uses bounded `scipy.optimize.least_squares` with an analytic
Jacobian, a squared-range linear initial estimate, and a weighted anchor-centre
initial estimate. It selects the successful fit with lowest objective. Soft-L1
reduces the influence of inconsistent anchor ranges; `robust=False` enables
ordinary weighted least squares for controlled comparisons. Loss and Jacobian
semantics follow [SciPy's solver
documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

The localiser returns `None` for insufficient usable bursts, fewer than the
required distinct BSSIDs, conflicting coordinates for one BSSID, collinear or
ill-conditioned AP geometry, or unsuccessful optimization. Consistent duplicate
BSSIDs count once. Bounds constrain a rectangle supplied by the caller; they do
not enforce walls, compartment polygons, or deck transitions. Three APs are a
minimum for this 2D model; extra APs improve redundancy. Receiver height is an
assumption, not an additional estimated coordinate.

For an interior solution, covariance is approximated by the inverse of the
normal matrix from the weighted, loss-adjusted Jacobian. A fixed range-noise
floor prevents exact residuals from implying perfect position certainty.
Covariance is unavailable at an active bound. It is a local approximation, not
a calibrated confidence guarantee under correlated errors, robust loss, NLOS,
or uncertain AP/receiver height.

## NLOS and calibration limits

FTM is affected by device offsets, blockers, multipath, fluctuations, and outliers.
Robust aggregation/loss can suppress isolated inconsistent errors; a persistent
or geometrically consistent NLOS bias can still move the entire solution.
This implementation does not infer NLOS from the simulator's `is_nlos` or
`wall_count` fields. The earlier claim that walls cannot bias RTT was incorrect.
See Dong, Shi, Arslan and Yang, [*Error Investigation on Wi-Fi RTT in Commercial
Consumer Devices*, Algorithms 15(12), 464,
2022](https://doi.org/10.3390/a15120464).

## Retained baselines

`positioning.py` provides nearest-AP and weighted-centroid baselines plus
nonlinear least-squares RSSI and legacy RTT fits. RSSI distance inverts the
configured log-distance relation; unmodelled wall attenuation biases it.
These legacy range fits treat ranges as horizontal, preserving their comparison
role; FTM handles slant geometry explicitly. Weighted AP spread and fingerprint
neighbour spread describe voting dispersion rather than calibrated error bars.

`fingerprinting.py` accepts a supplied radio map. Signal-space nearest-neighbour
matching is based on the fingerprinting approach established by Bahl and
Padmanabhan's [*RADAR: An In-Building RF-based User Location and Tracking System*,
INFOCOM 2000](https://www.microsoft.com/en-us/research/publication/radar-an-in-building-rf-based-user-location-and-tracking-system/).
Inverse-distance voting/coordinate weights, missing-signal floor and neighbour
count are implementation choices. Creating a synthetic map is exclusively
`simulator.calibration.build_radio_map`; a deployment supplies surveyed entries.

## Validation and reproducibility

Run `python -B -m unittest discover -s tests -v`. Tests cover timestamp units and
clock-offset cancellation, signed noisy samples, calibration, uncertainty and
outlier handling, known 3D anchor geometry, degeneracy, capability/failure cases,
short ranges, simulated end-to-end positioning, and RNG invariance. A subprocess
test blocks all simulator/plot/YAML/Shapely imports while loading the algorithm
modules, checking the dependency boundary independently of the current process.

The simulation feeds the same scan to every benchmark method. FTM uses a
separate repeatable random stream and RSSI calibration disables FTM generation.
Consequently, changing FTM burst settings does not change the RSSI/legacy RTT
scan sequence. Performance figures remain evidence about the assumed simulation,
not a validation of accuracy aboard a vessel or of a physical hardware adapter.
FTM receives up to eight ranging exchanges per AP while the legacy RTT baseline
receives one. Their benchmark comparison includes both measurement averaging
and estimator changes; it does not isolate robust fitting at an equal airtime
or energy budget.
