# Quality-aware fusion

`crew_tracking.fusion.FusionLocaliser` combines fingerprint and FTM estimates
from the same scan. Nearest-AP and power-centroid are used only if both primary
methods fail. They do not add separate votes alongside fingerprinting because
all three reuse RSSI measurements. `FixedWeightFusionLocaliser` preserves the
previous four-component blend for comparison.

## Information boundary

The algorithm accepts only `Scan` observations, surveyed AP metadata, an onsite
radio map, configured receiver height/bounds, and optional externally measured
range biases. It never receives a simulated receiver, true receiver position,
wall geometry/counts, LOS/NLOS labels, ideal RSSI, shadowing values or channel
parameters. Calibration positions and AP positions are legitimate deployment
survey information; live receiver ground truth is not.

Both estimators now attach observable diagnostics to `LocalisationEstimate.quality`.
Fusion uses these rather than the existing covariance matrices. Fingerprint
covariance describes selected reference-point spread; FTM covariance is a local
least-squares approximation. Neither is a calibrated field-error distribution.
Fusion leaves `covariance=None` and records weights/decision in `FusionQuality`.
It reports the maximum contributing AP count rather than adding overlapping sets.

For real hardware, the integration must validate BSSID, units, status, age and
observation timing. The existing FTM implementation requires actual independent
exchanges/corrected propagation RTT samples. A platform aggregate must not be
duplicated to invent a burst or sample retention statistics. An adapter for such
summary-only APIs remains necessary. Without usable FTM data, fusion uses the
available fingerprint result. No scan freshness or clock information is invented
by the algorithm.

## Quality scores

The scales below are heuristic risk scores in metres for relative weighting.
They are not estimated standard deviations, probabilities or confidence bounds.
The weighting floors and score conversion controls are exposed through
`FusionConfig`; the formula below also states the fixed conservative penalties.
These choices are provisional and were not fit to the benchmark's true receiver
coordinates.

For fingerprinting, retain the existing k-neighbour position calculation and
add diagnostics:

- `r`: nearest signal-space distance divided by sqrt(number of heard known APs),
  in dB. The existing missing-AP floor still contributes to this distance.
- `s`: RMS spatial spread of the selected, weighted neighbours.
- `a`: RMS spatial distance from the estimate to every reference point with
  signal RMS distance within `ambiguity_margin_db` of the best match (default
  2 dB). Searching beyond the selected k detects separated matching clusters.
- `f`: fraction of known APs observed; `n`: number heard.

```text
fingerprint_scale = hypot(0.75 m, s, a, 1.0 m * r / 6 dB)
                    / sqrt(f * min(n / 3, 1))
```

This does not replace fingerprinting with a probabilistic matcher or fix the
missing-AP model. It gives fusion observable indicators of a weak or ambiguous
match. Dense calibration, device drift or a confidently wrong unique match can
still defeat it. Identical reference signatures far apart increase ambiguity
even if k=1 would otherwise produce a zero covariance.

For FTM, let `G` be the unweighted XY slant-range Jacobian at the fitted point,
using the surveyed AP coordinates and configured receiver height:

```text
g = sqrt(trace(inverse(G.T @ G)))
r = RMS unwhitened range residual across all accepted radios, in metres
u = RMS aggregate range standard error, including the existing range floor
q = retained samples / submitted samples, across accepted distinct radios

ftm_scale = hypot(1.5 m, g * hypot(r, u)) / sqrt(q)
```

Then multiply by 1.5 when only three APs support the fit, by
`sqrt(max(1, 6 / minimum_retained_samples_per_AP))` for short bursts, and by 2
at an active position bound. Retention includes failed, malformed and rejected
exchanges in the denominator for radios actually used. It is not a network
delivery probability or a measure of all AP coverage.

Geometry is derived from raw range geometry, not from robust-loss curvature.
Residuals are computed before robust clipping; stable but inconsistent ranges
therefore remain visible. A consistent NLOS bias can evade the residual check.
More exchanges do not remove the 1.5 m fusion floor or establish LOS conditions.
These checks do not estimate or correct blocker bias.

Unknown or malformed diagnostics receive the configurable conservative
`unknown_quality_scale_m` (default 5 m). Zero covariance never implies zero
error. Both primary estimates remain usable even when their diagnostics are
unknown, with correspondingly conservative weighting.

## Decisions and fallback

Relative weights are proportional to the inverse square of these scores. This
is an engineering preference rule, not a product of independent Gaussian
likelihoods: RSSI and FTM share APs, obstructions and detection effects.

If positions are farther apart than
`3 * hypot(fingerprint_scale, ftm_scale)`, choose the lower-score estimate
instead of averaging the two hypotheses. Ties choose fingerprinting. The
decision is recorded as `disagreement_fingerprint` or `disagreement_ftm`.
Selecting a lower-score hypothesis cannot establish that it is correct;
coherent bias and common errors remain unresolved.

If one primary method misses or returns nonfinite coordinates, return the
other primary estimate. If both fail, try the power centroid, then nearest AP,
then return `None`. Small source uncertainty never creates a fused covariance
or a claimed accuracy improvement.

```python
from crew_tracking.fusion import FusionConfig, FusionLocaliser

localiser = FusionLocaliser(
    nearest_ap_localiser, weighted_power_localiser,
    fingerprint_localiser, ftm_localiser,
    config=FusionConfig(fingerprint_floor_m=0.75, ftm_floor_m=1.5),
)
estimate = localiser.locate(scan)
if estimate is not None:
    print(estimate.position)
    print(estimate.quality)  # Decision, weights, scores and separation.
```

Code previously passing `weights_with_ftm` or `weights_without_ftm` should use
`FixedWeightFusionLocaliser`. Adaptive fusion's four component constructor
arguments retain their original order.

## Simulation and deployment validation

`simulation.py` reports `Fusion`, `Fixed Fusion`, `Fingerprint`, `FTM` and the
other baselines on identical plain `Reading` scans. `simulator.evaluation`
copies measurement fields and discards all synthetic channel diagnostics.
Random evaluation positions are generated before scans so differences in
channel RNG consumption do not change the positions between profiles.

The default `matched` profile shares the calibration channel's static shadow
field. Dense survey spacing, fixed wall attenuation and favourable AP placement
can make room classification exceptionally easy. Independent scan noise does
not test a new device or a changed environment.

The `deployment` profile is a sensitivity scenario with explicitly assumed
values: independent shadow field, +4 dB receiver reporting offset, seeded per-AP
reporting offsets in [-3,+3] dB, 1.3 m true receiver height against the 1.0 m
survey/solver assumption, removal of one visible AP on 10% of scans, and +0.5 m
common FTM bias. The offsets affect reporting after detection, so they do not
claim to model transmit-power drift or alter beacon reception. Removing an AP
also removes its FTM burst. The survey network remains unchanged.

This is one sensitivity envelope, not a distribution fitted to the vessel;
an independent shadow field is a stress assumption rather than a measured
model of map ageing. It does not include body/orientation effects, doors,
motion, MAC contention, counter/clock errors, stale observations or surveyed
AP-coordinate errors. It cannot establish a real-world error bound.

Use multiple evaluation seeds and survey spacings. Tune floors and thresholds
with separate known-location deployment sessions/devices, then freeze them
before evaluating held-out sessions. Report errors, misses and compartment
confusion together; improved error among returned positions alone can conceal
outages. Comparing actual hardware captures from different days/devices is the
next required step before selecting production parameters.

### Implementation validation

The implementation passed 103 automated checks, including malformed metadata,
ambiguity, disagreement, fallback, observed-only hardware inputs, the stripped
simulation boundary and unchanged calibration state. The following paired CLI
runs used 60 positions per compartment (300 total per profile); each row uses
identical scans for fixed and adaptive fusion. These illustrative runs are
regression/sensitivity checks, not a statistical field-accuracy estimate.

| Survey spacing / scans, evaluation seed | Profile | Mean error, fixed → adaptive | P90 error, fixed → adaptive | Room accuracy, fixed → adaptive |
| --- | --- | --- | --- | --- |
| 1 m / 5, 73129 | matched | 1.07 → 1.02 m | 1.88 → 1.79 m | 99.3 → 100.0% |
| 1 m / 5, 73129 | deployment | 1.30 → 1.35 m | 2.29 → 2.32 m | 94.7 → 97.0% |
| 0.3 m / 15, 98431 | matched | 0.90 → 0.80 m | 1.54 → 1.48 m | 98.0 → 98.3% |
| 0.3 m / 15, 98431 | deployment | 1.08 → 1.06 m | 1.74 → 1.71 m | 95.3 → 96.3% |

Miss rates were unchanged by adaptive fusion in these paired comparisons
(0%, 0%, 0%, and 0.3% respectively). The coarser deployment run improved room
accuracy but worsened mean/P90 coordinate error. Fingerprinting alone still
beat or tied adaptive fusion on mean error in the dense-survey runs. The new
policy is not guaranteed to outperform either component or the old blend.
Different seeds are used for the two survey configurations, so their numbers
do not isolate the effect of survey spacing.

The research motivation is measurement fusion, for example
[Rizk, Elmogy and Yamaguchi, *A Robust and Accurate Indoor Localization Using
Learning-Based Fusion of Wi-Fi RTT and RSSI*, 2022](https://doi.org/10.3390/s22072700).
The current implementation is an explicit quality heuristic, not RRLoc's
deep-learning model or a reproduction of its reported performance. Persistent
hardware and blocker bias are documented by
[Dong et al., *Error Investigation on Wi-Fi RTT in Commercial Consumer Devices*,
2022](https://doi.org/10.3390/a15120464).
