# Kalman tracking after Fusion

`crew_tracking.tracking.KalmanTracker` is an optional position/velocity layer
after the existing localiser. Each device ID has independent state. The tracker
accepts an ordinary `LocalisationEstimate` or `None` plus a capture timestamp;
it never receives true location/velocity, room labels, walls, radio-map entries,
NLOS labels, or the simulation's channel diagnostics.

## API and timestamps

```python
from crew_tracking.tracking import KalmanConfig, KalmanTracker

tracker = KalmanTracker(KalmanConfig())

# In the scan consumer, using the actual observation capture time in seconds:
measurement = fusion_localiser.locate(scan)
tracked = tracker.update(device_id, capture_timestamp_s, measurement)
if tracked is not None:
    print(tracked.position, tracked.velocity, tracked.status)
    print(tracked.measurement_age_s, tracked.covariance)

# A display tick can request a later prediction without consuming a new scan:
display = tracker.predict(device_id, display_timestamp_s)

# An actual scan/delivery event with no valid position still advances time:
missing = tracker.update(device_id, next_capture_timestamp_s, None)

tracker.reset(device_id)  # Forget one device; tracker.reset() forgets all.
```

Capture timestamps must be finite nonnegative seconds and strictly increase
within a retained track. Duplicate/out-of-order captured updates raise
`ValueError` without changing state. Real adapters must validate identity,
measurement age and units, unwrap device clocks as needed, and place capture
and display timestamps on the same time base. The tracker has no arrival clock
and cannot infer whether an old packet was just delivered. Request a prediction
at actual display time to apply freshness at the display, rather than treating
a late capture's zero-age update as current.

`predict()` is a non-mutating snapshot at or after the latest captured update.
A display prediction at time 12 does not prevent accepting a newly delivered
capture at 11 when the preceding captured update was at 10. Display polling
does not refresh observations, advance the capture watermark, or break an
outlier-recovery streak. Snapshot covariance arrays are detached copies.

`update(..., None)` records a missing capture; invalid/nonfinite coordinates
also count as missing. Unknown devices without a usable position create no
track. Strict timestamp ordering restarts after explicit reset or removal of
an expired track. A display query alone does not delete an expired track;
the next captured update or explicit reset performs that lifecycle change.

## Model and covariance

State and measurement are:

```text
state = [x, y, vx, vy]     # vessel-local metres and metres/second
measurement = [x, y]

F(dt) = [[I, dt*I], [0, I]]
Q(dt) = q * [[dt^3/3*I, dt^2/2*I], [dt^2/2*I, dt*I]]
H = [I, 0]
```

`q` is a continuous white-acceleration noise density in m²/s³, not a new
acceleration variance per scan. This choice integrates over elapsed time;
changing the number of missing-scan ticks does not change the underlying
continuous process noise. The filter assumes approximately constant velocity
between updates, with process noise allowing starts, stops and turns.

Prediction uses `F P F.T + Q`. Updates solve the innovation system without
explicit inversion and use the Joseph covariance update for numerical stability:

```text
v = measurement - H state
S = H P H.T + R
K = P H.T inverse(S)
state = state + K v
P = (I-KH) P (I-KH).T + K R K.T
```

Returned covariance is full 4-by-4 in `[x, y, vx, vy]` order. It is a model
covariance, not a validated real-world confidence region. A minimum position
eigenvalue is enforced by positive covariance addition. This prevents unlimited
collapse when successive Wi-Fi positions are correlated, but does not explicitly
model their temporal correlation or make persistent errors observable.

By default `R = 1.5² I`. The source estimate's covariance and heuristic Fusion
quality are ignored because neither currently represents calibrated position
error. To use an externally validated, device/session-specific observation model:

```python
import numpy as np

tracked = tracker.update(
    device_id, capture_timestamp_s, measurement,
    measurement_covariance=np.array([[2.0, 0.2], [0.2, 1.5]]),  # m²
)
```

An explicit matrix must be finite, symmetric, positive semidefinite and 2-by-2.
Its eigenvalues are floored before use. Invalid covariance is rejected before
state changes. Do not construct a very small matrix from radio-map neighbour
spread or square a Fusion heuristic score and label it measured uncertainty.

## Defaults and lifecycle

All parameters below are configurable through `KalmanConfig` and provisional.
They were not fitted to the route benchmark's ground truth.

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `measurement_std_m` | 1.5 m | Default per-axis measurement noise |
| `acceleration_noise_density_m2_s3` | 1.0 m²/s³ | Motion variation; higher values follow turns faster but smooth less |
| `initial_velocity_std_mps` | 1.5 m/s | Initial velocity uncertainty; mean velocity starts at zero |
| `position_std_floor_m` | 0.5 m | Minimum model position standard deviation |
| `max_prediction_age_s` | 3 s | Maximum age of the last accepted fix for returned predictions |
| `reset_after_s` | 10 s | Expire or reinitialise at a captured update after an observation gap |
| `innovation_gate` | 16 | Squared Mahalanobis innovation threshold |
| `reacquire_after_rejections` | 3 | Consistent rejected observations needed for recovery |

An accepted fix produces `initialised`, `updated` or `reinitialised` status.
Missing data produces `predicted`; a gated fix produces `outlier_rejected`.
The latter two retain the last accepted measurement timestamp. Once its age
exceeds 3 s, return `None` rather than presenting an old prediction as current.
During shorter gaps covariance grows. At a captured update 10 s after the
last accepted fix, delete the state if data is still missing or initialise a
fresh position and zero velocity if a valid measurement has arrived.

For outlier recovery, consecutive rejected positions must agree with their
predecessors under the sum of their measurement covariances and the innovation
gate. Three consistent rejections reinitialise the track. An accepted fix,
inconsistent rejection or missing captured scan breaks that streak. This avoids
permanent rejection after a real location change, but repeated biased positions
can also satisfy the recovery rule. It is not proof that the recovered room is
correct. Display prediction ticks do not affect the streak.

## Validation and benchmarking

Run the suite from the project directory:

```powershell
& .\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Validation includes independent analytical prediction/update calculations,
off-diagonal measurement covariance, irregular time intervals, numerical
symmetry/positive semidefiniteness, device isolation, input/snapshot mutation,
stale captures, delayed delivery after display prediction, finite outage
prediction, expiry/recovery, inconsistent outliers and persistent bias. A noisy
straight-line fixture checks RMS error reduction under the model's assumptions.
A synthetic two-room crossing checks bounded transition delay; it does not
establish doorway performance on the vessel.

The `--tracking` CLI runs actual RSSI/FTM Fusion on five separate within-room
journeys with stops, 0.75 m/s walks, right-angle turns and irregular intervals
between 0.55 and 1 s. Each route starts a separate device track. The layout lacks
doorway metadata, so it invents no route through steel bulkheads. No inertial
or true velocity data is supplied to the tracker.

Each radio scan and Fusion fix is generated once. The same fixes are replayed
through raw Fusion and Kalman, then replayed with deliberate fix-delivery gaps
at 20–22 s and 42–55 s. The outage scenario suppresses coordinates after
localisation; it does not claim an additional radio propagation model.

Metrics make the trade-offs explicit:

- Mean/P90 error uses only points where both methods return a position.
- Returned percentage uses all journey points, as does room accuracy; a missing
  position counts as incorrect for room accuracy.
- Stop jitter is RMS distance around each stop's estimated centroid, while
  stop absolute error measures distance from truth. Both use common fixes
  beginning 3 s after stopping, to distinguish settled jitter from stop response.
- Turn error measures common-point coordinate error in the first 2 s after
  a right-angle change. It is not a direct estimate of turn latency.
- Prediction-only mean/P90 scores points where raw Fusion is unavailable but
  Kalman returns a prediction, so extra availability cannot conceal drift.
- Recovery reports time from delivery resumption until a fresh fix is accepted,
  including irregular scan timing, plus recovered/total outage counts.

The persistent-bias fixture deliberately produces zero jitter with a 2 m
position error. A correlated-error fixture combines slowly varying bias with
an isolated gross outlier. These check limitations rather than assuming
white-noise improvements generalise to deployment.

Use both evaluation profiles and multiple seeds. The simulated radio model
retains strong room signatures, fixed walls and an incomplete model of temporal
correlation, body blocking and environment changes. Filtering can smooth a wrong
position or introduce lag at turns. None of these checks supplies a field
accuracy guarantee or enforces walls. Tune observation/process noise using
separate recorded walking sessions, then evaluate held-out devices and days.

## Recorded implementation checks

The complete project suite passed 140 tests. Four CLI runs below used the same
default tracker configuration throughout. Each radio route set has 558 points
across five devices/compartments. Errors compare common returned fixes; these
are illustrative implementation checks, not a statistical field trial.

| Survey spacing / scans, seed | Profile | Mean error, raw → tracked | P90 error, raw → tracked | Stop jitter, raw → tracked | Room accuracy, raw → tracked |
| --- | --- | --- | --- | --- | --- |
| 1 m / 5, 73129 | matched | 0.88 → 0.84 m | 1.46 → 1.39 m | 0.40 → 0.28 m | 98.6 → 99.1% |
| 1 m / 5, 73129 | deployment | 1.33 → 1.23 m | 1.98 → 1.93 m | 1.20 → 0.85 m | 93.5 → 93.0% |
| 0.3 m / 15, 98431 | matched | 0.67 → 0.60 m | 1.27 → 1.12 m | 0.46 → 0.32 m | 99.5 → 98.7% |
| 0.3 m / 15, 98431 | deployment | 0.98 → 0.90 m | 1.68 → 1.50 m | 0.77 → 0.49 m | 96.6 → 97.0% |

In the normal radio replays, turn-window mean error decreased in these runs
(0.86 → 0.80, 1.07 → 0.97, 0.69 → 0.60 and 0.95 → 0.85 m respectively).
These averages are not estimates of latency. The synthetic boundary-crossing
fixture separately limits post-crossing delay to 2 s under its stated inputs.
Room accuracy did not improve consistently, even when coordinate error fell.

With the explicit 2 s and 13 s delivery gaps, the tracker returned positions
on 87.5% of all route points versus raw Fusion's 81.9–82.1%. It returned 30–31
short predictions and suppressed 70 stale outputs per run; every simulated
gap recovered at the first accepted fix, averaging 0.53 s after delivery resumed
(10/10 gaps, including scan timing). The extra positions have their own error:

| Survey / seed | Profile | Prediction-only count | Prediction-only mean / P90 |
| --- | --- | ---: | --- |
| 1 m / 73129 | matched | 30 | 0.88 / 1.47 m |
| 1 m / 73129 | deployment | 31 | 1.27 / 2.08 m |
| 0.3 m / 98431 | matched | 30 | 1.23 / 2.17 m |
| 0.3 m / 98431 | deployment | 30 | 1.89 / 3.08 m |

The last row illustrates the risk of extrapolating a biased or turning track.
Higher availability does not mean equal accuracy during missing data. Select
the freshness limit and model parameters from product requirements and real
recordings. Different seeds between survey configurations mean these numbers
do not isolate the effect of survey density. No parameters were tuned between
these four checks.
