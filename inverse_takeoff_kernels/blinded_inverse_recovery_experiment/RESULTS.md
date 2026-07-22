# AWS experiment results

Two blinded inverse-recovery sweeps were completed on 2026-07-22. Both used a
single `c7i.8xlarge` On-Demand instance with 32 worker processes, a no-ingress
security group, temporary private S3 storage, and instance-initiated
termination.

## Run inventory

| Run | Purpose | Schemas | Recovery trials | Wall time | Compute estimate |
|---|---:|---:|---:|---:|---:|
| `inverse-takeoff-20260722T203012Z` | Broad modal baseline | 575,440 | 13,810,560 | 18.6 min | $0.444 |
| `inverse-takeoff-20260722T205043Z` | Horizon-ratio confirmation | 589,302 | 18,857,664 | 14.2 min | $0.338 |
| **Combined** |  | **1,164,742** | **32,668,224** | **32.8 min** | **$0.782** |

The compute estimate uses the AWS price returned immediately before launch,
`$1.428/hour`. Temporary S3, EBS, request, and public-IPv4 charges are small but
not included because finalized billing was not yet available. The combined run
remained far below the requested `$5` ceiling, and each run was far below the
45-minute runtime ceiling.

Both processes exited with code zero. Both instances terminated, and both
temporary buckets and security groups were removed after result collection.

## Principal findings

### 1. Exact sub-lag blindness appears empirically

For the finite-horizon family, using noiseless windows beginning at the origin:

| Horizon ratio | Trials | Modal-class accuracy | Forecast success | Mean grouped median NRMSE |
|---|---:|---:|---:|---:|
| `T/L < 0.5` | 101,309 | 0.0% | 4.9% | 0.559 |
| `0.5 <= T/L < 1` | 132,303 | 0.0% | 0.0% | 0.740 |
| `1 <= T/L < 2` | 147,761 | 1.4% | 0.0% | 0.580 |
| `T/L >= 2` | 209,671 | 8.4% | 10.9% | 0.493 |

The zero recovery below `T/L=1` is consistent with the exact construction: the
observed prefix is independent of the hidden parameter `t`, while the hidden
recurrence polynomial changes. This is an information obstruction, not merely
an optimization failure.

Crossing the lag did not create rapid identification. Accuracy remained poor
through `T/L >= 2`, showing that a generic low-order matrix-pencil fit is not a
sufficient constrained decoder for this high-order, nearly periodic family.
This is a useful negative result and motivates the proposed integer,
fixed-speed reconstruction stage.

### 2. Failure is family-specific, not universal

In the broad baseline, noiseless random dyadic-split recurrences observed after
their maximum lag were substantially easier:

| Window | Modal-class accuracy | Forecast success | Median radius error |
|---:|---:|---:|---:|
| 24 | 78.3% | 95.6% | 0.0024 |
| 48 | 88.9% | 99.5% | 0.0007 |
| 96 | 86.6% | 99.8% | 0.0005 |

Thus the hard-family result cannot be dismissed as a completely broken
estimator. The same implementation accurately recovers generic short-lag
responses once the lag is visible.

### 3. Prediction and modal identification separate under noise

For post-lag random-split traces with a 96-sample window, increasing relative
noise from zero to `1e-3` changed:

- modal-class accuracy from 86.6% to 32.1%;
- median leading-radius error from 0.0005 to 0.4453;
- forecast success remained 99.8%.

Finite-window prediction can therefore remain excellent after the inferred
modal explanation has become unreliable. This empirically reinforces the
paper's reporting hierarchy: a good response fit must not be presented as a
recovered recursive mechanism.

## Interpretation and next claim

The experiment supports three distinct statements:

1. **Theorem-level obstruction:** without a lag bound, a finite prefix cannot
   identify the recurrence, even at fixed terminal speed and branch count.
2. **Empirical baseline:** generic modal recovery works well for many visible,
   short-lag recurrences but fails sharply on the constructed hidden-lag
   family.
3. **Methodological warning:** forecast accuracy and modal/source accuracy can
   diverge dramatically under noise.

The next algorithmic step should use the known constraint

\[
\sum_j a_j2^{-j}=1,
\qquad a_j\in\mathbb{Z}_{\ge0},
\]

together with a lag bound to enumerate or optimize over plausible recurrences.
The publishable comparison is then generic matrix pencil versus a
constraint-aware decoder as a function of `T/L`, noise, and modal separation.

## Artifacts

- Broad run: `runs/inverse-takeoff-20260722T203012Z/results/`
- Horizon run: `runs/inverse-takeoff-20260722T205043Z/results/`
- Each run contains `run_metadata.json`, `summary.json`, `summary.csv`, 32
  compressed raw shards, `hard_cases.csv`, and the captured instance log.
