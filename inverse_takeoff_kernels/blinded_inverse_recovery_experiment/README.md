# Blinded Inverse Takeoff Recovery

Standalone note: [PDF](blinded_inverse_note.pdf) | [LaTeX source](blinded_inverse_note.tex)

Publication figures are generated from the recurrence code and completed AWS
summaries by `python make_figures.py`; vector and high-resolution raster files
are written to `figures/`.

This experiment asks whether a finite observed redundancy-takeoff trace can
recover the modal structure of the hidden recursion that produced it. The
generator retains the recurrence and its poles as ground truth; the estimator
receives only a noisy finite window of

\[
Y_n = F_n-1,
\qquad
F_n = \frac{\log N_{n+1}-\log N_n}{\alpha}.
\]

The first cloud run measured modal recovery. The follow-up
`run_constraint_comparison.py` performs literal source reconstruction and
tests the sharp known-lag boundary. For maximum lag (D), the prefix with
(T=D-2) transitions can still have multiple compatible integer recurrences;
the prefix with (T=D-1) transitions identifies all coefficients once the
terminal speed is known. It compares:

1. a mixed-integer decoder enforcing nonnegativity, integrality, and
   \(\sum_j a_j2^{-j}=1\);
2. the same linear inverse equations with continuous nonnegative fitting and
   coefficient rounding; and
3. the order-six matrix pencil used in the original blinded benchmark.

Run a local comparison with:

```powershell
python run_constraint_comparison.py --output-dir constraint_run --workers 8 --worker-seconds 60
```

The capped AWS launcher selects it with:

```powershell
.\aws\launch_and_collect.ps1 -Experiment constraint -WorkerSeconds 720
```

## Academic novelty

The benchmark scale is not itself the novelty. Prony, matrix-pencil, ESPRIT,
Hankel-rank, and noisy exponential-sum recovery are classical system
identification tools. The new object is the finite response of memoizable
recursive redundancy, together with structural constraints inherited from a
nonnegative integer recurrence.

The strongest theorem candidate exposed by this experiment is finite-horizon
nonidentifiability even after first-order invariants are fixed.

Fix an observation horizon `T` and choose `L > T`. For an integer `t` satisfying

\[
1 \le t < \frac{2^{L+1}}{3},
\]

define

\[
\begin{aligned}
a_L &= t,\\
a_{L+1} &= 2^{L+1}-3t,\\
a_{L+2} &= 2t.
\end{aligned}
\]

Then

\[
a_L+a_{L+1}+a_{L+2}=2^{L+1}
\]

and

\[
a_L2^{-L}+a_{L+1}2^{-(L+1)}+a_{L+2}2^{-(L+2)}=1.
\]

Consequently every member of the family has terminal tree speed
`alpha = log(2)`, the same total branch count, an aperiodic one-dimensional
reachable state set, and polynomial memoized state growth. Under the boundary
convention `N_m = 1` for `m <= 0`, every recurrence also has

\[
N_1=N_2=\cdots=N_L=1+2^{L+1},
\]

independently of `t`. Its observed takeoff prefix through `F_{L-1}` is therefore
identical, while the recurrence polynomial and generally its non-dominant
modal spectrum change with `t`.

This yields a precise negative result: without a known lag bound, no estimator
can identify the recursive mechanism from an arbitrary finite prefix, even
with noiseless data, known terminal speed, and known total branching. It is
stronger than nonidentifiability of a latent split `a = b + c`, because these
are genuinely different combined recurrences.

With a known maximum lag `D` and terminal speed, the exact boundary is sharper:
`T=D-1` samples recover the full recurrence, while `T=D-2` samples fail
uniformly. The constraint sweep reproduced that transition across 38,796
schemas. Further robustness questions use scaling variables such as

\[
\frac{T}{L},
\qquad
T(1-\lambda_\star),
\qquad
\mathrm{SNR},
\qquad
\text{minimum pole separation}.
\]

A publishable version now combines:

1. the finite-horizon impossibility theorem;
2. the sharp `D-1`-sample identifiability theorem under a bounded lag;
3. a constraint-aware decoder using nonnegative integer, sparse, fixed-speed
   recurrence structure;
4. the blinded phase diagram produced here; and
5. with validation on real recursive traces remaining future work.

The novelty claim must remain scoped carefully: the generic spectral estimator
is classical; the recursive-redundancy observable, fixed-invariant comparison,
finite-horizon obstruction, and constraint-aware recovery problem are the
proposed contribution.

## Benchmark families

All synthetic recursive families satisfy

\[
\sum_j a_j2^{-j}=1,
\]

so their terminal speed is exactly `log(2)`.

- `immediate`: `a_1 = 2`.
- `slow`: `a_L = 2^L-1`, `a_{L+1}=2`.
- `finite_horizon`: the indistinguishable-prefix family above.
- `two_scale`: one local branch and a dyadically equivalent delayed channel.
- `random_split`: random mass-preserving dyadic branch refinements.

For each schema the experiment evaluates origin and post-lag observation
windows of 8, 16, 32, and 64 samples, hidden lags reaching 32, and several
relative noise levels. Every trial is tagged by its horizon ratio `T/L` and the
bins `<0.5`, `0.5--1`, `1--2`, and `>=2`. Recovery
uses a validation-selected matrix-pencil model. Ground truth is opened only
for scoring.

Reported outcomes include model order, leading-pole radius error, takeoff-class
accuracy, holdout normalized RMSE, and catastrophic failure examples.

## Local smoke test

```powershell
python inverse_takeoff_kernels/blinded_inverse_recovery_experiment/run_experiment.py `
  --output-dir inverse_takeoff_kernels/blinded_inverse_recovery_experiment/smoke `
  --workers 2 `
  --max-schemas-per-worker 8 `
  --worker-seconds 30
```

## AWS run

The launcher uses one `c7i.8xlarge` On-Demand instance in `us-east-1`, the
largest instance allowed by the account's current 32-vCPU standard-instance
quota. At the price checked on 2026-07-22 (`$1.428/hour`), its 42-minute forced
termination bounds instance compute at `$0.9996`. A private temporary S3 bucket,
no-ingress security group, small delete-on-termination EBS volume, presigned
result upload, process timeout, and `finally` cleanup provide independent
safety controls.

```powershell
powershell -ExecutionPolicy Bypass -File `
  inverse_takeoff_kernels/blinded_inverse_recovery_experiment/aws/launch_and_collect.ps1 `
  -Experiment constraint -WorkerSeconds 720
```

The launcher places collected artifacts under `runs/<run-id>/` and removes the
temporary AWS resources after collection.

Completed-run findings and interpretation are recorded in [RESULTS.md](RESULTS.md).
The constraint run completed in 13.4 minutes wall time for an estimated `$0.32`,
then verified the instance, temporary bucket, and security group were gone.
