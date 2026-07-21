# Long-horizon causal-composition test: AWS results

> Follow-up: the three-seed, three-epsilon controlled replication is reported
> in
> [`CAUSAL_COMPOSITION_REPLICATION.md`](CAUSAL_COMPOSITION_REPLICATION.md).

## Bottom line

The July 21, 2026 follow-up does **not** validate the manuscript's proposed
passive convolutional description of language-model training. In this tiny
closed-source language model, the measured influence curves often remained
large at the end of a 300-update window, and a measured two-block response did
not match the convolution of the two separately measured block responses.

This is a negative result for the manuscript's **operational scalar-kernel
construction**, not a disproof of the conditional theorem. The theorem is
mathematically true once independent passive convolutional stages are assumed.
The experiment shows that those assumptions did not emerge from this training
system under the paper's proposed influence-magnitude measurement.

The earlier source-exhaustion result is unchanged: the 16-bit closed corpus
could not provide more than 16 bits of new corpus information. What remains
unsupported is the novel bridge from a real optimizer trajectory to finite,
independent response kernels that compose by convolution.

## The actual experiment

The experiment trained the same miniature decoder-only Transformer used in the
first pilot: two Transformer blocks, width 64, four attention heads, and AdamW.
Sixty-four independent corpus replicas each contained 16 random red/blue facts.
The facts were replayed throughout training; no new source information entered.

At a chosen update, one selected training example was given weight
`1 + epsilon` in one branch and `1 - epsilon` in an otherwise identical branch.
The red-versus-blue logit difference between the branches was followed for the
next 300 updates. Dividing that difference by `2 * epsilon` produced a central
finite-difference response curve for each replica.

For each start time `s`, the test compared:

- the directly measured influence magnitude from `s` through `s + 200`; and
- the convolution of the measured `s` to `s + 100` curve with a separate
  response injected at `s + 100` and observed for another 100 updates.

Both sides were normalized to unit mass, exactly as an operational test of the
manuscript's passive influence-magnitude kernel. A perfect composition would
give zero Jensen--Shannon (JS) divergence, zero total variation (TV), and zero
Wasserstein displacement.

## Composition result

| Start update | JS divergence | Total variation | Wasserstein / 200-update horizon |
|---:|---:|---:|---:|
| 50 | 0.119 bits [0.090, 0.145] | 0.323 [0.262, 0.376] | 0.104 [0.083, 0.127] |
| 200 | 0.336 bits [0.093, 0.558] | 0.584 [0.285, 0.780] | 0.153 [0.092, 0.310] |
| 400 | 0.053 bits [0.036, 0.204] | 0.215 [0.159, 0.473] | 0.060 [0.019, 0.177] |

The intervals are 95% replica-bootstrap intervals. TV has the easiest reading:
the predicted curve would need 22% to 58% of its normalized response mass moved
to different lags to equal the measured curve. The middle trajectory is an
especially clear failure; the directly measured response develops a sharp late
surge that the convolution does not predict.

The result is not an artifact of one extreme replica. Replacing the mean
absolute response by a median across replicas gives TV distances of 0.302,
0.282, and 0.330. Ten-percent trimmed means give 0.321, 0.283, and 0.262.
Thus all three robust summaries retain a material shape mismatch.

## The responses did not die out within the window

| Injection update | Effective lags | Mass in final 50 lags | Endpoint / peak | Late log slope per update |
|---:|---:|---:|---:|---:|
| 50 | 265.3 | 0.110 | 0.267 | -0.00063 |
| 150 | 156.4 | 0.525 | 0.130 | +0.01706 |
| 200 | 110.6 | 0.451 | 0.427 | +0.03436 |
| 300 | 194.7 | 0.249 | 0.526 | +0.00017 |
| 400 | 227.3 | 0.113 | 0.212 | -0.00455 |
| 500 | 245.7 | 0.134 | 0.129 | +0.00177 |

Every response still had a non-negligible endpoint. Four of six late-window
slopes were positive, and two responses placed roughly half their observed mass
in the final 50 lags. This finite run cannot prove an infinite response tail,
but it does show that a 300-update cutoff is not a demonstrated finite or
well-decayed kernel. Normalizing these truncated curves makes their reported
shape depend on the observation cutoff.

The raw signed responses also cancel substantially: the ratio of aggregate
signed to aggregate absolute response ranged from 0.176 to 0.828. The
nonnegative kernel is therefore a coarse-graining of active signed dynamics,
as the manuscript acknowledges, rather than the optimizer's raw response law.

## Perturbation-size sanity check

At injection update 200, the flattened aggregate derivatives from
`epsilon=0.05` and `epsilon=0.10` had cosine similarity 0.0098. That alarming
aggregate number is driven mostly by rare unstable trajectories, not numerical
roundoff:

- the median per-replica cosine was 0.99998;
- 62 of 64 replicas had cosine at least 0.9;
- replicas 27 and 54 were unstable across perturbation sizes;
- replica 27 alone supplied 99.6% and 99.2% of the two arrays' squared energy;
- the median absolute response curves had cosine 0.99991.

So a typical replica was locally linear at these perturbation sizes, while a
small fraction underwent large trajectory-level amplification. This distinction
does not rescue the convolution test: median and trimmed composition curves
still disagree. It does warn against reporting only an unqualified ensemble
mean, and it is evidence against a uniform stable-kernel description over
training worlds.

## What this says about the paper

The clean interpretation is:

1. **Finite-source exhaustion:** supported in the earlier pilot, but it is the
   elementary side of the paper and follows from the finite source budget.
2. **Long-lived influence:** supported, although several curves look more like
   delayed amplification than passive decay.
3. **Finite passive response kernels:** not established in the 300-update
   window; the tails were often still active.
4. **Independent stages that compose by convolution:** rejected for this
   operational scalar construction in this toy language model.
5. **The abstract no-go theorem:** not falsified, because it assumes the kernel
   properties in items 3 and 4. Its applicability to ordinary model training
   is what the experiment fails to verify.

In plain English: we verified the obvious closed-box information ceiling, then
tested the paper's interesting claim about how training influence propagates.
The influence did not behave like two independent passive delays whose shapes
can simply be convolved. At present the paper has a valid conditional theorem,
but not empirical evidence that its causal assumptions describe this language
model.

## AWS execution, runtime, and cost

The capped run used one on-demand `c7i.2xlarge` in `us-east-1` with CPU-only
PyTorch 2.10.0.

- Instance: `i-0ba5eebced1d2922e`
- Launch: 2026-07-21 01:04:55 UTC
- Termination initiated: 2026-07-21 01:08:02 UTC
- Billable instance lifetime: approximately 187 seconds
- Scientific computation: 124.80 seconds
- End-to-end launch, setup, collection, and cleanup: about 4 minutes
- Exit status: 0 (`complete`)

The AWS Price List returned **$0.357/hour** for on-demand Linux
`c7i.2xlarge` in Northern Virginia, effective July 2026. Compute was therefore
approximately **$0.0185**. The short-lived root volume, public IPv4 use, and S3
requests/storage bring the estimated total to approximately **$0.019**, safely
under **$0.02** before tax and far below the $10 cap.

The instance is terminated. Post-run checks found no remaining
`causal-composition` S3 bucket, security group, or active tagged EC2 instance.

## Artifacts and reproducibility

The captured run is under
[`runs/composition-20260721T010442Z`](runs/composition-20260721T010442Z/):

- [`results/RUN_REPORT.md`](runs/composition-20260721T010442Z/results/RUN_REPORT.md): automatically generated primary metrics;
- [`results/metadata.json`](runs/composition-20260721T010442Z/results/metadata.json): configuration, environment, status, and code hashes;
- [`results/response_lags.csv`](runs/composition-20260721T010442Z/results/response_lags.csv): lag-level signed and absolute results;
- [`results/response_derivatives.npz`](runs/composition-20260721T010442Z/results/response_derivatives.npz): raw per-replica response arrays;
- [`results/composition_summary.csv`](runs/composition-20260721T010442Z/results/composition_summary.csv): composition metrics and bootstrap intervals;
- [`results/linearity_derivatives.npz`](runs/composition-20260721T010442Z/results/linearity_derivatives.npz): perturbation-size check;
- [`results/robustness_analysis.json`](runs/composition-20260721T010442Z/results/robustness_analysis.json): reproducible robust and outlier-sensitive summaries;
- [`results/causal_composition.png`](runs/composition-20260721T010442Z/results/causal_composition.png): measured-versus-convolved curves and long responses;
- [`results/instance_console.log`](runs/composition-20260721T010442Z/results/instance_console.log): complete remote execution log;
- [`results.tar.gz`](runs/composition-20260721T010442Z/results.tar.gz): downloaded archive, SHA-256 `966569e178985482a8ac087d6c5fcaddb6b46f5e52a42c19da3e05668584cdc2`.

The executed composition runner SHA-256 was
`5e20eeb39994aff19d135d08e8a2b0babae929d70f0d4c4c2438e4cab14db2e1`;
the imported base experiment runner SHA-256 was
`b94e64d3b80850f2a72fcc4ac00251bd16cd45a1e9817182f1fb68d7caa60777`.
The robustness file can be regenerated with
[`analyze_composition_run.py`](analyze_composition_run.py).
