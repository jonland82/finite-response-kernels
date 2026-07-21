# Closed-source language-model pilot: AWS results

## Bottom line

The July 21, 2026 AWS pilot provides a clean finite-horizon demonstration of
**observable closed-source exhaustion** in a miniature language model. It does
not verify the paper's full passive-causal dilution theorem.

In the closed condition, the model's fixed red/blue decoder reached a
variational information lower bound of **15.9887 bits from a 16-bit source**
(99.93%; bootstrap 95% interval 15.9882--15.9892). From update 192 through
update 400, another 208 updates increased this bound by only **0.0330 bits**,
while answer-token loss still fell by **68.2%**. This is the predicted
separation between exhausted source innovation and continued optimization.

The growing-source control behaved differently. As fresh record/code facts
expanded the available source from 16 to 32 bits, its fixed-decoder lower bound
continued rising and ended at **31.9114 of 32 bits** (99.72%; bootstrap 95%
interval 31.8590--31.9467).

The influence experiment found broad, non-instantaneous responses, but it also
found substantial horizon truncation and signed cancellation. Consequently,
the run supports the source-exhaustion narrative but does not establish finite
passive kernels, stage independence, convolutional composition, or the
entropy-power concentration law.

## Experiment

The model was a decoder-only Transformer trained from scratch:

- 2 Transformer blocks;
- width 64, 4 attention heads, feed-forward width 256;
- 128 independent training replicas with identical initial parameters;
- 400 optimizer updates, equivalent to 100 passes through the initial corpus;
- AdamW with learning rate 0.003 and weight decay 0.001.

Each closed corpus consisted of 16 sentences of the form

```text
<bos> record id07 code red <eos>
```

where each `red`/`blue` answer was an independent fair bit. Thus the random
source had exactly 16 bits of entropy. The closed condition replayed those
facts. The control introduced one new independently coded record every 20
updates until 32 bits were available.

At each checkpoint, the observable state was the vector of red-versus-blue
logit margins for all record prompts. For the primary derived analysis, the
fixed decoder

```text
q(D_i = red | z_i) = sigmoid(z_i)
```

gives a variational lower bound on information in those observable logits. A
second, grouped cross-fitted calibration decoder was recorded by the runner as
a deliberately conservative alternative. Both are lower bounds on observable
information, not measurements of exact mutual information in the complete
real-valued weights.

## Source results

| Condition | Update | Available source | Fixed-decoder lower bound | Decoder accuracy | Code loss |
|---|---:|---:|---:|---:|---:|
| Closed | 0 | 16 bits | -0.0038 bits | 50.49% | 3.6316 |
| Closed | 64 | 16 bits | 7.1328 bits | 83.64% | 0.4635 |
| Closed | 128 | 16 bits | 15.6746 bits | 99.66% | 0.0233 |
| Closed | 192 | 16 bits | 15.9557 bits | 100.00% | 0.00628 |
| Closed | 400 | 16 bits | 15.9887 bits | 100.00% | 0.00200 |
| Growing source | 128 | 22 bits | 20.5415 bits | 98.83% | 0.0647 |
| Growing source | 192 | 25 bits | 23.8043 bits | 99.38% | 0.0428 |
| Growing source | 256 | 28 bits | 27.5635 bits | 99.89% | 0.0217 |
| Growing source | 320 | 31 bits | 30.1823 bits | 99.42% | 0.0303 |
| Growing source | 400 | 32 bits | 31.9114 bits | 99.98% | 0.00406 |

The small negative value at initialization is allowed: a variational lower
bound may lie below zero even though true mutual information cannot. Its
bootstrap interval contains zero.

The runner's grouped cross-fitted decoder produced a more conservative closed
plateau of 12.176 bits. That decoder pooled calibration across independent
worlds and penalized confidence heterogeneity. The fixed model decoder is the
primary result because it is specified before observing any held-out world and
can be calculated directly from the archived logits.

## Influence results

One training example was assigned weights `1 + 0.1` and `1 - 0.1` in paired
branches at updates 50, 150, and 300. Subsequent minibatches were identical.
The central finite difference in the example's red/blue margin was followed
for 50 updates and normalized over that finite window.

| Injection update | Effective lag count | Peak mass | Mass in final 10 lags | Signed/absolute ratio |
|---:|---:|---:|---:|---:|
| 50 | 35.86 | 0.052 | 0.335 | 0.531 |
| 150 | 30.31 | 0.079 | 0.146 | 0.966 |
| 300 | 49.84 | 0.030 | 0.182 | 0.246 |

These curves rule out an approximately instantaneous response at the measured
resolution: no lag contains more than 8% of the finite-window magnitude. But
they do **not** establish a finite normalized response kernel. Between 14.6%
and 33.5% of measured magnitude lies in the final ten lags, so material
response may continue beyond the observation window.

The signed-to-absolute ratios at updates 50 and 300 also show strong
cross-replica cancellation. The manuscript's absolute-value kernel can still
be constructed as a coarse-graining, but raw training influence is not well
described as uniformly positive passive transport in those conditions.

No independently measured multi-stage response was compared with a convolution
of local stage kernels. The central convolution assumption therefore remains
untested.

## What this does and does not show

Supported by this run:

- a fixed random corpus supplied a bounded amount of recoverable information;
- recoverable information saturated under replay;
- optimization continued measurably after information was effectively
  saturated;
- admitting fresh random facts restored continuing information gains;
- example influence was temporally extended rather than delta-like.

Not established by this run:

- the exact conditional mutual information increments
  `I(D; W_t | W_0:t-1)`;
- an asymptotic limit as training time tends to infinity;
- a uniform anti-concentration bound independent of measurement resolution;
- passive signed dynamics;
- independence or convolution of training stages;
- entropy-power growth of an independently observed composite kernel.

The appropriate paper claim is therefore: **finite-horizon empirical support
for the observable source-exhaustion consequence, with mixed evidence for the
passive-response model.**

## AWS execution and cost

The account's quotas for both on-demand and Spot G/VT instances were zero. The
successful run therefore used one `c7i.2xlarge` in `us-east-1` with the
AWS-published Amazon Linux 2023 AMI and CPU-only PyTorch 2.10.0.

- Successful instance: `i-0e56a7ec222026546`
- Launch: 2026-07-21 00:38:41 UTC
- Termination initiated: 2026-07-21 00:40:20 UTC
- Instance lifetime: approximately 99 seconds
- Experiment runtime after setup: 36.09 seconds
- Experiment exit status: 0 (`complete`)

Two earlier bootstrap attempts used a GPU-oriented AMI on the CPU instance and
were terminated four seconds after launch. With EC2's 60-second minimum and a
current planning rate of $0.357/hour for `c7i.2xlarge`, estimated compute across
all three launches was approximately **$0.022**. Root-volume, S3-request, and
public-IPv4 charges add substantially less than one cent at these durations, so
the total estimated AWS cost is **under $0.03**, before tax.

All instances are terminated. The temporary S3 buckets and no-ingress security
groups were deleted after collection. No endpoint, NAT gateway, Elastic IP, or
persistent volume was created.

The local collector encountered a Windows character-encoding error while
exporting a redundant EC2 console copy after the archive had been downloaded.
The complete `instance_console.log` was already included inside the archive,
so this did not affect the experiment or its captured evidence.

## Artifacts

The successful run is under [`runs/aws-20260721T003831Z`](runs/aws-20260721T003831Z/):

- [`results/metadata.json`](runs/aws-20260721T003831Z/results/metadata.json): exact configuration and environment;
- [`results/trajectory.csv`](runs/aws-20260721T003831Z/results/trajectory.csv): runner checkpoint metrics;
- [`results/observables.npz`](runs/aws-20260721T003831Z/results/observables.npz): raw archived logit margins and per-bit statistics;
- [`results/direct_information.csv`](runs/aws-20260721T003831Z/results/direct_information.csv): derived fixed-decoder statistics and intervals;
- [`results/influence_lags.csv`](runs/aws-20260721T003831Z/results/influence_lags.csv): raw lag curves;
- [`results/influence_summary.csv`](runs/aws-20260721T003831Z/results/influence_summary.csv): response summaries;
- [`results/direct_information.png`](runs/aws-20260721T003831Z/results/direct_information.png): primary source comparison;
- [`results/closed_source_pilot.png`](runs/aws-20260721T003831Z/results/closed_source_pilot.png): original runner summary;
- [`results/instance_console.log`](runs/aws-20260721T003831Z/results/instance_console.log): complete bootstrap and run log;
- [`results.tar.gz`](runs/aws-20260721T003831Z/results.tar.gz): downloaded immutable archive, SHA-256 `020438b02b397a6b9036d30bf636a81af35f1a5717893eed84ff8441e3c5b26f`.

The executed runner's SHA-256 was
`b94e64d3b80850f2a72fcc4ac00251bd16cd45a1e9817182f1fb68d7caa60777`,
as recorded in `metadata.json`. Derived statistics can be regenerated with
[`analyze_aws_run.py`](analyze_aws_run.py).

The subsequent 300-update response and two-block convolution test is reported
separately in
[`CAUSAL_COMPOSITION_RESULTS.md`](CAUSAL_COMPOSITION_RESULTS.md).
