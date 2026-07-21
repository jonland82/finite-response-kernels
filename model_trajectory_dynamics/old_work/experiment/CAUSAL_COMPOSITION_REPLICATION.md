# Three-seed causal-composition replication: AWS results

## Bottom line

The three-seed AWS replication confirms the first run's main negative result:
the manuscript's operational scalar influence kernels do not reliably compose
by convolution in this miniature language model.

The result is strongest in the early and middle training regions. Median and
10%-trimmed response curves rejected a conservative split-half sampling-noise
control in all three seeds at starts 50 and 200. At start 400 the measured
shape mismatch remained similar in size, but only some seeds exceeded the
noise control because late-training trajectories were highly heterogeneous.

The long-tail result strengthened. Across all 18 measured response curves, the
final 50 of 300 lags contained a median 31.1% of the observed mass, the endpoint
averaged 51.4% of the curve's peak, and 13 of 18 late-window slopes were
positive. These are not well-decayed finite-window kernels.

This does not disprove the manuscript's conditional mathematical theorem. It
shows, across three model initializations, schedules, and synthetic datasets,
that the paper's proposed route from real language-model training to its
independent passive convolution assumptions is not empirically validated here.

## Protocol

Each seed trained 64 independent replicas of the same two-block, width-64
decoder-only Transformer on a closed corpus containing 16 independent red/blue
facts. The run measured 300-update central finite-difference responses at six
injection times and three two-block composition comparisons.

The replication added two controls:

- perturbation scales `epsilon = 0.025, 0.05, 0.10`; and
- 400 random split-half comparisons at each start and aggregation method.

The split-half control compares curve estimates from disjoint sets of 32
replicas. It is conservative because the tested full curves use all 64
replicas and therefore have less sampling noise than either control half.

## Composition across seeds

Total variation (TV) is the fraction of normalized response mass that must be
moved between lags to make the convolved prediction equal the direct
measurement. Zero is a perfect match.

| Start | Mean TV by seed | Median-response TV by seed | Cross-seed median of median-response TV |
|---:|---:|---:|---:|
| 50 | 0.323, 0.474, 0.467 | 0.302, 0.312, 0.287 | **0.302** |
| 200 | 0.584, 0.375, 0.585 | 0.282, 0.380, 0.267 | **0.282** |
| 400 | 0.215, 0.315, 0.517 | 0.330, 0.304, 0.289 | **0.304** |

The robust shape discrepancy is reproducible: a typical direct response needs
roughly 28%--30% of its mass relocated to look like the convolution. The
cross-seed median JS divergences for the median curves were 0.096, 0.093, and
0.117 bits at starts 50, 200, and 400.

Against the aggregation-matched 95th-percentile split-half control:

| Start | Mean exceeds control | Median exceeds control | 10%-trimmed mean exceeds control |
|---:|---:|---:|---:|
| 50 | 3/3 seeds | 3/3 | 3/3 |
| 200 | 0/3 seeds | 3/3 | 3/3 |
| 400 | 1/3 seeds | 2/3 | 1/3 |

The mid-training mean is a poor estimator because rare amplified replicas
dominate it; its median and trimmed versions reject the noise control in every
seed. Late training retains a material median mismatch, but the control result
is mixed, so that region should not be presented as a uniformly significant
failure on this sample size.

## Finiteness and decay

Across the 18 seed/injection combinations:

- mass in the last 50 lags ranged from 0.110 to 0.704, median **0.311**;
- endpoint-to-peak ratio ranged from 0.129 to 0.878, median **0.514**;
- 13 of 18 fitted late log-magnitude slopes were positive.

The experiment observes only 300 lags and therefore cannot prove that the
responses have infinite support. It can say that they had not decayed to a
stable finite kernel within that window. Normalizing at lag 300 consequently
produces a cutoff-dependent kernel.

## Perturbation-scale control

Typical replicas were extremely linear across all three perturbation sizes:

| Epsilon pair | Minimum median-curve cosine across seeds | Minimum median per-replica cosine | Replica comparisons below 0.9 |
|---:|---:|---:|---:|
| 0.025 vs 0.05 | 0.999682 | 0.999828 | 16/192 |
| 0.05 vs 0.10 | 0.999847 | 0.999943 | 22/192 |

In total, 38 of 384 replica/epsilon-pair comparisons fell below 0.9 cosine.
These minority trajectories sometimes carried enormous magnitude and made the
untrimmed aggregate derivative unstable. Thus the central finite difference is
well resolved and locally linear for the typical trajectory, but the response
law is not uniformly stable across training worlds.

This control removes two easy alternative explanations for the original
negative result: ordinary floating-point noise and a universally poor choice of
epsilon. Neither explains the robust convolution mismatch.

## Consequence for the manuscript

The evidentiary position is now:

1. The finite-source information ceiling remains supported, though it is the
   elementary part of the result.
2. Long-lived influence is reproducible across seeds.
3. A 300-update finite, well-decayed response kernel is not demonstrated.
4. The scalar passive convolution model fails reproducibly in early and middle
   training and has mixed evidence late in training.
5. The abstract theorem remains valid under its assumptions; its claimed
   relevance to ordinary model training is the unsupported step.

Another identical seed is unlikely to change this conclusion. The useful next
step is either to revise the manuscript around its explicitly conditional
scope, or begin a separate project on state-dependent response operators rather
than scalar independent-stage convolutions.

## AWS runtime, cost, and cleanup

All three runs used on-demand Linux `c7i.2xlarge` instances in `us-east-1`.

| Seed | Instance | Scientific runtime | Status |
|---:|---|---:|---|
| 20260720 | `i-0fecd61fcf02b6eb9` | 140.90 s | complete |
| 20260721 | `i-0868caaa156f5f37d` | 148.97 s | complete |
| 20260722 | `i-001ec27d6e867185a` | 140.46 s | complete |

Scientific computation totaled 430.33 seconds. Sequential launch, environment
setup, execution, collection, and cleanup took 844.7 seconds (**14 minutes 5
seconds**).

At the current AWS Price List rate of $0.357 per instance-hour, approximately
3.3--3.6 minutes of EC2 lifetime per instance costs about **$0.063** total.
Short-lived gp3 volumes, public IPv4 addresses, and S3 requests add less than a
cent; the conservative total estimate is **under $0.07 before tax**.

All three instances are terminated. Verification found no active tagged
instance, temporary `causal-composition` bucket, or experiment security group.

## Artifacts

Combined analysis:

- [`runs/composition-replication-summary-20260721T014215Z/summary.json`](runs/composition-replication-summary-20260721T014215Z/summary.json)
- [`runs/composition-replication-summary-20260721T014215Z/composition_all_seeds.csv`](runs/composition-replication-summary-20260721T014215Z/composition_all_seeds.csv)
- [`runs/composition-replication-summary-20260721T014215Z/robust_noise_floor_all_seeds.csv`](runs/composition-replication-summary-20260721T014215Z/robust_noise_floor_all_seeds.csv)
- [`runs/composition-replication-summary-20260721T014215Z/linearity_all_seeds.csv`](runs/composition-replication-summary-20260721T014215Z/linearity_all_seeds.csv)
- [`runs/composition-replication-summary-20260721T014215Z/responses_all_seeds.csv`](runs/composition-replication-summary-20260721T014215Z/responses_all_seeds.csv)

Complete seeded runs:

- [`runs/composition-seed20260720-20260721T012818Z`](runs/composition-seed20260720-20260721T012818Z/), archive SHA-256 `ea8349e4720450124db357775374262028f44527e4e534b2b1ee355204d3359e`;
- [`runs/composition-seed20260721-20260721T013258Z`](runs/composition-seed20260721-20260721T013258Z/), archive SHA-256 `de78535d379fd2eb27aad5a64fc9fce7a3efb34b16827bade5f702c17048a6b3`;
- [`runs/composition-seed20260722-20260721T013755Z`](runs/composition-seed20260722-20260721T013755Z/), archive SHA-256 `4e5fc700d9bd6f27346c35a0569a5784ecac477e422374a8a657d7fd28b06ace`.

Every archive contains raw per-replica derivative arrays, lag CSVs, response
summaries, robust composition summaries, noise controls, epsilon checks,
figures, metadata, and the instance console log. The executed runner SHA-256
was `9e3adf94f7ac155ec23c9bf535f916759df613d56b4bd6285e4cd7e1af201e62`
for all three runs and matches the retained
[`run_causal_composition.py`](run_causal_composition.py). Combined statistics
can be regenerated with
[`analyze_composition_replication.py`](analyze_composition_replication.py).
