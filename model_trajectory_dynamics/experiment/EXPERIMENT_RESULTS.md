# AWS LLM takeoff-kernel experiment

Run date: 2026-07-20/21 (America/New_York / UTC)  
Primary run: [`runs/llm-takeoff-20260721T022627Z`](runs/llm-takeoff-20260721T022627Z/)  
Status: training complete; local post-processing complete

## Bottom line

On this deliberately small language-model task, takeoff was not an instantaneous
one-checkpoint jump. All six confirmation seeds took 600--1,250 optimizer updates
to move from the first observed 50% held-out accuracy to the first observed 90%
(median 1,000 updates), while checkpoints were only 50 updates apart.

The full response was also incompatible with one nonnegative unimodal kernel at
the tested scales. A basis-free test selected three high--low--high delivery
windows on even checkpoints, then found a positive valley certificate on the
withheld odd checkpoints in all 6/6 seeds and at all three tested widths (18/18
seed--width evaluations). A two-sigmoid mixture also had lower checkpoint-level
BIC and cross-validated checkpoint RMSE than both a delta step and a single
sigmoid in all six seeds. Together these results establish multiplicity under a
weak shape assumption, then describe that multiplicity as two stages.

This is evidence about one controlled modular-arithmetic transformer, not a
proof about all LLMs and not a verification of a universal theorem. It rejects
the literal instantaneous-takeoff model at 50-update resolution for this task.

## What was trained

The model was a real, decoder-only causal language model trained from random
initialization:

- input token sequence: `[a, +, b, =]`;
- target next token: `(a + b) mod 67`;
- vocabulary: 70 tokens (67 integers plus three special tokens);
- architecture: one Transformer decoder block, width 64, four attention heads,
  feed-forward width 256, 59,136 trainable parameters;
- data: all 4,489 ordered input pairs, split once per seed into 1,347 training
  examples (30%) and 3,142 held-out examples (70%);
- optimizer: AdamW, learning rate 0.001, weight decay 1.0, batch size 512;
- observation: full train and held-out loss/accuracy every 50 optimizer updates;
- confirmation: six independent initialization/data-split seeds after a
  four-condition pilot selected the clearest delayed-generalization setting.

This is the standard grokking-style setting: the model memorizes its training
subset early, then generalizes the modular rule later. Training accuracy reached
99% at update 350 or 400 in every confirmation run. Held-out accuracy did not
reach 50% until updates 3,900--13,450.

## Direct measurements

| Seed | First 25% | First 50% | First 90% | 25% to 90% | 50% to 90% |
|---:|---:|---:|---:|---:|---:|
| 20260800 | 2,250 | 4,300 | 4,900 | 2,650 | 600 |
| 20260801 | 1,800 | 3,900 | 4,650 | 2,850 | 750 |
| 20260802 | 9,800 | 13,450 | 14,500 | 4,700 | 1,050 |
| 20260803 | 500 | 4,950 | 6,050 | 5,550 | 1,100 |
| 20260804 | 10,050 | 11,800 | 13,050 | 3,000 | 1,250 |
| 20260805 | 900 | 5,100 | 6,050 | 5,150 | 950 |
| **Median** | -- | -- | -- | **3,850** | **1,000** |

The bounded-score kernel was constructed from the isotonic held-out accuracy
profile. Its full 10--90% mass width was 4,262--9,611 updates (median 5,781),
because it includes both the early partial improvement and the late rise. Its
effective support was about 19--30 nonzero checkpoints. Thus even the sharp
late part was resolved across 12--25 checkpoints, and the entire response was
far more distributed.

## Can one response mode explain the curve?

Not if a single response is assumed to be nonnegative and unimodal. For three
ordered, equal-width windows, define the valley certificate

`V_w = min(early average rate, late average rate) - middle average rate`.

Every nonnegative unimodal kernel must have `V_w <= 0`: once its rate falls, it
cannot rise again without creating another mode. The test chose the three window
locations using only even-numbered checkpoints. It then froze those locations,
isotonic-projected the odd-numbered checkpoints separately, and evaluated the
certificate on that withheld half of the trajectory.

At the primary 1,000-update width, `V_1000` was positive in every seed, ranging
from 0.049 to 0.215 normalized response gain per 1,000 updates (median 0.134).
The result remained positive for every seed at widths 800, 1,000, and 1,500
updates. A deterministic endpoint-error calculation shows that the weakest
primary result survives arbitrary errors up to 1.2 percentage points at every
window endpoint; the median survives 3.3 points and the strongest 5.4 points.

The implementation was also tested on noiseless synthetic controls. A single
logistic response produced a nonpositive certificate, while two separated
responses produced a positive one. This is a shape theorem plus a held-out test,
not evidence that the two modes correspond to two distinct neural mechanisms.

## Does the curve decompose?

Three models were fit to the held-out accuracy trajectory:

1. an instantaneous delta/step model;
2. one finite-width logistic response;
3. a weighted mixture of two finite-width logistic responses.

Parameters were fit using the held-out correct counts. Model selection was
then evaluated at the checkpoint level, because the same test examples are
reused at every checkpoint and adjacent observations are temporally correlated.
Alternating checkpoints were excluded from fitting and used for cross-validated
RMSE. The RMSE ranges across the six seeds were:

| Model | Cross-validated checkpoint RMSE |
|---|---:|
| Delta step | 0.106--0.132 |
| One logistic | 0.051--0.071 |
| Two-logistic mixture | **0.030--0.047** |

The mixture also won checkpoint-level BIC in all six runs. Its fitted component
centers tracked the visible early and late rises. This is positive evidence for
more than one response component, but it does not prove that the components are
separate physical mechanisms.

A different decomposition test fit one versus two decaying exponentials to the
post-onset residual, over both a full-tail and a local 5--95% window. That test
selected one exponential in all 12 fits. The results are compatible: the data
support two separated *transition stages*, but do not require two exponential
*settling modes* inside either stage.

After centering and scaling each kernel, the median pairwise Wasserstein distance
was 0.206. The shapes recur approximately, while takeoff time and duration vary
substantially by seed; this is descriptive evidence, not yet a universality law.

## What this verifies, and what it does not

The experiment gives two useful results relative to the manuscript's claims:

- **Finite takeoff:** yes, for this task and resolution. The delta model loses in
  every seed, and the late 50--90% rise alone spans a median of 20 checkpoints.
- **More than one response mode:** yes, conditional on a nonnegative latent
  response and the tested 800--1,500-update scales; the held-out valley is
  positive in all 18 seed--width evaluations.
- **Descriptive decomposition:** yes for two transition components in accuracy;
  no evidence here for two exponential residual modes and no identification of
  two physical mechanisms.

It does **not** verify or falsify a general no-go theorem for closed-source LLMs.
The model is small, the task is synthetic, one hyperparameter condition was
selected after a pilot, and confirmation changes only the random seed. A stronger
next test should preregister the estimators and repeat them across task families,
model widths/depths, data fractions, and a larger open-weight language model.

## AWS execution, runtime, and cost

The successful job used one on-demand `c7i.2xlarge` instance in `us-east-1` with
a 25 GB delete-on-termination `gp3` volume. The remote workload took 392.4 seconds;
launch, package installation, artifact collection, and teardown brought total
launcher wall time to 516.6 seconds.

An earlier attempt (`llm-takeoff-20260721T021615Z`) ran for 458.3 seconds and
failed during confirmation because forked PyTorch workers tried to reset an
already initialized thread pool. It produced pilot data but is not counted as a
confirmation experiment. The runner was changed to fresh spawned workers before
the successful rerun.

At the AWS Price List rate captured on the run date, `c7i.2xlarge` cost
$0.357/hour. Conservative wall-time charging gives:

- successful attempt: about $0.051;
- failed attempt: about $0.045;
- compute total: about $0.097;
- storage and request charges: below one cent at this duration and data volume;
- **estimated total for both attempts: about $0.10, conservatively below $0.12.**

The offered 20-minute extension was not used. Both instances are terminated,
both temporary buckets and security groups are deleted, and no tagged experiment
volumes remain.

## Artifacts and provenance

- Main visual: [`takeoff_trajectories.png`](runs/llm-takeoff-20260721T022627Z/results/takeoff_trajectories.png)
- Paper results overview: [`trajectory_results_overview.pdf`](runs/llm-takeoff-20260721T022627Z/results/trajectory_results_overview.pdf)
- Explicit trajectory/kernel sum: [`kernel_composition_representative.pdf`](runs/llm-takeoff-20260721T022627Z/results/kernel_composition_representative.pdf)
- Two-component fits for all seeds: [`kernel_composition_all_seeds.pdf`](runs/llm-takeoff-20260721T022627Z/results/kernel_composition_all_seeds.pdf)
- Figure-generation source: [`plot_kernel_composition.py`](plot_kernel_composition.py)
- Threshold crossings: [`transition_thresholds.csv`](runs/llm-takeoff-20260721T022627Z/results/transition_thresholds.csv)
- Transition fits: [`transition_model_comparison.csv`](runs/llm-takeoff-20260721T022627Z/results/transition_model_comparison.csv)
- Kernel statistics: [`takeoff_kernel_summary.csv`](runs/llm-takeoff-20260721T022627Z/results/takeoff_kernel_summary.csv)
- Held-out one-mode test: [`single_mode_valley.csv`](runs/llm-takeoff-20260721T022627Z/results/single_mode_valley.csv)
- Modal fits: [`modal_decomposition.csv`](runs/llm-takeoff-20260721T022627Z/results/modal_decomposition.csv)
- Seed summary: [`run_summary.csv`](runs/llm-takeoff-20260721T022627Z/results/run_summary.csv)
- Machine-readable result: [`analysis_summary.json`](runs/llm-takeoff-20260721T022627Z/results/analysis_summary.json)
- Raw collected archive: [`results.tar.gz`](runs/llm-takeoff-20260721T022627Z/results.tar.gz)

SHA-256 identifiers:

- training script used on AWS: `a380fa68771d786e9c2ea56706da697e7af7c906351b967dc9a260c265826ddc`;
- final local analysis script: `19c868b4efb184254cc969cb1c837e67702906c4d1343cb8d5ad9447e5e6fa8e`;
- collected AWS archive: `85cfa25be9e4eb09a29538c28d1335da7b87edcb99491b3a48acfa10da7aa03a`.

The cloud training completed, but its first post-processing invocation failed
while writing heterogeneous model rows to CSV. The captured raw archive was not
modified. The corrected analyzer was run locally against that archive; details
are recorded in the run's
[`POSTPROCESSING_PROVENANCE.md`](runs/llm-takeoff-20260721T022627Z/POSTPROCESSING_PROVENANCE.md).
