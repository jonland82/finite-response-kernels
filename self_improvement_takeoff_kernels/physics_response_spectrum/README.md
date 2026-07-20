# Physics Response Spectrum Experiment

## Status

Completed on July 19--20, 2026. The exact-fitness pilot produced no exact
answers, triggering the separately configured continuous-fitness fallback
preregistered below. The primary and matched context controls are reported in
[RESULTS.md](RESULTS.md). The standalone six-page manuscript is available as
[physics_response_spectrum.pdf](physics_response_spectrum.pdf), with source in
[physics_response_spectrum.tex](physics_response_spectrum.tex).

## Objective

Replace the artificial-record population in *The Response Spectrum of Recursive
Self-Improvement* with populations of genuine physics solution attempts. Use a
single continuous selection parameter to make the same fixed model exhibit
collapse, neutral churn, and improvement, then test whether any useful effect
survives cross-model criticism or weight-level fine-tuning.

The central question is not merely whether filtering improves answers. It is
whether recursive solution populations have measurable response geometry:
thresholds, reversals, delayed collapse, diversity loss, rapid improvement, or
different critical behavior across physics domains.

## Experimental layers

The three layers must be reported separately because they support different
claims.

1. **Fixed-weight recursive prompting:** the model weights remain fixed while
   candidate solutions are generated, revised, and selected. This measures
   inference-time population dynamics.
2. **Solver--critic recursion:** a second call or stronger model critiques the
   solver. This measures interaction dynamics and changes the generation
   channel.
3. **Fine-tuning:** selected solution traces modify model weights. Evaluation on
   untouched problems tests genuine transfer, improvement, or degradation.

The first layer is the primary experiment. The later layers are gated on an
interesting and reproducible first result.

## Dataset

Begin with the 100 fully annotated examples in
[PHYBench](https://huggingface.co/datasets/Eureka-Lab/PHYBench). These problems
are text-only, span several physics domains, have single symbolic answers, and
include reference answers and solutions. PHYBench also provides symbolic
evaluation code and its Expression Edit Distance (EED) score.

Create a deterministic, stratified split by physics domain and difficulty. The
symbolic-parser gate retained 78 of 100 records, so the realized split is:

- 16 calibration problems for prompt, parser, and scorer development;
- 31 spectrum problems for the reported recursive-prompting experiment;
- 31 locked transfer problems that are not inspected or used for selection,
  prompt tuning, or fine-tuning-data construction.

Record the dataset revision, file hashes, problem IDs, and split seed. Never
send reference solutions or gold answers to the generator.

## Candidate format

For each problem, a candidate is a complete solution attempt with a stable ID
and the following machine-readable output:

```json
{
  "derivation": "Concise step-by-step derivation in LaTeX.",
  "final_answer": "Single symbolic or numerical expression.",
  "unit": "Unit if required, otherwise null"
}
```

Invalid JSON, missing answers, forbidden prose outside the object, or
unparseable expressions remain failed attempts. They are recorded and scored;
they are not silently repaired.

## Initial population and recursive reproduction

For each spectrum problem:

1. Generate four independent round-zero solutions from the problem alone.
2. At each later round, give each of the four surviving candidates to a new Nova
   Lite call together with the original problem.
3. Ask the model to diagnose the attempt, correct it if necessary, and return
   one child solution in the required JSON format.
4. Score the four children without showing the scores or answers to Nova.
5. Resample four survivors **with replacement** from the four children using the
   selection law below. Parents are never eligible to survive directly.
6. Repeat for six recursive rounds.

Let $m=4$ be the population size, let $X_r=(x_{r1},\ldots,x_{rm})$ be the
population at round $r$, and let $\mathcal G$ be Nova Lite's stochastic revision
operator. Before selection, the child population is

$$
Y_{r+1}=(y_{r+1,1},\ldots,y_{r+1,m}),
\qquad y_{r+1,i}\sim\mathcal G(\text{problem},x_{ri}).
$$

Selection maps $Y_{r+1}$ to $X_{r+1}$. Because sampling is with replacement, a
single child can have several descendants. Those copies receive distinct IDs
and independent model samples in the following round. This makes concentration
and recovery observable while avoiding elitism: even a correct parent can be
lost if its child is wrong.

Use fixed Amazon Nova Lite weights through Amazon Bedrock in `us-east-1`. Start
with temperature `0.8`, top-p `0.9`, and a maximum of 900 output tokens. Store
the exact prompt template and inference configuration with every run.

The model sees the problem and its inherited solution attempt. It never sees
the reference answer, fitness, selection parameter $\rho$, or selector's reason
for retaining an attempt.

## Deterministic answer scoring

Let $C(x)\in\{0,1\}$ be the primary correctness score of candidate $x$, based
only on its final answer:

- canonicalize LaTeX and parse supported expressions with SymPy;
- set $C(x)=1$ for symbolic equivalence or an accepted numerical answer within
  the dataset tolerance;
- set $C(x)=0$ for an incorrect answer;
- enforce unit compatibility when the problem requires a unit;
- assign $C(x)=0$ to missing, invalid, or unparseable answers.

Use $s(x)=C(x)$ as the primary selection fitness. PHYBench EED remains a
secondary diagnostic and tie-analysis variable, not the default fitness. A
surface-level expression resemblance is not necessarily physical correctness.

If binary fitness is too sparse, a continuous score may be introduced only in
a separately configured and reported experiment. The preferred extension is a
validated numerical-substitution score: compare candidate and reference
expressions on several admissible substitutions, aggregate relative errors, and
set exact equivalents to one. The substitutions, tolerances, singularity
handling, and unit checks must be preregistered and tested manually before that
score affects selection.

### Continuous-fitness fallback preregistration

This subsection was added after the exact-fitness pilot and before making any
fallback calls. The binary pilot had zero exact answers at every round and
selection value, so it could not distinguish collapse from improvement.

The fallback uses the official PHYBench Expression Edit Distance transformation
as the selection fitness $s_{\mathrm{EED}}\in[0,1]$. Exact symbolic matches have
$s_{\mathrm{EED}}=1$. For nonmatches with gold expression-tree size $L$ and
extended Zhang--Shasha distance $d$, PHYBench defines

$$
s_{\mathrm{EED}}=
\max\!\left(0,0.6-\frac{d}{L}\right).
$$

Invalid or unparseable answers receive zero. The EED fallback first repeats the
five-problem, three-round pilot with the same population, $\rho$ grid, model,
and sampling parameters. It proceeds to the 31-problem spectrum split only if
the pilot has nonzero within-population EED variance and no material scorer or
format failures. The primary fallback uses two paired replicates and six
recursive rounds. Exact accuracy remains a separately reported outcome and EED
results must not be described as verified physical correctness.

This is a deliberate deviation from the preferred numerical-substitution
extension: the official, deterministic EED implementation was already available
and could be validated before additional paid calls, whereas designing safe
substitution domains for dozens of heterogeneous expressions would require a
separate manual study.

EED and any later continuous score are reported separately from exact accuracy.
Scoring must be tested against the reference answers, deliberately incorrect
answers, and known equivalent expressions before any paid run. No LLM judge is
used in the primary oracle experiment.

## Continuous selection law

For child $i$ with fitness $s_i=s(y_{r+1,i})$, define

$$
w_i(\rho)=\exp\!\left[\rho\left(s_i-\frac12\right)\right],
\qquad
p_i(\rho)=\frac{w_i(\rho)}{\sum_{\ell=1}^{m}w_\ell(\rho)}.
$$

Draw the next population counts from

$$
(N_1,\ldots,N_m)\sim
\operatorname{Multinomial}\!\left(m;p_1(\rho),\ldots,p_m(\rho)\right),
$$

so child $i$ appears $N_i$ times in $X_{r+1}$. Sweep

$$
\rho\in\{-6,-2,0,2,6\}.
$$

Interpretation:

- $\rho=-6$: strong preference for incorrect solutions; collapse control;
- $\rho=-2$: weak negative selection;
- $\rho=0$: score-blind random propagation; neutral control;
- $\rho=2$: weak correctness selection;
- $\rho=6$: strong correctness selection; improvement condition.

With binary fitness, the odds ratio of selecting a correct child over an
incorrect child is

$$
\frac{p(\text{correct})}{p(\text{incorrect})}=e^{\rho}.
$$

Thus $\rho$ has a direct interpretation as log selection strength. When every
child has the same correctness, selection is uniform and improvement must wait
for the generator to create fitness variation.

This is a calibrated one-parameter family. It is preferable to several
unrelated intervention prompts because only the selection pressure changes.
Negative selection is an explicit experimental control, not a claim that an
uncontrolled model naturally chooses its worst answers.

Together, generation and selection define a transition kernel

$$
P_\rho(x,B)
=\Pr\!\left\{\mathcal S_\rho(\mathcal G(x))\in B\right\},
$$

where $x$ is a solution population and $B$ is a set of possible successor
populations. The fixed model determines $\mathcal G$; the scalar $\rho$
changes only $\mathcal S_\rho$.

## Recursive and nonrecursive controls

Selection alone can create an ordered response curve. To isolate inheritance,
run a matched control on a small, stratified problem subset near the most
interesting values of $\rho$:

1. **Recursive:** each child receives the original problem and its selected
   parent attempt.
2. **Restart:** each child receives only the original problem at every round.
3. **Frozen context:** each child always receives its assigned round-zero
   attempt, regardless of subsequent selection.

Use the same model settings, round-zero populations, number of calls, and
selection law. Define the inheritance and moving-context effects as

$$
I_r(\rho)=Q_r^{\mathrm{recursive}}(\rho)
          -Q_r^{\mathrm{restart}}(\rho),
$$

$$
M_r(\rho)=Q_r^{\mathrm{recursive}}(\rho)
          -Q_r^{\mathrm{frozen}}(\rho).
$$

A positive selection curve with $I_r\approx0$ is primarily repeated sampling
and selection, not recursive improvement. A nonzero $I_r$ or $M_r$ identifies
an effect attributable to inherited model output.

The saved candidate pools should also support an offline resampling null:
rerun $\mathcal S_\rho$ many times without additional model calls to quantify
how much uncertainty comes from multinomial selection rather than generation.

## Primary run

Before scaling, run a cost-and-validity pilot on five calibration problems, all
five values of $\rho$, three rounds, and one replicate. The pilot must estimate
real input length, output length, parser failures, retry frequency, latency, and
cost per problem-round. Pilot outputs are not included in the reported spectrum.

If the measured upper confidence bound fits the budget, run the five values of
$\rho$ on up to 40 spectrum problems with:

- population size $m=4$;
- recursive horizon $R=6$ after round zero;
- two independent replicates initially;
- paired initial generations and recorded seeds where possible;
- a third replicate only near a reproducible transition or anomalous region.

The full initial design requires at most

$$
40\times4\times6\times5\times2=9{,}600
$$

revision calls. If the measured cost does not fit, reduce the number of
spectrum problems before reducing the number of $\rho$ values or rounds. Let
$\widehat c_{\mathrm{pr}}$ be the conservative pilot cost per problem for the
complete five-$\rho$, two-replicate trajectory and let $B_{\mathrm{primary}}$
be the remaining primary allocation. The maximum problem count is

$$
n_{\max}=\min\!\left(40,
\left\lfloor\frac{B_{\mathrm{primary}}}{\widehat c_{\mathrm{pr}}}\right\rfloor
\right).
$$

Round-zero calls, calibration calls, retries, and any critic calls are budgeted
separately. Concurrency may reduce elapsed time but does not change the spend.

All $\rho$ conditions within a replicate must begin from the same saved
round-zero populations. This pairing makes differences at later rounds
attributable to selection and the induced recursive state, rather than different
starting samples.

## Response quantities

For problem $p$, replicate $j$, selection value $\rho$, and round $r$, define

$$
q_{pjr}(\rho)=\frac1m\sum_{i=1}^{m}C(x_{pjri}),
$$

and let

$$
Q_r(\rho)=\frac{1}{nJ}\sum_{p=1}^{n}\sum_{j=1}^{J}q_{pjr}(\rho)
$$

be mean population accuracy over $n$ problems and $J$ replicates. The
one-round conditional drift from population state $x$ is

$$
b_\rho(x)=\mathbb E_\rho\!\left[Q(X_{r+1})-Q(X_r)\mid X_r=x\right].
$$

Record:

- mean population accuracy $Q_r$;
- exact-answer accuracy;
- pass@4, indicating whether any survivor is exactly correct;
- best and worst candidate scores;
- score variance within each population;
- distinct normalized final answers;
- lexical or structural diversity of derivations;
- regression rate: correct parents producing incorrect children;
- recovery rate: incorrect parents producing correct children;
- invalid-output and API-error rates.

For final round $R$, calculate

$$
\Delta_r(\rho)=Q_r(\rho)-Q_{r-1}(\rho),
\qquad
g(\rho)=Q_R(\rho)-Q_0(\rho),
$$

$$
A(\rho)=\sum_{r=1}^{R}\left|\Delta_r(\rho)\right|.
$$

The triangle inequality gives the monotone wall

$$
A(\rho)\geq |g(\rho)|.
$$

Height above this wall,

$$
E(\rho)=A(\rho)-|g(\rho)|,
$$

is response erased by reversals.

The signed normalized kernel is

$$
\kappa_r(\rho)=\frac{\Delta_r(\rho)}{|g(\rho)|},
\qquad
\sum_{r=1}^{R}\kappa_r(\rho)=\operatorname{sgn}g(\rho),
$$

only when $|g(\rho)|$ is safely separated from zero. Near-neutral runs should be
reported with the unnormalized pair $(g,A)$ because normalization becomes
unstable.

Define terminal susceptibility and roundwise response susceptibility by

$$
\chi_R(\rho)=\frac{\partial Q_R(\rho)}{\partial\rho},
\qquad
\tau_r(\rho)=\frac{\partial\Delta_r(\rho)}{\partial\rho}.
$$

Estimate these with centered finite differences where the grid permits. Large
$|\chi_R|$ marks a steep selection-response region; a localized $\tau_r$ shows
when that sensitivity enters the recursive trajectory.

For normalized final-answer frequencies $f_a$ within a population, also record

$$
D_r=\frac{|\{a:f_a>0\}|}{m},
\qquad
N_{\mathrm{eff},r}=\frac{1}{\sum_a f_a^2}.
$$

$D_r$ is distinct-answer occupancy and $N_{\mathrm{eff},r}$ is the effective
number of answers. Both expose concentration caused by resampling with
replacement.

Uncertainty bands should bootstrap problems, the independent experimental
units, with replicates nested inside problems. Report paired contrasts between
$\rho$ conditions that share round-zero populations.

## Required figures

1. Mean $Q_r$ trajectories for every $\rho$, with uncertainty bands.
2. Exact accuracy and pass@4 by round.
3. Terminal gain $g$ versus total response $A$, with the monotone wall
   $A=|g|$.
4. Terminal response $Q_R(\rho)$ and $g(\rho)$ across selection pressure.
5. A round-by-$\rho$ heat map of $\Delta_r(\rho)$ or finite-difference
   susceptibility.
6. Accuracy versus solution diversity, to identify diversity loss that precedes
   or accompanies collapse.
7. Domain-stratified trajectories for mechanics, electromagnetism,
   thermodynamics, optics, and other represented fields when sample sizes allow.
8. Recursive, restart, and frozen-context trajectories on the control subset.

## Results worth following

Continue beyond the primary run if at least one of the following survives
replication:

- a sharp transition rather than a smooth response to $\rho$;
- delayed collapse under negative or weak selection;
- improvement followed by regression;
- diversity collapse before accuracy collapse;
- hysteresis or path dependence after changing $\rho$ mid-run;
- a large difference in critical selection pressure across physics domains;
- a near-neutral endpoint with large total internal response;
- a solver that systematically corrupts correct parents or repairs incorrect
  ones at particular rounds.

If all curves simply order monotonically with no meaningful internal structure,
the result is a useful control but does not justify immediate fine-tuning.

## Oracle versus endogenous selection

The primary $\rho$ sweep uses an oracle selector because correctness is computed
from the hidden gold answer. It maps the controlled response spectrum but is not
autonomous self-improvement. Around the most interesting $\rho$ region, compare
the oracle fitness with selectors that never see the gold answer:

1. **Answer consensus:** $s_i$ is the frequency of candidate $i$'s canonical
   final answer in the child population.
2. **Nova Lite self-ranking:** a separate Nova Lite call ranks the four children
   for correctness and returns a normalized score or rank.
3. **Independent Nova Lite critic:** one call critiques each solution before a
   separate solver call revises it.
4. **Nova Pro critic:** Nova Pro ranks or critiques; Nova Lite remains the
   generator.

Apply the same softmax family $p_i(\rho)\propto e^{\rho s_i}$ after normalizing
each endogenous score to $[0,1]$. The autonomous-selection gap is

$$
G_r^{\mathrm{selector}}(\rho)
=Q_r^{\mathrm{oracle}}(\rho)-Q_r^{\mathrm{selector}}(\rho).
$$

This gap measures how much of the oracle response is lost when the system must
identify good reasoning without an answer key.

## Solver--critic extension

Run this only on a small problem subset and around the most interesting $\rho$
region. Keep the selection law unchanged while comparing:

1. Nova Lite revises its own inherited attempt.
2. An independent Nova Lite call critiques the attempt, then Nova Lite revises
   it using the critique.
3. Nova Pro critiques the attempt, then Nova Lite revises it.
4. Nova Lite regenerates without an explicit critique.

The critic must not see the gold answer. This extension tests whether response
geometry changes when error detection is separated from solution generation.
It is a separate family of transition laws, not another point on the original
$\rho$ axis.

## Fine-tuning extension

Amazon Bedrock supports supervised fine-tuning of Nova Lite in `us-east-1` and
on-demand invocation of eligible custom Nova models. Fine-tuning changes model
weights and therefore supports a stronger claim than recursive prompting.

Construct equally sized training sets from spectrum problems only:

- **positive set:** verified-correct, manually audited derivations from
  positive-$\rho$ runs;
- **negative set:** plausible but incorrect derivations from negative-$\rho$
  runs, audited to exclude accidentally correct answers;
- **base model:** unchanged Nova Lite as the neutral reference.

Use at least 200 diverse, audited examples for any supervised fine-tuning job.
The positive records must be accurate; errors in the negative records must be
deliberate and documented. Keep problem IDs disjoint from the 40 locked transfer
problems. Match training-set size, topic mixture, response length, formatting,
epochs, and other hyperparameters between positive and negative custom models.

Evaluate base, positive, and negative models on the locked set with identical
prompts and deterministic answer scoring. Report both target-task change and
broad regression checks. Do not describe better training-set performance as
self-improvement; the relevant result is held-out transfer.

Let $\theta_0$, $\theta_+$, and $\theta_-$ denote the base, positive-tuned, and
negative-tuned weights. Define weight-level held-out gains

$$
g_{\mathrm{FT}}^+
=Q_{\mathrm{holdout}}(\theta_+)-Q_{\mathrm{holdout}}(\theta_0),
\qquad
g_{\mathrm{FT}}^-
=Q_{\mathrm{holdout}}(\theta_-)-Q_{\mathrm{holdout}}(\theta_0).
$$

Evidence for a weight-level spectrum requires $g_{\mathrm{FT}}^-<0$ and
$g_{\mathrm{FT}}^+>0$ with uncertainty intervals that exclude zero. Without
held-out separation, the result is memorization or task adaptation rather than
generalized improvement.

Fine-tuning is permitted only after:

- the prompt-level effect is replicated;
- the generated training records pass manual and programmatic audits;
- Bedrock customization permissions and quotas are confirmed;
- AWS supplies a preflight estimate that fits the remaining experiment budget.

If two matched custom-model jobs do not fit, prioritize one positive model and
retain the base model as control. Do not spend past the cap merely to complete a
symmetric design.

## AWS budget and stopping rules

Total authorized experimental spend: **\$10.00**.

Initial allocation:

| Work | Maximum allocation |
|---|---:|
| Prompt, scorer, and measured-cost pilot | \$0.50 |
| Primary five-point response spectrum | \$3.00 |
| Recursive controls and endogenous selectors | \$1.25 |
| Replication near an interesting region | \$0.75 |
| Fine-tuning reserve | \$4.00 |
| Contingency | \$0.50 |

Operational rules:

- maintain an append-only usage ledger for every Bedrock call;
- record input tokens, output tokens, model ID, retry status, and estimated cost;
- obtain current AWS prices immediately before launching a run;
- calculate a conservative worst-case cost from call count and token caps;
- refuse any stage whose preflight estimate exceeds its remaining allocation;
- stop launching new calls at **\$9.50** total estimated spend;
- do not automatically spend unused fine-tuning reserve on more inference;
- require an explicit decision before proceeding past the primary spectrum.

The earlier synthetic-population run is a useful operational calibration but
physics derivations are substantially longer, so its cost per call must not be
reused without adjusting for the new token caps.

## Reproducibility artifacts

The implementation should eventually create:

```text
physics_response_spectrum/
  README.md
  prompts/
  scripts/
  data/
    source/
    splits/
  runs/
    <run_id>/
      config.json
      manifest.json
      candidates.jsonl
      selections.jsonl
      trajectory.csv
      usage.csv
      metadata.json
  results/
  figures/
```

Every run manifest must include the Git commit, dataset revision and hashes,
problem split, seeds, model IDs, inference parameters, prompt hashes, scorer
version, selection values, timestamps, and current price table used for cost
estimation. Raw responses should be retained locally unless dataset terms or
security requirements prohibit doing so.

## Interpretation boundary

The primary experiment can establish that selection changes recursive
inference-time solution dynamics. It cannot by itself establish autonomous
capability growth or weight-level recursive self-improvement. The fine-tuning
extension can test transfer after weight changes, but even that remains a small,
externally orchestrated experiment on a bounded physics benchmark.

## External references

- [PHYBench dataset](https://huggingface.co/datasets/Eureka-Lab/PHYBench)
- [Amazon Bedrock supported fine-tuning models](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-model-fine-tuning.html)
- [Amazon Nova fine-tuning data preparation](https://docs.aws.amazon.com/bedrock/latest/nova2-userguide/fine-tune-prepare-data-understanding.html)
- [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/)
