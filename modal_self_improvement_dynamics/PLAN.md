# Modal Self-Improvement Dynamics — Execution Plan

## Project decision

Start a **new standalone paper/project** rather than expanding the existing two-page sketch. The sketch remains a compact conceptual bridge; this project will contain theorems, identification methods, experiments, and a reproducible empirical test.

**Working title:** *Beyond Single-Exponential Self-Improvement: Modal Takeoff Kernels for Solver–Verifier Dynamics*

Related local notes:

- [Conceptual self-improvement sketch](../self_improvement_takeoff_kernels/self_improvement_takeoff_kernels.tex)
- [Velocity takeoff formalism](../velocity_takeoff_revised/velocity_takeoff_revised.tex)
- [Inverse takeoff kernels](../inverse_takeoff_kernels/inverse_takeoff_kernels.tex)

## Current status

**Phases 0, 1, and 2 are complete, and Phase 3 has an exploratory first pass.** The scope and claim map are in [SCOPE.md](SCOPE.md). The working manuscript is in [modal_self_improvement_dynamics.tex](modal_self_improvement_dynamics.tex), with a compiled [PDF](modal_self_improvement_dynamics.pdf). The mathematical spine includes the single-mode equivalence, exact rank-one solver–verifier application, joint empirical null, finite-dimensional modal representation, settling theorem, Hankel-rank and matrix-pencil recovery results, endpoint separation, and finite-modal boundary cases. The synthetic construction and known-endpoint power study are in [scripts/synthetic_modes.py](scripts/synthetic_modes.py) and [scripts/modal_power_study.py](scripts/modal_power_study.py). Joint-asymptote rank-one/rank-two fitting and held-out calibration are implemented in [scripts/fit_modal_models.py](scripts/fit_modal_models.py) and [scripts/asymptote_forecast_study.py](scripts/asymptote_forecast_study.py), with results in [figures/fig3_asymptote_forecast.pdf](figures/fig3_asymptote_forecast.pdf). The public-data inventory and qualified Phi-4 vector-figure audit are documented in [DATA_AUDIT.md](DATA_AUDIT.md).

Phase 4 now has a completed three-seed, 21-checkpoint resource-scaled reproduction in [reproduction/](reproduction/). The controlled arithmetic run reproduces qualitative per-token gap closure, but neither a stable rank-two advantage nor a reliable modal advantage over persistence. A five-boundary comparison with power-law and local-linear baselines is documented in [BASELINE_AUDIT.md](BASELINE_AUDIT.md). It finds power-law forecasts most competitive on the published curves and persistence dominant on the local reproduction.

The direct inverse experiment is complete in [SPECTRAL_MODE_AUDIT.md](SPECTRAL_MODE_AUDIT.md). Matrix-pencil recovery is exact in noiseless controls but cannot reliably count heterogeneous modes at 20--21 checkpoints and 1% noise. Conditional one-pole summaries are stable but differ systematically across published solver, verifier, and gap curves, contradicting the exact shared-rate restriction without identifying a unique replacement order. The complete handoff state is in [RESTART_STATE.md](RESTART_STATE.md).

Primary empirical/theoretical target:

- Sun et al., *Theoretical Modeling of Large Language Model Self-Improvement Training Dynamics Through Solver–Verifier Gap* (ICLR 2026), [arXiv:2507.00075](https://arxiv.org/abs/2507.00075).

## Central question

Do observed self-improvement trajectories behave like a single geometric/exponential mode, or do they contain multiple identifiable response modes—fast and slow stages, delay, overshoot, ringing, or long tails—that are hidden by endpoint or aggregate-rate descriptions?

The kernel approach should contribute at three levels:

1. **Verification:** turn the standard single-rate model into falsifiable local identities.
2. **Generalization:** replace one rate by a finite modal response kernel.
3. **Identification:** infer the number and location of modes from trajectories and compare their predictive value.

## Scope and clocks

Keep two time indices distinct throughout:

- **Inner clock:** optimization checkpoints within one training run.
- **Outer clock:** complete generate–verify–retrain self-improvement rounds.

The same mathematics can be applied to either clock, but their kernels have different interpretations and must not be pooled without an explicit coupled model.

## High-level formalism

### Normalized trajectory and response kernel

For an increasing performance statistic $J_t$, define

$$
e_t=\frac{J_\infty-J_t}{J_\infty-J_0},
\qquad
F_t=1-e_t,
\qquad
\kappa_t=F_t-F_{t-1}=e_{t-1}-e_t.
$$

Here $e_t$ is the normalized residual gap, $F_t$ is the cumulative realized improvement, and $\kappa_t$ is the improvement arriving at step $t$. When improvement is monotone, $\kappa_t\ge 0$ and $\sum_t\kappa_t=1$. Signed kernels permit reversals or oscillation.

For a decreasing uncertainty statistic $U_t$, use

$$
e_t=\frac{U_t-U_\infty}{U_0-U_\infty}.
$$

### Result 1: single-mode equivalence

Prove the exact equivalence

$$
e_t=\lambda^t
\quad\Longleftrightarrow\quad
\kappa_t=(1-\lambda)\lambda^{t-1}.
$$

This yields local diagnostics:

$$
\frac{\kappa_{t+1}}{\kappa_t}=\lambda,
\qquad
\kappa_{t+1}\kappa_{t-1}-\kappa_t^2=0.
$$

These identities turn a global exponential fit into a testable claim about every sufficiently well-measured segment of the trajectory.

### Result 2: modal state-space representation

Let the hidden solver–verifier state satisfy

$$
x_{t+1}-x_\infty=A(x_t-x_\infty),
\qquad
J_\infty-J_t=b^\top(x_t-x_\infty).
$$

Under diagonalizability, the observable residual has the form

$$
J_\infty-J_t=\sum_{r=1}^{R}c_r\theta_r^t,
$$

and therefore

$$
\kappa_t=
\frac{\sum_{r=1}^{R}c_r(1-\theta_r)\theta_r^{t-1}}
     {\sum_{r=1}^{R}c_r}.
$$

Interpret $\theta_r$ as response modes, not merely fit parameters: positive real modes give monotone stages, negative or complex modes allow alternating or ringing responses, and a slow dominant mode controls the tail.

### Result 3: shape and settling behavior

For modal radius

$$
\lambda_\star=\max_r|\theta_r|,
$$

establish, under ordinary non-cancellation conditions,

$$
1-F_t=O(\lambda_\star^t),
\qquad
T_\varepsilon=O\!\left(\frac{\log(1/\varepsilon)}{|\log\lambda_\star|}\right).
$$

Also state the qualitative classification:

- positive real modes: monotone, possibly multi-stage takeoff;
- negative modes: alternating corrections;
- complex-conjugate modes: damped ringing;
- algebraic tails: evidence against an exact finite-modal model.

### Result 4: inverse identification by Hankel rank

Form the Hankel matrix

$$
H_{ij}=\kappa_{i+j}.
$$

For distinct active modes and nonzero coefficients, prove that its exact rank equals the number of active modes. Thus:

- rank one corresponds to a single exponential;
- rank two is the minimal two-stage alternative;
- the noisy singular spectrum estimates effective modal dimension;
- Prony or matrix-pencil methods recover candidate modes.

### Endpoint-separation corollary

Construct trajectories with identical $J_0$, $J_\infty$, and total gain but different $\lambda_\star$, kernel shape, and $T_\varepsilon$. This is the simplest formal demonstration that endpoint comparisons do not determine takeoff dynamics.

## Claim hierarchy

Keep the paper's claims tiered so that partial empirical results still yield a coherent contribution.

1. **Core theorem claim:** the standard single-rate model is exactly the rank-one kernel case.
2. **Diagnostic claim:** local ratio and Hankel tests can reject that case.
3. **Generalization claim:** coupled solver–verifier dynamics naturally generate multiple modes.
4. **Empirical claim:** at least one published or reproduced trajectory is better predicted by a multi-mode kernel.
5. **Interpretive claim:** recovered modes suggest distinct fast and slow self-improvement mechanisms; present this as an interpretation unless separately identified causally.

## Experimental program

### Experiment A — synthetic validation

Generate trajectories from:

- single modes $\lambda\in\{0.2,0.5,0.8,0.95\}$;
- two-mode mixtures with varied weights and separation;
- negative modes;
- complex-conjugate modes;
- algebraic-tail controls;
- multiple noise levels and checkpoint counts.

Compare:

- nonlinear least-squares single exponential;
- finite multi-mode matrix-pencil fit;
- power-law tail;
- nonparametric kernel estimate.

Report:

- mode and modal-rank recovery;
- held-out forecast error;
- errors in $J_\infty$, $\lambda_\star$, and $T_\varepsilon$;
- shape-classification accuracy;
- minimum checkpoint count needed at each noise level.

**Gate A:** proceed to strong empirical modal claims only if the synthetic study shows reliable discrimination in the sample regime available from real curves.

### Experiment B — published-trajectory reanalysis

Preferred acquisition order:

1. request numerical trajectories and seeds from the authors;
2. extract published tables or supplements;
3. digitize figures for exploratory analysis only.

Analyze solver uncertainty, verifier uncertainty, solver–verifier gap, and task performance separately. Use leave-last-$k$-out forecasting rather than relying on in-sample $R^2$.

Planned result table:

| Model/task | Clock | Final gain | Effective rank | $\lambda_\star$ | $T_{0.05}$ | Signed response? | Best forecast model |
|---|---|---:|---:|---:|---:|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Bootstrap over seeds where available; otherwise make uncertainty from digitization and fit instability explicit.

**Gate B:** if trajectories have too few checkpoints, an unstable inferred asymptote, or no recoverable numerical data, present this only as an illustrative reanalysis and move the main empirical claim to Experiment C.

### Experiment C — controlled reproduction

Minimal first reproduction:

- Phi-3-mini or Llama-3.2-3B;
- GSM8K first, MBPP second if resources permit;
- LoRA fine-tuning;
- True/False and quality-evaluation verifier variants;
- at least three random seeds;
- 15–30 checkpoints per trajectory.

Track:

- task accuracy and pass@$k$;
- solver uncertainty;
- verifier accuracy;
- acceptance rate;
- solver–verifier gap;
- data quantity and quality per outer round.

Fit the inner-clock and outer-clock kernels independently before attempting a coupled model.

**Gate C:** a full reproduction is warranted only if published data are inadequate or if the synthetic power analysis indicates that the planned checkpoint density can identify more than one mode.

## Paper sketch

1. **Introduction** — standard single-rate descriptions omit response shape.
2. **Related work** — self-improvement, solver–verifier dynamics, scaling/training laws, system identification, and takeoff kernels.
3. **Two clocks of self-improvement** — distinguish optimization from recursive rounds.
4. **Kernel formalism** — define residuals, cumulative response, and signed kernels.
5. **Finite-modal theory** — state-space derivation, single-mode equivalence, settling behavior, and endpoint separation.
6. **Inverse identification** — Hankel rank, singular spectrum, Prony/matrix-pencil recovery, noise limitations.
7. **Experiments** — synthetic validation, published reanalysis, and controlled reproduction as available.
8. **Discussion** — what modes can and cannot say about recursive takeoff.
9. **Limitations** — finite horizons, asymptote estimation, identifiability, bounded metrics, and causal interpretation.

## Execution phases

### Phase 0 — freeze scope and claim map

- [x] Preserve the existing two-page sketch as a separate conceptual note.
- [x] Fix the primary target formalism and observable trajectories from Sun et al.
- [x] Write a one-paragraph claim for each tier in the claim hierarchy.
- [x] Decide which results are theorems, propositions, diagnostics, or empirical hypotheses.
- [x] Record the exact meaning of the inner and outer clocks.

**Deliverable:** one-page scope memo and stable notation table.

### Phase 1 — fill in the mathematics

- [x] State and prove the single-mode equivalence.
- [x] Derive the modal kernel from the linear state-space model.
- [x] State precise assumptions for the settling-time bound.
- [x] Prove the finite-modal Hankel-rank result.
- [x] Construct the endpoint-separation example.
- [x] Add counterexamples: cancellation, repeated eigenvalues/Jordan blocks, and algebraic tails.
- [x] Mark which extensions are conjectural or deferred.

**Deliverable:** theorem-ready LaTeX sections with complete proofs or clearly labeled proof sketches.

### Phase 2 — build and validate the estimators

- [x] Implement synthetic trajectory generation.
- [x] Implement single-exponential and multi-mode fits.
- [x] Implement Hankel singular-spectrum diagnostics.
- [x] Implement matrix-pencil recovery with calibrated order selection and pole-stability checks.
- [x] Add power-law and cross-validated local-linear baselines.
- [x] Run checkpoint-count and noise-level sweeps.
- [x] Produce Figure 1: same endpoints, different kernels.
- [x] Produce Figure 2: rank and mode recovery phase diagram.
- [x] Profile over an unknown asymptote and compare held-out rank-one/rank-two forecasts.
- [x] Produce Figure 3: detection power versus endpoint-estimation reliability.

**Deliverable:** reproducible synthetic benchmark and an evidence-based minimum-data requirement.

### Phase 3 — acquire and reanalyze published data

- [x] Inventory figures, tables, appendices, and stated hyperparameters.
- [x] Search for an author repository or data archive.
- [x] Decide not to request author data; qualify the public-artifact audit instead.
- [x] Draft a precise author data request.
- [x] Digitize selected figures only if raw values remain unavailable.
- [x] Normalize the selected vector trajectories by an affine response scale.
- [x] Compare rank-one, multi-mode, power-law, persistence, and local-linear forecasts across five rolling splits.
- [x] Bootstrap rank-one versus rank-two held-out error on the selected figure.
- [x] Write a limitations note for the selected dataset.

**Deliverable:** a clean trajectory dataset, analysis notebook/script, and one empirical results table.

### Phase 4 — decide and run controlled reproduction

- [x] Use Phase 2 power analysis to set checkpoint density.
- [x] Lock model, task, verifier variants, seeds, and compute budget.
- [x] Reproduce the baseline single-round dynamics first.
- [ ] Add outer recursive rounds only after the baseline is stable.
- [x] Log every kernel observable at every checkpoint.
- [x] Fit models on early checkpoints and evaluate held-out tail forecasts.
- [x] Test whether modes replicate across three fresh seeds; verifier variants remain open.

**Deliverable:** controlled evidence for or against a multi-modal self-improvement response.

### Phase 5 — assemble the manuscript

- [x] Draft the mathematical and synthetic-results sections.
- [ ] Draft related work from primary sources.
- [x] Insert synthetic results before making claims about real trajectories.
- [x] Add the strongest credible empirical section available.
- [x] Separate mathematical identification from mechanistic interpretation.
- [ ] Write abstract, introduction, discussion, and limitations last.

**Deliverable:** complete compilable manuscript.

### Phase 6 — validation and release

- [ ] Run all figures and tables from a clean environment.
- [ ] Check notation and theorem dependencies.
- [x] Verify every current empirical number against generated artifacts.
- [x] Add a reproducibility README and fixed random seeds.
- [x] Compile and visually inspect the current PDF.
- [x] Perform a claim-versus-evidence audit of the current results.

**Deliverable:** paper PDF, source, data provenance, and reproducible scripts.

## Proposed repository layout

```text
modal_self_improvement_dynamics/
  PLAN.md
  README.md                              # later
  modal_self_improvement_dynamics.tex    # later
  references.bib                         # later
  data/
    raw/
    processed/
  scripts/
    synthetic_modes.py
    fit_modal_models.py
    digitize_published_curves.py
    make_figures.py
  figures/
  results/
```

Create later files only as their phase begins, so the repository records actual work rather than placeholders.

## Main risks and mitigations

| Risk | Mitigation |
|---|---|
| $J_\infty$ is jointly estimated and unstable | Use longer runs, profile sensitivity over plausible asymptotes, and report interval-valued conclusions. |
| Too few checkpoints to identify modes | Require roughly 15–20 or more observations unless synthetic power analysis supports fewer. |
| Digitization error dominates the signal | Treat digitized results as exploratory and propagate extraction uncertainty. |
| Inner and outer dynamics are conflated | Maintain separate time indices, datasets, fits, and interpretations. |
| Accuracy is bounded and noisy | Analyze uncertainty and verifier quantities alongside task accuracy. |
| Multi-mode models overfit | Use held-out forecasting, bootstrap stability, information criteria, and singular-spectrum diagnostics. |
| Recovered modes are given causal labels too quickly | Call them response modes unless an intervention identifies a mechanism. |

## Immediate work queue

1. Decide whether to position the empirical contribution as quantified nonidentifiability or wait for longer, lower-noise raw trajectories.
2. If better data become available, prioritize paired seed-level curves, an independently constrained endpoint, and 40--60 checkpoints.
3. On improved data, compare a sparse finite set of inverse-Laplace atoms with a regularized diffuse spectrum.
4. Draft related work from primary sources and run the notation/theorem dependency audit.
5. Add outer recursive rounds only after the inner-clock baseline is stable.

## Minimum viable paper: definition of done

The first complete version requires:

- rigorous statements and proofs for the single-mode, state-space, settling, and Hankel-rank results;
- synthetic validation showing when modal recovery works and fails;
- at least one credible empirical trajectory family;
- a held-out comparison of rank-one and multi-mode predictions;
- explicit separation of inner and outer clocks;
- a compiled paper and reproducible analysis scripts.

The project does **not** require proving that recursive self-improvement universally has multiple modes. A strong negative result—showing that available data cannot distinguish rank one from higher rank, together with quantified sample requirements—would also be a valid and useful result.
