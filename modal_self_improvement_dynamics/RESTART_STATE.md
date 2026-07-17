# Research restart state

Last updated: 2026-07-16

## Plain-language state

The project has a complete mathematical core, synthetic validation, a qualified audit of published curves, and a controlled local reproduction. The strongest empirical conclusion is not "there are exactly two modes." It is:

> The standard shared one-exponential description is too rigid. The published curves contain smooth, observably different relaxation scales, but current 20--21 checkpoint data cannot determine whether the richer structure is two discrete modes, several modes, a power-law continuum, or changing dynamics. The local reproduction is dominated by plateaus and discrete jumps rather than stable modal relaxation.

Do not foreground this project in the repository-level `README.md` or `index.html` yet. It has been committed as a self-contained research folder so work can resume without making it a featured project.

## What is complete

### Mathematical core

- The standard solver--verifier exponential law is exactly a rank-one takeoff kernel.
- Ratio constancy and vanishing second-order Hankel minors give local falsification tests.
- A finite-dimensional stable hidden state produces a finite modal kernel.
- Settling behavior is controlled by the largest active modal radius.
- Exact noiseless Hankel rank equals the number of distinct active modes.
- A matrix pencil recovers distinct active modes in the noiseless nondegenerate case.
- Endpoint-equal rank-one and rank-two trajectories can have very different settling times.
- Cancellation, Jordan blocks, complex modes, and algebraic tails are explicitly separated.

### Synthetic experiments

- `scripts/synthetic_modes.py`: same endpoints, different response kernels.
- `scripts/modal_power_study.py`: known-endpoint Hankel power study.
- `scripts/asymptote_forecast_study.py`: unknown-endpoint rank-one/rank-two forecast calibration.
- `scripts/forecast_baseline_audit.py`: persistence, local-linear, power-law, rank-one, and rank-two rolling forecasts.
- `scripts/spectral_mode_audit.py`: calibrated Hankel order selection and matrix-pencil pole recovery.

Important calibration facts:

- Held-out modal-order detection can show rank-one forecast failure before endpoints or individual modes are reliable.
- With 21 checkpoints and 1% noise, the earlier calibrated forecast test has rank-two power of 0.43--0.625 across splits.
- The endpoint-free direct spectral selector is much weaker: it selects the correct rank-two order only 1.15% of the time in the heterogeneous calibration.
- Supplying the true rank to the matrix pencil at 21 checkpoints recovers both rank-two poles within 0.05 only 12.5% of the time; ranks three and four are not recovered.
- Noiseless matrix-pencil recovery is exact through rank four, so this is a finite-data/noise limitation rather than a code failure.

### Published-curve audit

Source: 12 vector-extracted Phi-4 curves, 20 checkpoints each.

- Original 15/5 split: rank two forecasts better than rank one on 10/12 curves; seven fitted-null rejections survive Holm correction.
- Five rolling splits: power law wins 27/60 cases, rank two 16, rank one 13, local linear four, persistence zero.
- Power law beats both modal forecasts in 30/60 cases.
- No curve has the same winning forecast model at all five splits.
- The direct spectral order selector accepts rank one in 60/60 cases, but its synthetic power against rank two is below its 5% false-positive rate; this is non-informative acceptance.
- Conditional dominant poles are stable and ordered in all four panels: gap below solver below verifier. Median poles are 0.740, 0.790, and 0.889, respectively, with pairwise non-overlapping 10th--90th percentile perturbation intervals in every panel.
- Best claim: the exact shared-single-mode restriction fails descriptively; the number of replacement modes is unidentified.

### Local reproduction

Three fixed seeds, 21 checkpoints, six observables per seed, 0.5B Qwen, controlled arithmetic task.

- Per-token solver--verifier gap narrows in all three seeds.
- Solver accuracy improves in only one seed.
- Rank two beats rank one in 9/18 original comparisons, but the better modal model beats persistence in only 5/18.
- Across five rolling splits, persistence wins 52/90 cases and is the consensus winner for 15/18 trajectories.
- The spectral audit labels most cases high-rank, but no full trajectory has perturbation-stable poles. This reflects plateaus and jumps, not evidence for many physical modes.

## Current claim map

| Claim | State |
|---|---|
| Standard constant-rate law implies one shared mode | Proven |
| Exact finite-modal Hankel rank equals active mode count | Proven |
| Noiseless matrix pencil recovers active modes | Proven and numerically validated |
| Published curves are adequately described by one shared exponential | Evidence against |
| Published curves require more than one effective timescale | Plausible and forecast-supported, but not directly identified |
| Exactly two discrete modes are present | Unresolved |
| A broad spectrum/power-law tail is competitive | Supported predictively |
| Local reproduction contains stable response modes | Not supported |
| Kernel methods improve the result type | Supported: they distinguish model failure, order recovery, and mechanism identification |

## Main documents

- `modal_self_improvement_dynamics.tex` and compiled PDF: working paper.
- `SCOPE.md`: scope and claim hierarchy.
- `PLAN.md`: execution plan and checkboxes.
- `DATA_AUDIT.md`: public-data provenance and first modal audit.
- `BASELINE_AUDIT.md`: rolling forecast baselines.
- `SPECTRAL_MODE_AUDIT.md`: direct inverse spectral experiment.
- `reproduction/REPRODUCTION_REPORT.md`: local reproduction.

## Reproduction commands

Run from `modal_self_improvement_dynamics` with the existing environment:

```powershell
.\reproduction\.venv\Scripts\python.exe .\scripts\forecast_baseline_audit.py `
  --project-root . --synthetic-replicates 200 `
  --synthetic-noise 0.01 --seed 20260721

.\reproduction\.venv\Scripts\python.exe .\scripts\spectral_mode_audit.py `
  --project-root . --replicates 400 `
  --bootstrap-replicates 200 --seed 20260722

pdflatex -interaction=nonstopmode -halt-on-error modal_self_improvement_dynamics.tex
pdflatex -interaction=nonstopmode -halt-on-error modal_self_improvement_dynamics.tex
```

## Best next steps

1. Treat mode count as nonidentified unless longer or lower-noise raw trajectories become available.
2. If improving the empirical result, prioritize raw paired seed-level curves, an independently constrained endpoint, and roughly 40--60 checkpoints over adding a more elaborate inverse algorithm.
3. With better data, fit a regularized inverse-Laplace measure

   $$
   e(t)=\int_0^\infty e^{-\lambda t}\,d\mu(\lambda)
   $$

   and compare a few stable atoms with a diffuse spectrum.
4. Finish related work and a notation/theorem dependency audit only after deciding whether the paper will be positioned as a positive multimode result or a quantified nonidentifiability result.
5. Add outer recursive rounds only after the inner-clock empirical baseline is stable.

## Repository checkpoint

- Intended branch: `master`, tracking `origin/master`.
- Intended commit message: `Add modal self-improvement dynamics study`.
- Publication scope: the `modal_self_improvement_dynamics/` folder only.
- Deliberately excluded: `self_improvement_takeoff_kernels/`, root `README.md`, and root `index.html`.
- On restart, verify with `git status -sb` and `git log -1 --oneline` before editing.
