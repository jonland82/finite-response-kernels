# Local-versus-global dynamics audit

## Question

Is the shared exponential in the solver--verifier model a global law over the full training run, a persistent multimode response, or a local law that changes across training phases?

This is a first-pass audit of the trajectories already available in the project. It is not a new large-model training run.

## Models

Solver, verifier, and gap are fit jointly on rolling training prefixes and evaluated on untouched held-out tails. Errors are normalized by each series' training range and then averaged across the three observables.

The structural comparison is:

1. **Shared rank one:** one global mode for all three observables.
2. **Shared rank two:** two persistent global modes with observable-specific amplitudes.
3. **Shared two phase:** a continuous exponential response with one rate before a shared change point and another rate afterward.

The two-phase response is

$$
Y_n=Y_\infty+c
\lambda_1^{\min(n,\tau)}
\lambda_2^{\max(n-\tau,0)}.
$$

It has no larger a parameter budget than the shared rank-two model. Separate rank-one, power-law, local-linear, and persistence forecasts are also included so that a structural winner need not be the overall winner.

An endpoint-free local-rate diagnostic fits

$$
\Delta Y_{n+1}\approx\lambda_{\mathrm{local}}\Delta Y_n
$$

over rolling seven-checkpoint windows. A stable exponential decay should keep $0<\lambda_{\mathrm{local}}<1$ with limited drift.

## Data and calibration

- Four published Phi-4 task/verifier panels, each containing solver, verifier, and gap curves at 20 figure-derived checkpoints.
- Three controlled arithmetic reproduction seeds, each containing paired per-token solver and verifier uncertainty and their derived gap at 21 checkpoints.
- One hundred matched synthetic trajectory groups for each truth: shared rank one, shared rank two, and shared two phase.
- Five rolling training-prefix sizes per trajectory.
- Synthetic observation noise of 1% of response scale.

The gap is not treated as independent information. In the synthetic controls and reproduction it is derived from solver minus verifier; in the published audit it remains the plotted aggregate gap and the missing seed-level covariance is a limitation.

## Results

### Published curves

Across 20 rolling panel--split cases:

- shared rank two wins all 20 comparisons restricted to the three structural models;
- shared two phase beats shared rank one in 15 cases but never beats shared rank two;
- across the full baseline set, separate power-law fits win 12 cases, shared rank two wins six, separate rank one wins one, and local linear wins one;
- the two-phase model never wins overall.

The endpoint-free local rates are comparatively smooth. All 56 published rolling windows have rates in $(0,1)$. Panel medians range from $0.751$ to $0.843$, and within-panel interquartile ranges range from $0.040$ to $0.089$. There is no consistent early-to-late direction across the four panels.

The fitted two-phase change point moves later as the training prefix grows and never wins structurally. That boundary tracking is more consistent with a flexible approximation than with a stable physical transition.

**Interpretation:** the figure-derived published curves do not support the current two-phase alternative. They remain better described predictively by a global multiscale or nonmodal smooth relaxation than by one shared change point.

### Controlled reproduction

Across 15 rolling seed--split cases:

- shared two phase wins 11 of 15 structural comparisons;
- shared rank two wins the other four;
- across the full baseline set, shared two phase wins nine, persistence four, shared rank two one, and local linear one.

The fitted change points are perfectly stable across the five training prefixes within each seed:

| Seed | Change epoch | Early mode | Late mode | Structural wins |
|---:|---:|---:|---:|---:|
| 20260718 | 1.5 | 0.995 | 0.05 | 1/5 |
| 20260719 | 1.0 | 0.96 | 0.60 | 5/5 |
| 20260720 | 3.5 | 0.995 | 0.05 | 5/5 |

The extreme $0.995\rightarrow0.05$ fits in two seeds describe a delay followed by an abrupt jump into a plateau, not a smooth startup exponential. The endpoint-free rolling rates reinforce that reading: only $27\%$, $13\%$, and $7\%$ of the windows in the three seeds have a stable decay rate in $(0,1)$, and the median local rates are nonpositive.

**Interpretation:** the reproduction contains reproducible within-seed transition locations, but not a shared local exponential law across seeds. Its time-local structure is primarily discrete change plus plateau behavior.

### Dense-start follow-up

The seed-20260719 reproduction was rerun with checkpoints after every optimizer
update through epoch 4, giving quarter-epoch spacing in the region where the
first audit found change points. The original half-epoch checkpoints were then
continued through epoch 10. The 21 overlapping checkpoints match the original
trajectory exactly across all logged metrics (maximum absolute difference 0),
so the denser evaluation schedule did not perturb training; the runner restores
Python, NumPy, CPU, and CUDA RNG states around evaluation.

The additional checkpoints do not reveal a smooth local exponential. From epoch
1.25 to 1.5, solver total uncertainty falls from 10.416 to 2.394 while mean
solver response length falls from 14.375 to 6.875 tokens. Solver accuracy falls
from 0.50 to 0.375 and verifier accuracy from 0.75 to 0.625. Solver per-token
uncertainty decreases only from 0.379 to 0.333, while verifier per-token
uncertainty increases from 0.327 to 0.410. A second verifier transition from
epoch 3.0 to 3.25 coincides with a length increase from 14.125 to 18.25 tokens
and a verifier-accuracy decrease. These are one-update behavioral and
measurement transitions, not clean capability takeoff evidence.

**Interpretation:** the dense run supports time-local measurement as the right
experimental lens, but its first result is a confound warning. Total uncertainty
cannot be treated as capability without tracking response length and accuracy,
and a visible jump should not be promoted to a modal or causal source claim.

### Synthetic identifiability

Using raw held-out error among the three structural models, the correct model is selected at the following rates:

| True model | Correct selection |
|---|---:|
| Shared rank one | 57.2% |
| Shared rank two | 90.2% |
| Shared two phase | 51.0% |

Two-phase accuracy remains between $49\%$ and $54\%$ at every tested rolling split. Thus the present horizon can identify the particular rank-two controls well, but it only weakly separates a phase change from rank one or rank two.

## Bottom line

The first experiment does not establish that the published rate differences come from a local takeoff phase. It does sharpen the next question:

- the published aggregate curves look smooth and globally multiscale or nonmodal;
- the small reproduction looks piecewise and jump-dominated;
- 20--21 checkpoints are insufficient to reliably distinguish a true two-phase response from the global alternatives.

The time-local direction remains promising as an experimental design, not yet as a positive empirical claim. The dense-start run shows that quarter-epoch sampling can localize the apparent transitions, but one seed and an eight-prompt panel cannot establish their generality. Future work should retain paired seed-level solver and verifier measurements, independent capability metrics, and response-length metadata before attempting source separation.

## Artifacts

- Analysis: [scripts/local_dynamics_audit.py](scripts/local_dynamics_audit.py)
- Dense-start analysis: [scripts/dense_start_audit.py](scripts/dense_start_audit.py)
- Full rolling fits: [results/local_dynamics_group_details.csv](results/local_dynamics_group_details.csv)
- Model summary: [results/local_dynamics_summary.csv](results/local_dynamics_summary.csv)
- Local-rate windows: [results/local_rate_windows.csv](results/local_rate_windows.csv)
- Local-rate summary: [results/local_rate_summary.csv](results/local_rate_summary.csv)
- Metadata and synthetic confusion: [results/local_dynamics_metadata.json](results/local_dynamics_metadata.json)
- Figure: [figures/fig6_local_dynamics_audit.pdf](figures/fig6_local_dynamics_audit.pdf)
- Dense-start metadata: [results/dense_start_metadata.json](results/dense_start_metadata.json)
- Dense-start step changes: [results/dense_start_step_changes.csv](results/dense_start_step_changes.csv)
- Dense-start figure: [figures/fig7_dense_start_audit.pdf](figures/fig7_dense_start_audit.pdf)

## Reproduce

From `modal_self_improvement_dynamics`:

```powershell
.\reproduction\.venv\Scripts\python.exe .\scripts\local_dynamics_audit.py `
  --project-root . --synthetic-replicates 100 `
  --synthetic-noise 0.01 --seed 20260723

.\reproduction\.venv\Scripts\python.exe .\scripts\dense_start_audit.py `
  --project-root .
```

The script is deterministic at the declared seed.
