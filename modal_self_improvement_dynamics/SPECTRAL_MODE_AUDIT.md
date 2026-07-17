# Spectral mode-count audit

## Question

Can an inverse spectral method determine whether the self-improvement curves contain one, two, or more response modes?

The short answer is: **not at the present resolution**. The method exactly recovers modes in noiseless controls and gives stable dominant-rate summaries for the published curves, but its calibrated power to count multiple modes with 20--21 noisy checkpoints is extremely low. The useful positive result is a stable separation between solver, verifier, and gap rates, which contradicts the exact shared-single-mode law without identifying a unique larger mode count.

## Formalization

Write a finite-modal response as

$$
Y_n=Y_\infty+\sum_{j=1}^{R}a_j\theta_j^n,
\qquad 0<\theta_j<1.
$$

First differences remove the unknown endpoint:

$$
\Delta Y_n
=Y_{n+1}-Y_n
=\sum_{j=1}^{R}a_j(\theta_j-1)\theta_j^n.
$$

In exact arithmetic, the Hankel matrix of $\Delta Y_n$ has rank $R$. Equivalently, the sequence has an annihilating polynomial

$$
q(z)=\prod_{j=1}^{R}(z-\theta_j),
\qquad
\sum_{k=0}^{R}q_k\Delta Y_{n+k}=0.
$$

This is the continuous/noisy analogue of the syndrome construction in the algebraic-coherence work: the rank-$R$ sequence is the candidate lawful subspace, and the residual outside it is measured by the tail Hankel energy

$$
T_R
=
\frac{\sum_{j>R}\sigma_j(H)^2}
{\sum_j\sigma_j(H)^2}.
$$

For each checkpoint count and noise assumption, $T_R$ is calibrated at the 95th percentile of a matched rank-$R$ synthetic null. We select the smallest $R\in\{1,2,3\}$ not rejected; if all three are rejected, the descriptive label is $4+$. Candidate poles are then recovered with a truncated matrix pencil and perturbed with 200 fixed-seed bootstrap noise draws.

The important cost is visible in the differenced equation: a slow mode's amplitude is multiplied by $1-\theta_j$, while independent observation noise is amplified by differencing. The endpoint-free transformation is exact but statistically hostile to slow-mode recovery.

## Data and calibration

- 400 synthetic trajectories per true order $R=1,2,3,4$;
- 20--21 checkpoints and response-scale noise levels $0.5\%,1\%,2\%,3\%$;
- five rolling prefix lengths: 10, 12, 14, 15, and 16 checkpoints;
- 12 digitized published curves;
- 18 local reproduction trajectories;
- 200 perturbation-bootstrap draws for every full real trajectory;
- fixed seed `20260722`.

The matrix-pencil code also passes a noiseless control: all 20 curves at every order from one through four are recovered, with maximum pole errors at numerical precision.

## Synthetic result

At 1% response-scale noise, the rolling calibrated selector behaves as follows:

| True order | Exact order selected | Rank one rejected |
|---:|---:|---:|
| 1 | 95.00% | 5.00% |
| 2 | 1.15% | 3.95% |
| 3 | 0.55% | 2.20% |
| 4 | 0.00% | 0.05% |

This is not evidence that the higher-order curves are effectively rank one. It shows that the endpoint-free Hankel statistic has essentially no power for the heterogeneous slow-mode mixture at this horizon. Indeed, it rejects rank one less often under the higher-order alternatives than under the rank-one null because differencing suppresses the slow components.

Even when the true order is supplied to the matrix pencil, full 21-checkpoint pole recovery is difficult:

| True order | All poles real and stable | All poles within 0.05 |
|---:|---:|---:|
| 1 | 100.00% | 100.00% |
| 2 | 40.75% | 12.50% |
| 3 | 0.50% | 0.00% |
| 4 | 0.00% | 0.00% |

Thus model-failure detection is substantially easier than pole recovery. This agrees with the earlier held-out forecast calibration, which detected rank-two misspecification more often than the direct spectral inverse.

## Published curves

The calibrated order selector returns rank one in all 60 rolling curve--split cases. Because it rejects rank one on only 3.95% of the rank-two controls, this acceptance is **non-informative** and must not be read as support for one true mode.

Conditional one-pole fits are nevertheless perturbation-stable. Their median poles, pooled by observable role, are:

| Observable | Median dominant pole |
|---|---:|
| Gap | 0.740 |
| Solver | 0.790 |
| Verifier | 0.889 |

In all four task/metric panels,

$$
\widehat\theta_{\mathrm{gap}}
<\widehat\theta_{\mathrm{solver}}
<\widehat\theta_{\mathrm{verifier}},
$$

and all three pairwise 10th--90th percentile perturbation intervals are non-overlapping. This is a direct descriptive failure of the exact standard restriction that solver, verifier, and gap share one exponential rate. It does not tell us whether the correct replacement is two discrete modes, several modes, a power-law continuum, or time-varying dynamics.

The rolling forecast audit supplies the complementary comparison: power law wins 27 of 60 cases, rank two 16, and rank one 13; power law beats both modal models in 30 cases. Taken together, the evidence favors richer or multiscale structure over one shared rate, while leaving the number of discrete modes unidentified.

## Local reproduction

At the primary 1% calibration, 74 of 90 rolling reproduction cases are labeled $4+$, 11 rank three, three rank two, and two rank one. On the full trajectories, 16 of 18 are labeled $4+$ and two rank three. However, **none** yields perturbation-stable poles.

This is not evidence for four physical modes. The trajectories contain plateaus and discrete jumps, persistence wins 52 of 90 rolling forecasts, and the finite-exponential inverse tries to represent those irregularities as high rank. The correct reading is failure of a stable finite-modal law at this scale.

## Conclusion

The experiment separates three statements:

1. Matrix-pencil recovery is mathematically correct and exact in noiseless controls.
2. The published observables have stable but different dominant relaxation rates, contradicting one exact shared mode.
3. The available curves do not identify whether the underlying response has two, three, many, or continuously distributed timescales.

The most accurate state is therefore:

> One shared exponential mode is forecast-incomplete and its cross-observable equality is contradicted. More than one effective timescale remains plausible, but exactly two discrete modes are neither verified nor ruled out. Direct spectral counting is nonidentifiable with the present endpoint uncertainty, checkpoint count, and noise.

## Reproduce

From `modal_self_improvement_dynamics`:

```powershell
.\reproduction\.venv\Scripts\python.exe .\scripts\spectral_mode_audit.py `
  --project-root . --replicates 400 `
  --bootstrap-replicates 200 --seed 20260722
```

Generated artifacts:

- `results/spectral_calibration_summary.csv`;
- `results/spectral_selection_confusion.csv`;
- `results/matrix_pencil_recovery.csv`;
- `results/spectral_mode_details.csv`;
- `results/spectral_mode_series_summary.csv`;
- `results/spectral_mode_metadata.json`;
- `figures/fig5_spectral_mode_audit.png` and `.pdf`.
