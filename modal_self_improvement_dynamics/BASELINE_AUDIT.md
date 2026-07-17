# Rolling forecast baseline audit

## Question

Do rank-one or rank-two takeoff kernels predict held-out self-improvement trajectories better than plausible non-modal baselines, and is the answer stable under the train/test split?

## Models

Every model is fit using only a training prefix and forecasts the untouched tail:

1. persistence, $Y_{n+h}=Y_n$;
2. a cross-validated local-linear smoother;
3. power-law relaxation,

   $$
   Y(t)=Y_\infty+c(t+\tau)^{-p};
   $$

4. a signed rank-one residual,

   $$
   Y_n=Y_\infty+c\lambda^n;
   $$

5. a signed rank-two residual,

   $$
   Y_n=Y_\infty+c_1\lambda_1^n+c_2\lambda_2^n.
   $$

The audit uses five prefix fractions, $0.50,0.60,0.70,0.75,0.80$. Model selection uses raw held-out mean-squared error within each trajectory and split. Normalized errors use only the training range.

## Data

- two exact synthetic endpoint-separation trajectories;
- 400 fixed-seed synthetic trajectories at 1% response-scale noise, half rank one and half rank two;
- 12 digitized published Phi-4 trajectories;
- 18 local reproduction trajectories: six observables for each of three seeds.

This gives 2,160 forecast cases and 10,800 model fits.

## Results

### Published curves

Across 60 curve--split cases, the winning forecasts are:

| Model | Wins |
|---|---:|
| Power law | 27 |
| Rank two | 16 |
| Rank one | 13 |
| Local linear | 4 |
| Persistence | 0 |

At least one modal model beats persistence in all 60 cases, confirming that the published curves contain a smooth trend. But the power law beats both modal models in 30 of 60 cases. No published curve has the same winning model at all five split points; the mean within-curve winner consensus is only 0.53.

Thus the earlier rank-two advantage is not specific evidence for two modes. It is evidence against a rigid rank-one exponential, but a non-modal long tail often predicts at least as well.

### Local reproduction

Across 90 trajectory--split cases, the winning forecasts are:

| Model | Wins |
|---|---:|
| Persistence | 52 |
| Power law | 13 |
| Local linear | 10 |
| Rank one | 8 |
| Rank two | 7 |

The better modal forecast beats persistence in only 27 of 90 cases. Persistence is the consensus winner for 15 of the 18 trajectories. Only two trajectories have one winner at all five split points.

This strengthens the original pilot conclusion: the local dynamics are dominated by plateaus and discrete response changes rather than a stable relaxation law.

### Known-rank synthetic calibration

Naively selecting the lower-MSE modal order is correct in 65.1% of noisy synthetic cases. It is strongly biased toward rank two: accuracy is 40.1% under rank-one truth and 90.0% under rank-two truth. Extra flexibility can therefore look like mode detection even out of sample.

We calibrate the log forecast-error ratio at the 95th percentile of a heterogeneous rank-one null separately for every split. This fixes the simulated false-positive rate at 5%. Rank-two detection power is then only 0.43--0.625 across the five splits. With 21 checkpoints and 1% noise, reliable modal-order classification is still underpowered for the studied mixture.

## Interpretation

The broader baseline suite separates two failures:

- The published curves reject a simple rank-one description, but do not distinguish a second exponential mode from a power-law tail.
- The local reproduction does not support a stable smooth response law at all; persistence usually wins.

The kernel formalism remains useful because it makes these distinctions testable. The appropriate empirical claim is now narrower: single-exponential self-improvement is falsifiable and often forecast-inadequate, while modal mechanism identification remains unresolved.

## Spectral follow-up

The direct Hankel/matrix-pencil experiment is recorded in [SPECTRAL_MODE_AUDIT.md](SPECTRAL_MODE_AUDIT.md). It explains why the forecast comparison is more informative than direct mode counting at this resolution. With 21 checkpoints and 1% noise, an endpoint-free spectral selector chooses the correct rank-two order in only 1.15% of heterogeneous synthetic cases, and a matrix pencil supplied with the true rank recovers both rank-two poles within 0.05 in only 12.5% of cases.

The published curves therefore cannot be called rank one merely because the spectral selector accepts rank one. That test is underpowered. The useful spectral signal is cross-observable: conditional dominant poles are stable and ordered gap below solver below verifier in all four panels, with non-overlapping perturbation intervals. This contradicts one exact shared mode while leaving two discrete modes versus a broader spectrum unresolved.

## Limitations

- The published values are digitized and their split errors are correlated.
- The synthetic power calculation is conditional on the chosen mode, weight, endpoint, and noise distributions.
- The local-linear and power-law grids are explicit baselines, not exhaustive nonparametric model selection.
- Winner counts are descriptive; they are not independent hypothesis tests.
- The reproduction uses a small evaluation set and a 0.5B model.

## Reproduce

From `modal_self_improvement_dynamics`:

```powershell
.\reproduction\.venv\Scripts\python.exe .\scripts\forecast_baseline_audit.py `
  --project-root . --synthetic-replicates 200 `
  --synthetic-noise 0.01 --seed 20260721
```

Generated artifacts:

- `results/rolling_forecast_details.csv`;
- `results/rolling_forecast_summary.csv`;
- `results/rolling_forecast_series_summary.csv`;
- `results/rolling_forecast_metadata.json`;
- `figures/fig4_rolling_forecast.png` and `.pdf`.

The follow-up spectral artifacts are listed in [SPECTRAL_MODE_AUDIT.md](SPECTRAL_MODE_AUDIT.md).
