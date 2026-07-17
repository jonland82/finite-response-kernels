# Sun et al. trajectory-data audit

## Bottom line

The public artifact supports an exploratory modal audit, but not a definitive reanalysis. The arXiv vector figure preserves 240 plotted points: 12 Phi-4-mini trajectories, each sampled at 20 half-epoch checkpoints. On held-out checkpoints, rank two improves on rank one for 10 of the 12 curves. Nine curve-level tests reject a fitted rank-one null at the uncorrected 5% level, and seven remain after Holm-Bonferroni correction across all 12 curves.

A broader rolling forecast audit changes the interpretation. Across five train/test boundaries and five forecast models, power-law relaxation wins 27 of 60 curve--split cases, rank two wins 16, rank one wins 13, and local-linear smoothing wins four. Power law beats both modal models in 30 cases. No curve has the same winner at all five boundaries. The curves are therefore inconsistent with a rigid one-exponential description, but they do not specifically identify a second exponential mode.

The stronger kernel restriction also looks too rigid. The standard coupled law predicts one shared mode for solver uncertainty, verifier uncertainty, and their gap. In each of the four task/metric panels, a shared rank-one fit forecasts worse than three separately fitted rank-one curves. The held-out mean-squared-error factors are 4.09 (Math/QE), 18.55 (GSM8K/QE), 10.06 (Math/TF), and 4.32 (GSM8K/TF).

These results are evidence that the published one-mode description is forecast-incomplete. They are not evidence that two physical mechanisms have been identified.

A direct inverse spectral audit sharpens this point. The endpoint-free Hankel selector accepts rank one on all 60 rolling published cases, but matched rank-two simulations are rejected as rank one only 3.95% of the time at a 5% false-positive rate. The acceptance is therefore non-informative. Conditional one-pole estimates are perturbation-stable and satisfy gap below solver below verifier in all four panels, with pairwise non-overlapping 10th--90th percentile perturbation intervals. This is evidence against the exact shared-single-rate restriction, not an identified modal count.

## Public-artifact inventory

Source inspected: Sun et al., *Theoretical Modeling of Large Language Model Self-Improvement Training Dynamics Through Solver-Verifier Gap*, arXiv:2507.00075v4.

- The arXiv source has 79 files: 11 TeX files, 59 PDF figures, one bibliography, and arXiv build metadata.
- It has no CSV files, analysis scripts, notebooks, or data archives.
- No author code or data repository is linked in the TeX or bibliography.
- The main text states that the validation fits use 10 self-improvement epochs and reports in-sample $R^2>0.9$ for separate exponential fits.
- The selected Phi-4-mini training figure contains four panels: Math/QE, GSM8K/QE, Math/TF, and GSM8K/TF. Each panel plots solver uncertainty, verifier uncertainty, and their gap.

## Exploratory extraction and test

The source figure `fig/fit/dataset_exponential_fit_train_Phi_4.pdf` was converted to SVG. Its colored scatter markers are vector paths, so their centers can be extracted without raster-pixel ambiguity. For every series, the script:

1. extracts 20 marker centers and assigns checkpoints at 0.5-epoch spacing;
2. applies an affine vertical-coordinate map, which preserves exponential modes;
3. fits

   $$
   Y_n=Y_\infty+c\lambda^n
   $$

   and

   $$
   Y_n=Y_\infty+c_1\theta_1^n+c_2\theta_2^n
   $$

   to the first 15 checkpoints;
4. compares mean squared prediction error on the final five checkpoints;
5. calibrates the log error ratio under a fitted rank-one null using 2,000 parametric-bootstrap trajectories with i.i.d. Gaussian residuals; and
6. applies Holm-Bonferroni correction across the 12 curve-level tests.

The seven corrected rejections are Math/QE solver; all three Math/TF curves; and all three GSM8K/TF curves. The joint shared-mode comparison is descriptive rather than a formal test because solver, verifier, and gap are algebraically related and the figure does not provide their seed-level covariance.

## Artifacts

- Extraction and audit script: [scripts/digitize_sun_phi4.py](scripts/digitize_sun_phi4.py)
- Extracted vector points: [results/sun_phi4_figure4_digitized.csv](results/sun_phi4_figure4_digitized.csv)
- Curve-level summary: [results/sun_phi4_figure4_modal_summary.csv](results/sun_phi4_figure4_modal_summary.csv)
- Full fits, forecasts, bootstrap results, and provenance hash: [results/sun_phi4_figure4_modal_audit.json](results/sun_phi4_figure4_modal_audit.json)
- Rolling modal/power-law/nonparametric audit: [BASELINE_AUDIT.md](BASELINE_AUDIT.md)
- Calibrated Hankel and matrix-pencil audit: [SPECTRAL_MODE_AUDIT.md](SPECTRAL_MODE_AUDIT.md)

## Limits on the claim

- The values are figure-derived aggregate points, not raw examples or independent seed trajectories.
- The bootstrap assumes independent Gaussian residuals and estimates noise from the same short curve.
- The gap shares data with solver and verifier, so the 12 curve tests are not independent. Holm correction controls the stated family under arbitrary dependence, but it does not repair the fitted noise model.
- The modes lie on a fixed grid. With only 15 training and five test checkpoints, mode and endpoint estimates are substantially less reliable than detecting a forecast failure.
- A better two-mode forecast does not identify a causal mechanism. It can also represent endpoint misspecification, time-varying rates, averaging across seeds, or another smooth departure from one exponential.
- Power-law forecasts often match the digitized curves better, and the best model changes with the split. This makes the departure from rank one less mechanism-specific.
- The original paper's high in-sample $R^2$ and this held-out comparison answer different questions. A high $R^2$ does not establish the shared-rate restriction or rule out a better multi-mode forecast.

## What a paper-grade raw-data test would need

If exact numerical curves ever become available, the useful fields would be solver, verifier, and gap values at every checkpoint for every seed, together with checkpoint timing, uncertainty definitions, model/task/verifier labels, fitting code, and the aggregation used in each figure. At roughly 1% response-scale noise, the synthetic studies show that rank-one forecast failure can be detectable with short curves while modal-order recovery remains unreliable.
