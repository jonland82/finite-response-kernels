# LLM takeoff-kernel experiment

Status: **complete**.

Synthetic estimator validation passed: **True**.
Completed confirmation trajectories analyzed: **6**.
Finite takeoff preferred to a checkpoint-delta model: **6/6**.
A two-component transition won held-out checkpoint likelihood: **6/6**.
A held-out 1,000-update valley rejected a single unimodal kernel: **6/6**.
The valley remained positive across 800-, 1,000-, and 1,500-update windows: **18/18** seed-by-width checks.
Two settling modes passed the BIC threshold: **0/12** fitted windows.
Median first-crossing width from 25% to 90% accuracy: **3850 updates**.
Median first-crossing width from 50% to 90% accuracy: **1000 updates**.

The two-component transition result concerns two separated rises in held-out accuracy. The settling-mode test asks a different question---whether the residual after onset is better represented by one or two exponentials---and selected one mode here. The abrupt-versus-finite conclusion is relative to the 50-update checkpoint spacing; neither fit by itself establishes distinct physical mechanisms.

```json
{
  "status": "complete",
  "synthetic_validation_passed": true,
  "confirmation_runs_analyzed": 6,
  "finite_takeoff_count": 6,
  "mixture_best_heldout_count": 6,
  "single_mode_rejected_count": 6,
  "single_mode_confirmation_count": 6,
  "valley_sensitivity_positive_count": 18,
  "valley_sensitivity_comparison_count": 18,
  "median_primary_valley_certificate_per_1000": 0.13367191553159236,
  "minimum_primary_valley_error_radius": 0.012244707513464513,
  "median_primary_valley_error_radius": 0.03341797888289809,
  "valley_estimator_validation": {
    "passed": true,
    "one_mode_certificate_per_1000": -0.005199244214992869,
    "two_mode_certificate_per_1000": 0.26393035654158525
  },
  "two_mode_selected_count": 0,
  "modal_fit_count": 12,
  "median_standardized_kernel_wasserstein": 0.20606317576085748,
  "median_kernel_10_90_width_steps": 5780.598522251447,
  "median_logistic_width_steps": 395.5235022160583,
  "median_observed_25_90_width_steps": 3850.0,
  "median_observed_50_90_width_steps": 1000.0,
  "analysis_script_sha256": "19c868b4efb184254cc969cb1c837e67702906c4d1343cb8d5ad9447e5e6fa8e"
}
```
