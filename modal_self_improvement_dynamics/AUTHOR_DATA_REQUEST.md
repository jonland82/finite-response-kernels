# Draft data request

Subject: Numerical trajectories for solver-verifier self-improvement dynamics

Dear authors,

I am studying response kernels and possible finite-modal extensions of the solver-verifier dynamics in your ICLR 2026 paper. Your public vector figures allow an exploratory held-out comparison, but a reliable test requires the underlying trajectories.

Would you be willing to share the numerical solver uncertainty, verifier uncertainty, and uncertainty-gap values at every checkpoint for each seed, especially the trajectories underlying Figure 4 and the corresponding appendix figures? If available, the following would be particularly helpful:

- checkpoint or epoch times before aggregation;
- denser early-training checkpoints, if they were recorded;
- seed-level rather than mean-only trajectories;
- exact QE and TF uncertainty definitions and preprocessing;
- model, task, split, verifier, and training-stage labels;
- optimizer-step, learning-rate, data-refresh, and schedule-transition metadata;
- curve-fitting code, endpoint constraints, and fitted parameters; and
- any covariance or paired-seed information connecting solver, verifier, and gap;
- independent capability, response-length, and decoding statistics that can
  distinguish uncertainty changes from output-format transitions; and
- condition-level observables or interventions that would support source
  separation rather than only response-mode fitting.

The immediate question is whether the three trajectories share one exponential
rate over the full run, whether that law is valid only during an early training
phase, or whether a broader response kernel forecasts better. A fitted response
mode would not by itself be interpreted as a causal mechanism; source
separation would require independent observables or interventions. I would
clearly attribute the data and share the analysis and derived results with you.

Best,

Jonathan R. Landers
