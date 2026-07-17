# Draft data request

Subject: Numerical trajectories for solver-verifier self-improvement dynamics

Dear authors,

I am studying a finite-modal extension of the exponential solver-verifier dynamics in your ICLR 2026 paper. Your public vector figures allow an exploratory held-out comparison, but a reliable test requires the underlying trajectories.

Would you be willing to share the numerical solver uncertainty, verifier uncertainty, and uncertainty-gap values at every checkpoint for each seed, especially the trajectories underlying Figure 4 and the corresponding appendix figures? If available, the following would be particularly helpful:

- checkpoint or epoch times before aggregation;
- seed-level rather than mean-only trajectories;
- exact QE and TF uncertainty definitions and preprocessing;
- model, task, split, verifier, and training-stage labels;
- curve-fitting code, endpoint constraints, and fitted parameters; and
- any covariance or paired-seed information connecting solver, verifier, and gap.

The immediate question is whether the three trajectories share one discrete exponential mode, as the coupled constant-coefficient law predicts, or whether a finite two-mode response improves held-out tail prediction. I would clearly attribute the data and share the analysis and derived results with you.

Best,

Jonathan R. Landers
