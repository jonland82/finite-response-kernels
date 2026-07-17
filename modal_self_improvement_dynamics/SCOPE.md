# Modal Self-Improvement Dynamics — Scope and Claim Map

## Working thesis

The standard linear solver–verifier model of self-improvement is exactly a **rank-one takeoff-kernel model** after sampling training time at checkpoints. The takeoff kernel itself is more general: it is the normalized response sequence, which may be signed, phase-changing, broad-spectrum, or nonmodal. Rank-one and finite-modal laws are parameterizations of that response. The rank-one observation supplies stronger tests than fitting separate exponential curves: the normalized solver, verifier, and gap trajectories must share one decay mode; their kernel ratios must be constant; and every $2\times2$ Hankel minor must vanish.

The project will first test that rank-one null. It will not assume in advance that real self-improvement is multi-modal, that a single stationary law governs the full training run, or that response modes correspond to distinct mechanisms. The empirical comparison asks whether the exponential restriction is global, local to startup, replaced by changing training phases, or simply a poor description of an aggregate measurement.

## Primary target

The initial application is Sun et al., *Theoretical Modeling of Large Language Model Self-Improvement Training Dynamics Through Solver–Verifier Gap* (ICLR 2026), [arXiv:2507.00075](https://arxiv.org/abs/2507.00075).

Their phenomenological model defines solver and verifier uncertainties $U_s(t)$ and $U_v(t)$, the uncertainty gap

$$
G(t)=U_s(t)-U_v(t),
$$

and a gap potential $E(t)=f(G(t))$. The assumed dynamics are

$$
\frac{dU_s}{dt}=-\alpha E(t),
\qquad
\frac{dU_v}{dt}=-\beta E(t),
\qquad
\alpha>\beta\geq 0.
$$

With the local linearization $E(t)\approx kG(t)-b$, their convergence rate is

$$
\gamma=k(\alpha-\beta),
$$

so $U_s-U_{s,\infty}$, $U_v-U_{v,\infty}$, and $G-G_\infty$ are all proportional to $e^{-\gamma t}$. At checkpoint spacing $\Delta$, the shared discrete mode is therefore

$$
\lambda=e^{-\gamma\Delta}.
$$

This is the precise bridge to takeoff kernels.

## Two clocks

- **Inner clock $t$ or $n$:** optimization time or checkpoints within one self-improvement training run. The Sun et al. differential equations model this clock.
- **Outer clock $r$:** complete generate–verify–retrain rounds in which a successor model supplies data or feedback for the next successor.

Results for the inner clock do not by themselves establish sustained recursive takeoff. The outer-round process can have a different kernel, and coupling the two clocks is a later modeling step.

## Stable notation

| Symbol | Meaning |
|---|---|
| $t$ | Continuous inner training time |
| $n$ | Discrete checkpoint index, with $t=n\Delta$ |
| $r$ | Outer self-improvement round |
| $U_s,U_v$ | Solver and verifier uncertainty; lower is better |
| $G=U_s-U_v$ | Solver–verifier uncertainty gap |
| $E=f(G)$ | Gap potential driving the standard dynamics |
| $\alpha,\beta$ | Solver and verifier response coefficients in the target paper |
| $k$ | Local slope of the gap potential |
| $\gamma=k(\alpha-\beta)$ | Continuous-time decay rate under the linearized target model |
| $Y_n$ | Generic scalar observable with limit $Y_\infty$ |
| $e_n$ | Normalized residual distance to the limit |
| $F_n=1-e_n$ | Cumulative realized response |
| $\kappa_n=F_n-F_{n-1}$ | Takeoff kernel at checkpoint $n$ |
| $\lambda$ | Single discrete decay mode |
| $A$ | Sampled linear state-transition matrix |
| $\theta_j$ | A mode/eigenvalue of $A$ |
| $R$ | Number of distinct active observable modes |

For any scalar trajectory $Y_n\to Y_\infty$ with $Y_0\neq Y_\infty$, use the sign-independent normalization

$$
e_n=\frac{Y_n-Y_\infty}{Y_0-Y_\infty},
\qquad
F_n=1-e_n,
\qquad
\kappa_n=e_{n-1}-e_n.
$$

This covers both increasing performance and decreasing uncertainty.

## Claim hierarchy

### Claim 1 — exact reduction

The target paper's linearized exponential law is not merely similar to a takeoff kernel. At equally spaced checkpoints it is exactly the one-mode case

$$
e_n=\lambda^n,
\qquad
\kappa_n=(1-\lambda)\lambda^{n-1}.
$$

This will be a theorem plus a direct corollary for the solver–verifier equations.

### Claim 2 — stronger verification test

Separate high-$R^2$ exponential fits are weaker than the joint restrictions imposed by the theory. The solver, verifier, and gap series must have the same $\lambda$ after normalization; successive kernel ratios must be constant; and the exact kernel Hankel matrix must have rank one. These become empirical diagnostics, not new assumptions.

### Claim 3 — finite-modal response subclass

A locally linear coupled hidden state with transition matrix $A$ produces

$$
e_n=\frac{\sum_{j=1}^{R}c_j\theta_j^n}{\sum_{j=1}^{R}c_j}.
$$

This is a tractable response subclass. It preserves exponential settling while allowing multiple stages, alternation, or ringing; it is not a universal representation of kernels and does not by itself identify latent sources.

### Claim 4 — inverse response identification

Under distinct active modes and nonzero amplitudes, the exact Hankel rank of $\kappa$ equals $R$. With noisy finite data, singular spectra and held-out forecasting can test whether the rank-one null is adequate. This identifies response order within the finite-modal model, not the physical sources that generated the response.

### Claim 5 — mechanistic interpretation is conditional

Recovered modes identify response timescales before they identify literal mechanisms. A fast and a slow mode may be consistent with solver learning, verifier drift, data refresh, or optimizer effects, but causal labels require interventions or additional measurements.

## Result classification

| Item | Planned status |
|---|---|
| Single-mode/kernel equivalence | Theorem |
| Sun et al. model is rank one after sampling | Corollary/application proposition |
| Shared-mode and local-ratio restrictions | Corollary and empirical null hypothesis |
| Modal state-space representation | Theorem |
| Settling rate from modal radius | Proposition |
| Hankel rank equals active modal count | Theorem |
| Same endpoints, different response modes | Corollary plus constructed example |
| Real trajectories require more than one mode | Empirical hypothesis, not assumed |
| One exponential is valid only during a startup phase | Empirical hypothesis requiring dense early checkpoints |
| A piecewise-rate model beats persistent modes at 20--21 checkpoints | First-pass forecast result on the reproduction; dense follow-up shows one-update behavioral jumps rather than a smooth local law |
| Modes correspond to named mechanisms | Not identified; requires source-separation data or interventions |

## Immediate falsifiable null

For each trajectory family, jointly test

$$
H_0:\quad
e_n^{(s)}=e_n^{(v)}=e_n^{(G)}=\lambda^n
\quad\text{for one shared }\lambda\in(0,1).
$$

Rejecting a separately fitted exponential is not enough; the main comparison is shared rank one versus a regularized shared multi-mode alternative, evaluated by held-out checkpoints and stability across seeds.

## Competing empirical hypotheses

The updated empirical program distinguishes four explanations rather than treating every global rank-one failure as evidence for additional fixed modes.

### H1 — global shared rank one

One mode $\lambda$ governs solver, verifier, and gap over the full observation window.

### H2 — global shared modal set

A fixed set $\{\theta_j\}$ persists across the full run. Solver and verifier have different amplitudes, while the gap amplitudes are their differences because $G=U_s-U_v$. The gap is therefore a derived observable, not a third independent channel. Even if this response model fits, its modes are not automatically causal sources.

### H3 — local or piecewise rank one

One exponential rate is approximately valid within a startup window or training phase, but the active rate changes at one or more transition times. A continuous two-phase exponential is the minimal alternative for the first audit.

### H4 — no resolved smooth law

Plateaus, jumps, measurement noise, or decoding transitions dominate at the available resolution, so persistence or a nonmodal baseline forecasts as well as or better than exponential models.

The dense-start follow-up currently favors this as a measurement warning for the
small reproduction: the sharpest changes occur in one optimizer update and
coincide with response-length and accuracy changes. It does not establish H4
for the larger published systems.

The primary comparison uses held-out rolling forecasts with matched parameter budgets. Local rates or change points count as empirical structure only if they are stable across nearby windows and fresh seeds.

## Boundaries of the first paper

- The main mathematical object is the finite response across observed checkpoints.
- The first empirical target is bounded training improvement, not unbounded capability growth.
- Finite modes are a local/system-identification model, not a claim that neural training is globally linear.
- A global finite-modal fit can imitate a time-varying one-mode process or a broad spectrum; piecewise-rate and nonmodal alternatives must therefore be included before interpreting extra modes.
- A kernel concentrated near startup is still a full-run descriptive kernel, but a kernel fitted inside a changing phase is only a conditional/windowed response and must be labeled accordingly.
- A failure to reject rank one is informative if the synthetic study quantifies what modes the available sampling could have detected.
- A failure of all finite-modal fits may indicate changing parameters, nonlinear regime shifts, or a broad/algebraic response spectrum.
