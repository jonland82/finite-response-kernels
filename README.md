# Finite Response Kernels

**Jonathan R. Landers**

[Website](https://jonland82.github.io/finite-response-kernels/)

This repository presents a small research program on finite-response kernels:
a cohesive set of manuscripts and computational notes about takeoff kernels,
mediated response, language-model grokking, and the information carried by the
finite shape of response rather than only its limiting rate.

A related side exploration looks at smoothness in classical motion, tracing how
discontinuities, impulsive jerk, and ballistic or diffusive mediation sit as a
nearby twig of the same finite-response picture.

The shared thread is that a limiting rate or asymptotic invariant is not the
full response. Finite response has shape. A physical mediator spreads an
idealized instantaneous source through a causal kernel; a recursive computation
reveals redundancy through a velocity takeoff curve before memoization collapses
the recursion tree into a DAG; and an observed takeoff curve can be read
backward as a system-identification problem. In each case, the kernel is the
object that records how the response turns on, spreads, settles, or decomposes
into modes.

The newest paper applies this viewpoint to delayed generalization in a small
causal Transformer. Across six modular-addition runs, the apparent late
"takeoff" resolves into a two-stage response: roughly 30 percent of the fitted
gain arrives early and 70 percent in a later, sharper component.

## Formal Thread

A mediated physical response is written as a source convolved with a causal
kernel:

$$F_{\mathrm{actual}}(x,t)=\int_{\mathbb{R}^d}\int_{-\infty}^{t}G(x-x',t-s)F_{\mathrm{ideal}}(x',s)\,ds\,dx'.$$

$$K(t)\ge 0,\qquad K(t)=0\text{ for }t<0,\qquad \int_0^\infty K(t)\,dt=1.$$

Sequential stages compose by convolution, while parallel channels mix
convexly:

$$G_{\mathrm{total}}=G_2*G_1,\qquad G_\lambda=(1-\lambda)G_1+\lambda G_2.$$

For recursive redundancy, the same finite-response viewpoint appears in the
tree-size recurrence

$$N_n=1+\sum_{j\in J}a_jN_{n-j},\qquad u_n=\log N_{n+1}-\log N_n.$$

$$F_n=\frac{u_n}{\alpha},\qquad \kappa_n=F_n-F_{n-1}.$$

Here $F_n$ is the normalized takeoff profile and $\kappa_n$ is the causal
takeoff kernel. The inverse manuscript asks how much hidden recurrence
structure can be recovered from an observed takeoff profile:

$$\text{observed takeoff}\longrightarrow\text{modal decomposition}\longrightarrow\text{candidate recurrence mechanisms}.$$

## Contents

- Core manuscripts

  - [`model_trajectory_dynamics/takeoff_as_finite_response.pdf`](model_trajectory_dynamics/takeoff_as_finite_response.pdf) studies delayed generalization as a resolved finite response. Six independent modular-addition runs reject a single unimodal response kernel and support an early component followed by a sharper second lobe.
  - [`entropy_of_finite_response/`](entropy_of_finite_response/) develops passive causal kernels as finite-response mediators, including closure under convolution and mixing, moment growth, entropy power, and the geometry between instantaneous and spreading response.
  - [`velocity_takeoff_revised/`](velocity_takeoff_revised/) introduces velocity takeoff kernels for recursive redundancy and shows how systems with the same limiting overlap velocity can have different finite takeoff shapes.
  - [`inverse_takeoff_kernels/`](inverse_takeoff_kernels/) asks what hidden recurrence structure can be recovered from an observed takeoff curve, separating identifiable modal structure from underdetermined mechanisms.

- Future work

  - [`response_spectrum_of_recursive_self_improvement/`](self_improvement_takeoff_kernels/response_spectrum_of_recursive_self_improvement/) extends the program to recursive model feedback. An LLM generates outputs, a successor rule selects what survives, and those survivors seed the next round. The resulting signed kernel records whether the loop collapses, churns without lasting gain, or improves—and when that change occurs.
    - [`recursive_prompting/`](modal_self_improvement_dynamics/recursive_prompting/) contains the fixed-weight pilot, scripts, and scaled results behind the note. The next step is to sweep intervention strength continuously before moving to weight-updating recursive training.

- Side exploration

  - [`classical_motion/`](classical_motion/) studies the smoothness hierarchy of classical motion, connecting impulsive jerk and ballistic or diffusive mediation to the broader finite-response picture.

## Primary Artifacts

- `entropy_of_finite_response/entropy_of_finite_response.tex`
- `entropy_of_finite_response/entropy_of_finite_response.pdf`
- `velocity_takeoff_revised/velocity_takeoff_revised.tex`
- `velocity_takeoff_revised/velocity_takeoff_revised.pdf`
- `velocity_takeoff_revised/velocity_takeoff_revised.html`
- `inverse_takeoff_kernels/inverse_takeoff_kernels.tex`
- `inverse_takeoff_kernels/inverse_takeoff_kernels.pdf`
- `model_trajectory_dynamics/takeoff_as_finite_response.tex`
- `model_trajectory_dynamics/takeoff_as_finite_response.pdf`
- `self_improvement_takeoff_kernels/response_spectrum_of_recursive_self_improvement/response_spectrum_of_recursive_self_improvement.tex`
- `self_improvement_takeoff_kernels/response_spectrum_of_recursive_self_improvement/response_spectrum_of_recursive_self_improvement.pdf`
- `classical_motion/smoothness_hierarchy_classical_motion.html`

The `.tex` files are the source of record for the manuscripts. PDFs and the
HTML exports are included for convenient reading.

## Building the Manuscripts

The manuscripts use standard LaTeX packages and inline bibliographies. Run each
build from its manuscript directory so local figure paths resolve correctly:

```powershell
Push-Location entropy_of_finite_response
pdflatex -interaction=nonstopmode -halt-on-error entropy_of_finite_response.tex
Pop-Location

Push-Location velocity_takeoff_revised
pdflatex -interaction=nonstopmode -halt-on-error velocity_takeoff_revised.tex
Pop-Location

Push-Location inverse_takeoff_kernels
pdflatex -interaction=nonstopmode -halt-on-error inverse_takeoff_kernels.tex
Pop-Location

Push-Location model_trajectory_dynamics
pdflatex -interaction=nonstopmode -halt-on-error takeoff_as_finite_response.tex
Pop-Location

Push-Location self_improvement_takeoff_kernels/response_spectrum_of_recursive_self_improvement
pdflatex -interaction=nonstopmode -halt-on-error response_spectrum_of_recursive_self_improvement.tex
Pop-Location
```

Run `pdflatex` a second time if references or table-of-contents entries need to
settle after edits.

## Regenerating Figures

Figure scripts are included with the relevant manuscripts.

```powershell
python entropy_of_finite_response/kernels.py
python velocity_takeoff_revised/velocity_takeoff_figures/experiments.py
python self_improvement_takeoff_kernels/response_spectrum_of_recursive_self_improvement/figures.py
```

The scripts expect a scientific Python environment with packages such as
`numpy`, `scipy`, and `matplotlib`.

## Repository Hygiene

Generated LaTeX auxiliaries such as `.aux`, `.log`, and `.out` are intentionally
ignored. Keep manuscript sources, figures, PDFs, scripts, and substantial notes
under version control; leave disposable build products out of commits.
