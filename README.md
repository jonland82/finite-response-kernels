# Finite Response Kernels

**Jonathan R. Landers**

This repository collects draft manuscripts and supporting computational material
on finite-response kernels, takeoff kernels, and the shape of mediated
response.

The shared thread is that a limiting rate or asymptotic invariant is not the
full response. Finite response has shape. A physical mediator spreads an
idealized instantaneous source through a causal kernel; a recursive computation
reveals redundancy through a velocity takeoff curve before memoization collapses
the recursion tree into a DAG; and an observed takeoff curve can be read
backward as a system-identification problem. In each case, the kernel is the
object that records how the response turns on, spreads, settles, or decomposes
into modes.

## Formal Thread

A mediated physical response is written as a source convolved with a causal
kernel:

$$
F_{\mathrm{actual}}(x,t)
=
\int_{\mathbb R^d}\int_{-\infty}^{t}
G(x-x',t-s)F_{\mathrm{ideal}}(x',s)\,ds\,dx'.
$$

$$
K(t)\ge 0,\qquad K(t)=0\text{ for }t<0,\qquad
\int_0^\infty K(t)\,dt=1.
$$

Sequential stages compose by convolution, while parallel channels mix
convexly:

$$
G_{\mathrm{total}}=G_2*G_1,
\qquad
G_\lambda=(1-\lambda)G_1+\lambda G_2.
$$

For recursive redundancy, the same finite-response viewpoint appears in the
tree-size recurrence

$$
N_n=1+\sum_{j\in J}a_jN_{n-j},
\qquad
u_n=\log N_{n+1}-\log N_n.
$$

$$
F_n=\frac{u_n}{\alpha},
\qquad
\kappa_n=F_n-F_{n-1}.
$$

Here $F_n$ is the normalized takeoff profile and $\kappa_n$ is the causal
takeoff kernel. The inverse manuscript asks how much hidden recurrence
structure can be recovered from an observed takeoff profile:

$$
\text{observed takeoff}
\longrightarrow
\text{modal decomposition}
\longrightarrow
\text{candidate recurrence mechanisms}.
$$

## Contents

| Directory | Description |
| --- | --- |
| `entropy_of_finite_response/` | Develops passive causal kernels as finite-response mediators. It studies closure under convolution and mixing, moment growth, entropy power, and the geometry between the delta-like instantaneous boundary and the Gaussian spreading boundary. |
| `velocity_takeoff_revised/` | Introduces velocity takeoff kernels for recursive redundancy. It shows that recursive schemas can have the same terminal overlap velocity while exhibiting different finite takeoff shapes, so the kernel carries information that first-order asymptotics miss. |
| `inverse_takeoff_kernels/` | Poses the inverse problem. Given an observed redundancy takeoff curve, it separates recoverable modal structure from underdetermined source-level mechanisms and frames reconstruction as constrained system identification. |

## Primary Artifacts

- `entropy_of_finite_response/entropy_of_finite_response.tex`
- `entropy_of_finite_response/entropy_of_finite_response.pdf`
- `velocity_takeoff_revised/velocity_takeoff_revised.tex`
- `velocity_takeoff_revised/velocity_takeoff_revised.pdf`
- `velocity_takeoff_revised/velocity_takeoff_revised.html`
- `inverse_takeoff_kernels/inverse_takeoff_kernels.tex`
- `inverse_takeoff_kernels/inverse_takeoff_kernels.pdf`

The `.tex` files are the source of record for the manuscripts. PDFs and the
HTML export are included for convenient reading.

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
```

Run `pdflatex` a second time if references or table-of-contents entries need to
settle after edits.

## Regenerating Figures

Figure scripts are included with the relevant manuscripts.

```powershell
python entropy_of_finite_response/kernels.py
python velocity_takeoff_revised/velocity_takeoff_figures/experiments.py
```

The scripts expect a scientific Python environment with packages such as
`numpy`, `scipy`, and `matplotlib`.

## Repository Hygiene

Generated LaTeX auxiliaries such as `.aux`, `.log`, and `.out` are intentionally
ignored. Keep manuscript sources, figures, PDFs, scripts, and substantial notes
under version control; leave disposable build products out of commits.
