# Finite Response Kernels

**Jonathan R. Landers**

This repository collects draft manuscripts and supporting computational material
on finite-response kernels, recursive redundancy, and takeoff behavior in
mediated systems.

The common theme is that a limiting rate or asymptotic invariant is often not
the full story. Finite response has shape: causal mediation kernels spread
instantaneous sources, and recursive programs expose measurable redundancy
takeoff before memoization collapses repeated work.

## Contents

| Directory | Description |
| --- | --- |
| `entropy_of_finite_response/` | Draft note on passive causal kernels, convolution, entropy power, and the geometry of finite physical response. Includes LaTeX source, rendered PDF, figures, and figure-generation code. |
| `velocity_takeoff_revised/` | Manuscript on velocity takeoff kernels for memoization and recursive redundancy. Includes LaTeX source, rendered PDF/HTML, figures, and experiment scripts. |
| `inverse_takeoff_kernels/` | Working draft for the inverse problem: identifying response modes and candidate recursive mechanisms from an observed takeoff curve. |

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
