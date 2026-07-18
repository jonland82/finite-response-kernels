# The Response Spectrum of Recursive Self-Improvement

This folder contains the standalone five-page note and its reproducibility
artifacts.

- `response_spectrum_of_recursive_self_improvement.tex` is the manuscript source.
- `response_spectrum_of_recursive_self_improvement.pdf` is the compiled note.
- `figures.py` rebuilds both figures from the copied result tables in `data/`.
- `data/` is a local snapshot of the scaled recursive-population run originally
  produced in `modal_self_improvement_dynamics/recursive_prompting/population_results/`.

The experiment used fixed Amazon Nova Lite weights. It recursively transformed
prompt-level populations; it did not train or fine-tune a model. The copied
metadata records 640 successful calls, zero API errors, and an estimated run
cost of $0.10036 at the public rates used by the experiment script.

To rebuild locally:

```powershell
python .\figures.py
pdflatex -interaction=nonstopmode -halt-on-error .\response_spectrum_of_recursive_self_improvement.tex
pdflatex -interaction=nonstopmode -halt-on-error .\response_spectrum_of_recursive_self_improvement.tex
```
