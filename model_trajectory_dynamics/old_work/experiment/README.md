# Closed-source language-model pilot

This experiment is a finite-horizon empirical companion to
`closed_source_causal_dilution.tex`. It trains independent replicas of a small
decoder-only Transformer on synthetic record/code sentences.

Each closed corpus contains 16 independently sampled red/blue code tokens, so
its source entropy is exactly 16 bits. The same sentences are replayed for 400
optimizer updates. The open control starts with the same number of facts and
admits one fresh random fact every 20 updates until 32 facts are available.

The principal observable is the model's red/blue log-odds for every record.
For a source bit `D_i` and logit margin `z_i`, the decoder
`q(D_i=1|z_i)=sigmoid(z_i)` yields the variational lower bound

```text
I(D_i; Z) >= 1 - binary_cross_entropy(D_i, sigmoid(z_i)) / log(2).
```

Summing across bits and averaging across independent corpus realizations gives
the reported recoverable-information lower bound. Differences between
checkpoints are an information-increment proxy, not an estimator of the exact
conditional mutual information in the complete weight trajectory.

The influence experiment forks selected closed runs, gives one example weights
`1 + epsilon` and `1 - epsilon` for one update, and measures the resulting
central finite difference in later logits. The curves remain unnormalized in
`influence_lags.csv`; `normalized_mass` is a finite-window descriptive summary.

## Local smoke test

```powershell
python model_trajectory_dynamics/experiment/run_experiment.py `
  --smoke `
  --output-dir model_trajectory_dynamics/experiment/smoke_results
```

## Full run

```bash
python run_experiment.py --output-dir results --max-runtime-seconds 2700
```

Artifacts include raw CSV data, compressed NumPy observables, a PNG summary,
machine-readable metadata, and an automatically generated run report.

## Interpretation boundary

This experiment can demonstrate saturation of recoverable source information
in an observable language-model trajectory and contrast it with a growing
source. It cannot verify an asymptotic theorem, and it does not identify the
exact mutual information of unquantized weights. The influence measurements
test passivity and finite-window response shape; they do not establish that
training stages compose by convolution.

## Long-horizon composition test

`run_causal_composition.py` performs the follow-up operational test of the
kernel construction. It measures 300-update responses, compares independently
measured two-block responses with convolutions of two 100-update response
curves, and checks sensitivity to perturbation size. The AWS results and their
interpretation are documented in
[`CAUSAL_COMPOSITION_RESULTS.md`](CAUSAL_COMPOSITION_RESULTS.md).

The controlled three-seed confirmation is in
[`CAUSAL_COMPOSITION_REPLICATION.md`](CAUSAL_COMPOSITION_REPLICATION.md).
