# Recursive prompting pilots

## Recursive output populations

`bedrock_recursive_population.py` is the primary experiment. Nova Lite
reproduces a structured population over outer rounds. Raw replacement is
compared with retained anchors, exact diversity selection, and an explicit
diversity prompt. Marginal entropy and distinct-combination coverage provide
fully programmatic collapse/improvement measurements.

```powershell
python .\modal_self_improvement_dynamics\recursive_prompting\bedrock_recursive_population.py
```

The population script refuses a run whose conservative preflight estimate
exceeds `$1` unless `--max-estimated-usd` is explicitly changed. This is a
per-run guard; cumulative spending must still be tracked across runs.

## Hidden-rule calibration

`bedrock_recursive_prompting.py` is retained as a negative calibration rather
than primary evidence. It treats an in-context demonstration set as the
recursively updated state while Nova Lite chooses among four Boolean rules.
Nova Lite showed strong default-answer behavior and did not reliably recover
the rule even from perfect contexts. A concrete value in an early JSON-format
example also induced a striking response bias. The population experiment was
introduced to remove this reasoning confound.

The calibration implements four regimes:

- `raw_replace`: replace the context with unfiltered model outputs;
- `verified`: retain only outputs accepted by an exact verifier;
- `anchored`: retain the original correct example alongside new outputs;
- `gold`: replace the context with externally correct examples.

The two initial examples are deliberately compatible with all four rules. This
is an inexpensive prompt-level analogue of recursive self-training, not a
claim about weight updates. Data generation is stochastic while evaluation is
deterministic, so trajectory changes reflect changes in recursive context rather
than evaluation sampling noise. Because the task has an exact hidden rule,
verification and evaluation do not use an LLM judge.

Run from the repository root:

```powershell
python .\modal_self_improvement_dynamics\recursive_prompting\bedrock_recursive_prompting.py
```

The default run makes at most 480 Bedrock calls (8 worlds, four evaluation
regimes, and three model-generated update regimes) with a 128-token output cap
per call. Results are written under `recursive_prompting/results/`.
