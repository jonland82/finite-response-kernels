# Recursive population experiment: first scaled result

## Scope

This is a prompt-level cultural-transmission experiment with fixed model
weights, not recursive parameter training. It tests whether the takeoff-kernel
language can distinguish collapse, near-neutral propagation, and improvement
across outer feedback rounds.

The run used Amazon Nova Lite, 16 independent replicates, 10 recursive rounds,
24 records per population, and four interventions. Each record has four
four-valued categorical attributes. The initial population covers every
attribute value but has deliberately imbalanced marginals. The response score
is the mean of normalized marginal entropy and the fraction of distinct joint
attribute combinations.

## Result

| Intervention | Initial | Round 10 | Gain | Replicates with positive gain |
|---|---:|---:|---:|---:|
| Raw replacement | 0.8699 | 0.8777 | +0.0078 | 10/16 |
| Fixed anchors | 0.8699 | 0.7387 | -0.1312 | 0/16 |
| Diversity verification | 0.8699 | 0.9957 | +0.1259 | 16/16 |
| Diversity-guided prompt | 0.8699 | 0.9622 | +0.0923 | 16/16 |

The interventions produce distinct response structures:

- Diversity verification has a strongly front-loaded positive response and
  reaches its plateau by roughly rounds 3--4.
- Diversity-guided prompting also jumps early, then relaxes to a slightly lower
  positive plateau.
- Raw replacement is mixed and approximately neutral: it improves initially,
  then partially regresses, with positive terminal gain in 10 of 16 replicates.
- Fixed anchoring has a delayed negative tail. Marginal entropy remains high,
  but distinct joint combinations fall from 0.875 to 0.581. Repeatedly injecting
  the same small anchor subset preserves individual attribute values while
  concentrating the joint distribution.

This is direct evidence for the proposed two-part description

\[
\text{intervention }\xi \longmapsto (g_\xi,\kappa_\xi):
\]

terminal outcome and temporal response shape vary separately. It is not yet
evidence about weight-level model collapse or capability takeoff.

## Cost and artifacts

The scaled run made 640 successful calls, used 344,406 input tokens and 332,067
output tokens, and cost an estimated `$0.10036` at the cited public Nova Lite
rates. The complete measured total over all completed calibration and population
runs in this session was about `$0.16009`; interrupted Pro and Haiku probes did
not write usage summaries, so a conservative all-in upper bound is below `$0.50`.

Primary artifacts are in `population_results/`:

- `recursive_population.png`
- `aggregate.csv`
- `trajectory.csv`
- `terminal_summary.csv`
- `kernel.csv`
- `usage.csv`
- `metadata.json`
