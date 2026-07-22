# Constraint-aware sharp-boundary comparison

- Schemas: 38,796
- Method trials: 2,327,760
- Boundary: T = D - 1 observed transitions F_0,...,F_{D-2}.
- Nonidentifiable control: T = D - 2 observed transitions F_0,...,F_{D-3}.

## Noiseless boundary snapshot

| Method | Regime | Exact source | Forecast success | Median NRMSE |
|---|---:|---:|---:|---:|
| constraint_integer | nonidentifiable | 0.450 | 0.633 | 0.000101 |
| constraint_integer | sharp_boundary | 1.000 | 1.000 | 0 |
| continuous_round | nonidentifiable | 0.002 | 0.310 | 0.512 |
| continuous_round | sharp_boundary | 0.023 | 0.498 | 0.103 |
| matrix_pencil_6 | nonidentifiable | n/a | 0.191 | 0.769 |
| matrix_pencil_6 | sharp_boundary | n/a | 0.197 | 0.495 |

See `summary.csv` for the full noise-by-regime phase diagram.

## Boundary robustness

At the identifying horizon, the constraint-aware decoder retained exact source
recovery in 100.0%, 99.98%, 70.1%, and 38.6% of trials at relative noise
levels 0, $10^{-8}$, $10^{-5}$, and $10^{-3}$. Its corresponding 16-step
forecast success rates were 100.0%, 100.0%, 99.6%, and 94.2%.

At $T=D-2$, the reported 45.0% exact-source rate is not a uniform guarantee:
the aggregate mixes uniquely determined boundary cases with arbitrary selection
inside multi-schema version sets. Overall, 74.1% of schemas had multiple
compatible tails; the median version count was 43 and the maximum was 174,763.
The theoretical validation construction has 342 compatible sources at $D=11$.

## Runtime and resources

- Instance: `c7i.8xlarge`, 32 workers
- Timed worker phase: 12 minutes
- Wall time from launch to archived results: 13.43 minutes
- Estimated EC2 cost to results: $0.32; configured hard ceiling: $1.00
- Exit status: 0
- Instance terminated; temporary S3 bucket and security group deleted and
  independently verified absent
