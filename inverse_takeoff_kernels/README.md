# Inverse Takeoff Kernels

Working seed for a follow-up paper to the velocity takeoff kernel note.

Main question:

Can an observed composite redundancy takeoff curve be used to infer the
latent recurrence modes, and then propose candidate recursive mechanisms?

The draft skeleton is in `inverse_takeoff_kernels.tex`. It frames the problem
as system identification for recursive redundancy:

- recover a minimal modal model from the observed takeoff response;
- separate what is identifiable at the modal level from what is not
  identifiable at the branch/source level;
- add constrained reconstruction assumptions such as sparsity, integer
  branch counts, bounded lag, and separated fast/slow supports;
- test the method on synthetic recurrences and real recursive traces.

The intended thesis is:

```
Takeoff identifies response modes before it identifies source mechanisms.
```
