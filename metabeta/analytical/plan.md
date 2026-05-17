Analytical GLMM Plan
====================

- [plan_normal.md](plan_normal.md) — Normal GLMM: production MAP path now carries MAP β
  for `d > 4`; active priority is validating `normal_eb` and the remaining R-INLA gap.
- [plan_bernoulli.md](plan_bernoulli.md) — Bernoulli GLMM: default P14-cal Laplace-EB path;
  no full INLA or separate amortized-correction branch.

Testing Scheme
--------------

### Required benchmark (our method, fast)

1. Always start with `size=small`. Proceed to `medium`, then `large`/`huge` only if no regressions.
2. `ds_type=mixed` — use the first two training epochs (`--n-epochs 2`).
3. `ds_type=sampled` — run both `valid` and `test` partitions separately.

### Competitor comparisons (CAVI, R-INLA, or any slower reference)

- Use the first 1000 dataset indices of each set for both the reference method and our method.
