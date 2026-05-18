Analytical GLMM Plan
====================

- [plan_normal.md](plan_normal.md) — Normal GLMM: production EB now defaults to scalar
  β sigma-grid plus direct σ_rfx EB coordinate grid plus the rare BLUP/sigma guard for
  high-d aliased rows. Post-EB, axis, and variance-ratio beta-grid variants were tested
  and removed; curvature-aware β shrinkage was also tested and removed. The remaining
  actionable gap to INLA is FFX posterior-mean behavior in rare ill-conditioned rows,
  not another broad σ-grid branch.
- [plan_bernoulli.md](plan_bernoulli.md) — Bernoulli GLMM: default Bernoulli EB path;
  no full INLA or separate amortized-correction branch.

Testing Scheme
--------------

### Required benchmark (our method, fast)

1. Always start with `size=small`. Proceed to `medium`, then `large`/`huge` only if no regressions.
2. `ds_type=mixed` — use the first two training epochs (`--n-epochs 2`).
3. `ds_type=sampled` — run both `valid` and `test` partitions separately.

### Competitor comparisons (CAVI, R-INLA, or any slower reference)

- Use the first 1000 dataset indices of each set for both the reference method and our method.
