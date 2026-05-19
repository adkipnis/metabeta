Analytical GLMM Plan
====================

- [plan_normal.md](plan_normal.md) — Normal GLMM: production EB now defaults to scalar
  β sigma-grid, tail-gated damped β posterior-mean correction, direct σ_rfx EB
  coordinate grid, and the rare BLUP/sigma guard for high-d aliased rows. Post-EB,
  axis, variance-ratio, and curvature-shrink beta variants were tested and removed.
  The remaining gap to INLA is narrowed to rare ill-conditioned fixed-effect tails and
  σ_rfx scale accuracy, not another broad inference branch.
- [plan_bernoulli.md](plan_bernoulli.md) — Bernoulli GLMM: default Bernoulli EB path;
  no full INLA or separate amortized-correction branch.
- [plan_poisson.md](plan_poisson.md) — Poisson GLMM: production now defaults to Poisson
  EB, reusing the retained Bernoulli diagonal Laplace-EB structure with Poisson-specific
  likelihood stabilization, accepted-row σ calibration, and RAW/PQL BLUP fallback.

Testing Scheme
--------------

### Required benchmark (our method, fast)

1. Always start with `size=small`. Proceed to `medium`, then `large`/`huge` only if no regressions.
2. `ds_type=mixed` — use the first two training epochs (`--n-epochs 2`).
3. `ds_type=sampled` — run both `valid` and `test` partitions separately.

### Competitor comparisons (R-INLA or any slower reference)

- Use the first 1000 dataset indices of each set for both the reference method and our method.
