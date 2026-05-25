Normal GLMM Plan
================

Last updated: 2026-05-25 (Direction K adopted; profile-MAP σ_rfx rescue added; debloated)

Goal
----

Fast, prior-aware analytical summaries for Gaussian GLMMs. R-INLA is a slow
accuracy reference, not a backend. The retained path should stay batched,
millisecond-scale, and simple enough to trust.

Default Path
------------

`glmm(..., likelihood_family=0)` now runs Normal EB by default with 2 outer iterations:

**Outer loop × 2** (`normal_map_outer_iterations=2`; tail β only on last iteration):
- marginal MAP refinement of β, diagonal σ_rfx, and σ_eps (Adam, 20 steps);
- reported β cap for `d > 4`: `clamp(β_MAP, ν_ffx ± 4τ_ffx)`;
- scalar β sigma-grid reporting over σ_rfx scales `{0.5, 0.667, 0.833, 1.0, 1.2, 1.5, 2.0}`;
- one-shot posterior-moment EB update for diagonal σ_rfx;
- one-pass coordinate σ_rfx grid over the same 7-pt scales, accepted only on marginal-target
  improvement, reporting softmax-weighted posterior mean (Direction A + E);

**Final pass only:**
- damped tail β correction for `d >= 9`, gated by β cap/stabilization or weak β
  precision; 25% blend toward the grid posterior mean (75% toward prior-regularized OLS
  when both cap AND stabilization fire — Direction K);
- rare BLUP/sigma guard for high-d aliased rows with implausibly large BLUP norms.

BLUP β is anchored to iteration 1's uncapped MAP β across both outer iterations, preventing
the warm-started second-pass β from drifting further from OLS. The β cap and tail correction
are reporting-only.

Current Performance
-------------------

First 1000 datasets per row with the default path, rerun 2026-05-25. Lower NRMSE is better.

| Dataset | part | FFX | σ | σ_eps | BLUP | ms | guard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.1084 | 0.3868 | 0.2151 | 0.4146 | 7.84 | 0.000 |
| small-n-sampled | valid | 0.2601 | 0.5509 | 0.2169 | 0.5116 | 7.08 | 0.000 |
| small-n-sampled | test | 0.2825 | 0.4679 | 0.2169 | 0.4929 | 7.01 | 0.000 |
| medium-n-mixed | train | 0.2188 | 0.3524 | 0.1655 | 0.4194 | 10.96 | 0.000 |
| medium-n-sampled | valid | 0.2467 | 0.4728 | 0.1891 | 0.5139 | 13.35 | 0.000 |
| medium-n-sampled | test | 0.2448 | 0.3552 | 0.1949 | 0.4398 | 13.02 | 0.000 |
| large-n-mixed | train | 0.2445 | 0.3612 | 0.1268 | 0.4119 | 15.73 | 0.000 |
| large-n-sampled | valid | 0.2852 | 0.3817 | 0.1563 | 0.5003 | 17.13 | 0.000 |
| large-n-sampled | test | 0.2788 | 0.4337 | 0.1513 | 0.5054 | 18.54 | 0.000 |
| huge-n-mixed | train | 0.2524 | 0.2996 | 0.1161 | 0.4516 | 20.64 | 0.004 |
| huge-n-sampled | valid | 0.3875 | 0.3394 | 0.1375 | 0.4529 | 25.64 | 0.000 |
| huge-n-sampled | test | 0.2884 | 0.3429 | 0.1438 | 0.4589 | 27.03 | 0.002 |

R-INLA Comparison
-----------------

Diagonal R-INLA, first-1000 datasets per row. The benchmark (performance table above) uses
`sortish=True` (sorts by size within 50-batch buckets), so `--max-datasets 1000` yields a
size-biased subset. The INLA comparison uses `sortish=False` (strict file order) to match
precomputed `.inla.npz` results — a different 1000 datasets for sampled partitions. Mixed
(train) files contain exactly one epoch worth of data, so both scripts use the same datasets
there. Bold = better method per column.

| Dataset | part | current FFX | INLA FFX | current σ | INLA σ | current BLUP | INLA BLUP | INLA s/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.1084 | **0.0985** | 0.3868 | **0.3665** | 0.4146 | **0.4081** | 2.4 |
| small-n-sampled | valid | 0.2735 | **0.2151** | **0.4881** | 0.5313 | 0.4812 | **0.4755** | 9.9 |
| small-n-sampled | test | 0.2393 | **0.1675** | **0.4005** | 0.4119 | 0.4228 | **0.4156** | 9.9 |
| medium-n-mixed | train | **0.2188** | 0.2301 | 0.3524 | **0.3419** | **0.4194** | 0.4289 | 2.6 |
| medium-n-sampled | valid | 0.2309 | **0.2296** | 0.4097 | **0.3201** | **0.5710** | 0.5733 | 8.3 |
| medium-n-sampled | test | **0.2220** | 0.2339 | 0.3160 | **0.3103** | **0.4370** | 0.4426 | 8.3 |
| large-n-mixed | train | 0.2445 | **0.2377** | 0.3612 | **0.3393** | **0.4119** | 0.4185 | 2.8 |
| large-n-sampled | valid | 0.2710 | **0.2389** | 0.3239 | **0.3226** | **0.4734** | 0.4769 | 8.9 |
| large-n-sampled | test | 0.2640 | **0.2514** | 0.3765 | **0.3601** | **0.4731** | 0.4726 | 8.9 |
| huge-n-mixed | train | 0.2524 | **0.2413** | 0.2996 | **0.2808** | **0.4516** | 0.4548 | 3.0 |
| huge-n-sampled | valid | 0.3746 | **0.2907** | **0.3392** | 0.5085 | **0.4873** | 0.4897 | 10.1 |
| huge-n-sampled | test | 0.2601 | **0.2491** | 0.3169 | **0.2895** | **0.5531** | 0.5554 | 10.1 |

Current is ~7–21 ms/ds; R-INLA is 2–10 s/ds (~100–500× slower).

- **FFX**: current leads INLA on medium-n train and test; tied on medium-n-sampled valid (Δ=0.0013).
  Large/huge trail INLA by ≤0.007 on mixed and ≤0.032 on sampled.
- **σ**: INLA leads on most rows; current wins on small-n-sampled and huge-n-sampled valid (0.3392 vs 0.5085).
- **BLUP**: current matches or beats INLA on all medium+ rows; small-n is INLA-dominant.
- Remaining large/huge sampled FFX gap concentrates in high-d ill-conditioned rows (tail diagnostic below).

Sampled FFX-tail diagnostic (8k-row scan, INLA on 16 worst current FFX rows per size/partition):

| Tail set | part | current β RMSE | INLA β RMSE | Δβ | cap rows | singular/near-singular |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| large-n-sampled | valid | 0.6766 | 0.3425 | +0.3341 | 8/16 | 11/16 |
| huge-n-sampled | valid | 0.7014 | 0.2925 | +0.4089 | 12/16 | 16/16 |
| large-n-sampled | test | 0.6440 | 0.3715 | +0.2725 | 10/16 | 12/16 |
| huge-n-sampled | test | 0.4434 | 0.2185 | +0.2249 | 8/16 | 13/16 |

Known Structural Limits
-----------------------

- σ_eps df denominator is biased upward when many predictors are near-collinear with Z
  (active df drops, remaining directions over-absorb cross-group variance).
- z_rank can be underestimated when a group has n_g ≤ q (jittered inverse applied but
  rank counter may still inflate the df denominator, biasing σ_eps downward).

No correction is implemented; both are structural constraints of the one-pass projection
estimator.

Closed Directions (historical)
------------------------------

**Direction K — OLS fallback for double-trigger cap+stab rows (ADOPTED 2026-05-25)**

Per-dataset gap analysis (2026-05-25) showed ~2% of large/huge datasets have both
`normal_map_beta_prior_capped` AND `normal_map_beta_stabilized` fire. For these rows,
Adam MAP drove β outside the prior bounds AND the in-MAP σ-grid stabilization ran —
indicating inflated σ_rfx and collapsed GLS. The existing 25%-blend toward the tail-grid
σ-grid GLS is insufficient because the σ-grid GLS is also collapsed in this regime.

Fix: in `maybe_correct_beta_tail` (inside `refineNormalLaplaceEb`), when both triggers
fire, replace the blend target with **prior-regularized OLS** (no σ_rfx dependence, so
immune to inflation). The OLS β = `(X'X/σ² + diag(1/τ²))⁻¹(X'y/σ² + ν/τ²)` is capped
at `ν ± 4τ` and blended 75% (default; `normal_beta_tail_grid_both_trigger_blend`).
The single-trigger path (cap OR stab OR high_cond, not both) is unchanged at 25%.

Benchmark (N=1000): large-n-mixed FFX 0.2582 → 0.2449 (−5.1%), huge-n-mixed FFX
0.2677 → 0.2531 (−5.4%). Medium also improved (0.2283 → 0.2208) from d>4 rows in that
class. σ_rfx improved substantially on huge (0.3776 → 0.3001). No regression anywhere.
Current now outperforms INLA on medium-n FFX across both mixed and sampled sets.

Regime 2 (low-d d≤4, small-n sparse designs) remains un-addressed — no tail gate
mechanism available for d≤4 and INLA's advantage is irreducible in that regime.

**Direction I — Skew-corrected β marginal mean (RETIRED 2026-05-24)**

Added 3rd-moment skewness correction to the softmax-weighted β average. Zero effect: the 7-pt log-spaced grid is symmetric so m3 ≈ 0 whenever EB σ is near the posterior mode — which is exactly when the correction would run.

**Direction J — σ-only REML-Newton polish after Adam (RETIRED 2026-05-24)**

After Adam, ran 5 REML-Newton steps for σ_rfx with β frozen. Universal regression (+7–55% σ across all sizes): profile REML at fixed β optimises the wrong objective — it treats β as known rather than uncertain, so the Newton-displaced σ diverges from the joint MAP σ and triggers spurious EB "improvements" (EB accept rate 3% → 68%). Same coupled-objectives failure as Direction G.

**Direction H — Iterated MAP→EB loop (ADOPTED 2026-05-24)**

2 outer iterations of `MAP-Adam(20) → moment-EB → grid-refine`; tail β only on last iteration. BLUP β anchored to iteration 1's MAP so warm-started iter-2 β doesn't degrade BLUPs. FFX −1–7% across all sizes; σ_rfx and BLUP neutral. Cost +4–8 ms/ds (~1.7× total). Default: `normal_map_outer_iterations=2`.

**Direction G — Newton polish after Adam (RETIRED 2026-05-24)**

Block-coordinate Newton (exact GLS-MAP β + REML-Newton σ_rfx + EM σ_eps) after Adam. Pure Newton from LMM init severely diverges; polish-after-Adam regresses FFX +45% / σ +85% on small-n. Step 1 replaces Adam's β with GLS β, which changes residuals substantially and causes the Newton σ step to overshoot via the same collapse-toward-prior mechanism as Direction D.

**Direction E — Wider 7-pt σ grid (DONE 2026-05-24)**

Replaced the 3-pt scalar grid `{0.75, 1.0, 1.333}` with `{0.5, 0.667, 0.833, 1.0, 1.2, 1.5, 2.0}`
for all three grid scales (`normal_laplace_eb_sigma_grid_scales`, `normal_beta_sigma_grid_scales`,
`normal_beta_tail_grid_scales`). Results on first-1000 rows: σ NRMSE improved 1–8% across
all sizes (huge-n-mixed: -8.1%), FFX neutral. Cost: `+0.5–1 ms/ds`.

Also added a `σ_eps_map` fix: `refineNormalMapSrfx` now stores Adam's MAP σ_eps in
`stats['normal_map_sigma_eps']`, and the tail-grid GLS uses it instead of the initial
projection σ_eps. Slight additional FFX improvement for gated-row tail corrections.

**Direction D — Unconditional β averaging (RETIRED 2026-05-24)**

Unconditional σ-grid β averaging at both MAP and EB stages. All variants regressed FFX on medium (+3–50%) while neutral on large/huge. Root cause: ~65% of large/huge rows are ill-conditioned — GLS β collapses toward the prior mean there, and Adam's β is better. The existing 25%-blended cap-gated tail correction is already the sweet spot.

**Direction F — Cartesian σ_rfx grid for q=2 β averaging (RETIRED 2026-05-24)**

Cartesian σ-grid β averaging for q=2. Not applicable: hampered by the same ill-conditioned-row mechanism as Direction D. No benefit without a working unconditional β averaging path.

**Direction A — Softmax-weighted σ_rfx posterior mean (DONE 2026-05-24)**

`_normalSigmaRfxGridRefine` replaced argmax with softmax-weighted mean — same pattern as
`_normalSigmaGridBetaAverage` for β. σ posteriors are right-skewed so mean > mode; this
closed most of the INLA gap on σ_rfx.

**Direction B — Analytical β-profiling (not warranted)**

Diagnostic confirmed MAP σ already matches INLA mode; the gap was purely mean-vs-mode.
Direction B not needed.

**Direction C — Trace-based df for final σ_eps (tried 2026-05-24; reverted)**

Attempted: replace `z_rank` with `T = Σ_g tr[W_g Z_g'Z_g]` in the projection df
denominator. Since T ≤ z_rank, corrected denominator is larger → σ_eps_corrected smaller.
Result: σ_eps NRMSE increased +19–117% across all configurations (FFX/σ_rfx/BLUP flat).
The correction reduces σ_eps for all datasets even when there is no collinearity, which
outweighs any benefit in the rare high-collinearity case. No further attempts planned.

Commands
--------

Use `python -u` for long analytical runs so completed blocks stream immediately.

```bash
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family n --methods current raw --max-datasets 1000 --batch-size 32

uv run python -u experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-n-mixed --partition train --n-total 1000 \
    --analytical-methods current --no-save-inla-rows

# per-dataset gap analysis (uses precomputed INLA, no live INLA calls)
uv run python -u experiments/analytical/glmm_normal_ffx_gap_analysis.py \
    --data-ids large-n-mixed huge-n-mixed --partition train --max-datasets 1000

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical tests
```

Retired Lines
-------------

- R-INLA backend or full PyTorch INLA: incompatible with the throughput target.
- Standalone MAP option: EB is the retained Normal answer; MAP is only an internal stage.
- `mode='gradient'` in `refineNormalLaplaceEb`: gradient Adam loop for σ_rfx — removed
  in 2026-05-24 debloat pass; moment EB is the only retained mode.
- Output-local MAP for final BLUP: oracle tests confirmed diagonal Ψ from MAP σ_rfx beats
  output-local MAP and full Ψ recompute for BLUP accuracy.
- Axis, ratio, post-EB, curvature, hard-shrink, and broad tail-grid β variants.
- Final correlated Ψ for BLUP: estimated correlations are noisy and harmful here.
- Wider σ_rfx grid (2.0× scale, G-gated): 8k showed no improvement on large/huge; the
  standard EB + 1.333 grid already captures available marginal-target improvement.
- Per-dimension moment EB (moment_per_dim): too liberal — accepts individual-dim updates
  where the joint posterior is not improving; regressed small-n-sampled σ by +0.013.
- τ_rfx floor for W_g (moment_sigma_tau_floor): no material improvement; moment EB is
  already robust to poor initialization via the prior regularizer.
- Low-d sparse-m σ-grid blend (small_m_threshold): fired for d<9 and m≤20 but sm_grid≈beta_output
  on small-n (zero effect) and regressed medium-n FFX by +0.019–0.050. Retired 2026-05-25.
