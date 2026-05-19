Poisson GLMM Plan
=================

Last updated: 2026-05-19

Goal
----

Provide a fast, high-accuracy analytical Poisson GLMM estimator. The estimator should be
stable, prior-aware, and cheap; it is not intended to be a full posterior engine.

Do not pursue R-INLA as a backend or full PyTorch INLA as the main path. Use INLA concepts
only where they stay compatible with batched CPU/GPU processing and the `~100 ms/dataset`
target.

Default Path
------------

**RAW / POISSON-PQL** is the current Poisson analytical path in `glmm()`. It uses PQL/IRLS
with Laplace-linearised working responses.

A default-off **Poisson EB** prototype now exists behind
`glmm(..., poisson_laplace_eb='poisson_eb')` and the benchmark method `poisson_eb`.
It reuses the Bernoulli diagonal Laplace-EB structure with a Poisson log-link target:

- Adam over β and diagonal log σ_rfx, with nested per-group random-effect mode solves;
- fixed-effect and σ_rfx priors in the Laplace target;
- per-row accept gate against incoming RAW/PQL;
- accepted-row BLUP fallback to RAW/PQL, because Poisson EB modes currently regress BLUP on
  most small/medium cells;
- accepted-row-only σ cap at `2.5 * tau_rfx` for effective `d >= 5`.

Do not make this the default until it clears the full 8k mixed + sampled benchmark.

Performance Snapshot
--------------------

Matched first-1000 per comparison row. CPU ms/dataset refers to the analytical (RAW) method.
Lower NRMSE is better. Large/huge rows pending (INLA runs in progress as of 2026-05-19).

| Dataset          | part  | RAW FFX   | INLA FFX  | RAW σ     | INLA σ    | RAW BLUP  | INLA BLUP | INLA s/ds |
| ---              | ---   | ---:      | ---:      | ---:      | ---:      | ---:      | ---:      | ---:      |
| small-p-mixed    | train | 0.4806    | **0.2261** | 0.7185   | **0.4242** | 0.5713   | **0.521** | 2.932     |
| small-p-sampled  | valid | 0.7351    | **0.2589** | 0.7369   | **0.4597** | 0.5823   | **0.5364** | 3.042    |
| small-p-sampled  | test  | 0.6525    | **0.2369** | 0.7524   | **0.4375** | 0.6708   | **0.5326** | 3.058    |
| medium-p-mixed   | train | 0.5185    | **0.1901** | 0.9753   | **0.3706** | 0.6445   | **0.5113** | 3.429     |
| medium-p-sampled | valid | 0.822     | **0.2394** | 1.2083   | **0.4556** | 0.7509   | **0.5568** | 3.409     |
| medium-p-sampled | test  | 0.6852    | **0.2446** | 0.9222   | **0.3592** | 0.6261   | **0.5491** | 3.406     |
| large-p-mixed    | train | —         | —         | —         | —         | —         | —         | —         |
| large-p-sampled  | valid | —         | —         | —         | —         | —         | —         | —         |
| large-p-sampled  | test  | —         | —         | —         | —         | —         | —         | —         |
| huge-p-mixed     | train | —         | —         | —         | —         | —         | —         | —         |
| huge-p-sampled   | valid | —         | —         | —         | —         | —         | —         | —         |
| huge-p-sampled   | test  | —         | —         | —         | —         | —         | —         | —         |

INLA logs are under `experiments/analytical/inla_runs/poisson_sampled/` and
`experiments/analytical/inla_runs/poisson_mixed/`. All runs use data regenerated on
2026-05-19 with the Poisson LP scale calibration (`_calibratePoissonEtaScale`, caps [1.0, 2.0]).

Poisson EB Prototype Snapshot
-----------------------------

Expanded quick gate: first 1000 rows for small and medium mixed train (`n_epochs=2`) plus
sampled valid/test. Lower NRMSE is better.

| Dataset | part | RAW FFX | EB FFX | RAW σ | EB σ | RAW BLUP | EB BLUP | EB ms/ds | accept | σ cap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.4806 | **0.4269** | 0.7185 | **0.5272** | 0.5713 | 0.5713 | 29.13 | 0.879 | 0.000 |
| small-p-sampled | valid | 0.7351 | **0.6375** | 0.7369 | **0.5645** | 0.5823 | 0.5823 | 24.05 | 0.897 | 0.000 |
| small-p-sampled | test | 0.6525 | **0.5429** | 0.7524 | **0.6323** | 0.6708 | 0.6708 | 24.34 | 0.894 | 0.000 |
| medium-p-mixed | train | 0.5185 | **0.4525** | 0.9753 | **0.5695** | 0.6445 | 0.6445 | 43.48 | 0.813 | 0.069 |
| medium-p-sampled | valid | 0.8220 | **0.7291** | 1.2083 | **0.5646** | 0.7509 | 0.7509 | 44.61 | 0.896 | 0.074 |
| medium-p-sampled | test | 0.6852 | **0.5963** | 0.9222 | **0.5979** | 0.6261 | 0.6261 | 41.81 | 0.837 | 0.083 |

Incremental findings:

- The first direct Poisson EB module improved FFX/σ but regressed BLUP on small mixed and
  valid rows. The accepted-row BLUP fallback fixes this without giving up FFX/σ gains.
- The `24` step, `lr=0.05` preset was clearly better than the initial `12` step,
  `lr=0.03` prototype on all small FFX/σ cells, with runtime still below the 100 ms/dataset
  target.
- A relaxed BLUP fallback threshold (`β jump >= 1.0`) was rejected: it improved only
  medium sampled valid BLUP and regressed four of the other five small/medium cells.
- The `2.5 * tau_rfx` σ cap improves all medium σ rows while leaving small rows unchanged.
  A tighter `2.0` cap fired more often and was slightly worse on all medium σ rows.

Takeaways
---------

From small and medium rows; large/huge pending.

- RAW has a large FFX gap vs INLA at small scale (~2.5–3× worse NRMSE). This mirrors the
  Bernoulli pattern before Bernoulli EB: PQL/IRLS fixed effects are biased relative to
  full INLA.
- RAW σ_rfx is similarly ~1.7× worse than INLA. BLUP is closer (~1.1× worse), consistent
  with BLUPs being mainly driven by the within-group signal once σ is roughly in range.
- No separation analogue for Poisson, but heavy-tailed count overdispersion may introduce
  analogous instabilities at higher d.
- The INLA ms/ds (~3 s) vs RAW (~4 ms) ratio is ~750× — consistent with the Bernoulli gap.

Bernoulli Lessons for Poisson
-----------------------------

Reuse these pieces:

- **Diagonal Laplace EB over β and log σ.** This is the main Bernoulli win and maps cleanly
  to Poisson by replacing the Bernoulli logit likelihood with the stabilized Poisson
  log-link likelihood.
- **Nested per-group mode refresh.** Re-solving random-effect modes inside the outer EB
  loop is the part that makes the β target close to INLA-like behavior without full INLA
  hyperparameter integration.
- **Prior-aware target.** Keep fixed-effect priors, σ_rfx priors, and the log-σ Jacobian in
  the EB target from the start. A likelihood-only Poisson refinement is likely to repeat the
  failed optimizer-free Normal experiment pattern: faster but biased in σ.
- **Acceptance gate against RAW/PQL.** Bernoulli EB is safe because every dataset can fall
  back to the incoming PQL estimate if the Laplace target worsens. Poisson should have the
  same per-row target gate before any default promotion.
- **Diagnostics and full-suite promotion gate.** Track accept rate, β jump, σ cap/fallback
  rows, and BLUP fallback rows. Promote only after small/medium/large/huge mixed train plus
  sampled valid/test all improve or stay neutral.

Skip these initially:

- **Bernoulli nAGQ and nested-β pre-refiners.** They were useful stepping stones before the
  retained EB path, but adding them to Poisson would create extra branches before we know
  whether direct EB closes the RAW gap.
- **Separation-specific β output cap.** Poisson has no Bernoulli separation analogue. Keep
  the existing Poisson η/beta clamps for numerical stability; add a Poisson-specific cap
  only if diagnostics show high-count tails causing β explosions.
- **Broad sigma grids, multi-starts, EP, or full PyTorch INLA.** These add moving parts and
  target the expensive part of INLA. Start with one diagonal Laplace mode and one optimizer.
- **BLUP variance inflation changes.** First improve point estimates. Coverage calibration
  can follow if `glmm_error_analysis.py` shows Poisson interval undercoverage.

Next Steps
----------

1. **Keep `poisson_eb` default-off until the full benchmark clears.** The prototype is
   promising on small/medium, but promotion requires large/huge first-1000 and then 8k
   mixed + sampled results.

2. **Run the expanded quick gate for any further patch.** Use small + medium mixed train and
   sampled valid/test, first 1000 rows each. A patch must improve FFX/σ without BLUP
   regressions before moving to large/huge.

3. **Next candidate: diagnose whether EB BLUPs can be selectively trusted.** The safe
   default is RAW/PQL BLUP fallback, but medium sampled valid suggests some rows benefit from
   EB modes. Only keep a selector if it improves the expanded gate without mixed-row BLUP
   regressions.

4. **Then run all sizes first-1000, followed by the 8k analytical benchmark.** If large/huge
   follow the medium pattern, consider promoting `poisson_eb` to the default Poisson path.

5. **Complete the INLA comparison baseline in parallel.** Fill in medium/large/huge rows as
   logs finish. Do not wait for the full INLA table before prototyping EB, because the small
   rows already show a large and actionable RAW gap.

Commands
--------

```bash
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods raw current poisson_eb \
    --combos small-p-mixed:train:2 small-p-sampled:valid small-p-sampled:test \
        medium-p-mixed:train:2 medium-p-sampled:valid medium-p-sampled:test \
        large-p-mixed:train:2 large-p-sampled:valid large-p-sampled:test \
        huge-p-mixed:train:2 huge-p-sampled:valid huge-p-sampled:test \
    --batch-size 32 --max-datasets 1000

uv run python -u experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-p-sampled --partition valid \
    --n-inla 1000 --n-total 1000 --analytical-methods raw,current
# Repeat the INLA command for each sampled size and valid/test partition.
# For mixed/train: add --partition train --n-epochs 2 and use -p-mixed data-ids.

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/fit.py metabeta/analytical/glmm \
    experiments/analytical/glmm_inla_comparison.py tests/utils/test_glmm.py
```

Retired Lines
-------------

- R-INLA backend and full PyTorch INLA: incompatible with the throughput target.
- Bernoulli nAGQ/nested-β staging for Poisson: defer unless direct Poisson EB fails in a
  clearly diagnosed way.
- Unconditional Bernoulli-style β output cap: not justified for Poisson without a
  high-count tail diagnostic.
- Relaxed BLUP fallback at `β jump >= 1.0`: regresses most small/medium BLUP rows.
- Poisson σ cap `2.0 * tau_rfx`: slightly worse than `2.5` on all medium σ rows.
