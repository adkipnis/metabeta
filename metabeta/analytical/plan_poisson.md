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

**Poisson EB** is now the default Poisson analytical path in `glmm()`. It starts from
RAW/PQL, then applies a diagonal Laplace-EB refinement with a Poisson log-link target:

- Adam over β and diagonal log σ_rfx, with nested per-group random-effect mode solves;
- fixed-effect and σ_rfx priors in the Laplace target;
- per-row accept gate against incoming RAW/PQL;
- accepted-row BLUP fallback to RAW/PQL, because Poisson EB modes currently regress BLUP on
  most benchmark cells;
- accepted-row-only σ cap at `2.5 * tau_rfx` for effective `d >= 5`.

The old RAW/PQL path remains available with `map_refine=False` or
`poisson_laplace_eb=False`. The explicit EB preset remains available for benchmark
equivalence checks:

```python
glmm(
    ...,
    poisson_laplace_eb='poisson_eb',
)
```

Performance Snapshot
--------------------

Matched first-1000 per comparison row. CPU ms/dataset refers to the analytical (RAW) method.
Lower NRMSE is better. Large rows are refreshed; huge rows are running as of 2026-05-19.

| Dataset          | part  | RAW FFX   | INLA FFX  | RAW σ     | INLA σ    | RAW BLUP  | INLA BLUP | INLA s/ds |
| ---              | ---   | ---:      | ---:      | ---:      | ---:      | ---:      | ---:      | ---:      |
| small-p-mixed    | train | 0.4806    | **0.2261** | 0.7185   | **0.4242** | 0.5713   | **0.521** | 2.932     |
| small-p-sampled  | valid | 0.7351    | **0.2589** | 0.7369   | **0.4597** | 0.5823   | **0.5364** | 3.042    |
| small-p-sampled  | test  | 0.6525    | **0.2369** | 0.7524   | **0.4375** | 0.6708   | **0.5326** | 3.058    |
| medium-p-mixed   | train | 0.5185    | **0.1901** | 0.9753   | **0.3706** | 0.6445   | **0.5113** | 3.429     |
| medium-p-sampled | valid | 0.822     | **0.2394** | 1.2083   | **0.4556** | 0.7509   | **0.5568** | 3.409     |
| medium-p-sampled | test  | 0.6852    | **0.2446** | 0.9222   | **0.3592** | 0.6261   | **0.5491** | 3.406     |
| large-p-mixed    | train | 0.869     | **0.1974** | 1.1677   | **0.3416** | 0.9667   | **0.5316** | 4.705     |
| large-p-sampled  | valid | 1.0474    | **0.2716** | 1.5415   | **0.4563** | 0.7538   | **0.6062** | 4.754     |
| large-p-sampled  | test  | 0.893     | **0.234**  | 1.3776   | **0.3713** | 0.9427   | **0.5447** | 4.743     |
| huge-p-mixed     | train | —         | —         | —         | —         | —         | —         | —         |
| huge-p-sampled   | valid | —         | —         | —         | —         | —         | —         | —         |
| huge-p-sampled   | test  | —         | —         | —         | —         | —         | —         | —         |

INLA logs are under `experiments/analytical/inla_runs/poisson_sampled/` and
`experiments/analytical/inla_runs/poisson_mixed/`. All runs use data regenerated on
2026-05-19 with the Poisson LP scale calibration (`_calibratePoissonEtaScale`, caps [1.0, 2.0]).

Poisson EB Default Snapshot
---------------------------

First 1000 rows for mixed train (`n_epochs=2`) plus sampled valid/test. Lower NRMSE is
better. `current` and explicit `poisson_eb` were verified to match.

| Dataset | part | RAW FFX | EB FFX | RAW σ | EB σ | RAW BLUP | EB BLUP | EB ms/ds | accept | σ cap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.4806 | **0.4269** | 0.7185 | **0.5272** | 0.5713 | 0.5713 | 29.13 | 0.879 | 0.000 |
| small-p-sampled | valid | 0.7351 | **0.6375** | 0.7369 | **0.5645** | 0.5823 | 0.5823 | 24.05 | 0.897 | 0.000 |
| small-p-sampled | test | 0.6525 | **0.5429** | 0.7524 | **0.6323** | 0.6708 | 0.6708 | 24.34 | 0.894 | 0.000 |
| medium-p-mixed | train | 0.5185 | **0.4525** | 0.9753 | **0.5695** | 0.6445 | 0.6445 | 43.48 | 0.813 | 0.069 |
| medium-p-sampled | valid | 0.8220 | **0.7291** | 1.2083 | **0.5646** | 0.7509 | 0.7509 | 44.61 | 0.896 | 0.074 |
| medium-p-sampled | test | 0.6852 | **0.5963** | 0.9222 | **0.5979** | 0.6261 | 0.6261 | 41.81 | 0.837 | 0.083 |
| large-p-mixed | train | 0.8690 | **0.7710** | 1.1677 | **0.6788** | 0.9667 | 0.9667 | 56.33 | 0.756 | 0.105 |
| large-p-sampled | valid | 1.0474 | **0.9243** | 1.5415 | **0.7873** | 0.7538 | 0.7538 | 53.25 | 0.783 | 0.100 |
| large-p-sampled | test | 0.8930 | **0.7968** | 1.3776 | **0.6296** | 0.9427 | 0.9427 | 58.99 | 0.790 | 0.112 |
| huge-p-mixed | train | 0.9756 | **0.9225** | 1.3812 | **0.7202** | 1.3458 | 1.3458 | 67.33 | 0.750 | 0.139 |
| huge-p-sampled | valid | 1.4947 | **1.3965** | 2.1001 | **1.1463** | 1.8398 | 1.8398 | 86.01 | 0.748 | 0.150 |
| huge-p-sampled | test | 1.4947 | **1.4128** | 1.7253 | **0.8989** | 1.2842 | 1.2842 | 79.12 | 0.786 | 0.174 |

Incremental findings:

- The first direct Poisson EB module improved FFX/σ but regressed BLUP on small mixed and
  valid rows. The accepted-row BLUP fallback fixes this without giving up FFX/σ gains.
- The `24` step, `lr=0.05` preset was clearly better than the initial `12` step,
  `lr=0.03` prototype on all small FFX/σ cells, with runtime still below the 100 ms/dataset
  target.
- A relaxed BLUP fallback threshold (`β jump >= 1.0`) was rejected: it improved only
  medium sampled valid BLUP and regressed four of the other five small/medium cells.
- The `2.5 * tau_rfx` σ cap improves all medium+ σ rows while leaving small rows unchanged.
  A tighter `2.0` cap fired more often and was slightly worse on all medium σ rows.
- Large and huge first-1000 rows have the same pattern as small/medium: FFX and σ improve,
  BLUP is neutral, and runtime remains below the `~100 ms/dataset` target.

INLA-direction diagnostic:

- Script:
  `experiments/analytical/glmm_poisson_inla_direction_diagnostic.py`.
- Saved row diagnostics:
  `experiments/analytical/inla_runs/poisson_direction/mixed_train_20.csv` and
  `experiments/analytical/inla_runs/poisson_direction/sampled_valid_20.csv`.
- On 20 small + 20 medium mixed train rows, current EB improved mean FFX RMSE from
  `0.4307` to `0.3736`, but INLA was `0.1461`. Medium rows moved strongly toward INLA
  (median β distance gain `0.6729`, mean shift cosine `0.9213`); small rows were mostly
  neutral and had unstable distance-gain means because some RAW rows were already close to
  INLA.
- On 20 small + 20 medium sampled-valid rows, current EB improved mean FFX RMSE from
  `0.3634` to `0.2875`, while INLA was `0.1618`. Medium sampled rows again moved toward
  INLA (median β distance gain `0.8137`, mean shift cosine `0.9497`). Small sampled rows
  were the weak spot: median β distance gain `-1.8049` and mean shift cosine `-0.4000`.
- The largest FFX gaps are not solved by σ cap tuning alone. Typical outliers have RAW and
  EB β RMSE in the `2–5` range while INLA is around `0.1–0.5`; EB usually moves in the INLA
  direction but not far enough. Some low-d (`d=1`) sampled rows move away from INLA despite
  small absolute errors, so a broad stronger β update is likely unsafe.

Takeaways
---------

From first-1000 rows across all sizes:

- RAW has a large FFX gap vs INLA at small scale (~2.5–3× worse NRMSE). This mirrors the
  Bernoulli pattern before Bernoulli EB: PQL/IRLS fixed effects are biased relative to
  full INLA.
- RAW σ_rfx is similarly ~1.7× worse than INLA. BLUP is closer (~1.1× worse), consistent
  with BLUPs being mainly driven by the within-group signal once σ is roughly in range.
- Poisson EB consistently improves FFX and σ over RAW/PQL on all first-1000 mixed and
  sampled rows. The biggest gain is σ on medium+ rows. Row-level INLA-direction diagnostics
  show that medium rows usually move toward INLA, but low-d small/sampled rows need a guard.
- BLUP is deliberately neutral at this stage. Direct EB BLUP modes regressed most cells, so
  the retained default feeds RAW/PQL BLUPs through while using EB β and σ.
- The current Bernoulli-style EB transplant is not enough for Poisson FFX. The likely
  missing piece is Poisson log-link marginalization: random-effect variance changes the
  marginal mean via an `exp(0.5 * zΨz)` factor, so β needs a correction that directly
  targets marginal mean calibration rather than only the conditional mode target. This
  should be gated; low-d sampled rows can move away from INLA under the current EB shift.
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
  retained EB path, but Poisson now needs a Poisson-specific β correction before adding
  more Bernoulli-derived optimizer branches.
- **Separation-specific β output cap.** Poisson has no Bernoulli separation analogue. Keep
  the existing Poisson η/beta clamps for numerical stability; add a Poisson-specific cap
  only if diagnostics show high-count tails causing β explosions.
- **Broad sigma grids, multi-starts, EP, or full PyTorch INLA.** These add moving parts and
  target the expensive part of INLA. Consider only a tiny targeted quadrature/grid if the
  INLA diagnostic shows the remaining gap is concentrated in low-q or high-σ rows.
- **BLUP variance inflation changes.** First improve point estimates. Coverage calibration
  can follow if `glmm_error_analysis.py` shows Poisson interval undercoverage.

Next Steps
----------

1. **Prototype a gated Poisson marginal-mean β correction.** Use the current Ψ estimate to add a
   cheap marginalization correction,
   `η_marg ≈ Xβ + 0.5 * diag(ZΨZ')`, then re-fit or adjust β so the Poisson marginal mean is
   calibrated. Gate it first to medium/high-severity rows, because the INLA-direction
   diagnostic shows medium rows move toward INLA while low-d sampled rows can move away.
   Candidate gates: `d >= 5`, large prior-standardized β magnitude, large β jump, or a
   count-tail/zero-fraction severity score.

2. **Try a variance-aware β refinement before touching BLUPs.** Build a batched β update
   using Poisson weights/offsets from the current random-effect variance and fitted μ. Keep
   RAW/PQL BLUP fallback unless this β-focused path first improves FFX/σ on the quick gate.

3. **Use tiny targeted quadrature/grid only if marginal β correction is insufficient.** If
   the remaining INLA gap is concentrated in `q <= 2`, high-σ, or high-count-tail rows, test
   a narrow adaptive correction for those rows. Skip broad grids, multi-starts, and full
   PyTorch INLA unless a targeted diagnostic shows that the simpler marginal correction
   cannot move the gap.

4. **Postpone the full 8k analytical benchmark.** Run it only after a more serious
   Poisson-specific improvement beats the current EB path on the first-1000 quick gate.

5. **Complete the INLA comparison baseline in parallel.** Large rows are now refreshed for
   first-1000 comparisons and huge rows are running. Integrate huge values once the logs
   finish, but do not block the Poisson-specific prototypes on the full INLA table.

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

# Postpone the full 8k run until a Poisson-specific patch beats the current EB default on
# the first-1000 quick gate.

uv run python -u experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-p-sampled --partition valid \
    --n-inla 1000 --n-total 1000 --analytical-methods raw,current
# Repeat the INLA command for each sampled size and valid/test partition.
# For mixed/train: add --partition train --n-epochs 2 and use -p-mixed data-ids.

uv run python -u experiments/analytical/glmm_poisson_inla_direction_diagnostic.py \
    --data-ids small-p-mixed medium-p-mixed --partition train \
    --n-inla 20 --n-epochs 1 --batch-size 16 \
    --output-csv experiments/analytical/inla_runs/poisson_direction/mixed_train_20.csv

uv run python -u experiments/analytical/glmm_poisson_inla_direction_diagnostic.py \
    --data-ids small-p-sampled medium-p-sampled --partition valid \
    --n-inla 20 --batch-size 16 \
    --output-csv experiments/analytical/inla_runs/poisson_direction/sampled_valid_20.csv

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
