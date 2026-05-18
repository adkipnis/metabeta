Normal GLMM Plan
================

Last updated: 2026-05-17

Goal
----

Fast, prior-aware analytical summaries for Gaussian GLMMs. R-INLA is a slow reference,
not a backend. The production path must stay batched and in the low-millisecond range.

Current Path
------------

`glmm()` uses:

- raw Gaussian LMM initialization;
- marginal MAP refinement of σ_rfx, β, and σ_eps;
- diagonal final Ψ for GLS/BLUP recompute;
- d-gated MAP β carry-forward for `d > 4`;
- prior-capped reported β for `d > 4`: `β_report = clamp(β_MAP, ν_ffx ± 4τ_ffx)`;
- default posterior-moment diagonal σ_rfx EB calibration.

The d gate is intentional. A direct MAP β carry-forward improves medium/large/huge rows
but badly overfits small `d <= 4` rows, so small keeps the older GLS/OLS final β path.
The prior cap is output-only for β: final BLUPs continue to use the uncapped MAP β, which
preserves the strong BLUP performance while removing rare FFX tail explosions.

Current Performance
-------------------

First 1000 datasets per row. Lower NRMSE is better.

| Dataset | part | MAP FFX | EB FFX | MAP σ | EB σ | MAP σ_eps | EB σ_eps | MAP BLUP | EB BLUP | MAP ms | EB ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.1096 | 0.1095 | 0.4814 | 0.4203 | 0.2151 | 0.2151 | 0.4192 | 0.4173 | 2.95 | 2.52 |
| small-n-sampled | valid | 0.2587 | 0.2588 | 0.5654 | 0.5646 | 0.2169 | 0.2169 | 0.5130 | 0.5125 | 2.91 | 2.50 |
| small-n-sampled | test | 0.2828 | 0.2827 | 0.4864 | 0.4684 | 0.2169 | 0.2169 | 0.4926 | 0.4924 | 2.22 | 2.54 |
| medium-n-mixed | train | 0.2515 | 0.2515 | 0.3798 | 0.3619 | 0.1655 | 0.1655 | 0.4212 | 0.4198 | 2.73 | 3.05 |
| medium-n-sampled | valid | 0.2766 | 0.2766 | 0.5709 | 0.4131 | 0.1891 | 0.1891 | 0.5172 | 0.5151 | 3.32 | 3.71 |
| medium-n-sampled | test | 0.2623 | 0.2623 | 0.4505 | 0.3964 | 0.1949 | 0.1949 | 0.4451 | 0.4417 | 3.28 | 3.63 |
| large-n-mixed | train | 0.4075 | 0.4075 | 0.4148 | 0.3711 | 0.1268 | 0.1268 | 0.4155 | 0.4148 | 3.71 | 4.23 |
| large-n-sampled | valid | 0.3009 | 0.3009 | 0.4982 | 0.4316 | 0.1563 | 0.1563 | 0.5135 | 0.5069 | 3.86 | 4.39 |
| large-n-sampled | test | 0.3579 | 0.3579 | 0.8721 | 0.4415 | 0.1513 | 0.1513 | 0.5171 | 0.5126 | 4.08 | 4.66 |
| huge-n-mixed | train | 0.3314 | 0.3314 | 0.4280 | 0.3776 | 0.1161 | 0.1161 | 0.4573 | 0.4545 | 4.74 | 5.34 |
| huge-n-sampled | valid | 0.4485 | 0.4485 | 0.4100 | 0.3694 | 0.1375 | 0.1375 | 0.4595 | 0.4574 | 5.63 | 6.43 |
| huge-n-sampled | test | 0.3398 | 0.3398 | 0.4092 | 0.3870 | 0.1438 | 0.1438 | 0.4630 | 0.4619 | 5.84 | 6.63 |

Relative to the previous normal prototype, carrying and prior-capping MAP β for `d > 4`
substantially reduces FFX NRMSE on every medium/large/huge row. EB then improves
σ_rfx and BLUP on every row, while leaving σ_eps unchanged. The β cap is sparse: it fires
on about `1.1%` of medium mixed rows, `4.4%` of large mixed rows, `4.0%` of huge mixed
rows, and `1.7-5.9%` of medium+ sampled rows.

R-INLA Reference
----------------

The mixed/train 1k R-INLA rerun completed on the same first-1000 rows. INLA uses diagonal
random effects because the exact correlated Gaussian R-INLA branch is unstable on these
datasets.

| Dataset | EB FFX | INLA FFX | EB σ | INLA σ | EB BLUP | INLA BLUP | EB ms | INLA s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | 0.1095 | 0.0985 | 0.4203 | 0.3665 | 0.4173 | 0.4081 | 2.42 | 2.406 |
| medium-n-mixed | 0.2515 | 0.2301 | 0.3619 | 0.3419 | 0.4198 | 0.4289 | 3.05 | 2.604 |
| large-n-mixed | 0.4075 | 0.2377 | 0.3711 | 0.3393 | 0.4148 | 0.4185 | 4.23 | 2.786 |
| huge-n-mixed | 0.3314 | 0.2413 | 0.3776 | 0.2808 | 0.4545 | 0.4548 | 5.34 | 2.965 |

Small diagnostic run:

```bash
uv run python experiments/analytical/glmm_normal_inla_diagnostic.py --n-inla 8 --batch-size 8
```

Findings before the prior cap: after the β carry-forward patch, the mean per-dataset FFX
gap was small on the first 8 mixed datasets per size. Remaining INLA advantages were mostly
BLUP on small `d<=4` and tail cases; medium σ_rfx still favored INLA even after `normal_eb`.

Patched MAP FFX tail diagnostic, first 1000 mixed/train datasets:

| Dataset | aggregate FFX NRMSE | mean dataset RMSE | p95 RMSE | p99 RMSE | max RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| medium-n-mixed | 0.4953 | 0.0656 | 0.1351 | 0.2667 | 3.3842 |
| large-n-mixed | 1.6137 | 0.0865 | 0.1338 | 0.5532 | 10.0530 |
| huge-n-mixed | 0.9507 | 0.0644 | 0.1132 | 0.5213 | 5.6540 |

Interpretation: the remaining Gaussian FFX problem is not a broad approximation bias like
the old Bernoulli PQL/IRLS failure. Most datasets have modest β RMSE; aggregate NRMSE is
dominated by a small number of ill-conditioned FE/RE confounding cases. INLA likely wins
there by posterior averaging and stronger hyperparameter uncertainty propagation, while
our analytical path is still a plug-in MAP plus final GLS/BLUP recompute.

Tail-targeted INLA diagnostic, first 1000 mixed/train rows per size:

```bash
uv run python experiments/analytical/glmm_normal_inla_diagnostic.py \
    --tail-scan 1000 --tail-k 12 --batch-size 32 --top-k 16 \
    --output-csv /private/tmp/normal_inla_tail_1k_top12.csv
```

This scans 4000 rows analytically and runs INLA only on the 12 selected tail rows per
dataset. Selection is cap-prioritized FFX error (`cap_ffx_eb_rmse`).

| Tail subset | N | EB FFX | INLA FFX | EB σ | INLA σ | EB BLUP | INLA BLUP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | 12 | 0.7367 | 0.6962 | 0.2593 | 0.1929 | 0.5497 | 0.5180 |
| medium-n-mixed | 12 | 0.2645 | 0.0931 | 0.1143 | 0.1278 | 0.1777 | 0.1791 |
| large-n-mixed | 12 | 0.6927 | 0.0950 | 0.0531 | 0.0546 | 0.1045 | 0.1066 |
| huge-n-mixed | 12 | 0.4109 | 0.1063 | 0.1131 | 0.0877 | 0.1808 | 0.1755 |

Tail interpretation:

- Cap-hit rows are the actionable failure mode. Across the selected tail rows, cap-hit
  rows have EB FFX `0.4559` versus INLA `0.0984`; INLA is better on `34/35` cap-hit
  rows. Non-cap rows are mostly small low-d rows where INLA is not consistently better.
- Large/huge FFX failures are not caused by BLUP reconstruction. On large tail rows BLUP
  is essentially tied (`0.1045` EB versus `0.1066` INLA), while FFX differs by `0.5977`.
- The worst large rows are high-d, usually q=1, and have residualized-X condition numbers
  in the hundreds to thousands. Examples: large idx `341` has EB FFX `1.4337` versus
  INLA `0.1283` with `rx_cond=1227`; large idx `344` has EB FFX `0.9448` versus
  INLA `0.1319` with `rx_cond=1235`.
- FE/RE explainability alone is not a sufficient trigger. Some tail rows have low measured
  max slope RE R² but still enormous residualized-X condition. The robust trigger should
  combine prior-cap hit and residualized-X conditioning, not only FE/RE R².

Likely implementation cause:

The current normal path optimizes a plug-in marginal MAP β and then hard-clips reported β
to `ν_ffx ± 4τ_ffx`. In ill-conditioned high-d rows, MAP can move far along weakly
identified fixed-effect directions. The output cap prevents explosions but still reports a
boundary value, whereas INLA reports a posterior mean that shrinks continuously through
the local curvature and hyperparameter uncertainty. Because BLUP uses the uncapped MAP β,
BLUP accuracy remains good; the failure is mostly the reported β summary.

Next Steps
----------

1. **Keep the d-gated MAP β carry-forward plus prior reporting cap internally.** It is simple,
   sparse, and directly fixes the observed β tail failures.
2. **Use EB as the retained normal answer.** It is a one-shot posterior-moment update,
   not another optimizer, and improves σ/BLUP broadly.
3. **Prototype a conditional β posterior-mean stabilizer for reporting only.** Trigger on
   `normal_map_beta_prior_capped` or large residualized-X condition for `d > 4`. Replace
   hard boundary clipping with a smooth shrinkage toward `ν_ffx` using a cheap diagonal
   or low-rank approximation to the local β curvature. BLUP should continue to use the
   uncapped MAP β unless a benchmark shows otherwise.
4. **Benchmark the stabilizer against the first-1000 required normal rows before any INLA
   rerun.** The target is to reduce large/huge FFX without moving σ/BLUP backward.
5. **Next sampled-set R-INLA only if needed.** The mixed/train R-INLA rerun and tail
   diagnostic are enough to guide the next patch; sampled R-INLA is expensive and should
   be used only if analytical sampled rows regress.
6. **Avoid broad new machinery.** Do not add multi-starts, EP, or full posterior
   integration unless a new benchmark shows a systematic, not tail-only, Gaussian FFX gap.

Commands
--------

```bash
uv run python experiments/analytical/glmm_required_benchmark.py \
    --family n --methods current raw --max-datasets 1000 --batch-size 32

uv run python experiments/analytical/glmm_normal_inla_diagnostic.py --n-inla 8 --batch-size 8

uv run python experiments/analytical/glmm_normal_inla_diagnostic.py \
    --tail-scan 1000 --tail-k 12 --batch-size 32 --top-k 16 \
    --output-csv /private/tmp/normal_inla_tail_1k_top12.csv

uv run python experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-n-mixed,medium-n-mixed,large-n-mixed,huge-n-mixed \
    --partition train --n-epochs 2 --n-inla 1000 --n-total 1000 \
    --analytical-methods raw,current --normal-re-correlation diagonal

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical tests
```

Retired Lines
-------------

- REML variants: worse or unstable on required normal rows.
- Final correlated Ψ for BLUP: estimated correlations are noisy and harmful.
- Prior-mean β shrinkage as a fix for INLA FFX: prior means are around `0.97-0.99`
  FFX NRMSE on mixed rows and do not explain INLA's advantage.
- Additional long optimizers: not compatible with the speed target unless they replace,
  rather than extend, current MAP.
