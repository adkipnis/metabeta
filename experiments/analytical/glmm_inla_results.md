R-INLA vs Analytical GLMM Comparison Results
=============================================

`glmm_inla_comparison.py` — full analytical pipeline (`glmm()`, `map_refine=True`) vs
R-INLA. For Bernoulli, the analytical column is the current default Bernoulli EB path. Results
are reported on the matched subset used for INLA where available. mixed=train/ep2,
sampled=test.

NRMSE Summary
-------------

Bold = better method per column. The mixed rows were rerun after Bernoulli EB became the
default Bernoulli path. The sampled rows combine the current Bernoulli EB benchmark with the
existing R-INLA reference run.

| Dataset           | part  | EB FFX    | INLA FFX  | EB σ      | INLA σ    | EB BLUP   | INLA BLUP | INLA s/ds |
| ---               | ---   | ---:      | ---:      | ---:      | ---:      | ---:      | ---:      | ---:      |
| small-b-mixed     | train | **0.267** | 0.451     | **0.510** | 0.567     | **0.614** | 0.618     | n/a       |
| small-b-sampled   | test  | **0.293** | 0.447     | **0.504** | 0.556     | **0.609** | 0.625     | 2.129     |
| medium-b-mixed    | train | **0.313** | 0.332     | 0.539     | **0.522** | 0.686     | **0.648** | 2.566     |
| medium-b-sampled  | test  | **0.339** | 0.400     | **0.584** | 4.490 †   | 0.707     | **0.692** | 4.306     |
| large-b-mixed     | train | 0.332     | **0.323** | 0.542     | **0.521** | 0.685     | **0.676** | 3.273     |
| large-b-sampled   | test  | **0.357** | 0.365     | 0.620     | **0.603** | 0.727     | **0.710** | 4.682     |
| huge-b-mixed      | train | 0.335     | **0.330** | 0.600     | **0.550** | 0.737     | **0.713** | 4.605     |
| huge-b-sampled    | test  | **0.378** | 0.394     | 0.627     | **0.579** | 0.753     | **0.740** | 4.960     |

† medium-b-sampled INLA σ_rfx = 4.490: outlier driven by 4th quartile (σ>0.9,
  RMSE=5.5, positive-skewed).  Likely numerical instability for high-σ datasets
  with q≤3 correlated RE. The analytical path wins this cell.

Key Findings
-------------

**FFX**: Bernoulli EB closes the old medium/large/huge Bernoulli fixed-effect failure. It is
now better than INLA on small/medium and sampled rows, and essentially tied on
large/huge mixed rows.

**σ_rfx**: The remaining consistent INLA edge is variance scale on medium+ rows. Bernoulli EB is
close on large mixed and sampled rows, but still over-shrinks high-σ cases more than INLA.
The medium-sampled INLA σ cell remains a numerical outlier and should not drive decisions.

**BLUP**: Bernoulli EB is tied or slightly better at small scale; INLA keeps a small but
consistent edge on medium+ rows, mostly tracking the remaining σ_rfx gap.

**Speed**: Bernoulli EB remains in the tens of milliseconds per dataset; R-INLA is seconds per
dataset, roughly two orders of magnitude slower on these benchmarks.

Normal Diagonal R-INLA Snapshot
-------------------------------

The retained normal analytical path now carries MAP β internally for `d > 4`, reports a
prior-capped β (`ν_ffx ± 4τ_ffx`) to remove rare FFX tail explosions, keeps the uncapped
MAP β for BLUP residuals, and applies the one-shot EB σ update by default.

Small-scale rough placement, mixed/train, first 100 datasets per row. MAP columns are
historical internal-stage diagnostics, not a retained standalone mode.

| Dataset | RAW FFX | MAP FFX | EB FFX | INLA FFX | RAW σ | MAP σ | EB σ | INLA σ | RAW BLUP | MAP BLUP | EB BLUP | INLA BLUP | RAW ms | MAP ms | EB ms | INLA s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | 0.3355 | 0.3349 | 0.3349 | 0.3150 | 0.6666 | 0.5441 | 0.5400 | 0.4761 | 0.5040 | 0.4975 | 0.4974 | 0.4770 | 0.35 | 9.42 | 2.44 | 2.568 |
| medium-n-mixed | 0.4407 | 0.4380 | 0.4345 | 0.3035 | 0.8218 | 0.6634 | 0.6099 | 0.4939 | 0.5697 | 0.5436 | 0.5398 | 0.5258 | 0.73 | 3.56 | 4.03 | 2.964 |
| large-n-mixed | 1.5804 | 1.5953 | 1.5972 | 0.3387 | 0.8528 | 0.7878 | 0.6572 | 0.5475 | 0.6719 | 0.6586 | 0.6640 | 0.6139 | 1.10 | 42.77 | 4.88 | 2.699 |
| huge-n-mixed | 0.7923 | 0.7751 | 0.7747 | 0.3054 | 0.6018 | 0.3913 | 0.3501 | 0.3273 | 0.5095 | 0.4750 | 0.4737 | 0.4516 | 1.19 | 4.92 | 5.60 | 2.858 |

Current analytical mixed/train rows after the prior β cap, first 1000 datasets per row:

| Dataset | MAP FFX | EB FFX | MAP σ | EB σ | MAP BLUP | EB BLUP | MAP ms | EB ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | 0.1096 | 0.1095 | 0.4814 | 0.4203 | 0.4192 | 0.4173 | 2.95 | 2.52 |
| medium-n-mixed | 0.2515 | 0.2515 | 0.3798 | 0.3619 | 0.4212 | 0.4198 | 2.73 | 3.05 |
| large-n-mixed | 0.4075 | 0.4075 | 0.4148 | 0.3711 | 0.4155 | 0.4148 | 3.71 | 4.23 |
| huge-n-mixed | 0.3314 | 0.3314 | 0.4280 | 0.3776 | 0.4573 | 0.4545 | 4.74 | 5.34 |

Mixed/train diagonal R-INLA reference, first 1000 datasets per row. The INLA cells are
from the completed rerun; analytical FFX values above include the post-rerun prior β cap
on the same first-1000 rows.

| Dataset | RAW FFX | MAP FFX | INLA FFX | RAW σ | MAP σ | INLA σ | RAW BLUP | MAP BLUP | INLA BLUP | RAW ms | MAP ms | INLA s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | 0.1124 | 0.1096 | 0.0985 | 0.7978 | 0.4814 | 0.3665 | 0.4761 | 0.4192 | 0.4081 | 0.37 | 7.26 | 2.368 |
| medium-n-mixed | 0.5758 | 0.2515 | 0.2301 | 0.5236 | 0.3798 | 0.3419 | 0.4550 | 0.4212 | 0.4289 | 0.63 | 2.73 | 2.604 |
| large-n-mixed | 1.7363 | 0.4075 | 0.2377 | 0.5449 | 0.4148 | 0.3393 | 0.4675 | 0.4155 | 0.4185 | 0.96 | 3.71 | 2.786 |
| huge-n-mixed | 1.0635 | 0.3314 | 0.2413 | 0.5752 | 0.4280 | 0.2808 | 0.4925 | 0.4573 | 0.4548 | 1.37 | 4.74 | 2.965 |

Normal takeaways:

- The prior β cap closes most of the normal FFX gap on small/medium/huge and reduces the
  large-row FFX gap substantially.
- EB closes additional σ/BLUP error with a one-shot update.
- The remaining Gaussian FFX issue appears tail-dominated: on patched MAP, large/huge
  mixed/train have mean per-dataset β RMSE around `0.06-0.09`, but rare maxima above
  `5-10` drove aggregate NRMSE before the prior cap.
- INLA remains about seconds per dataset; analytical EB is milliseconds per dataset.

Normal tail diagnostic, first 1000 mixed/train rows per size:

The diagnostic scanned 4000 rows with Normal EB, selected the 12 highest
cap-prioritized FFX-error rows per size, and ran diagonal R-INLA only on those selected
tail rows.

```bash
uv run python experiments/analytical/glmm_normal_inla_diagnostic.py \
    --tail-scan 1000 --tail-k 12 --batch-size 32 --top-k 16 \
    --output-csv /private/tmp/normal_inla_tail_1k_top12.csv
```

| Tail subset | N | EB FFX | INLA FFX | EB σ | INLA σ | EB BLUP | INLA BLUP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | 12 | 0.7367 | 0.6962 | 0.2593 | 0.1929 | 0.5497 | 0.5180 |
| medium-n-mixed | 12 | 0.2645 | 0.0931 | 0.1143 | 0.1278 | 0.1777 | 0.1791 |
| large-n-mixed | 12 | 0.6927 | 0.0950 | 0.0531 | 0.0546 | 0.1045 | 0.1066 |
| huge-n-mixed | 12 | 0.4109 | 0.1063 | 0.1131 | 0.0877 | 0.1808 | 0.1755 |

Tail takeaways:

- Cap-hit rows dominate the remaining medium/large/huge FFX gap: selected cap-hit rows
  have EB FFX `0.4559` versus INLA `0.0984`, and INLA is better on `34/35` of them.
- The large tail is almost pure fixed-effect reporting error: EB and INLA BLUP are tied
  (`0.1045` versus `0.1066`), while FFX differs by `0.5977`.
- Worst large rows have high residualized-X condition numbers (`~200-2000`) and prior-cap
  hits. This points to weakly identified β directions where hard output clipping is too
  crude compared with INLA posterior means.
- The most promising patch is now reporting-only σ-grid averaging for cap-hit `d > 4`
  rows. It keeps uncapped MAP β for BLUP and replaces only reported cap-hit fixed effects
  with a small marginal-target-weighted average over nearby σ_rfx values.

The old direct cap-shrinkage modes were removed after benchmarking. They were useful
diagnostics, but the σ-grid approximation is the better match to INLA's hyperparameter
averaging behavior.

First-1000 required normal rows, all sizes and dataset types:

| Dataset | Part | sigma FFX | tail FFX | sigma ms | tail ms |
| --- | --- | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.1095 | 0.1095 | 3.50 | 3.25 |
| small-n-sampled | valid | 0.2588 | 0.2588 | 2.66 | 2.48 |
| small-n-sampled | test | 0.2827 | 0.2827 | 2.62 | 2.46 |
| medium-n-mixed | train | 0.2283 | 0.2270 | 4.02 | 3.59 |
| medium-n-sampled | valid | 0.2626 | 0.2677 | 4.84 | 4.27 |
| medium-n-sampled | test | 0.2594 | 0.2592 | 4.86 | 4.18 |
| large-n-mixed | train | 0.2630 | 0.2634 | 5.55 | 5.27 |
| large-n-sampled | valid | 0.2994 | 0.2969 | 6.06 | 5.08 |
| large-n-sampled | test | 0.2878 | 0.2869 | 6.41 | 5.49 |
| huge-n-mixed | train | 0.2799 | 0.2807 | 7.03 | 6.14 |
| huge-n-sampled | valid | 0.4448 | 0.4440 | 8.34 | 7.35 |
| huge-n-sampled | test | 0.3037 | 0.3104 | 8.56 | 7.68 |

Accuracy is effectively tied. Tail-grid is slightly faster here because it only changes
the severe cap-excess rows, but it is slightly worse on huge sampled test. Keep
three-point `sigma_grid` as the main candidate and `tail_grid` as a fallback diagnostic.
