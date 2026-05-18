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

The retained Normal path is EB: MAP β/σ/σ_eps refinement, reporting-only prior cap for
`d > 4`, uncapped MAP β for BLUP residuals, diagonal final Ψ, and one-shot posterior-moment
σ_rfx EB calibration.

Mixed/train diagonal R-INLA reference, first 1000 datasets per row:

| Dataset | EB FFX | σ-grid FFX | INLA FFX | EB σ | INLA σ | EB BLUP | INLA BLUP | EB ms | INLA s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | 0.1095 | 0.1095 | 0.0985 | 0.4203 | 0.3665 | 0.4173 | 0.4081 | 2.52 | 2.406 |
| medium-n-mixed | 0.2515 | 0.2283 | 0.2301 | 0.3619 | 0.3419 | 0.4198 | 0.4289 | 3.05 | 2.604 |
| large-n-mixed | 0.4075 | 0.2630 | 0.2377 | 0.3711 | 0.3393 | 0.4148 | 0.4185 | 4.23 | 2.786 |
| huge-n-mixed | 0.3314 | 0.2799 | 0.2413 | 0.3776 | 0.2808 | 0.4545 | 0.4548 | 5.34 | 2.965 |

Takeaways:

- σ-grid closes much of the large/huge FFX gap but still trails INLA in the hardest rows.
- INLA keeps the best σ_rfx accuracy; analytical BLUP is already tied on medium+ rows.
- R-INLA remains seconds per dataset, while analytical EB/σ-grid is milliseconds.
- Sampled-set INLA rows are still pending. Run them with unbuffered output:

```bash
uv run python -u experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-n-sampled,medium-n-sampled,large-n-sampled,huge-n-sampled \
    --partition valid --n-inla 1000 --n-total 1000 \
    --analytical-methods normal_eb,normal_sigma_grid --re-correlation diagonal
```
