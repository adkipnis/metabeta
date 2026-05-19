R-INLA vs Analytical GLMM Comparison Results
=============================================

`glmm_inla_comparison.py` — full analytical pipeline (`glmm()`, `map_refine=True`) vs
R-INLA. For Bernoulli, the analytical column is the current default Bernoulli EB path.
Results are reported on the matched first-1000 rows used for INLA. Bernoulli mixed rows
use train/ep2; Bernoulli sampled rows now include both valid and test with explicit
diagonal R-INLA.

NRMSE Summary
-------------

Bold = better method per column. The mixed rows were rerun after Bernoulli EB became the
default Bernoulli path. Sampled valid/test rows were rerun on 2026-05-19, one
unbuffered log per size/partition under `experiments/analytical/inla_runs/bernoulli_sampled/`.

| Dataset           | part  | EB FFX    | INLA FFX  | EB σ      | INLA σ    | EB BLUP   | INLA BLUP | INLA s/ds |
| ---               | ---   | ---:      | ---:      | ---:      | ---:      | ---:      | ---:      | ---:      |
| small-b-mixed     | train | **0.267** | 0.451     | **0.510** | 0.567     | **0.614** | 0.618     | n/a       |
| small-b-sampled   | valid | **0.313** | 0.559     | **0.538** | 0.539     | **0.639** | 0.639     | 2.753     |
| small-b-sampled   | test  | **0.293** | 0.446     | **0.504** | 0.517     | **0.609** | 0.618     | 2.764     |
| medium-b-mixed    | train | **0.313** | 0.332     | 0.539     | **0.522** | 0.686     | **0.648** | 2.566     |
| medium-b-sampled  | valid | **0.333** | 0.345     | 0.557     | **0.541** | 0.691     | **0.670** | 2.975     |
| medium-b-sampled  | test  | **0.339** | 0.400     | 0.584     | **0.563** | 0.707     | **0.690** | 2.973     |
| large-b-mixed     | train | 0.332     | **0.323** | 0.542     | **0.521** | 0.685     | **0.676** | 3.273     |
| large-b-sampled   | valid | **0.367** | 0.371     | 0.594     | **0.591** | 0.742     | **0.737** | 3.248     |
| large-b-sampled   | test  | **0.357** | 0.365     | 0.620     | **0.569** | 0.727     | **0.708** | 3.258     |
| huge-b-mixed      | train | 0.335     | **0.330** | 0.600     | **0.550** | 0.737     | **0.713** | 4.605     |
| huge-b-sampled    | valid | **0.386** | 0.393     | 0.611     | **0.569** | 0.767     | **0.751** | 3.763     |
| huge-b-sampled    | test  | **0.378** | 0.394     | 0.627     | **0.548** | 0.753     | **0.737** | 3.783     |

The old medium-b-sampled σ_rfx outlier was not reproduced under the current explicit
diagonal R-INLA specification.

Key Findings
-------------

**FFX**: Bernoulli EB closes the old medium/large/huge Bernoulli fixed-effect failure. It is
better than INLA on all sampled rows and essentially tied on large/huge mixed rows.

**σ_rfx**: The remaining consistent INLA edge is variance scale on medium+ rows. Bernoulli EB is
close on large mixed and sampled rows, but still over-shrinks high-σ cases more than INLA.

**BLUP**: Bernoulli EB is tied or slightly better at small scale; INLA keeps a small but
consistent edge on medium+ rows, mostly tracking the remaining σ_rfx gap.

**Speed**: Bernoulli EB remains in the tens of milliseconds per dataset; R-INLA is seconds per
dataset, roughly two orders of magnitude slower on these benchmarks.

Normal Diagonal R-INLA Snapshot
-------------------------------

The retained Normal path is now guarded EB plus tail β correction by default: MAP
β/σ/σ_eps refinement, reporting-only prior cap for `d > 4`, scalar β sigma-grid reporting,
damped tail-gated β posterior-mean correction, one-shot posterior-moment σ_rfx EB
calibration, direct σ_rfx coordinate grid, diagonal final Ψ, and the rare BLUP/sigma guard
for high-d aliased rows.

Mixed/train diagonal R-INLA reference, first 1000 datasets per row:

| Dataset | EB FFX | current FFX | INLA FFX | EB σ | INLA σ | EB BLUP | INLA BLUP | current ms | INLA s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | 0.1095 | 0.1089 | 0.0985 | 0.4203 | 0.3665 | 0.4173 | 0.4081 | 3.42 | 2.406 |
| medium-n-mixed | 0.2515 | 0.2283 | 0.2301 | 0.3619 | 0.3419 | 0.4198 | 0.4289 | 3.05 | 2.604 |
| large-n-mixed | 0.4075 | 0.2582 | 0.2377 | 0.3711 | 0.3393 | 0.4148 | 0.4185 | 6.20 | 2.786 |
| huge-n-mixed | 0.3314 | 0.2677 | 0.2413 | 0.3776 | 0.2808 | 0.4545 | 0.4548 | 7.89 | 2.965 |

Takeaways:

- guarded EB plus damped tail β correction closes more of the large/huge FFX gap but
  still trails INLA in the hardest rows.
- INLA keeps the best σ_rfx accuracy; analytical BLUP is already tied on medium+ rows.
- R-INLA remains seconds per dataset, while analytical EB/σ-grid is milliseconds.
- A 2026-05-18 FFX-tail diagnostic scanned 8000 medium/large/huge mixed rows and ran
  INLA on the 16 worst FFX rows per size. The remaining large/huge gap is rare and
  concentrated in high-d or ill-conditioned rows; INLA's β posterior-mean shift is
  strongly aligned with the analytical β error in those tail rows.
- A sampled FFX-tail diagnostic on large/huge valid/test shows the same signature:
  high-d rows, frequent singular or near-singular fixed/random design, many β prior-cap
  hits, and no material BLUP gap. The worst valid tails remain behind INLA, but this is a
  narrow tail rather than a broad sampled-set failure.
- The retained tail β correction is scalar-grid-only and damped (`25%` toward the grid
  posterior mean). It improved the saved large/huge INLA FFX tail rows while leaving
  medium unchanged.
- Sampled-set INLA rows were completed on 2026-05-18 with one saved unbuffered block per
  size/partition under `experiments/analytical/inla_runs/`.

Normal sampled first-1000 rows with diagonal R-INLA:

| Dataset | part | current FFX | INLA FFX | current σ | INLA σ | current BLUP | INLA BLUP | current ms | INLA s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-sampled | valid | 0.2608 | 0.2023 | 0.5695 | 0.4450 | 0.5119 | 0.4988 | 2.87 | 2.377 |
| small-n-sampled | test | 0.2828 | 0.2008 | 0.4759 | 0.4357 | 0.4931 | 0.4756 | 2.73 | 2.382 |
| medium-n-sampled | valid | 0.2626 | 0.2490 | 0.4186 | 0.4048 | 0.5145 | 0.5201 | 4.81 | 3.032 |
| medium-n-sampled | test | 0.2594 | 0.2594 | 0.3825 | 0.3419 | 0.4403 | 0.4506 | 5.35 | 3.020 |
| large-n-sampled | valid | 0.2970 | 0.2527 | 0.4159 | 0.3428 | 0.5045 | 0.5069 | 6.21 | 3.430 |
| large-n-sampled | test | 0.2872 | 0.2710 | 0.4346 | 0.3984 | 0.5126 | 0.5052 | 6.65 | 3.431 |
| huge-n-sampled | valid | 0.4240 | 0.3110 | 0.3562 | 4.4979 † | 0.4555 | 0.4598 | 10.27 | 3.784 |
| huge-n-sampled | test | 0.2947 | 0.2732 | 0.3689 | 0.3122 | 0.4604 | 0.4639 | 9.63 | 3.768 |

† `huge-n-sampled valid` INLA σ_rfx is a numerical outlier: the second true-σ quartile
has RMSE `2.1247`, while the other quartiles are around `0.07-0.12`.

Sampled FFX-tail diagnostic, 8000-row scan with diagonal R-INLA on the 16 worst current
FFX rows per size/partition:

| Tail set | part | current β RMSE | INLA β RMSE | Δβ | cap rows | singular/near-singular rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| large-n-sampled | valid | 0.6766 | 0.3425 | +0.3341 | 8/16 | 11/16 |
| huge-n-sampled | valid | 0.7014 | 0.2925 | +0.4089 | 12/16 | 16/16 |
| large-n-sampled | test | 0.6440 | 0.3715 | +0.2725 | 10/16 | 12/16 |
| huge-n-sampled | test | 0.4434 | 0.2185 | +0.2249 | 8/16 | 13/16 |
