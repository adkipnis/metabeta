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
damped tail-gated β posterior-mean correction (OLS fallback for double-trigger cap+stab
rows, Direction K 2026-05-25), one-shot posterior-moment σ_rfx EB calibration, direct
σ_rfx coordinate grid, τ_rfx floor in coordinate search, profile-MAP σ_rfx rescue for
stuck-zero components (2026-05-25), diagonal final Ψ, and the rare BLUP/sigma guard for
high-d aliased rows.

Mixed/train diagonal R-INLA reference, first 1000 datasets per row:

| Dataset | EB FFX | current FFX | INLA FFX | current σ | INLA σ | current BLUP | INLA BLUP | current ms | INLA s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | 0.1095 | 0.1085 | 0.0985 | 0.3863 | 0.3665 | 0.4148 | 0.4081 | 5.93 | 2.406 |
| medium-n-mixed | 0.2515 | **0.2208** | 0.2301 | 0.3558 | 0.3419 | **0.4193** | 0.4289 | 7.77 | 2.604 |
| large-n-mixed | 0.4075 | 0.2449 | **0.2377** | 0.3587 | **0.3393** | **0.4124** | 0.4185 | 12.52 | 2.786 |
| huge-n-mixed | 0.3314 | 0.2531 | **0.2413** | 0.3001 | **0.2808** | **0.4515** | 0.4548 | 14.94 | 2.965 |

Takeaways:

- Direction K (OLS fallback for the ~2% of large/huge datasets where both β-cap and
  σ-grid stabilization fire) significantly narrowed the FFX gap: large-n-mixed improved
  from 0.2582 → 0.2449, huge-n-mixed from 0.2677 → 0.2531. Medium also benefited
  (0.2283 → 0.2208) from the small fraction of d>4 rows in that size class.
- The profile-MAP σ_rfx rescue (2026-05-25) targets stuck-zero σ_rfx components where
  the coordinate grid can't escape a wrong local minimum. It fired for ~2% of huge-n rows
  and improved huge-n-sampled valid FFX from 0.4112 → 0.3746. No regressions on any
  other size/partition.
- Current now outperforms INLA on medium-n FFX; large/huge still trail INLA by ≤0.015.
- INLA keeps the best σ_rfx accuracy on medium+ test rows; current wins on huge valid σ
  (0.3392 vs 0.5085). Analytical BLUP beats INLA on medium+ rows.
- R-INLA remains ~8-10s per dataset, while analytical EB/σ-grid is ~40-160ms per
  dataset (increasing with d due to rescue and grid refinement passes).
- A 2026-05-18 FFX-tail diagnostic scanned 8000 medium/large/huge mixed rows and ran
  INLA on the 16 worst FFX rows per size. The remaining large/huge gap is rare and
  concentrated in high-d or ill-conditioned rows; INLA's β posterior-mean shift is
  strongly aligned with the analytical β error in those tail rows.
- A sampled FFX-tail diagnostic on large/huge valid/test shows the same signature:
  high-d rows, frequent singular or near-singular fixed/random design, many β prior-cap
  hits, and no material BLUP gap. The worst valid tails remain behind INLA, but this is a
  narrow tail rather than a broad sampled-set failure.
- Sampled-set INLA rows were completed on 2026-05-18 with one saved unbuffered block per
  size/partition under `experiments/analytical/inla_runs/`.

Normal sampled first-1000 rows with diagonal R-INLA (rerun 2026-05-25, with σ_rfx rescue):

| Dataset | part | current FFX | INLA FFX | current σ | INLA σ | current BLUP | INLA BLUP | current ms | INLA s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-sampled | valid | 0.2735 | **0.2151** | **0.4881** | 0.5313 | 0.4812 | **0.4755** | 38.63 | 9.853 |
| small-n-sampled | test | 0.2393 | **0.1675** | **0.4005** | 0.4119 | 0.4228 | **0.4156** | 42.04 | 9.910 |
| medium-n-sampled | valid | 0.2806 | **0.2296** | 0.4097 | **0.3201** | **0.5710** | 0.5733 | 78.98 | 8.329 |
| medium-n-sampled | test | 0.2410 | **0.2339** | 0.3160 | **0.3103** | **0.4370** | 0.4426 | 75.08 | 8.326 |
| large-n-sampled | valid | 0.2710 | **0.2389** | 0.3239 | **0.3226** | **0.4734** | 0.4769 | 100.36 | 8.897 |
| large-n-sampled | test | 0.2640 | **0.2514** | 0.3765 | **0.3601** | **0.4731** | 0.4726 | 88.81 | 8.877 |
| huge-n-sampled | valid | 0.3746 | **0.2907** | **0.3392** | 0.5085 | **0.4873** | 0.4897 | 159.90 | 10.143 |
| huge-n-sampled | test | 0.2601 | **0.2491** | 0.3169 | **0.2895** | **0.5531** | 0.5554 | 126.41 | 10.135 |

Sampled FFX-tail diagnostic, 8000-row scan with diagonal R-INLA on the 16 worst current
FFX rows per size/partition:

| Tail set | part | current β RMSE | INLA β RMSE | Δβ | cap rows | singular/near-singular rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| large-n-sampled | valid | 0.6766 | 0.3425 | +0.3341 | 8/16 | 11/16 |
| huge-n-sampled | valid | 0.7014 | 0.2925 | +0.4089 | 12/16 | 16/16 |
| large-n-sampled | test | 0.6440 | 0.3715 | +0.2725 | 10/16 | 12/16 |
| huge-n-sampled | test | 0.4434 | 0.2185 | +0.2249 | 8/16 | 13/16 |
