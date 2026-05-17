R-INLA vs Analytical GLMM Comparison Results
=============================================

`glmm_inla_comparison.py` — full analytical pipeline (`glmm()`, `map_refine=True`) vs
R-INLA. For Bernoulli, the analytical column is the current default P14-cal path. Results
are reported on the matched subset used for INLA where available. mixed=train/ep2,
sampled=test.

NRMSE Summary
-------------

Bold = better method per column. The mixed rows were rerun after P14-cal became the
default Bernoulli path. The sampled rows combine the current P14-cal benchmark with the
existing R-INLA reference run.

| Dataset           | part  | P14 FFX   | INLA FFX  | P14 σ     | INLA σ    | P14 BLUP  | INLA BLUP | INLA s/ds |
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

**FFX**: P14-cal closes the old medium/large/huge Bernoulli fixed-effect failure. It is
now better than INLA on small/medium and sampled rows, and essentially tied on
large/huge mixed rows.

**σ_rfx**: The remaining consistent INLA edge is variance scale on medium+ rows. P14-cal is
close on large mixed and sampled rows, but still over-shrinks high-σ cases more than INLA.
The medium-sampled INLA σ cell remains a numerical outlier and should not drive decisions.

**BLUP**: P14-cal is tied or slightly better at small scale; INLA keeps a small but
consistent edge on medium+ rows, mostly tracking the remaining σ_rfx gap.

**Speed**: P14-cal remains in the tens of milliseconds per dataset; R-INLA is seconds per
dataset, roughly two orders of magnitude slower on these benchmarks.

Normal Diagonal R-INLA Snapshot
-------------------------------

Mixed/train rows, first 1000 datasets per row. This run used `raw,map` only; the
`normal_eb` prototype was added afterward. The normal reference uses diagonal random
effects because the exact correlated Gaussian INLA branch was numerically unstable on
these datasets.

| Dataset | RAW FFX | MAP FFX | INLA FFX | RAW σ | MAP σ | INLA σ | RAW BLUP | MAP BLUP | INLA BLUP | RAW ms | MAP ms | INLA s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | 0.1124 | 0.1096 | 0.0985 | 0.7978 | 0.4814 | 0.3665 | 0.4761 | 0.4192 | 0.4081 | 0.37 | 7.26 | 2.368 |
| medium-n-mixed | 0.5758 | 0.5489 | 0.2301 | 0.5236 | 0.3798 | 0.3421 | 0.4550 | 0.4409 | 0.4288 | 0.57 | 2.81 | 2.614 |
| large-n-mixed | 1.7363 | 1.8207 | 0.2377 | 0.5449 | 0.4148 | 0.3397 | 0.4675 | 0.4361 | 0.4185 | 0.85 | 3.80 | 2.786 |
| huge-n-mixed | 1.0635 | 1.3100 | 0.2413 | 0.5752 | 0.4280 | 0.2809 | 0.4925 | 0.4742 | 0.4548 | 1.22 | 5.11 | 3.071 |

Normal takeaways:

- INLA is best on all mixed/train normal rows, especially FFX on medium/large/huge.
- MAP closes much of the raw σ/BLUP gap but leaves measurable σ/BLUP room for a
  fast diagonal EB calibration.
- INLA remains about seconds per dataset; analytical MAP is milliseconds per dataset.
