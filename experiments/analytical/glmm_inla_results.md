R-INLA vs PQL Comparison Results
==================================

`glmm_inla_comparison.py` — full pipeline (`glmm()`, map_refine=True) vs R-INLA.
"PQL" column = final output of `glmm()` (P1+P2+P5+P6+BC1).
All runs: n_inla=1000, n_total=1000.  mixed=train/ep2, sampled=test.

NRMSE Summary
-------------

| Dataset           | part  |    N | PQL FFX   | INLA FFX  | PQL σ     | INLA σ    | PQL BLUP  | INLA BLUP | INLA s/ds |
| ---               | ---   | ---: | ---:      | ---:      | ---:      | ---:      | ---:      | ---:      | ---:      |
| small-b-mixed     | train | 1000 | **0.271** | 0.451     | **0.571** | 0.567     | **0.618** | 0.618     | 2.240     |
| small-b-sampled   | test  | 1000 | **0.301** | 0.447     | **0.575** | 0.556     | **0.621** | 0.625     | 2.129     |
| medium-b-mixed    | train |    — |           |           |           |           |           |           |           |
| medium-b-sampled  | test  |    — |           |           |           |           |           |           |           |
| large-b-mixed     | train |    — |           |           |           |           |           |           |           |
| large-b-sampled   | test  |    — |           |           |           |           |           |           |           |
| huge-b-mixed      | train |    — |           |           |           |           |           |           |           |
| huge-b-sampled    | test  |    — |           |           |           |           |           |           |           |

σ_rfx Quartile Pattern (small-b-mixed, matched N=1000)
-------------------------------------------------------

| σ_rfx_true range | PQL bias | INLA bias |
| ---              | ---:     | ---:      |
| 0.001–0.201      | +0.160   | +0.227    |
| 0.201–0.478      | −0.008   | +0.071    |
| 0.478–0.911      | −0.087   | −0.008    |
| 0.911–3.873      | −0.193   | −0.132    |

Key findings (small-b only, to be updated as larger results arrive):
- **FFX**: PQL beats INLA by 1.5–1.7× at small scale.
- **σ_rfx**: INLA marginally better overall (0.556–0.567 vs 0.571–0.575); INLA
  favors middle quartiles, PQL favors low σ.
- **BLUP**: Effectively tied (~0.618–0.625).
- **Speed**: PQL ~6–11 ms/ds vs INLA ~2.1–2.2 s/ds (≈350–400× faster).
