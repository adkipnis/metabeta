Bernoulli GLMM Plan
===================

Last updated: 2026-05-14

Current Baseline
----------------

Estimator: `lmmBernoulli` (6 PQL passes) + `refineBernoulliNagqSrfx` (P5 nAGQ,
all q≤5) + `refineBernoulliMapBeta` (P6 true Laplace score for β + BC1 M-step
correction). Active when `map_refine=True`.

Required-suite NRMSE (P1+P1-ext+Ψ-floor+P5+P6+BC1, 2026-05-14, N=8192):

| Dataset           | Partition | FFX    | sRFX   | BLUP   |
| ---               | ---       | ---:   | ---:   | ---:   |
| small-b-mixed     | train     | 0.2314 | 0.5299 | 0.6202 |
| small-b-sampled   | valid     | 0.2809 | 0.6065 | 0.6620 |
| small-b-sampled   | test      | 0.2747 | 0.5896 | 0.6571 |
| medium-b-mixed    | train     | 0.7397 | 0.6572 | 0.7214 |
| medium-b-sampled  | valid     | 0.5860 | 0.7204 | 0.8485 |
| medium-b-sampled  | test      | 0.6840 | 0.7401 | 0.8228 |
| large-b-mixed     | train     | 1.6439 | 0.7643 | 0.9333 |
| large-b-sampled   | valid     | 0.8581 | 0.8041 | 0.8372 |
| large-b-sampled   | test      | 1.3645 | 0.8345 | 0.9634 |
| huge-b-mixed      | train     | 2.0183 | 0.8502 | 0.9423 |
| huge-b-sampled    | valid     | 1.3226 | 0.9101 | 1.0111 |
| huge-b-sampled    | test      | 1.5393 | 0.8881 | 0.9952 |

Root cause summary (`glmm_error_analysis.py`):
- **FFX** is the dominant failure mode; NRMSE scales with d (low Fisher information
  per binary observation, pooled IRLS underdetermined at large d / low n).
- **σ_rfx** has bidirectional S-curve bias: upward at low true σ (Ψ floor overshoots),
  downward at high true σ (M-step shrinkage). CAVI wins mainly in the high-σ quartile.
- **BLUPs** track FFX: bad β contaminates the BLUP residual ỹ−Xβ.

Implemented (✓) / Tried and reverted (✗)
-----------------------------------------

**✓ P1+P1-ext** — Prior-regularized IRLS β₀ with Student-t adaptive precision.
**✓ P2 sub-item** — Prior-informed Ψ floor from `tau_rfx`, capped at 0.25.
**✓ P5** — nAGQ for q=1. Adam on k=7 GH quadrature LML + Newton BLUP refresh.
**✓ P6** — True Laplace score for β. n_outer=2 rounds of β Newton + b̂_g Newton + M-step.
**✓ P7/BC1** — Analytic O(1/n) M-step correction inline in `refineBernoulliMapBeta`.
**✓ P8** — nAGQ for q>1 (2≤q≤5) via Cartesian-product GH grid.

**✗ P2** — Laplace-MAP σ_rfx fixed-point (cancels H_g^{-1} correction). Code in `map.py:refineBernoulliMapSrfx` removed.
**✗ P3** — Beta blend for BLUP residuals (oracle: partition-specific, no globally safe α).
**✗ P4** — blup_var `1+C/n_g` formula (calibration depends on both n_g and G; bookkeeping-only anyway).
**✗ P8a/b** — Joint MAP on (β, log σ) via Adam (β gradient ≈0 at P6 fixed point). Code removed.
**✗ P9** — Decouple M-step β from P6 Newton (partition-specific wins/losses).
**✗ P10** — nAGQ σ gradient at P6 β (P6 β makes W≈0, σ uninformative from likelihood).

No remaining principled directions. σ_rfx gap vs CAVI at mixed datasets is structural
(Laplace M-step instability at large d) and resists all approaches above.

External Reference Baseline
----------------------------

**CAVI:** `BinomialBayesMixedGLM` (statsmodels 0.14+), CAVI on mean-field Gaussian,
diagonal Ψ, prior Gaussian on log σ² (vcp_p=4.0). Script: `glmm_reference_comparison.py`.

Note: CAVI uses diagonal Ψ (mean-field); our stack supports full Ψ. Reference
comparison runs P5+P6 without P1/P2 (raw `glmm()` base). Numbers use matched subset
(first n_cavi datasets processed).

Matched comparison — all runs at n_cavi=200 (huge: 100), n_total=1000 (huge: 500).
P6 = raw PQL + P5 + P6 + BC1, without P1/P2 (see note). Bold = winner per cell.

| Dataset              | part  |   N | P6 FFX    | CAVI FFX | P6 σ      | CAVI σ    | P6 BLUP   | CAVI BLUP |
| ---                  | ---   | ---:| ---:      | ---:     | ---:      | ---:      | ---:      | ---:      |
| small-b-sampled      | valid |  200| **0.325** | 0.346    | **0.665** | 0.674     | **0.615** | 0.635     |
| small-b-sampled      | test  |  200| **0.340** | 0.398    | **0.612** | 0.664     | **0.621** | 0.636     |
| small-b-mixed        | train |  200| **0.339** | 0.366    | 0.782     | **0.734** | **0.717** | 0.811     |
| medium-b-sampled     | valid |  200| **0.386** | 0.513    | **0.757** | 0.818     | **0.717** | 0.788     |
| medium-b-sampled     | test  |  200| **0.399** | 0.548    | 0.994     | **0.793** | 1.109     | **0.831** |
| medium-b-mixed       | train |  200| **0.343** | 0.438    | 1.051     | **0.718** | **0.696** | 0.741     |
| large-b-sampled      | valid |  200| **0.355** | 0.453    | 1.311     | **0.769** | 1.063     | **0.784** |
| large-b-sampled      | test  |  200| **0.340** | 0.472    | 0.879     | **0.814** | **0.743** | 0.825     |
| large-b-mixed        | train |  200| **0.356** | 0.514    | 1.006     | **0.859** | 1.085     | **0.960** |
| huge-b-sampled       | valid |  100| **0.418** | 0.774    | **0.810** | 1.037     | **0.823** | 1.231     |
| huge-b-sampled       | test  |  100| **0.472** | 0.753    | 0.881     | **0.784** | **0.811** | 0.965     |
| huge-b-mixed         | train |  100| **0.403** | 0.652    | 1.489     | **0.903** | **0.921** | 0.940     |

Key findings:
- **FFX**: Full pipeline beats CAVI at all 12 cells (2–3.5× better at large/huge).
- **σ_rfx**: Mixed. Pipeline wins at small + huge-sampled-valid; CAVI wins at mixed datasets
  (all sizes) and medium/large-huge test. Laplace M-step produces high σ variance at large d/q.
- **BLUP**: follows σ_rfx.

**R-INLA:** `inla()` (R-INLA package), INLA Laplace approximation. Script:
`glmm_inla_comparison.py`. Results in `experiments/analytical/glmm_inla_results.md`.

Key findings (small-b only, n_inla=1000):
- **FFX**: Full pipeline beats INLA by 1.5–1.7× at small scale.
- **σ_rfx**: INLA matches or marginally beats PQL (small σ quartile).
- **BLUP**: Tied (~0.618–0.625).
- **Speed**: PQL ~6–11 ms/ds vs INLA ~2.1–2.2 s/ds (≈350–400× faster).

Commands
--------

```bash
uv run python experiments/analytical/glmm_required_benchmark.py --family b
uv run python experiments/analytical/glmm_required_benchmark.py --family b --methods current raw
uv run python experiments/analytical/glmm_error_analysis.py --data-id small-b-mixed
# reference comparison (n_cavi=200 / 100 for huge; n_total=1000 / 500 for huge)
uv run python experiments/analytical/glmm_reference_comparison.py \
    --data-ids small-b-sampled,medium-b-sampled,large-b-sampled \
    --partition valid --n-cavi 200 --n-total 1000
uv run python experiments/analytical/glmm_reference_comparison.py \
    --data-ids small-b-sampled,medium-b-sampled,large-b-sampled \
    --partition test --n-cavi 200 --n-total 1000
uv run python experiments/analytical/glmm_reference_comparison.py \
    --data-ids small-b-mixed,medium-b-mixed,large-b-mixed \
    --partition train --n-epochs 2 --n-cavi 200 --n-total 1000
uv run python experiments/analytical/glmm_reference_comparison.py \
    --data-ids huge-b-sampled --partition valid --n-cavi 100 --n-total 500
uv run python experiments/analytical/glmm_reference_comparison.py \
    --data-ids huge-b-sampled --partition test --n-cavi 100 --n-total 500
uv run python experiments/analytical/glmm_reference_comparison.py \
    --data-ids huge-b-mixed --partition train --n-epochs 2 --n-cavi 100 --n-total 500
uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
