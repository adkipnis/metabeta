Bernoulli GLMM Plan
===================

Last updated: 2026-05-15

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

Implemented (✓) / Tried and reverted (✗) / In progress (→)
--------------------------------------------------------------

**✓ P1+P1-ext** — Prior-regularized IRLS β₀ with Student-t adaptive precision.
**✓ P2 sub-item** — Prior-informed Ψ floor from `tau_rfx`, capped at 0.25.
**✓ P5** — nAGQ for q=1. Adam on k=7 GH quadrature LML + Newton BLUP refresh.
**✓ P6** — True Laplace score for β. n_outer=2 rounds of β Newton + b̂_g Newton + M-step.
**✓ P7/BC1** — Analytic O(1/n) M-step correction inline in `refineBernoulliMapBeta`.
**✓ P8** — nAGQ for q>1 (2≤q≤5) via Cartesian-product GH grid.
**→ P11/INLA-lite** — Grid integration of σ_rfx replacing P6 MAP β.

  Root cause of FFX failure at medium+ scale: IRLS underdetermined at large d, P6 anchors β
  to the PQL point estimate of σ_rfx. INLA wins because it marginalizes over σ_rfx.

  Design:
  - After `refineBernoulliNagqSrfx` (P5/P8), build a uniform log-space grid of σ_rfx values
    centered on the nAGQ estimate ± 2.5 log-units. Grid size: K_per_dim^{q_act} with
    K={1:15, 2:7, 3:5, 4:4, 5:3} to keep total nodes ≤300.
  - For each grid point k: run penalized IRLS with σ_rfx fixed (`_pqlPass(fixed_psi=True)`)
    to get MAP (β̂_k, b̂_k). Evaluate Laplace LML:
      log p(y|σ_rfx_k) = Σ_g { ℓ_g(β̂, b̂_g) − ½ b̂_g'Ψ⁻¹b̂_g − ½ log|Ψ| − ½ log|H_g| }
    where H_g = ZWZ_g + Ψ⁻¹.
  - Weight: w_k = softmax(LML_k + log p_prior(σ_rfx_k)).
  - Return: β_marg = Σ_k w_k β̂_k, σ_rfx_marg = Σ_k w_k σ_rfx_k.
  - Final: 3-Newton BLUP refresh at (β_marg, σ_rfx_marg) + BC1 M-step.
  - `fixed_psi=True` in `_pqlPass` skips the internal Ψ M-step so the mode is found
    under the grid Ψ exactly (matches INLA's per-hyperparameter mode-finding).
  - Replaces `refineBernoulliMapBeta` in the call chain; nAGQ still runs first for grid center.
  - Chunked over K to bound memory: chunk_k = max(1, 64//n_elig) grid points at a time.

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

**R-INLA:** `inla()` (R-INLA package), INLA Laplace approximation. Uncorrelated
datasets (eta_rfx=0): independent `f(group, model='iid')` per RE dimension with PC
prior P(σ>τ_rfx)=0.317. Correlated (eta_rfx>0, q=2): `iid2d` + copy with Wishart
prior matching HalfNormal(τ_rfx) marginals. FE prior: N(ν_ffx, τ_ffx²) via
control.fixed. Script: `glmm_inla_comparison.py`.
Full results in `experiments/analytical/glmm_inla_results.md`.

Matched comparison — n_inla=1000, n_total=1000. Bold = winner per cell.
† medium-b-sampled INLA σ outlier driven by 4th quartile instability (RMSE=5.5).

| Dataset           | part  | PQL FFX   | INLA FFX  | PQL σ     | INLA σ    | PQL BLUP  | INLA BLUP |
| ---               | ---   | ---:      | ---:      | ---:      | ---:      | ---:      | ---:      |
| small-b-mixed     | train | **0.271** | 0.451     | **0.571** | 0.567     | **0.618** | 0.618     |
| small-b-sampled   | test  | **0.301** | 0.447     | **0.575** | 0.556     | **0.621** | 0.625     |
| medium-b-mixed    | train | 1.782     | **0.331** | 0.835     | **0.519** | 1.150     | **0.648** |
| medium-b-sampled  | test  | **0.345** | 0.400     | **0.651** | 4.490 †   | **0.707** | 0.692     |
| large-b-mixed     | train | 2.501     | **0.323** | 0.723     | **0.521** | 0.974     | **0.676** |
| large-b-sampled   | test  | 1.811     | **0.365** | 0.879     | **0.603** | 0.953     | **0.710** |
| huge-b-mixed      | train | 1.043     | **0.330** | 1.016     | **0.550** | 1.070     | **0.713** |
| huge-b-sampled    | test  | **0.385** | 0.394     | 0.791     | **0.579** | 0.835     | **0.740** |

Key findings (n_inla=1000 per split, 2026-05-15):
- **FFX**: PQL wins only at small scale. INLA dominates from medium onward: 5–8×
  better on mixed (high-d train) splits, 5× at large-sampled, near-tied at
  medium-sampled (0.345 vs 0.400) and huge-sampled (0.385 vs 0.394). Reversal
  driven by d: more covariates make PQL's IRLS underdetermined; INLA's marginal
  Laplace scales better.
- **σ_rfx**: INLA better at medium+ (except medium-sampled outlier). Quartile
  pattern consistent across scales: both methods overshoot low σ, PQL undershoots
  high σ more severely.
- **BLUP**: INLA better at medium+ scale (0.65–0.74 vs 0.71–1.15 PQL). Tied at
  small. Tracks FFX improvement.
- **Speed**: PQL 20–41 ms/ds vs INLA 4.3–5.0 s/ds (≈100–200× faster at medium+).

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
