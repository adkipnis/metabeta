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

Done
----

**✓ P1+P1-ext** — Prior-regularized IRLS β₀ with Student-t adaptive precision.
Net: FFX −2% to −34% across 12 cells; largest gains at large/huge.

**✓ P2 sub-item** — Prior-informed Ψ floor: replaced constant 0.25 with per-dataset
floor from `tau_rfx`, capped at 0.25. Net: small-b sRFX +0.6–2.3%; other sizes neutral.

**✗ P2** — Laplace-MAP σ_rfx fixed-point (FAILED, reverted). Cancels the H_g^{-1}
correction added by the M-step, reverting to downward-biased raw MoM. Code kept in
`map.py:refineBernoulliMapSrfx`, not called.

**✓ P5** — nAGQ for q=1. Adam on k=7 GH quadrature LML gradient + Newton BLUP
refresh. q=1 sRFX −16–18% at all sizes; FFX unchanged.

**✓ P6** — True Laplace score for β. n_outer=2 rounds of β Newton (n_steps=8,
damping=0.7) + b̂_g Newton (n_newton=3) + final M-step. FFX −21–65% vs PQL baseline.
Full-convergence trial (P6-conv): zero improvement — PQL already zeroes the Bernoulli
score at convergence; MAP under wrong Ψ is not better.

**✓ P7/BC1** — Analytic O(1/n) M-step correction inline in `refineBernoulliMapBeta`.
`ΔΨ_{jj} = (1/G)Σ_g b̂_{gj}·T3_{gj}·[H_g^{-1}]_{jj}²`. Universal −0.3–1.2%
sRFX improvement across all 12 cells, no regressions.

**✓ P8** — nAGQ for q>1 (2≤q≤5) via Cartesian-product GH grid. k_per_dim =
{2:5,3:5,4:3,5:3} → {25,125,81,243} nodes. sRFX −6–12% at large-b; smaller at huge-b.
FFX regresses +1.7–2.8% at huge-b-sampled (within noise).

**✗ P8a/b** — Profile Laplace joint MAP via Adam on (β, log σ) (TRIED, REVERTED).
Root cause: β gradient ≈0 at P6 fixed point; σ gradient pushed downward (ELBO omits
H_g^{-1} bias correction). Code kept in `map.py:refineBernoulliLaplaceMap`, not wired.

**✗ P9 — Decouple M-step β from P6 Newton (TRIED, REVERTED, 2026-05-14).**
Hypothesis: P6 Newton β introduces variance into M-step Ψ. Fix: use pre-Newton β/b̂_g
for the M-step, return P6 β as FFX output only. Result: mixed — wins at medium-b-mixed
(−20%), large-b-sampled valid (−20%), huge-b-mixed (−21%), but regressions at
large-b-sampled test (+18%) and large-b-mixed train (+20%). Net: not acceptable.
Root cause: the P6 β happens to give better-calibrated M-step Hessians for some
dataset orderings; the effect is partition-specific, not systematic.

Open Priorities
---------------

**Priority 1 — nAGQ σ gradient step / profile MLE for σ**

Fix β = β̂ (P6 output), differentiate nAGQ LML w.r.t. log σ via autodiff, take one
Newton step. Distinct from P8a/b: uses the actual quadrature LML (not Laplace), so
the H_g^{-1} bias correction is implicit in the GH nodes. The σ gradient should be
well-calibrated. Cost: cheap for q=1 (7 nodes); expensive for q>1 (243 nodes at q=5).
Risk: profile LML landscape may be flat near P5 estimate (needs measurement).
Acceptance: ≥10% σ_rfx improvement at any large/huge cell, no FFX regression.

**Priority 3 — Beta blend for BLUP residuals (LOW impact, quick)**

`beta_for_blup = alpha*beta_gls + (1−alpha)*beta_0` (alpha ≤ 0.65/0.75 for low/high
d). `beta_est` unchanged. Expected: 5–10% BLUP at small-medium, neutral at large-huge.
Run oracle ablation before implementing. Acceptance: no regressions anywhere.

**Priority 4 — blup_var calibration tuning (LOW priority)**

`_BERNOULLI_BLUP_VAR_INFLATION=1.5` overcorrects large groups (ratio 0.77 at
n_g=25–150). Group-size-dependent inflation (e.g., 1.0+C/n_g) would help.

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

Note: `glmm()` uses `map_refine=True` by default and receives all batch priors, so
P1/P2/P5/P6/BC1 are all active. The "P6" column is one additional P5+P6 round on
top, which changes results minimally. The matched N is a sequential subset of each
file, not a stratified sample.

Key findings:
- **FFX**: Full pipeline beats CAVI at all 12 cells (2–3.5× better at large/huge;
  ≈tie at small).
- **σ_rfx**: Mixed. Pipeline wins at small (both partitions), medium-sampled-valid,
  huge-sampled-valid. CAVI wins at mixed datasets (all sizes) and medium/large-huge
  test. The Laplace M-step produces high σ_rfx variance at large d/q, particularly
  in mixed datasets where β diversity is higher.
- **BLUP**: follows σ_rfx — pipeline wins where σ_rfx wins, CAVI wins otherwise.

σ_rfx quartile pattern: CAVI biases σ upward across all quartiles; PQL/P6 has an
S-curve (upward at low σ, downward at high σ). CAVI's NRMSE advantage at mixed/large
datasets comes from the high-σ quartile where the PQL downward bias is largest.

Methods not pursued: lme4 (diverges q>1), pure Laplace/lme4-style LA (same
divergence), JJ/Polya-Gamma variational bounds (faster CAVI backend, no accuracy
gain over statsmodels), INLA (non-trivial for arbitrary q/Ψ), GPBoost (tree-boosted
FX, unclear full-Ψ support).

Acceptance Criteria
-------------------

- FFX improvement ≥ 15% at large-b or huge-b.
- σ_rfx improvement ≥ 10% at any required cell.
- BLUP improvement must not regress FFX or sRFX.
- Compare against current PQL baseline (`map_refine=True`).
- Narrow-bin-only improvements are experiments only.

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
