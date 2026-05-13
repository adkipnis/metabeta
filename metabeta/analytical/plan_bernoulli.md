Bernoulli GLMM Plan
===================

Last updated: 2026-05-13

Current Baseline
----------------

Estimator: `lmmBernoulli` (6 PQL passes). Active when `map_refine=True`: prior-
regularized IRLS β₀ (P1+P1-ext) and prior-informed Ψ floor (P2 sub-item). Raw
baseline: `glmm(..., map_refine=False)`.

Required-suite NRMSE (post-P1+P1-ext+Ψ-floor, 2026-05-13):

| Dataset           | Partition | FFX    | sRFX   | BLUP   |
| ---               | ---       | ---:   | ---:   | ---:   |
| small-b-mixed     | train     | 0.6682 | 0.6134 | 0.6369 |
| small-b-sampled   | valid     | 0.9844 | 0.6854 | 0.6854 |
| small-b-sampled   | test      | 0.7652 | 0.6742 | 0.6839 |
| medium-b-mixed    | train     | 1.4230 | 0.7142 | 0.7387 |
| medium-b-sampled  | valid     | 1.3452 | 0.7846 | 0.8497 |
| medium-b-sampled  | test      | 1.4008 | 0.8018 | 0.8442 |
| large-b-mixed     | train     | 2.0853 | 0.8000 | 0.9482 |
| large-b-sampled   | valid     | 1.9543 | 0.8491 | 0.8463 |
| large-b-sampled   | test      | 2.0433 | 0.8651 | 0.9752 |
| huge-b-mixed      | train     | 2.5970 | 0.8796 | 0.9497 |
| huge-b-sampled    | valid     | 2.7898 | 0.9393 | 1.0304 |
| huge-b-sampled    | test      | 2.7488 | 0.9263 | 1.0016 |

Root cause summary (`glmm_error_analysis.py`):
- **FFX** is the dominant failure mode; NRMSE scales with d (low Fisher information
  per binary observation, pooled IRLS underdetermined at large d / low n).
- **σ_rfx** has bidirectional S-curve bias: upward at low true σ (Ψ floor overshoots),
  downward at high true σ (M-step shrinkage). CAVI wins mainly in the high-σ quartile.
- **BLUPs** track FFX: bad β contaminates the BLUP residual ỹ−Xβ.

Closed / Done
-------------

**✓ P1+P1-ext — Prior-regularized IRLS β₀ with Student-t adaptive precision (DONE 2026-05-13)**

Added diagonal prior N(ν_ffx, diag(τ²)) to pooled IRLS normal equations. Key
constraints: initialize β from zeros (not ν_ffx — warm-starting destabilizes first
IRLS step); inactive dimensions (τ_ffx==0) get zero precision (not 1e-4 clamp). For
Student-t priors (20% of datasets, df=5), precision is EM-adaptive:
`6.0/(5τ²+(β−ν)²)`, recomputed each iteration. GLS prior in `_pqlPass` was tried
and reverted — shrunk β causes RFX to compensate, inflating Ψ̂_Lap.

Net result vs raw: FFX −2% to −34% across all 12 cells; large/huge largest gains
(−20–34%). One regression: large-b-mixed train BLUP +13% (⚠️); sampled counterparts
improve −19/−20%. Active when `map_refine=True` only.

**✓ P2 sub-item — Prior-informed Ψ floor (DONE 2026-05-13)**

Replaced constant `_BERNOULLI_INITIAL_PSI_FLOOR=0.25` with per-dataset floor derived
from `tau_rfx`, capped at 0.25 so it can only be lowered, never raised. Uncapped
formula raised the floor for most datasets, causing FFX regressions up to +6.5% at
large/huge. Net effect: small-b sRFX improves 0.6–2.3%; other sizes unaffected
(τ_rfx_mean > 0.5 for most, making the cap a no-op there).

**✗ P2 — Laplace-MAP σ_rfx refinement option (a) (FAILED, reverted)**

Fixed-point MAP for σ_rfx using `G·(Ψ_lap−H_mean)` as the sufficient statistic
consistently worsened sRFX at small-b. Root cause: this cancels the H_g^{-1}
correction the M-step added to make the estimate unbiased — it reverts to the
downward-biased raw MoM, and the prior then pushes σ further down. `refineBernoulliMapSrfx`
exists in map.py but is not called. P6 (below) is the correct path.

Open Priorities
---------------

**Priority 3 — Beta blend for BLUP residuals (LOW impact, quick)**

Apply the Normal-path technique to Bernoulli final BLUP residuals:
`beta_for_blup = alpha*beta_gls + (1−alpha)*beta_0` (alpha ≤ 0.65/0.75 for low/high d).
`beta_est` (reported) is unchanged. Expected gain: 5–10% at small-medium, possibly
neutral at large-huge. Run oracle ablation before implementing.

Acceptance: no regressions on any dataset × partition. Small BLUP improvement at
small-b-mixed is sufficient.

**Priority 4 — blup_var calibration tuning (LOW priority)**

`_BERNOULLI_BLUP_VAR_INFLATION=1.5` overcorrects large groups (ratio 0.77 at
n_g=25–150) while marginal at small groups (1.31 at n_g=5–9). A group-size-dependent
inflation (e.g., 1.0+C/n_g) would help. Defer until P5/P6 are stable.

**Priority 5 — nAGQ for q=1 (OPEN)**

For datasets with a scalar random effect (q=1), replace single-Laplace marginal
with k=7 adaptive Gauss-Hermite quadrature to remove the Breslow-Lin downward
bias in Ψ̂. Gates on `active_q.sum() == 1`; not tractable for q>1 (k^q nodes).

After the final PQL Newton loop has found b̂_g and H_g = ZWZ_g + Ψ^{-1}:

    z_j, w_j   : standard GH nodes/weights (k=7, pre-computed once per call)
    σ_g        : H_g^{-0.5}  (scalar curvature scale)
    b_{g,j}    : b̂_g + √2 · σ_g · z_j
    ℓ_{g,j}   : log p(y_g | β, b_{g,j}) + log p(b_{g,j} | Ψ)
    LML_g      : logsumexp_j(w_j + ℓ_{g,j} + z_j²) + log(√2 · σ_g)

Use ∂(ΣgLML_g)/∂(log σ²) for a single gradient step on σ_rfx after the PQL outer
loop, then one final `_pqlPass` with the nAGQ-refined Ψ to recompute BLUPs.
The `+z_j²` term cancels the implicit Gaussian weight in standard GH.

Acceptance: ≥10% σ_rfx NRMSE reduction at large-b-sampled (current ≈0.85) with
no FFX or BLUP regressions, restricted to q=1 cells only.

**Priority 6 — True Laplace score for β (OPEN, highest priority)**

Current PQL updates β via GLS on the linearized working response, which is NOT
the same as the true Laplace score ∂L/∂β = Σ_g X_g'(y_g − μ_g(β, b̂_g)).
At large d / small n the alternating sequence can converge to a biased β–Ψ pair.

After PQL convergence, take gradient steps on (β, log σ_rfx) differentiating
the Laplace log-marginal at fixed (b̂_g, H_g) (envelope-theorem approximation):

    L(β, Ψ) = Σ_g [log p(y_g | β, b̂_g) + log p(b̂_g | Ψ) − ½ log|H_g|]
    ∂L/∂β ≈ Σ_g X_g'(y_g − μ_g(β, b̂_g))   [true Bernoulli score]

Autograd on the Bernoulli log-likelihood at fixed b̂_g gives the gradient cheaply.
The Ψ M-step (= mean_g(b̂_g b̂_g' + H_g^{-1})) is already the correct first-order
condition for the Laplace LML, so Ψ estimation is near-optimal; β is the main lever.

Acceptance: ≥5% FFX NRMSE improvement at large-b or huge-b with no sRFX or BLUP
regressions. If only sRFX improves, the PQL M-step was not at fault.

**Priority 7 — BC1 σ_rfx correction for q>1 (contingent on P6, OPEN)**

Hold until P6 is complete. If σ_rfx gap persists in q>1 datasets after P6, the
Breslow-Lin (1995) BC1 adds an analytic O(1/n) correction to the M-step that
reduces downward bias in the high-σ_rfx quartile. Unlike P5 nAGQ, tractable for
arbitrary q; weaker (first-order only). Do not pursue before P6 — the σ_rfx gap
is largely downstream of FFX error and should cascade when β improves.

σ_rfx bias direction: S-curve — upward bias at low true σ (Ψ floor overshoots),
downward at high true σ (M-step shrinkage). BC1 addresses only the downward half.

External Reference Baseline
----------------------------

**CAVI:** `BinomialBayesMixedGLM` (statsmodels 0.14+), CAVI on mean-field Gaussian,
diagonal Ψ, prior Gaussian on log σ² (vcp_p=4.0). Script: `glmm_reference_comparison.py`.

Measured results (sampled=test / mixed=train×2, n_total=1000–2000, 2026-05-13):

| Dataset          | N    | PQL FFX | CAVI FFX | PQL σ | CAVI σ | PQL BLUP | CAVI BLUP |
| ---              | ---: | ---:    | ---:     | ---:  | ---:   | ---:     | ---:      |
| small-b-sampled  | 1024 | 0.754   | **0.353** | 0.668 | **0.647** | 0.657 | **0.641** |
| small-b-mixed    | 2016 | 0.782   | **0.283** | 0.633 | **0.614** | 0.641 | **0.600** |
| medium-b-sampled | 1024 | 1.256   | **0.445** | 0.770 | **0.718** | **0.743** | 0.752 |
| medium-b-mixed   | 2016 | 2.142   | **0.327** | 0.808 | **0.647** | 0.952 | **0.649** |

σ_rfx RMSE by true σ_rfx quartile:

| Quartile          | PQL Bias   | PQL RMSE  | CAVI Bias  | CAVI RMSE | Winner  |
| ---               | ---:       | ---:      | ---:       | ---:      | ---     |
| Low (≤0.20)       | +0.26–0.28 | 0.30–0.32 | +0.33–0.39 | 0.37–0.47 | **PQL** |
| Med-low (0.20–0.46) | +0.07–0.14 | 0.17–0.44 | +0.22–0.27 | 0.28–0.38 | **PQL** |
| Med-high (0.46–0.91) | −0.10–0.17 | 0.22–0.31 | +0.06–0.11 | 0.23–0.28 | ~tie |
| High (≥0.91)      | −0.43–0.62 | 0.64–0.83 | −0.25–0.39 | 0.46–0.58 | **CAVI** |

Key findings: FFX gap is 2–6.5× (main target for P6). σ_rfx CAVI advantage is driven
by the high-σ quartile and is likely downstream of better β. BLUP gap tracks FFX.
P6 should cascade to both.

**Methods not pursued:**

- **lme4:** ML without regularization diverges at q>1 (σ_rfx NRMSE 33–64 on medium).
  pymer4 removed.
- **Pure Laplace / lme4-style LA:** PQL is already Laplace-based; P6 closes the
  remaining β-linearization gap. Same divergence problem without regularization.
- **JJ / Polya-Gamma variational bounds:** faster CAVI backends; would not improve
  our PQL estimator, only replace the statsmodels reference with a marginally faster
  variant. statsmodels is fast enough at 1000–2016 datasets per data_id.
- **INLA:** accuracy ceiling, but P5 and P6 already target the same improvements
  internally; setup for arbitrary q/unstructured Ψ is non-trivial.
- **GPBoost:** designed for tree-boosted FX; unclear support for full unstructured Ψ
  at arbitrary q.

Acceptance Criteria
-------------------

A Bernoulli change must improve at least one primary metric without material regressions:

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
uv run python experiments/analytical/glmm_reference_comparison.py
uv run python experiments/analytical/glmm_reference_comparison.py \
    --data-ids small-b-sampled,small-b-mixed,medium-b-sampled,medium-b-mixed \
    --partition test --n-cavi 1000 --n-total 1000
uv run python experiments/analytical/glmm_reference_comparison.py \
    --data-ids small-b-mixed,medium-b-mixed --partition train --n-epochs 2 \
    --n-cavi 2000 --n-total 2000
uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
