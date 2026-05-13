Bernoulli GLMM Plan
===================

Last updated: 2026-05-13

Current Baseline
----------------

Estimator: `lmmBernoulli` (6 PQL passes) + `refineBernoulliNagqSrfx` (P5 nAGQ,
q=1 gated). Active when `map_refine=True`: prior-regularized IRLS β₀ (P1+P1-ext),
prior-informed Ψ floor (P2 sub-item), and nAGQ σ_rfx refinement (P5). Raw
baseline: `glmm(..., map_refine=False)`.

Required-suite NRMSE (post-P1+P1-ext+Ψ-floor+P5-nAGQ, 2026-05-13):

| Dataset           | Partition | FFX    | sRFX   | BLUP   |
| ---               | ---       | ---:   | ---:   | ---:   |
| small-b-mixed     | train     | 0.6682 | 0.5573 | 0.6209 |
| small-b-sampled   | valid     | 0.9844 | 0.6227 | 0.6688 |
| small-b-sampled   | test      | 0.7652 | 0.6098 | 0.6588 |
| medium-b-mixed    | train     | 1.4230 | 0.6737 | 0.7322 |
| medium-b-sampled  | valid     | 1.3452 | 0.7431 | 0.8255 |
| medium-b-sampled  | test      | 1.4008 | 0.7597 | 0.8325 |
| large-b-mixed     | train     | 2.0853 | 0.7690 | 0.9481 |
| large-b-sampled   | valid     | 1.9543 | 0.8221 | 0.8379 |
| large-b-sampled   | test      | 2.0433 | 0.8326 | 0.9667 |
| huge-b-mixed      | train     | 2.5970 | 0.8538 | 0.9522 |
| huge-b-sampled    | valid     | 2.7898 | 0.9102 | 1.0284 |
| huge-b-sampled    | test      | 2.7488 | 0.8975 | 1.0029 |

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

**✓ P5 — nAGQ for q=1 (DONE 2026-05-13)**

Implemented as `refineBernoulliNagqSrfx` in `map.py`. Gates on `active_q.sum() == 1`
per batch item; q>1 datasets are returned unchanged. Called explicitly after
`lmmBernoulli` in the reference comparison; not yet wired into `glmm()`.

Algorithm: k=7 Gauss-Hermite quadrature of the group marginal log-likelihood
∂(ΣgLML_g)/∂(log σ²) via n_steps=10 Adam steps (lr=0.1) at fixed β and fixed
b̂_g centers. After the gradient step, b̂_g is recomputed via n_newton=3 Newton
steps under the refined Ψ. GH nodes are standard Hermite (np.polynomial.hermite.hermgauss);
the `+z_j²` term in the logsumexp cancels the implicit Gaussian weight.

Results (N=2016, 2026-05-13):

| Dataset          | PQL σ  | P5 σ   | Δ%    | PQL BLUP | P5 BLUP |
| ---              | ---:   | ---:   | ---:  | ---:     | ---:    |
| small-b-sampled  | 0.6773 | 0.6099 | −9.9% | 0.686    | 0.6588  |
| small-b-mixed    | 0.6333 | 0.5764 | −9.0% | 0.6408   | 0.6186  |
| medium-b-sampled | 0.7688 | 0.7296 | −5.1% | 0.7532   | 0.7445  |
| medium-b-mixed   | 0.8078 | 0.7484 | −7.4% | 0.9517   | 0.9481  |
| large-b-sampled  | 0.8792 | 0.8500 | −3.3% | 0.8511   | 0.8410  |

FFX is unchanged (P5 does not touch β). BLUPs improve at small/medium; neutral
at large. No regressions anywhere.

q=1-only NRMSE (acceptance criterion cell):

| Dataset          | PQL q=1 σ | P5 q=1 σ | Δ%     |
| ---              | ---:      | ---:     | ---:   |
| small-b-sampled  | 0.6519    | 0.5349   | −17.9% |
| medium-b-sampled | 0.7763    | 0.6430   | −17.2% |
| large-b-sampled  | 0.8709    | 0.7271   | −16.5% |

Acceptance criterion met: ≥10% at large-b-sampled q=1 cells (achieved 16.5%)
with no FFX or BLUP regressions.

Wired into `glmm()` Bernoulli branch (after `lmmBernoulli`, gated on `map_refine`).
Required benchmark confirmed (N=8192 per cell, 2026-05-13).

**P5 → P6 composition (confirmed 2026-05-13):** Running P5 then P6 on medium-b-mixed
closes the long-standing FFX gap: PQL=2.132 → P6=1.242 → P5+P6=0.308 (vs CAVI=0.327).
The improved Ψ from P5 breaks the P6 stall exactly as predicted.

**✓ P6 — True Laplace score for β (DONE 2026-05-13)**

Implemented as `refineBernoulliMapBeta` in `map.py`. Called explicitly after
`lmmBernoulli` in the reference comparison; not yet wired into `glmm()`.

Algorithm: n_outer=2 rounds of alternating:
1. β Newton (n_steps=8 damped Newton steps, damping=0.7) at fixed b̂_g
2. b̂_g Newton (n_newton=3 steps) at fixed β with PQL Ψ
Final Ψ M-step once at end from the last b̂_g.  Wall time: +1–2 ms/dataset.

Results (sampled=test / mixed=train×2, n_total=2000, 2026-05-13, N=2016):

| Dataset          | PQL FFX | P6 FFX   | CAVI FFX | PQL σ | P6 σ  | CAVI σ | PQL BLUP | P6 BLUP | CAVI BLUP |
| ---              | ---:    | ---:     | ---:     | ---:  | ---:  | ---:   | ---:     | ---:    | ---:      |
| small-b-sampled  | 0.720   | **0.284**| 0.329    | 0.677 | 0.652 | **0.644** | 0.686 | 0.684   | **0.637** |
| small-b-mixed    | 0.782   | **0.256**| 0.283    | 0.633 | **0.605** | 0.614 | 0.641 | 0.640   | **0.601** |
| medium-b-sampled | 1.487   | **0.333**| 0.419    | 0.769 | 0.748 | **0.705** | 0.753 | 0.753   | **0.717** |
| medium-b-mixed   | 2.132   | 1.242    | **0.327**| 0.808 | 0.794 | **0.646** | 0.952 | 0.923   | **0.649** |

P6 beats CAVI on FFX for 3/4 datasets (2–4.5× improvement over PQL).
medium-b-mixed FFX gap persists (1.24 vs 0.327).

σ_rfx: P6 beats PQL uniformly (all quartiles, 2–6% RMSE reduction).
P6 beats CAVI on small-b-mixed; CAVI still leads at medium where σ_rfx is
downstream of the FFX gap.  BLUP: marginal improvement over PQL (<1%).

Root cause of medium-b-mixed gap (confirmed resolved 2026-05-13): The Bernoulli
joint log-posterior in (β, b̂_g) is globally concave for fixed Ψ, so P6 alone
was already at the global MAP but under a biased Ψ. The gap was entirely a Ψ
estimation problem. P5 (nAGQ) corrects the Ψ bias; running P5 → P6 achieves
0.308 on medium-b-mixed (vs PQL=2.132, P6-alone=1.242, CAVI=0.327).

Investigated dead ends: (a) P6-ext — Ψ M-step inside the outer loop — structural
no-op for n_outer=2 and regresses σ_rfx at small datasets. (b) Multiple restarts:
moot — joint concavity guarantees unique global MAP.

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
    --data-ids small-b-sampled,medium-b-sampled \
    --partition test --n-cavi 0 --n-total 2000
uv run python experiments/analytical/glmm_reference_comparison.py \
    --data-ids small-b-mixed,medium-b-mixed --partition train --n-epochs 2 \
    --n-cavi 0 --n-total 2000
# add --n-cavi 2000 to also run CAVI (slow: ~60-240 ms/dataset)
uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
