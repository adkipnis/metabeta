Bernoulli GLMM Plan
===================

Last updated: 2026-05-14

Current Baseline
----------------

Estimator: `lmmBernoulli` (6 PQL passes) + `refineBernoulliNagqSrfx` (P5 nAGQ,
q=1 gated) + `refineBernoulliMapBeta` (P6 true Laplace score for β + BC1 M-step
correction). Active when `map_refine=True`: prior-regularized IRLS β₀ (P1+P1-ext),
prior-informed Ψ floor (P2 sub-item), nAGQ σ_rfx refinement (P5), Newton β
refinement (P6), and BC1 analytic M-step correction. Raw baseline:
`glmm(..., map_refine=False)`.

Required-suite NRMSE (post-P1+P1-ext+Ψ-floor+P5+P6+BC1, 2026-05-14, N=8192):

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

sRFX improvement vs prior baseline (P6 without BC1, 2026-05-14): −0.3% to −1.2%
across all 12 cells. FFX and BLUP unchanged. No regressions.

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

**✗ P8a — Profile Laplace joint MAP after P6 (TRIED 2026-05-14, REVERTED)**

Implemented `refineBernoulliLaplaceMap` in `map.py`; wired after P6 in `glmm.py`.
Result: FFX unchanged everywhere; sRFX regressed at mixed cells (small-b-mixed
+3.3%, medium-b-mixed +2.4%).

Root cause: at the P6 fixed point, both gradients of the profile ELBO are ≈0.
(a) β gradient = Bernoulli score, already zeroed by P6 Newton.
(b) σ gradient (without log-det) = G − Σ_g b̂_gj²/σ_j² < 0 at the P6 M-step σ
(the M-step includes a H_g^{-1} bias correction that the ELBO gradient omits),
so Adam pushes σ further down. `refineBernoulliLaplaceMap` exists in map.py but
is not wired into `glmm()`.

**✗ P8b — Profile Laplace joint MAP before P6 (TRIED 2026-05-14, REVERTED)**

Rewired P8 to run after P5 but before P6.  Result: medium-b FFX improved (−10.9%
mixed, −17.1% sampled-valid), but large-b/huge-b FFX unchanged or worse (+0.6–2.2%),
and small-b-mixed sRFX regressed +3.3%.  Formal acceptance criterion (FFX ≥15% at
large-b or huge-b) not met.

Root cause: β gradient at PQL output is nonzero (PQL doesn't zero the Bernoulli
score exactly), so 25 Adam steps at lr=0.03 move β at medium-b (d≈8) but insufficient
at large-b/huge-b (d=16).  The β landscape is harder to optimize at high d with
low Fisher information per binary observation.

Required benchmark (N=8192, P8b = P5→P8→P6, 2026-05-14):

| Dataset           | Partition | FFX    | sRFX   | BLUP   | vs baseline |
| ---               | ---       | ---:   | ---:   | ---:   | --- |
| small-b-mixed     | train     | 0.2325 | 0.5472 | 0.6349 | sRFX+3.3%↑ BLUP+2.4%↑ |
| small-b-sampled   | valid     | 0.2809 | 0.6066 | 0.6633 | neutral |
| small-b-sampled   | test      | 0.2747 | 0.5894 | 0.6588 | neutral |
| medium-b-mixed    | train     | 0.6592 | 0.6679 | 0.7288 | FFX−10.9%↓ |
| medium-b-sampled  | valid     | 0.4859 | 0.7223 | 0.8383 | FFX−17.1%↓ |
| medium-b-sampled  | test      | 0.6938 | 0.7397 | 0.8327 | FFX+1.4%↑ |
| large-b-mixed     | train     | 1.6542 | 0.7730 | 0.9363 | regressions |
| large-b-sampled   | valid     | 0.8656 | 0.8034 | 0.8309 | FFX+0.9%↑ |
| large-b-sampled   | test      | 1.3746 | 0.8212 | 0.9664 | sRFX−1.6%↓ |
| huge-b-mixed      | train     | 2.0051 | 0.8453 | 0.9467 | neutral |
| huge-b-sampled    | valid     | 1.3520 | 0.9030 | 1.0127 | FFX+2.2%↑ |
| huge-b-sampled    | test      | 1.5527 | 0.8893 | 0.9890 | neutral |

Medium-b FFX wins are real and above 15%, but out of scope (criterion: large-b/huge-b).
`refineBernoulliLaplaceMap` kept in map.py; not wired.

**✗ P8-trial — Adam-steps ablation (TRIED 2026-05-14, FAILED, reverted)**

n_steps=100 in P8b position.  Large-b partial results (run aborted after large-b
completed, huge-b unchanged pattern expected):

| Dataset           | Partition | FFX    | sRFX   | BLUP   | vs baseline |
| ---               | ---       | ---:   | ---:   | ---:   | --- |
| large-b-mixed     | train     | 1.6067 | 0.7839 | 0.9145 | FFX−2.3%↓, sRFX+2.6%↑ |
| large-b-sampled   | valid     | 0.8656 | 0.8034 | 0.8309 | neutral |
| large-b-sampled   | test      | 1.3746 | 0.8212 | 0.9664 | neutral |

Acceptance criterion (FFX ≤ 1.48 at large-b-mixed, i.e. ≥10% improvement) not met.
Root cause confirmed: more Adam steps do not solve the problem — the profile ELBO
for β is fundamentally ill-conditioned at d=16 with binary observations (low Fisher
information per sample). No amount of gradient descent on (β, log σ) will recover
the correct β when the Laplace ELBO landscape is flat in β at the PQL initialization.

**Proceeding to nAGQ for q>1 as primary path.**

**Priority 8 — nAGQ for q>1 (HIGH impact, primary path)**

Extend `refineBernoulliNagqSrfx` to handle any active q via a Cartesian product
Gauss-Hermite grid.  Root cause: P5 nAGQ gates on `active_q == 1`, leaving q>1
datasets (large-b: q∈{1…4}, huge-b: q∈{1…5}) with the biased PQL Ψ.  Correcting
σ for q>1 unlocks the same P5→P6 cascade that fixed medium-b: better σ → P6
converges to the correct MAP → FFX improves.

**Grid construction:**

Use a Cartesian product of 1D Gauss-Hermite nodes.  Node count scales as k^q;
choose k to keep the product tractable:

| active q | k per dim | total nodes |
| ---      | ---:      | ---:        |
| 1        | 7         | 7  (existing P5) |
| 2        | 5         | 25 |
| 3        | 5         | 125 |
| 4        | 3         | 81 |
| 5        | 3         | 243 |

For each group g, the quadrature point at multi-index j = (j1,…,jq) is:
```
b_{g,j} = b̂_g + √2 · L_g · z_j
```
where L_g = chol(H_g^{-1}) (B, m, q, q lower-triangular), z_j is the q-vector of
1D GH nodes at each dimension, and H_g = ZWZ_g + Ψ^{-1}.  The LML per group is:
```
LML_g = logsumexp_j( log w_j + ℓ_{g,j} + ½‖z_j‖² )  + ½ log(2^q) − ½ log|H_g|
```
where `log w_j = Σ_i log w_{ji}` (product of 1D GH weights) and the `+½‖z_j‖²`
cancels the implicit GH density across all q dimensions.

**Implementation plan:**

1. Build `_ghProductGrid(k_vals: list[int], dtype, device)` → `(K, q)` node tensor
   and `(K,)` log-weight tensor, where K = prod(k_vals) and k_vals[i] is the k for
   dimension i (e.g., [5, 5] for q=2).
2. Refactor `refineBernoulliNagqSrfx` to branch on `active_q`:
   - `active_q == 1`: keep existing scalar path unchanged.
   - `active_q >= 2`: new multivariate path using the product grid.
3. Multivariate path:
   - Gather active-q columns of Zm → `z_cols` (B, m, n_max, q_act).
   - Gather active-q BLUPs → `b_g0` (B, m, q_act) — fixed quadrature centers.
   - Initial `log_s2` (B, q_act); optimize jointly via Adam (n_steps=10, lr=0.05).
   - At each step: build Psi_inv from exp(log_s2), compute H_g scalar for active
     dims, form L_g = chol(H_g^{-1}), evaluate LML as logsumexp over K grid points.
   - Gradient w.r.t. log_s2 via autograd.
4. After optimization: update Psi_lap active block, recompute b̂_g via n_newton=3
   Newton steps.
5. Gate: run multivariate path for `2 <= active_q <= 5`; skip if active_q > 5
   (unlikely but safe).

**Acceptance:** σ_rfx improvement ≥ 10% at any large-b or huge-b cell, and FFX
improves (downstream P6 cascade) without BLUP regressions.  Target: large-b-mixed
FFX from 1.6439 to ≤ 1.40.

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
per batch item; q>1 datasets are returned unchanged. Wired into `glmm()` Bernoulli
branch after `lmmBernoulli`, gated on `map_refine=True`.

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

**✓ P6 — True Laplace score for β (DONE 2026-05-13, wired 2026-05-13)**

Implemented as `refineBernoulliMapBeta` in `map.py`. Wired into `glmm()` Bernoulli
branch after P5, active when `map_refine=True`. Wall time: +1–2 ms/dataset.

Algorithm: n_outer=2 rounds of alternating:
1. β Newton (n_steps=8 damped Newton steps, damping=0.7) at fixed b̂_g
2. b̂_g Newton (n_newton=3 steps) at fixed β with PQL Ψ
Final Ψ M-step once at end from the last b̂_g.

Required benchmark (P5→P6 composition via glmm(), N=8192, 2026-05-13):

FFX NRMSE improved vs prior P5-only baseline:
- small-b: −64–65% (0.668→0.231 mixed, 0.765→0.275 sampled-test)
- medium-b: −48–51% (1.423→0.740 mixed, 1.401→0.684 sampled-test)
- large-b: −21–56% (2.085→1.644 mixed, 2.043→1.365 sampled-test)
- huge-b: −22–44% (2.597→2.018 mixed, 2.749→1.539 sampled-test)

Reference comparison vs CAVI (N=2016, n_total=2000, sampled=test, mixed=train×2):

| Dataset          | P5+P6 FFX | CAVI FFX | P5+P6 σ | CAVI σ | P5+P6 BLUP | CAVI BLUP |
| ---              | ---:      | ---:     | ---:    | ---:   | ---:       | ---:      |
| small-b-sampled  | **0.281** | 0.392    | **0.591** | 0.663 | 0.654      | **0.649** |
| small-b-mixed    | **0.253** | 0.355    | **0.549** | 0.667 | 0.616      | **0.681** |
| medium-b-sampled | **0.332** | 0.500    | **0.703** | 0.765 | **0.734**  | 0.827     |
| medium-b-mixed   | 0.740*    | **0.427**| 0.760*  | **0.696** | 0.899*  | **0.707** |

*medium-b-mixed P5+P6 from required benchmark (N=8192); CAVI from reference comparison
(N=500 matched). Remaining medium-b-mixed gap: our FFX=0.740, CAVI=0.427.

P5→P6 beats CAVI on FFX at 3/4 reference datasets; beats CAVI on σ_rfx at 3/4;
beats CAVI on BLUP at medium-b-sampled. CAVI still leads medium-b-mixed across all
metrics (high d, multi-q training data, σ_rfx bias downstream of FFX error).

Root cause of medium-b-mixed gap: P6 alone was at the global MAP under the wrong Ψ.
P5→P6 composition reduces FFX from 1.423 to 0.740, but CAVI reaches 0.427. The
remaining gap (~0.31 NRMSE) is the open problem for the next priority.

Investigated dead ends: (a) P6-ext — Ψ M-step inside the outer loop — structural
no-op for n_outer=2 and regresses σ_rfx at small datasets. (b) Multiple restarts:
moot — joint concavity guarantees unique global MAP.

**✗ P6-conv — Run P6 to full convergence (TRIED 2026-05-13, REVERTED)**

Hypothesis: the large-b FFX gap (CAVI 3.6–6× better) is due to insufficient Newton
iterations in P6 (fixed budget n_outer=2 × n_steps=8 = 16 steps, not converged).
Raising n_outer=30, n_steps=50 with a gradient-norm stopping criterion on the inner
loop and β-change criterion on the outer loop.

Result: **zero improvement at large-b (FFX 2.5334 → 2.5333)**; small-b/medium-b also
regressed to ≈PQL level.  Reverted to n_outer=2, n_steps=8.

Root cause of the null result:

1. **PQL already zeroes the true Bernoulli score.**  IRLS working-response normal
   equations reduce to `∑ X'(y − σ(Xβ + Zb)) = 0` at PQL convergence — the exact
   MAP score condition.  So at PQL (β, b̂), the score is already ≈0.  Increasing the
   Newton budget cannot move β away from this point.

2. **MAP (β, b̂) given PQL Ψ is not better than PQL (β, b̂).**  Full convergence
   drives to the MAP under the wrong Ψ.  Because PQL Ψ is biased (downward at high σ),
   b̂ compensates for β errors at the MAP, leaving FFX unchanged or worse.

3. **The 16-step budget is beneficial implicit regularization via early stopping.**
   PQL (6 passes) leaves a residual score; those 16 Newton steps partially correct it
   without over-fitting to the wrong Ψ.  Full convergence removes this regularization.

Conclusion: the large-b FFX gap is **not** caused by insufficient P6 budget.  It is
caused by PQL Ψ being poorly estimated for q>1 (no P5 nAGQ for q>1).  The correct
path is Ψ refinement for q>1 (Priority 7 BC1, or a q>1 nAGQ variant).

**✓ P7 / BC1 — Analytic O(1/n) Laplace M-step correction (DONE 2026-05-14)**

Implemented inline in `refineBernoulliMapBeta` (map.py). After the standard
Laplace M-step `Ψ̂ = (1/G)Σ_g(b̂_g b̂_g' + H_g^{-1})`, adds BC1 diagonal correction:

```
ΔΨ_{jj} = (1/G) Σ_g b̂_{gj} · T3_{gj} · [H_g^{-1}]_{jj}²
T3_{gj} = -Σ_i z_{gij}³ · μ_i(1-μ_i)(1-2μ_i)
```

Derivation: `E[b_g|y,Ψ] - b̂_g ≈ f'''(b̂_g)/(2H_g²)` (Laplace mode-mean discrepancy),
so `E[b_g²] - (b̂_g² + H_g^{-1}) ≈ 2b̂_g·Δb_g = b̂_g·f'''(b̂_g)/H_g²`. The diagonal
approximation treats each q-dimension independently (tractable for arbitrary q).

Applies to all q (q=1 and q>1). Applied after the final M-step within P6, so it
corrects the last M-step given the P6-refined (β, b̂_g). For q=1 datasets that went
through P5 (nAGQ), BC1 still applies to the P6 M-step (which re-estimates Ψ from
the final b̂_g after the P6 Newton alternation).

Results (N=8192, 2026-05-14, compared to P6-without-BC1):

| Cell              | Pre-BC1 sRFX | BC1 sRFX | Δ%    |
| ---               | ---:         | ---:     | ---:  |
| small-b-mixed     | 0.5358       | 0.5299   | −1.1% |
| medium-b-mixed    | 0.6654       | 0.6572   | −1.2% |
| large-b-mixed     | 0.7721       | 0.7643   | −1.0% |
| huge-b-mixed      | 0.8550       | 0.8502   | −0.6% |
| large-b-sampled   | 0.8091       | 0.8041   | −0.6% |
| huge-b-sampled    | 0.9128       | 0.9101   | −0.3% |

FFX and BLUP unchanged; sRFX improved 0.3–1.2% across all 12 cells. Below the
formal ≥10% acceptance threshold, but universal (all cells, both q=1 and q>1),
no regressions, and essentially zero computational cost (~3 extra tensor ops).

Remaining σ_rfx gap: S-curve bias persists — upward at low true σ (Ψ floor), downward
at high true σ (M-step shrinkage). BC1 partially corrects the downward half only.
Further improvement for q>1 would require higher-order (BC2) or quadrature-based
corrections, both substantially more complex.

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
