Bernoulli GLMM Plan
===================

Last updated: 2026-05-15 (P13 experiments)

Current Baseline
----------------

Estimator: `lmmBernoulli` (6 PQL passes) + `refineBernoulliNagqSrfx` (P5/P8 nAGQ,
all q≤5) + `refineBernoulliNestedBeta` (P12 nested β/b̂_g Newton + BC1 M-step).
Active when `map_refine=True`.

Required-suite NRMSE (P1+P1-ext+Ψ-floor+P5+P8+P12+BC1, 2026-05-15, N=8192):

| Dataset           | Partition | FFX    | sRFX   | BLUP   |
| ---               | ---       | ---:   | ---:   | ---:   |
| small-b-mixed     | train     | 0.2290 | 0.4788 | 0.5957 |
| small-b-sampled   | valid     | 0.2788 | 0.5625 | 0.7048 |
| small-b-sampled   | test      | 0.3100 | 0.5567 | 0.6789 |
| medium-b-mixed    | train     | 0.6450 | 0.5662 | 0.6726 |
| medium-b-sampled  | valid     | 0.4515 | 0.6388 | 0.8037 |
| medium-b-sampled  | test      | 0.6565 | 0.6851 | 0.7976 |
| large-b-mixed     | train     | 1.4763 | 0.6797 | 0.8521 |
| large-b-sampled   | valid     | 0.8266 | 0.7422 | 0.7949 |
| large-b-sampled   | test      | 1.3627 | 0.7885 | 0.9380 |
| huge-b-mixed      | train     | 1.9566 | 0.8518 | 0.9427 |
| huge-b-sampled    | valid     | 1.2558 | 0.9136 | 0.9687 |
| huge-b-sampled    | test      | 1.5085 | 0.8554 | 0.9810 |

P6 → P12 delta (P6 was 2 outer rounds of 8β+3b̂_g; P12 is 12 outer × 4 inner steps):
FFX improves 1–23% (medium largest), sRFX improves 2–14%, BLUP improves 1–9%.
Main gains at medium+. small-b-sampled test shows minor FFX regression (within noise).

Root cause summary (`glmm_error_analysis.py`):
- **FFX** is the dominant failure mode; NRMSE scales with d (low Fisher information
  per binary observation, pooled IRLS underdetermined at large d / low n).
- **σ_rfx** has bidirectional S-curve bias: upward at low true σ (Ψ floor overshoots),
  downward at high true σ (M-step shrinkage). CAVI wins mainly in the high-σ quartile.
- **BLUPs** track FFX: bad β contaminates the BLUP residual ỹ−Xβ.

Strategic Direction
-------------------

Goal: fast, high-accuracy Bernoulli summaries for downstream hierarchical NPE context.
The analytical estimator does not need to be a complete posterior engine; it needs stable,
prior-aware point summaries and uncertainty proxies that improve the NPE's conditioning.
Because `glmm()` outputs are already passed to the hierarchical NPE, do not add a separate
amortized-correction branch here; the NPE is the correction mechanism.

Decision: do **not** pursue a full PyTorch INLA implementation as the main branch. A faithful
INLA path would integrate over variance/correlation hyperparameters and repeatedly solve
latent modes. With current regimes (`d≤16`, `q≤5`, `m≤200`, `n_total≤3000`), diagonal Ψ already
needs many hyperparameter evaluations, and full Ψ has up to `q(q+1)/2=15` hyperparameters.
Even with batched small-matrix kernels, that multiplier is unlikely to fit a robust
`~100 ms/dataset` target. R-INLA remains a reference only, not a backend.

Ranked branches, ordered by expected accuracy per implementation risk:

1. **→ P14/single-mode Laplace-EB** — Implement a direct marginal-Laplace empirical-Bayes
   solver, not full INLA and not another PQL patch. Optimize β and log σ on the true
   Bernoulli profile/Laplace objective, with vectorized per-group `q×q` Newton modes
   and priors in the objective. Start diagonal Ψ only, active `q≤5`, and use a σ
   continuation schedule: initialize σ tiny so b̂_g is effectively zero and β learns
   first, then relax/optimize σ. Expected improvement: high for high-d FFX and high-σ
   sRFX; risk: medium. Target runtime: `50–150 ms/dataset` batched CPU/GPU.

   Staging:
   - **→ P14a/objective smoke test** — Implemented as `refineBernoulliLaplaceEb`
     in `map.py` (direct callable, not wired into `glmm()`). Current form uses
     diagonal Ψ, stats-β initialization by default, σ continuation from a tiny cap,
     true Bernoulli Laplace target, prior terms when available, and an objective
     acceptance gate against incoming stats. Smoke tests pass. First-batch checks
     are finite and close on β but not yet better on σ, so this remains experimental.
   - **→ P14b/log-σ continuation tuning** — Implemented 2026-05-15. The underlying
     σ parameter now starts from current stats while the effective σ is capped by
     continuation, and the target includes the log-σ Jacobian so optimization is
     on the log-hyperposterior rather than the zero-mode σ density. Four-batch
     sanity benchmark (N=128/split) improved FFX, sRFX, and BLUP on
     small-sampled-test, medium-mixed-train, large-sampled-test, and
     huge-sampled-test with ~4–10 ms/dataset extra CPU time.
   - **→ P14c/opt-in production path** — Implemented 2026-05-15. `glmm()` now exposes
     P14 behind `bernoulli_laplace_eb=True`; the default Bernoulli path is unchanged.
     The refinement remains batched over the incoming mini-batch, supports optional
     late-stage early stopping, and can return tensor diagnostics
     (`laplace_eb_accept`, `laplace_eb_steps`, target/base-target) when requested via
     `bernoulli_laplace_eb_diagnostics=True`. Focused GLMM tests pass. Superseded
     operationally by P15's explicit auto-gated route.

2. **→ P15/diagnostic fallback gate** — Implemented 2026-05-15. The current hybrid path
   remains the default (`bernoulli_laplace_eb=False`). `bernoulli_laplace_eb=True` still
   routes every Bernoulli dataset through P14, while `bernoulli_laplace_eb='auto'`
   applies P14 only to datasets selected by a simple diagnostic gate:
   effective `d >= 4`, mean estimated `σ_rfx >= 0.75`, or max fitted `|η| >= 8`.
   The thresholds are configurable via `bernoulli_laplace_eb_gate_min_d`,
   `bernoulli_laplace_eb_gate_min_sigma`, and `bernoulli_laplace_eb_gate_eta_abs`;
   setting a threshold to `None` disables that component. Diagnostics add
   `laplace_eb_gate` alongside the P14 accept/step/target tensors, with skipped
   datasets receiving zero accept/step diagnostics. This intentionally excludes
   pooled-Fisher and β-update gates for now: the first production gate should have
   few moving parts and be easy to benchmark. Next decision: matched-subset benchmark
   `False` vs `True` vs `'auto'` against CAVI/INLA references and tune thresholds only
   if the first gate misses clear P14 wins.

   Initial smoke check on the first 128 datasets: small-b-sampled/test selected 27%
   and landed between baseline and full P14 (σ improved, FFX/BLUP roughly flat);
   large-b-sampled/test selected 100% and improved FFX/σ but regressed BLUP on that
   first batch. Do not make `'auto'` the default before the matched benchmark; if the
   BLUP regression persists, prefer a BLUP-specific fallback over adding more gates.

3. **✗ P13/prior-seeded P12 / cold-start** — Tried and reverted (2026-05-15). See P13a/b/c
   entries in the tried section below. Result informs P14: any cold-start route must keep
   σ tiny while β learns; otherwise b̂_g absorbs the fixed-effect signal.

Deprioritized branches:
- **Full PyTorch INLA** — Too many hyperparameter-mode solves for the `~100 ms/dataset` target,
  especially with full Ψ. Use INLA concepts, not full INLA integration.
- **Batched EP / Pólya-Gamma variational GLMM** — Potentially accurate and GPU-friendly, but
  more moving parts than P14. Revisit only if P14 fails on accuracy.
- **More PQL-local patches** — Low expected upside after P13 unless they simplify or stabilize P14.

Implemented (✓) / Tried and reverted (✗) / In progress (→)
--------------------------------------------------------------

**✓ P1+P1-ext** — Prior-regularized IRLS β₀ with Student-t adaptive precision.
**✓ P2 sub-item** — Prior-informed Ψ floor from `tau_rfx`, capped at 0.25.
**✓ P5** — nAGQ for q=1. Adam on k=7 GH quadrature LML + Newton BLUP refresh.
**✓ P6** — True Laplace score for β. n_outer=2 rounds of β Newton + b̂_g Newton + M-step. Superseded by P12.
**✓ P7/BC1** — Analytic O(1/n) M-step correction inline in `refineBernoulliMapBeta` / `refineBernoulliNestedBeta`.
**✓ P8** — nAGQ for q>1 (2≤q≤5) via Cartesian-product GH grid.
**✗ P11/INLA-lite** — Grid integration of σ_rfx with β averaging.

  Implemented and benchmarked (2026-05-15). σ_rfx improved (0.447 vs 0.602 nAGQ) but FFX
  catastrophically regressed (3–4× worse). Root cause: β̂(σ) is monotone in σ (small σ →
  OLS, large σ → within-group); posterior mass is asymmetric (~60% below LML peak). Averaging
  pulls β toward the OLS regime, undoing P6's improvements. Reverted.

  Insight from envelope theorem: ∂LML/∂β = Σ_g X_g'(y_g − μ_g) — the exact Bernoulli score —
  which P6 already computes. P6 and INLA optimize the same β objective. The INLA advantage at
  high d is not about marginalizing σ_rfx; it is about nested optimization (b_g re-optimized
  per β step) vs P6 block coordinate ascent from a poor PQL starting point.

**✓ P12/nested-β** — Nested β/b̂_g Newton: re-converge b̂_g at each outer β step.

  Implemented 2026-05-15 as `refineBernoulliNestedBeta`. Replaces `refineBernoulliMapBeta`
  (P6) in the call chain. n_beta_steps=12 outer β Newton steps, n_inner=4 inner b̂_g Newton
  steps per outer step, n_final=3 final b̂_g steps, damping=0.7. Hessian: XtWX (not Schur —
  Schur correction tried but caused overshoot at large d/σ; XtWX + damping is safer).

  Gains vs P6 (N=8192): FFX 1–23%, sRFX 2–14%, BLUP 1–9%. Largest at medium. Remaining gap
  to INLA at large/huge FFX is structural: PQL initializes from IRLS which underdetermines β
  at high d; nested Newton improves convergence but cannot fix a poor starting basin.

**✗ P2** — Laplace-MAP σ_rfx fixed-point (cancels H_g^{-1} correction). Code in `map.py:refineBernoulliMapSrfx` removed.
**✗ P3** — Beta blend for BLUP residuals (oracle: partition-specific, no globally safe α).
**✗ P4** — blup_var `1+C/n_g` formula (calibration depends on both n_g and G; bookkeeping-only anyway).
**✗ P8a/b** — Joint MAP on (β, log σ) via Adam (β gradient ≈0 at P6 fixed point). Code removed.
**✗ P9** — Decouple M-step β from P6 Newton (partition-specific wins/losses).
**✗ P10** — nAGQ σ gradient at P6 β (P6 β makes W≈0, σ uninformative from likelihood).
**✗ P13a/cold-start-full** — Reset β=ν_ffx and b̂_g=0 before P12, bypassing PQL entirely.

  Hypothesis: PQL/IRLS underdetermines β at large d → bad starting basin → P12 can't
  escape. A neutral start (β=prior, b̂_g=0) would reach the true Laplace MAP.

  Result: catastrophic FFX regression at all scales (small-b-sampled test 0.91 vs 0.31).
  Root cause: FE/RE confounding. With b̂_g=0 and β=ν_ffx, the inner loop (n_inner=4)
  converges b̂_g to absorb all between-group variance before β can learn — the β score
  ∑_g X_g'(y_g − μ(Xβ + Zb̂_g)) is near zero after the inner loop, so β never moves.
  At large σ_rfx, Ψ^{-1} shrinkage is weak, so blups absorb even more freely.

  Key insight: PQL is not just an initializer for β — it is necessary to give β a head
  start before blups are fitted. Without PQL, b̂_g absorbs what β should explain.

**✗ P13b/cold-start-beta-only** — Reset β=ν_ffx, keep PQL b̂_g.

  Hypothesis: PQL blups are reasonable; resetting only β avoids the confounding and
  lets β find a better basin while blups warm-start from PQL.

  Result: even worse than P13a (small-b-sampled test 1.11 vs 0.31). PQL blups are
  calibrated to PQL's β; using them with a different β creates a misleading gradient
  from step 0 that actively pushes β away from the MAP.

**✗ P13c/outer-loop-nAGQ** — After P12, re-run nAGQ at P12's β then run P12 again.

  Hypothesis: nAGQ estimates σ_rfx at PQL's β (bad at large d); re-running it at P12's
  improved β gives better Ψ, which then improves a second P12 round.

  Result: diverges and prohibitively slow (9+ minutes for small benchmark at N=8192,
  vs ~30 s baseline; tests went from 3.6 s to 18.6 s). Root cause: nAGQ (Adam, 10 steps
  on the marginal Laplace LML for σ_rfx) and P12's M-step (closed-form posterior-mean
  E[b_g b_g' + H_g^{-1}]) optimize different objectives. Alternating them produces
  oscillation, not convergence.

Revised gap analysis (2026-05-15, after P13 experiments):
- The cold-start experiments confirmed that the PQL initialization is load-bearing, not
  just a convenient starting point. Without it, the FE/RE confounding prevents β from
  being estimated altogether.
- The real INLA advantage is the marginal-Laplace objective with a profile-likelihood
  approach: optimize σ_rfx on the profile LML (with b̂_g nested), start from tiny
  σ_rfx so that blups ≈ 0 and β can learn first, then grow σ_rfx. This sidesteps
  confounding by design. It is structurally different from PQL+refinement.
- Closing the gap requires implementing P14 (single-mode Laplace-EB): jointly optimize
  (β, log σ_rfx) on the true Bernoulli Laplace LML from a proper cold start with small
  initial σ_rfx. The PQL+P12 framework cannot be patched to achieve this.
- P14 target: ~50–150 ms/dataset batched, matching P12 structure but on the correct
  joint objective. Next step once capacity allows.

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
