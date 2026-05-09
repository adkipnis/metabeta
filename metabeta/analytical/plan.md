Plan
Last updated: 2026-05-09

Current code state: Fix 1 + Fix 2 + Fix 4 + Fix 7 + Fix 9 + Fix C
  Fix 1  — Ψ/G_mom additive blup_var floor (addresses WP-V2, WP-V3)
  Fix 2  — Delta-method cap loosened to min=2 (addresses WP-V1, minimal impact)
  Fix 4  — Off-diagonal Ψ shrinkage in Normal path (addresses WP-C1)
  Fix 5  — EM early-exit on batch-max convergence (micro-opt, max=5 unchanged)
  Fix 7  — Per-component count floor in component-wise MoM (addresses WP-Ψ4)
  Fix 9  — Adaptive M-step BLUP winsorization at 10× (partially addresses WP-EM1)
  Fix C  — Outlier-trimmed Ψ_em M-step at 3× mean BLUP norm (addresses WP-EM1 P3)

---

Key lessons from Fixes 1–9

  1. Root cause fixes cascade; downstream workarounds don't.
     Fix 7 (one threshold change) was the largest single improvement (−12 to −50%
     FFX/BLUPs). It was impactful because it fixed a specific algorithmic error —
     single-group component signals were inflating psi_eig_cap — and once Ψ was
     correctly bounded, GLS and EM both improved as cascades. By contrast, Fix 1
     (blup_var floor) halved calibration ratios but had zero NRMSE impact. Fixes
     targeting upstream errors are worth ~10× more than downstream corrections.

  2. The EM fixed point can be biased, not just slow to converge.
     Fixes 5 and 8 both tried to run more EM iterations. Both caused catastrophic
     regressions on medium-n-mixed (FFX +143% and +183%) because the M-step
     estimator (blup_outer + post_cov) / G_mom is biased when BLUPs are over-shrunk
     from a bad Ψ initialization. More iterations = stronger convergence to the wrong
     attractor. The hard cap of 5 is inadvertent regularization. Any extension must
     first verify the fixed point is unbiased, not just that G_mom is large.
     G_mom is NOT a reliable proxy; psi_df = Σ(ns_g − q − 1) over mom groups is better.

  3. Adaptive thresholds beat static ones.
     Fix 3 (static cap at 6×sqrt(Ψ̂_init)) was catastrophic; Fix 9 (cap at
     10×sqrt(Ψ̂_current)) succeeded. The adaptive cap loosens as Ψ improves each
     iteration. Fix 6 (median-based winsorization) failed because the per-component
     sample sizes were too small for the median (returned 0 for 1–2 group masks).
     Fix 7 succeeded by not estimating at all for sparse components — falling back
     to a conservative default. Lesson: when sample size for the threshold estimate
     is < 5, any estimator is worse than a fixed conservative fallback.

  4. Fix 7's sEps regression reveals a variance partitioning problem.
     By tightening psi_eig_cap, Fix 7 prevented Ψ overestimation and correctly
     moved variance to σ², but the EM converges to a σ² fixed point that is farther
     from ground truth. The Ψ/σ² partition is weakly identified in component-wise
     fallback regimes. This means WP-σ1/σ2 (Stage 1 σ̂_eps biases) now matter more
     — the Stage 1 estimate is the only unbiased anchor available.

---

Remaining open problems (ordered by priority)

  P1 — small-n-mixed BLUPs stuck at NRMSE 1.14 [root cause: WP-Ψ2]
  Every fix has left this unchanged. The fundamental problem is that for bg_df=4–7,
  the MoM estimates Ψ from 4–7 noisy outer products. Ψ̂ ≪ Ψ_true → BLUPs → 0 →
  M-step Ψ_em → 0 — a self-reinforcing underestimation cycle the EM cannot escape.
  No downstream fix (blup_var floor, winsorization) can recover BLUP accuracy without
  first recovering Ψ accuracy.

  P2 — sEps regression from Fix 7 [root cause: variance partitioning + WP-σ1/σ2]
  Most datasets saw 20–66% worse σ̂_eps after Fix 7. The EM σ² update now wanders
  further from the Stage 1 estimate. The Stage 1 estimate has its own biases
  (WP-σ1/σ2) but at least it doesn't depend on Ψ quality.

  P3 — huge-n-mixed regression from Fix 9 (+10% BLUPs) [partially addressed by Fix C]
  Fix C reduced huge-n-mixed BLUPs from 0.5133 → 0.4663 (−9%) without any regressions.
  The outlier-trimmed Ψ_em (3× mean norm threshold) detects and excludes groups with
  anomalously large BLUP norms, giving a more robust M-step estimate that doesn't
  over-shrink when a few groups have large BLUPs. Still above the Fix 7 baseline
  (0.4644), but the remaining gap is small.

---

Proposed fixes (ordered by impact-to-effort)

  Fix A — Use beta_wg in MoM residual [ABANDONED — catastrophic FFX regression]
  Tried 2026-05-09 (combined with Fix D). medium-n-mixed FFX: 0.1454 → 80.18 (+55,000%).
  Root cause: beta_wg is poorly identified for datasets where predictors are confounded
  with group membership (high-d, mixed-n). Using it as the MoM beta produces noisy
  bhat_g → bad Ψ_initial → GLS blow-up. The assumption that beta_wg is "cleaner" breaks
  down when the within-Z projection is itself unstable. Would require per-dataset stability
  checking before using beta_wg, which is not tractable.

  Fix B — psi_df-gated EM extension [ABANDONED — same failure as Fix 8]
  Implemented 2026-05-09 and reverted immediately. medium-n-mixed/train BLUPs:
  0.3748 → 0.9601 (+156% regression). Root cause: medium-n-mixed has large psi_df
  (many groups × moderate obs → psi_df >> 300) but its EM fixed point is still biased
  from poor Ψ initialization. psi_df is not a reliable gate for fixed-point quality,
  just as G_mom wasn't. The cap of 5 iterations is not a gating problem but a bias
  problem — no count statistic can distinguish unbiased from biased fixed points
  without ground truth. Any further EM extension attempts must first establish that
  the fixed point is unbiased for the target regime.

  Fix C — Outlier-trimmed Ψ_em in M-step [IMPLEMENTED 2026-05-09, targets P3, WP-EM1]
  Implemented with adaptive 3× mean threshold after two failed intermediate versions:
    v1 (top 10% fixed trim): medium-n-mixed FFX +55%, BLUPs +22% regression ✗
    v2 (G_mom >= 20 gate, 5% trim): medium-n-mixed FFX +125%, BLUPs +85% regression ✗
    v3 (adaptive 3× mean norm threshold): no regressions ✓

    mom_norm_mean = (blup_norm * mom_mask_1d.float()).sum(dim=1) / G_mom  # (B,)
    outlier_thresh = 3.0 * mom_norm_mean[:, None]   # exclude groups with norm > 3× mean
    trim_mask = mom_mask_1d & (blup_norm <= outlier_thresh)  # (B, m)
    Psi_em = _psdProject(((blup_outer + post_cov) * trim_mask_4d).sum(dim=1) /
                          trim_count[:, None, None])

  When all groups have similar norms (no genuine outliers), trim_mask = mom_mask_1d
  and the result is identical to the original full mean. When one or more groups are
  anomalous, they're excluded, giving a more stable Ψ_em.

  Results (vs Fix 7+9 baseline): huge-n-mixed BLUPs 0.5133→0.4663 (−9%) ✓,
  large-n-mixed BLUPs 0.3729→0.3670 (−1.6%) ✓, medium-n-mixed ≈unchanged ✓.

  Fix D — Regularize EM σ² toward Stage 1 [ABANDONED — regresses BLUPs]
  Tried 2026-05-09 alone (after Fix A reverted). Results at 0.8/0.2 blend:
    sEps improvement: 2.5–7% (small; sEps regression from Fix 7 was 20–66%)
    huge-n-mixed BLUPs: 0.5133 → 0.6725 (+31% regression) ✗
    large-n-mixed BLUPs: 0.3729 → 0.4035 (+8% regression) ✗
  Root cause: Stage 1 σ̂_eps is biased upward for large-n datasets (WP-σ1: OLS on
  Z-partialled residuals absorbs cross-group variance). Blending toward Stage 1
  raises σ², increasing BLUP shrinkage. For large n_g, BLUPs are large in magnitude
  so even a small σ² increase causes over-shrinkage. The tradeoff is unacceptable:
  −3% sEps improvement at the cost of +8–31% BLUP regression.

  Fix E — Refresh mom4 mask after first EM iteration [ABANDONED — 2026-05-09, no-op]

  Key finding: _groupZDiagnostics is purely structural (ZtZ, ns, q — no Ψ/σ²).
  Calling it again after iteration 1 returns the IDENTICAL mask regardless of how
  much Ψ improves. The original plan sketch was wrong that sigma_rfx would change
  the result.

  Implemented as: relax cond_cap to infinity after i=0 (only condition that
  could admit new groups). Result: zero impact on all 6 datasets — confirmed by
  identical BLUPs/FFX/sEps to Fix C baseline across every partition.

  Root cause of zero impact: cond_cap = 1e6 is already so permissive that
  essentially no groups are excluded by the condition number criterion alone.
  Groups excluded by mom_mask fail the rank or ns conditions (rank < active_count,
  or ns <= active_count + 1) — both of which depend only on data size and ZtZ
  structure, not Ψ quality. There is no structural criterion that Ψ-improvement
  can unlock.

  Lesson: G_mom is bounded by sample size (bg_df = number of groups with ns > q+1
  AND full ZtZ rank), not by condition number. For small-n-mixed, G_mom = 4–7 is
  irreducible with the current data sizes — no mask refresh can change this.

---

Investigation items

  I1 — Measure Fix A impact on small-n-mixed specifically.
  Run glmm_error_analysis.py --data-id small-n-mixed with beta_wg → beta_for_mom
  in _initialPsiMom. Check whether sigma_rfx RMSE at bg_df=4–7 improves.
  If MoM signal improves by 20%+, pursue. If negligible, the bottleneck is
  WP-Ψ2 (irreducible noise at low G_mom), not beta contamination.

  I2 — Characterize the psi_df distribution for medium-n-mixed datasets.
  Log psi_df = Σ(ns_g − q − 1) for medium-n-mixed batches. Verify that the
  problematic datasets (those where Fix 5/8 caused regressions) fall below
  psi_df = 300. If they do, Fix B's threshold is valid. If the distribution
  overlaps, increase threshold to 500+.

  I3 — REML as a replacement for Stage 2 + EM.
  REML gives unbiased Ψ and σ² for small-sample LMMs and directly fixes
  WP-Ψ1/Ψ2. The tradeoff is compute: REML needs log-det of V per iteration
  in O(N³). Profile on a typical batch (B=32, N≤3000): if runtime is < 2× the
  current analytical path, REML is the cleanest fix for small-n-mixed.

---

Priority order

  ┌────────────────────────────┬──────────────────────────┬───────────┬────────┐
  │          Fix               │ Expected impact          │ Effort    │ Status │
  ├────────────────────────────┼──────────────────────────┼───────────┼────────┤
  │ Fix A (beta_wg in MoM)     │ ABANDONED — FFX=80 on    │ —         │ Dead   │
  │                            │ medium-n-mixed           │           │        │
  ├────────────────────────────┼──────────────────────────┼───────────┼────────┤
  │ Fix D (σ² regularization)  │ ABANDONED — +31% BLUPs   │ —         │ Dead   │
  │                            │ regression huge-n-mixed  │           │        │
  ├────────────────────────────┼──────────────────────────┼───────────┼────────┤
  │ Fix C (outlier-trim Ψ_em)  │ P3: huge-n-mixed −9%     │ 10 lines  │ Done   │
  │                            │ BLUPs, no regressions    │           │        │
  ├────────────────────────────┼──────────────────────────┼───────────┼────────┤
  │ Fix B (psi_df-gated EM)    │ ABANDONED — same failure │ —         │ Dead   │
  │                            │ as Fix 8, reverted       │           │        │
  ├────────────────────────────┼──────────────────────────┼───────────┼────────┤
  │ Fix E (refresh mom4)       │ ABANDONED — _groupZDiag  │ —         │ Dead   │
  │                            │ is structural; no-op     │           │        │
  ├────────────────────────────┼──────────────────────────┼───────────┼────────┤
  │ I3 (REML)                  │ Potentially solves P1    │ High      │ Third  │
  │                            │ root cause               │           │ pass   │
  └────────────────────────────┴──────────────────────────┴───────────┴────────┘
