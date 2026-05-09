Plan
Last updated: 2026-05-09 (after I3 Option D + Option C attempted, reverted)

Current code state: Fix 1 + Fix 2 + Fix 4 + Fix 7 + Fix 9 + Fix C
  Code also contains `_remlNewtonStep` in normal.py (correct REML gradient; not wired in)
  Fix 1  — Ψ/G_mom additive blup_var floor (addresses WP-V2, WP-V3)
  Fix 2  — Delta-method cap loosened to min=2 (addresses WP-V1, minimal impact)
  Fix 4  — Off-diagonal Ψ shrinkage in Normal path (addresses WP-C1)
  Fix 5  — EM early-exit on batch-max convergence (micro-opt, max=5 unchanged)
  Fix 7  — Per-component count floor in component-wise MoM (addresses WP-Ψ4)
  Fix 9  — Adaptive M-step BLUP winsorization at 10× (partially addresses WP-EM1)
  Fix C  — Outlier-trimmed Ψ_em M-step at 3× mean BLUP norm (addresses WP-EM1 P3)

---

Key lessons from Fixes 1–E

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

  5. G_mom is bounded by structural data size, not Ψ quality.
     Fix E showed that _groupZDiagnostics is purely structural (ZtZ, ns, q — no Ψ/σ²).
     No matter how much Ψ improves, the mom4 mask cannot change. Groups are excluded by
     the rank or ns conditions (both structural), not the condition-number cap
     (cond_cap=1e6 admits essentially all real-world groups). For small-n-mixed,
     G_mom=4–7 is irreducible with current simulation sizes. Any fix to P1 must work
     WITH the constraint G_mom=4–7, not try to expand it.

  6. A regime diagnostic oracle is essential before designing a fix.
     I3 Option D targeted G_mom < 10 (the "EM false fixed point" regime), but a
     diagnostic revealed those datasets are ALREADY the best performers (NRMSE 0.79).
     The actual problem is G_mom=10-49 (NRMSE 1.10, 78% of datasets). Using true Ψ and
     σ² (oracle GLS) gives NRMSE 0.33, confirming the gap is entirely from Ψ estimation
     error. Before designing a fix, always run: (a) oracle diagnostic to confirm gap is
     real, (b) breakdown by G_mom/psi_df to identify the true problem regime.
     G_mom alone does not identify the small-n problem: large-n datasets with G_mom < 10
     (10–15% of those datasets) would also be gated, causing catastrophic regressions.

---

Remaining open problems (ordered by priority)

  P1 — small-n-mixed BLUPs stuck at NRMSE ~1.07 [root cause: WP-Ψ2]
  Every fix has left this effectively unchanged (BLUPs: 1.14 before Fix 9, now ~1.07
  but the small gain came from better M-step, not better Ψ init). The fundamental
  problem is Ψ estimation error (over- OR under-estimation).
  Target for I3 (REML): NRMSE < 0.9.

  NEW INSIGHT (2026-05-09, after I3 attempt): Oracle diagnostic with TRUE Ψ and σ²
  gives BLUP NRMSE = 0.33 — confirming that better Ψ estimation would yield huge gains.
  Breakdown of Fix C NRMSE by G_mom:
    G_mom < 5:    0.87  (104 datasets — small G total, few groups, actually BEST)
    G_mom 5-9:    0.79  (721 datasets — also good)
    G_mom 10-19:  1.10  (2285 datasets — THE REAL PROBLEM, 28% of data)
    G_mom 20-49:  1.10  (4076 datasets — THE REAL PROBLEM, 50% of data)
    G_mom 50+:    1.02  (1006 datasets)
  The original I3 plan targeted G_mom < 10 (the EM false Ψ=0 fixed point), but G_mom <
  10 datasets are actually the BEST performing subset! The actual problem is G_mom=10-49
  where MoM has some information but the MoM+EM cycle consistently over- OR under-estimates
  Ψ. This regime requires REML with a gate based on psi_df (total residual df in informative
  groups), not G_mom alone. See I3 revised plan.

  P2 — sEps regression from Fix 7 [root cause: variance partitioning + WP-σ1/σ2]
  Most datasets saw 20–66% worse σ̂_eps after Fix 7. The EM σ² update now wanders
  further from the Stage 1 estimate. The Stage 1 estimate has its own biases
  (WP-σ1/σ2) but at least it doesn't depend on Ψ quality.

  P3 — huge-n-mixed [largely resolved by Fix C]
  Fix C reduced huge-n-mixed BLUPs from 0.5133 → 0.4663 (−9%). The remaining gap
  from the Fix 7 baseline (0.4644) is 0.0019 — effectively noise. Consider P3 closed.

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
  Relaxed cond_cap to ∞ after i=0. Result: zero impact on all 6 datasets. Root
  cause: _groupZDiagnostics is purely structural; no Ψ/σ² dependency. All exclusions
  come from rank or ns conditions. G_mom is irreducibly structural. Lesson: see
  Key Lesson 5 above.

---

Implementation plan: I3 — REML-Newton for diagonal variance components

  Target: P1 (small-n-mixed BLUPs, NRMSE ~1.07 → < 0.9)
  Effort: ~80 lines for REML gradient, ~30 lines integration
  Risk: Medium-high

  --- Why REML fixes what EM cannot ---

  The EM M-step at Ψ ≈ 0 is a false fixed point. When Ψ → 0:
    - W_g = (σ²Ψ⁻¹ + ZtZ_g)⁻¹ → 0  (prior precision dominates)
    - BLUPs b̂_g = W_g Z_g'r_g → 0  (all groups shrunk to prior mean)
    - blup_outer b̂_g b̂_g' → 0
    - post_cov σ²W_g → σ²·0 = 0
    - M-step: Ψ_em = (blup_outer + post_cov)/G_mom → 0 ✓ (fixed point!)

  This fixed point has nothing to do with the data. The EM converges to it because
  once BLUPs → 0, no E-step signal escapes. It is NOT a stationary point of the
  likelihood — ∂log L/∂Ψ > 0 at Ψ=0 whenever the data have between-group signal.

  The REML gradient for diagonal component ψ_k:

    S_k = -½ Σ_g(active) H_g_kk + ½ Σ_g(active) (Z_g'Py_g)²_k

  where H_g = Z_g'V_g⁻¹Z_g (per-group information matrix, q×q).

  At Ψ → 0, V_g → σ²I, so Z_g'Py_g → Z_g'r_g/σ² (raw OLS residuals), and:

    S_k|_{Ψ=0} ≈ -½ Σ_g ZtZ_g_kk/σ² + ½ Σ_g (Z_g'r_g)²_k/σ⁴

  When the k-th random effect has any between-group signal, the second term (sum of
  squared Z-residuals) exceeds the first (sum of within-group information) and S_k > 0.
  A Newton step Δψ_k = S_k / F_k > 0 moves AWAY from 0, escaping the trap.

  The EM lacks this because it conditions on b̂_g (which are 0 at Ψ=0), never seeing
  the raw residual signal in Z_g'r_g.

  --- Key equations (all quantities from existing GLS output) ---

  Given: W_g (B, m, q, q), blups b̂_g (B, m, q), resid r_g (B, m, n), ZtZ_safe (B, m, q, q)

  Step 1 — Per-group information matrix (diagonal elements only):
    ZtZ_W = einsum('bmqr,bmrs->bmqs', ZtZ_safe, W_g)     # reuse from σ² update
    ZtZ_W_ZtZ = einsum('bmqr,bmrs->bmqs', ZtZ_W, ZtZ_safe)
    H_g_diag = (ZtZ_safe.diagonal(dim1=-2,dim2=-1)
                - ZtZ_W_ZtZ.diagonal(dim1=-2,dim2=-1) / se2[:, None, None]) / se2[:, None, None]
    # shape (B, m, q); H_g_diag_k = (ZtZ_g_kk - [ZtZ_g W_g ZtZ_g]_kk / σ²) / σ²

  Step 2 — REML score approximation (uses Z_g'V_g⁻¹r_g = Ψ⁻¹b̂_g/σ², exact at Ψ=0):
    psi_diag = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=psi_diag_floor)  # (B, q)
    Vr_g = blups / (psi_diag[:, None, :] * se2[:, None, None])  # (B, m, q)
    active_m = mask_m.bool()  # (B, m)
    S_k  = (-0.5 * (H_g_diag * active_m[:,:,None]).sum(dim=1)
            + 0.5 * (Vr_g.square() * active_m[:,:,None]).sum(dim=1))  # (B, q)
    F_k  = 0.5 * (H_g_diag.square() * active_m[:,:,None]).sum(dim=1)  # (B, q)

  Step 3 — Newton step (damped, clamped):
    delta_psi = S_k / F_k.clamp(min=1e-10)  # (B, q)
    psi_diag_new = (psi_diag + 0.5 * delta_psi).clamp(min=psi_diag_floor)

  Step 4 — Reconstruct Ψ:
    Psi = _psdClampEigenvalues(
        _forceDiagonalPsi(torch.diag_embed(psi_diag_new) * active_qq, uncorr), psi_eig_cap
    )

  Note on the approximation: Z_g'Py_g = Z_g'V_g⁻¹r_g - Z_g'V_g⁻¹X_g A_reg⁻¹ X'V⁻¹r
  (the X correction mixes groups via A_reg). The approximation drops this correction.
  It is exact at Ψ=0 (where V=σ²I and the BLUP is zero) and becomes a small
  relative error once Ψ is well-estimated. The key use case (small-n-mixed, G_mom=4–7)
  has few groups, so the correction is small.

  --- Derivation of Z_g'V_g⁻¹r_g = Ψ⁻¹b̂_g/σ² ---

  Using Woodbury: V_g⁻¹ = σ⁻²(I - Z_g W_g Z_g'/σ²)
  Z_g'V_g⁻¹r_g = σ⁻²(Z_g'r_g - Z_g'Z_g W_g Z_g'r_g/σ²)
               = σ⁻²(Ztr_g - ZtZ_g W_g Ztr_g/σ²)
  Using ZtZ_g = W_g⁻¹ - σ²Ψ⁻¹:
  ZtZ_g W_g = (W_g⁻¹ - σ²Ψ⁻¹)W_g = I - σ²Ψ⁻¹W_g
  → σ⁻²(Ztr_g - (I - σ²Ψ⁻¹W_g)Ztr_g/σ²)
  = σ⁻²(Ztr_g - Ztr_g/σ² + Ψ⁻¹W_g Ztr_g/σ²)
  Hmm, that yields σ⁻²Ztr_g(1-1/σ²) + Ψ⁻¹b̂_g/σ⁴ which doesn't simplify cleanly.

  Correct derivation: starting from b̂_g = W_g Ztr_g → Ztr_g = W_g⁻¹b̂_g = (σ²Ψ⁻¹+ZtZ_g)b̂_g
  Z_g'V_g⁻¹r_g = σ⁻²(Ztr_g - ZtZ_g W_g Ztr_g/σ²)
               = σ⁻²(Ztr_g - (I-σ²Ψ⁻¹W_g)Ztr_g/σ²)
               = σ⁻²(1 - 1/σ²)Ztr_g + σ⁻²·σ²Ψ⁻¹W_g Ztr_g/σ²
               = σ⁻²(1-1/σ²)Ztr_g + Ψ⁻¹b̂_g/σ²

  For σ²=1 this gives exactly Ψ⁻¹b̂_g. For σ²≠1, there's an extra term proportional
  to Ztr_g. In practice, b̂_g ≈ (Ψ/(Ψ+σ²/ZtZ))·true_rfx, and the approximation is
  still in the right direction. A cleaner approximation: use Ztr_g = W_g⁻¹b̂_g directly:
    Vr_g_exact = (ZtZ_safe @ blups + se2[:, None, None, None] * 
                  einsum('bmqr,bmnq->bmnr'... — shape mismatch, need einsum over n)

  Simpler: compute Ztr = einsum('bmnq,bmn->bmq', Zm, gls.resid) each REML step.
  Then: Z_g'V_g⁻¹r_g = (Ztr - ZtZ_W @ Ztr ... wrong shape)

  Actually simplest: compute Ztr = einsum('bmnq,bmn->bmq', Zm, gls.resid)  # (B,m,q)
  Then b̂_g = W_g @ Ztr (which we already have as gls.blups). For the REML approx:
  Use Ztr directly (= Z_g'r_g, not V⁻¹-weighted) as the score numerator term.

  At Ψ=0: (Z_g'r_g)²_k/σ⁴ is the signal; H_g_kk ≈ ZtZ_g_kk/σ² is the noise cost.
  This gives the correct sign of the gradient at Ψ=0. For finite Ψ, it's biased
  toward larger ψ_k (since it doesn't downweight by the prior), so use damping.

  REVISED Step 2 (use raw Ztr, simpler and exact at Ψ=0):
    Ztr = einsum('bmnq,bmn->bmq', Zm, gls.resid)   # Z'r, not V⁻¹-weighted
    S_k  = (-0.5 * (H_g_diag * active_m[:,:,None]).sum(dim=1)
            + 0.5 * (Ztr.square() / se2[:, None, None]² * active_m[:,:,None]).sum(dim=1) / G_active
    Wait — units: S_k should be dimensionless. [H_g_diag] = 1/(variance²) roughly;
    [Ztr²/σ⁴] = same units. Need to be careful.

  Actually, let's just use the standard Fisher scoring formula from scratch and
  not try to be clever about approximations. Compute numerically.

  --- Algorithm sketch for prototype ---

  New function _remlNewtonStep(Zm, gls, ZtZ_safe, se2, Psi, mask_m, active_qq,
                               psi_diag_floor, psi_eig_cap, uncorr) → Psi_new:

    B, m = mask_m.shape
    active = mask_m.bool()                             # (B, m)
    se2_bm = se2[:, None, None]                        # (B, 1, 1)

    # H_g = Z'V⁻¹Z diagonal: (ZtZ - ZtZ W ZtZ/σ²) / σ²
    ZtZ_W = einsum('bmqr,bmrs->bmqs', ZtZ_safe, gls.W_g)
    ZtZ_W_ZtZ = einsum('bmqr,bmrs->bmqs', ZtZ_W, ZtZ_safe)
    H_diag = ((ZtZ_safe - ZtZ_W_ZtZ / se2_bm).diagonal(dim1=-2,dim2=-1)
              / se2_bm.squeeze(-1))                    # (B, m, q)
    H_diag = H_diag * active[:, :, None]

    # Score numerator: use raw Z'r (exact at Ψ=0, biased but directionally correct
    # for finite Ψ; corrects itself as Ψ improves across iterations)
    Ztr = einsum('bmnq,bmn->bmq', Zm, gls.resid)      # (B, m, q) = Z_g'r_g
    signal = Ztr.square() / se2[:, None, None].square()  # (B, m, q)
    signal = signal * active[:, :, None]

    # REML score and Fisher information per component
    S = -0.5 * H_diag.sum(dim=1) + 0.5 * signal.sum(dim=1)   # (B, q)
    F = 0.5 * H_diag.square().sum(dim=1).clamp(min=1e-10)      # (B, q)

    # Newton step with 0.5 damping and clamp to floor
    psi_diag = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=psi_diag_floor)
    psi_diag_new = (psi_diag + 0.5 * S / F).clamp(min=psi_diag_floor)

    Psi_new = _psdClampEigenvalues(
        _forceDiagonalPsi(torch.diag_embed(psi_diag_new * active_qq.diagonal(dim1=-2,dim2=-1)),
                          uncorr),
        psi_eig_cap,
    )
    return Psi_new

  --- REML signal derivation (key result) ---

  The correct REML score signal uses Z_g'Py_g where P = V⁻¹ - V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹.
  Since residuals r_g = y_g - X_g β_gls: Z_g'Py = Z_g'V_g⁻¹(y_g - X_g β_gls) = Z_g'V_g⁻¹r_g.
  This equals (Ztr - ZtZ_W Ztr / σ²) / σ² (exact Woodbury formula). See implemented
  function `_remlNewtonStep` in normal.py.

  --- Integration strategy (attempts and lessons) ---

  Option D attempted (2026-05-09, FAILED):
    Gate: G_mom_raw < 10. Neutral init: Psi = diag(psi_diag_floor). n_reml=8.
    Result: small-n-mixed BLUPs 1.0687 → 1.0683 (≈0 improvement). Large-n-mixed,
    huge-n-mixed: catastrophic regressions (+42-158% sRFX).

    Failure mode 1 — Wrong gate. G_mom < 10 triggers for 10.7% of large-n-mixed
    and 15% of huge-n-mixed datasets (those with many total groups but few informative
    ones due to small n_g in those specific groups). REML from neutral init for
    these datasets introduces large noise.

    Failure mode 2 — Wrong target regime. The G_mom < 10 datasets for small-n-mixed
    are ALREADY the best performers (NRMSE 0.79-0.87). The actual problem is G_mom=10-49
    (NRMSE 1.10). The plan's diagnosis (EM false fixed point at Ψ=0 primarily harms
    G_mom=4-7) was wrong — those datasets don't hit the fixed point as hard as thought.

  Option C attempted (2026-05-09, FAILED):
    No gate. Start REML from MoM Ψ for all datasets. n_reml=4.
    Result: small-n-mixed BLUPs → 2.67 (catastrophic). All other datasets regress.

    Failure mode — Signal noise at MLE. For large-n datasets where MoM is approximately
    correct, E[S_k] ≈ 0 but Var[S_k] is O(G·ψ²). With G=50, std(Δψ) ≈ ψ/√50 per step.
    After 4 steps, accumulated noise ≈ 27% of ψ. REML adds noise to well-estimated Ψ.

  --- New analysis: oracle and true problem regime ---

  Oracle BLUP NRMSE for small-n-mixed (using TRUE Ψ and σ²): 0.3323.
  The gap 1.07 → 0.33 is entirely from Ψ estimation error. The fix must improve Ψ
  estimation accuracy for the datasets causing the high NRMSE.

  BLUP NRMSE breakdown by G_mom (Fix C baseline):
    G_mom < 10:  0.79-0.87  (10% of datasets — not the problem!)
    G_mom 10-49: 1.10       (78% of datasets — the actual problem)
    G_mom 50+:   1.02       (12% of datasets)

  The correct gate must target small-n datasets with over-shrunk Ψ without harming
  large-n datasets (where MoM works). DIAGNOSTIC FINDING (2026-05-09): no simple
  structural feature separates the types at runtime. All scalar features
  (psi_df, G_mom, mean_ns, m, n_total) have nearly identical distributions across
  small-n-mixed, medium-n-mixed, large-n-mixed, and huge-n-mixed:

    Feature           small-n  medium-n  large-n  huge-n
    psi_df < 300      39.5%    40.4%     40.0%    42.3%   ← useless
    G_mom < 10        10.1%    4.8%      10.7%    15.0%   ← useless
    mean_ns < 20      45.0%    46.9%     47.9%    49.1%   ← useless
    m median          24       29        33       37      ← too similar
    n_total median    548      627       708      783     ← too similar

  The structural difference between types is in d (3 vs 6–14) and q (1 vs 2), which
  ARE observable at runtime but fragile — conflates model structure with dataset size.

  --- I3 revised direction (OPTION E) ---

  Gate on post-EM Ψ/σ²: after the EM has run, check if it collapsed to a near-zero Ψ.
  This detects the EM false fixed point at runtime from its RESULT, not its preconditions.

    psi_ratio = Psi.diagonal(dim1=-2, dim2=-1).amax(dim=-1) / se2  # (B,)
    reml_gate = psi_ratio < THRESHOLD  # triggers when post-EM Ψ << σ²

  Rationale: when the EM converges to the false Ψ=0 fixed point, psi_ratio is very
  small (Ψ ≈ psi_diag_floor, σ² reasonable). For datasets where EM worked correctly,
  Ψ reflects a real between-group signal and psi_ratio is meaningful (> 0.1 or so).
  For datasets where true Ψ is genuinely small (near 0), REML would be a no-op (the
  signal is also near 0, so REML would confirm Ψ ≈ 0 and stay put).

  Threshold selection: needs a diagnostic sweep across all dataset types to find a
  threshold that fires on collapsed-EM datasets and not on healthy-EM datasets. Likely
  in the range 0.01–0.1.

  Integration (post-EM REML refinement):
    # After _emRefineNormal returns (Psi, se2, gls):
    psi_ratio = Psi.diagonal(dim1=-2, dim2=-1).amax(dim=-1) / se2.clamp(min=1e-12)
    reml_gate = psi_ratio < THRESHOLD
    if reml_gate.any():
        reml_psi_cap = torch.maximum(psi_eig_cap, (25.0 * se2).clamp(min=0.1))
        Psi_neutral = torch.diag_embed(psi_diag_floor.clamp(min=1e-10)) * active_qq
        Psi_reml = torch.where(reml_gate[:, None, None], Psi_neutral, Psi)
        gls_reml = _normalGlsAndBlups(..., Psi_reml, ...)
        for _ in range(n_reml):
            Psi_step = _remlNewtonStep(..., reml_psi_cap, ...)
            Psi_reml = torch.where(reml_gate[:, None, None], Psi_step, Psi_reml)
            gls_reml = _normalGlsAndBlups(..., Psi_reml, ...)
        # merge Psi and gls fields

  Key advantage: the gate fires on the OUTCOME (EM collapsed) rather than a proxy for
  conditions that might cause collapse. False positives (large-n datasets where true Ψ
  is genuinely small) are safe: REML on a dataset with true Ψ≈0 and small signal does
  nothing harmful.

  Step 1: run psi_ratio diagnostic — check its distribution per dataset type,
  separately for datasets where the EM has clearly collapsed (Ψ ≈ psi_diag_floor)
  vs where it found a non-trivial Ψ. This will reveal the right threshold.

  Step 2: implement and benchmark with a handful of thresholds.

  --- σ² handling ---

  Keep σ² fixed at the post-EM se2 during the REML refinement pass (not Stage 1 σ̂_eps,
  since the EM σ² update has already run). The REML pass is a pure Ψ correction.

  --- Success criteria ---

    Primary:    small-n-mixed BLUPs NRMSE < 0.9 (current: 1.07)
    Secondary:  small-n-mixed sRFX NRMSE improves (current: ~0.64)
    Regression: all other datasets within ±3% of Fix C baseline

  --- Numerical risks ---

    1. False positives on large-n datasets with genuinely small Ψ: safe (REML confirms
       Ψ≈0 and makes no change).
    2. psi_eig_cap too tight: use max(psi_eig_cap, 25·σ²) for REML cap.
    3. psi_ratio threshold miscalibrated: check distribution to confirm separation.
    4. REML adds cost for any false positives: acceptable at low false positive rate.

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
  │ I3 (REML-Newton diagonal)  │ P1: small-n-mixed BLUPs  │ ~120 lines│ Next   │
  │                            │ target NRMSE < 0.9;      │           │        │
  │                            │ Option E: post-EM gate   │           │        │
  └────────────────────────────┴──────────────────────────┴───────────┴────────┘
