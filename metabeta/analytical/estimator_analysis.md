Estimator-by-estimator analysis
Last updated: 2026-05-09 (after Fixes 1–9, Fix C, Fix E attempted; I3 Option D+C attempted, reverted)

Status key:  ✓ addressed  ~  partially addressed  ✗ open  ! new insight

  Stage 1: σ̂_ε (within-group projection)

  How it works:
  Project Z out of X and y via Frisch-Waugh (My = y - Z(Z'Z)⁻¹Z'y), fit OLS on
  the Z-partialled residuals, compute RSS / (N - z_rank_sum - mx_rank).

  Weakpoints:

  - WP-σ1 [✗ open] — df denominator at high d. mx_rank = _gramRank(MXtMX). When
  many predictors are near-collinear with Z (e.g., group indicators span most of
  X's column space), mx_rank << d and the active df drops. But the remaining
  "identified" directions may still over-absorb cross-group variance, biasing
  σ̂² upward. There's no correction for this.

  - WP-σ2 [✗ open] — z_rank can be underestimated. z_rank is the sum of per-group
  Z ranks. When a group has very few observations (n_g ≤ q), its ZtZ is
  rank-deficient and we use a jittered inverse, but the rank counter may still
  count it incorrectly, inflating the df denominator → σ̂² biased down.

  ! The sEps regression introduced by Fix 7 (20–66% worse across most datasets)
  made σ̂_eps a more visible failure mode. Fix 7 tightened psi_eig_cap, shifting
  variance from Ψ to σ² in the EM. The Stage 1 estimate (unaffected by EM) may
  be the right anchor for σ²; WP-σ1/σ2 bias in that estimate propagates directly
  into the EM fixed point via se2_anchor.

  ---
  Stage 2: Initial Ψ̂ (MoM)

  How it works:
  Compute beta_ols from global OLS (no rfx). Get per-group OLS random effects
  bhat_g = (ZtZ_g)⁻¹ Z'(y - X·beta_ols). Solve the MoM equation Σ_{g in mom}
  ZtZ_g · bhat_g bhat_g' · ZtZ_g = (Σ ZtZ_g) Ψ + G_mom σ² I. Fall back to
  per-component scalar variance if G_mom < d + 1.

  Weakpoints:

  - WP-Ψ1 [✗ open — Fix A attempted 2026-05-09, abandoned] — OLS beta inflates
  residuals incorrectly. beta_ols is computed without any random effects, so it
  absorbs between-group variance into the group-level residuals. Fix A tried
  using beta_wg (within-Z projection estimator) for the MoM residual. Result:
  medium-n-mixed FFX blew up by 55,000% — beta_wg is poorly identified when
  predictors are confounded with group membership (high-d, mixed-n). The assumption
  that beta_wg is "cleaner" breaks down in the collinear regime. REML (I3) addresses
  this indirectly: the REML score uses ALL groups weighted by their information,
  not just mom_mask groups with beta_ols residuals.

  - WP-Ψ2 [✗ open — hardest remaining problem, target for I3] — Extremely noisy
  at bg_df = 4–7. The MoM works on G_mom informative groups. With bg_df=4 (4 groups
  with ns > active_count + 1), you're fitting a q×q matrix from 4 data points. The
  winsorization at 6× mean helps but is still based on a mean computed from those
  same 4 noisy groups. The diagnostic confirmed RMSE for sigma_rfx at these df
  values is 2–4× worse than at bg_df ≥ 20.

  Current BLUPs NRMSE for small-n-mixed: ~1.07 (marginally improved from 1.14 before
  Fix 9 due to better M-step, not better Ψ init). Fix E confirmed G_mom is irreducibly
  structural (= number of groups with ns > active_count + 1 AND full ZtZ rank) —
  no mask refresh or parameter improvement can increase G_mom.

  I3 attempt (2026-05-09, reverted): REML-Newton with gate G_mom<10 + neutral init.
  Two failure modes discovered:
    (a) Wrong target: G_mom<10 datasets have NRMSE 0.79-0.87 (the BEST subset!). The
        actual problem is G_mom=10-49 (NRMSE 1.10, 78% of datasets). Oracle with true
        Ψ gives NRMSE 0.33 — confirming 3× gap entirely from estimation error.
    (b) Wrong gate: G_mom<10 triggers for 10-15% of large/huge-n-mixed datasets
        (those with many total groups but few informative ones), causing catastrophic
        sRFX regressions (+42-158%). The gate must incorporate n_g/group context.
  Revised I3 direction: gate on psi_df = Σ_g(ns_g - active_count - 1) < 300 over
  mom groups. This separates small-n (psi_df≈100) from large-n (psi_df≈960) even at
  same G_mom. See revised I3 section in plan.md.

  The EM cannot recover from a badly underestimated Ψ: when Ψ̂ → 0, BLUPs → 0,
  M-step Ψ_em → 0, feeding back into worse Ψ. This is a FALSE FIXED POINT of the
  EM iteration — it is NOT a stationary point of the likelihood. The REML gradient
  at Ψ=0 is strictly positive whenever the data have between-group signal, so
  REML-Newton escapes this trap where EM cannot. See revised I3 in plan.md.

  - WP-Ψ3 [✗ open — attempted Fix 6, reverted] — Winsorization uses the mean of
  the noisy signal. signal_cap = 6.0 × signal_mean. When 1 of 4 groups dominates,
  signal_mean is already contaminated and the cap is too high. Fix 6 tried a
  median-based cap but the per-component masks had too few groups (1–2) for the
  median to be meaningful — it returned 0 and collapsed the cap. A median-based
  cap is only safe in the _initialPsiMom mom_mask path (shared across q) where
  G_mom ≥ 5; the _componentwisePsiDiagSignal path must remain mean-based.

  - WP-Ψ4 [✓ addressed by Fix 7] — Component-wise fallback was imprecise.
  Previously, components with only 2 valid groups were used for floor and cap
  estimation, allowing single-group signals to inflate psi_eig_cap and cause
  massive Ψ overestimation. Fix 7 raised the threshold to 5 groups and gates
  diag_cap_signal per-component: components with < 5 valid groups fall back to
  sigma_eps_sq / active_ns_mean. This was the largest single improvement (−12% to
  −50% FFX/BLUPs across large-n datasets) but introduced 20–66% sEps regression
  as a tradeoff (see Stage 1 note above).

  ---
  Stage 3: GLS + BLUPs (Woodbury)

  How it works:
  W_g = (σ²Ψ⁻¹ + ZtZ_g)⁻¹. BLUP b̂_g = W_g · Z_g'·(y_g - X_g·β̂_gls). Posterior
  variance Var(b_g | data) = σ²·W_g.

  Weakpoints:

  - WP-B1 [~ partially addressed by Fix 1] — The posterior variance formula is
  only correct under known Ψ. The formula σ²·W_g is the exact BLUP posterior
  variance given Ψ and σ² are the true parameters. When Ψ̂ ≪ Ψ_true (which is
  common at bg_df=4–7), the BLUP is over-shrunk AND σ²W_g is tiny. Fix 1 added
  an additive Ψ/G_mom floor to blup_var, roughly halving calibration ratios (from
  38× to 17× at n_g=53–150). The root cause (Ψ underestimation) is unaddressed —
  the floor is a downstream workaround. The remaining ratios at large n_g bins are
  still 4–17×.

  - WP-B2 [✗ open] — KH correction collapses at large n_g. The Kackar-Harville
  correction adds (W_g·ZtX)² × Var(β̂). For large n_g, W_g ≈ 0 → KH → 0.
  Correct in principle when Ψ is known, but does nothing when Ψ̂ ≈ 0.

  ---
  Stage 4: EM refinement

  How it works:
  5 EM iterations (hard cap). M-step: Ψ_em = (1/G_mom) Σ_{g in mom}
  (b̂_g b̂_g' + σ²W_g). Damped update: Ψ ← 0.5Ψ + 0.5Ψ_em. σ² updated from
  full residuals with effective df.

  Weakpoints:

  - WP-EM1 [~ partially addressed by Fix 9 + Fix C] — M-step had no outlier protection.
  Fix 9 winsorizes GLS BLUPs at ±10×sqrt(Ψ_diag) before forming blup_outer. Adapts
  as Ψ improves. Result: large-n-mixed BLUPs −35%, large-n-sampled/test −6%, but
  huge-n-mixed regression +10% (cap clips legitimate large BLUPs when Ψ underestimated).
  Fix C (2026-05-09) adds outlier-trimmed Ψ_em: groups with ||b̂_g||² > 3× mean norm
  among mom groups are excluded from the M-step average. This is adaptive — when all
  groups have similar norms no exclusion happens (no bias introduced). Result: huge-n-mixed
  BLUPs further reduced from 0.5133→0.4663 (−9%) on top of Fix 9, no regressions.
  Key lesson: a group-level outlier norm threshold (relative to the distribution of
  ||b̂_g||) is more robust than a per-BLUP cap (Fix 9) because it detects true outlier
  groups rather than extreme individual components, and avoids clipping legitimate BLUPs
  when Ψ is underestimated.

  - WP-EM2 [✗ open — Fix E attempted 2026-05-09, no-op] — mom4 mask is frozen
  from Stage 2. _groupZDiagnostics is purely structural (ZtZ, ns, q — no Ψ/σ²),
  so calling it again after iteration 1 returns the identical mask regardless of
  how much Ψ improves. Relaxing cond_cap to ∞ after i=0 changed nothing across
  all 6 datasets. Root cause: groups excluded from mom_mask fail the rank or ns
  condition (both structural), not the condition-number cap. G_mom is bounded by
  data size (bg_df) and is irreducible — no mask refresh can change this.

  - WP-EM3 [~ partially addressed by Fix 5, EM extension dead-end] — Fixed 5
  iterations regardless of convergence. Fix 5 added a batch-max early exit (break
  when all datasets satisfy |Ψ_delta| < 1e-3 and |σ²_delta| < 1e-3) — retained
  as a micro-optimization. Increasing max beyond 5 was attempted three times:
  Fix 5 (max=10 for all), Fix 8 (max=10 gated on G_mom >= 20), Fix B (max=10
  gated on psi_df >= 300). All three caused catastrophic regressions on medium-n
  datasets. No count statistic (G_mom, psi_df, or any structural proxy) reliably
  distinguishes a biased EM fixed point from an unbiased one. The EM extension
  direction is CLOSED. The underlying issue (biased fixed point at Ψ=0) requires
  a different algorithm (REML, see I3), not more iterations.

  - WP-EM4 [✗ open] — σ² update is fragile at the cap. The T_safe =
  T.clamp(max=0.9×(N−beta_rank)) prevents negative denominator but can mask
  legitimate large σ² values. The se2 = min(se2_next, 100·se2_anchor) cap
  prevents runaway but also blocks recovery when the initial σ̂² from Stage 1
  was badly off. Fix 7's sEps regression suggests the 100× cap is permissive
  enough that EM σ² can wander far from the Stage 1 anchor.

  ---
  Stage 5: blup_var corrections (post-EM)

  How it works:
  Three layered corrections:
  1. Delta-method: blup_var × (1 + 2/max(G-d, 2)) — caps at 100% inflation.
  2. Kackar-Harville: blup_var + (W_g ZtX)² × Var(β̂).
  3. Floor: max(blup_var, Ψ̂/(2n_g)) + Ψ̂/G_mom (additive).

  Weakpoints:

  - WP-V1 [✓ addressed by Fix 2] — Delta-method cap loosened from min=4 to min=2.
  Effect was negligible (activates only at G−d < 4, rare in practice).

  - WP-V2 [✓ addressed by Fix 1] — The floor Ψ̂/(2n_g) goes to zero for large n_g.
  Fix 1 added an additive Ψ̂/G_mom term that is independent of n_g and stays finite
  when G_mom is small. This roughly halved worst-case calibration ratios.

  - WP-V3 [~ partially addressed by Fix 1] — All three corrections collapse
  simultaneously when σ²W_g ≈ 0. Fix 1's additive floor provides a non-collapsing
  term. Remaining ratios at large n_g are still 4–17×, limited by Fix 1 floor
  magnitude (Ψ/G_mom may itself be small when G_mom is large).

  ---
  Stage 6: Correlation structure (Ψ off-diagonals)

  Weakpoints:

  - WP-C1 [✓ addressed by Fix 4] — Off-diagonal shrinkage now applied in Normal
  path via _shrinkOffDiagonal(Psi, G/(G+5.0)) before returning sigma_rfx. Matches
  PQL path. Effect on diagonal NRMSE is minimal; Psi_corr RMSE improves ~1%.

  - WP-C2 [✗ open] — Off-diagonal MoM is noisy at large q. For huge-n-sampled
  (q up to max_q), sign accuracy drops to 56–58%. The MoM cross-product
  bhat_g bhat_g' has off-diagonal noise that scales as sqrt(Ψ_ij / G_mom), and
  with small G_mom and high q, the signal-to-noise in off-diagonals is poor.
  Not a priority given sRFX NRMSE is dominated by diagonal estimation error.
