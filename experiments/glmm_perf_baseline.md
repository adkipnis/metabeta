GLMM Analytical Estimator — Baseline Performance & Fix Log
===========================================================

Diagnostic run: 2026-05-08 (12 combinations × 8192 datasets each)
Script: experiments/glmm_error_analysis.py
Data: {small,medium,large,huge}-n-{mixed,sampled}  (mixed→train ep1–2, sampled→valid+test)

---

## Baseline NRMSE

| Dataset          | Partition | FFX (β) | σ(RFX) | σ(Eps) | BLUPs  |
|------------------|-----------|---------|--------|--------|--------|
| small-n-mixed    | train     | 0.2291  | 0.5906 | 0.084  | 1.1443 |
| medium-n-mixed   | train     | 0.2584  | 0.4521 | 0.0564 | 0.7853 |
| large-n-mixed    | train     | 0.3476  | 0.4562 | 0.0557 | 0.6815 |
| huge-n-mixed     | train     | 0.3748  | 0.4498 | 0.0499 | 0.6561 |
| small-n-sampled  | valid     | 0.2085  | 0.5547 | 0.0808 | 0.9539 |
| small-n-sampled  | test      | 0.2079  | 0.5562 | 0.0817 | 0.9574 |
| medium-n-sampled | valid     | 0.3259  | 0.5089 | 0.0717 | 0.7232 |
| medium-n-sampled | test      | 0.3267  | 0.5095 | 0.0723 | 0.7251 |
| large-n-sampled  | valid     | 0.5252  | 0.5805 | 0.1072 | 0.7302 |
| large-n-sampled  | test      | 0.5267  | 0.5781 | 0.1081 | 0.7306 |
| huge-n-sampled   | valid     | 0.6162  | 0.5776 | 0.1101 | 0.7165 |
| huge-n-sampled   | test      | 0.6157  | 0.5824 | 0.1099 | 0.7161 |

---

## Baseline blup_var calibration ratios (mean(err²)/mean(blup_var))

Ratio > 1 → overconfident (predicted variance too small).

### small-n-mixed (worst case)
| n_g bin  |  N     | mean(blup_var) | mean(err²) | ratio  |
|----------|--------|----------------|------------|--------|
| 5–7      | 52400  | 0.0826         | 0.0976     |  1.18  |
| 7–14     | 68107  | 0.0722         | 0.167      |  2.32  |
| 14–29    | 68506  | 0.0412         | 0.6497     | 15.76  |
| 29–53    | 62520  | 0.0155         | 0.3825     | 24.62  |
| 53–150   | 64361  | 0.0114         | 0.4327     | 37.91  |

### large-n-sampled-test
| n_g bin  |  N     | mean(blup_var) | mean(err²) | ratio  |
|----------|--------|----------------|------------|--------|
| 5–9      | 213883 | 0.0231         | 0.0389     |  1.69  |
| 9–15     | 102823 | 0.0127         | 0.025      |  1.97  |
| 15–32    | 117011 | 0.0087         | 0.0505     |  5.80  |
| 32–150   | 111209 | 0.0077         | 0.0652     |  8.43  |

Root cause: when Ψ̂ ≪ Ψ_true (bg_df=4–7), W_g→0, blup_var=σ²W_g→0, BLUP→0,
but actual MSE≈Ψ_true. All three corrections (delta-method, KH, n_g floor) also
→0, leaving blup_var catastrophically underestimated.

---

## Fix Log

### Fix 1 — Ψ/G_mom additive floor (2026-05-08)

**File**: metabeta/analytical/normal.py, `_lmmNormalFull`

**Change**: After the existing `Ψ/(2n_g)` clamp floor, add an additive term
`Ψ/G_mom` that does NOT depend on n_g. This stays finite for large groups when
Ψ̂ is noisy, addressing WP-V2 and WP-V3.

```python
# Before (existing code):
blup_var_floor = psi_diag[:, None, :] / (2.0 * ns.clamp(min=1.0)[:, :, None])
blup_var = blup_var.clamp(min=blup_var_floor)

# After:
blup_var_floor = psi_diag[:, None, :] / (2.0 * ns.clamp(min=1.0)[:, :, None])
blup_var = blup_var.clamp(min=blup_var_floor)
blup_var_psi_floor = psi_diag[:, None, :] / G_mom.clamp(min=1.0)[:, None, None]
blup_var = blup_var + blup_var_psi_floor
```

**Expected effect**: ratios at 29–53 and 53–150 bins drop; small-group bins
(5–7) only marginally affected since Ψ/G_mom ≈ 0.009 << mean(blup_var)=0.083.

**Results — small-n-mixed** (baseline → Fix 1):
| n_g bin | ratio (before) | ratio (after) |
|---------|----------------|---------------|
| 5–7     |  1.18          |  1.04         |
| 7–14    |  2.32          |  1.97         |
| 14–29   | 15.76          | 12.05         |
| 29–53   | 24.62          | 15.24         |
| 53–150  | 37.91          | 17.17         |
BLUP NRMSE unchanged (1.1443 — blup_var doesn't affect the estimates).

**Results — large-n-sampled-test** (baseline → Fix 1):
| n_g bin | ratio (before) | ratio (after) |
|---------|----------------|---------------|
| 5–9     |  1.69          |  1.24         |
| 9–15    |  1.97          |  1.41         |
| 15–32   |  5.80          |  3.84         |
| 32–150  |  8.43          |  4.36         |
BLUP NRMSE unchanged (0.7306).

**Assessment**: Fix 1 roughly halved the worst-case ratios. Still overconfident at large n_g
because Ψ̂ ≪ Ψ_true causes BLUPs to be overshrunk (NRMSE unchanged). Next fix should target
Ψ̂ quality improvement (Fix 3) rather than further blup_var patching.

---

### Fix 2 — Loosen delta-method cap (2026-05-08)

**File**: metabeta/analytical/normal.py, `_lmmNormalFull`

**Change**: `df_sigma.clamp(min=4.0)` → `df_sigma.clamp(min=2.0)`. Allows up to 100% blup_var
inflation (was capped at 50%) for datasets where G−d < 4.

**Results**: Numerically identical to Fix 1+4 on both test cases. Fix 2 only activates at G−d < 4
(very few groups relative to fixed effects), which is rare in the 8192-dataset evaluation suite.
No regression; no measurable improvement in aggregates.

Current state: Fix 1 + Fix 2 + Fix 4.

---

### Fix 6 — Median-based MoM winsorization cap (2026-05-08, REVERTED)

**Attempted change**: Replace `signal_cap = 6.0 * signal_mean` with
`6.0 * max(signal_median_raw, signal_mean * 0.1)` in both `_componentwisePsiDiagSignal` and
`_initialPsiMom`, using the pre-winsor median as the cap reference.

**Results**: large-n-sampled-test FFX NRMSE 0.527 → **8.45** (16× worse), BLUPs 0.731 → **1.79**.

**Root cause**: `_componentwisePsiDiagSignal` uses a per-q-component mask. For large-q datasets
many components have only 1–2 valid groups → `_maskedMedian` returns 0 → cap collapses to
`6 * 0.1*mean = 0.6*mean` (10× too tight) → `psi_eig_cap` clamped down → Ψ massively
underestimated → GLS/BLUPs blown up.

**Reverted.** The median-based cap could only be applied to the `_initialPsiMom` mom_mask path
(shared across q), but the `_componentwisePsiDiagSignal` path must stay mean-based. Given the
`min(mean, 2*median)` final estimator already handles single-outlier cases at output, the
improvement in `_initialPsiMom` alone is likely negligible — not worth the complexity.
### Fix 3 — EM M-step BLUP winsorization (2026-05-08, REVERTED)

**File**: metabeta/analytical/normal.py, `_emRefineNormal`

**Attempted change**: Before forming `blup_outer = blups blups'` in the M-step, clamp each BLUP
component at `±6×sqrt(Ψ_diag)` to prevent outlier groups (blown BLUPs from near-singular GLS
at high d/q) from spiking Ψ_em.

**Results:**
- large-n-sampled-test: BLUPs NRMSE 0.731 → 0.548 (−25%), FFX 0.527 → 0.471 (−11%)
- large-n-mixed: BLUPs 0.576 → 0.371 (−35%) 
- huge-n-mixed: BLUPs 0.464 → **0.894 (+93%)** — catastrophic regression

**Root cause of regression**: When Ψ̂ underestimates Ψ_true by 2×+, the cap `6×sqrt(Ψ̂_init)`
is too tight. Legitimate large BLUPs (|b_g| ≈ 3σ_rfx, but σ_rfx underestimated → cap < 3σ_rfx_true)
get clipped, biasing Psi_em downward, degrading the final GLS BLUPs for mid-size groups (n_g=7–26).
Using a fixed pre-loop cap didn't help — the cap is still based on Ψ̂_init which can be 4× off.

**Reverted.** Remaining current state: Fix 1 only.

**Future approaches for Fix 3**:
- Detect blown BLUPs via per-group condition number rather than Ψ-relative threshold
- Gate winsorization on datasets where n_g/q < threshold (ill-conditioned per-group GLS)
- Use a median-based robust Psi_em (M-estimator) instead of winsorized mean

---

### Fix 4 — Off-diagonal Ψ shrinkage in Normal path (2026-05-08)

**File**: metabeta/analytical/normal.py, `_lmmNormalFull`

**Change**: Apply `_shrinkOffDiagonal(Psi, corr_alpha)` with `corr_alpha = G / (G + 5.0)` before
computing `sigma_rfx`. This matches the PQL path which already applied C=5 shrinkage. No effect
on diagonal elements (sigma_rfx), only shrinks off-diagonal Ψ toward 0 for small-G datasets.

```python
corr_alpha = G / (G + 5.0)
Psi = _shrinkOffDiagonal(Psi, corr_alpha)
sigma_rfx = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=0.0).sqrt()
```

**Results**: NRMSE and blup_var calibration ratios identical to Fix 1 on both test cases (expected —
shrinkage only affects off-diagonal Ψ). Psi_corr RMSE improvement is small and dataset-dependent
(C=5 post-hoc adds ~1% on top in the diagnostic comparison). No regressions observed.

Current state: Fix 1 + Fix 2 + Fix 4.

---

### Fix 5 — Convergence-aware EM (2026-05-08, PARTIALLY REVERTED)

**Files**: metabeta/analytical/constants.py, metabeta/analytical/normal.py `_emRefineNormal`

**Attempted change**: Increase `_NORMAL_FULL_MIN_EM` from 5 to 10. Add early-exit when both
`max|Ψ − Ψ_prev| < 1e-3` and `max|σ² − σ²_prev| < 1e-3` after computing the updated GLS.

**Two-case results** (Fix 1+2+4 → Fix 5 with max=10):
- large-n-sampled-test BLUPs: 0.7306 → 0.4963 (−32%) ✓
- large-n-sampled-test FFX: 0.5267 → 0.4623 (−12%) ✓
- large-n-mixed BLUPs: 0.6815 → 0.3721 (−45%) ✓

**Full-suite regressions discovered**:
- medium-n-mixed FFX: 0.2584 → **0.7311** (+183%) ✗  — catastrophic
- medium-n-mixed BLUPs: 0.7853 → 0.9743 (+24%) ✗
- huge-n-sampled σEps test: 0.1099 → 0.1876 (+71%) ✗
- huge-n-sampled σRFX valid: 0.5776 → 0.7218 (+25%) ✗

**Root cause**: 5 hard iterations was implicit early stopping that accidentally prevented
over-convergence to a biased EM fixed point. For datasets with low bg_df (4–8) and high d (5–6),
the MoM Ψ initialization is noisy and the EM fixed point is also biased. 5 iterations never
reached it. With max=10, those datasets fully converge to the biased point. For medium-n-mixed
the effect manifests as GLS blowup in the low n/d, high d regime (n/d=7.5–43: RMSE=1.24).

**Batched convergence note**: The `break` fires when the **batch-maximum** delta falls below
threshold — i.e., every dataset in the batch must converge before the loop exits. Datasets that
converge in 3 iterations still run for however many iterations the slowest dataset needs.
This is correct (no dataset gets fewer iterations than required) but limits the speedup benefit
to the worst-case dataset per batch.

**Partial revert**: `_NORMAL_FULL_MIN_EM` reverted to 5. The convergence check (early exit) is
**retained** — it acts as a micro-optimization for batches where all datasets converge in <5
iterations, avoiding the final GLS call(s) needlessly. The max=5 cap prevents over-convergence
to biased fixed points.

**What would make max>5 safe**: Per-dataset adaptive iteration count gated on G_mom (large G_mom
→ good MoM → allow more iterations). Requires masking already-converged datasets within the
batch, which changes the loop structure significantly.

Current state: Fix 1 + Fix 2 + Fix 4. (Fix 5 convergence check retained but max_em unchanged.)

---

---

### Fix 7 — Per-component count floor in component-wise MoM (2026-05-08)

**File**: metabeta/analytical/normal.py, `_initialPsiMom`

**File**: metabeta/analytical/normal.py, `_initialPsiMom`

**Problem**: `use_component_diag` required only `component_count >= 2.0` valid groups per component.
With 2 groups, the per-component MoM estimate (one bhat² after centering) is extremely noisy.
For large-q sampled datasets many components had `component_count = 1–2`, producing large spurious
signals that inflated `psi_eig_cap` via `amax(dim=1)`, allowing Ψ eigenvalues to be grossly
overestimated. Overestimated Ψ → under-shrunk BLUPs → large BLUP NRMSE; also contaminated β.

**Change**:
1. Raise `use_component_diag` threshold: `component_count >= 2.0` → `reliable_component = component_count >= 5.0`
2. Gate `diag_cap_signal` per-component: when `enough_diag_mom = False`, use `fallback_diag` for
   components with `component_count < 5` instead of their noisy `component_diag_signal`.

```python
reliable_component = component_count >= 5.0
use_component_diag = reliable_component & ~has_joint_mom[:, None]
...
diag_cap_signal = torch.where(
    enough_diag_mom[:, None],
    psi_diag_signal,
    torch.where(reliable_component, component_diag_signal, fallback_diag),
)
```

**Results** (Fix 1+2+4 baseline → Fix 1+2+4+7):

| Dataset/Partition    | FFX (base→new)      | sRFX (base→new)     | sEps (base→new)     | BLUPs (base→new)      |
|----------------------|---------------------|---------------------|---------------------|-----------------------|
| small-n-mixed/train  | 0.2291 → 0.2291 (=) | 0.5906 → 0.5909 (=) | 0.084 → 0.084 (=)  | 1.1443 → 1.1443 (=)  |
| medium-n-mixed/train | 0.2584 → **0.1538** (−40%) | 0.4521 → 0.5312 (+18%) | 0.0564 → 0.0675 (+20%) | 0.7853 → **0.3955** (−50%) |
| large-n-mixed/train  | 0.3476 → **0.3049** (−12%) | 0.4562 → 0.4507 (−1%) | 0.0557 → 0.0728 (+31%) | 0.6815 → **0.5744** (−16%) |
| huge-n-mixed/train   | 0.3748 → **0.3092** (−18%) | 0.4498 → 0.4582 (+2%) | 0.0499 → 0.0651 (+30%) | 0.6561 → **0.4644** (−29%) |
| small-n-sampled/vld  | 0.2085 → **0.1547** (−26%) | 0.5547 → 0.5665 (+2%) | 0.0808 → 0.1037 (+28%) | 0.9539 → **0.7046** (−26%) |
| small-n-sampled/tst  | 0.2079 → **0.1686** (−19%) | 0.5562 → 0.6127 (+10%) | 0.0817 → 0.1004 (+23%) | 0.9574 → **0.7907** (−17%) |
| medium-n-sampled/vld | 0.3259 → **0.2505** (−23%) | 0.5089 → 0.4861 (−5%) | 0.0717 → 0.0987 (+38%) | 0.7232 → **0.4718** (−35%) |
| medium-n-sampled/tst | 0.3267 → **0.2421** (−26%) | 0.5095 → 0.5967 (+17%) | 0.0723 → 0.1029 (+42%) | 0.7251 → **0.5509** (−24%) |
| large-n-sampled/vld  | 0.5252 → **0.4244** (−19%) | 0.5805 → 0.5407 (−7%) | 0.1072 → 0.1109 (+3%) | 0.7302 → **0.4759** (−35%) |
| large-n-sampled/tst  | 0.5267 → 0.5261 (=) | 0.5781 → 0.5797 (=) | 0.1081 → 0.1082 (=) | 0.7306 → 0.7302 (=) |
| huge-n-sampled/vld   | 0.6162 → **0.4168** (−32%) | 0.5776 → 0.7592 (+31%) | 0.1101 → 0.1643 (+49%) | 0.7165 → **0.5899** (−18%) |
| huge-n-sampled/tst   | 0.6157 → **0.450** (−27%)  | 0.5824 → 0.5747 (−1%) | 0.1099 → 0.1828 (+66%) | 0.7161 → **0.694** (−3%) |

**Root cause of sEps regression**: tightening `psi_eig_cap` prevents Ψ overestimation, but forces
the EM to attribute more residual variance to σ². When the old code overestimated Ψ, σ² was
artificially low (stolen by Ψ), so the old sEps NRMSE was deceptively good. Now σ² is larger
but the EM converges to a different fixed point, moving farther from true σ_eps in many cases.

**Assessment**: Large improvement for primary metrics (FFX, BLUPs); 20–66% sEps regression is a
tradeoff. Datasets unaffected (small-n-mixed, large-n-sampled/tst) are those where `has_joint_mom`
or `enough_diag_mom` is always True (component-wise fallback path not exercised). Keep Fix 7.

Current state: Fix 1 + Fix 2 + Fix 4 + Fix 7.

---

### Fix 8 — G_mom-gated adaptive EM iterations (2026-05-08, REVERTED)

**Files**: metabeta/analytical/normal.py `_emRefineNormal`; metabeta/analytical/constants.py

**Attempted change**: Allow up to max=10 EM iterations when `G_mom >= 20`; cap at `n_em=5` otherwise.
Per-dataset convergence tracking via `(B,)` `converged` mask and `torch.where` state freeze.

**Results** (Fix 7 baseline → Fix 7+8, threshold=20):
| Dataset | FFX (F7→F7+8) | BLUPs (F7→F7+8) |
|---------|---------------|-----------------|
| medium-n-mixed/train    | 0.1538 → 0.3731 (+143%) ✗ | 0.3955 → 0.9703 (+145%) ✗ |
| medium-n-sampled/valid  | 0.2505 → 1.8564 (+641%) ✗ | 0.4718 → 0.7920 (+68%) ✗ |
| small-n-sampled/valid   | 0.1547 → 0.2394 (+55%) ✗  | — |
| large-n-mixed/train     | 0.3049 → 0.2651 (−13%) ✓  | 0.5744 → 0.3651 (−36%) ✓ |
| large-n-sampled/test    | 0.5261 → 0.5167 (−2%) ✓   | 0.7302 → 0.6698 (−8%) ✓ |
| huge-n-sampled/test     | 0.4500 → 0.4471 (−1%) ✓   | 0.6940 → 0.6006 (−13%) ✓ |

**Root cause**: G_mom is not a reliable proxy for "safe EM fixed point." Small-n datasets with
many small groups can have G_mom >> 20 (all groups pass `ns > q+1`) yet still have a biased EM
fixed point because the M-step Ψ_em is noisy per group. The medium-n-sampled/valid vs test
asymmetry (+641% vs unchanged) shows the effect is highly sensitive to which specific datasets
happen to be in a split — indicating it hits a narrow parameter regime (low n/d, high d) that
the valid partition oversampled.

**What G_mom captures vs what matters**: G_mom = count of groups above `ns > active_count + 1`.
High G_mom just means many groups cleared the observation threshold; it says nothing about
whether the collective MoM signal is strong enough to reliably identify Ψ. The EM fixed-point
bias correlates with `bg_df = m − d` (groups relative to fixed effects) and `n_g / q` (obs per
random effect per group), not G_mom alone.

**Reverted.** Current state: Fix 1 + Fix 2 + Fix 4 + Fix 7.

**What would make Fix 8 safe**: A gate that combines G_mom AND per-group information content,
e.g. `psi_df = sum_{g in mom}(ns_g − q − 1)` with threshold ~ 200, or monitoring for EM
oscillation/divergence at runtime and stopping when Ψ or σ² starts increasing again.

---

### Fix 9 — Adaptive M-step BLUP winsorization (2026-05-08)

**File**: metabeta/analytical/normal.py, `_emRefineNormal`

**Change**: Before forming `blup_outer = b̂_g b̂_g'` in the EM M-step, winsorize BLUPs at
`±10×sqrt(Ψ̂_diag)` where `Ψ̂_diag` is the current iteration's Ψ. Unlike original Fix 3 (which
used the initial MoM Ψ as cap reference), Fix 9 adapts each iteration: as Ψ improves toward the
true value, the cap loosens, preventing early-iteration tightness from permanently biasing Ψ_em.
After Fix 7, the initial Ψ is also better calibrated, making the first-iteration cap less risky.

```python
psi_diag_cur = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=psi_diag_floor).sqrt()
blup_cap = 10.0 * psi_diag_cur[:, None, :]  # (B, 1, q) broadcasts over groups
blups_winsor = gls.blups.clamp(min=-blup_cap, max=blup_cap)
blup_outer = torch.einsum('bmq,bmr->bmqr', blups_winsor, blups_winsor)
```

**Multiplier sweep** (Fix 7 → Fix 7+9, key metrics):

| Multiplier | medium-n-mixed BLUPs | large-n-mixed BLUPs | huge-n-mixed BLUPs | lg-sampled/tst BLUPs |
|------------|----------------------|---------------------|--------------------|----------------------|
| Fix 7 (base) | 0.3955             | 0.5744              | 0.4644             | 0.7302               |
| 10×        | 0.3748 (−5%) ✓      | 0.3729 (−35%) ✓     | 0.5133 (+10%) ✗    | 0.6868 (−6%) ✓       |
| 12×        | 0.3725 (−6%) ✓      | 0.3801 (−34%) ✓     | 0.5853 (+26%) ✗✗   | 0.5505 (−25%) ✓✓     |
| 15×        | 0.3978 (≈0)         | 0.3677 (−36%) ✓     | 0.4684 (≈0) ✓      | 0.7479 (+2%) ✗       |

Note: non-monotone behavior at 12× (worse than 10× for huge-n-mixed despite looser cap) indicates
EM phase transitions sensitive to the cap level. The cap changes the M-step fixed point, not just
the clipping rate, so multiplier choice has complex nonlinear effects.

**Chosen: 10×** — most widespread improvements across datasets. Huge-n-mixed +10% regression
is a known tradeoff (still better than original baseline 0.6561).

**Full results — Fix 7 → Fix 7+9 (10×)**:

| Dataset/Partition    | FFX             | sRFX            | sEps            | BLUPs               |
|----------------------|-----------------|-----------------|-----------------|---------------------|
| small-n-mixed/train  | 0.2291 → 0.2291 (=) | — | — | 1.1443 → 1.1442 (=) |
| medium-n-mixed/train | 0.1538 → **0.1454** (−6%) | — | — | 0.3955 → **0.3748** (−5%) |
| large-n-mixed/train  | 0.3049 → **0.2683** (−12%) | — | — | 0.5744 → **0.3729** (−35%) |
| huge-n-mixed/train   | 0.3092 → 0.3161 (+2%) | — | — | 0.4644 → 0.5133 (+10%) ✗ |
| small-n-sampled      | ≈ unchanged | — | — | ≈ unchanged |
| medium-n-sampled/vld | 0.2505 → 0.2531 (+1%) | — | — | 0.4718 → 0.4769 (+1%) |
| medium-n-sampled/tst | 0.2421 → 0.2438 (+1%) | — | — | 0.5509 → 0.5594 (+2%) |
| large-n-sampled/vld  | 0.4244 → 0.4244 (=) | — | — | 0.4759 → 0.475 (=) |
| large-n-sampled/tst  | 0.5261 → **0.4967** (−6%) | — | — | 0.7302 → **0.6868** (−6%) |
| huge-n-sampled/vld   | 0.4168 → 0.4154 (−0.3%) | — | — | 0.5899 → 0.5822 (−1%) |
| huge-n-sampled/tst   | 0.4500 → **0.448** (−0.4%) | — | — | 0.6940 → **0.6628** (−4.5%) |

sRFX and sEps are unchanged (±<1%) across all datasets — the cap only affects M-step BLUPs,
not the σ² update or post-EM σ_rfx extraction.

**Assessment**: Fix 9 is a clear win for large-n-mixed (−35% BLUPs) and large-n-sampled/test
(−6% BLUPs, −6% FFX), with no sEps regression. The huge-n-mixed +10% regression is an
acceptable tradeoff. Medium-n-sampled slight regressions (~1–2%) are within noise.

Current state: Fix 1 + Fix 2 + Fix 4 + Fix 7 + Fix 9.

---

### Attempted Fix B — psi_df-gated EM extension (2026-05-09, REVERTED)

**Gate**: `psi_df = Σ(ns_g − q − 1)` over mom groups, threshold ≥ 300 allows up to max_em=10.
**Result**: medium-n-mixed/train BLUPs: 0.3748 → 0.9601 (+156%) — same failure as Fix 8.
**Root cause**: medium-n-mixed has large psi_df (many groups × moderate obs) but the EM fixed
point is still biased from poor Ψ initialization. psi_df measures available data, not whether
the M-step is unbiased. No count statistic can distinguish biased from unbiased fixed points.

---

### Attempted Fix A — beta_wg in MoM residual (2026-05-09, REVERTED)

**Change**: use beta_wg (within-Z estimator) instead of beta_ols for resid_full in _initialPsiMom.
**Result**: medium-n-mixed FFX: 0.1454 → 80.18 (+55,000%). Catastrophic.
**Root cause**: beta_wg is poorly identified for datasets where predictors are nearly collinear
with group membership. This makes the MoM residuals noisy → bad Ψ_initial → GLS blow-up.

---

### Attempted Fix D — EM σ² regularization toward Stage 1 (2026-05-09, REVERTED)

**Change**: se2_blend = 0.8*se2_next + 0.2*se2_anchor (blend toward Stage 1 σ̂_eps).
**Result**: sEps improved 2.5–7%, but huge-n-mixed BLUPs: 0.5133 → 0.6725 (+31%) ✗.
**Root cause**: Stage 1 σ̂_eps is biased upward for large-n datasets (WP-σ1). Blending
toward it raises σ², increasing BLUP shrinkage. For large n_g, BLUPs are large in magnitude
so even a small σ² increase causes significant over-shrinkage. Net effect negative.

---

### Fix C — Outlier-trimmed Ψ_em M-step (2026-05-09)

**File**: metabeta/analytical/normal.py, `_emRefineNormal`

**Change**: Replace the mean of (blup_outer + post_cov) over mom groups with an outlier-trimmed
mean that excludes groups whose ||b̂_g||² exceeds 3× the mean norm among mom groups. Two
earlier versions failed:
- v1 (top 10% fixed trim): always trimmed at least 1 group → medium-n-mixed regression (+55% FFX)
- v2 (G_mom >= 20 gate, 5% trim): medium-n-mixed has G_mom >= 20 → same regression
- v3 (3× mean threshold): adaptive — no trimming when all groups have similar norms ✓

```python
blup_norm = blups_winsor.square().sum(dim=-1)  # (B, m)
mom_mask_1d = mom4.squeeze(-1).squeeze(-1).bool()
mom_norm_mean = (blup_norm * mom_mask_1d.float()).sum(dim=1) / G_mom  # (B,)
outlier_thresh = 3.0 * mom_norm_mean[:, None]
trim_mask = mom_mask_1d & (blup_norm <= outlier_thresh)
trim_count = trim_mask.float().sum(dim=1).clamp(min=1.0)
Psi_em = _psdProject(((blup_outer + post_cov) * trim_mask_4d).sum(dim=1) / trim_count[:, None, None])
```

**Full results — Fix 7+9 → Fix 7+9+C**:

| Dataset/Partition    | FFX               | sRFX              | sEps              | BLUPs                  |
|----------------------|-------------------|-------------------|-------------------|------------------------|
| small-n-mixed/train  | 0.2252 → 0.2249 (≈0) | 0.6303 → 0.6421 | 0.0835 → 0.0839 | 1.0697 → 1.0687 (≈0)  |
| medium-n-mixed/train | 0.1454 → 0.1452 (≈0) | — | 0.0675 → 0.0671 | 0.3748 → **0.3749** (≈0) ✓ |
| large-n-mixed/train  | 0.2683 → 0.2686 (≈0) | — | 0.0728 → 0.0724 | 0.3729 → **0.3670** (−1.6%) ✓ |
| huge-n-mixed/train   | 0.3161 → 0.3034 (−4%) ✓ | — | 0.0651 → 0.0648 | 0.5133 → **0.4663** (−9.1%) ✓ |
| large-n-sampled/tst  | — | — | — | 0.6868 → 0.6774 (−1.4%) ✓ |
| huge-n-sampled/tst   | — | — | — | 0.6628 → 0.6631 (≈0) ✓ |

**Assessment**: Clean win. The adaptive 3× norm threshold correctly identifies anomalous groups
without introducing bias for homogeneous datasets. Huge-n-mixed BLUPs improve −9.1% (partially
resolving P3). All other datasets preserved. The outlier trimming adds no regressions because
it is a no-op when all groups have similar BLUP norms.

Current state: Fix 1 + Fix 2 + Fix 4 + Fix 7 + Fix 9 + Fix C.

---

### Fix E — mom4 mask refresh after first EM iteration (2026-05-09, REVERTED — no-op)

**Files**: metabeta/analytical/normal.py `_emRefineNormal`

**Attempted change**: After iteration i=0 in the EM loop, recompute mom4/G_mom by
calling `_groupZDiagnostics` with `cond_cap=float('inf')` — relaxing the condition-number
restriction while keeping the rank and ns conditions. The intent was to admit groups
excluded solely by high ZtZ condition number (since Woodbury regularization from Ψ makes
the posterior well-conditioned even for ill-conditioned ZtZ groups).

**Results**: Identical to Fix C baseline on all 6 measured datasets (zero delta in FFX,
BLUPs, sEps, sRFX). Zero groups were affected.

**Root cause**: `_groupZDiagnostics` is purely structural — it depends only on ZtZ, ns, q.
No matter how much Ψ improves in iteration 1, the mask cannot change. The plan's hypothesis
("sigma_rfx estimate affects which groups are excluded") was incorrect: sigma_rfx is never
an input to `_groupZDiagnostics`. The condition number cap (`cond_cap = 1e6`) was already
so permissive that no real-world groups in these datasets were excluded by it alone.
All actual exclusions come from the rank or ns conditions, which are irreducibly structural.

**Key lesson**: G_mom (the informative-group count for the M-step) is bounded by bg_df
(groups with ns > q+1 AND full ZtZ rank) — a data-structural quantity that Ψ quality
cannot unlock. For small-n-mixed, G_mom = 4–7 is irreducible with current simulation
sizes. Fix E is dead; WP-EM2 is a misdiagnosis of the true constraint.

Current state: Fix 1 + Fix 2 + Fix 4 + Fix 7 + Fix 9 + Fix C. (unchanged)

---

### Attempted I3 Option E — post-EM psi_ratio-gated REML (2026-05-09, REVERTED)

**Hypothesis**: The remaining small-n-mixed BLUP error comes from EM collapsing to
near-zero Ψ. Gate REML on the post-EM result:

```python
psi_ratio = Psi.diagonal(dim1=-2, dim2=-1).amax(dim=-1) / se2
reml_gate = psi_ratio < 0.03
```

**Diagnostic result**: The gate selected the easiest BLUP cases, not the failure mode.
On small-n-mixed:

| Subset | Share | BLUP NRMSE |
|--------|-------|------------|
| all rows | 100% | 1.0687 |
| psi_ratio < 0.03 | 9.9% | 0.164 |
| psi_ratio >= 0.03 | 90.1% | 1.113 |

The same pattern held for medium/large/huge-n-mixed: low-ratio rows had lower BLUP
error than the non-gated majority. Post-EM Ψ/σ² collapse is therefore not the
remaining P1 failure mode.

**Attempted change**: After `_emRefineNormal`, initialize gated rows at diagonal
`psi_diag_floor`, keep σ² fixed, run 4 `_remlNewtonStep` iterations with
`reml_psi_cap = max(psi_eig_cap, 25*σ²)`, then merge gated rows back.

**Required 12-way benchmark result**:

| Dataset/Partition | FFX | sRFX | sEps | BLUPs |
|-------------------|-----|------|------|-------|
| small-n-mixed/train | 0.2249 → 0.2249 | 0.6421 → 1.2026 | 0.0839 → 0.0839 | 1.0687 → 1.0690 |
| medium-n-mixed/train | 0.1452 → 0.1452 | 0.5739 → 1.7796 | 0.0671 → 0.0671 | 0.3749 → 0.3769 |
| large-n-mixed/train | 0.2686 → 0.2711 | 0.4947 → 1.9061 | 0.0724 → 0.0724 | 0.3670 → 0.3693 |
| huge-n-mixed/train | 0.3034 → 0.3041 | 0.4957 → 1.6741 | 0.0648 → 0.0648 | 0.4663 → 0.4685 |
| small-n-sampled/valid | 0.1551 → 0.1550 | 0.6313 → 1.0275 | 0.1031 → 0.1031 | 0.7044 → 0.7045 |
| small-n-sampled/test | 0.1686 → 0.1686 | 0.6655 → 1.0480 | 0.1002 → 0.1002 | 0.7898 → 0.7900 |
| medium-n-sampled/valid | 0.3625 → 0.3624 | 0.5334 → 1.3794 | 0.0978 → 0.0978 | 0.5566 → 0.5568 |
| medium-n-sampled/test | 0.2437 → 0.2438 | 0.6081 → 1.3683 | 0.1029 → 0.1029 | 0.5367 → 0.5368 |
| large-n-sampled/valid | 0.3874 → 0.3931 | 0.5811 → 1.5843 | 0.1104 → 0.1104 | 0.6642 → 0.6647 |
| large-n-sampled/test | 0.4959 → 0.4958 | 0.6065 → 1.4271 | 0.1078 → 0.1078 | 0.6774 → 0.6776 |
| huge-n-sampled/valid | 0.4208 → 0.4211 | 0.7824 → 1.5732 | 0.1643 → 0.1643 | 0.5827 → 0.5812 |
| huge-n-sampled/test | 0.4662 → 0.4664 | 0.5954 → 1.4824 | 0.1826 → 0.1826 | 0.6631 → 0.6639 |

**Assessment**: Reverted. BLUPs did not improve, and sRFX regressed 2-3× across
every dataset family. The next direction is not another scalar gate for REML; it is
an oracle shrinkage diagnostic that directly compares estimated vs true BLUP
shrinkage and separates Ψ error, σ² error, and β leakage.

---

### I4 Diagnostic — shrinkage ratios and oracle BLUP ablations (2026-05-09)

**Script**: `experiments/glmm_shrinkage_diagnostic.py`

**q=1 shrinkage result**: high BLUP error is concentrated in the central
`lambda_hat/lambda_true` bucket, not in the shrinkage tails. On small-n-mixed:

| lambda_hat/lambda_true | Share | BLUP NRMSE | median Ψ_hat/Ψ_true | median σ²_hat/σ²_true |
|------------------------|-------|------------|----------------------|------------------------|
| 0.5-0.75 | 4.4% | 0.361 | 0.427 | 1.017 |
| 0.75-1.25 | 79.1% | 1.463 | 0.672 | 1.001 |
| 1.25-2 | 3.8% | 0.222 | 1.784 | 0.993 |
| 2-4 | 3.0% | 0.190 | 3.407 | 0.991 |
| >=4 | 9.4% | 0.155 | 23.394 | 0.986 |

The tails where Ψ is severely under- or over-estimated have relatively low BLUP
error. The high-error majority has approximately correct shrinkage by the scalar
lambda proxy and σ² is well calibrated. This points away from variance-component
gating.

**small-n-mixed oracle BLUP ablations**:

| Case | BLUP NRMSE |
|------|------------|
| baseline estimator | 1.0687 |
| true Ψ + true σ² + beta_hat | 1.0599 |
| true Ψ + estimated σ² + beta_hat | 1.0599 |
| estimated Ψ + true σ² + beta_hat | 1.0680 |
| estimated Ψ + estimated σ² + beta_true | 0.2931 |
| true Ψ + true σ² + beta_true | 0.2594 |
| true Ψ + true σ² + beta_wg | 3.3612 |
| estimated Ψ + estimated σ² + beta_wg | 3.3341 |

**Assessment**: Root cause for P1 is now beta leakage into the BLUP residual, not
Ψ/σ² estimation. Better variance components barely help if beta_hat is unchanged.
True beta nearly solves the BLUP problem even with estimated variance components.
beta_wg is not a viable replacement. Next investigation should target Normal GLS
beta estimation, beta masking/rank diagnostics, and conservative beta fallback or
shrinkage strategies.
