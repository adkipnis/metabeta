"""NPE failure mode analysis: detailed per-dataset error and failure characterization.

Compares MB (amortized NPE), NUTS (gold-standard MCMC), and ADVI (variational)
at the per-dataset level to identify and characterize failure modes specific to
amortized posterior estimation.

Unlike evaluate.py (which produces aggregate tables and plots), this reports:
  1. Distribution statistics for per-dataset NRMSE, coverage error, and LOO-NLL
  2. Head-to-head error ratios: MB vs NUTS, MB vs ADVI (per dataset)
  3. Coverage calibration at each alpha level for all three methods
  4. Worst-K failure cases: MB datasets with highest combined error vs NUTS
  5. GLMM diagnostics (from glmm.py) correlated with NPE failure
  6. Dataset property correlations with MB estimation error

Run from the repo root:
    uv run python benchmarks/bench_npe.py
    uv run python benchmarks/bench_npe.py --checkpoint <path> [--device cuda]
"""

import numpy as np
import torch
from scipy.stats import spearmanr

from metabeta.evaluation.evaluate import Evaluator
from metabeta.evaluation.evaluate import setup as evaluatorSetup
from metabeta.evaluation.intervals import ALPHAS, getCredibleInterval
from metabeta.evaluation.point import getPointEstimates
from metabeta.evaluation.predictive import (
    getPosteriorPredictive,
    psisLooNLL,
)
from metabeta.utils.dataloader import toDevice
from metabeta.utils.evaluation import Proposal, getMasks
from metabeta.utils.glmm import glmm
from metabeta.utils.preprocessing import rescaleData

SEP = '=' * 80
THIN = '-' * 60
TOP_K = 10      # failure cases to list in detail
EPS_CI = 1e-6   # tolerance for inside-CI check


# ─── Formatting helpers ───────────────────────────────────────────────────────


def _pct(arr: np.ndarray, q: float) -> float:
    return float(np.nanpercentile(arr, q * 100))


def _dist(arr: np.ndarray, label: str = '', indent: str = '  ') -> None:
    """Print median [Q25, Q75] (Q10, Q90) mean for a 1-D array."""
    med = np.nanmedian(arr)
    q25, q75 = _pct(arr, 0.25), _pct(arr, 0.75)
    q10, q90 = _pct(arr, 0.10), _pct(arr, 0.90)
    mean = float(np.nanmean(arr))
    nan_frac = float(np.isnan(arr).mean())
    tag = f'  nan={nan_frac:.1%}' if nan_frac > 0 else ''
    prefix = f'{indent}{label:<16}' if label else indent
    print(
        f'{prefix}  med={med:7.4f}  [Q25={q25:7.4f}, Q75={q75:7.4f}]'
        f'  [Q10={q10:7.4f}, Q90={q90:7.4f}]  mean={mean:7.4f}{tag}'
    )


def _corr(x: np.ndarray, y: np.ndarray, label: str, indent: str = '    ') -> None:
    """Print Spearman ρ between x and y, skipping NaN."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return
    rho, pval = spearmanr(x[ok], y[ok])
    stars = (
        '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
    )
    print(f'{indent}{label:<50s}  ρ={rho:+.3f}  p={pval:.2e}  {stars}')


def _section(title: str) -> None:
    print()
    print(SEP)
    print(f'  {title}')
    print(SEP)


# ─── Per-dataset NRMSE ───────────────────────────────────────────────────────


def _maskedStd(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-last-dim std across all other dims, matching maskedStd in point.py."""
    dims = tuple(range(x.dim() - 1))
    n = mask.float().sum(dims).clamp_min(1.0)
    mean = (x * mask.float()).sum(dims) / n
    sq_diff = (x - mean).square() * mask.float()
    return (sq_diff.sum(dims) / n).clamp_min(0.0).sqrt()


def perDatasetNRMSE(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    """Per-dataset NRMSE for each parameter group.

    Uses the same per-last-dimension normalization as getRMSE in point.py:
    std_d is computed across all B datasets for each parameter dimension d
    separately.  Avoids division-by-zero for scalar parameters (q=1, d=1)
    because the std is across datasets, not within a single dataset.

    Note: evaluate.py reports mean_d(sqrt(mean_B(err²)) / std_d) — aggregate
    over B first, then dims.  This benchmark reports
    median_B(sqrt(mean_d(err²/std_d²))) — aggregate over dims first, then
    datasets.  Because sqrt(mean(x²)) ≥ mean(sqrt(x²)) and median ≤ mean for
    right-skewed distributions, the benchmark median will always be lower than
    evaluate.py's figure even with identical normalization.

    Returns {param_key: (B,) float32 array}.
    """
    est = getPointEstimates(proposal, 'mean')
    masks = getMasks(data, proposal.has_sigma_eps)
    out: dict[str, np.ndarray] = {}

    for key, est_k in est.items():
        if key == 'corr_rfx':
            continue

        gt = data[key]
        mask = masks.get(key)

        if key == 'sigma_eps':
            gt = gt.unsqueeze(-1)
            est_k = est_k.unsqueeze(-1)

        se = (gt - est_k).square()

        if key == 'rfx':
            mask_mq = masks['rfx']  # (B, m, q)
            # per-q std across (B, m) — same as getRMSE's maskedStd with dims=(0,1)
            std_per_q = _maskedStd(gt, mask_mq).clamp_min(1e-8)     # (q,)
            mask_f = mask_mq.float()
            n_act = mask_f.sum(dim=(1, 2)).clamp_min(1.0)            # (B,)
            se_normed = se / std_per_q.square()                       # (B, m, q)
            mse_normed_b = (se_normed * mask_f).sum(dim=(1, 2)) / n_act
            out[key] = mse_normed_b.sqrt().numpy()

        elif mask is not None:
            mask_f = mask.float()
            # per-dim std across B — same as getRMSE's maskedStd with dims=(0,)
            std_per_dim = _maskedStd(gt, mask_f).clamp_min(1e-8)    # (d,) or (q,)
            n_act = mask_f.sum(-1).clamp_min(1.0)                    # (B,)
            se_normed = se / std_per_dim.square()                     # (B, d)
            mse_normed_b = (se_normed * mask_f).sum(-1) / n_act      # (B,)
            out[key] = mse_normed_b.sqrt().numpy()

        else:
            # sigma_eps — std across B, same as getRMSE's gt.std(0, unbiased=False)
            mse_b = se.squeeze(-1)                                    # (B,)
            norm = float(gt.squeeze(-1).std(unbiased=False).clamp_min(1e-8))
            out[key] = (mse_b.sqrt() / norm).numpy()

    return out


def combinedNRMSE(nrmse: dict[str, np.ndarray], exclude: tuple[str, ...] = ('rfx',)) -> np.ndarray:
    """Per-dataset NRMSE averaged over all global parameter groups."""
    arrs = [v for k, v in nrmse.items() if k not in exclude]
    return np.stack(arrs).mean(0) if arrs else np.zeros(1)


# ─── Per-dataset coverage error ───────────────────────────────────────────────


def perDatasetCoverageError(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
    alphas: list[float] = ALPHAS,
) -> dict[str, np.ndarray]:
    """Mean absolute (coverage − nominal) per dataset, averaged over alphas.

    Returns {param_key: (B,) float32 array}.
    """
    masks = getMasks(data, proposal.has_sigma_eps)
    acc: dict[str, list[np.ndarray]] = {}

    for alpha in alphas:
        nominal = 1.0 - alpha
        ci_dict = getCredibleInterval(proposal, alpha)

        for key, ci in ci_dict.items():
            if key == 'corr_rfx':
                continue

            gt = data[key]
            mask = masks.get(key)

            if key == 'sigma_eps':
                gt = gt.unsqueeze(-1)
                ci = ci.unsqueeze(-1)

            # ci: (B, [m,] 2, d)  –  lower at [..., 0, :], upper at [..., 1, :]
            above = ci[..., 0, :] - EPS_CI <= gt
            below = gt <= ci[..., 1, :] + EPS_CI
            inside = (above & below).float()

            if key == 'rfx':
                mask_mq = masks['rfx']
                n_act = mask_mq.float().sum(dim=(1, 2)).clamp_min(1.0)
                cov_b = (inside * mask_mq).sum(dim=(1, 2)) / n_act
            elif mask is not None:
                mask_f = mask.float()
                n_act = mask_f.sum(-1).clamp_min(1.0)
                cov_b = (inside * mask_f).sum(-1) / n_act
            else:
                cov_b = inside.squeeze(-1)

            err_b = (cov_b - nominal).abs().numpy()
            acc.setdefault(key, []).append(err_b)

    return {k: np.stack(v).mean(0) for k, v in acc.items()}


# ─── Per-dataset LOO-NLL ──────────────────────────────────────────────────────


def perDatasetLooNLL(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
    likelihood_family: int,
) -> np.ndarray:
    """PSIS-LOO NLL per dataset. Returns (B,) float32 array."""
    pp = getPosteriorPredictive(proposal, data, likelihood_family)
    loo_nll, _ = psisLooNLL(pp, data, w=proposal.weights)
    return loo_nll.numpy()


# ─── Coverage calibration table ───────────────────────────────────────────────


def printCoverageCalibration(
    proposals: dict[str, Proposal],
    data: dict[str, torch.Tensor],
) -> None:
    """Print overall empirical coverage at each alpha level for all methods."""
    _section('COVERAGE CALIBRATION BY ALPHA LEVEL  (empirical vs nominal)')
    masks = getMasks(data)
    header = f'  {"alpha":>6}  {"nominal":>8}'
    for label in proposals:
        header += f'  {label + "_cov":>10}  {label + "_ECE":>8}'
    print(header)
    print(f'  {THIN}')

    for alpha in ALPHAS:
        nominal = 1.0 - alpha
        row = f'  {alpha:6.2f}  {nominal:8.4f}'
        for label, proposal in proposals.items():
            ci_dict = getCredibleInterval(proposal, alpha)
            covs: list[float] = []
            for key, ci in ci_dict.items():
                if key == 'corr_rfx':
                    continue
                gt = data[key]
                mask = masks.get(key)
                if key == 'sigma_eps':
                    gt = gt.unsqueeze(-1)
                    ci = ci.unsqueeze(-1)
                inside = (
                    (ci[..., 0, :] - EPS_CI <= gt) & (gt <= ci[..., 1, :] + EPS_CI)
                ).float()
                if key == 'rfx':
                    m = masks['rfx']
                    cov = (inside * m).sum() / m.float().sum().clamp_min(1.0)
                elif mask is not None:
                    m = mask.float()
                    cov = (inside * m).sum() / m.sum().clamp_min(1.0)
                else:
                    cov = inside.mean()
                covs.append(cov.item())
            actual = float(np.mean(covs))
            ece = actual - nominal
            row += f'  {actual:10.4f}  {ece:+8.4f}'
        print(row)


# ─── GLMM diagnostics ────────────────────────────────────────────────────────


def runGLMMDiagnostics(
    data: dict[str, torch.Tensor],
    likelihood_family: int,
) -> dict[str, np.ndarray] | None:
    """Run glmm.py on the full batch using the same API as diagnose_glmm.py.

    The standard test batch stores grouped/padded tensors under keys X, y, Z
    (shape B × m_max × n_max × d/q), which is exactly what glmm() expects.

    Returns a dict of per-dataset (B,) arrays, or None if tensors are missing.

    Per-dataset NRMSE metrics (same normalisation as perDatasetNRMSE):
      glmm_ffx_nrmse      — GLMM beta vs true ffx
      glmm_sigma_nrmse    — GLMM sigma_rfx vs true
      glmm_blup_nrmse     — BLUPs vs true rfx
      glmm_seps_nrmse     — sigma_eps vs true  (Normal only)
      glmm_combined_nrmse — mean(ffx, sigma[, seps]) — for ranking/correlation
    BLUP calibration:
      glmm_blup_var_ratio — mean(err²) / mean(blup_var) per dataset;
                            > 1 → GLMM overconfident (BLUPs too shrunk)
                            < 1 → GLMM underconfident
    Family-specific diagnostics (GLMM only):
      phi_pearson         — overdispersion estimate
      psi_0               — initial variance-component estimate
    """
    required = ['X', 'y', 'Z', 'mask_n', 'mask_m', 'ns', 'n']
    missing = [k for k in required if k not in data]
    if missing:
        print(f'  (GLMM diagnostics unavailable: missing tensors {missing})')
        return None

    max_q = data['sigma_rfx'].shape[-1]
    Zm = data['Z'][..., :max_q]

    try:
        result = glmm(
            data['X'], data['y'], Zm,
            data['mask_n'].float(), data['mask_m'].float(),
            data['ns'].clamp(min=1).float(), data['n'].float(),
            likelihood_family=likelihood_family,
            eta_rfx=data.get('eta_rfx'),
        )
    except Exception as exc:
        print(f'  [warn] GLMM failed: {exc}')
        return None

    diag: dict[str, np.ndarray] = {}

    # ── FFX ───────────────────────────────────────────────────────────────────
    beta_est = result['beta_est'].cpu()      # (B, d)
    gt_ffx   = data['ffx']                   # (B, d)
    mask_d   = data['mask_d']
    std_d    = _maskedStd(gt_ffx, mask_d).clamp_min(1e-8)        # (d,)
    mask_f_d = mask_d.float()
    n_d      = mask_f_d.sum(-1).clamp_min(1.0)                   # (B,)
    se_normed = (beta_est - gt_ffx).square() / std_d.square()    # (B, d)
    diag['glmm_ffx_nrmse'] = (
        (se_normed * mask_f_d).sum(-1).div(n_d).sqrt().numpy()
    )

    # ── Sigma_rfx ─────────────────────────────────────────────────────────────
    sigma_est = result['sigma_rfx_est'].cpu()  # (B, q)
    gt_sigma  = data['sigma_rfx']               # (B, q)
    mask_q    = data['mask_q']
    std_q     = _maskedStd(gt_sigma, mask_q).clamp_min(1e-8)     # (q,)
    mask_f_q  = mask_q.float()
    n_q       = mask_f_q.sum(-1).clamp_min(1.0)                  # (B,)
    se_normed = (sigma_est - gt_sigma).square() / std_q.square()
    diag['glmm_sigma_nrmse'] = (
        (se_normed * mask_f_q).sum(-1).div(n_q).sqrt().numpy()
    )

    # ── BLUPs ─────────────────────────────────────────────────────────────────
    blup_est  = result['blup_est'].cpu()   # (B, m, q)
    gt_rfx    = data['rfx']                # (B, m, q)
    mask_mq   = data['mask_mq']
    std_mq    = _maskedStd(gt_rfx, mask_mq).clamp_min(1e-8)      # (q,)
    mask_f_mq = mask_mq.float()
    n_mq      = mask_f_mq.sum(dim=(1, 2)).clamp_min(1.0)         # (B,)
    blup_err  = blup_est - gt_rfx
    se_normed = blup_err.square() / std_mq.square()
    diag['glmm_blup_nrmse'] = (
        (se_normed * mask_f_mq).sum(dim=(1, 2)).div(n_mq).sqrt().numpy()
    )

    # ── BLUP variance calibration ─────────────────────────────────────────────
    blup_var  = result['blup_var'].cpu()   # (B, m, q)
    mse_b     = (blup_err.square() * mask_f_mq).sum(dim=(1, 2)) / n_mq
    mvar_b    = (blup_var * mask_f_mq).sum(dim=(1, 2)) / n_mq
    diag['glmm_blup_var_ratio'] = (mse_b / mvar_b.clamp_min(1e-30)).numpy()

    # ── Sigma_eps (Normal only) ───────────────────────────────────────────────
    if likelihood_family == 0 and 'sigma_eps_est' in result:
        seps_est = result['sigma_eps_est'].squeeze(-1).cpu()  # (B,)
        gt_seps  = data['sigma_eps']                           # (B,)
        norm_eps = float(gt_seps.std(unbiased=False).clamp_min(1e-8))
        diag['glmm_seps_nrmse'] = (
            (seps_est - gt_seps).abs().div(norm_eps).numpy()
        )

    # ── Combined NRMSE for ranking (global params only, matching combinedNRMSE) ─
    combined_keys = ['glmm_ffx_nrmse', 'glmm_sigma_nrmse']
    if 'glmm_seps_nrmse' in diag:
        combined_keys.append('glmm_seps_nrmse')
    diag['glmm_combined_nrmse'] = np.stack([diag[k] for k in combined_keys]).mean(0)

    # ── Family-specific diagnostics ───────────────────────────────────────────
    if 'phi_pearson' in result:
        diag['phi_pearson'] = result['phi_pearson'].cpu().numpy()
    if 'psi_0' in result:
        diag['psi_0'] = result['psi_0'].cpu().numpy()

    return diag


# ─── Dataset property extraction ─────────────────────────────────────────────


def datasetProperties(data: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Extract per-dataset structural properties useful for failure analysis."""
    mask_m = data['mask_m'].numpy().astype(bool)
    ns = data['ns'].numpy()

    props: dict[str, np.ndarray] = {}
    n_groups = mask_m.sum(-1).astype(float)
    props['n_groups'] = n_groups

    active_ns = np.where(mask_m, ns, 0.0)
    props['mean_obs_per_group'] = active_ns.sum(-1) / n_groups.clip(1)
    # minimum observations across active groups (proxy for "hardest" group)
    ns_active = np.where(mask_m, ns, ns.max())
    props['min_obs_per_group'] = ns_active.min(-1).astype(float)
    props['total_obs'] = active_ns.sum(-1)

    mask_q = data['mask_q'].numpy().astype(bool)
    true_sigma = data['sigma_rfx'].numpy()
    n_q = mask_q.sum(-1).clip(1).astype(float)
    props['mean_sigma_rfx'] = (true_sigma * mask_q).sum(-1) / n_q
    props['max_sigma_rfx'] = np.where(mask_q, true_sigma, -np.inf).max(-1)
    props['min_sigma_rfx'] = np.where(mask_q, true_sigma, np.inf).min(-1)

    mask_d = data['mask_d'].numpy().astype(bool)
    true_ffx = data['ffx'].numpy()
    n_d = mask_d.sum(-1).clip(1).astype(float)
    props['mean_abs_ffx'] = (np.abs(true_ffx) * mask_d).sum(-1) / n_d

    if 'sigma_eps' in data:
        sigma_eps = data['sigma_eps'].numpy()
        props['sigma_eps'] = sigma_eps
        props['snr_rfx'] = props['mean_sigma_rfx'] / (sigma_eps + 1e-8)

    return props


# ─── Report sections ──────────────────────────────────────────────────────────


def printDistributions(
    nrmse_all: dict[str, dict[str, np.ndarray]],
    cov_err_all: dict[str, dict[str, np.ndarray]],
    loo_nll_all: dict[str, np.ndarray],
) -> None:
    _section('PER-DATASET METRIC DISTRIBUTIONS  (med [Q25, Q75]  [Q10, Q90]  mean)')
    for label in nrmse_all:
        print(f'\n  ── {label} ──')
        print(f'  NRMSE:')
        for key, arr in nrmse_all[label].items():
            _dist(arr, key.upper())
        print(f'  Coverage error (mean |cov - nominal|, avg over alphas):')
        for key, arr in cov_err_all[label].items():
            _dist(arr, key.upper())
        print(f'  LOO-NLL:')
        _dist(loo_nll_all[label], 'LOO-NLL')


def printHeadToHead(
    nrmse_mb: dict[str, np.ndarray],
    nrmse_ref: dict[str, np.ndarray],
    loo_mb: np.ndarray,
    loo_ref: np.ndarray,
    ref_label: str,
) -> None:
    """Print per-dataset error ratios MB / ref for each parameter group."""
    print(f'\n  ── vs {ref_label} ──  (ratio > 1 means MB is worse)')
    for key in nrmse_mb:
        if key not in nrmse_ref:
            continue
        ratio = nrmse_mb[key] / (nrmse_ref[key] + 1e-8)
        ok = np.isfinite(ratio)
        frac_worse = float((ratio[ok] > 1.0).mean())
        _dist(ratio[ok], f'{key.upper()} ratio')
        print(f'    ↳ MB > {ref_label}: {frac_worse:.1%} of datasets')

    loo_ratio = loo_mb / (np.abs(loo_ref) + 1e-8)
    ok = np.isfinite(loo_ratio)
    _dist(loo_ratio[ok], 'LOO-NLL ratio')
    print(f'    ↳ MB > {ref_label}: {float((loo_ratio[ok] > 1.0).mean()):.1%} of datasets')


def printHeadToHeadSection(
    nrmse_all: dict[str, dict[str, np.ndarray]],
    loo_nll_all: dict[str, np.ndarray],
) -> None:
    _section('HEAD-TO-HEAD: MB vs NUTS and MB vs ADVI')
    if 'NUTS' in nrmse_all:
        printHeadToHead(
            nrmse_all['MB'], nrmse_all['NUTS'],
            loo_nll_all['MB'], loo_nll_all['NUTS'], 'NUTS',
        )
    if 'ADVI' in nrmse_all:
        printHeadToHead(
            nrmse_all['MB'], nrmse_all['ADVI'],
            loo_nll_all['MB'], loo_nll_all['ADVI'], 'ADVI',
        )


def printFailureAnalysis(
    combined_mb: np.ndarray,
    combined_nuts: np.ndarray | None,
    props: dict[str, np.ndarray],
    glmm_diag: dict[str, np.ndarray] | None,
    nrmse_mb: dict[str, np.ndarray],
    nrmse_nuts: dict[str, np.ndarray] | None,
) -> None:
    """Detailed characterization of the worst MB datasets."""
    _section(f'FAILURE CASE ANALYSIS (TOP-{TOP_K} WORST MB DATASETS BY COMBINED NRMSE)')
    B = len(combined_mb)

    # Rank by combined MB NRMSE
    order = np.argsort(combined_mb)[::-1]
    worst_idx = order[:TOP_K]
    rest_idx = order[TOP_K:]

    # Column header
    cols = ['ds', 'MB_nrmse']
    if combined_nuts is not None:
        cols.append('NUTS_nrmse')
        cols.append('ratio')
    cols += list(props.keys())
    if glmm_diag is not None:
        cols += [k for k in glmm_diag if k not in ('psi_trace',)]

    header = f'  {"ds":>4}  {"MB_nrmse":>9}'
    if combined_nuts is not None:
        header += f'  {"NUTS_nrmse":>10}  {"ratio":>7}'
    for k in props:
        header += f'  {k[:12]:>12}'
    if glmm_diag is not None:
        for k in glmm_diag:
            if k == 'psi_trace':
                continue
            header += f'  {k[:12]:>12}'
    print(header)
    print(f'  {THIN}')

    for idx in worst_idx:
        row = f'  {idx:4d}  {combined_mb[idx]:9.4f}'
        if combined_nuts is not None:
            nuts_val = combined_nuts[idx]
            ratio = combined_mb[idx] / (nuts_val + 1e-8)
            row += f'  {nuts_val:10.4f}  {ratio:7.3f}'
        for k, arr in props.items():
            row += f'  {arr[idx]:12.4f}'
        if glmm_diag is not None:
            for k, arr in glmm_diag.items():
                if k == 'psi_trace':
                    continue
                row += f'  {arr[idx]:12.4f}'
        print(row)

    # Compare worst-K vs rest on all properties
    print(f'\n  Comparison: worst-{TOP_K} vs remaining {B - TOP_K} datasets')
    print(f'  {"property":<30}  {"worst-K med":>12}  {"rest med":>10}  {"Δ":>8}')
    print(f'  {THIN}')

    def _med(arr, idx):
        return float(np.nanmedian(arr[idx])) if len(idx) > 0 else float('nan')

    all_prop = {**props}
    if glmm_diag is not None:
        all_prop.update(glmm_diag)
    all_prop['MB_combined_nrmse'] = combined_mb
    if combined_nuts is not None:
        all_prop['NUTS_combined_nrmse'] = combined_nuts

    for k, arr in all_prop.items():
        wm = _med(arr, worst_idx)
        rm = _med(arr, rest_idx)
        delta = wm - rm
        print(f'  {k:<30}  {wm:12.4f}  {rm:10.4f}  {delta:+8.4f}')

    # Per-key NRMSE breakdown for worst-K
    print(f'\n  Per-parameter NRMSE breakdown (worst-{TOP_K} vs rest):')
    print(f'  {"param":<14}  {"worst-K med MB":>14}', end='')
    if nrmse_nuts is not None:
        print(f'  {"worst-K med NUTS":>16}', end='')
    print()
    for key in nrmse_mb:
        mb_arr = nrmse_mb[key]
        wm = _med(mb_arr, worst_idx)
        row = f'  {key:<14}  {wm:14.4f}'
        if nrmse_nuts is not None and key in nrmse_nuts:
            nuts_arr = nrmse_nuts[key]
            row += f'  {_med(nuts_arr, worst_idx):16.4f}'
        print(row)


def printGLMMSection(
    glmm_diag: dict[str, np.ndarray],
    nrmse_all: dict[str, dict[str, np.ndarray]],
    combined_mb: np.ndarray,
    combined_nuts: np.ndarray | None,
) -> None:
    """GLMM diagnostic distributions, head-to-head, and NPE failure correlations."""
    _section('GLMM DIAGNOSTICS')

    # ── Distribution of GLMM NRMSE ───────────────────────────────────────────
    print('\n  GLMM point-estimate NRMSE distributions:')
    nrmse_keys = [k for k in glmm_diag if k.endswith('_nrmse')]
    for k in nrmse_keys:
        _dist(glmm_diag[k], k)
    if 'glmm_blup_var_ratio' in glmm_diag:
        print('\n  BLUP variance calibration (mean err² / mean blup_var per dataset):')
        print('  > 1 → GLMM overconfident (BLUPs overshrunk)  < 1 → underconfident')
        _dist(glmm_diag['glmm_blup_var_ratio'], 'blup_var_ratio')
    for k in ('phi_pearson', 'psi_0'):
        if k in glmm_diag:
            _dist(glmm_diag[k], k)

    # ── GLMM vs NPE head-to-head per parameter ────────────────────────────────
    _section('GLMM vs NPE HEAD-TO-HEAD  (same NRMSE scale)')
    glmm_nrmse_map = {
        'ffx':       'glmm_ffx_nrmse',
        'sigma_rfx': 'glmm_sigma_nrmse',
        'rfx':       'glmm_blup_nrmse',
    }
    if 'glmm_seps_nrmse' in glmm_diag:
        glmm_nrmse_map['sigma_eps'] = 'glmm_seps_nrmse'

    for method, nrmse_dict in nrmse_all.items():
        print(f'\n  ── {method} vs GLMM ──  (ratio > 1 means {method} is worse)')
        for param_key, glmm_key in glmm_nrmse_map.items():
            if param_key not in nrmse_dict:
                continue
            ratio = nrmse_dict[param_key] / (glmm_diag[glmm_key] + 1e-8)
            ok = np.isfinite(ratio)
            _dist(ratio[ok], f'{param_key.upper()} ratio')
            print(f'    ↳ {method} > GLMM: {float((ratio[ok] > 1.0).mean()):.1%} of datasets')

    # ── Propagation: does GLMM failure predict NPE failure? ──────────────────
    _section('GLMM FAILURE PROPAGATION TO NPE  (Spearman ρ)')
    print(
        '  High ρ(GLMM_err, MB_err) → shared data difficulty.\n'
        '  Low ρ or negative gap → NPE-specific failure (amortization error).'
    )

    nrmse_mb = nrmse_all['MB']
    print(f'\n  GLMM combined NRMSE vs NPE/NUTS/ADVI combined NRMSE:')
    gc = glmm_diag['glmm_combined_nrmse']
    _corr(gc, combined_mb, 'GLMM combined vs MB combined')
    if combined_nuts is not None:
        _corr(gc, combined_nuts, 'GLMM combined vs NUTS combined')

    print(f'\n  Per-parameter GLMM NRMSE vs MB NRMSE:')
    for param_key, glmm_key in glmm_nrmse_map.items():
        if param_key not in nrmse_mb:
            continue
        _corr(glmm_diag[glmm_key], nrmse_mb[param_key], f'{glmm_key} vs MB {param_key}')

    print(f'\n  Per-parameter GLMM NRMSE vs NUTS NRMSE (sanity check):')
    nrmse_nuts = nrmse_all.get('NUTS', {})
    for param_key, glmm_key in glmm_nrmse_map.items():
        if param_key not in nrmse_nuts:
            continue
        _corr(glmm_diag[glmm_key], nrmse_nuts[param_key], f'{glmm_key} vs NUTS {param_key}')

    # ── Amortization gap: MB_nrmse − GLMM_nrmse ──────────────────────────────
    gap = combined_mb - gc
    print(f'\n  Amortization gap = MB_combined_nrmse − GLMM_combined_nrmse:')
    print('  Positive → MB fails more than GLMM (NPE-specific failure).')
    _dist(gap, 'gap')
    frac_positive = float((gap > 0).mean())
    print(f'    ↳ gap > 0 on {frac_positive:.1%} of datasets')


def printPropertyCorrelations(
    props: dict[str, np.ndarray],
    nrmse_mb: dict[str, np.ndarray],
    combined_mb: np.ndarray,
    glmm_diag: dict[str, np.ndarray] | None,
) -> None:
    """Spearman correlations between dataset properties and MB NRMSE."""
    _section('DATASET PROPERTY CORRELATIONS WITH MB NRMSE  (Spearman ρ)')
    print('  Shows which structural characteristics predict NPE difficulty.')

    for pk, pa in props.items():
        print(f'\n  {pk}:')
        _corr(pa, combined_mb, 'vs combined global NRMSE')
        for nk, na in nrmse_mb.items():
            _corr(pa, na, f'vs {nk} NRMSE')

    # Correlate dataset properties with the amortization gap (MB − GLMM error)
    if glmm_diag is not None and 'glmm_combined_nrmse' in glmm_diag:
        gap = combined_mb - glmm_diag['glmm_combined_nrmse']
        print(f'\n  dataset properties vs amortization gap (MB_nrmse − GLMM_nrmse):')
        for pk, pa in props.items():
            _corr(pa, gap, f'vs gap ({pk})')


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    cfg = evaluatorSetup()
    cfg.plot = False
    cfg.save_tables = False

    ev = Evaluator(cfg)
    lf = cfg.likelihood_family
    full_batch = ev.dl_test.fullBatch()
    B = full_batch['ffx'].shape[0]

    print(SEP)
    print('  NPE Failure Mode Analysis')
    print(f'  likelihood_family={lf}  B={B} test datasets')
    print(SEP)

    # ── Sample / load proposals ───────────────────────────────────────────────
    print('\nSampling MB proposals...')
    proposal_mb = ev.sampleMinibatched(ev.dl_test, 'MB')

    print('Loading NUTS and ADVI proposals...')
    proposal_nuts = ev._fit2proposal(full_batch, prefix='nuts')
    proposal_advi = ev._fit2proposal(full_batch, prefix='advi')

    proposals = {'MB': proposal_mb, 'NUTS': proposal_nuts, 'ADVI': proposal_advi}

    # CPU + optional rescaling
    data = toDevice(full_batch, 'cpu')
    if cfg.rescale:
        data = rescaleData(data)
    for p in proposals.values():
        p.to('cpu')

    # ── Per-dataset metrics ───────────────────────────────────────────────────
    print('Computing per-dataset NRMSE, coverage error, and LOO-NLL...')
    nrmse_all: dict[str, dict[str, np.ndarray]] = {}
    cov_err_all: dict[str, dict[str, np.ndarray]] = {}
    loo_nll_all: dict[str, np.ndarray] = {}

    for label, proposal in proposals.items():
        nrmse_all[label] = perDatasetNRMSE(proposal, data)
        cov_err_all[label] = perDatasetCoverageError(proposal, data)
        loo_nll_all[label] = perDatasetLooNLL(proposal, data, lf)

    combined_mb = combinedNRMSE(nrmse_all['MB'])
    combined_nuts = combinedNRMSE(nrmse_all['NUTS']) if 'NUTS' in nrmse_all else None

    # ── Dataset properties and GLMM diagnostics ───────────────────────────────
    props = datasetProperties(data)

    print('Running GLMM diagnostics...')
    glmm_diag = runGLMMDiagnostics(data, lf)

    # ── Report ────────────────────────────────────────────────────────────────

    printDistributions(nrmse_all, cov_err_all, loo_nll_all)

    printHeadToHeadSection(nrmse_all, loo_nll_all)

    printCoverageCalibration(proposals, data)

    printFailureAnalysis(
        combined_mb,
        combined_nuts,
        props,
        glmm_diag,
        nrmse_all['MB'],
        nrmse_all.get('NUTS'),
    )

    if glmm_diag is not None:
        printGLMMSection(glmm_diag, nrmse_all, combined_mb, combined_nuts)

    printPropertyCorrelations(props, nrmse_all['MB'], combined_mb, glmm_diag)

    print()
    print(SEP)
    print('  Done.')
    print(SEP)


if __name__ == '__main__':
    main()
