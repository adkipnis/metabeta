import torch
import numpy as np


def _batchCorrcoef(
    x: np.ndarray,  # (n_sim, m, q)
) -> np.ndarray:
    a, b = x[..., 0], x[..., 1]  # (n_sim, m)
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    num = (a * b).sum(axis=-1)
    den = np.sqrt((a**2).sum(axis=-1) * (b**2).sum(axis=-1)).clip(min=1e-12)
    return num / den  # (n_sim)


def _corrSamplingDistribution(
    m: int,
    rho: float,
    n_sim: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Empirical distribution of sample-correlation estimates for fixed (m, rho)."""
    rho = float(np.clip(rho, -0.999, 0.999))
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=(n_sim, m))
    return _batchCorrcoef(z)


def _empiricalBounds(
    m: int,
    rhos: np.ndarray,
    n_sim: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical lower and upper quantiles for r-hat over a rho grid."""
    lowers = np.zeros_like(rhos, dtype=float)
    uppers = np.zeros_like(rhos, dtype=float)
    for i, rho in enumerate(rhos):
        rs = _corrSamplingDistribution(m=m, rho=float(rho), n_sim=n_sim, seed=seed + i)
        lowers[i] = np.quantile(rs, alpha / 2)
        uppers[i] = np.quantile(rs, 1 - alpha / 2)
    return lowers, uppers


def posteriorCorrelation(
    rfx: torch.Tensor,  # (b, m, s, q)
    mask_m: torch.Tensor,  # (b, m)
) -> torch.Tensor:
    """compute pairwise correlation matrices from posterior rfx samples"""
    mask = mask_m[:, :, None, None].float()  # (b, m, 1, 1)
    n = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (b, 1, 1, 1)
    mean = (rfx * mask).sum(dim=1, keepdim=True) / n  # (b, 1, s, q)
    centered = (rfx - mean) * mask  # (b, m, s, q)
    cov = torch.einsum('bmsi,bmsj->bsij', centered, centered) / n.view(-1, 1, 1, 1)  # (b, s, q, q)
    std = cov.diagonal(dim1=-2, dim2=-1).clamp(min=1e-12).sqrt()  # (b, s, q)
    return cov / (std.unsqueeze(-1) * std.unsqueeze(-2))  # (b, s, q, q)


def evaluateCorrelation(
    rfx: torch.Tensor,  # (b, m, s, q)
    data: dict[str, torch.Tensor],
    n_sim: int = 2000,
    rho_grid_step: float = 0.02,
    alpha: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Evaluate correlation recovery from posterior rfx samples.

    Uses non-parametric rank test: for each upper-triangular pair, ranks
    |corr_mean[i,j]| against the null distribution of |r| when rho=0.
    """
    mask_m = data['mask_m']  # (b, m)
    corr_true = data['corr_rfx']  # (b, q, q)
    eta_rfx = data['eta_rfx']  # (b,)
    b_size = rfx.shape[0]
    q = rfx.shape[-1]

    # trivial case
    if q < 2:
        return {
            'corr_mean': torch.ones(b_size, 1, 1, device=rfx.device),
            'corr_true': torch.ones(b_size, 1, 1, device=rfx.device),
            'corr_q025': torch.ones(b_size, 1, 1, device=rfx.device),
            'corr_q975': torch.ones(b_size, 1, 1, device=rfx.device),
            'offdiag_mae': torch.zeros(b_size, device=rfx.device),
            'percentile': torch.zeros(b_size, device=rfx.device),
            'percentile_pairs': torch.zeros(b_size, 1, device=rfx.device),
            'eta_rfx': eta_rfx,
        }

    # posterior correlation
    corr_samples = posteriorCorrelation(rfx, mask_m)  # (b, s, q, q)
    # isnull = corr_samples[..., 0, -1, -1] == 0
    # isnullforq1 = isnull == (data['mask_q'].sum(-1) == 1)
    corr_mean = corr_samples.mean(dim=1)  # (b, q, q)

    # upper-triangular indices (unique pairs only)
    ri, ci = torch.triu_indices(q, q, offset=1)

    # upper-tri MAE
    diff = (corr_mean - corr_true).abs()
    offdiag_mae = diff[:, ri, ci].mean(dim=-1)  # (b,)
    # n_off = len(ri)

    # per-m caches: null r and empirical r_hat quantile bounds
    ms = data['m']
    rho_grid = np.arange(-0.95, 0.95 + rho_grid_step, rho_grid_step)
    null_cache: dict[int, np.ndarray] = {}
    bounds_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for m_val in ms.unique().tolist():
        m_int = int(m_val)
        null_dist = _corrSamplingDistribution(m_int, rho=0.0, n_sim=n_sim)
        null_dist.sort()
        null_cache[m_int] = null_dist
        bounds_cache[m_int] = _empiricalBounds(
            m=m_int,
            rhos=rho_grid,
            n_sim=n_sim,
            alpha=alpha,
        )

    # pair-wise empirical envelope conditioned on true rho and m
    # use interpolation instead of individual computation for each rho
    corr_true_pairs = corr_true[:, ri, ci].cpu().numpy()  # (b, n_off)
    q025 = np.zeros_like(corr_true_pairs, dtype=float)
    q975 = np.zeros_like(corr_true_pairs, dtype=float)
    for i in range(b_size):
        m_i = int(ms[i])
        lowers, uppers = bounds_cache[m_i]
        q025[i] = np.interp(corr_true_pairs[i], rho_grid, lowers)
        q975[i] = np.interp(corr_true_pairs[i], rho_grid, uppers)

    # percentile: rank |corr_mean| against null via searchsorted
    abs_uppertri = corr_mean[:, ri, ci].abs().cpu().numpy()  # (b, n_off)
    percentiles = np.zeros(b_size)
    percentile_pairs = np.zeros_like(abs_uppertri)
    for i in range(b_size):
        null = null_cache[int(ms[i])]
        ranks = np.searchsorted(null, abs_uppertri[i]) / len(null)  # (n_off,)
        percentile_pairs[i] = ranks
        percentiles[i] = ranks.mean()
    percentiles = torch.as_tensor(percentiles, device=rfx.device, dtype=rfx.dtype)
    percentile_pairs = torch.as_tensor(percentile_pairs, device=rfx.device, dtype=rfx.dtype)
    corr_q025 = torch.as_tensor(q025, device=rfx.device, dtype=rfx.dtype)
    corr_q975 = torch.as_tensor(q975, device=rfx.device, dtype=rfx.dtype)

    return {
        'corr_mean': corr_mean[:, ri, ci],
        'corr_true': corr_true[:, ri, ci],
        'corr_q025': corr_q025,
        'corr_q975': corr_q975,
        'offdiag_mae': offdiag_mae,
        'percentile': percentiles,
        'percentile_pairs': percentile_pairs,
        'eta_rfx': eta_rfx,
    }


def summarizeCorrelation(
    results: dict[str, torch.Tensor],
    qs: torch.Tensor,
    threshold: float = 0.90,
) -> dict[str, float]:
    """Aggregate correlation evaluation into summary metrics"""
    eta = results['eta_rfx']
    mae = results['offdiag_mae']
    pct = results['percentile']

    nontrivial = qs > 1
    correlated = eta > 0
    uncorrelated = (eta == 0) & nontrivial

    out = {
        'mae_all': mae[nontrivial].median().item(),
    }
    if correlated.any():
        out['mae_correlated'] = mae[correlated].median().item()
        out['mperc_correlated'] = pct[correlated].median().item()
        out['detection_rate'] = (pct[correlated] > threshold).float().mean().item()
        # out['detection_rate'] = (results['corr_mean'][correlated] > threshold).float().mean().item()
    if uncorrelated.any():
        out['mae_uncorrelated'] = mae[uncorrelated].median().item()
        out['mperc_uncorrelated'] = pct[uncorrelated].median().item()
        out['false_positive_rate'] = (pct[uncorrelated] > threshold).float().mean().item()
        # out['false_positive_rate'] = (results['corr_mean'][uncorrelated] > threshold).float().mean().item()
    return out
