from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def psislw(log_w: np.ndarray, r_eff: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Pareto-smoothed importance sampling (PSIS) log-weights along the last axis.

    Wraps ``arviz_stats.base.array_stats.psislw`` (Vehtari et al. 2024,
    https://arxiv.org/abs/1507.02646) with the ArviZ < 1 conventions this package was
    written against: the input is log importance *ratios* (arviz-stats expects the
    log-likelihood and negates internally), the output is log-normalised, and rows whose
    tail cannot be fitted return the raw normalised weights with k = inf instead of
    raising. That covers too few draws for a 5-draw tail and a tail of identical values
    (an observation every draw predicts equally well, common with Bernoulli outcomes).
    """
    # imported lazily so constructing samplers never pulls in the arviz stack
    from arviz_stats.base import array_stats

    log_w = np.asarray(log_w, dtype=np.float64)
    n = log_w.shape[-1]
    # tail length rule of arviz-stats (Vehtari et al. 2024, Appendix G)
    n_tail = int(np.floor(3 * np.sqrt(n / r_eff))) if n * r_eff > 225 else n // 5
    log_w_norm = log_w - logsumexp(log_w, axis=-1, keepdims=True)
    k = np.full(log_w.shape[:-1], np.inf)
    if n_tail < 5:
        return log_w_norm, k
    tail = np.sort(log_w, axis=-1)[..., -n_tail:]  # (..., n_tail)
    fittable = np.abs(tail[..., -1] - tail[..., 0]) >= np.finfo(float).tiny
    if not fittable.any():
        return log_w_norm, k
    lw_fit, k_fit = array_stats.psislw(-log_w[fittable], r_eff=r_eff, axis=-1)
    log_w_norm[fittable] = lw_fit
    k[fittable] = k_fit
    return log_w_norm, k
