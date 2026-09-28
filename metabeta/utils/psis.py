from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def psislw(log_w: np.ndarray, r_eff: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Pareto-smoothed importance sampling (PSIS) log-weights along the last axis.

    Wraps ``arviz_stats.base.array_stats.psislw`` (Vehtari et al. 2024,
    https://arxiv.org/abs/1507.02646) with the ArviZ < 1 conventions this package was
    written against: the input is log importance *ratios* (arviz-stats expects the
    log-likelihood and negates internally), the output is log-normalised, and too few
    draws for a 5-draw tail fit return the raw normalised weights with k = inf instead of
    raising.
    """
    # imported lazily so constructing samplers never pulls in the arviz stack
    from arviz_stats.base import array_stats

    n = log_w.shape[-1]
    if n * r_eff <= 225 and n // 5 < 5:
        log_w_norm = log_w - logsumexp(log_w, axis=-1, keepdims=True)
        return log_w_norm, np.full(log_w.shape[:-1], np.inf)
    return array_stats.psislw(-log_w, r_eff=r_eff, axis=-1)
