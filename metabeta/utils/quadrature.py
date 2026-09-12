"""Gauss-Hermite quadrature grids shared by the analytical GLMM refiners and the
posthoc AGQ marginal likelihood."""

import numpy as np
import torch


def ghProductGrid(
    k_vals: list[int], dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cartesian-product Gauss-Hermite grid; returns z_nodes (K, q) and log_w (K,)."""
    z_1d, logw_1d = [], []
    for k in k_vals:
        z_np, w_np = np.polynomial.hermite.hermgauss(k)
        z_1d.append(torch.tensor(z_np, dtype=dtype, device=device))
        logw_1d.append(torch.tensor(np.log(w_np), dtype=dtype, device=device))
    if len(k_vals) == 1:
        return z_1d[0].unsqueeze(-1), logw_1d[0]
    grids_z = torch.meshgrid(*z_1d, indexing='ij')
    grids_w = torch.meshgrid(*logw_1d, indexing='ij')
    z_nodes = torch.stack([g.reshape(-1) for g in grids_z], dim=-1)
    log_w = sum(g.reshape(-1) for g in grids_w)
    return z_nodes, log_w
