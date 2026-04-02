import torch
from metabeta.utils.least_squares import _adaptive_ridge


def _sufficientStats(
    Xm: torch.Tensor,  # (B, m, n, d)
    ym: torch.Tensor,  # (B, m, n)
    Zm: torch.Tensor,  # (B, m, n, q)
) -> tuple[torch.Tensor, ...]:
    """Group-level sufficient statistics"""
    XtX_g = torch.einsum('bmnd,bmnk->bmdk', Xm, Xm)   # (B, m, d, d)
    Xty_g = torch.einsum('bmnd,bmn->bmd', Xm, ym)     # (B, m, d)
    yty_g = (ym * ym).sum(dim=2)                      # (B, m)
    ZtZ_g = torch.einsum('bmnq,bmnr->bmqr', Zm, Zm)   # (B, m, q, q)
    ZtX_g = torch.einsum('bmnq,bmnd->bmqd', Zm, Xm)   # (B, m, q, d)
    Zty_g = torch.einsum('bmnq,bmn->bmq', Zm, ym)     # (B, m, q)
    return XtX_g, Xty_g, yty_g, ZtZ_g, ZtX_g, Zty_g


def _getL(
    l_diag: torch.Tensor,  # (B, q)   log L_jj
    l_off: torch.Tensor,  # (B, q*(q-1)//2)  L_ij, i > j
    q: int,
) -> torch.Tensor:         # (B, q, q) lower triangular
    """assemble the Cholesky factor L from unconstrained parameters"""
    L = torch.diag_embed(l_diag.exp())  # (B, q, q) positive diagonal
    if q > 1:
        rows, cols = torch.tril_indices(q, q, offset=-1, device=l_diag.device)
        n_off = rows.shape[0]
        basis = torch.zeros(n_off, q, q, device=l_diag.device, dtype=l_diag.dtype)
        basis[torch.arange(n_off), rows, cols] = 1.0
        L = L + torch.einsum('bk,kij->bij', l_off, basis)
    return L


