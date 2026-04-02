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


