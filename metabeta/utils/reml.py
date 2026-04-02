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


def _GLSNLL(
    l_diag: torch.Tensor,  # (B, q)
    l_off: torch.Tensor,  # (B, q*(q-1)//2)
    log_sigma_eps: torch.Tensor,  # (B,)
    XtX_g: torch.Tensor,  # (B, m, d, d)
    Xty_g: torch.Tensor,  # (B, m, d)
    yty_g: torch.Tensor,  # (B, m)
    ZtZ_g: torch.Tensor,  # (B, m, q, q)
    ZtX_g: torch.Tensor,  # (B, m, q, d)
    Zty_g: torch.Tensor,  # (B, m, q)
    ns: torch.Tensor,  # (B, m)  group sizes (float)
    mask_m: torch.Tensor,  # (B, m)  1 for active groups
) -> tuple[torch.Tensor, torch.Tensor]:
    """GLS β̂ and REML negative log-likelihood for given variance components.

    Returns (beta, nll) with shapes (B, d) and (B,).
    """
    B, _, q = Zty_g.shape
    inv_se2 = torch.exp(-2 * log_sigma_eps)  # (B,)

    # --- Σ^{-1} from Cholesky factor L via triangular solve
    L = _getL(l_diag, l_off, q)  # (B, q, q)
    eye_q = torch.eye(q, device=l_diag.device, dtype=l_diag.dtype).unsqueeze(0).expand(B, -1, -1)
    L_inv = torch.linalg.solve_triangular(L, eye_q, upper=False)  # (B, q, q)
    Sigma_inv = L_inv.transpose(-1, -2) @ L_inv                 # (B, q, q)
    log_det_Sigma = 2.0 * l_diag.sum(dim=-1)  # (B,)

    # --- Woodbury identity
    M_g = Sigma_inv.unsqueeze(1) + inv_se2[:, None, None, None] * ZtZ_g
    MinvZtX = torch.linalg.solve(M_g, ZtX_g)  # (B, m, q, d)
    MinvZty = torch.linalg.solve(M_g, Zty_g)  # (B, m, q)
    wood_XX = torch.einsum('bmqd,bmqk->bmdk', ZtX_g, MinvZtX)  # (B, m, d, d)
    wood_Xy = torch.einsum('bmqd,bmq->bmd', ZtX_g, MinvZty)    # (B, m, d)
    wood_yy = (Zty_g * MinvZty).sum(dim=-1)                    # (B, m)
    inv_se2_4d = inv_se2[:, None, None, None]
    inv_se2_3d = inv_se2[:, None, None]
    inv_se2_2d = inv_se2[:, None]

    # --- GLS cross-products via Woodbury
    A_g = inv_se2_4d * (XtX_g - inv_se2_4d * wood_XX)  # (B, m, d, d)
    b_g = inv_se2_3d * (Xty_g - inv_se2_3d * wood_Xy)  # (B, m, d)
    c_g = inv_se2_2d * (yty_g - inv_se2_2d * wood_yy)  # (B, m)

    # sum over active groups
    A = (A_g * mask_m[:, :, None, None]).sum(dim=1)  # (B, d, d)
    b_vec = (b_g * mask_m[:, :, None]).sum(dim=1)    # (B, d)
    c = (c_g * mask_m).sum(dim=1)                    # (B,)

    # GLS: β̂ = (A + ridge)^{-1} b
    beta = torch.linalg.solve(A + _adaptive_ridge(A), b_vec)  # (B, d)

    # --- REML NLL
    # quadratic form y'Py = c − b'β̂  (≥ 0)
    quad = (c - (b_vec * beta).sum(dim=-1)).clamp(min=0.0)  # (B,)

    # logdet of compound covariance matrix
    _, log_det_M = torch.linalg.slogdet(M_g)  # (B, m)
    log_se2 = 2 * log_sigma_eps
    log_V_i = log_det_M + log_det_Sigma[:, None] + ns * log_se2[:, None]
    sum_log_V = (log_V_i * mask_m).sum(dim=1)  # (B,)

    # logdet of A
    _, log_det_A = torch.linalg.slogdet(A)  # (B,)
    nll = 0.5 * (sum_log_V + log_det_A + quad)
    return beta, nll


def remlSolve(
    Xm: torch.Tensor,  # (B, m, n, d)  masked X
    ym: torch.Tensor,  # (B, m, n)     masked y
    Zm: torch.Tensor,  # (B, m, n, q)  masked Z  (= Xm[..., :q] in metabeta)
    ns: torch.Tensor,  # (B, m)        group sizes
    n_iter: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched REML for mixed effects regression with full rfx covariance.

    Parameterises Σ_rfx = L Lᵀ (Cholesky) and optimizes the REML log-likelihood
    over (l_diag, l_off, log σ_ε) with L-BFGS. β̂ is obtained analytically via
    GLS at the optimal variance components.

    """
    # init
    B, _, _, q = Zm.shape
    n_off = q * (q - 1) // 2
    mask_m = (ns > 0).to(Xm.dtype)
    ns_f = ns.float().clamp(min=1.0)

    # inner products over n
    stats = _sufficientStats(Xm, ym, Zm)

    # optimization parameters [l_diag (q), l_off (n_off), log_se (1)].
    l_params = torch.zeros(B, q + n_off + 1, device=Xm.device, dtype=Xm.dtype, requires_grad=True)

    # batched REML of covariance parameters
    optimizer = torch.optim.LBFGS(
        [l_params],
        max_iter=n_iter,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-5,
        tolerance_change=1e-7,
    )

    def closure():
        optimizer.zero_grad()
        l_diag = l_params[:, :q]
        l_off = l_params[:, q : q + n_off]  # shape (B, 0) when q == 1
        log_se = l_params[:, -1]
        _, nll = _GLSNLL(l_diag, l_off, log_se, *stats, ns_f, mask_m)   # type: ignore
        loss = nll.sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    # batched GLS of fixed effects
    with torch.no_grad():
        l_diag = l_params[:, :q]
        l_off = l_params[:, q : q + n_off]
        log_se = l_params[:, -1]
        beta, _ = _GLSNLL(l_diag, l_off, log_se, *stats, ns_f, mask_m)   # type: ignore
        L = _getL(l_diag, l_off, q)
        Sigma_u = L @ L.transpose(-1, -2)   # (B, q, q)
        sigma_eps = log_se.exp()             # (B,)

    return beta, Sigma_u, sigma_eps
