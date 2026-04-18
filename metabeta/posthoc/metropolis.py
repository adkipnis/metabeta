"""
posthoc/metropolis.py — Independence Metropolis-Hastings (IMH) correction for flow posteriors.

Design
------
The flow q(θ) approximates p(θ|y) but can be systematically off (e.g. the sigma_eps/sigma_rfx
ridge in Normal models).  IMH uses the flow as proposal in a Markov chain, accepting/rejecting
with ratio w(θ') / w(θ).  Unlike IS, degenerate proposals are corrected by rejection rather
than by collapsed weights, and the chain is guaranteed to target the correct posterior.

Three modes differ in how rfx (local params) are handled:

  'global'
      log_w = log p(y|θ_g, θ_l) + log p(θ_g) − log q_g
      rfx enter the likelihood but carry no IS correction.  When a global proposal is accepted
      the paired flow rfx from the same draw travels with it.  Works for all likelihoods.

  'marginal'  [default for Normal; requires likelihood_family == 0]
      log_w = log p_marginal(y|θ_g) + log p(θ_g) − log q_g
      rfx are integrated out analytically via the Normal-Normal conjugate marginal, so the
      chain operates only in the low-dimensional global space.  After acceptance, fresh rfx
      are drawn from the exact conditional posterior p(rfx | θ_g, y) (Rao-Blackwellised).
      Best mixing; theoretically optimal for Normal.

  'joint'
      log_w = log p(y|θ_g,θ_l) + log p(θ_l|θ_g) + log p(θ_g) − log q_g − log q_l
      Full joint target; rfx are subject to acceptance alongside globals.  Correct for GLMMs
      but degenerates with many groups (high rfx dimension).

Chain mechanics
---------------
A pool of s = n_chains × n_steps proposals is drawn upfront.  All log weights are computed in
one vectorised pass.  The MH loop is vectorised over (b, n_chains) and iterates as a Python
loop over n_steps — typically 50–250 steps.  The first `burnin` steps are discarded; the
remaining n_chains × (n_steps − burnin) samples are returned as a Proposal.

Empirical comparison
--------------------
Call MetropolisSampler with each mode in turn and evaluate the resulting Proposal via
Evaluator.summary() / plotComparison to choose the best mode for a given dataset type.
The diagnostics dict returned by __call__ contains per-chain acceptance rates as a quick
quality indicator before running the full evaluation.
"""

import argparse
from typing import Literal

import torch
from torch import Tensor

from metabeta.models.approximator import Approximator
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.utils.evaluation import Proposal
from metabeta.utils.families import (
    hasSigmaEps,
    logMarginalLikelihoodNormal,
    logProbFfx,
    logProbSigma,
)
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.regularization import unconstrainedToCholeskyCorr

Mode = Literal['global', 'marginal', 'joint']


class MetropolisSampler:
    def __init__(
        self,
        data: dict[str, Tensor],
        n_chains: int = 4,
        n_steps: int = 250,
        burnin: int = 50,
        mode: Mode = 'marginal',
        likelihood_family: int = 0,
        eps: float = 1e-12,
    ) -> None:
        if mode == 'marginal' and likelihood_family != 0:
            raise ValueError("mode='marginal' requires likelihood_family=0 (Normal)")
        if burnin >= n_steps:
            raise ValueError('burnin must be < n_steps')

        self.n_chains = n_chains
        self.n_steps = n_steps
        self.burnin = burnin
        self.mode = mode
        self.likelihood_family = likelihood_family
        self.has_sigma_eps = hasSigmaEps(likelihood_family)
        self.eps = eps

        # Reuse ImportanceSampler for prior log-prob computation and unnormalizedPosterior.
        # 'joint' needs full=True (includes rfx prior and local log-prob).
        self._is = ImportanceSampler(
            data, full=(mode == 'joint'), likelihood_family=likelihood_family, eps=eps
        )

        # Data tensors for the Normal-Normal conditional (marginal mode).
        # These mirror what ImportanceSampler stores but are kept as direct references.
        self._X = data['X']           # (b, m, n, d)
        self._Z = data['Z']           # (b, m, n, q)
        self._y = data['y'].unsqueeze(-1)  # (b, m, n, 1)
        self._mask_n = data['mask_n'].unsqueeze(-1)   # (b, m, n, 1)
        self._mask_m = data['mask_m'].unsqueeze(-1)   # (b, m, 1)

    # ------------------------------------------------------------------
    # Log-weight computation
    # ------------------------------------------------------------------

    def _logWeights(self, proposal: Proposal) -> Tensor:
        """Compute unnormalised log IS weights (b, s) according to self.mode."""
        log_q_g = proposal.log_prob_g  # (b, s)

        if self.mode == 'marginal':
            ffx = proposal.ffx         # (b, s, d)
            sigma_rfx = proposal.sigma_rfx   # (b, s, q)
            sigma_eps = proposal.sigma_eps   # (b, s)
            ll = logMarginalLikelihoodNormal(
                ffx, sigma_rfx, sigma_eps,
                self._y, self._X, self._Z, self._mask_n, self._mask_m,
            )
            lp = logProbFfx(ffx, self._is.nu_ffx, self._is.tau_ffx,
                            self._is.family_ffx, self._is.mask_d)
            lp = lp + logProbSigma(sigma_rfx, self._is.tau_rfx,
                                   self._is.family_sigma_rfx, self._is.mask_q)
            lp = lp + logProbSigma(sigma_eps, self._is.tau_eps,
                                   self._is.family_sigma_eps)
            return ll + lp - log_q_g

        # 'global' or 'joint': delegate to ImportanceSampler
        ll, lp = self._is.unnormalizedPosterior(proposal)
        if self.mode == 'joint':
            log_q_l = proposal.log_prob_l   # (b, m, s)
            lq = log_q_g + (log_q_l * self._is.mask_m).sum(1)
            return ll + lp - lq

        return ll + lp - log_q_g

    # ------------------------------------------------------------------
    # MH chain
    # ------------------------------------------------------------------

    def _runChains(
        self,
        log_w: Tensor,          # (b, s)
        sg: Tensor,             # (b, s, D_g)
        sl: Tensor | None,      # (b, m, s, q) — None for marginal mode
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Run n_chains independent IMH chains, return (sg_out, sl_out, accept_rate).

        Outputs:
            sg_out       (b, C*T_post, D_g)
            sl_out       (b, m, C*T_post, q)  or None
            accept_rate  (b, C)  — fraction of proposals accepted after burnin
        """
        b, s, D_g = sg.shape
        C, T = self.n_chains, self.n_steps
        T_post = T - self.burnin

        # Reshape pool into (b, C, T, ...)
        sg_ct = sg.reshape(b, C, T, D_g)
        lw_ct = log_w.reshape(b, C, T)
        if sl is not None:
            m, q = sl.shape[1], sl.shape[-1]
            sl_ct = sl.permute(0, 2, 1, 3).reshape(b, C, T, m, q)

        # Initialise at the best-weight sample in each chain's block.
        # This prevents chains from getting permanently stuck when the first
        # proposal has very high weight and subsequent ones never beat it.
        best_t = lw_ct.argmax(dim=-1)                          # (b, C)
        bt_g = best_t.unsqueeze(-1).unsqueeze(-1).expand(b, C, 1, D_g)
        cur_g = sg_ct.gather(2, bt_g).squeeze(2).clone()       # (b, C, D_g)
        bt_lw = best_t.unsqueeze(-1)
        cur_lw = lw_ct.gather(2, bt_lw).squeeze(-1).clone()    # (b, C)
        if sl is not None:
            bt_l = best_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(b, C, 1, m, q)
            cur_l = sl_ct.gather(2, bt_l).squeeze(2).clone()   # (b, C, m, q)

        keep_g: list[Tensor] = []
        keep_l: list[Tensor] = []
        keep_acc: list[Tensor] = []

        for t in range(1, T):
            prop_lw = lw_ct[:, :, t]   # (b, C)

            log_alpha = (prop_lw - cur_lw).clamp(max=0.0)
            accept = torch.rand_like(log_alpha).log() < log_alpha  # (b, C)

            cur_g = torch.where(accept.unsqueeze(-1), sg_ct[:, :, t], cur_g)
            cur_lw = torch.where(accept, prop_lw, cur_lw)
            if sl is not None:
                cur_l = torch.where(accept[:, :, None, None], sl_ct[:, :, t], cur_l)

            if t >= self.burnin:
                keep_g.append(cur_g.clone())
                if sl is not None:
                    keep_l.append(cur_l.clone())
                keep_acc.append(accept.float())

        # Stack post-burnin samples: (T_post, b, C, D_g) → (b, C*T_post, D_g)
        sg_out = torch.stack(keep_g, dim=0).permute(1, 2, 0, 3).reshape(b, C * T_post, D_g)

        sl_out = None
        if sl is not None:
            sl_out = (
                torch.stack(keep_l, dim=0)        # (T_post, b, C, m, q)
                .permute(1, 2, 0, 3, 4)           # (b, C, T_post, m, q)
                .reshape(b, C * T_post, m, q)
                .permute(0, 2, 1, 3)              # (b, m, C*T_post, q)
            )

        accept_rate = torch.stack(keep_acc, dim=0).mean(0)  # (b, C)
        return sg_out, sl_out, accept_rate

    # ------------------------------------------------------------------
    # Normal-Normal conditional rfx posterior
    # ------------------------------------------------------------------

    def _sampleRfxConditional(
        self,
        sg_out: Tensor,   # (b, s_out, D_g)
        d: int,
        q: int,
        d_corr: int,
    ) -> Tensor:
        """Draw rfx ~ p(rfx | θ_g, y) using Normal-Normal conjugacy.

        For each accepted global sample and each group j:
            V_j⁻¹ = Σ_rfx⁻¹ + Zⱼᵀ Zⱼ / σ²_eps
            μ_j   = V_j (Zⱼᵀ rⱼ / σ²_eps)     where rⱼ = yⱼ − Xⱼ β
            rfx_j ~ N(μ_j, V_j)

        Σ_rfx = diag(σ_rfx²) for independent rfx, or D R Dᵀ for correlated
        (D = diag(σ_rfx), R reconstructed from the unconstrained z_corr dims).

        Returns (b, m, s_out, q).
        """
        b, s_out, _ = sg_out.shape
        m = self._X.shape[1]

        # Extract global parameter components from sg_out
        ffx = sg_out[..., :d]                     # (b, s_out, d)
        sigma_rfx = sg_out[..., d:d + q]          # (b, s_out, q)
        sigma_eps = sg_out[..., d + q]             # (b, s_out)

        # Residuals: r = y - X @ ffx  (b, m, n, s_out)
        mu_ffx = torch.einsum('bmnd,bsd->bmns', self._X, ffx)
        r = (self._y - mu_ffx) * self._mask_n     # (b, m, n, s_out)

        # Group-level sufficient statistics
        Z_m = self._Z * self._mask_n              # (b, m, n, q) — zero out padded obs
        ZtZ = torch.einsum('bmnq,bmnr->bmqr', Z_m, Z_m)              # (b, m, q, q)
        Ztr = torch.einsum('bmnq,bmns->bmsq', Z_m, r)                # (b, m, s_out, q)

        # Σ_rfx Cholesky factor L such that Σ_rfx = L @ Lᵀ
        sigma_rfx_c = sigma_rfx.clamp(min=1e-6)
        if d_corr > 0:
            L_corr = unconstrainedToCholeskyCorr(sg_out[..., -d_corr:], q)  # (b, s_out, q, q)
            L_rfx = L_corr * sigma_rfx_c.unsqueeze(-1)                      # (b, s_out, q, q)
        else:
            L_rfx = torch.diag_embed(sigma_rfx_c)   # (b, s_out, q, q) diagonal

        # Σ_rfx⁻¹ via Cholesky solve: solve L Lᵀ x = I
        eye = torch.eye(q, device=sg_out.device, dtype=sg_out.dtype).expand(b, s_out, q, q)
        Sigma_inv = torch.cholesky_solve(eye, L_rfx)   # (b, s_out, q, q)

        # M = Σ_rfx⁻¹ + ZᵀZ / σ²_eps  (b, m, s_out, q, q)
        s2e = sigma_eps.pow(2)  # (b, s_out)
        M = Sigma_inv.unsqueeze(1) + ZtZ.unsqueeze(2) / s2e[:, None, :, None, None]

        # Cholesky of M (posterior precision)
        jitter = 1e-6 * torch.eye(q, device=M.device, dtype=M.dtype)
        chol_M = torch.linalg.cholesky(M + jitter)   # (b, m, s_out, q, q)

        # Posterior mean: μ = M⁻¹ (Zᵀr / σ²_eps)
        rhs = Ztr / s2e[:, None, :, None]             # (b, m, s_out, q)
        post_mean = torch.cholesky_solve(
            rhs.unsqueeze(-1), chol_M
        ).squeeze(-1)                                  # (b, m, s_out, q)

        # Sample: rfx = μ + chol_M⁻ᵀ z  where z ~ N(0, I)
        # chol_M is Chol(M) = Chol(V⁻¹), so Chol(V) = chol_M⁻ᵀ (upper-triangular solve)
        z = torch.randn_like(post_mean)
        rfx_centered = torch.linalg.solve_triangular(
            chol_M.mT, z.unsqueeze(-1), upper=True
        ).squeeze(-1)                                  # (b, m, s_out, q)

        rfx = (post_mean + rfx_centered) * self._mask_m.unsqueeze(-1).float()
        return rfx   # (b, m, s_out, q)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, proposal: Proposal) -> tuple[Proposal, dict]:
        """Run IMH chains on a pre-drawn flow proposal.

        Parameters
        ----------
        proposal : Proposal
            Output of model.estimate(data, n_samples=n_chains * n_steps).

        Returns
        -------
        proposal_out : Proposal
            Post-burnin samples; n_chains * (n_steps - burnin) samples per dataset.
        diagnostics : dict
            'accept_rate' (b, n_chains) — fraction of proposals accepted post-burnin.
        """
        s_expected = self.n_chains * self.n_steps
        if proposal.n_samples != s_expected:
            raise ValueError(
                f'proposal has {proposal.n_samples} samples; '
                f'expected n_chains × n_steps = {s_expected}'
            )

        d, q = proposal.d, proposal.q
        d_corr = proposal.d_corr

        log_w = self._logWeights(proposal)   # (b, s)

        # Run chain — rfx travels with globals for 'global' and 'joint' modes
        sl_in = proposal.samples_l if self.mode != 'marginal' else None
        sg_out, sl_out, accept_rate = self._runChains(log_w, proposal.samples_g, sl_in)

        # Attach rfx
        if self.mode == 'marginal':
            sl_out = self._sampleRfxConditional(sg_out, d, q, d_corr)
        # 'global' and 'joint': sl_out already set by _runChains

        b, s_out = sg_out.shape[0], sg_out.shape[1]
        m = sl_out.shape[1]
        proposed = {
            'global': {'samples': sg_out, 'log_prob': sg_out.new_zeros(b, s_out)},
            'local': {'samples': sl_out, 'log_prob': sl_out.new_zeros(b, m, s_out)},
        }
        out = Proposal(proposed, has_sigma_eps=proposal.has_sigma_eps, d_corr=d_corr)
        out.tpd = proposal.tpd
        return out, {'accept_rate': accept_rate}


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


def runIMH(
    model: Approximator,
    data: dict[str, torch.Tensor],
    cfg: argparse.Namespace,
) -> tuple[Proposal, dict]:
    """Draw a flow proposal and correct it with IMH.

    cfg fields
    ----------
    n_chains       : int  — number of independent chains (default 4)
    n_steps        : int  — steps per chain including burnin (default 250)
    imh_burnin     : int  — burnin steps to discard (default 50)
    imh_mode       : str  — 'global' | 'marginal' | 'joint'
                     defaults to 'marginal' for Normal, 'global' otherwise
    rescale        : bool
    likelihood_family : int
    """
    lf = getattr(cfg, 'likelihood_family', 0)
    n_chains = getattr(cfg, 'n_chains', 4)
    n_steps = getattr(cfg, 'n_steps', 250)
    burnin = getattr(cfg, 'imh_burnin', 50)
    default_mode = 'marginal' if lf == 0 else 'global'
    mode: Mode = getattr(cfg, 'imh_mode', default_mode)

    proposal = model.estimate(data, n_samples=n_chains * n_steps)

    if cfg.rescale:
        proposal.rescale(data['sd_y'])
        data = rescaleData(data)

    sampler = MetropolisSampler(
        data, n_chains=n_chains, n_steps=n_steps, burnin=burnin,
        mode=mode, likelihood_family=lf,
    )
    return sampler(proposal)
