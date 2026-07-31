"""
posthoc/importance.py — self-normalized importance sampling (SNIS) correction for flow posteriors.

`getImportanceWeights` already implements standard SNIS: log_w = log_likelihood + log_prior -
log_q, softmax-normalized (optionally PSIS-smoothed via `pareto`, dampened, temperature-scaled).
Current diagnostics returned: `pareto_k` (PSIS shape, when `pareto=True`) and `n_eff` /
`sample_efficiency` (effective sample size).

Findings (2026-07 posthoc ablation, 128 validation datasets per family, small models)
-------------------------------------------------------------------------------------
The default 'is' configuration (full=False, marginal=False, constrain=True, pareto=True)
barely changed any metric relative to raw flow samples. Two compounding causes:

1. Biased pseudo-target. With full=False and marginal=False the weight is
   ll(y | θ_g, rfx_flow) + lp(θ_g) − log q_g: the flow's rfx draws are plugged into the
   likelihood without either subtracting their proposal density (full=True) or integrating
   them out (marginal=True). The resulting "correction" does not target the posterior of
   the global parameters — it reweights toward datasets' flow-conditional rfx fit instead.
2. Dampening before PSIS. constrain=True applied dampen(log_w, p=0.5) — a signed sqrt of
   the log-weights — *before* az.psislw, flattening the weights toward uniform, so PSIS
   then smoothed already-neutered weights. PSIS is itself the principled regularizer;
   dampening on top mostly cancels whatever signal the weights carried.

Remedies implemented here: `marginal=True` integrates rfx out exactly (Normal; supports
correlated Σ_rfx via the L_corr path in logMarginalLikelihoodNormal and includes the LKJ
prior via corr_prior), `rb_redraw=True` re-draws rfx from the exact Normal-Normal
conditional per weighted global sample (Rao-Blackwellisation), and dampening is no longer
applied in the pareto branch.

Resample-move (ResampleMoveSampler, added for the 2026-07 post-log-det-fix ablation):
systematic resampling on the PSIS-smoothed marginal weights followed by K vectorized
independence-MH rejuvenation sweeps with fresh flow proposals — converts weight
degeneracy into actual correction instead of a uniform-weight fallback. Implemented
because the huge model keeps tripping the pareto_k guardrail on ~42% of test datasets
(and IMH's duplicate-heavy chains under-disperse the globals there).

TODO
----
- Target energy gap diagnostic: mean weighted `-log p̃(theta)` under the corrected samples
  minus the same quantity under a reference (e.g. NUTS) sample, when a reference is
  available — flags whether corrected samples land in the right density region, not just
  whether weights are stable (see Ko & Domke, arXiv:2605.26419, Section D.3/D.6).
"""

import argparse
import math
import time

import arviz as az
import torch
from metabeta.models.approximator import Approximator
from metabeta.utils.dataloader import toDevice
from metabeta.utils.results import Proposal, joinProposals
from metabeta.utils.regularization import dampen, corrLowerToUnconstrained, unconstrainedToCholesky
from metabeta.utils.constants import hasSigmaEps
from metabeta.utils.families import (
    logProbFfx,
    logProbSigma,
    logProbRfx,
    logProbRfxCorrelated,
    logProbCorrRfx,
    logLikelihood,
    logMarginalLikelihoodNormal,
    sampleRfxConditionalNormal,
)
from metabeta.utils.preprocessing import rescaleData


class ImportanceSampler:
    def __init__(
        self,
        data: dict[str, torch.Tensor],
        constrain: bool = True,  # dampen log-weights (non-pareto branch only)
        full: bool = False,  # incorporate RFX priors and local log-prob in IS weight
        corr_prior: bool = True,  # include LKJ prior on z_corr in IS weight (global param — should be True)
        marginal: bool = False,  # use marginal likelihood (Normal only); integrates rfx out
        rb_redraw: bool = False,  # redraw rfx from the exact conditional (requires marginal)
        temperature: float = 1.0,  # softmax temperature
        pareto: bool = False,  # use Pareto smoothing (PSIS)
        k_threshold: float = 0.7,  # PSIS k above which a dataset falls back to uniform weights
        sir: bool = False,  # use Sampling Importance Resampling (SIR)
        n_sir: int = 25,  # size of SIR re-sample
        likelihood_family: int = 0,
        eps: float = 1e-12,
    ) -> None:
        if marginal and likelihood_family != 0:
            raise ValueError('marginal IS is only implemented for the Normal likelihood family')
        if rb_redraw and not marginal:
            raise ValueError('rb_redraw requires marginal=True (Rao-Blackwellised weights)')
        self.constrain = constrain
        self.full = full
        self.corr_prior = corr_prior
        self.marginal = marginal
        self.rb_redraw = rb_redraw
        self.temperature = temperature
        self.pareto = pareto
        self.k_threshold = k_threshold
        self.sir = sir
        self.n_sir = n_sir
        self.likelihood_family = likelihood_family
        self.has_sigma_eps = hasSigmaEps(likelihood_family)
        self.eps = eps

        # prior
        self.nu_ffx = data['nu_ffx'].unsqueeze(-2)   # (b, 1, d)
        self.tau_ffx = data['tau_ffx'].unsqueeze(-2) + self.eps   # (b, 1, d)
        self.tau_rfx = data['tau_rfx'].unsqueeze(-2) + self.eps   # (b, 1, q)
        self.family_ffx = data['family_ffx']   # (b,)
        self.family_sigma_rfx = data['family_sigma_rfx']   # (b,)
        if self.has_sigma_eps:
            self.tau_eps = data['tau_eps'].unsqueeze(-1) + self.eps   # (b, 1)
            self.family_sigma_eps = data['family_sigma_eps']   # (b,)
        self.eta_rfx = data.get('eta_rfx')   # (b,) or None

        # observations
        self.X = data['X']   # (b, m, n, d)
        self.Z = data['Z']   # (b, m, n, q)
        self.y = data['y'].unsqueeze(-1)   # (b, m, n, 1)

        # masks
        self.mask_d = data['mask_d'].unsqueeze(-2)    # (b, 1, d)
        self.mask_q = data['mask_q'].unsqueeze(-2)    # (b, 1, q)
        self.mask_mq = data['mask_mq'].unsqueeze(-2)  # (b, m, 1, q)
        self.mask_m = data['mask_m'].unsqueeze(-1)    # (b, m, 1)
        self.mask_n = data['mask_n'].unsqueeze(-1)    # (b, m, n, 1)

    def _getLCorr(self, proposal: Proposal) -> torch.Tensor | None:
        """Cholesky of the rfx correlation matrix from the proposal's z_corr dims."""
        if proposal.d_corr == 0:
            return None
        r_corr = proposal.samples_g[..., -proposal.d_corr :]  # (b, s, d_corr)
        z_corr = corrLowerToUnconstrained(r_corr, proposal.q)
        return unconstrainedToCholesky(z_corr, proposal.q)  # (b, s, q, q)

    def _logPriorGlobals(
        self, proposal: Proposal
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log prior of the global parameters (ffx, sigmas, z_corr).

        Returns (lp, ffx, sigma_eps) with ffx zero-padded to the data's d and
        sigma_eps zeros when the likelihood has none — shared by the exact and
        Laplace (posthoc/laplace_glmm.py) marginal samplers.
        """
        ffx = proposal.ffx

        pad_d = self.nu_ffx.shape[-1] - ffx.shape[-1]
        if pad_d > 0:
            ffx = torch.nn.functional.pad(ffx, (0, pad_d), 'constant', 0)
        lp = logProbFfx(ffx, self.nu_ffx, self.tau_ffx, self.family_ffx, self.mask_d)

        if self.has_sigma_eps:
            sigma_eps = proposal.sigma_eps
            lp = lp + logProbSigma(sigma_eps, self.tau_eps, self.family_sigma_eps)
        else:
            sigma_eps = ffx.new_zeros(ffx.shape[:2])

        # sigma_rfx is a global parameter included in log q_global for all families,
        # so its prior must be in the numerator to keep the IS weight balanced.
        lp = lp + logProbSigma(proposal.sigma_rfx, self.tau_rfx, self.family_sigma_rfx, self.mask_q)

        # corr_rfx: stored as constrained r (lower triangle); unconstrain to z for the prior
        if self.corr_prior and proposal.d_corr > 0 and self.eta_rfx is not None:
            r_corr = proposal.samples_g[..., -proposal.d_corr :]  # (b, s, d_corr)
            z_corr = corrLowerToUnconstrained(r_corr, proposal.q)
            lp = lp + logProbCorrRfx(z_corr, proposal.q, self.eta_rfx)

        return lp, ffx, sigma_eps

    def unnormalizedPosterior(self, proposal: Proposal) -> tuple[torch.Tensor, torch.Tensor]:
        lp, ffx, sigma_eps = self._logPriorGlobals(proposal)
        sigma_rfx = proposal.sigma_rfx

        if self.marginal:
            # integrate rfx out analytically — weight is a function of global params only;
            # pass L_corr so correlated-rfx samples target the correct marginal
            ll = logMarginalLikelihoodNormal(
                ffx,
                sigma_rfx,
                sigma_eps,
                self.y,
                self.X,
                self.Z,
                self.mask_n,
                self.mask_m,
                L_corr=self._getLCorr(proposal),
            )
        else:
            rfx = proposal.rfx
            if self.full:
                if proposal.d_corr > 0:
                    r_corr = proposal.samples_g[..., -proposal.d_corr :]
                    z_corr_full = corrLowerToUnconstrained(r_corr, proposal.q)
                    L = unconstrainedToCholesky(z_corr_full, proposal.q)
                    lp = lp + logProbRfxCorrelated(rfx, sigma_rfx, L, self.mask_mq)
                else:
                    lp = lp + logProbRfx(rfx, sigma_rfx, self.mask_mq)
            ll = logLikelihood(
                ffx,
                sigma_eps,
                rfx,
                self.y,
                self.X,
                self.Z,
                self.mask_n,
                likelihood_family=self.likelihood_family,
            )
        return ll, lp

    def __call__(self, proposal: Proposal) -> Proposal:
        # posterior log probs
        log_q_g, log_q_l = proposal.log_probs
        lq = log_q_g
        if self.full:
            lq = lq + (log_q_l * self.mask_m).sum(1)

        # log likelihood, log prior, log proposal posterior
        ll, lp = self.unnormalizedPosterior(proposal)

        # importance sampling
        proposal.is_results = self.getImportanceWeights(ll, lp, lq)

        # Rao-Blackwellisation: replace flow rfx with exact conditional draws so the
        # weighted (θ_g, rfx | θ_g) pairs form a consistent joint-posterior sample
        if self.rb_redraw:
            self._redrawRfx(proposal)

        # take subset for SIR
        if self.sir:
            idx = self.getSirIndices(proposal.is_results['weights'])
            proposal.subset(idx)
            proposal.is_results = {}
        return proposal

    def _redrawRfx(self, proposal: Proposal) -> None:
        ffx = proposal.ffx
        pad_d = self.X.shape[-1] - ffx.shape[-1]
        if pad_d > 0:
            ffx = torch.nn.functional.pad(ffx, (0, pad_d), 'constant', 0)
        rfx = sampleRfxConditionalNormal(
            ffx,
            proposal.sigma_rfx,
            proposal.sigma_eps,
            self.y,
            self.X,
            self.Z,
            self.mask_n,
            self.mask_m,
            L_corr=self._getLCorr(proposal),
        )
        proposal.data['local']['samples'] = rfx
        # flow rfx density no longer applies; nothing downstream reads log_prob_l
        # once is_results['weights'] is set
        proposal.data['local']['log_prob'] = torch.zeros_like(proposal.log_prob_l)

    def getImportanceWeights(
        self,
        log_likelihood: torch.Tensor,
        log_prior: torch.Tensor,
        log_q: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = {}
        log_w = log_likelihood + log_prior - log_q

        # regularize
        if self.pareto:
            # No dampening here: PSIS is the principled tail regularizer, and dampening
            # log-weights before smoothing flattens them toward uniform, cancelling the
            # correction (see module docstring Findings).
            log_w_np, pareto_k_np = az.psislw(log_w)
            out['log_w'] = log_w.new_tensor(log_w_np)
            out['pareto_k'] = log_w.new_tensor(pareto_k_np)
        else:
            if self.constrain:
                log_w = dampen(log_w, p=0.80)
            log_w_max = torch.quantile(log_w, 0.99, dim=-1).unsqueeze(-1)
            out['log_w'] = log_w.clamp(max=log_w_max) - log_w_max

        # normalized weights
        w = torch.softmax(out['log_w'] / self.temperature, dim=-1)
        w = torch.where(torch.isfinite(w), w, 0)

        # guardrail: datasets with unreliable weights (PSIS k above threshold) fall
        # back to uniform weights (i.e. raw flow) instead of applying a bad correction
        if self.pareto:
            fallback = out['pareto_k'] > self.k_threshold  # (b,)
            uniform = torch.full_like(w, 1.0 / w.shape[-1])
            w = torch.where(fallback.unsqueeze(-1), uniform, w)
            out['fallback'] = fallback
        out['weights'] = w

        # diagnostics (on the weights actually used downstream)
        out['n_eff'] = w.sum(-1).square() / (w.square().sum(-1) + 1e-12)
        out['sample_efficiency'] = out['n_eff'] / w.shape[-1]
        out['max_weight'] = w.max(-1).values
        out['entropy_ratio'] = -(w * (w + 1e-12).log()).sum(-1) / math.log(w.shape[-1])
        return out

    def getSirIndices(self, w: torch.Tensor) -> torch.Tensor:
        """Use inverse method to get {n_sir} coupled draws from w. Return the indices of said draws."""
        n_sir = self.n_sir
        b, s = w.shape

        # get
        cdf = torch.cumsum(w, dim=-1)

        # get random offset and {n_sir} equidistant quantiles
        u0 = torch.rand(b, 1, device=w.device) / n_sir
        u = u0 + torch.arange(n_sir, device=w.device).view(1, -1) / n_sir

        # get indices of these quantiles, drawing proportionally from w
        idx = torch.searchsorted(cdf, u, right=True).clamp(max=s - 1)
        return idx


class ResampleMoveSampler:
    """Resample-move correction: marginal SNIS → systematic resampling → K
    independence-MH rejuvenation sweeps with fresh flow proposals.

    Where plain marginal SNIS falls back to uniform weights when PSIS k exceeds
    its threshold (i.e. applies no correction), this resamples from the
    PSIS-smoothed weights and rejuvenates every particle with fresh flow
    proposals: weight degeneracy becomes particle duplication, and the MH sweeps
    (which leave the marginal posterior invariant) restore diversity. Unlike
    IMH — whose low-acceptance chains duplicate samples and under-disperse the
    globals — every rejuvenation proposal here is fresh, so K sweeps at
    acceptance rate a leave only ≈ (1 − a)^K of particles unmoved.

    Normal likelihood only (exact marginal weights); rfx are drawn from the
    exact Normal-Normal conditional after the final sweep, as in rb_redraw.
    """

    def __init__(
        self,
        model: Approximator,
        data_raw: dict[str, torch.Tensor],  # unrescaled batch, fed to model.estimate
        data: dict[str, torch.Tensor],  # rescaled batch, target space of the weights
        n_sweeps: int = 15,
        likelihood_family: int = 0,
        device: str = 'cpu',
        eps: float = 1e-12,
    ) -> None:
        if likelihood_family != 0:
            raise ValueError('resample-move requires the Normal likelihood (marginal weights)')
        self.model = model
        self.n_sweeps = n_sweeps
        self.device = device
        self.sd_y = data_raw['sd_y']
        self.data_dev = toDevice(data_raw, device)
        self._is = ImportanceSampler(
            data, marginal=True, corr_prior=True, pareto=True, likelihood_family=0, eps=eps
        )

    def _logWeights(self, proposal: Proposal) -> torch.Tensor:
        """Raw (unsmoothed) marginal log IS weights (b, s) — the MH target ratio."""
        ll, lp = self._is.unnormalizedPosterior(proposal)
        return ll + lp - proposal.log_prob_g

    def _freshProposal(self, n_samples: int) -> Proposal:
        with torch.no_grad():
            fresh = self.model.estimate(self.data_dev, n_samples=n_samples)
        fresh.to('cpu')
        fresh.rescale(self.sd_y)
        return fresh

    def __call__(self, proposal: Proposal) -> tuple[Proposal, dict]:
        t0 = time.perf_counter()
        b, s = proposal.samples_g.shape[:2]

        # initial resample from the PSIS-smoothed weights (raw weights drive the
        # MH ratio below; smoothing only stabilises the resampling step)
        lw = self._logWeights(proposal)  # (b, s)
        log_w_np, pareto_k_np = az.psislw(lw)
        w = torch.softmax(lw.new_tensor(log_w_np), dim=-1)
        w = torch.where(torch.isfinite(w), w, 0)
        self._is.n_sir = s
        idx = self._is.getSirIndices(w)  # (b, s)
        cur_g = torch.gather(
            proposal.samples_g, 1, idx.unsqueeze(-1).expand(-1, -1, proposal.samples_g.shape[-1])
        ).clone()
        cur_lw = torch.gather(lw, 1, idx).clone()

        # rejuvenation sweeps: one fresh independent proposal per particle
        moved = torch.zeros_like(cur_lw, dtype=torch.bool)
        acc_hist = []
        for _ in range(self.n_sweeps):
            fresh = self._freshProposal(s)
            fresh_lw = self._logWeights(fresh)  # (b, s)
            log_alpha = (fresh_lw - cur_lw).clamp(max=0.0)
            accept = torch.rand_like(log_alpha).log() < log_alpha  # (b, s)
            cur_g = torch.where(accept.unsqueeze(-1), fresh.samples_g, cur_g)
            cur_lw = torch.where(accept, fresh_lw, cur_lw)
            moved |= accept
            acc_hist.append(accept.float().mean(-1))  # (b,)

        # final rfx from the exact conditional given the rejuvenated globals
        m = self._is.X.shape[1]
        q = self._is.Z.shape[-1]
        proposed = {
            'global': {'samples': cur_g, 'log_prob': cur_g.new_zeros(b, s)},
            'local': {'samples': cur_g.new_zeros(b, m, s, q), 'log_prob': cur_g.new_zeros(b, m, s)},
        }
        out = Proposal(proposed, has_sigma_eps=self._is.has_sigma_eps, d_corr=proposal.d_corr)
        self._is._redrawRfx(out)

        out.tpd = (proposal.tpd or 0.0) + (time.perf_counter() - t0)
        diagnostics = {
            'accept_rate': torch.stack(acc_hist, dim=-1),  # (b, n_sweeps)
            'moved_frac': moved.float().mean(-1),  # (b,)
            'pareto_k': lw.new_tensor(pareto_k_np),  # (b,)
        }
        return out, diagnostics


def runIS(
    model: Approximator,
    data: dict[str, torch.Tensor],
    cfg: argparse.Namespace,
) -> Proposal:
    # raw proposal
    proposal = model.estimate(data, n_samples=cfg.n_samples)

    # unnormalize proposal and batch
    if cfg.rescale:
        proposal.rescale(data['sd_y'])
        data = rescaleData(data)

    # importance weighing
    lf = getattr(cfg, 'likelihood_family', 0)
    imp_sampler = ImportanceSampler(data, sir=False, likelihood_family=lf)
    proposal = imp_sampler(proposal)
    return proposal


def runSIR(
    model: Approximator,
    data: dict[str, torch.Tensor],
    cfg: argparse.Namespace,
) -> Proposal:
    # prepare rescaling
    if cfg.rescale:
        data_eval = rescaleData(data)
    else:
        data_eval = data

    # init importance sampler
    lf = getattr(cfg, 'likelihood_family', 0)
    n_sir = cfg.n_samples // cfg.sir_iter
    n_proposal = getattr(cfg, 'sir_n_proposal', cfg.n_samples)
    imp_sampler = ImportanceSampler(data_eval, sir=True, n_sir=n_sir, likelihood_family=lf)
    selected = []
    n_remaining = cfg.n_samples
    while n_remaining > 0:
        proposal = model.estimate(data, n_samples=n_proposal)
        if cfg.rescale:
            proposal.rescale(data['sd_y'])
        proposal = imp_sampler(proposal)
        selected.append(proposal)
        n_remaining -= proposal.n_samples
    return joinProposals(selected)
