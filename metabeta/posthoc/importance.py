import argparse
import arviz as az
import torch
from metabeta.models.approximator import Approximator
from metabeta.utils.evaluation import Proposal, joinProposals
from metabeta.utils.regularization import dampen, unconstrainedToCholeskyCorr
from metabeta.utils.families import (
    hasSigmaEps,
    logProbFfx,
    logProbSigma,
    logProbRfx,
    logProbRfxCorrelated,
    logProbCorrRfx,
    logLikelihood,
    logMarginalLikelihoodNormal,
)
from metabeta.utils.preprocessing import rescaleData


class ImportanceSampler:
    def __init__(
        self,
        data: dict[str, torch.Tensor],
        constrain: bool = True,
        full: bool = False,  # incorporate RFX priors and local log-prob in IS weight
        corr_prior: bool = True,  # include LKJ prior on z_corr in IS weight (global param — should be True)
        marginal: bool = False,  # use marginal likelihood (Normal only); integrates rfx out
        temperature: float = 1.0,  # softmax temperature
        pareto: bool = False,  # use Pareto smoothing (PSIS)
        sir: bool = False,  # use Sampling Importance Resampling (SIR)
        n_sir: int = 25,  # size of SIR re-sample
        likelihood_family: int = 0,
        eps: float = 1e-12,
    ) -> None:
        if marginal and likelihood_family != 0:
            raise ValueError('marginal IS is only implemented for the Normal likelihood family')
        self.constrain = constrain
        self.full = full
        self.corr_prior = corr_prior
        self.marginal = marginal
        self.temperature = temperature
        self.pareto = pareto
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

    def unnormalizedPosterior(self, proposal: Proposal) -> tuple[torch.Tensor, torch.Tensor]:
        ffx = proposal.ffx
        sigma_rfx = proposal.sigma_rfx

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
        lp = lp + logProbSigma(sigma_rfx, self.tau_rfx, self.family_sigma_rfx, self.mask_q)

        # corr_rfx: modeled in unconstrained z-space by the global flow — add matching prior
        if self.corr_prior and proposal.d_corr > 0 and self.eta_rfx is not None:
            z_corr = proposal.samples_g[..., -proposal.d_corr :]  # (b, s, d_corr)
            lp = lp + logProbCorrRfx(z_corr, proposal.q, self.eta_rfx)

        if self.marginal:
            # integrate rfx out analytically — weight is a function of global params only
            ll = logMarginalLikelihoodNormal(
                ffx, sigma_rfx, sigma_eps, self.y, self.X, self.Z, self.mask_n, self.mask_m,
            )
        else:
            rfx = proposal.rfx
            if self.full:
                if proposal.d_corr > 0:
                    L = unconstrainedToCholeskyCorr(
                        proposal.samples_g[..., -proposal.d_corr :], proposal.q
                    )
                    lp = lp + logProbRfxCorrelated(rfx, sigma_rfx, L, self.mask_mq)
                else:
                    lp = lp + logProbRfx(rfx, sigma_rfx, self.mask_mq)
            ll = logLikelihood(
                ffx, sigma_eps, rfx, self.y, self.X, self.Z, self.mask_n,
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

        # take subset for SIR
        if self.sir:
            idx = self.getSirIndices(proposal.is_results['weights'])
            proposal.subset(idx)
            proposal.is_results = {}
        return proposal

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
            if self.constrain:
                log_w = dampen(log_w, p=0.50)
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
        out['weights'] = w

        # effective sample size
        out['n_eff'] = w.sum(-1).square() / (w.square().sum(-1) + 1e-12)
        out['sample_efficiency'] = out['n_eff'] / w.shape[-1]
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
