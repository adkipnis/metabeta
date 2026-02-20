import arviz as az
import torch
from metabeta.utils.evaluation import Proposal
from metabeta.utils.regularization import dampen
from metabeta.utils.probabilities import (
    logPriorFfx,
    logPriorSigmaRfx,
    logPriorSigmaEps,
    logPriorRfx,
    logLikelihoodCond,
)


class ImportanceSampler:
    def __init__(
        self,
        data: dict[str, torch.Tensor],
        constrain: bool = True,
        full: bool = True,  # incorporate sigma_rfx and rfx priors
        temperature: float = 3.0, # softmax temperature
        pareto: bool = False,  # use Pareto smoothing (PSIS)
        sir: bool = False,  # use Sampling Importance Resampling (SIR)
        n_sir: int = 25,  # size of SIR re-sample
        eps: float = 1e-12,
    ) -> None:
        self.constrain = constrain
        self.full = full
        self.temperature = temperature
        self.pareto = pareto
        self.sir = sir
        self.n_sir = n_sir
        self.eps = eps

        # prior
        self.nu_ffx = data['nu_ffx'].unsqueeze(-2)   # (b, 1, d)
        self.tau_ffx = data['tau_ffx'].unsqueeze(-2) + self.eps   # (b, 1, d)
        self.tau_rfx = data['tau_rfx'].unsqueeze(-2) + self.eps   # (b, 1, q)
        self.tau_eps = data['tau_eps'].unsqueeze(-1) + self.eps   # (b, 1)

        # observations
        self.X = data['X']   # (b, m, n, d)
        self.Z = data['Z']   # (b, m, n, q)
        self.y = data['y'].unsqueeze(-1)   # (b, m, n, 1)

        # masks
        self.mask_d = data['mask_d'].unsqueeze(-2)   # (b, 1, d)
        self.mask_q = data['mask_q'].unsqueeze(-2)   # (b, 1, q)
        self.mask_mq = data['mask_mq'].unsqueeze(-2)   # (b, m, 1, q)
        self.mask_m = data['mask_m'].unsqueeze(-1)   # (b, m, 1)
        self.mask_n = data['mask_n'].unsqueeze(-1)   # (b, m, n, 1)

    def unnormalizedPosterior(self, proposal: Proposal) -> tuple[torch.Tensor, ...]:
        # unpack parameters
        ffx, sigma_rfx, sigma_eps, rfx = proposal.parameters

        # log priors
        lp_ffx = logPriorFfx(ffx, self.nu_ffx, self.tau_ffx, self.mask_d)
        lp_sigma_eps = logPriorSigmaEps(sigma_eps, self.tau_eps)
        lp = lp_ffx + lp_sigma_eps

        # regularize Sigma(RFX)
        if self.full:
            lp_sigma_rfx = logPriorSigmaRfx(sigma_rfx, self.tau_rfx, self.mask_q)
            lp_rfx = logPriorRfx(rfx, sigma_rfx, self.mask_mq)
            lp = lp + lp_sigma_rfx + lp_rfx

        # conditional log likelihood
        ll = logLikelihoodCond(
            ffx, sigma_eps, rfx, self.y, self.X, self.Z, self.mask_n)
        return ll, lp

    def __call__(self, proposal: Proposal) -> Proposal:
        # posterior log probs
        log_q_g, log_q_l = proposal.log_probs
        lq = log_q_g + (log_q_l * self.mask_m).sum(1)
        
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
                log_w = dampen(log_w, p=0.70)
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
