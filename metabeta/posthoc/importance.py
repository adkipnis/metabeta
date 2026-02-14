import math
import arviz as az
import torch
from torch import distributions as D
from metabeta.utils.evaluation import Proposal
from metabeta.utils.regularization import dampen


class ImportanceSampler:
    def __init__(
        self,
        data: dict[str, torch.Tensor],
        constrain: bool = True,
        full: bool = True,  # incorporate sigma_rfx and rfx priors
        pareto: bool = False,  # use Pareto smoothing (PSIS)
        eps: float = 1e-12,
    ) -> None:
        self.constrain = constrain
        self.full = full
        self.pareto = pareto
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

    def logPriorFfx(self, ffx: torch.Tensor) -> torch.Tensor:
        # ffx (b, s, d)
        dist = D.Normal(self.nu_ffx, self.tau_ffx)
        lp = dist.log_prob(ffx)
        lp = (lp * self.mask_d).sum(-1)
        return lp   # (b, s)

    def logPriorSigmaRfx(self, sigma_rfx: torch.Tensor) -> torch.Tensor:
        # sigma_rfx (b, s, q)
        dist = D.HalfNormal(scale=self.tau_rfx)
        lp = dist.log_prob(sigma_rfx)
        lp = (lp * self.mask_q).sum(-1)
        return lp   # (b, s)

    def logPriorSigmaEps(self, sigma_eps: torch.Tensor) -> torch.Tensor:
        # sigma_eps (b, s)
        dist = D.StudentT(df=4, loc=0, scale=self.tau_eps)
        lp = dist.log_prob(sigma_eps) + math.log(2.0)
        return lp   # (b, s)

    def logPriorRfx(self, rfx: torch.Tensor, sigma_rfx: torch.Tensor) -> torch.Tensor:
        # rfx (b, m, s, q), sigma_rfx (b, s, q)
        scale = sigma_rfx.unsqueeze(1) + self.eps
        dist = D.Normal(loc=0, scale=scale)
        lp = dist.log_prob(rfx)
        lp = (lp * self.mask_mq).sum(1)
        lp = lp.sum(-1)
        return lp   # (b, s)

    def logLikelihoodCond(
        self,
        ffx: torch.Tensor,  # (b, s, d)
        sigma_eps: torch.Tensor,  # (b, s)
        rfx: torch.Tensor,  # (b, m, s, q)
    ) -> torch.Tensor:
        mu_g = torch.einsum('bmnd,bsd->bmns', self.X, ffx)
        mu_l = torch.einsum('bmnq,bmsq->bmns', self.Z, rfx)
        loc = mu_g + mu_l
        scale = sigma_eps.unsqueeze(1).unsqueeze(1) + 1e-12
        dist = D.Normal(loc=loc, scale=scale)
        ll = dist.log_prob(self.y)
        lp = (ll * self.mask_n).sum(dim=(1, 2))
        return lp   # (b, s)

    def __call__(self, proposal: Proposal) -> dict[str, torch.Tensor]:
        # unpack parameters
        ffx, sigma_rfx, sigma_eps, rfx = proposal.parameters

        # posterior log probs
        log_q_g, log_q_l = proposal.log_probs
        lq = log_q_g + (log_q_l * self.mask_m).sum(1)

        # log priors
        lp_ffx = self.logPriorFfx(ffx)
        lp_sigma_eps = self.logPriorSigmaEps(sigma_eps)
        lp = lp_ffx + lp_sigma_eps

        # regularize Sigma(RFX)
        if self.full:
            lp_sigma_rfx = self.logPriorSigmaRfx(sigma_rfx)
            lp_rfx = self.logPriorRfx(rfx, sigma_rfx)
            lp = lp + lp_sigma_rfx + lp_rfx

        # conditional log likelihood
        ll = self.logLikelihoodCond(ffx, sigma_eps, rfx)

        return self.getImportanceWeights(ll, lp, lq)

    def getImportanceWeights(
        self,
        log_likelihood: torch.Tensor,
        log_prior: torch.Tensor,
        log_q: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        log_w = log_likelihood + log_prior - log_q

        # regularize
        if self.pareto:
            if self.constrain:
                log_w = dampen(log_w, p=0.50)
            log_w_, pareto_k = az.psislw(log_w)
            log_w = log_w.new_tensor(log_w_)
            pareto_k = log_w.new_tensor(pareto_k)
        else:
            if self.constrain:
                log_w = dampen(log_w, p=0.70)
            log_w_max = torch.quantile(log_w, 0.99, dim=-1).unsqueeze(-1)
            log_w = log_w.clamp(max=log_w_max) - log_w_max

        # weights and number of effective samples
        w = log_w.exp()
        w = torch.where(torch.isfinite(w), w, 0)
        n_eff = w.sum(-1).square() / (w.square().sum(-1) + 1e-12)

        # normalized weights
        log_w_norm = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)
        w_norm = log_w_norm.exp()
        w_norm = torch.where(torch.isfinite(w_norm), w_norm, 0)

        # finalize
        out = {
            'weights': w,
            'weights_norm': w_norm,
            'n_eff': n_eff,
            'sample_efficiency': n_eff / w.shape[-1],
        }
        if self.pareto:
            out['pareto_k'] = pareto_k
        return out
