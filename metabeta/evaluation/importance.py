import torch
from torch import distributions as D
from metabeta.utils import dampen, maskedStd, weightedMean
from metabeta.evaluation.resampling import replace, powersample


def getImportanceWeights(
    log_likelihood: torch.Tensor,
    log_prior: torch.Tensor,
    log_q: torch.Tensor,
    constrain: bool = True,
) -> dict[str, torch.Tensor]:
    log_w = log_likelihood + log_prior - log_q
    if constrain:
        log_w = dampen(log_w, p=0.75)
    log_w_max = torch.quantile(log_w, 0.99, dim=-1).unsqueeze(-1)
    log_w = log_w.clamp(max=log_w_max)
    log_w = log_w - log_w_max
    w = log_w.exp()
    w = w / w.mean(dim=-1, keepdim=True)
    n_eff = w.sum(-1).square() / (w.square().sum(-1) + 1e-12)
    sample_efficiency = n_eff / w.shape[-1]

    return {"weights": w, "n_eff": n_eff, "sample_efficiency": sample_efficiency}


# =============================================================================
# Mixed Effects
# -----------------------------------------------------------------------------
class ImportanceLocal:
    def __init__(self, data: dict[str, torch.Tensor]):
        self.nu_ffx = data["nu_ffx"]
        self.tau_ffx = data["tau_ffx"]
        self.tau_rfx = data["tau_rfx"]
        self.tau_eps = data["tau_eps"]
        self.mask_d = data["mask_d"]
        self.mask_n = data["mask_n"]
        self.mask_m = data["mask_m"]
        self.y = data["y"].unsqueeze(-1)
        self.X = data["X"]
        self.Z = data["Z"]
        self.n = data["n"].unsqueeze(-1)
        self.n_i = data["n_i"].unsqueeze(-1)
        self.q = data["q"].unsqueeze(1)
        self.max_d = data["d"].max()

    def logPriorBeta(self, beta: torch.Tensor) -> torch.Tensor:
        # beta: (b, d)
        nu, tau, mask = self.nu_ffx, self.tau_ffx, self.mask_d
        lp = D.Normal(nu, tau + 1e-12).log_prob(beta)
        lp = (lp * mask).sum(dim=1)  # (b,)
        return lp

    def logPriorSigmasRfx(self, sigmas_rfx: torch.Tensor) -> torch.Tensor:
        # sigmas_rfx: (b, q)
        tau, mask = (self.tau_rfx + 1e-12), self.q
        lp = D.HalfNormal(tau).log_prob(sigmas_rfx)
        lp = (lp * mask).sum(dim=1)  # (b,)
        return lp

    def logPriorNoise(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (b,)
        tau = self.tau_eps
        lp = D.HalfNormal(tau).log_prob(sigma)
        return lp

    def logLikelihoodCond(
        self,
        ffx: torch.Tensor,  # (b, d)
        sigma: torch.Tensor,  # (b)
        rfx: torch.Tensor,  # (b, m, q, s)
    ) -> torch.Tensor:
        # conditional likelihood for rfx IS
        mu_g = torch.einsum("bmnd,bd->bmn", self.X, ffx).unsqueeze(-1)
        mu_l = torch.einsum("bmnd,bmds->bmns", self.Z, rfx)
        eps = self.y - mu_g - mu_l  # (b, m, n, s)
        ssr = eps.square().sum(dim=2)  # (b, m, s)
        sigma_ = sigma.view(-1, 1, 1)
        ll = (
            # - 0.5 * self.n_i * torch.tensor(2 * torch.pi).log()
            -self.n_i * sigma_.log() - 0.5 * ssr / sigma_.square()
        )  # (b, m, s)
        return ll

    def logLikelihoodRfx(
        self,
        rfx: torch.Tensor,  # (b, m, q, s)
        sigmas_rfx: torch.Tensor,  # (b, q)
    ) -> torch.Tensor:
        S = sigmas_rfx.unsqueeze(1).unsqueeze(-1) + 1e-12  # (b, 1, q, 1)
        means = torch.zeros_like(S)
        ll = D.Normal(means, S).log_prob(rfx)  # (b, m, q, s)
        mask = rfx != 0
        ll = (ll * mask).sum(dim=-2)  # (b, m, s)
        return ll

    def __call__(
        self,
        proposed: dict[str, dict[str, torch.Tensor]],
        resample: bool = False,
        upsample: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]]:
        # unpack
        log_q = proposed["local"]["log_prob"].clone()
        rfx = proposed["local"]["samples"].clone()
        samples_g = proposed["global"]["samples"].clone()
        weights_g = proposed["global"].get("weights", None)
        mean_g = weightedMean(samples_g, weights_g)
        ffx = mean_g[:, : self.max_d]
        sigmas_rfx = mean_g[:, self.max_d : -1]
        sigma_eps = mean_g[:, -1]

        # priors
        log_prior_beta = self.logPriorBeta(ffx)
        log_prior_noise = self.logPriorNoise(sigma_eps)
        log_prior_sigmas_rfx = self.logPriorSigmasRfx(sigmas_rfx)
        log_prior = log_prior_beta + log_prior_noise + log_prior_sigmas_rfx

        # likelihoods
        log_likelihood_cond = self.logLikelihoodCond(ffx, sigma_eps, rfx)
        log_likelihood_rfx = self.logLikelihoodRfx(rfx, sigmas_rfx)
        log_likelihood = log_likelihood_cond + log_likelihood_rfx

        # importance sampling
        factor = 5.0
        out = getImportanceWeights(
            log_likelihood, log_prior.view(-1, 1, 1), factor * log_q
        )
        out["weights"] = out["weights"].unsqueeze(-2).expand(*rfx.shape).clone()

        # resampling
        if resample or upsample:
            resamples = rfx.clone()
            if resample:
                resamples = replace(resamples, out["weights"], t=200)
                out.pop("weights", None)
            if upsample:
                b, m, d, s = resamples.shape
                resamples = resamples.reshape(b, m * d, s)
                resamples, _ = powersample(resamples, t=1000, method="yeo-johnson")
                resamples = resamples.reshape(b, m, d, 1000)
                out.pop("weights", None)
            out["samples"] = resamples

        # finalize
        proposed["local"].update(**out)
        return proposed


# -----------------------------------------------------------------------------
