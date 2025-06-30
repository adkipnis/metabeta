from typing import Dict
import torch
from torch import distributions as D


def logPriorBeta(beta: torch.Tensor, mask: torch.Tensor):
    # beta: (b, d, s)
    # mask: (b, d)
    lp = D.Normal(0., 3.).log_prob(beta)
    lp = lp * mask.unsqueeze(-1)
    return lp.sum(dim=1) # (b, s)


def logPrior(beta: torch.Tensor, sigma: torch.Tensor, mask: torch.Tensor):
    lp_beta = logPriorBeta(beta, mask)
    lp_sigma = D.HalfNormal(1.).log_prob(sigma.sqrt())
    return lp_beta + lp_sigma


def logLikelihood(beta: torch.Tensor, sigma: torch.Tensor, batch: dict):
    # y: (b, n)
    # X: (b, n, d)
    # beta: (b, d, s)
    # sigma: (b, 1) or (b, s)
    y, X, ns = batch['y'], batch['X'], batch['n'].unsqueeze(-1)

    # compute sum of squared residuals
    mu = torch.einsum('bnd,bds->bns', X, beta) # (b, n, s)
    ssr = (y - mu).square().sum(dim=1) # (b, s)

    # Compute log likelihood per batch
    ll = (
        - 0.5 * ns * torch.tensor(2 * torch.pi).log() 
        - ns * sigma.log()
        - 0.5 * ssr / sigma.square()
    )
    return ll # (b, s)


def getImportanceWeights(log_likelihood: torch.Tensor,
                         log_prior: torch.Tensor,
                         log_q: torch.Tensor) -> torch.Tensor:
    log_w = log_likelihood + log_prior - log_q
    log_w_min, log_w_max = torch.quantile(
        log_w, torch.tensor([0.0, 0.98]), dim=-1).unsqueeze(-1)
    log_w = log_w.clamp(min=log_w_min, max=log_w_max)
    w = (log_w - log_w_max).exp()
    w = w / w.mean(dim=-1, keepdim=True)
    return w


def importanceSample(batch: dict, proposed: dict) -> Dict[str, torch.Tensor]:
    samples = proposed['samples']
    beta, sigma = samples[:, :-1], samples[:, -1] + 1e-9
    log_prior = logPrior(beta, sigma, batch['mask_d'])
    log_likelihood = logLikelihood(beta, sigma, batch)
    w = getImportanceWeights(log_likelihood, log_prior, proposed['log_prob'])
    n_eff = w.sum(-1).square() / (w.square().sum(-1) + 1e-5)
    sample_efficiency = n_eff / samples.shape[-1]
    return {'weights': w,
            'n_eff': n_eff,
            'sample_efficiency': sample_efficiency}


# =============================================================================
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from metabeta.utils import dsFilename
    from metabeta.dataset import LMDataset
    import numpy as np
    np.set_printoptions(suppress=True)

    # load data
    b = 250
    d = 1
    fn = dsFilename('ffx', 'test', d, 0, 50, b, -1)
    ds_raw = torch.load(fn, weights_only=False)
    ds = LMDataset(**ds_raw)
    dl = DataLoader(ds, batch_size=b, shuffle=False)
    batch = first_batch = next(iter(dl))
 
    # IS
    s = 10
    beta_ = batch['ffx'].unsqueeze(-1)
    sample_noise = D.Normal(0, 0.1).sample((b, d+1, s))
    beta = beta_ + sample_noise
    sigma = batch['sigma_error'].unsqueeze(-1).unsqueeze(-1).expand((b, 1, s))
    
    ll = logLikelihood(beta, sigma, batch)
    lp = logPrior(beta, sigma, batch['mask_d'])
    log_q = D.Normal(beta_, 0.1).log_prob(beta).sum(1, keepdim=True)
    
    w = getImportanceWeights(ll, lp, log_q)

