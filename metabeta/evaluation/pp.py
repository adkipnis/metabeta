import torch
from torch import distributions as D
import seaborn as sns


def posteriorPredictiveSample(
    ds: dict[str, torch.Tensor],
    proposed: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    from metabeta.utils import weightedMean

    # prepare observed
    X, Z, y = ds["X"], ds["Z"], ds["y"]
    d = X.shape[-1]
    mask = (y != 0).unsqueeze(-1)

    # prepare samples
    samples_g = proposed["global"]["samples"]
    samples_l = proposed["local"]["samples"]
    weights_l = proposed["local"].get("weights", None)
    rfx = weightedMean(samples_l, weights_l).to(X.dtype)
    ffx = samples_g[:, :d].to(X.dtype)
    b = len(ffx)
    sigma_eps = samples_g[:, -1].view(b, 1, 1, -1)

    # construct posterior predictive
    mu_g = torch.einsum("bmnd,bds->bmns", X, ffx)
    mu_l = torch.einsum("bmnq,bmq->bmn", Z, rfx).unsqueeze(-1)
    posterior_predictive = D.Normal(mu_g + mu_l, sigma_eps)

    # sample from posterior predictive
    y_rep = posterior_predictive.sample((1,)).squeeze(0)
    y_rep *= mask
    return y_rep


def weightSubset(
    weights: torch.Tensor,  #  (b, s)
    alpha: float = 0.01,
) -> torch.Tensor:
    # get a mask of shape (b, s) that subsets the 1-alpha most likely samples
    b, s = weights.shape
    root = torch.tensor(alpha).expand(b, 1)
    w_sorted, w_idx = weights.sort(dim=-1, descending=False)
    w_inv = torch.argsort(w_idx, -1)
    cdf = torch.cumsum(w_sorted, -1) / s
    cdf = cdf.contiguous()
    r_idx = torch.searchsorted(cdf, root).clamp(max=s - 1)
    mask = torch.arange(s).unsqueeze(0) >= r_idx
    mask = mask.gather(-1, w_inv)
    return mask


