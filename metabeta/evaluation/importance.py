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
