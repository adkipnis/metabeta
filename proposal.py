from math import sqrt
import torch
from torch import nn
from torch import distributions as D
mse = nn.MSELoss(reduction='none')

# --------------------------------------------------------------------------
# discrete posterior
def equidistantBins(low: float, high: float, n_bins: int) -> torch.Tensor:
    return torch.linspace(low, high, steps=n_bins)

def normalBins(scale: float, n_bins: int) -> torch.Tensor:
    quantiles = torch.linspace(0., 1., n_bins+2)
    quantiles = quantiles[1:-1]
    return D.Normal(0, scale).icdf(quantiles)

def halfNormalBins(scale: float, n_bins: int) -> torch.Tensor:
    quantiles = torch.linspace(0., 1., n_bins+1)
    quantiles = quantiles[:-1]
    return D.HalfNormal(scale).icdf(quantiles)

class DiscreteProposal(nn.Module):
    def __init__(self, bins: torch.Tensor, # (m,)
                 ):
        super().__init__()
        self.register_buffer('bins', bins)
        self.register_buffer('widths', bins[1:] - bins[:-1])
        self.centers = self.bins[:-1] + self.widths/2
        self.n_bins = len(bins) - 1

    def getBindex(self, vals: torch.Tensor) -> torch.Tensor:
        vals = vals.contiguous()
        indices = torch.searchsorted(self.bins, vals) - 1
        indices[vals <= self.bins[0]] = 0
        indices[vals >= self.bins[-1]] = self.n_bins - 1
        return indices

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits, -1)
        return p @ self.centers

    def mode(self, logits: torch.Tensor) -> torch.Tensor:
        indices = logits.argmax(-1)
        return self.centers[indices]

    def variance(self, logits: torch.Tensor, mean: torch.Tensor, # (batch, seq_len)
                 ) -> torch.Tensor:
        p = torch.softmax(logits, -1)
        squared_diff = (self.centers.view(1, 1, -1) - mean.unsqueeze(-1)).square()
        return torch.sum(squared_diff * p, dim=-1)
 
    def mseLoss(self, targets: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        means = self.mean(logits)
        return mse(targets, means)
 
    def nllLoss(self,
                targets: torch.Tensor, # (batch, n, d)
                logits: torch.Tensor, # (batch, n, d, n_bins)
                ) -> torch.Tensor:
        indices = self.getBindex(targets)
        bin_log_probs = torch.log_softmax(logits, -1)
        scaled_bucket_log_probs = bin_log_probs - torch.log(self.widths)
        return -scaled_bucket_log_probs.gather(-1, indices.unsqueeze(-1)).squeeze(-1)

    def loss(self, loss_type: str, targets, logits):
        if loss_type == "mse":
            return self.mseLoss(targets, logits)
        elif loss_type == "nll":
            return self.nllLoss(targets, logits)
        else:
            raise ValueError(f"loss type {loss_type} unknown")


# --------------------------------------------------------------------------
# mixture posterior

class MixtureProposal(nn.Module):
    def __init__(self, comp_dist = D.Normal):
        super().__init__()
        self.comp_dist = comp_dist
    
    def construct(self, locs: torch.Tensor, scales: torch.Tensor, weights: torch.Tensor) -> D.Distribution:
        m = locs.shape[-1]
        if m > 1:
            mix = D.Categorical(weights)
            comp = self.comp_dist(locs, scales)
            proposal = D.MixtureSameFamily(mix, comp)
        else:
            proposal = self.comp_dist(locs.squeeze(-1), scales.squeeze(-1))
        return proposal

    def mean(self, locs: torch.Tensor, scales: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if self.comp_dist == D.Normal:
            return (locs * weights).sum(dim=-1)
        else:
            proposal = self.construct(locs, scales, weights)
            return proposal.mean

    def variance(self, locs: torch.Tensor, scales: torch.Tensor,
                 weights: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        if self.comp_dist == D.Normal:
            second_moments = locs.square() + scales.square()
            return (second_moments * weights).sum(dim=-1) - mean.square()
        else:
            proposal = self.construct(locs, scales, weights)
            return proposal.variance

    def mseLoss(self, locs: torch.Tensor, scales: torch.Tensor, weights: torch.Tensor,
                target: torch.Tensor, target_type: str) -> torch.Tensor:
        m = locs.shape[-1]
        if m > 1:
            loc = self.mean(locs, scales, weights)
        else:
            loc = locs.squeeze(-1)
        target_expanded = target.unsqueeze(1).expand_as(loc)
        if target_type in ["rfx", "noise"]:
            return mse(loc.log(), target_expanded.log())
        else:
            return mse(loc, target_expanded)

    def nllLoss(self, locs: torch.Tensor, scales: torch.Tensor, weights: torch.Tensor,
                target: torch.Tensor, target_type: str) -> torch.Tensor:
        b, n, d, _ = locs.shape
        target = target.unsqueeze(1).expand((b, n, d))
        proposal = self.construct(locs, scales, weights)
        return -proposal.log_prob(target)

    def loss(self, loss_type: str, locs, scales, weights, target, target_type):
        if loss_type == "mse":
            return self.mseLoss(locs, scales, weights, target, target_type)
        elif loss_type == "nll":
            return self.nllLoss(locs, scales, weights, target, target_type)
        else:
            raise ValueError(f"loss type {loss_type} unknown")


if __name__ == '__main__':
    m = 32
    # bins = equidistantBins(-5., 5., m+1)
    bins = normalBins(3., m+1)
    # bins = halfNormalBins(2., m+1)
    print(bins)
    
    prop = DiscreteProposal(bins)
    targets = torch.tensor([0., 3., 7.]).unsqueeze(0)
    logits = D.Normal(0,1).sample((3,m)).unsqueeze(0)
    mean = prop.mean(logits)
    variance = prop.variance(logits, mean)

