from typing import Tuple, Type, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch import distributions as D
import torch.nn.functional as F
from metabeta.models.normalizingflows.coupling import CouplingFlow
from metabeta.models.normalizingflows.flowmatching import FlowMatching
from metabeta.utils import maskLoss
plt.rcParams.update({'font.size': 16})
mse = nn.MSELoss(reduction='none')

# -----------------------------------------------------------------------------
# helpers
def inverseSoftplus(y: torch.Tensor) -> torch.Tensor:
    return torch.where(
        y == 0.,
        torch.zeros_like(y),
        torch.where(
            y > 20.,
            y,
            torch.log(torch.expm1(y))
        )
    )
        

class GeneralizedProposer(nn.Module):
    # get m linear layers from hidden_size to d_output
    def __init__(self, d_model: int, d_output: int, c: int, m: int = 1):
        super().__init__()
        self.d_output = d_output
        self.m = m # multiplier
        self.c = c # number of bins / components
        layers = [nn.Linear(d_model, d_output) for _ in range(c * m)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        out = [layer(x).unsqueeze(-1) for layer in self.layers]
        out = torch.cat(out, dim=-1)
        if out.dim() == 3:
            b, d = out.shape[:2]
            out = out.reshape(b, d, self.c, -1) # type: ignore
        else:
            b, m, d = out.shape[:3]
            out = out.reshape(b, m, d, self.c, -1) # type: ignore
        return out



# -----------------------------------------------------------------------------
# base class
class Posterior(nn.Module):
    def __init__(self):
        super().__init__()
        self.other = None

    def propose(self, s: torch.Tensor) -> torch.Tensor:
        ''' x (batch, d, emb) -> h (batch, ...) '''
        h = self.prop(s) # type: ignore
        return h

    def mean(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def variance(self, h: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def getLocScale(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        loc = self.mean(h)
        scale = self.variance(h, loc).sqrt()
        return loc, scale
    
    def getCDF(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns x, cdf (batch, d, n_samples)
        raise NotImplementedError

    def getQuantiles(self, h: torch.Tensor, roots: list = [.05, .50, .95]) -> torch.Tensor:
        x, cdf = self.getCDF(h)
        cdf = cdf.contiguous()
        b, d, s, r = *cdf.shape, len(roots)
        roots_ = torch.tensor(roots).view(1, 1, -1).expand((b, d, r)).contiguous()
        indices = torch.searchsorted(cdf, roots_).clamp(max=s-1)
        quantiles = x.gather(dim=-1, index=indices)
        return quantiles

    def logProb(self, s: torch.Tensor, values: torch.Tensor, **kwargs):
        h = self.propose(s)
        return self._logProb(h, values)

    def _logProb(self, h: torch.Tensor, values: torch.Tensor):
        # from hidden proposal representations
        raise NotImplementedError

    def loss(self, h: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        ''' calculate losses for each target dim, mask out padded target dims,
        aggregate over targets and average over batch '''
        raise NotImplementedError

    def forward(self, s: torch.Tensor, targets: torch.Tensor,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.propose(s)
        loss = self.loss(h, targets)
        return loss, h

    def plot(self, h: torch.Tensor, target: torch.Tensor, names: List[str],
             batch_idx: int = 0, **kwargs):
        raise NotImplementedError

# -----------------------------------------------------------------------------
# point posterior
class PointPosterior(Posterior):
    def __init__(self, d_model: int, d_output: int):
        super().__init__()
        self.prop = nn.Linear(d_model, d_output)

    def mean(self, h: torch.Tensor):
        return h

    def variance(self, h: torch.Tensor, mean: torch.Tensor):
        return torch.zeros_like(h)

    def loss(self, h: torch.Tensor, targets: torch.Tensor, **kwargs):
        losses = mse(h, targets) 
        return maskLoss(losses, targets)


# -----------------------------------------------------------------------------
# mixture posterior
class MixturePosterior(Posterior):
    def __init__(self,
                 d_model: int,
                 d_data: int,
                 n_components: int,
                 comp_dist: Type[D.Normal] | Type[D.LogNormal],
                 lamb: float = 0.1,
                 ):
        super().__init__()
        self.comp_dist = comp_dist
        self.prop = GeneralizedProposer(d_model, d_data, n_components, 3)
        self.lamb = lamb

    def locScaleLogit(self, h: torch.Tensor):
        loc = h[..., 0]
        scale = h[..., 1].exp()
        logit = h[..., 2]
        return loc, scale, logit

    def construct(self, h: torch.Tensor) -> D.Distribution:
        # builds a proposal distribution object from summary projection h
        locs, scales, logits = self.locScaleLogit(h)
        m = locs.shape[-1]
        if m > 1:
            mix = D.Categorical(logits=logits)
            comp = self.comp_dist(locs, scales)
            proposal = D.MixtureSameFamily(mix, comp)
        else:
            proposal = self.comp_dist(locs.squeeze(-1), scales.squeeze(-1))
        return proposal

    def mean(self, h: torch.Tensor):
        if self.comp_dist == D.Normal:
            locs, _, logits = self.locScaleLogit(h)
            weights = torch.softmax(logits, -1)
            return (locs * weights).sum(dim=-1)
        else:
            proposal = self.construct(h)
            return proposal.mean

    def variance(self, h: torch.Tensor, mean: torch.Tensor):
        if self.comp_dist == D.Normal:
            locs, scales, logits = self.locScaleLogit(h)
            second_moments = locs.square() + scales.square()
            weights = torch.softmax(logits, -1)
            return (second_moments * weights).sum(dim=-1) - mean.square()
        else:
            proposal = self.construct(h)
            return proposal.variance

    def getCDF(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.comp_dist == D.Normal:
            x = torch.linspace(-10., 10., 1000)
        else:
            x = torch.linspace(1e-5, 10., 1000)
        x_ = x.view(-1, 1, 1)
        proposal = self.construct(h)
        cdf = proposal.cdf(x_).permute(1,2,0)
        x_ = x_.permute(1,2,0).expand_as(cdf)
        return x_, cdf

    def _logProb(self, h: torch.Tensor, values: torch.Tensor, **kwargs):
        proposal = self.construct(h)
        return proposal.log_prob(values)

    def loss(self, h: torch.Tensor, targets: torch.Tensor, **kwargs):
        _, _, log_w = self.locScaleLogit(h)
        entropy = -(log_w.exp() * log_w).sum(-1)
        proposal = self.construct(h)
        log_p = proposal.log_prob(targets + 1e-12)
        losses = -log_p - self.lamb * entropy
        return maskLoss(losses, targets)
    
    def plot(self, h: torch.Tensor | Dict[str, torch.Tensor], target: torch.Tensor,
             names: List[str], batch_idx: int = 0, mcmc: torch.Tensor | None = None, **kwargs):
        
        # join global effects
        if isinstance(h, dict):
            h = torch.cat([h['ffx'], h['sigmas']], dim=1)
        x1 = torch.linspace(-10., 10., steps=1000)
        x2 = torch.linspace(1e-5, 10., steps=1000)
        
        # apply target mask
        mask = (target[batch_idx] != 0.)
        target_ = target[batch_idx][mask]
        h_ = h[batch_idx][mask]
        names_ = names[mask.numpy()]
        mc_samples = mcmc[batch_idx][mask] if mcmc is not None else None
        
        # dims
        d = int(mask.sum())
        w = int(torch.tensor(d).sqrt().ceil())
        _, axs = plt.subplots(figsize=(8 * w, 6 * w), ncols=w, nrows=w)
        axs = axs.flatten()
        for i in range(d):
            # prepare density
            if names_[i][2] == 'b':
                x_i = x1
                prop_i = self.construct(h_[i])
            elif names_[i][2] == 's':
                x_i = x2
                prop_i = self.other.construct(h_[i]) # type: ignore
            else:
                raise NameError
            p_i = prop_i.log_prob(x_i.view(1,1,-1)).exp().squeeze()
            indices = (p_i > 1e-3).nonzero(as_tuple=True)[0]
            first, last = indices[0], indices[-1]
            p_i_ = p_i[first:last+1]
            x_i_ = x_i[first:last+1]
            
            # begin plot
            ax = axs[i]
            ax.set_axisbelow(True)
            ax.grid(True)
            ax.plot(x_i_, p_i_, color='darkgreen')
            if mc_samples is not None:
                sns.histplot(mc_samples[i], kde=True, ax=ax,
                             stat="density", color='darkorange', alpha=0.5)
            ax.axvline(x=target_[i], linestyle='--', 
                       linewidth=2.5, color='black')
            ax.set_xlabel(names_[i], fontsize=20)
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelcolor='w')
        for i in range(d, w*w):
            axs[i].set_visible(False)


# -----------------------------------------------------------------------------
# discrete posterior
def equidistantBins(low: float, high: float, steps: int) -> torch.Tensor:
    return torch.linspace(low, high, steps=steps)

def normalBins(scale: float, steps: int, eps: float = 1e-5) -> torch.Tensor:
    quantiles = torch.linspace(0. + eps, 1. - eps, steps)
    return D.Normal(0, scale).icdf(quantiles)

def halfNormalBins(scale: float, steps: int, eps: float = 1e-5) -> torch.Tensor:
    quantiles = torch.linspace(0. + eps, 1. - eps, steps)
    return D.HalfNormal(scale).icdf(quantiles)

class DiscretePosterior(Posterior):
    def __init__(self,
                 d_model: int,
                 d_data: int,
                 bins: torch.Tensor, # (m,)
                 lamb: float = 0.1
                 ):
        super().__init__()
        self.register_buffer('bins', bins)
        self.register_buffer('widths', bins[1:] - bins[:-1])
        self.register_buffer('centers', self.bins[:-1] + self.widths/2)
        self.n_bins = len(bins) - 1
        self.prop = GeneralizedProposer(d_model, d_data, self.n_bins)
        self.lamb = lamb # entropy regularizer

    def getBindex(self, vals: torch.Tensor) -> torch.Tensor:
        vals = vals.contiguous()
        indices = torch.searchsorted(self.bins, vals) - 1
        indices[vals <= self.bins[0]] = 0
        indices[vals >= self.bins[-1]] = self.n_bins - 1
        return indices

    def propose(self, s: torch.Tensor):
        logits = self.prop(s)[..., 0]
        log_p = torch.log_softmax(logits, -1)
        log_p = log_p - torch.log(self.widths) # adjust for (variable) widths
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        return log_p

    def mean(self, h: torch.Tensor):
        p = h.exp()
        return p @ self.centers

    def mode(self, h: torch.Tensor) -> torch.Tensor:
        p = h.exp()
        idx = p.argmax(-1)
        return self.centers[idx]

    def variance(self, h: torch.Tensor, mean: torch.Tensor):
        p = h.exp()
        squared_diff = (self.centers.view(1, -1) - mean.unsqueeze(-1)).square()
        return torch.sum(squared_diff * p, dim=-1)

    def getCDF(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = h.exp()
        cdf = p.cumsum(dim=-1)
        x = self.centers.view(1,1,-1).expand_as(cdf)
        return x, cdf

    def _logProb(self, h: torch.Tensor, values: torch.Tensor, **kwargs):
        indices = self.getBindex(values)
        return h.gather(-1, indices.unsqueeze(-1)).squeeze(-1)

    def loss(self, h: torch.Tensor, targets: torch.Tensor, **kwargs):
        log_p_ = h # (batch, d, n_bins)
        diffs = log_p_[..., 1:] - log_p_[..., :-1] # (batch, d)
        spikiness = diffs.square().mean(-1) / log_p_.abs().mean(-1)
        log_p = self._logProb(log_p_, targets)
        losses = -log_p + self.lamb * spikiness
        return maskLoss(losses, targets)

    def plot(self, h: torch.Tensor | dict, target: torch.Tensor, names: List[str],
             batch_idx: int = 0, mcmc: torch.Tensor | None = None, **kwargs):
        if isinstance(h, dict):
            h = torch.cat([h['ffx'], h['sigmas']], dim=1)
        # apply target mask
        mask = target[batch_idx] != 0.
        d = int(mask.sum())
        target_ = target[batch_idx][mask]
        log_p_ = h[batch_idx][mask]
        names_ = names[mask.numpy()]
        mc_samples = mcmc[batch_idx][mask] if mcmc is not None else None

        w = int(torch.tensor(d).sqrt().ceil())
        _, axs = plt.subplots(figsize=(8 * w, 6 * w), ncols=w, nrows=w)
        axs = axs.flatten()
        for i in range(d):
            ax = axs[i]
            ax.set_axisbelow(True)
            ax.grid(True)

            # get probabilities and find support
            p_i = log_p_[i].exp()
            indices = (p_i > 1e-3).nonzero(as_tuple=True)[0]
            first, last = indices[0], indices[-1]
            p_i_ = p_i[first:last+1]

            # get x_values and limit to support
            lefts = self.bins[:-1] if names_[i][2] == 'b' else self.other.bins[:-1] # type: ignore
            widths = self.widths if names_[i][2] == 'b' else self.other.widths # type: ignore
            lefts_ = lefts[first:last+1]
            widths_ = widths[first:last+1]

            # subplot
            ax.bar(lefts_, p_i_, width=widths_, align='edge',
                   color='darkgreen',edgecolor='black', alpha=0.8)
            if mc_samples is not None:
                sns.histplot(mc_samples[i], kde=True, ax=ax, # type: ignore
                             color='darkorange', alpha=0.5, stat="density")
            ax.axvline(x=target_[i], linestyle='--',
                       linewidth=2.5, color='black')
            ax.set_xlabel(names_[i], fontsize=20)
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelcolor='w')
        for i in range(d, w*w):
            axs[i].set_visible(False)

# -----------------------------------------------------------------------------
# flow posterior
class FlowPosterior(nn.Module):
    def __init__(self, d_data: int, fx_type: str):
        super().__init__()
        self.d_data = d_data
        self.fx_type = fx_type

    def mean(self,
             samples: torch.Tensor,
             weights: torch.Tensor | None = None):
        if weights is not None:
            s = samples.shape[-1]
            weighted_mean = (samples * weights.unsqueeze(1)).sum(-1) / s
            return weighted_mean
        else:
            return samples.mean(-1)

    def std(self,
            samples: torch.Tensor,
            mean: torch.Tensor,
            weights: torch.Tensor | None = None,
            n_eff: torch.Tensor | None = None):
        if weights is not None:
            s = samples.shape[-1]
            denom = (s - s / n_eff).unsqueeze(-1)
            d_sq = (samples - mean.unsqueeze(-1)).square()
            weighted_d_sq = d_sq * weights.unsqueeze(1)
            weighted_var = weighted_d_sq.sum(-1) / (denom + 1e-6)
            return weighted_var.sqrt()
        else:
            return samples.std(-1)

    def getLocScale(self, h): 
        return self._moments(h)

    def _moments(self, proposed: Dict[str, torch.Tensor] | torch.Tensor):
        if isinstance(proposed, torch.Tensor):
            proposed = {'samples': proposed}
        samples = proposed['samples']
        weights = proposed.get('weights', None)
        n_eff = proposed.get('n_eff', None)
        loc = self.mean(samples, weights)
        scale = self.std(samples, loc, weights, n_eff)
        return loc, scale

    def getCDF(self, proposed: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = proposed['samples']
        x, idx = samples.sort(dim=-1, descending=False)
        if 'weights' in proposed:
            s = samples.shape[-1]
            weights = proposed['weights'].unsqueeze(1).expand(*samples.shape)
            w_sorted =  torch.gather(weights, -1, idx)
            cdf = torch.cumsum(w_sorted, -1) / s
        else:
            cdf = torch.linspace(0, 1, x.shape[-1])
            cdf = cdf.view(1,1,-1).expand_as(x)
        return x, cdf
    
    def getQuantiles(self, h: Dict[str, torch.Tensor], roots: list = [.05, .50, .95]) -> torch.Tensor:
        x, cdf = self.getCDF(h)
        cdf = cdf.contiguous()
        b, d, s, r = *cdf.shape, len(roots)
        roots_ = torch.tensor(roots).view(1, 1, -1).expand((b, d, r)).contiguous()
        indices = torch.searchsorted(cdf, roots_).clamp(max=s-1)
        quantiles = x.gather(dim=-1, index=indices)
        return quantiles

    def logProb(self, summary: torch.Tensor, values: torch.Tensor, mask=None):
        raise NotImplementedError

    def loss(self, summary: torch.Tensor, targets: torch.Tensor, mask=None):
        raise NotImplementedError

    def sample(self, summary: torch.Tensor, mask=None, n: int = 1000, log_prob=False):
        raise NotImplementedError

    def processTargets(self, targets: torch.Tensor):
        d_real = (self.d_data - 1)
        if self.fx_type == 'mfx':
            d_real //= 2 
        subtargets = targets[:, d_real:]
        real = inverseSoftplus(subtargets)
        targets[:, d_real:] = real
        return targets

    def processSamples(self, samples):
        d_real = (self.d_data - 1)
        if self.fx_type == 'mfx':
            d_real //= 2
        subsamples = samples[:, d_real:]
        positive = F.softplus(subsamples)
        samples[:, d_real:] = positive
        return samples

    def forward(self, summary: torch.Tensor, targets: torch.Tensor,
                sample=True, n: int = 100, log_prob=False, constrain=True, **kwargs):
        mask = (targets != 0.).float()
        targets = self.processTargets(targets.clone()) if constrain else targets
        loss = self.loss(summary, targets, mask=mask)
        proposed = dict()
        if sample:
            samples, log_q = self.sample(summary, mask, n, log_prob)
            if samples.dim() == 3:
                samples = samples.permute(0, 2, 1) # (b, d, s)
            elif samples.dim() == 4:
                samples = samples.permute(0, 1, 3, 2) # (b, m, d, s)
            samples = self.processSamples(samples.clone()) if constrain else samples
            proposed = {'samples': samples, 'log_prob': log_q}
        return loss, proposed

    def plot(self,
             proposed: dict[str, torch.Tensor],
             target: torch.Tensor,
             names: List[str],
             batch_idx: int = 0,
             mcmc: torch.Tensor | None = None,
             **kwargs):
        # apply target mask
        mask = (target[batch_idx] != 0.)
        target_ = target[batch_idx][mask]
        samples = proposed['samples']
        samples_ = samples[batch_idx][mask]
        if 'weights' in proposed:
            weights = proposed['weights'].unsqueeze(1).expand_as(samples)
            weights_ = weights[batch_idx][mask]
        else:
            weights_ = None
        mc_samples = mcmc['samples'][batch_idx][mask] if mcmc is not None else None
        names_ = names[mask.numpy()]
        d = int(mask.sum())
        w = int(torch.tensor(d).sqrt().ceil())
        _, axs = plt.subplots(figsize=(8 * w, 6 * w), ncols=w, nrows=w)
        axs = axs.flatten()
        for i in range(d):
            ax = axs[i]
            ax.set_axisbelow(True)
            ax.grid(True)
            weights_i = weights_[i] if weights_ is not None else None
            sns.histplot(x=samples_[i], weights=weights_i,
                         kde=True, ax=ax, bins=30,
                         color='darkgreen', alpha=0.5, stat="density")
            if mc_samples is not None:
                sns.histplot(mc_samples[i],
                             kde=True, ax=ax, bins=30,
                             color='darkorange', alpha=0.5, stat="density")
            
            ax.axvline(x=target_[i], linestyle='--', 
                       linewidth=2.5, color='black')
            ax.set_xlabel(names_[i], fontsize=20)
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelcolor='w')
        for i in range(d, w*w):
            axs[i].set_visible(False)


class CouplingPosterior(FlowPosterior):
    def __init__(self,
                 d_model: int,
                 d_data: int,
                 n_flows: int = 6,
                 permute_mode: str | None = 'shuffle',
                 use_actnorm: bool = True,
                 base_dist: str = 'gaussian',
                 transform: str = 'affine',
                 net_kwargs: dict = {},
                 fx_type: str = 'mfx'
                 ):
        super().__init__(d_data, fx_type)
        if transform in ['affine', 'rq']:
            self.cf = CouplingFlow(
                d_target=d_data,
                d_context=d_model,
                n_flows=n_flows,
                permute_mode=permute_mode,
                use_actnorm=use_actnorm,
                use_linear=(transform=='rq'),
                base_dist=base_dist,
                transform=transform,
                net_kwargs=net_kwargs
                )
        else:
            raise NotImplementedError()

    def logProb(self, summary, values, mask=None):
        return self.cf.log_prob(values, condition=summary, mask=mask)

    def loss(self, summary, targets, mask=None):
        return -self.logProb(summary, targets, mask=mask)
 
    def sample(self, summary: torch.Tensor, mask=None, n: int = 100, log_prob=False):
        with torch.no_grad():
            return self.cf.sample(n, context=summary, mask=mask, log_prob=log_prob)


