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


