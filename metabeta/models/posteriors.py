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


