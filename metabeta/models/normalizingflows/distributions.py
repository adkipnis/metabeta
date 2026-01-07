import numpy as np
import scipy
import torch
from torch import nn
from torch import distributions as D
from torch.nn import functional as F


class StaticDist(nn.Module):
    def __init__(self, d_data: int, family: str) -> None:
        super().__init__()
        assert family in ['normal', 'student'], 'distribution family unknown'
        self.d_data = d_data
        self.family = family
        self._sampling_dist = self._dist['scipy'](**self._default_params)
        self._training_dist = self._dist['torch'](**self._default_params)
        self.device = self._training_dist.mean.device

    def __repr__(self) -> str:
        params = ''
        for k,v in self._default_params.items():
            params += f'{k}: {v:.1f}, '
        return f'Static {self.family.title()} ({params[:-2]})'

    @property
    def _default_params(self) -> dict[str, float]:
        out = {}
        if self.family == 'student':
            out['df'] = 5.
        out['loc'] = 0.
        out['scale'] = 1.
        return out

    @property
    def _dist(self) -> dict:
        if self.family == 'normal':
            return {'torch': D.Normal, 'scipy': scipy.stats.norm} 
        elif self.family == 'student':
            return {'torch': D.StudentT, 'scipy': scipy.stats.t}
        else:
            raise ValueError

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        x = self._sampling_dist.rvs(size=shape).astype(np.float32)
        return torch.from_numpy(x).to(self.device)

    def logProb(self, x: torch.Tensor) -> torch.Tensor:
        return self._training_dist.log_prob(x)


class TrainableDist(nn.Module):
    def __init__(self, d_data: int, family: str) -> None:
        super().__init__()
        self.d_data = d_data
        self.family = family
        self._initParams()
        self.device = self._params['loc'].device

    def __repr__(self) -> str:
        params = ''
        for k,v in self._params.items():
            params += f'\n    {k}: {v.cpu().detach().numpy()}, '
        return f'Trainable {self.family.title()} ({params[:-2]})'
 
    def _initParams(self) -> None:
        if self.family == 'student':
            self._log_df = nn.Parameter(torch.log(torch.exp(5 * torch.ones(self.d_data)) - 1))
        self._loc = nn.Parameter(torch.zeros(self.d_data))
        self._log_scale = nn.Parameter(torch.log(torch.exp(torch.ones(self.d_data)) - 1))

    @property
    def _params(self) -> dict[str, torch.Tensor]:
        out = {}
        if self.family == 'student':
            out['df'] = F.softplus(self._log_df + 1e-6)
        out['loc'] = self._loc * 1.
        out['scale'] = F.softplus(self._log_scale + 1e-6)
        return out

    @property
    def _dist(self) -> dict:
        if self.family == 'normal':
            return {'torch': D.Normal, 'scipy': scipy.stats.norm} 
        elif self.family == 'student':
            return {'torch': D.StudentT, 'scipy': scipy.stats.t}
        else:
            raise ValueError

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        assert shape[-1] == len(self._params['loc']), 'shape mismatch'
        with torch.no_grad():
            sampling_dist = self._dist['scipy'](**self._params)
            x = sampling_dist.rvs(size=shape).astype(np.float32)
            return torch.from_numpy(x).to(self.device)

    def logProb(self, x: torch.Tensor) -> torch.Tensor:
        training_dist = self._dist['torch'](**self._params)
        return training_dist.log_prob(x)


