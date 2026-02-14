import numpy as np
import scipy
import torch
from torch import nn
from torch import distributions as D
from torch.nn import functional as F


class StaticDist(nn.Module):
    def __init__(self, d_data: int, family: str, seed: int = 1) -> None:
        super().__init__()
        assert family in ['normal', 'student'], 'distribution family unknown'
        self.d_data = d_data
        self.family = family
        self._sampling_dist = self._dist['scipy'](**self._default_params)
        self._training_dist = self._dist['torch'](**self._default_params)
        self.rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        params = ''
        for k, v in self._default_params.items():
            params += f'{k}: {v:.1f}, '
        return f'Static {self.family.title()} ({params[:-2]})'

    @property
    def _default_params(self) -> dict[str, float]:
        out = {}
        if self.family == 'student':
            out['df'] = 5.0
        out['loc'] = 0.0
        out['scale'] = 1.0
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
        x = self._sampling_dist.rvs(size=shape, random_state=self.rng)
        return torch.from_numpy(x)

    def logProb(self, x: torch.Tensor) -> torch.Tensor:
        return self._training_dist.log_prob(x)


class TrainableDist(nn.Module):
    def __init__(self, d_data: int, family: str, seed: int = 1) -> None:
        super().__init__()
        self.d_data = d_data
        self.family = family
        self.rng = np.random.default_rng(seed)
        self._initParams()

    def __repr__(self) -> str:
        params = ''
        for k, v in self._params.items():
            params += f'\n    {k}: {v.cpu().detach().numpy()}, '
        return f'Trainable {self.family.title()} ({params[:-2]})'

    def _initParams(self) -> None:
        if self.family == 'student':
            self._log_df = nn.Parameter(torch.log(torch.exp(6 * torch.ones(self.d_data)) - 1))
        self._loc = nn.Parameter(torch.zeros(self.d_data))
        self._log_scale = nn.Parameter(torch.log(torch.exp(torch.ones(self.d_data)) - 1))

    @property
    def _params(self) -> dict[str, torch.Tensor]:
        out = {}
        if self.family == 'student':
            out['df'] = F.softplus(self._log_df + 1e-6) + 3.0   # clamps df at min=3
        out['loc'] = self._loc * 1.0
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
        if shape[-1] != len(self._params['loc']):
            raise ValueError(
                f'last shape dim should be {len(self._params["loc"])}, but found {shape[-1]}'
            )
        with torch.no_grad():
            params = {k: v.cpu() for k, v in self._params.items()}
            sampling_dist = self._dist['scipy'](**params)
            x = sampling_dist.rvs(size=shape, random_state=self.rng)
            return torch.from_numpy(x)

    def logProb(self, x: torch.Tensor) -> torch.Tensor:
        training_dist = self._dist['torch'](**self._params)
        return training_dist.log_prob(x)


class BaseDist(nn.Module):
    """Wrapper for Normalizing Flow base distribution
    - supports Normal and StudentT distribution family
    - allows trainable distribution parameters
    - log_prob evaluation via pytorch
    - CPU sampling via scipy
      (circumvents issues with torch sampling in model.eval() when not on CPU)
    """

    def __init__(
        self,
        d_data: int,
        family: str,
        trainable: bool = True,
        seed: int = 1,
    ) -> None:
        super().__init__()
        self.trainable = trainable
        if trainable:
            self.base = TrainableDist(d_data, family, seed)
        else:
            self.base = StaticDist(d_data, family, seed)

    def __repr__(self) -> str:
        return self.base.__repr__()

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        return self.base.sample(shape)

    def logProb(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.logProb(x)


if __name__ == '__main__':
    b, d = 8, 3
    dist = BaseDist(d, family='student', trainable=True)
    x = dist.sample((b, d))
    log_prob = dist.logProb(x)
