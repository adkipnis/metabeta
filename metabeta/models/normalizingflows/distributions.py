import numpy as np
from scipy.stats import norm, t
import torch
from torch import nn
from torch import distributions as D

# TODO introduce seeding to scipy

class DiagDist(nn.Module):
    def __init__(self, base: D.Distribution, dim: int, trainable: bool = True):
        super().__init__()
        self.dim = dim
        self.trainable = trainable
        if trainable:
            self.initParams()
            self.base = base
        else:
            default_params = self.getParams()
            self.base = base(**default_params) # type: ignore

    def initParams(self) -> None:
        raise NotImplementedError

    def getParams(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def dist(self) -> D.Distribution:
        if self.trainable:
            params = self.getParams()
            dist = self.base(**params) # type: ignore
        else:
            dist = self.base
        return dist

    def log_prob(self, z: torch.Tensor):
        dist = self.dist()
        log_prob = dist.log_prob(z)
        return log_prob

    # def sample(self, shape: tuple):
    #     dist = self.sampleDist()
    #     if self.trainable:
    #         shape = shape[:-1]
    #     sample = dist.sample(shape)
    #     return sample
    
    def sample(self, shape: tuple) -> torch.Tensor:
        with torch.no_grad():
            params = self.getParams()
            device = params['loc'].device
            params = {k: v.cpu().numpy() for k,v in params.items()}
        x = self.sample_dist(**params, size=shape).astype(np.float32)
        x = torch.from_numpy(x).to(device)
        return x


class DiagGaussian(DiagDist):
    def __init__(self, dim):
        super().__init__(D.Normal, dim)
        self.sample_dist = norm.rvs

    def getParams(self):
        if self.trainable:
            scale = nn.functional.softplus(self.log_scale + 1e-3)
            return {"loc": self.loc, "scale": scale}
        return {"loc": 0, "scale": 1}

    def initParams(self):
        self.loc = nn.Parameter(torch.ones(self.dim) * 0)
        self.log_scale = nn.Parameter(torch.log(torch.exp(torch.ones(self.dim) * 1) - 1))
    


class DiagStudent(DiagDist):
    def __init__(self, dim):
        super().__init__(D.StudentT, dim)
        self.sample_dist = t.rvs

    def getParams(self):
        if self.trainable:
            df = nn.functional.softplus(self.log_df + 1e-3)
            scale = nn.functional.softplus(self.log_scale + 1e-3)
            return {"df": df, "loc": self.loc, "scale": scale}
        return {"df": 5, "loc": 0, "scale": 1}

    def initParams(self):
        self.log_df = nn.Parameter(torch.log(torch.exp(torch.ones(self.dim) * 5) - 1))
        self.loc = nn.Parameter(torch.ones(self.dim) * 0)
        self.log_scale = nn.Parameter(torch.log(torch.exp(torch.ones(self.dim) * 1) - 1))
    