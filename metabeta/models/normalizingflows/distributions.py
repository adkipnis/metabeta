import torch
from torch import nn
from torch import distributions as D


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

    def sample(self, shape: tuple):
        dist = self.dist()
        if self.trainable:
            shape = shape[:-1]
        sample = dist.sample(shape)
        return sample


class DiagGaussian(DiagDist):
    def __init__(self, dim):
        super().__init__(D.Normal, dim)

    def getParams(self):
        if self.trainable:
            return {"loc": self.loc, "scale": self.scale}
        return {"loc": 0, "scale": 1}

    def initParams(self):
        self.loc = nn.Parameter(torch.ones(self.dim) * 0)
        self.scale = nn.Parameter(torch.ones(self.dim) * 1)


