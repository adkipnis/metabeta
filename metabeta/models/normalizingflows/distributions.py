import torch
from torch import nn
from torch import distributions as D

class DiagDist(nn.Module):
    def __init__(self, base: D.Distribution = D.Normal(0,1)):
        super().__init__()
        self.base = base

    def sample(self, shape: tuple):
        return self.base.sample(shape)

    def log_prob(self, z: torch.Tensor):
        mask = ~torch.isnan(z)
        log_prob = torch.empty_like(z)
        log_prob[mask] = self.base.log_prob(z[mask])
        log_prob[~mask] = float('nan')
        return log_prob

class DiagGaussian(DiagDist):
    def __init__(self):
        super().__init__(D.Normal(loc=0, scale=1))

class DiagStudent(DiagDist):
    def __init__(self):
        super().__init__(D.StudentT(df=4, loc=0, scale=1))

class DiagUniform(DiagDist):
    def __init__(self):
        super().__init__(D.Uniform(low=0, high=1))
