import torch
from torch import nn
from torch import distributions as D

# --- probabilistic activations
class Bernoulli(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bernoulli(torch.sigmoid(x))

class Poisson(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.poisson(torch.where(x > 0, x**2, 0))

class Geometric(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(x) + 1e-9
        return D.Geometric(probs=p).sample().sqrt()

class Gamma(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return D.Gamma(2, x.exp()).sample().sqrt()

class Beta(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.sigmoid(x + 1e-9)
        alpha = m * 10
        beta = (1-m) * 10
        return D.Beta(alpha, beta).sample()


