import math
import torch
from typing import Dict, Tuple

class Task:
    def __init__(self,
                 n_predictors: int, # without intercept
                 sigma_error: float, # standard deviation of the additive noise
                 data_dist: torch.distributions.Distribution,
                 ):

        # data distribution
        self.n_predictors = n_predictors
        self.data_dist = data_dist

        # beta distribution
        self.beta_error = math.sqrt(5)
        self.beta_dist = torch.distributions.Normal(0., self.beta_error)

        # error distribution
        self.sigma_error = sigma_error
        self.noise_dist = torch.distributions.Normal(0., self.sigma_error)
        
    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=0)
        sd = torch.std(x, dim=0)
        return (x-mean)/(sd + 1e-8)

    def _sampleBeta(self, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        d = self.n_predictors + 1
        return self.beta_dist.sample((d, 1)) # type: ignore

    def _sampleFeatures(self, n_samples: int, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        x = self.data_dist.sample((n_samples, self.n_predictors)) # type: ignore
        x = self._standardize(x)
        intercept = torch.ones(n_samples, 1)
        return torch.cat([intercept, x], dim=1)

    def _sampleNoise(self, n_samples: int, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        return self.noise_dist.sample((n_samples, 1)) # type: ignore

    def sample(self, n_samples: int, seed: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _posteriorPrecision(self, x: torch.Tensor) -> torch.Tensor:
        d = self.n_predictors + 1
        precision = torch.tensor(1. / self.beta_error).square().repeat(d)
        L_0 = torch.diag(precision)
        S = x.T @ x
        return L_0 + S

class FixedEffects(Task):
    def __init__(self, n_predictors: int, sigma_error: float, data_dist: torch.distributions.Distribution):
        super().__init__(n_predictors, sigma_error, data_dist)

    def sample(self, n_samples: int, seed: int) -> Dict[str, torch.Tensor]:
        x = self._sampleFeatures(n_samples, seed)
        betas = self._sampleBetas(seed)
        eps = self._sampleNoise(n_samples, seed)
        y = x @ betas + eps
        return {
                "predictors": x,
                "y": y,
                "params": betas,
                "sigma_error": torch.tensor(self.sigma_error),
                "seed": torch.tensor(seed),
                }
    

