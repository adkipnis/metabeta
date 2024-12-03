import torch
from typing import Dict

class Task:
    def __init__(self,
                 n_predictors: int, # without intercept
                 sigma_error: float,
                 data_dist: torch.distributions.Distribution,
                 ):

        # data distribution
        self.n_predictors = n_predictors
        self.data_dist = data_dist

        # beta distribution
        self.beta_dist = torch.distributions.Normal(0, 5)

        # error distribution
        self.sigma_error = sigma_error
        self.noise_dist = torch.distributions.Normal(0, self.sigma_error)

    def sampleBetas(self, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        return self.beta_dist.sample((self.n_predictors+1,)) # type: ignore

    def sampleFeatures(self, n_samples: int, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        x = self.data_dist.sample((n_samples, self.n_predictors)) # type: ignore
        intercept = torch.ones(n_samples, 1)
        return torch.cat([intercept, x], dim=1)

    def sampleNoise(self, n_samples: int, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        return self.noise_dist.sample((n_samples,)) # type: ignore

    def sample(self, n_samples: int, seed: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


    def sample(self, n_samples: int, seed: int) -> Dict[str, torch.Tensor]:
        x = self.sampleFeatures(n_samples, seed)
        y = self.predict(x, seed).unsqueeze(1)
        return {
                "predictors": x,
                "y": y,
                "params": self.weights.unsqueeze(-1), # torch.cat([self.weights, self.sigma])
                "seed": torch.tensor(seed),
                }


