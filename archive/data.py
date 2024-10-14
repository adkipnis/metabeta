import torch
from typing import Tuple

class Task:
    def __init__(self,
                 n_predictors: int = 1,
                 sigma_error: float = 0.1,
                 sigma_beta: float = 0.1,
                 cov_beta: float = 0.0,
                 ):
        self.n_predictors = n_predictors

        # error distribution
        self.sigma_error = sigma_error
        self.noise_dist = torch.distributions.Normal(0, self.sigma_error)

        # beta distribution
        self.sigma_beta = sigma_beta
        self.cov_beta = cov_beta
        self.mean = self.sampleMean()
        self.cov = self.sampleCov(self.sigma_beta, self.cov_beta)
        self.weight_dist = torch.distributions.MultivariateNormal(self.mean, self.cov)

    def sampleMean(self) -> torch.Tensor:
        return torch.randn(self.n_predictors + 1)

    def sampleCov(self, var: float, cov: float) -> torch.Tensor:
        main_diag = torch.eye(self.n_predictors + 1)
        off_diag = torch.ones(self.n_predictors + 1, self.n_predictors + 1) - main_diag
        return var * main_diag + cov * off_diag

    def sampleBeta(self) -> torch.Tensor:
        return self.weight_dist.sample()


class LinearModel:
    def __init__(self,
                 task: Task):
        self.task = task
        self.n_predictors = task.n_predictors
        self.weights = self.task.sampleBeta()
        self.dataDist = torch.distributions.Normal(0, 1) # this should be choosable

    def _data(self, n_samples: int):
        x = self.dataDist.sample((n_samples, self.n_predictors))
        intercept = torch.ones(n_samples, 1)
        return torch.cat([intercept, x], dim=1)

    def _target(self, x: torch.Tensor) -> torch.Tensor:
        # assumes x is a tensor of shape (n_samples, n_predictors + 1)
        # with the first column being 1
        n = x.shape[0]
        error = self.task.noise_dist.sample((n,))
        return x @ self.weights + error

    def dataset(self, n_samples: int) -> torch.Tensor:
        x = self._data(n_samples)
        y = self._target(x)
        return torch.cat([x, y.unsqueeze(1)], dim=1)


def main():
    task = Task(n_predictors=2)
    print(f"Mean: {task.mean},\nCov: {task.cov}\nNoise: {task.sigma_error}")
    model = LinearModel(task)
    print(f"Weights: {model.weights}")
    data = model.dataset(10)
    print(f"Data: {data}")
    
if __name__ == "__main__":
    main()
