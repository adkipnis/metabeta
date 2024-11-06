import torch
from typing import Dict

class Task:
    def __init__(self,
                 n_predictors: int,
                 seed: int,
                 sigma_error: float = 0.1,
                 sigma_beta: float = 0.1,
                 cov_beta: float = 0.0,
                 ):
        self.n_predictors = n_predictors
        self.seed = seed

        # error distribution
        self.sigma_error = sigma_error
        self.noise_dist = torch.distributions.Normal(0, self.sigma_error)

        # beta distribution
        self.sigma_beta = sigma_beta
        self.cov_beta = cov_beta
        self.mean = self.sampleMean()
        self.cov = self.getCov(self.sigma_beta, self.cov_beta)
        self.weight_dist = torch.distributions.MultivariateNormal(self.mean, self.cov)

    def sampleMean(self) -> torch.Tensor:
        torch.manual_seed(self.seed)
        return torch.randn(self.n_predictors + 1)

    def getCov(self, var: float, cov: float) -> torch.Tensor:
        main_diag = torch.eye(self.n_predictors + 1)
        off_diag = torch.ones(self.n_predictors + 1, self.n_predictors + 1) - main_diag
        return var * main_diag + cov * off_diag

    def sampleBeta(self) -> torch.Tensor:
        torch.manual_seed(self.seed)
        return self.weight_dist.sample()


class LinearModel:
    def __init__(self,
                 task: Task,
                 dataDist: torch.distributions.Distribution) -> None:
        self.task = task
        self.dataDist = dataDist
        self.weights = self.task.sampleBeta()
        self.sigma = torch.tensor([self.task.sigma_error])

    @property
    def n_predictors(self):
        return self.task.n_predictors

    def sampleFeatures(self, n_samples: int, seed: int):
        torch.manual_seed(seed)
        x = self.dataDist.sample((n_samples, self.n_predictors)) # type: ignore
        intercept = torch.ones(n_samples, 1)
        return torch.cat([intercept, x], dim=1)

    def predict(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        # x: (n_samples, n_predictors + 1)
        torch.manual_seed(seed)
        n_samples = x.shape[0]
        error = self.task.noise_dist.sample((n_samples,)) # type: ignore
        return x @ self.weights + error

    def sample(self, n_samples: int, seed: int) -> Dict[str, torch.Tensor]:
        x = self.sampleFeatures(n_samples, seed)
        y = self.predict(x, seed).unsqueeze(1)
        return {
                "predictors": x,
                "y": y,
                "params": self.weights # torch.cat([self.weights, self.sigma])
                }


def main():
    seed = 0
    task = Task(n_predictors=1, seed=seed)
    print(f"Mean: {task.mean},\nCov: {task.cov}\nNoise: {task.sigma_error}")
    model = LinearModel(task)
    sample = model.sample(10, seed=seed)
    print(f"Features: {sample['features']}")
    print(f"Target: {sample['target']}")
    print(f"Params: {sample['params']}")
    
if __name__ == "__main__":
    main()
