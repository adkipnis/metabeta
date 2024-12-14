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
        self.beta_error = math.sqrt(5.)
        self.beta_dist = torch.distributions.Normal(0., self.beta_error)

        # error distribution
        self.sigma_error = sigma_error
        self.noise_dist = torch.distributions.Normal(0., self.sigma_error)
        
    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=0)
        sd = torch.std(x, dim=0)
        return (x-mean)/(sd + 1e-8)

    def _sampleBeta(self) -> torch.Tensor:
        d = self.n_predictors + 1
        return self.beta_dist.sample((d,)) # type: ignore

    def _sampleFeatures(self, n_samples: int) -> torch.Tensor:
        x = self.data_dist.sample((n_samples, self.n_predictors)) # type: ignore
        x = self._standardize(x)
        intercept = torch.ones(n_samples, 1)
        return torch.cat([intercept, x], dim=1)

    def _sampleNoise(self, n_samples: int) -> torch.Tensor:
        return self.noise_dist.sample((n_samples,)) # type: ignore

    def sample(self, n_samples: int, seed: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    
class FixedEffects(Task):
    def __init__(self, n_predictors: int, sigma_error: float, data_dist: torch.distributions.Distribution):
        super().__init__(n_predictors, sigma_error, data_dist)

    def _priorPrecision(self) -> torch.Tensor:
        d = self.n_predictors + 1
        precision = torch.tensor(1. / self.beta_error).square().repeat(d)
        L_0 = torch.diag(precision)
        return L_0

    def _posteriorPrecision(self, x: torch.Tensor) -> torch.Tensor:
        L_0 = self._priorPrecision()
        S = x.T @ x
        L_n = L_0 + S
        return L_n

    def _posteriorCovariance(self, L_n: torch.Tensor) -> torch.Tensor:
        lower = torch.linalg.cholesky(L_n)
        S_n = torch.cholesky_inverse(lower)
        return S_n

    def _posteriorMean(self, x: torch.Tensor, y: torch.Tensor, S_n: torch.Tensor) -> torch.Tensor:
        # simplified form under zero prior mean
        mu_n = S_n @ (x.T @ y)
        return mu_n

    def _posteriorA(self, x: torch.Tensor) -> torch.Tensor:
        a_0 = 3.
        n = x.shape[0]
        a_n = torch.tensor(a_0 + n / 2.)
        return a_n

    def _posteriorB(self, y: torch.Tensor, mu_n: torch.Tensor, L_n: torch.Tensor) -> torch.Tensor:
        b_0 = 1.
        y_inner = torch.dot(y, y)
        mu_n_inner_scaled = torch.linalg.multi_dot([mu_n, L_n, mu_n])
        b_n = b_0 + (y_inner - mu_n_inner_scaled) / 2.
        return b_n

    def posteriorParams(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        L_n = self._posteriorPrecision(x)
        S_n = self._posteriorCovariance(L_n)
        mu_n = self._posteriorMean(x, y, S_n)
        a_n = self._posteriorA(x)
        b_n = self._posteriorB(y, mu_n, L_n)
        return mu_n, S_n, a_n, b_n

    def allPosteriorParams(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n, d = x.shape
        mus = torch.zeros(n, d)
        sigmas = torch.zeros(n, d, d)
        for i in range(n):
            mus[i], sigmas[i] = self.posteriorParams(x[:i+1], y[:i+1])
        return mus, sigmas

    def sample(self, n_samples: int, seed: int, include_posterior: bool = False) -> Dict[str, torch.Tensor]:
        torch.manual_seed(seed)
        X = self._sampleFeatures(n_samples)
        beta = self._sampleBeta()
        eps = self._sampleNoise(n_samples)
        y = X @ beta + eps
        out = {"X": X, # (n, d)
               "y": y, # (n,)
               "beta": beta, # (d,)
               "sigma_error": torch.tensor(self.sigma_error),
               "seed": torch.tensor(seed),}
        if include_posterior:
            mu_n, Sigma_n = self.allPosteriorParams(X, y)
            out.update({"mu_n": mu_n,
                        "Sigma_n": Sigma_n,})
        return out


def main():
    seed = 0
    n_predictors = 2
    n_obs = 20
    noise_std = 0.1
    datadist = torch.distributions.Normal(0., 3.)
    fe = FixedEffects(n_predictors, noise_std, datadist)
    ds = fe.sample(n_obs, seed)
    X, y = ds["X"], ds["y"]
    # mu_n, Sigma_n = fe.posteriorParams(X, y)
    # print(f"true:\n{ds['params']}")
    # print(f"post. mean:\n{mu_n}")
    # print(f"post. cov:\n{Sigma_n}")
    mus, sigmas = fe.allPosteriorParams(X, y)
    print(mus)

if __name__ == "__main__":
    main()

