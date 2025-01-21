import math
import torch
from typing import Dict, Tuple
from utils import symmetricMatrix2Vector, symmetricMatrixFromVector


class Task:
    def __init__(self,
                 n_predictors: int, # without bias
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
        intercept = torch.ones(n_samples, 1)
        if self.n_predictors == 0:
            return intercept
        x = self.data_dist.sample((n_samples, self.n_predictors)) # type: ignore
        x = self._standardize(x)
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

    def allPosteriorParams(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x (n, d), y (n, 1)
        n, d = x.shape

        # weights posterior
        mus = torch.zeros(n, d)
        sigmas = torch.zeros(n, d, d)

        # noise variance posterior
        alphas = torch.zeros(n)
        betas = torch.zeros(n)

        for i in range(n):
            mus[i], sigmas[i], alphas[i], betas[i] = self.posteriorParams(x[:i+1], y[:i+1])
        return mus, sigmas, alphas, betas

    def sample(self, n_samples: int, seed: int, include_posterior: bool = False) -> Dict[str, torch.Tensor]:
        torch.manual_seed(seed)
        X = self._sampleFeatures(n_samples)
        beta = self._sampleBeta()
        eps = self._sampleNoise(n_samples)
        y = X @ beta + eps
        out = {"X": X, # (n, d)
               "y": y, # (n,)
               "beta": beta, # (d,)
               "sigma_error": torch.tensor(self.sigma_error), # (n,)
               "seed": torch.tensor(seed)}
        if include_posterior:
            mu_n, Sigma_n, a_n, b_n = self.allPosteriorParams(X, y)
            out.update({"mu_n": mu_n, "Sigma_n": Sigma_n, "a_n": a_n, "b_n": b_n})
        return out


class MixedEffects(Task):
    def __init__(self, n_predictors: int, sigma_error: float, data_dist: torch.distributions.Distribution,
                 n_random_effects: int):
        super().__init__(n_predictors, sigma_error, data_dist)
        self.n_random_effects = n_random_effects
        self._initRfxStructure()

    def _initRfxStructure(self, a: float = 3., b: float = 3.) -> None:
        ''' given n_random_effects, draw diagonal elements of S (covariance matrix of random effects) '''
        dist = torch.distributions.inverse_gamma.InverseGamma(a,b)
        q = self.n_random_effects
        self.m = torch.zeros((q,))
        self.S = torch.diag_embed(dist.sample((q,))) # type: ignore
        self.b_dist = torch.distributions.multivariate_normal.MultivariateNormal(self.m, covariance_matrix=self.S)

    def _sampleRandomEffects(self, n_samples: int) -> torch.Tensor:
        return self.b_dist.sample((n_samples,)) # type: ignore

    def sample(self, n_samples: int, seed: int) -> Dict[str, torch.Tensor]:
        torch.manual_seed(seed)
        X = self._sampleFeatures(n_samples)
        beta = self._sampleBeta()
        q = self.n_random_effects
        rfx = self._sampleRandomEffects(n_samples)
        Z = X[:,:q]
        eta = torch.bmm(Z.unsqueeze(1), rfx.unsqueeze(2)).flatten() # eta_i = z_i^T b_i
        eps = self._sampleNoise(n_samples)
        y = X @ beta + eta + eps
        out = {"X": X, # (n, d)
               "y": y, # (n,)
               "beta": beta, # (d,)
               "S": symmetricMatrix2Vector(self.S), # (q, q)
               "sigma_error": torch.tensor(self.sigma_error), # (n,)
               "seed": torch.tensor(seed)}
        return out


def plotExample(beta: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import pandas as pd
    import numpy as np

    def paramDataFrame(means, stds) -> pd.DataFrame:
        values_m = means.flatten()
        values_s = stds.flatten()
        row_indices = np.repeat(np.arange(means.shape[0]), means.shape[1]) + 1
        column_indices = np.tile(np.arange(means.shape[1]), means.shape[0])
        return pd.DataFrame({
            'n' : row_indices,
            'mean' : values_m,
            'std' : values_s,
            'd': column_indices
        })
    
    def plotParams(df, targets):
        cmap = colors.LinearSegmentedColormap.from_list("custom_blues", ["#add8e6", "#000080"])
        unique_d = df['d'].unique()
        norm = colors.Normalize(vmin=unique_d.min(), vmax=unique_d.max())
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for d_value, group in df.groupby('d'):
            target = targets[d_value].item()
            color = cmap(norm(d_value))
            ax.plot(group['n'], group['mean'], label=f'd={d_value}', color=color)
            ax.fill_between(group['n'], 
                            group['mean'] - group['std'], 
                            group['mean'] + group['std'], 
                            color=color, alpha=0.1)  # Shade ± SD
            ax.axhline(y=target, color=color, linestyle=':', linewidth=1.5)
            
        # Adding labels and title
        plt.xlabel('n')  # X-axis label
        plt.ylabel(f'analytical posterior')      # Y-axis label
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.ylim(-6, 6)
        plt.grid(True)           # Show grid
        plt.show()               # Display the plot
        
    df = paramDataFrame(mu, torch.diagonal(sigma, dim1=-1, dim2=-2).sqrt())
    plotParams(df, beta)


if __name__ == "__main__":
    seed = 1
    n_predictors = 3
    n_obs = 20
    noise_var = 0.5 ** 2
    datadist = torch.distributions.uniform.Uniform(0., 1.)

    # # fixed effects
    # fe = FixedEffects(n_predictors, math.sqrt(noise_var), datadist)
    # ds = fe.sample(n_obs, seed)
    # X, y, beta = ds["X"], ds["y"], ds["beta"]
    # print(f"true beta: {beta}")

    # # analytical posterior
    # mu, sigma, a, b = fe.allPosteriorParams(X, y)
    # print(f"posterior mu:\n{mu}")
    # print(f"posterior (margial) sigma^2:\n{torch.diagonal(sigma, dim1=-1, dim2=-2)}")
    # print(f"posterior a:\n{a}")
    # print(f"posterior b:\n{b}")
    # plotExample(beta, mu, sigma)

    # # noise variance
    # eps = y - X @ beta
    # noise_var_ml = 1/(n_obs - n_predictors - 1) * torch.dot(eps, eps)
    # expected_noise_var = torch.distributions.inverse_gamma.InverseGamma(a[-1], b[-1]).mean
    # print(f"true error variance: {noise_var:.3f}")
    # print(f"ML estimate: {noise_var_ml:.3f}")
    # print(f"EAP estimate: {expected_noise_var:.3f}")
    # snr = torch.var(y)/noise_var
    # print(f"SNR: {snr:.3f}")

    # mixed effects
    n_random_effects = 3
    me = MixedEffects(n_predictors, math.sqrt(noise_var), datadist, n_random_effects)
    ds = me.sample(n_obs, seed)
    y, S = ds["y"], symmetricMatrixFromVector(ds["S"], n_random_effects)

    print(f"random effects covariance:\n{S}")
    print(f"targets:\n{y}")



