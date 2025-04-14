import math
import torch
from torch import distributions as D
from typing import Dict, Tuple, Union
from utils import symmetricMatrix2Vector, symmetricMatrixFromVector
from alternatives import evalVI, fitFfxVI, fitMfxVI


def covarySeries(series: torch.Tensor, correction: int = 1) -> torch.Tensor:
    ''' Calculate sample variance for each subset of series (as n increases) '''
    # series (n, q)
    n = series.shape[0]
    k = torch.arange(1, n+1).unsqueeze(1)
    means = torch.cumsum(series, dim=0) / k
    outer = means * means
    inner = torch.cumsum(series**2, dim=0) / k
    bessel = k / (k - correction) # unbiased variance estimate
    bessel[0] = 1. # prevent NaN generation due to division by inf
    variances = bessel * (inner - outer)
    # test:
    # i = 9
    # torch.isclose(variances[i], torch.var(series[:i+1], dim=0, correction = correction))
    return torch.max(torch.tensor(0), variances) # guarantee non-negative values


class Task:
    def __init__(self,
                 sigma_error: Union[float, torch.Tensor], # standard deviation of the additive noise
                 n_ffx: int, # without bias
                 ):

        # data distribution
        self.d = n_ffx
        self.dist_data = D.Uniform(0., 1.)

        # ffx distribution
        self.sigma_ffx = 3.
        self.dist_ffx = D.Normal(0., self.sigma_ffx)

        # noise distribution
        self.sigma_error = sigma_error
        self.dist_error = D.Normal(0., sigma_error)
 
    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=0)
        sd = torch.std(x, dim=0)
        return (x-mean)/(sd + 1e-8)

    def _sampleFfx(self) -> torch.Tensor:
        d = self.d + 1
        return self.dist_ffx.sample((d,)) # type: ignore

    def _sampleFeatures(self, n_samples: int) -> torch.Tensor:
        intercept = torch.ones(n_samples, 1)
        if self.d == 0:
            return intercept
        x = self.dist_data.sample((n_samples, self.d)) # type: ignore
        x = self._standardize(x)
        return torch.cat([intercept, x], dim=1)

    def _sampleError(self, n_samples: int) -> torch.Tensor:
        return self.dist_error.sample((n_samples,)) # type: ignore

    def sample(self, n_samples: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def signalToNoiseRatio(self, y: torch.Tensor, eta: torch.Tensor):
        eps = y - eta
        ess = torch.sum((eta - torch.mean(eta)).square())
        rss = torch.sum(eps.square())
        snr = ess / rss
        return snr



class FixedEffects(Task):
    def __init__(self,
                 sigma_error: Union[float, torch.Tensor],
                 n_ffx: int,
                 **kwargs
                 ):
        super().__init__(sigma_error, n_ffx)

    def sample(self, n_samples: int, include_posterior: bool = False) -> Dict[str, torch.Tensor]:
        X = self._sampleFeatures(n_samples)
        ffx = self._sampleFfx()
        eps = self._sampleError(n_samples)
        sigma_error_emp = covarySeries(eps.unsqueeze(-1)).sqrt().squeeze()
        eta = X @ ffx
        y = eta + eps
        snr = self.signalToNoiseRatio(y, eta)
        out = {"X": X, # (n, d)
               "y": y, # (n,)
               "ffx": ffx, # (d,)
               "sigma_error": self.sigma_error, # (1,)
               "sigma_error_emp": sigma_error_emp, # (n,)
               "snr": snr, # (1,)
               "n": torch.tensor(n_samples), # (1,)
               "d": torch.tensor(self.d), # (1,)
               }
        if include_posterior:
            mu, Sigma, alpha, beta = self.posteriorParams(X, y)
            ffx_stats = {"mu": mu, "Sigma": Sigma}
            noise_stats = {"alpha": alpha, "beta": beta}
            out.update({"optimal": {"ffx": ffx_stats, "noise": noise_stats}})
        return out

    # ----------------------------------------------------------------
    # analytical solution assuming Normal-IG-prior

    def _priorPrecision(self) -> torch.Tensor:
        d = self.d + 1
        precision = torch.tensor(1. / self.sigma_ffx).square().repeat(d)
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
        L_i = self._posteriorPrecision(x)
        S_i = self._posteriorCovariance(L_i)
        mu_i = self._posteriorMean(x, y, S_i)
        a_i = self._posteriorA(x)
        b_i = self._posteriorB(y, mu_i, L_i)
        return mu_i, S_i, a_i, b_i

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



class MixedEffects(Task):
    def __init__(self,
                 sigma_error: Union[float, torch.Tensor],
                 sigmas_rfx: torch.Tensor,
                 n_ffx: int,
                 n_rfx: int,
                 n_groups: int,
                 ):
        super().__init__(sigma_error, n_ffx)
        self.q = n_rfx 
        self.m = n_groups 
        self.S = torch.diag_embed(sigmas_rfx[:self.q]).square() # type: ignore
        self._initRfxStructure()

    def _initRfxStructure(self) -> None:
        ''' given q, draw diagonal elements of S (covariance matrix of random effects) '''
        loc = torch.zeros((self.q,))
        self.dist_rfx = D.MultivariateNormal(loc, covariance_matrix=self.S)

    def _sampleRfx(self) -> torch.Tensor:
        return self.dist_rfx.sample((self.m,)) # type: ignore

        # fixed effects and noise
        beta = self._sampleBeta() # (d,)
        X = self._sampleFeatures(n_samples)
        eps = self._sampleNoise(n_samples)
        
        # random effects and target
        groups = torch.arange(0, self.m).repeat(n_samples//self.m) # (n,)
        b = self._sampleRandomEffects() # (m, q)
        B = b[groups] # (n, q)
        Z = X[:,:self.q]
        y = X @ beta + (Z * B).sum(dim=-1) + eps

        # empricial covariance
        S_emp = self._covarySeries(B)
        sigma_error_emp = self._covarySeries(eps.unsqueeze(-1)).sqrt()

        # outputs
        out = {"X": X, # (n, d)
               "y": y, # (n,)
               "groups": groups, # (n,)
               "ffx": beta, # (d,)
               "rfx": B, # (n, q)
               "S": torch.diag(self.S), # once we allow correlation: symmetricMatrix2Vector(self.S),
               "S_emp": S_emp, # for now only marginal variances
               "sigma_error": torch.tensor(self.sigma_error), # (1,)
               "sigma_error_emp": sigma_error_emp, # (n,)
               "seed": torch.tensor(seed)}
        if include_posterior:
            approx = self.fitVI(y, X, groups)
            ffx_vp = self.evalVI(approx, "beta") # fixed effects variational posterior
            rfx_vp = self.evalVI(approx, "sigma_b")
            noise_vp = self.evalVI(approx, "sigma_e")
            out.update({"ffx_vp": ffx_vp, "rfx_vp": rfx_vp, "noise_vp": noise_vp})
        return out
    
    def fitVI(self, y: torch.Tensor, X: torch.Tensor, groups: torch.Tensor) -> pm.variational.approximations.MeanField:
        ''' perform variational inference with automatic diffenentiation '''
        d = X.shape[1]
        Z = X[:,:self.q]
        with pm.Model() as model:
            beta = pm.Normal("beta", mu=0., sigma=5., shape=d) # priors
            sigma_b = pm.HalfNormal("sigma_b", sigma=1., shape=self.q) # rfx SD
            b = pm.Normal("b", mu=0., sigma=sigma_b, shape=(self.m, self.q)) # rfx
            B = b[groups.numpy()]
            mu = pt.dot(X.numpy(), beta) + pt.sum(Z.numpy() * B, axis=1) # linear predictor
            sigma_e = pm.HalfNormal("sigma_e", sigma=1.)  # noise SD
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_e, observed=y.numpy())
            approx = pm.fit(method="advi", n=10000) 
        return approx
    
    # def allVariationalPosteriors(self, y: torch.Tensor, X: torch.tensor,
    #                              groups: torch.Tensor) -> Dict[str, torch.Tensor]:
    #     n, d = X.shape
    #     ffx_quantiles   = torch.zeros(n, d, 3)
    #     rfx_quantiles   = torch.zeros(n, self.q, 3)
    #     noise_quantiles = torch.zeros(n, 3)
        
    #     for i in range(n):
    #         approx = me.fitVI(y[:i+1], X[:i+1], groups[:i+1])
    #         ffx_quantiles[i]   = me.evalVI(approx, "beta")["ci95"].T
    #         rfx_quantiles[i]   = me.evalVI(approx, "sigma_b")["ci95"].T
    #         noise_quantiles[i] = me.evalVI(approx, "sigma_e")["ci95"]
    #     return {"ffx_vp":   ffx_quantiles,
    #             "rfx_vp":   rfx_quantiles,
    #             "noise_vp": noise_quantiles}


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
    n_predictors = 5
    n_obs = 50
    noise_var = 0.5 ** 2
    datadist = D.Uniform(0., 1.)

    # # --- fixed effects
    # fe = FixedEffects(n_predictors, math.sqrt(noise_var), datadist)
    # ds = fe.sample(n_obs, seed)
    # X, y, beta, noise_var_emp = ds["X"], ds["y"], ds["beta"], ds["sigma_error_emp"].square()
    # print(f"true beta: {beta}")
    # print(f"true noise variance: {noise_var}")
    # print(f"empirical noise variance: {noise_var_emp}")
    
    # # analytical posterior
    # mu, sigma, a, b = fe.allPosteriorParams(y, X)
    # print(f"posterior mu: {mu[-1]}")
    # print(f"posterior (margial) sigma^2: {torch.diagonal(sigma, dim1=-1, dim2=-2)[-1]}")
    # # print(f"posterior a:\n{a}")
    # # print(f"posterior b:\n{b}")
    # plotExample(beta, mu, sigma)
    
    # # VI posterior
    # approx = fe.fitVI(y, X)
    # vi_stats = fe.evalVI(approx, "beta")
    # mu_vi, sigma_vi = vi_stats["means"], vi_stats["stds"]
    # print(f"VI posterior mu: {mu_vi}")
    # print(f"VI posterior sigma^2: {sigma_vi.square()}")

    # # noise variance
    # eps = y - X @ beta
    # noise_var_ml = 1/(n_obs - n_predictors - 1) * torch.dot(eps, eps)
    # expected_noise_var = torch.distributions.inverse_gamma.InverseGamma(a[-1], b[-1]).mean
    # print(f"true error variance: {noise_var:.3f}")
    # print(f"ML estimate: {noise_var_ml:.3f}")
    # print(f"EAP estimate: {expected_noise_var:.3f}")
    # snr = torch.var(y)/noise_var
    # print(f"SNR: {snr:.3f}")

    # --- mixed effects
    n_obs = 50
    n_rfx = 2
    n_groups = 5
    me = MixedEffects(n_predictors, math.sqrt(noise_var), datadist, n_rfx, n_groups)
    ds = me.sample(n_obs, seed, include_posterior=True)
    X, y, groups, beta = ds["X"], ds["y"], ds["groups"], ds["beta"]
    S, S_emp, rfx = ds["S"], ds["S_emp"], ds["rfx"]
    print(f"true beta: {beta}")
    print(f"random effects variances:\n{S}")
    
    # # VI Posterior
    # approx = me.fitVI(y, X, groups)
    # vi_stats_beta = me.evalVI(approx, "beta")
    # print(f"VI beta posterior mu: {vi_stats_beta['means']}")
    # print(f"VI beta posterior sigma^2: {vi_stats_beta['stds'].square()}")
    
    # vi_stats_rfx = me.evalVI(approx, "sigma_b")
    # print(f"VI beta posterior mu: {vi_stats_rfx['means']}")
    # print(f"VI beta posterior sigma^2: {vi_stats_rfx['stds'].square()}")
    
    # # rfx
    # print(f"random effects variances:\n{S}")
    # print(f"random effects variances (empirical):\n{S_emp}")
    # print(f"random effects:\n{rfx}")



