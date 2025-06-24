from math import sqrt
import torch
from torch import distributions as D
from typing import Dict, Tuple, Union, List

# -----------------------------------------------------------------------------
# base class
class Task:
    def __init__(self,
                 sigma_error: Union[float, torch.Tensor], # standard deviation of the additive noise
                 n_ffx: int, # without bias
                 ):

        # data distribution
        self.d = n_ffx
        self.dist_data = D.Uniform(-sqrt(12), sqrt(12)) # mean = 0, sd = 1

        # ffx distribution
        self.sigma_ffx = 3.
        self.dist_ffx = D.Normal(0., self.sigma_ffx)

        # noise distribution
        self.sigma_error = sigma_error
 
    def _sampleFfx(self) -> torch.Tensor:
        d = self.d + 1
        return self.dist_ffx.sample((d,)) # type: ignore

    def _sampleFeatures(self, n_samples: int) -> torch.Tensor:
        return self.dist_data.sample((n_samples, self.d)) # type: ignore

    def _addIntercept(self, x: torch.Tensor):
        n = x.shape[0]
        intercept = torch.ones(n, 1)
        return torch.cat([intercept, x], dim=-1)

    def _sampleError(self, n_samples: int) -> torch.Tensor:
        eps = torch.randn((n_samples,))
        eps = eps / torch.std(eps)
        return self.sigma_error * eps

    def sample(self, n_samples: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def signalToNoiseRatio(self, y: torch.Tensor, eta: torch.Tensor):
        eps = y - eta
        ess = torch.sum((eta - torch.mean(eta)).square())
        rss = torch.sum(eps.square())
        snr = ess / rss
        return snr


# -----------------------------------------------------------------------------
# FFX
class FixedEffects(Task):
    def __init__(self,
                 sigma_error: Union[float, torch.Tensor],
                 n_ffx: int,
                 **kwargs
                 ):
        super().__init__(sigma_error, n_ffx)

    def sample(self, n_samples: int, include_posterior: bool = False) -> Dict[str, torch.Tensor]:
        X = self._sampleFeatures(n_samples)
        X = self._addIntercept(X)
        ffx = self._sampleFfx()
        eps = self._sampleError(n_samples)
        eta = X @ ffx
        y = eta + eps
        snr = self.signalToNoiseRatio(y, eta)
        out = {"X": X, # (n, d)
               "y": y, # (n,)
               "ffx": ffx, # (d+1,)
               "sigma_error": self.sigma_error, # (1,)
               "snr": snr, # (1,)
               "n": torch.tensor(n_samples), # (1,)
               "d": torch.tensor(self.d), # (1,)
               }
        if include_posterior:
            mu, Sigma, alpha, beta = self.posteriorParams(X, y)
            ffx_stats = {"mu": mu, "Sigma": Sigma}
            noise_stats = {"alpha": alpha, "beta": beta}
            out['analytical'] = {"ffx": ffx_stats, "noise": noise_stats}
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

# -----------------------------------------------------------------------------
# MFX
class MixedEffects(Task):
    def __init__(self,
                 sigma_error: Union[float, torch.Tensor],
                 sigmas_rfx: torch.Tensor,
                 n_ffx: int,
                 n_rfx: int,
                 n_groups: int,
                 n_obs: List[int],
                 ):
        super().__init__(sigma_error, n_ffx)
        self.q = n_rfx
        self.m = n_groups 
        self.n_i = torch.tensor(n_obs) # per group
        assert sigmas_rfx.shape[0] == n_rfx, "mismatch between rfx and provided variances"
        assert len(self.n_i) == self.m, "mismatch between number of groups and individual observations"
        self.sigmas_rfx = sigmas_rfx

    def _sampleRfx(self) -> torch.Tensor:
        b = torch.randn((self.m, self.q))
        b = b / torch.std(b, dim=0)
        sigmas = self.sigmas_rfx.unsqueeze(0)
        return b * sigmas

    def sample(self, include_posterior: bool = False) -> Dict[str, torch.Tensor]:
        if include_posterior:
            raise NotImplementedError('posterior inference not implemented for MFX')
        n_samples = self.n_i.sum()
        
        # fixed effects and noise
        X = self._sampleFeatures(n_samples)
        X = self._addIntercept(X)
        ffx = self._sampleFfx()
        eps = self._sampleError(n_samples)

        # random effects and target
        groups = torch.repeat_interleave(torch.arange(self.m), self.n_i)
        b = self._sampleRfx() # (m, q)
        B = b[groups] # (n, q)
        Z = X[:,:self.q]
        eta = X @ ffx 
        y = eta + (Z * B).sum(dim=-1) + eps
        snr = self.signalToNoiseRatio(y, eta)

        # outputs
        out = {"X": X, # (n, d)
               "y": y, # (n,)
               "groups": groups, # (n,)
               "ffx": ffx, # (d,)
               "rfx": b, # (m, q)
               "sigmas_rfx": self.sigmas_rfx, # (q,)
               "sigma_error": self.sigma_error, # (1,)
               "snr": snr,
               "n": n_samples, # (1,)
               "n_i": self.n_i, # (m,)
               "m": torch.tensor(self.m), # (1,)
               "d": torch.tensor(self.d), # (1,)
               "q": torch.tensor(self.q), # (1,)
               }
        return out
 

