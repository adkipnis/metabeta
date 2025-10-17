import random
import torch
from torch import distributions as D
from metabeta.data.distributions import (
    Normal,
    StudentT,
    Uniform,
    Bernoulli,
    NegativeBinomial,
    ScaledBeta,
)
from metabeta.utils import fullCovary
from metabeta import plot

# -----------------------------------------------------------------------------
probs = torch.tensor([0.10, 0.40, 0.05, 0.25, 0.10, 0.10])
dists = [
    Normal,
    StudentT,
    Uniform,
    Bernoulli,
    NegativeBinomial,
    ScaledBeta,
]


class Task:
    """base class for generating regression datasets"""

    def __init__(
        self,
        nu_ffx: torch.Tensor,  # ffx prior means
        tau_ffx: torch.Tensor,  # ffx prior stds
        tau_eps: torch.Tensor,  # noise prior std
        n_ffx: int,  # with bias
        features: torch.Tensor | None = None,
        limit: float = 300,  # try to sd(y) below this
        use_default: bool = False,  # use default prior parameters for predictors
        correlate: bool = True,  # correlate numerical predictors
    ):
        self.d = n_ffx
        self.features = features
        self.original_limit = limit
        self.limit = limit
        self.use_default = use_default
        self.correlate = correlate

        # data distribution
        idx = D.Categorical(probs).sample((n_ffx,))
        self.dist_data_ = [dists[i] for i in idx]
        self.dist_data = []
        self.cidx = []

        # ffx distribution
        assert len(nu_ffx) == len(tau_ffx) == self.d, "dimension mismatch"
        self.nu_ffx = nu_ffx
        self.tau_ffx = tau_ffx
        self.dist_ffx = D.Normal(self.nu_ffx, self.tau_ffx)

        # noise distribution
        self.tau_eps = tau_eps
        self.sigma_eps = D.HalfNormal(self.tau_eps).sample((1,))[0] + 1e-3  # type: ignore

    def sampleFfx(self) -> torch.Tensor:
        return self.dist_ffx.sample()

    def sampleFeatures(self, n_samples: int, ffx: torch.Tensor) -> torch.Tensor:
        """instantiate feature distributions within variance bounds and sample from them"""
        features = [torch.empty((n_samples, 0))]
        for i in range(len(ffx) - 1):
            weight = ffx[i + 1]
            dist = self.dist_data_[i](
                weight, limit=self.limit, use_default=self.use_default
            )
            if str(dist)[:4] == "Bern":
                self.cidx.append(i)
            self.dist_data += [dist]
            x = dist.sample((n_samples, 1))
            self.limit -= (x * weight).abs().max()
            features += [x]
        out = torch.cat(features, dim=-1)
        return out

    def induceCorrelation(self, x: torch.Tensor) -> torch.Tensor:
        """sample a correlation matrix and apply it to the continuous design matrix columns"""
        if self.d < 3 or not self.correlate:
            return x
        mean, std = x.mean(dim=0), x.std(dim=0)
        x_ = (x - mean) / std
        L = D.LKJCholesky(self.d - 1, 10.0).sample()

        # correlate continuous
        x_cor = (x_ @ L.T) * std + mean
        x_cor[:, self.cidx] = x[:, self.cidx]  # preserve categorial

        # correlate categorial
        R = L @ L.T
        for i in self.cidx:
            nidx = list(range(self.d - 1))
            nidx.pop(i)
            j = random.choice(nidx)
            x_cor[:, i] = self.correlateBinary(x_cor[:, j], R[i, j])
        return x_cor

    def correlateBinary(self, v: torch.Tensor, r: float | torch.Tensor):
        """generate a categorial variable whose correlation with variable v is r"""
        v = (v - v.mean()) / v.std()
        z = torch.randn_like(v)
        z = r * v + (1 - r**2) ** 0.5 * z
        probs = torch.sigmoid(z)
        z = torch.bernoulli(probs)
        return z

    def addIntercept(self, x: torch.Tensor):
        n_samples = x.shape[0]
        intercept = torch.ones(n_samples, 1)
        out = torch.cat([intercept, x], dim=-1)
        return out

    def sampleError(self, n_samples: int) -> torch.Tensor:
        eps = torch.randn((n_samples,))
        eps = (eps - torch.mean(eps)) / torch.std(eps)
        return eps * self.sigma_eps

    def signalToNoiseRatio(self, y: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        eps = y - eta
        ess = (eta - eta.mean()).square().sum()
        rss = eps.square().sum()
        snr = ess / rss
        return snr

    def relativeNoiseVariance(self, y: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        eps = y - eta
        return eps.var() / y.var()


# -----------------------------------------------------------------------------
# MFX
class MixedEffects(Task):
    def __init__(
        self,
        nu_ffx: torch.Tensor,  # ffx prior means
        tau_ffx: torch.Tensor,  # ffx prior stds
        tau_eps: torch.Tensor,  # noise prior std
        tau_rfx: torch.Tensor,  # rfx prior stds
        n_ffx: int,  # d
        n_rfx: int,  # q
        n_groups: int,  # m
        n_obs: list[int],  # n
        features: torch.Tensor | None = None,
        groups: torch.Tensor | None = None,
        use_default: bool = False,
    ):
        super().__init__(
            nu_ffx, tau_ffx, tau_eps, n_ffx, features=features, use_default=use_default
        )
        assert len(n_obs) == n_groups, (
            "mismatch between number of groups and individual observations"
        )
        assert len(tau_rfx) == n_rfx, (
            "mismatch between number of random effects and their prior stds"
        )
        self.q = n_rfx
        self.m = n_groups
        self.n_i = torch.tensor(n_obs)  # per group
        self.groups = groups

        # rfx distribution
        self.tau_rfx = tau_rfx
        if (tau_rfx == 0).any():
            self.sigmas_rfx = torch.zeros_like(tau_rfx)
        else:
            self.sigmas_rfx = D.HalfNormal(self.tau_rfx).sample()

    def sampleRfx(self) -> torch.Tensor:
        if self.q == 0:
            return torch.zeros((self.m, self.q))
        b = torch.randn((self.m, self.q))
        b = (b - b.mean(0, keepdim=True)) / b.std(0, keepdim=True)
        b = torch.where(b.isnan(), 0, b)
        b *= self.sigmas_rfx.unsqueeze(0)  # type: ignore
        return b

    def sample(self, include_posterior: bool = False) -> dict[str, torch.Tensor]:
        if include_posterior:
            raise NotImplementedError("posterior inference not implemented for MFX")
        okay = True

        # fixed effects and noise
        ffx = self.sampleFfx()
        if self.features is None:
            n_samples = int(self.n_i.sum())
            X = self.sampleFeatures(n_samples, ffx)
            X = self.induceCorrelation(X)
            X = self.addIntercept(X)
        else:
            # use predetermined features
            X = self.features
            n_i = torch.unique(self.groups, return_counts=True)[1]
            subjects = torch.randperm(len(n_i))[: self.m]
            self.n_i = n_i[subjects]
            n_samples = int(self.n_i.sum())
            subjects_mask = (self.groups.unsqueeze(-1) == subjects.unsqueeze(0)).any(-1)  # type: ignore
            X = X[subjects_mask]

            # subsample features
            d = min(self.d, X.size(1))
            features = (torch.randperm(X.size(1) - 1)[: d - 1] + 1).tolist()
            features = [0] + features
            X = X[:, features]

            # optionally add more
            if d < self.d:
                X_ = self.sampleFeatures(n_samples, ffx[d - 1 :])
                X = torch.cat([X, X_], dim=1)
        # plot.dataset(X[:, 1:])
        eps = self.sampleError(n_samples)
        eta = X @ ffx

        # check which variables are categorial
        is_binary = (X == 0) | (X == 1)
        categorial = is_binary.all(dim=0)[1:]

        # random effects and target
        groups = torch.repeat_interleave(torch.arange(self.m), self.n_i)  # (n,)
        rfx = self.sampleRfx()  # (m, q)
        B = rfx[groups]  # (n, q)
        Z = X[:, : self.q]
        y_hat = eta + (Z * B).sum(dim=-1)
        y = y_hat + eps
        snr = self.signalToNoiseRatio(y, eta)
        rnv = self.relativeNoiseVariance(y, y_hat)

        # check if dataset is within limits
        if eta.std() > self.original_limit:
            okay = False

        # Cov(mean Z, rfx), needed for standardization
        if self.q:
            weighted_rfx = Z.mean(0, keepdim=True) * rfx
            cov = fullCovary(weighted_rfx)
            cov_sum = cov.sum() - cov[0, 0]
        else:
            cov_sum = torch.tensor(0.0)

        # outputs
        out = {
            # data
            "X": X,  # (n, d-1)
            "y": y,  # (n,)
            "groups": groups,  # (n,)
            # params
            "ffx": ffx,  # (d,)
            "rfx": rfx,  # (m, q)
            "sigmas_rfx": self.sigmas_rfx,  # (q,)
            "sigma_eps": self.sigma_eps,  # (1,)
            # priors
            "nu_ffx": self.nu_ffx,  # (d,)
            "tau_ffx": self.tau_ffx,  # (d,)
            "tau_rfx": self.tau_rfx,  # (q,)
            "tau_eps": self.tau_eps,  # (1,)
            # misc
            "m": torch.tensor(self.m),  # (1,)
            "n": torch.tensor(n_samples),  # (1,)
            "n_i": self.n_i,  # (m,)
            "d": torch.tensor(self.d),  # (1,)
            "q": torch.tensor(self.q),  # (1,)
            "cov_sum": cov_sum,  # (1,)
            "snr": snr,  # (1,)
            "rnv": rnv,  # (1,)
            "categorial": categorial, # (d-1,)
            "okay": torch.tensor(okay),
        }
        return out


# =============================================================================
if __name__ == "__main__":
    # seed = 1
    # torch.manual_seed(seed)
    n_ffx = 3
    nu = torch.tensor([0.0, 1.0, 1.0])
    tau_beta = torch.tensor([50.0, 50.0, 50.0])
    tau_eps = torch.tensor(50.0)
    n_obs = 50

    # -------------------------------------------------------------------------
    # mixed effects
    print("\nmixed effects example\n----------------------------")

    n_obs = [50, 40, 30]
    n_rfx = 2
    n_groups = 3
    tau_rfx = torch.tensor([10.0, 20.0])
    me = MixedEffects(
        nu,
        tau_beta,
        tau_eps,
        tau_rfx,
        n_ffx=n_ffx,
        n_rfx=n_rfx,
        n_groups=n_groups,
        n_obs=n_obs,
    )
    ds = me.sample()

    print(f"true ffx: {ds['ffx']}")
    print(f"true noise variance: {ds['sigma_eps'] ** 2:.3f}")
    print(f"relative noise variance: {ds['rnv']:.2f}")
    print(f"random effects variances:\n{ds['sigmas_rfx']}")
