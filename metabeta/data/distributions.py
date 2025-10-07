from collections.abc import Iterable
import torch
from torch import distributions as D


# ==============================================================================
# for data simulation
class DistWithPrior:
    """Base class for 1-d distributions with automated prior used for sampling dataset features"""

    def __init__(
        self,
        weight: float,
        limit: float = 1e3,
        center: bool = True,
        use_default: bool = False,
    ):
        self.weight = weight
        self.limit = limit  # absolute limit for weight * samples
        self.center = center
        self.use_default = use_default
        if use_default:
            self.params = self.defaultParams()
        else:
            self.params = self.findParams()
        self.dist = self.base(**self.params)

    def __repr__(self):
        return self.dist.__repr__()

    @property
    def base(self):
        raise NotImplementedError

    def defaultParams(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def samplePrior(self, n: int = 1) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def findParams(self) -> dict[str, torch.Tensor]:
        params = self.samplePrior(100)
        under_limit = self.check(params)
        if under_limit.any():
            idx = torch.where(under_limit)[0][0]
            params = {k: v[idx] for k, v in params.items()}
        else:
            params = self.defaultParams()
        return params

    def check(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        # check if x * weight is likely surpass set limit
        raise NotImplementedError

    def checkICDF(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        dist = self.base(**params)
        icdf_0 = dist.icdf(torch.tensor(0.05))
        icdf_1 = dist.icdf(torch.tensor(0.95))
        xb_0 = (icdf_0 * self.weight).abs()
        xb_1 = (icdf_1 * self.weight).abs()
        return (xb_0 < self.limit) * (xb_1 < self.limit)

    def checkSample(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        dist = self.base(**params)
        x = dist.sample((100,))
        xb = (x * self.weight).abs().max(0)[0]
        return xb < self.limit

    def sample(self, dims: Iterable[int]) -> torch.Tensor:
        return self.dist.sample(dims)


class Normal(DistWithPrior):
    @property
    def base(self):
        return D.Normal

    def defaultParams(self):
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        return dict(loc=loc, scale=scale)

    def samplePrior(self, n=1):
        scale = D.Uniform(0.5, 25.0).sample((n,))
        if self.center:
            loc = torch.zeros_like(scale)
        else:
            loc = D.Uniform(-25.0, 25.0).sample((n,))
        return dict(loc=loc, scale=scale)

    def check(self, params):
        return self.checkICDF(params)


class StudentT(DistWithPrior):
    @property
    def base(self):
        return D.StudentT

    def defaultParams(self):
        df = torch.tensor(3.0)
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        return dict(df=df, loc=loc, scale=scale)

    def samplePrior(self, n=1):
        scale = D.Uniform(0.5, 25.0).sample((n,))
        if self.center:
            loc = torch.zeros_like(scale)
        else:
            loc = D.Uniform(-25.0, 25.0).sample((n,))
        df = torch.randint(3, 100, (n,), dtype=loc.dtype)
        return dict(df=df, loc=loc, scale=scale)

    def check(self, params):
        return self.checkSample(params)


class Uniform(DistWithPrior):
    @property
    def base(self):
        return D.Uniform

    def defaultParams(self):
        a = torch.tensor(12.0).sqrt()
        return dict(low=-a, high=a)

    def samplePrior(self, n: int = 1):
        if self.center:
            b = D.Uniform(0.5, 100.0).sample((n,))
            a = -b
        else:
            ab = D.Uniform(0.5, 100.0).sample((2, n))
            a = ab.min(0)[0] - 0.5
            b = ab.max(0)[0] + 0.5

        return dict(low=a, high=b)

    def check(self, params):
        return self.checkICDF(params)


class Bernoulli(DistWithPrior):
