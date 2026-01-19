import numpy as np
from scipy import stats

class ParametricDistribution:
    def __init__(self, rng: np.random.Generator, truncate: bool = True):
        self.rng = rng
        self.params = self.initParams()
        self.dist = self.base(**self.params)
        self.truncate = truncate and isinstance(self.dist, stats.rv_continuous)
        if self.truncate:
            self.borders = self.initBorders()

    @property
    def base(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def initParams(self) -> dict[str, float|int]:
        raise NotImplementedError

    def initBorders(self) -> tuple[float, float]:
        lower, upper = -np.inf, np.inf
        p_use = self.rng.uniform(size=2)
        use = (p_use < 0.25)
        if use.any():
            p = self.rng.uniform(size=2)
            if use.all():
                lower = self.dist.ppf(p.min())
                upper = self.dist.ppf(p.max())
            elif use[0]:
                lower = self.dist.ppf(p.min())
            else:
                upper = self.dist.ppf(p.max())
        return lower, upper

    def sample(self, n: int) -> np.ndarray:
        if self.truncate:
            left, right = self.borders
            out = []
            reached = 0
            while reached < n:
                z = self.dist.rvs(n, 1)
                inliers = (left < z) * (z < right)
                out.append(z[inliers])
                reached += inliers.sum()
            x = np.concat(out)[:n, None]
        else:
            x = self.dist.rvs((n,1))
        return x


# --- continuous

class Normal(ParametricDistribution):
    @property
    def base(self):
        return stats.norm

    def __repr__(self):
        loc, scale = self.params.values()
        return f'Normal(loc={loc:.3f}, scale={scale:.3f})'

    def initParams(self):
        loc = self.rng.uniform(-100, 100)
        scale = self.rng.uniform(0.1, 100)
        return dict(loc=loc, scale=scale)


class Student(ParametricDistribution):
    @property
    def base(self):
        return stats.t

    def __repr__(self):
        df, loc, scale = self.params.values()
        return f't(df={df}, loc={loc:.3f}, scale={scale:.3f})'
 
    def initParams(self):
        df = int(self.rng.integers(1, 50))
        loc = self.rng.uniform(-100, 100)
        scale = self.rng.uniform(0.1, 100)
        return dict(df=df, loc=loc, scale=scale)


class LogNormal(ParametricDistribution):
    @property
    def base(self):
        return stats.lognorm

    def __repr__(self):
        s, scale = self.params.values()
        return f'LogNormal(s={s:.3f}, scale={scale:.3f})'

    def initParams(self):
        s = self.rng.uniform(0.1, 5)
        scale = self.rng.uniform(1, 100)
        return dict(s=s, scale=scale)


class Uniform(ParametricDistribution):
    @property
    def base(self):
        return stats.uniform

    def __repr__(self):
        lb, ub = self.dist.support()
        return f'U(lb={lb:.3f}, ub={ub:.3f})'

    def initParams(self):
        ab = self.rng.uniform(-100, 100, size=(2,))
        a = ab.min() - 0.5
        b = ab.max() + 0.5
        scale = np.abs(b-a)
        return dict(loc=a, scale=scale)


class ScaledBeta(ParametricDistribution):
    @property
    def base(self):
        return stats.beta

    def __repr__(self):
        a, b, scale = self.params.values()
        return f'Beta(a={a:.3f}, b={b:.3f}, scale:{scale:.3f})'

    def initParams(self):
        ab = self.rng.uniform(1e-3, 10, size=(2,))
        scale = self.rng.uniform(0.5, 100)
        return dict(a=ab[0], b=ab[1], scale=scale)


# --- non-continuous
class Bernoulli(ParametricDistribution):
    @property
    def base(self):
        return stats.bernoulli

    def __repr__(self):
        p = self.params.values()
        return f'Bernoulli(p={p:.3f})'

    def initParams(self) -> dict[str, float]:
        p = self.rng.uniform(0.05, 0.95)
        return dict(p=p)

class NegativeBinomial(ParametricDistribution):
    @property
    def base(self):
        return stats.nbinom

    def __repr__(self):
        n, p = self.params.values()
        return f'Bernoulli(n={n}, p={p:.3f})'

    def initParams(self):
        n = int(self.rng.integers(1, 100))
        p = self.rng.uniform(0.05, 0.95)
        return dict(n=n, p=p)



if __name__ == '__main__':
    seed = 0
    rng = np.random.default_rng(seed)

    n = 1000
    dists = [Normal, Student, LogNormal, Uniform, ScaledBeta, Bernoulli, NegativeBinomial]
    samples = np.column_stack([dist(rng).sample(n) for dist in dists])


