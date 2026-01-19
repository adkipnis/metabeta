import numpy as np
from scipy import stats


class ParametricDistribution:
    def __init__(self, rng: np.random.Generator, truncate: bool = True):
        self.rng = rng
        self.params = self.initParams()
        self.dist = self.base(**self.params)
        self.truncate = truncate and self.is_continuous
        if self.truncate:
            self.borders = self.initBorders()

    @property
    def base(self):
        raise NotImplementedError

    @property
    def is_continuous(self) -> bool:
        return True

    @property
    def infinite_borders(self) -> bool:
        if not self.truncate:
            return True
        return (np.isfinite(self.borders) == False).all()

    def __repr__(self) -> str:
        raise NotImplementedError

    def initParams(self) -> dict[str, float|int]:
        raise NotImplementedError

    def initBorders(self) -> tuple[float, float]:
        lower, upper = -np.inf, np.inf
        p_use = self.rng.uniform(size=2)
        use = (p_use < 0.25)
        if use.any():
            eps = 1e-6
            min_diff = 0.1
            p0, p1 = self.rng.uniform(eps, 1-eps, size=2)
            if p0 > p1:
                p1, p0 = p0, p1
            if p1 - p0 < min_diff:
                p0 = max(eps, p0-min_diff/2)
                p1 = min(1-eps, p1+min_diff/2)
            if use[0]:
                lower = self.dist.ppf(p0)
            if use[1]:
                upper = self.dist.ppf(p1)
        return lower, upper

    def sample(self, n: int) -> np.ndarray:
        # untruncated sampling
        if not self.truncate or self.infinite_borders:
            return self.dist.rvs(size=(n,1), random_state=self.rng)

        # truncated sampling
        left, right = self.borders
        out = []
        reached = 0
        max_iters = 10_000

        for _ in range(max_iters):
            remaining = n - reached
            batch = max(256, remaining)
            z = self.dist.rvs(size=(batch,1), random_state=self.rng)

            inliers = (left < z) & (z < right)
            if np.any(inliers):
                z = z[inliers].reshape(-1,1)
                out.append(z)
                reached += int(inliers.sum())
                if reached >= n:
                    break

        if reached < n:
            raise RuntimeError(
                f'truncated sampling failed: {reached}/{n}, borders={self.borders}, dist={self}')

        return np.concatenate(out, axis=0)[:n]



# --- continuous

class Normal(ParametricDistribution):
    @property
    def base(self):
        return stats.norm

    def __repr__(self):
        loc = self.params['loc']
        scale = self.params['scale']
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
        df = self.params['df']
        loc = self.params['loc']
        scale = self.params['scale']
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
        s = self.params['s']
        scale = self.params['scale']
        return f'LogNormal(s={s:.3f}, scale={scale:.3f})'

    def initParams(self):
        s = self.rng.uniform(0.1, 3)
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
        ab = self.rng.uniform(-100, 100, size=2)
        a = ab.min() - 0.5
        b = ab.max() + 0.5
        scale = np.abs(b-a)
        return dict(loc=a, scale=scale)


class ScaledBeta(ParametricDistribution):
    @property
    def base(self):
        return stats.beta

    def __repr__(self):
        a = self.params['a']
        b = self.params['b']
        scale = self.params['scale']
        return f'Beta(a={a:.3f}, b={b:.3f}, scale={scale:.3f})'

    def initParams(self):
        ab = self.rng.uniform(1e-3, 10, size=2)
        scale = self.rng.uniform(0.5, 100)
        return dict(a=ab[0], b=ab[1], scale=scale)


# --- non-continuous
class Bernoulli(ParametricDistribution):
    @property
    def base(self):
        return stats.bernoulli

    @property
    def is_continuous(self) -> bool:
        return False

    def __repr__(self):
        p = self.params['p']
        return f'Bernoulli(p={p:.3f})'

    def initParams(self) -> dict[str, float]:
        p = self.rng.uniform(0.05, 0.95)
        return dict(p=p)

class NegativeBinomial(ParametricDistribution):
    @property
    def base(self):
        return stats.nbinom

    @property
    def is_continuous(self) -> bool:
        return False

    def __repr__(self):
        n = self.params['n']
        p = self.params['p']
        return f'NegBinom(n={n}, p={p:.3f})'

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


