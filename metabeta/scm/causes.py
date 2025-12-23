import numpy as np
import torch
from torch import nn
from typing import Iterable


class CauseSampler(nn.Module):
    def __init__(self,
                 n_samples: int,
                 n_causes: int,
                 dist: str = 'normal', # [mixed, normal, uniform]
                 fixed: bool = False, # random parameters for dist
                 **kwargs
                 ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.n_causes = n_causes
        self.dist = {'normal': self._normal,
                     'uniform': self._uniform,
                     'mixed': self._mixed}[dist]
        self.fixed = fixed
        if not self.fixed:
            self.mu = torch.randn(n_causes)
            self.sigma = (torch.randn(n_causes) * self.mu).abs()

    def _normal(self, shape: Iterable) -> torch.Tensor:
        x = torch.randn(*shape)
        if not self.fixed:
            mu, sigma = self.mu.unsqueeze(0), self.sigma.unsqueeze(0)
            x = mu + x * sigma
        return x

    def _uniform(self, shape: Iterable) -> torch.Tensor:
        x = torch.rand(*shape)
        if not self.fixed:
            mu, sigma = self.mu.unsqueeze(0), self.sigma.unsqueeze(0)
            x = mu + (x-0.5) * sigma * np.sqrt(12)
        return x

    def _multinomial(self, shape: Iterable) -> torch.Tensor:
        n, d = shape
        n_categories = np.random.randint(2, 20)
        probs = torch.rand((d, n_categories))
        x = torch.multinomial(probs, n, replacement=True).permute(1,0).float()
        x = (x - x.mean(0)) / x.std(0)
        if not self.fixed:
            mu, sigma = self.mu.unsqueeze(0), self.sigma.unsqueeze(0)
            x = mu + x * sigma
        return x

    def _zipf(self, shape: Iterable) -> torch.Tensor:
        a = 2 * np.random.rand() + 2
        x = np.random.zipf(a, shape) # type: ignore
        x = torch.tensor(x).clamp(max=10).float()
        x = (x - x.mean(0)) / x.std(0)
        if not self.fixed:
            mu, sigma = self.mu.unsqueeze(0), self.sigma.unsqueeze(0)
            x = mu + x * sigma
        return x

    def _mixed(self, shape: Iterable) -> torch.Tensor:
        out = []
        dists = [self._normal, self._uniform, self._multinomial, self._zipf]
        n, d = shape
 
        # draw distributions
        probs = np.random.dirichlet(alpha=np.ones((4,)), size=(d,))
        ids = np.sort(probs.argmax(-1))
        ids, counts = np.unique_counts(ids)

        # partition moments accordingly
        if not self.fixed:
            mu, sigma = self.mu, self.sigma
            parts = np.cumsum(np.pad(counts, (1,0)))
            mus = [mu[parts[i]:parts[i+1]] for i in range(len(parts) - 1)]
            sigmas = [sigma[parts[i]:parts[i+1]] for i in range(len(parts) - 1)]

        # draw from each distribution
        for idx, d_ in zip(ids, counts):
            if not self.fixed:
                self.mu = mus[idx]
                self.sigma = sigmas[idx]
            dist = dists[idx]
            x = dist((n, d_))
            out.append(x)

        # gather and permute positions
        x = torch.cat(out, dim=-1)
        x = x[:, torch.randperm(d)]
        return x


    def sample(self) -> torch.Tensor:
        shape = (self.n_samples, self.n_causes)
        return self.dist(shape)


if __name__ == '__main__':
