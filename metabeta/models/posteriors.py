import numpy as np
import torch
from torch import nn
from metabeta.models.normalizingflows.coupling import CouplingFlow
from metabeta.utils import weightedMean, weightedStd
from metabeta.evaluation.coverage import Calibrator
from metabeta import plot

mse = nn.MSELoss(reduction="none")


class Posterior(nn.Module):
    def __init__(self, d_target: int):
        super().__init__()
        self.d_target = d_target
        self.append = d_target == 1
        self.calibrator = Calibrator(d_target)

    def plot(
        self,
        proposed: dict[str, torch.Tensor],
        target: torch.Tensor,
        names: list[str],
        batch_idx: int = 0,
        mcmc: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ):
        plot.posterior(proposed, target, names, batch_idx=batch_idx, mcmc=mcmc, **kwargs)

    def mean(self,
             samples: torch.Tensor,
             weights: torch.Tensor | None = None,
             ):
        return weightedMean(samples, weights)

    def std(self, 
            samples: torch.Tensor,
            weights: torch.Tensor | None = None,
            n_eff: torch.Tensor | None = None,
            ):
        return weightedStd(samples, weights, n_eff)

    def getLocScale(self,
                    proposed: dict[str, torch.Tensor] | torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(proposed, torch.Tensor):
            proposed = {"samples": proposed}
        samples = proposed["samples"]
        weights = proposed.get("weights", None)
        n_eff = proposed.get("n_eff", None)
        loc = self.mean(samples, weights)
        scale = self.std(samples, weights, n_eff)
        return loc, scale

    def getCDF(
        self,
        proposed: dict[str, torch.Tensor],
        use_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        samples = proposed["samples"].clone()
        s = samples.shape[-1]
        samples, idx = samples.sort(dim=-1, descending=False)
        if 'weights' in proposed and use_weights:
            w_sorted =  torch.gather(proposed['weights'], -1, idx)
            cdf = torch.cumsum(w_sorted, -1) / s
        else:
            cdf = torch.linspace(0, 1, s)
            cdf = cdf.view(1, 1, -1).expand_as(samples).contiguous()
        return samples, cdf

    def getQuantiles(
        self,
        samples: torch.Tensor,
        roots: list[float] | torch.Tensor,
        calibrate: bool = False,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert len(roots) == 2, "roots must contain 2 values"
        assert np.isclose(0.5 - roots[0], roots[1] - 0.5), "roots must be even"

        # prepare
        samples, idx = samples.sort(dim=-1, descending=False)
        b, d, s = samples.shape
        if isinstance(roots, list):
            roots = torch.tensor(roots)

        # unweighed: get indices from roots
        if weights is None or calibrate:
            indices = (roots * s).round().int().clamp(max=s - 1)
            quantiles = samples[..., indices]
        # weighed: get indices from weights
        else:
            roots_ = torch.zeros(b, d, len(roots))
            roots_[:, :] = roots
            w_sorted = torch.gather(weights, -1, idx)
            cdf = torch.cumsum(w_sorted, -1) / s
            indices = torch.searchsorted(cdf, roots_).clamp(max=s - 1)
            quantiles = samples.gather(dim=-1, index=indices)

        # optionally calibrate
        if calibrate:
            i = (roots[1] - roots[0]) * 100
            i = int(np.round(i))
            quantiles = self.calibrator.apply(quantiles, i)

        return quantiles

    def hpdQuantiles(
        self, proposed: dict[str, torch.Tensor], mass: float
    ) -> torch.Tensor:
        # this calculates the quantiles using the HPDI
        # pro: considers joint distribution, honoring mutual dependencies
        # con: requires log_probs
        # TODO: incorporate weights
        log_p = proposed["log_prob"].clone()
        samples = proposed["samples"].clone()
        b, d, s = samples.shape
        border = torch.tensor(s * mass).ceil().int()
        log_p, idx = log_p.sort(dim=-1, descending=True)
        inside = idx[:, :border]
        quantiles = torch.zeros(b, d, 2)
        for i in range(b):
            subset = samples[i, :, inside[i]]
            quantiles[i, :, 0] = subset.min(-1)[0]
            quantiles[i, :, 1] = subset.max(-1)[0]
        return quantiles

    def getRanks(
        self, proposed: dict[str, torch.Tensor], values: torch.Tensor
    ) -> torch.Tensor:
        # get the relative rank of values within the proposed samples
        # this approximates the empirical cdf of the samples
        samples = proposed["samples"].clone()
        samples, _ = samples.sort(dim=-1, descending=False)
        closest = torch.searchsorted(samples, values.unsqueeze(-1), right=False)
        ranks = closest.squeeze(-1) / samples.shape[0]
        return ranks

    def logProb(self, summary: torch.Tensor, values: torch.Tensor, mask=None):
        raise NotImplementedError

    def loss(self, summary: torch.Tensor, targets: torch.Tensor, mask=None):
        raise NotImplementedError

    def sample(self, summary: torch.Tensor, mask=None, n: int = 100):
        raise NotImplementedError

    def forward(
        self,
        summary: torch.Tensor,
        targets: torch.Tensor,
        sample=False,
        n: int = 100,
        **kwargs,
    ):
        # init
        proposed = {}

        # handle 1-dim targets
        if self.append:
            intercept = torch.zeros_like(targets[..., 0:1])
            targets = torch.cat([intercept, targets], dim=-1)

        # forward pass with loss
        mask = (targets != 0.0).float()
        loss = self.loss(summary, targets, mask=mask)

        # optional backward pass with sampling
        if sample:
            proposed = self.estimate(summary, mask, n)
        return loss, proposed

    def estimate(self, summary: torch.Tensor, mask: torch.Tensor, n: int = 100):
        with torch.no_grad():
            # handle 1-dim targets
            if self.append and mask.shape[-1] == 1:
                intercept = torch.zeros_like(mask[..., 0:1])
                mask = torch.cat([intercept, mask], dim=-1)

            samples, log_q = self.sample(summary, mask=mask, n=n)
            if samples.dim() == 3:
                samples = samples.permute(0, 2, 1)  # (b, d, s)
            elif samples.dim() == 4:
                samples = samples.permute(0, 1, 3, 2)  # (b, m, d, s)

            if self.append:
                samples = samples[..., 1:, :]
            return {"samples": samples, "log_prob": log_q}


class CouplingPosterior(Posterior):
