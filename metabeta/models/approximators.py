import numpy as np
import torch
from torch import nn
from metabeta.utils import (
    maskedMean,
    maskedStd,
    batchCovary,
    maskedSoftplus,
    maskedInverseSoftplus,
    dampen,
    nParams,
)
from metabeta.models.transformers import (
    BaseSetTransformer,
    SetTransformer,
    DualTransformer,
)
from metabeta.models.posteriors import Posterior, CouplingPosterior
from metabeta import plot

mse = nn.MSELoss()

summary_defaults = {
    "type": "set-transformer",
    "d_model": 64,
    "n_blocks": 3,
    "d_ff": 128,
    "depth": 1,
    "d_output": 64,
    "n_heads": 8,
    "dropout": 0.01,
    "activation": "GELU",
}

posterior_defaults = {
    "type": "flow-affine",
    "flows": 3,
    "d_ff": 128,
    "depth": 3,
    "dropout": 0.01,
    "activation": "ReLU",
}

model_defaults = {"type": "mfx", "seed": 42, "d": 5, "q": 2, "tag": ""}

# -----------------------------------------------------------------------------


# base approximator
class Approximator(nn.Module):
    posterior: Posterior | None = None
    stats: dict = {}

    def __init__(
        self,
        constrain: bool = True,
        use_standardization: bool = True,
    ):
        super().__init__()
        self.constrain = constrain
        self.use_standardization = use_standardization

    @staticmethod
    def modelID(s_dict: dict, p_dict: dict, m_dict: dict) -> str:
        prefix = ""
        suffix = ""
        if m_dict["tag"]:
            suffix = "-" + m_dict["tag"]
        summary = f"{prefix}{s_dict['type']}-{s_dict['n_blocks']}-{s_dict['d_model']}-{s_dict['d_ff']}*{s_dict['depth']}-{s_dict['d_output']}-{s_dict['n_heads']}-{s_dict['activation']}-{s_dict['dropout']}"
        posterior = f"{p_dict['type']}-{p_dict['flows']}-{p_dict['d_ff']}*{p_dict['depth']}-{p_dict['activation']}-{p_dict['dropout']}"
        return f"{m_dict['type']}-d={m_dict['d']}-q={m_dict['q']}-{summary}-+-{posterior}-seed={m_dict['seed']}{suffix}"

    @property
    def device(self):
        return next(self.parameters()).device

    def inputs(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """prepare input tensor for the summary network"""
        raise NotImplementedError

    def targets(self, data: dict[str, torch.Tensor]):
        """prepare target tensor for the posterior network"""
        raise NotImplementedError

    def forward(self, data: dict[str, torch.Tensor], sample: bool = False):
        raise NotImplementedError

    def standardize(self,
                    x: torch.Tensor,
                    name: str,
                    mask: torch.Tensor | None = None,
                    categorial: torch.Tensor | None = None,
                    ) -> torch.Tensor:
        """z-standardization specific for each dataset"""
        dim = tuple(range(1, x.dim() - 1))
        if mask is not None:
            mean = maskedMean(x, dim, mask=mask)
            std = maskedStd(x, dim, mask=mask, mean=mean) + 1e-12
        else:
            mean = x.mean(dim, keepdim=True)
            std = x.std(dim, keepdim=True) + 1e-12
        if categorial is not None:
            categorial = categorial.unsqueeze(1).unsqueeze(1)
            mean[categorial] = 0.
            std[categorial] = 1.
        self.stats[name] = {"mean": mean, "std": std}
        out = (x - mean) / std
        if mask is not None:
            out *= mask
        return out

    def unpackMoment(
        self, names_list: list[str], moment: str, device: str | None = None
    ) -> dict[str, torch.Tensor]:
        if device is None:
            return {name: self.stats[name][moment] for name in names_list}
        return {name: self.stats[name][moment].to(device) for name in names_list}

    def unpackMean(self, names_list, device=None):
        return self.unpackMoment(names_list, "mean", device)

    def unpackStd(self, names_list, device=None):
        return self.unpackMoment(names_list, "std", device)

    def moments(
        self,
        proposed: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """wrapper for location and scale of the posterior"""
        assert 'samples' in proposed, "no samples in proposed"
        assert self.posterior is not None
        return self.posterior.getLocScale(proposed)

    def quantiles(
        self,
        proposed: dict[str, torch.Tensor],
        roots: list = [0.025, 0.975],
        calibrate: bool = False,
    ) -> torch.Tensor | None:
        """wrapper for desired quantiles of the posterior"""
        if "samples" in proposed:
            assert self.posterior is not None
            samples = proposed["samples"].clone()
            quantiles = self.posterior.getQuantiles(samples, roots, calibrate)
            return quantiles

    def ranks(
        self, proposed: dict[str, torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        assert self.posterior is not None
        return self.posterior.getRanks(proposed, targets)

    def plotRecovery(
        self,
        targets: torch.Tensor,
        names: list[str],
        means: torch.Tensor,
        color: str = "darkgreen",
        alpha: float = 0.3,
        return_stats: bool = True,
    ) -> None | tuple[float, float]:
        return plot.recovery(targets, names, means, color, alpha, return_stats)

    def recoveryGrouped(
        self,
        targets: list[torch.Tensor],
        names: list[list[str]],
        means: list[torch.Tensor],
        titles: list[str] = [],
        marker: str = "o",
        alpha: float = 0.2,
    ) -> None | tuple[float, float]:
        return plot.recoveryGrouped(targets, names, means, titles, marker, alpha)


class ApproximatorMFX(Approximator):
