from dataclasses import dataclass, asdict
import torch
from torch import nn
from metabeta.models.transformers import SetTransformer
from metabeta.models.normalizingflows import CouplingFlow
from metabeta.utils.regularization import maskedInverseSoftplus, maskedSoftplus, dampen

@dataclass(frozen=True)
class SummarizerConfig:
    d_model: int
    d_ff: int
    d_output: int
    n_blocks: int
    n_isab: int = 0
    activation: str = 'GELU'
    dropout: float = 0.01
    type: str = 'set-transformer'

    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop('type')
        return out

@dataclass(frozen=True)
class PosteriorConfig:
    n_blocks: int
    subnet_kwargs: dict | None = None
    type: str = 'flow'
    transform: str = 'spline'

    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop('type')
        return out

@dataclass(frozen=True)
class Config:
    d_ffx: int
    d_rfx: int
    summarizer: SummarizerConfig
    posterior: PosteriorConfig


class Approximator(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.d_ffx = cfg.d_ffx # number of fixed effects
        self.d_rfx = cfg.d_rfx # number of random effects
        self.build()

    def build(self) -> None:
        d_ffx = self.d_ffx
        d_rfx = self.d_rfx

        # --- summarizers
        s_cfg = self.cfg.summarizer
        if s_cfg.type == 'set-transformer':
            Summarizer = SetTransformer
        else:
            raise NotImplementedError('unknown summarizer type')
        d_input_l = 1 + (d_ffx - 1) + (d_rfx - 1)
        d_input_g = s_cfg.d_output + 1 # n_obs per group
        self.summarizer_l = Summarizer(d_input=d_input_l, **s_cfg.to_dict())
        self.summarizer_g = Summarizer(d_input=d_input_g, **s_cfg.to_dict())

        # --- posteriors
        p_cfg = self.cfg.posterior
        if p_cfg.type == 'flow':
            Posterior = CouplingFlow
        else:
            raise NotImplementedError('unknown posterior type')
        d_var = 1 + d_rfx # number of variance params
        d_prior = 2 * d_ffx + d_var # number of prior params
        d_context_g = s_cfg.d_output + d_prior + 2 # global summary, prior, n_groups, n_total
        d_context_l = d_ffx + d_var + d_input_g # global params, local summaries
        self.posterior_g = Posterior(d_ffx+d_var, d_context_g, **p_cfg.to_dict())
        self.posterior_l = Posterior(d_rfx, d_context_l, **p_cfg.to_dict())

    @property
    def device(self):
        return next(self.parameters()).device

    def _inputs(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        '''prepare input tensor for the summary network'''
        ...

    def _targets(self, data: dict[str, torch.Tensor]):
        '''prepare target tensor for the posterior network'''
        ...


if __name__ == '__main__':
    from metabeta.utils.logger import setupLogging
    setupLogging(1)

    cfg = {'d': 3, 'q': 1}
    cfg['summarizer'] = {
        'type': 'set-transformer',
        'd_model': 64,
        'd_ff': 128,
        'd_output': 64,
        'n_blocks': 4,
        'n_isab': 0,
        'activation': 'GeGLU',
        'dropout': 0.01,
    }
    cfg['posterior'] = {
        'type': 'flow',
        'n_blocks': 4,
        'transform': 'spline',
        'net_kwargs': {
            'activation': 'ReLU',
            },
    }

    model = Approximator(cfg)


