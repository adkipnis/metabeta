import torch
from torch import nn
from metabeta.models.transformers import SetTransformer
from metabeta.models.normalizingflows import CouplingFlow

class Approximator(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.build()

    def build(self) -> None:
        d_ffx = self.cfg['d']
        d_rfx = self.cfg['q']

        # --- summarizers
        s_cfg = self.cfg['summarizer']
        if s_cfg['type'] == 'set-transformer':
            Summarizer = SetTransformer
        else:
            raise NotImplementedError('unknown summarizer type')
        d_input_l = 1 + (d_ffx - 1) + (d_rfx - 1)
        d_input_g = s_cfg['d_output'] + 1 # n_obs per group
        self.summarizer_l = Summarizer(d_input=d_input_l, **s_cfg)
        self.summarizer_g = Summarizer(d_input=d_input_g, **s_cfg)

        # --- posteriors
        p_cfg = self.cfg['posterior']
        if p_cfg['type'] == 'flow':
            Posterior = CouplingFlow
        else:
            raise NotImplementedError('unknown posterior type')
        d_var = 1 + d_rfx # number of variance params
        d_prior = 2 * d_ffx + d_var # number of prior params
        d_context_g = s_cfg['d_output'] + d_prior + 2 # global summary, prior, n_groups, n_total
        d_context_l = d_ffx + d_var + d_input_g # global params, local summaries
        self.posterior_g = Posterior(d_ffx+d_var, d_context_g, **p_cfg)
        self.posterior_l = Posterior(d_rfx, d_context_l, **p_cfg)

    @property
    def device(self):
        return next(self.parameters()).device

    def _inputs(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        '''prepare input tensor for the summary network'''
        ...

    def _targets(self, data: dict[str, torch.Tensor]):
        '''prepare target tensor for the posterior network'''
        ...


