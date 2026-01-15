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
 
    @property
    def n_params(self) -> int:
        param_counts = (p.numel() for p in model.parameters() if p.requires_grad)
        return sum(param_counts)

    def _inputs(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        ''' get summarizer inputs '''
        d, q = self.d_ffx, self.d_rfx # TODO: adapt to batchwise max
        y = data['y'].unsqueeze(-1)
        X = data['X'][..., 1:d]
        Z = data['Z'][..., 1:q]
        return torch.cat([y, X, Z], dim=-1)

    def _targets(self, data: dict[str, torch.Tensor], local: bool = False) -> torch.Tensor:
        ''' get posterior targets '''
        if local:
            return data['rfx']
        out = [data['ffx'], data['sigma_rfx'], data['sigma_eps'].unsqueeze(-1)]
        return torch.cat(out, dim=-1)

    def _addMetadata(
        self,
        summary: torch.Tensor,
        data: dict[str, torch.Tensor],
        local: bool = False,
        numerator: int = 10, # for normalizing counts
    ) -> torch.Tensor:
        ''' append summary with selected metadata '''
        out = [summary]
        if local:
            n_obs = data['n_i'].unsqueeze(-1).sqrt() / numerator
            out += [n_obs]
        else:
            n_total = data['n'].unsqueeze(-1).sqrt() / numerator
            n_groups = data['m'].unsqueeze(-1).sqrt() / numerator
            nu_ffx = data['nu_ffx'].clone()
            tau_ffx = data['tau_ffx'].clone()
            tau_rfx = data['tau_rfx'].clone()
            tau_eps = data['tau_eps'].clone().unsqueeze(-1)
            # TODO: optionally dampen prior params
            out += [n_total, n_groups, nu_ffx, tau_ffx, tau_rfx, tau_eps]
        return torch.cat(out, dim=-1)

    def _preprocess(self, targets: torch.Tensor, local: bool = False) -> torch.Tensor:
        ''' constrain target parameters '''
        if local:
            return targets
        targets = targets.clone()
        d = self.d_ffx

        # project from R+ to R
        sigmas = targets[:, d:]
        sigmas = maskedInverseSoftplus(sigmas)
        targets[:, d:] = sigmas
        return targets

    def _postprocess(self, proposed: dict[str, dict[str, torch.Tensor]]):
        ''' reverse _preprocess for samples '''
        if 'samples' not in proposed['local']:
            return proposed
        d = self.d_ffx

        # local postprocessing
        ...

        # global postprocessing
        if 'global' in proposed:
            samples = proposed['global']['samples'].clone()
            sigmas = samples[:, d:]
            sigmas = maskedSoftplus(sigmas)
            samples[:, d:] = sigmas
            proposed['global']['samples'] = samples

        return proposed

    def forward(
            self, data: dict[str, torch.Tensor], n_samples: int = 0,
    ) -> dict[str, dict[str, torch.Tensor]]:
        # prepare
        proposed = {}
        inputs = self._inputs(data)

        # local summaries
        summary_l = self.summarizer_l(inputs, mask=data['mask_n'])
        summary_l = self._addMetadata(summary_l, data, local=True)

        # global summary
        summary_g = self.summarizer_g(summary_l, mask=data['mask_m'])
        summary_g = self._addMetadata(summary_g, data, local=False)

        # global posterior
        targets_g = self._targets(data, local=False)
        targets_g = self._preprocess(targets_g, local=False)
        loss_g = self.posterior_g.loss(
            targets_g, summary_g, mask=data['mask_d'])
        if n_samples > 0:
            proposed['global'] = self.posterior_g.sample(
                n_samples, summary_g, mask=data['mask_d'])

        # local posterior
        targets_l = self._targets(data, local=True)
        targets_l = self._preprocess(targets_l, local=True)






if __name__ == '__main__':
    s_cfg = SummarizerConfig(
        d_model=128,
        d_ff=256,
        d_output=64,
        n_blocks=2,
    )
    p_cfg = PosteriorConfig(
        transform='spline',
        subnet_kwargs={'activation': 'GeGLU'},
        n_blocks=6,
    )
    cfg = Config(
        d_ffx=3,
        d_rfx=1,
        summarizer=s_cfg,
        posterior=p_cfg,
    )

    model = Approximator(cfg)
    model.device
    model.n_params

