from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
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
        self.posterior_l = Posterior(max(d_rfx, 2), d_context_l, **p_cfg.to_dict())

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

    def _targets(
            self, data: dict[str, torch.Tensor], local: bool = False,
    ) -> torch.Tensor:
        ''' get posterior targets '''
        if local:
            targets = data['rfx']
            if targets.shape[-1] == 1: # handle 1D local params for flow
                targets = F.pad(targets, (0,1))
            return targets
        targets = [data['ffx'], data['sigma_rfx'], data['sigma_eps'].unsqueeze(-1)]
        return torch.cat(targets, dim=-1)

    def _masks(
            self, data: dict[str, torch.Tensor], local: bool = False,
    ) -> torch.Tensor:
        ''' get masks for the posterior targets '''
        if local:
            mask_q = data['mask_q']
            if mask_q.shape[-1] == 1: # handle 1D local params for flow
                mask_q = F.pad(mask_q, (0,1))
            return data['mask_m'].unsqueeze(-1) & mask_q.unsqueeze(-2)
        masks = [
            data['mask_d'],
            data['mask_q'],
            torch.ones_like(data['mask_d'][..., 0:1]),
        ]
        return torch.cat(masks, dim=-1)

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
            n_obs = data['ns'].unsqueeze(-1).sqrt() / numerator
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

    def _localContext(
        self,
        global_targets: torch.Tensor | None,
        local_summary: torch.Tensor,
        proposed: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        b, m, _ = local_summary.shape
        if 'global' in proposed:
            global_params = proposed['global']['samples']
            s = global_params.shape[-2]
            global_params = global_params.unsqueeze(1).expand(b, m, s, -1)
            local_summary = local_summary.unsqueeze(2).expand(b, m, s, -1)    
        elif global_targets is not None:
            global_params = global_targets.unsqueeze(1).expand(b, m, -1)
        else:
            raise ValueError('no samples or global targets provided')
        return torch.cat([local_summary, global_params], dim=-1)

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
        if not proposed:
            return proposed
        d = self.d_ffx

        # local postprocessing
        if 'local' in proposed:
            ...

        # global postprocessing
        if 'global' in proposed:
            samples = proposed['global']['samples'].clone()
            sigmas = samples[:, d:]
            sigmas = maskedSoftplus(sigmas)
            samples[:, d:] = sigmas
            proposed['global']['samples'] = samples

        return proposed

    def _expandLocal(
            self, targets: torch.Tensor | None, mask: torch.Tensor, s: int,
        ) -> tuple[torch.Tensor | None, torch.Tensor]:
        ''' when sampling, we condition the local posterior on all samples
            of the global posterior, so we need to expand the targets/mask '''
        if s > 0:
            b, m, q = mask.shape
            mask = mask.unsqueeze(-2).expand(b, m, s, q)
            if targets is not None:
                targets = targets.unsqueeze(-2).expand(b, m, s, q)
        return targets, mask

    def forward(
            self, data: dict[str, torch.Tensor], n_samples: int = 0,
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        # prepare
        proposed = {}
        inputs = self._inputs(data)

        # ---------------------------------------------------------------------
        # local summaries
        summary_l = self.summarizer_l(inputs, mask=data['mask_n'])
        summary_l = self._addMetadata(summary_l, data, local=True)

        # global summary
        summary_g = self.summarizer_g(summary_l, mask=data['mask_m'])
        summary_g = self._addMetadata(summary_g, data, local=False)

        # ---------------------------------------------------------------------
        # global loss
        targets_g = self._targets(data, local=False)
        mask_g = self._masks(data, local=False)
        targets_g = self._preprocess(targets_g, local=False)
        loss_g = self.posterior_g.loss(
            targets_g, context=summary_g, mask=mask_g)

        # global sampling
        if n_samples > 0:
            samples_g, log_prob_g = self.posterior_g.sample(
                n_samples, context=summary_g, mask=mask_g)
            proposed['global'] = {'samples': samples_g, 'log_prob': log_prob_g}

        # ---------------------------------------------------------------------
        # local loss
        targets_l = self._targets(data, local=True)
        targets_l = self._preprocess(targets_l, local=True)
        mask_l = self._masks(data, local=True)
        targets_l, mask_l = self._expandLocal(targets_l, mask_l, n_samples)
        context_l = self._localContext(targets_g, summary_l, proposed)
        loss_l = self.posterior_l.loss(
            targets_l, context=context_l, mask=mask_l
        )

        # local sampling
        if n_samples > 0:
            samples_l, log_prob_l = self.posterior_l.sample(
                1, context=context_l, mask=mask_l)
            samples_l, log_prob_l = samples_l.squeeze(-2), log_prob_l.squeeze(-1)
            proposed['local'] = {'samples': samples_l, 'log_prob': log_prob_l}
            loss_l = loss_l.mean(-1) # average over samples

        # ---------------------------------------------------------------------
        proposed = self._postprocess(proposed)
        loss = loss_g + loss_l.sum(-1) / data['m']
        return loss, proposed





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

