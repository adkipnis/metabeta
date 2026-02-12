import torch
from torch import nn
from torch.nn import functional as F

from metabeta.models.transformers import SetTransformer
from metabeta.models.normalizingflows import CouplingFlow
from metabeta.utils.regularization import maskedInverseSoftplus, maskedSoftplus
from metabeta.utils.config import ApproximatorConfig
from metabeta.utils.evaluation import Proposal


class Approximator(nn.Module):
    def __init__(self, cfg: ApproximatorConfig):
        super().__init__()
        self.cfg = cfg
        self.build()

    @property
    def d_ffx(self) -> int:
        return self.cfg.d_ffx

    @property
    def d_rfx(self) -> int:
        return self.cfg.d_rfx

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
        if p_cfg.type == 'coupling':
            Posterior = CouplingFlow
        else:
            raise NotImplementedError('unknown posterior type')
        d_var = 1 + d_rfx # number of variance params
        d_prior = 2 * d_ffx + d_var # number of prior params
        d_context_g = s_cfg.d_output + d_prior + 2 # global summary, prior, n_groups, n_total
        d_context_l = d_ffx + d_var + d_input_g # global params, local summaries
        self.posterior_g = Posterior(d_ffx+d_var, d_context_g, **p_cfg.to_dict())
        self.posterior_l = Posterior(max(d_rfx, 2), d_context_l, **p_cfg.to_dict())
 
    def compile(self) -> None:
        self.summarizer_l = torch.compile(self.summarizer_l)
        self.summarizer_g = torch.compile(self.summarizer_g)
        self.posterior_g = torch.compile(self.posterior_g)
        self.posterior_l = torch.compile(self.posterior_l)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
 
    @property
    def n_params(self) -> int:
        param_counts = (p.numel() for p in self.parameters() if p.requires_grad)
        return sum(param_counts)

    def _inputs(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        ''' get summarizer inputs '''
        d, q = self.d_ffx, self.d_rfx
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
            if self.d_rfx == 1: # handle 1D local params for flow
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
            # TODO: optionally dampen prior params, depending on how large they get
            out += [n_total, n_groups, nu_ffx, tau_ffx, tau_rfx, tau_eps]
        return torch.cat(out, dim=-1)

    def _localContext(
        self,
        local_summary: torch.Tensor,
        global_params: torch.Tensor,
    ) -> torch.Tensor:
        b, m, _ = local_summary.shape
        ndim = global_params.dim()
        if ndim == 3: # samples
            s = global_params.shape[-2]
            global_params = global_params.unsqueeze(1).expand(b, m, s, -1)
            local_summary = local_summary.unsqueeze(2).expand(b, m, s, -1)
        elif ndim == 2: # ground truth
            global_params = global_params.unsqueeze(1).expand(b, m, -1)
        else:
            raise ValueError(f'global_params has {ndim} dims, expected 2 or 3')
        return torch.cat([local_summary, global_params], dim=-1)

    def _preprocess(self, targets: torch.Tensor, local: bool = False) -> torch.Tensor:
        ''' constrain target parameters '''
        if local:
            return targets
        targets = targets.clone()
        d = self.d_ffx

        # project from R+ to R
        sigmas = targets[..., d:]
        sigmas = maskedInverseSoftplus(sigmas)
        targets[..., d:] = sigmas
        return targets

    def _postprocess(self, proposed: dict[str, dict[str, torch.Tensor]]):
        ''' reverse _preprocess for samples '''
        d = self.d_ffx
        q = self.d_rfx

        # local postprocessing
        if 'local' in proposed:
            samples_l = proposed['local']['samples']
            if q == 1 and samples_l.shape[-1] > 1:
                proposed['local']['samples'] = samples_l[..., :1]

        # global postprocessing
        if 'global' in proposed:
            samples = proposed['global']['samples'].clone()
            sigmas = samples[..., d:]
            sigmas = maskedSoftplus(sigmas)
            samples[..., d:] = sigmas
            proposed['global']['samples'] = samples
        return Proposal(proposed)

    def _summarize(
            self, data: dict[str, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._inputs(data)
        summary_l = self.summarizer_l(inputs, mask=data['mask_n'])
        summary_l = self._addMetadata(summary_l, data, local=True)
        summary_g = self.summarizer_g(summary_l, mask=data['mask_m'])
        summary_g = self._addMetadata(summary_g, data, local=False)
        return summary_g, summary_l

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ''' training method: learn conditional forward pass '''
        loss = {}

        # summaries
        summary_g, summary_l = self._summarize(data)

        # global posterior
        targets_g = self._targets(data, local=False)
        mask_g = self._masks(data, local=False)
        targets_g = self._preprocess(targets_g, local=False)
        loss['global'] = self.posterior_g.loss(
            targets_g, context=summary_g, mask=mask_g)

        # local posterior
        targets_l = self._targets(data, local=True)
        targets_l = self._preprocess(targets_l, local=True)
        mask_l = self._masks(data, local=True)
        context_l = self._localContext(summary_l, targets_g)
        loss['local'] = self.posterior_l.loss(
            targets_l, context=context_l, mask=mask_l
        )

        # total loss
        loss['total'] = loss['global'] + loss['local'].sum(-1) / data['m']
        return loss

    @torch.no_grad()
    def estimate(
            self, data: dict[str, torch.Tensor], n_samples: int = 100,
    ) -> Proposal:
        ''' inference method: sample and apply conditional backward pass '''
        assert n_samples > 0, 'n_samples must be positive'
        proposed = {}
        summary_g, summary_l = self._summarize(data)

        # global posterior
        mask_g = self._masks(data, local=False)
        samples_g, log_prob_g = self.posterior_g.sample(
            n_samples, context=summary_g, mask=mask_g)
        proposed['global'] = {'samples': samples_g, 'log_prob': log_prob_g}

        # local posterior
        mask_l = self._masks(data, local=True)
        b, m, q = mask_l.shape
        mask_l = mask_l.unsqueeze(-2).expand(b, m, n_samples, q)
        context_l = self._localContext(summary_l, samples_g)
        samples_l, log_prob_l = self.posterior_l.sample(
            1, context=context_l, mask=mask_l)
        samples_l, log_prob_l = samples_l.squeeze(-2), log_prob_l.squeeze(-1)
        proposed['local'] = {'samples': samples_l, 'log_prob': log_prob_l}

        # postprocess samples
        proposal = self._postprocess(proposed)
        return proposal

# =============================================================================
if __name__ == '__main__':
    from pathlib import Path
    from metabeta.utils.dataloader import Dataloader
    from metabeta.utils.config import dataFromYaml, modelFromYaml
    torch.manual_seed(0)

    # load toy data
    data_cfg_path = Path('..', 'simulation', 'configs', 'toy.yaml')
    data_fname = dataFromYaml(data_cfg_path, 'test')
    data_path = Path('..', 'outputs', 'data', data_fname)
    dl = Dataloader(data_path, batch_size=8)
    batch = next(iter(dl))

    # init toy model
    model_cfg_path = Path('..', 'models', 'configs', 'toy.yaml')
    model_cfg = modelFromYaml(model_cfg_path, dl.dataset.d, dl.dataset.q) # type: ignore
    model = Approximator(model_cfg)
    # model.compile()

    loss = model.forward(batch)
    proposal = model.estimate(batch, n_samples=100)
