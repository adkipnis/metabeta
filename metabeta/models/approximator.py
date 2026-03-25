import torch
from torch import nn
from torch.nn import functional as F

from metabeta.models.transformers import SetTransformer
from metabeta.models.normalizingflows import CouplingFlow
from metabeta.utils.regularization import getConstrainers
from metabeta.utils.config import ApproximatorConfig
from metabeta.utils.evaluation import Proposal, joinGlobals

constrainSigma, unconstrainSigma, logDetJacobian = getConstrainers(method='softplus')


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
        s_cfg = self.cfg.summarizer
        p_cfg = self.cfg.posterior
        d_ffx = self.d_ffx
        d_rfx = self.d_rfx
        d_var = 1 + d_rfx   # number of variance params

        # --- context
        # global context
        d_counts = 2   # n_groups, n_total
        d_prior = 2 * d_ffx + d_var + 1   # nu_ffx, tau_ffx, tau_sigma, tau_eps, eta_rfx
        d_stats = d_ffx + 2   # beta_ols, sigma_eps_ols, sigma_rfx0_ols
        d_meta_g = d_counts + d_prior + d_stats
        d_context_g = s_cfg.d_output + d_meta_g   # global summary + metadata
        # local context
        d_meta_l = 2   # n_obs + eta_rfx (fed to global summarizer)
        d_context_l = (
            d_ffx + d_var + s_cfg.d_output + d_meta_l
        )   # global samples + local summaries + metadata

        # --- summarizers
        if s_cfg.type == 'set-transformer':
            Summarizer = SetTransformer
        else:
            raise NotImplementedError('unknown summarizer type')
        d_input_l = 1 + (d_ffx - 1) + (d_rfx - 1)
        d_input_g = s_cfg.d_output + d_meta_l
        self.summarizer_l = Summarizer(d_input=d_input_l, **s_cfg.to_dict())
        self.summarizer_g = Summarizer(d_input=d_input_g, **s_cfg.to_dict())

        # --- posteriors
        if p_cfg.type == 'coupling':
            Posterior = CouplingFlow
        else:
            raise NotImplementedError('unknown posterior type')
        self.posterior_g = Posterior(d_ffx + d_var, d_context_g, **p_cfg.to_dict())
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
        """get summarizer inputs"""
        d, q = self.d_ffx, self.d_rfx
        y = data['y'].unsqueeze(-1)
        X = data['X'][..., 1:d]
        Z = data['Z'][..., 1:q]
        return torch.cat([y, X, Z], dim=-1)

    def _targets(self, data: dict[str, torch.Tensor], local: bool = False) -> torch.Tensor:
        """get posterior targets"""
        if local:
            targets = data['rfx']
            if self.d_rfx == 1:   # handle 1D local params for flow
                targets = F.pad(targets, (0, 1))
            return targets
        return joinGlobals(data)

    def _masks(self, data: dict[str, torch.Tensor], local: bool = False) -> torch.Tensor:
        """get masks for the posterior targets"""
        if local:
            mask_q = data['mask_q']
            if mask_q.shape[-1] == 1:   # handle 1D local params for flow
                mask_q = F.pad(mask_q, (0, 1))
            return data['mask_m'].unsqueeze(-1) & mask_q.unsqueeze(-2)
        masks = [
            data['mask_d'],
            data['mask_q'],
            torch.ones_like(data['mask_d'][..., 0:1]),
        ]
        return torch.cat(masks, dim=-1)

    @staticmethod
    def _dataStatistics(
        data: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute sufficient statistics from grouped data.

        Returns:
            beta_ols:       (B, d)  pooled OLS fixed-effect estimates
            sigma_eps_ols:  (B, 1)  pooled residual SD
            sigma_rfx_ols:  (B, 1)  between-group SD of group means (random intercept proxy)
        """
        X = data['X']                           # (B, m, n, d)
        y = data['y']                           # (B, m, n)
        mask = data['mask_n'].float()           # (B, m, n)
        mask_m = data['mask_m'].float()         # (B, m)
        ns = data['ns'].clamp(min=1).float()    # (B, m)
        n_total = data['n'].float()             # (B,)

        # --- pooled OLS: beta = (X'X)^{-1} X'y
        Xm = X * mask.unsqueeze(-1)
        ym = y * mask
        XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)   # (B, d, d)
        Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)      # (B, d)
        beta_ols = torch.linalg.lstsq(XtX, Xty).solution   # (B, d)

        # --- residual SD
        yhat = torch.einsum('bmnd,bd->bmn', X, beta_ols)   # (B, m, n)
        resid = (y - yhat) * mask                         # (B, m, n)
        ss_resid = (resid.square()).sum(dim=(1, 2))       # (B,)
        df = (n_total - X.shape[-1]).clamp(min=1)         # (B,)
        sigma_eps_ols = (ss_resid / df).sqrt().unsqueeze(-1)   # (B, 1)

        # --- between-group SD of group means (random intercept proxy)
        group_means = (ym).sum(dim=2) / ns                # (B, m)
        m_valid = mask_m.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        grand_mean = (group_means * mask_m).sum(dim=1, keepdim=True) / m_valid
        sq_dev = ((group_means - grand_mean).square() * mask_m).sum(dim=1, keepdim=True)
        sigma_rfx_ols = (sq_dev / m_valid.clamp(min=2)).sqrt()  # (B, 1)

        return beta_ols, sigma_eps_ols, sigma_rfx_ols

    def _addMetadata(
        self,
        summary: torch.Tensor,
        data: dict[str, torch.Tensor],
        local: bool = False,
    ) -> torch.Tensor:
        """append summary with selected metadata"""
        out = [summary]
        if local:
            n_obs = data['ns'].unsqueeze(-1).float().sqrt() / 10
            eta_rfx = data['eta_rfx'].unsqueeze(-1).expand(-1, summary.shape[1])
            out += [n_obs, eta_rfx.unsqueeze(-1)]
        else:
            n_total = data['n'].unsqueeze(-1).float().sqrt() / 10
            n_groups = data['m'].unsqueeze(-1).float().sqrt() / 10
            beta_ols, sigma_eps_ols, sigma_rfx_ols = self._dataStatistics(data)
            nu_ffx = data['nu_ffx'].clone()
            tau_ffx = data['tau_ffx'].clone()
            tau_rfx = data['tau_rfx'].clone()
            tau_eps = data['tau_eps'].clone().unsqueeze(-1)
            eta_rfx = data['eta_rfx'].clone().unsqueeze(-1)
            out += [
                n_total,
                n_groups,
                beta_ols,
                sigma_eps_ols,
                sigma_rfx_ols,
                nu_ffx,
                tau_ffx,
                tau_rfx,
                tau_eps,
                eta_rfx,
            ]
        return torch.cat(out, dim=-1)

    def _localContext(
        self,
        local_summary: torch.Tensor,
        global_params: torch.Tensor,
    ) -> torch.Tensor:
        b, m, _ = local_summary.shape
        ndim = global_params.dim()
        if ndim == 3:   # samples
            s = global_params.shape[-2]
            global_params = global_params.unsqueeze(1).expand(b, m, s, -1)
            local_summary = local_summary.unsqueeze(2).expand(b, m, s, -1)
        elif ndim == 2:   # ground truth
            global_params = global_params.unsqueeze(1).expand(b, m, -1)
        else:
            raise ValueError(f'global_params has {ndim} dims, expected 2 or 3')
        return torch.cat([local_summary, global_params], dim=-1)

    def _preprocess(self, targets: torch.Tensor, local: bool = False) -> torch.Tensor:
        """constrain target parameters"""
        if local:
            return targets
        targets = targets.clone()
        d = self.d_ffx

        # project from R+ to R
        sigmas = targets[..., d:]
        log_sigmas = unconstrainSigma(sigmas)
        targets[..., d:] = log_sigmas
        return targets

    def _postprocess(self, proposed: dict[str, dict[str, torch.Tensor]]):
        """reverse _preprocess for samples"""
        d = self.d_ffx
        q = self.d_rfx

        # local postprocessing
        if 'local' in proposed:
            samples_l = proposed['local']['samples']
            if q == 1 and samples_l.shape[-1] > 1:
                proposed['local']['samples'] = samples_l[..., :1]

        # global postprocessing
        if 'global' in proposed:
            samples = proposed['global']['samples']
            log_sigmas = samples[..., d:].clone()
            log_det = logDetJacobian(log_sigmas)
            log_prob_g = proposed['global']['log_prob'] - log_det.sum(dim=-1)
            proposed['global']['log_prob'] = log_prob_g
            sigmas = constrainSigma(log_sigmas)
            samples_out = torch.cat([samples[..., :d], sigmas], dim=-1)
            proposed['global']['samples'] = samples_out
        return Proposal(proposed)

    def summarize(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._inputs(data)
        summary_l = self.summarizer_l(inputs, mask=data['mask_n'])
        summary_l = self._addMetadata(summary_l, data, local=True)
        summary_g = self.summarizer_g(summary_l, mask=data['mask_m'])
        summary_g = self._addMetadata(summary_g, data, local=False)
        return summary_g, summary_l

    def forward(
        self,
        data: dict[str, torch.Tensor],
        summaries: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """training method: learn conditional forward pass"""
        log_probs = {}

        # summaries
        if summaries is None:
            summary_g, summary_l = self.summarize(data)
        else:
            summary_g, summary_l = summaries

        # global posterior
        targets_g = self._targets(data, local=False)
        mask_g = self._masks(data, local=False)
        targets_g = self._preprocess(targets_g, local=False)
        log_probs['global'] = self.posterior_g.logProb(  # type: ignore
            targets_g, context=summary_g, mask=mask_g
        )

        # local posterior
        targets_l = self._targets(data, local=True)
        targets_l = self._preprocess(targets_l, local=True)
        mask_l = self._masks(data, local=True)
        context_l = self._localContext(summary_l, targets_g)
        log_probs['local'] = self.posterior_l.logProb(  # type: ignore
            targets_l, context=context_l, mask=mask_l
        )
        return log_probs

    def backward(
        self,
        data: dict[str, torch.Tensor],
        summaries: tuple[torch.Tensor, torch.Tensor] | None = None,
        n_samples: int = 1,
    ) -> Proposal:
        """inference method: sample and apply conditional backward pass"""
        assert n_samples > 0, 'n_samples must be positive'
        proposed = {}

        # summaries
        if summaries is None:
            summary_g, summary_l = self.summarize(data)
        else:
            summary_g, summary_l = summaries

        # global posterior
        mask_g = self._masks(data, local=False)
        samples_g, log_prob_g = self.posterior_g.sample(  # type: ignore
            n_samples, context=summary_g, mask=mask_g
        )
        proposed['global'] = {'samples': samples_g, 'log_prob': log_prob_g}

        # local posterior
        mask_l = self._masks(data, local=True)
        b, m, q = mask_l.shape
        mask_l = mask_l.unsqueeze(-2).expand(b, m, n_samples, q)
        context_l = self._localContext(summary_l, samples_g)
        samples_l, log_prob_l = self.posterior_l.sample(  # type: ignore
            1, context=context_l, mask=mask_l
        )
        samples_l, log_prob_l = samples_l.squeeze(-2), log_prob_l.squeeze(-1)
        proposed['local'] = {'samples': samples_l, 'log_prob': log_prob_l}

        # postprocess samples
        proposal = self._postprocess(proposed)
        return proposal

    @torch.inference_mode()
    def estimate(self, data, summaries=None, n_samples=1) -> Proposal:
        return self.backward(data, summaries=summaries, n_samples=n_samples)


# =============================================================================
if __name__ == '__main__':
    from pathlib import Path
    from metabeta.utils.dataloader import Dataloader
    from metabeta.utils.config import dataFromYaml, modelFromYaml

    DIR = Path(__file__).resolve().parent
    torch.manual_seed(0)

    # load toy data
    data_cfg_path = Path(DIR, '..', 'simulation', 'configs', 'small.yaml')
    data_fname = dataFromYaml(data_cfg_path, 'test')
    data_path = Path(DIR, '..', 'outputs', 'data', data_fname)
    dl = Dataloader(data_path, batch_size=8)
    batch = next(iter(dl))

    # init toy model
    model_cfg_path = Path(DIR, '..', 'models', 'configs', 'toy.yaml')
    model_cfg = modelFromYaml(model_cfg_path, dl.dataset.d, dl.dataset.q)   # type: ignore
    model = Approximator(model_cfg)
    # model.compile()

    loss = model.forward(batch)
    proposal = model.estimate(batch, n_samples=100)
