import torch
from torch import nn
from torch.nn import functional as F

from metabeta.models.transformers import SetTransformer
from metabeta.models.normalizingflows import CouplingFlow
from metabeta.utils.regularization import (
    getConstrainers,
    corrToLower,
    corrToUnconstrained,
)
from metabeta.utils.config import ApproximatorConfig, SummarizerConfig, PosteriorConfig
from metabeta.utils.evaluation import Proposal, joinGlobals
from metabeta.utils.glmm import glmm
from metabeta.utils.families import (
    FFX_FAMILIES,
    SIGMA_FAMILIES,
    FamilyEncoder,
    hasSigmaEps,
)

constrainSigma, unconstrainSigma, logDetJacobian = getConstrainers(method='softplus')
CLAMP = 20.0


def _buildSummarizer(cfg: SummarizerConfig, d_input: int) -> nn.Module:
    if cfg.type == 'set-transformer':
        return SetTransformer(d_input=d_input, **cfg.to_dict())
    raise NotImplementedError(f'unknown summarizer type: {cfg.type}')


def _buildPosterior(cfg: PosteriorConfig, d_target: int, d_context: int) -> nn.Module:
    if cfg.type == 'coupling':
        return CouplingFlow(d_target=d_target, d_context=d_context, **cfg.to_dict())
    raise NotImplementedError(f'unknown posterior type: {cfg.type}')


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

    @property
    def likelihood_family(self) -> int:
        return self.cfg.likelihood_family

    @property
    def has_sigma_eps(self) -> bool:
        return hasSigmaEps(self.likelihood_family)

    @property
    def d_corr(self) -> int:
        return self.cfg.d_corr

    @property
    def analytical_context(self) -> str:
        return self.cfg.analytical_context

    def _analyticsGlobalDim(self) -> int:
        """Dimension added to global context by GLMM statistics.

        beta_est (d_ffx) + sigma_rfx (d_rfx) + eps/phi (1) + corr (d_corr).
        """
        if self.analytical_context == 'none':
            return 0
        return self.d_ffx + self.d_rfx + self.d_corr + 1

    def _analyticsLocalDim(self) -> int:
        """Dimension added to local context by GLMM stats.

        blup_est (d_rfx) + blup_std (d_rfx) + lambda_g (d_rfx) + resid_g (1)
        """
        if self.analytical_context == 'none':
            return 0
        return 3 * self.d_rfx + 1

    def build(self) -> None:
        d_ffx = self.d_ffx
        d_rfx = self.d_rfx
        d_sigma_eps = 1 if self.has_sigma_eps else 0
        d_var = d_sigma_eps + d_rfx  # number of variance params

        # --- family encoder
        if self.has_sigma_eps:
            n_families = (len(FFX_FAMILIES), len(SIGMA_FAMILIES), len(SIGMA_FAMILIES))
        else:
            n_families = (len(FFX_FAMILIES), len(SIGMA_FAMILIES))
        self.family_encoder = FamilyEncoder(n_families, embed_dim=None)

        # --- summarizers
        # local: y + X[1:d] + Z[1:q] per observation
        d_input_l = 1 + (d_ffx - 1) + (d_rfx - 1)
        self.summarizer_l = _buildSummarizer(self.cfg.summarizer_l, d_input_l)
        # global: local summaries + local metadata, aggregated across groups
        d_meta_l = 2 + self._analyticsLocalDim()   # n_obs + eta_rfx
        d_input_g = self.cfg.summarizer_l.d_output + d_meta_l
        self.summarizer_g = _buildSummarizer(self.cfg.summarizer_g, d_input_g)

        # --- posteriors
        # global: fixed effects + variance params conditioned on global summary + metadata
        d_prior = (
            2 * d_ffx + d_rfx + d_sigma_eps + 1
        )  # nu_ffx, tau_ffx, tau_rfx, [tau_eps], eta_rfx
        d_meta_g = 2 + d_prior + self.family_encoder.d_output + self._analyticsGlobalDim()
        d_context_g = self.cfg.summarizer_g.d_output + d_meta_g
        d_target_g = d_ffx + d_var + self.d_corr
        self.posterior_g = _buildPosterior(self.cfg.posterior_g, d_target_g, d_context_g)
        # local: random effects conditioned on global samples + local summary + local metadata
        # global params now include d_corr extra dims that the local flow also receives as context
        d_context_l = d_ffx + d_var + self.d_corr + self.cfg.summarizer_l.d_output + d_meta_l
        self.posterior_l = _buildPosterior(self.cfg.posterior_l, max(d_rfx, 2), d_context_l)

    def compile(self) -> None:
        # Only compile the posteriors: Set Transformers use variable-length masked sequences
        # that trigger an Inductor tiling assertion bug (pytorch/pytorch#139438).
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
        # d, q = self.d_ffx, self.d_rfx
        y = data['y'].unsqueeze(-1)
        X = data['X'][..., 1:]
        Z = data['Z'][..., 1:]
        return torch.cat([y, X, Z], dim=-1)

    def _targets(self, data: dict[str, torch.Tensor], local: bool = False) -> torch.Tensor:
        """get posterior targets"""
        if local:
            targets = data['rfx'][..., : self.d_rfx]
            if self.d_rfx == 1:  # handle 1D local params for flow
                targets = F.pad(targets, (0, 1))
            return targets
        # Slice to model dims before joining — data may carry max-dim columns.
        sliced = {
            **data,
            'ffx': data['ffx'][..., : self.d_ffx],
            'sigma_rfx': data['sigma_rfx'][..., : self.d_rfx],
        }
        targets = joinGlobals(sliced)
        if self.d_corr > 0:
            corr = data['corr_rfx'][..., : self.d_rfx, : self.d_rfx]  # (B, q, q)
            z_corr = corrToUnconstrained(corr)  # (B, d_corr)
            targets = torch.cat([targets, z_corr], dim=-1)
        return targets

    def _masks(self, data: dict[str, torch.Tensor], local: bool = False) -> torch.Tensor:
        """get masks for the posterior targets"""
        if local:
            mask_q = data['mask_q'][..., : self.d_rfx]
            if mask_q.shape[-1] == 1:  # handle 1D local params for flow
                mask_q = F.pad(mask_q, (0, 1))
            return data['mask_m'].unsqueeze(-1) & mask_q.unsqueeze(-2)
        masks = [data['mask_d'][..., : self.d_ffx], data['mask_q'][..., : self.d_rfx]]
        if self.has_sigma_eps:
            masks.append(torch.ones_like(data['mask_d'][..., 0:1]))
        if self.d_corr > 0:
            # Correlation is active only when the second rfx dim is active (q >= 2).
            corr_mask = data['mask_q'][..., 1:2].expand(-1, self.d_corr)
            masks.append(corr_mask)
        return torch.cat(masks, dim=-1)

    def _dataStatistics(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute sufficient statistics: GLS/GLMM β̂, σ̂, BLUPs."""
        Zm = data['Z'][..., : self.d_rfx]
        return glmm(
            data['X'],
            data['y'],
            Zm,
            data['mask_n'].float(),
            data['mask_m'].float(),
            data['ns'].clamp(min=1).float(),
            data['n'].float(),
            likelihood_family=self.likelihood_family,
            eta_rfx=data.get('eta_rfx'),
        )

    def _addMetadata(
        self,
        summary: torch.Tensor,
        data: dict[str, torch.Tensor],
        local: bool = False,
        stats: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """append summary with selected metadata"""
        out = [summary]
        ctx = self.analytical_context
        if local:
            # counts
            n_obs = data['ns'].unsqueeze(-1).float().sqrt() / 10  # (B, m, 1)

            # correlation prior
            eta_rfx = (
                data['eta_rfx'].unsqueeze(-1).expand(-1, summary.shape[1]).unsqueeze(-1)
            )   # (B, m, 1)
            out += [n_obs, eta_rfx]

            # point estimates
            if ctx != 'none' and stats is not None:
                blup_est = stats['blup_est'].clamp(-CLAMP, CLAMP)   # (B, m, q)
                blup_std = stats['blup_var'].clamp(min=0.0).sqrt().clamp(max=CLAMP)  # (B, m, q)
                sigma_rfx_sq = stats['sigma_rfx_est'].unsqueeze(-2) ** 2  # (B, 1, q)
                lambda_g = (1.0 - stats['blup_var'] / (sigma_rfx_sq + 1e-8)).clamp(0.0, 1.0)   # (B, m, q)
                resid_g = stats['resid_g'].clamp(-CLAMP, CLAMP)   # (B, m, 1)
                out += [blup_est, blup_std, lambda_g, resid_g]
        else:
            # counts
            n_total = data['n'].unsqueeze(-1).float().sqrt() / 10
            n_groups = data['m'].unsqueeze(-1).float().sqrt() / 10
            out += [n_total, n_groups]

            # prior parameters and families
            nu_ffx = data['nu_ffx'].clone()
            tau_ffx = data['tau_ffx'].clone()
            tau_rfx = data['tau_rfx'][..., : self.d_rfx].clone()
            eta_rfx = data['eta_rfx'].clone().unsqueeze(-1)
            if self.has_sigma_eps:
                tau_eps = data['tau_eps'].clone().unsqueeze(-1)
                out.append(tau_eps)
            out += [nu_ffx, tau_ffx, tau_rfx, eta_rfx]

            # prior families
            families = [data['family_ffx'], data['family_sigma_rfx']]
            if self.has_sigma_eps:
                families.append(data['family_sigma_eps'])
            family_enc = self.family_encoder(families)
            out.append(family_enc)

            # point estimates
            if ctx != 'none' and stats is not None:
                beta_est = stats['beta_est'].clamp(-CLAMP, CLAMP)   # (B, d)
                sigma_rfx_est = stats['sigma_rfx_est'].clamp(max=CLAMP)   # (B, q)
                out += [beta_est, sigma_rfx_est]
                if self.has_sigma_eps:
                    out.append(stats['sigma_eps_est'].clamp(max=CLAMP))   # (B, 1)
                else:
                    # phi_pearson fills the reserved +1 slot in _analyticsGlobalDim for GLMM
                    out.append(stats['phi_pearson'].clamp(max=CLAMP).unsqueeze(-1))  # (B, 1)
                if self.d_corr > 0:
                    Psi = stats['Psi'] if 'Psi' in stats else stats['Psi_lap']  # (B, q, q)
                    std = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()  # (B, q)
                    psi_corr = (Psi / (std.unsqueeze(-1) * std.unsqueeze(-2))).clamp(-1, 1)
                    out.append(corrToLower(psi_corr))
        return torch.cat(out, dim=-1)

    def _localContext(
        self,
        local_summary: torch.Tensor,
        global_params: torch.Tensor,
    ) -> torch.Tensor:
        b, m, _ = local_summary.shape
        ndim = global_params.dim()
        if ndim == 3:  # samples
            s = global_params.shape[-2]
            global_params = global_params.unsqueeze(1).expand(b, m, s, -1)
            local_summary = local_summary.unsqueeze(2).expand(b, m, s, -1)
        elif ndim == 2:  # ground truth
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
        q_var = self.d_rfx + (1 if self.has_sigma_eps else 0)

        # project the sigma block [d : d+q_var] from R+ to R
        # correlation scalars [d+q_var :] are already unconstrained — leave them
        sigmas = targets[..., d : d + q_var]
        targets[..., d : d + q_var] = unconstrainSigma(sigmas)
        return targets

    def _postprocess(self, proposed: dict[str, dict[str, torch.Tensor]]):
        """reverse _preprocess for samples"""
        d = self.d_ffx
        q = self.d_rfx
        d_corr = self.d_corr
        q_var = q + (1 if self.has_sigma_eps else 0)

        # local postprocessing
        if 'local' in proposed:
            samples_l = proposed['local']['samples']
            if q == 1 and samples_l.shape[-1] > 1:
                proposed['local']['samples'] = samples_l[..., :1]

        # global postprocessing
        if 'global' in proposed:
            samples = proposed['global']['samples']
            # only the sigma block [d : d+q_var] is log-scale; corr_z is already unconstrained
            log_sigmas = samples[..., d : d + q_var].clone()
            log_det = logDetJacobian(log_sigmas)
            log_prob_g = proposed['global']['log_prob'] - log_det.sum(dim=-1)
            proposed['global']['log_prob'] = log_prob_g
            sigmas = constrainSigma(log_sigmas)
            if d_corr > 0:
                corr_z = samples[..., d + q_var :]
                samples_out = torch.cat([samples[..., :d], sigmas, corr_z], dim=-1)
            else:
                samples_out = torch.cat([samples[..., :d], sigmas], dim=-1)
            proposed['global']['samples'] = samples_out
        return Proposal(proposed, has_sigma_eps=self.has_sigma_eps, d_corr=d_corr)

    def summarize(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._inputs(data)
        summary_l = self.summarizer_l(inputs, mask=data['mask_n'])
        if self.analytical_context != 'none':
            stats = self._dataStatistics(data)  # compute once, share across both contexts
        else:
            stats = None
        summary_l = self._addMetadata(summary_l, data, local=True, stats=stats)
        summary_g = self.summarizer_g(summary_l, mask=data['mask_m'])
        summary_g = self._addMetadata(summary_g, data, local=False, stats=stats)
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

    @torch.no_grad()
    def estimate(
        self,
        data,
        summaries=None,
        n_samples=1,
    ) -> Proposal:
        return self.backward(data, summaries=summaries, n_samples=n_samples)


# =============================================================================
if __name__ == '__main__':
    from pathlib import Path
    from metabeta.utils.dataloader import Dataloader
    from metabeta.utils.config import dataFromYaml, modelFromYaml

    DIR = Path(__file__).resolve().parent
    torch.manual_seed(0)

    # load toy data
    data_cfg_path = Path(DIR, '..', 'simulation', 'configs', 'small-n-sampled.yaml')
    data_fname = dataFromYaml(data_cfg_path, 'test')
    data_path = Path(DIR, '..', 'outputs', 'data', data_fname)
    dl = Dataloader(data_path, batch_size=8)
    batch = next(iter(dl))

    # init toy model
    model_cfg_path = Path(DIR, '..', 'models', 'configs', 'toy.yaml')
    model_cfg = modelFromYaml(model_cfg_path, dl.dataset.d, dl.dataset.q)  # type: ignore
    model = Approximator(model_cfg)
    # model.compile()

    loss = model.forward(batch)
    proposal = model.estimate(batch, n_samples=100)
