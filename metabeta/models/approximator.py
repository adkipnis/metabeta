import torch
from torch import nn
from torch.nn import functional as F

from metabeta.models.transformers import SetTransformer
from metabeta.models.normalizingflows import CouplingFlow
from metabeta.utils.regularization import (
    getConstrainers,
    corrToLower,
    corrToUnconstrained,
    corrLowerToUnconstrained,
    logDetJacobianCorr,
    unconstrainedToCholesky,
)
from metabeta.utils.config import ApproximatorConfig, SummarizerConfig, PosteriorConfig
from metabeta.utils.evaluation import Proposal, joinGlobals
from metabeta.utils.glmm import glmm, analyticalBLUPContext
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
        beta_est (d_ffx) + log_beta_var (d_ffx, normal only) + sigma_rfx (d_rfx) + eps/phi (1) + corr (d_corr).
        """
        if not self.analytical_context:
            return 0
        d_beta_var = self.d_ffx if self.has_sigma_eps else 0
        return self.d_ffx + d_beta_var + self.d_rfx + self.d_corr + 1

    def _analyticsLocalDim(self) -> int:
        """Dimension added to local context by GLMM stats.
        blup_est (d_rfx) + blup_std (d_rfx) + lambda_g (d_rfx)
        """
        if not self.analytical_context:
            return 0
        return 3 * self.d_rfx

    def _crossBlupDim(self) -> int:
        """Extra dims added to the global-transformer input only: cross-BLUP (d_corr).
        Separate from _analyticsLocalDim because cross-BLUP is not in the local-flow context.
        """
        if not self.analytical_context:
            return 0
        return self.d_corr

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
        d_input_g = self.cfg.summarizer_l.d_output + d_meta_l + self._crossBlupDim()
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
            r_corr = corrToLower(corr)  # (B, d_corr) — constrained lower triangle
            targets = torch.cat([targets, r_corr], dim=-1)
        return targets

    def _masks(self, data: dict[str, torch.Tensor], local: bool = False) -> torch.Tensor:
        """get masks for the posterior targets"""
        if local:
            mask_l = data['mask_mq'][..., : self.d_rfx]
            if mask_l.shape[-1] == 1:  # handle 1D local params for flow
                mask_l = F.pad(mask_l, (0, 1))
            return mask_l
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
        if local:
            # counts
            n_obs = data['ns'].unsqueeze(-1).float().sqrt() / 10  # (B, m, 1)

            # correlation prior
            eta_rfx = (
                data['eta_rfx'].unsqueeze(-1).expand(-1, summary.shape[1]).unsqueeze(-1)
            )   # (B, m, 1)
            out += [n_obs, eta_rfx]

            # point estimates
            if stats is not None:
                blup_est = stats['blup_est'].clamp(-CLAMP, CLAMP)   # (B, m, q)
                blup_std = stats['blup_var'].clamp(min=0.0).sqrt().clamp(max=CLAMP)  # (B, m, q)
                sigma_rfx_sq = stats['sigma_rfx_est'].unsqueeze(-2) ** 2  # (B, 1, q)
                lambda_g = (1.0 - stats['blup_var'] / (sigma_rfx_sq + 1e-8)).clamp(
                    0.0, 1.0
                )   # (B, m, q)
                # resid_g = stats['resid_g'].clamp(-CLAMP, CLAMP)   # (B, m, 1)
                out += [blup_est, blup_std, lambda_g]
                if self.d_corr > 0:
                    # Per-group normalized cross-BLUP: b_i*b_j/(σ_i*σ_j) for each pair i<j.
                    # Gives the global set transformer raw per-group correlation evidence
                    # so it can learn to aggregate (rather than only seeing pre-averaged psi_corr).
                    sr = stats['sigma_rfx_est'].unsqueeze(-2).clamp(min=1e-8)  # (B, 1, q)
                    blup_norm = blup_est / sr                                  # (B, m, q)
                    q = blup_norm.shape[-1]
                    cross = torch.stack(
                        [blup_norm[..., j] * blup_norm[..., i] for i in range(q) for j in range(i)],
                        dim=-1,
                    ).clamp(-CLAMP, CLAMP)                                     # (B, m, d_corr)
                    out.append(cross)
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
            if stats is not None:
                beta_est = stats['beta_est'].clamp(-CLAMP, CLAMP)   # (B, d)
                sigma_rfx_est = stats['sigma_rfx_est'].clamp(max=CLAMP)   # (B, q)
                out += [beta_est, sigma_rfx_est]
                if self.has_sigma_eps and 'beta_var' in stats:
                    out.append(stats['beta_var'].clamp(min=1e-8).log())   # (B, d)
                if self.has_sigma_eps:
                    out.append(stats['sigma_eps_est'].clamp(max=CLAMP))   # (B, 1)
                else:
                    # phi_pearson fills the reserved +1 slot in _analyticsGlobalDim for GLMM
                    out.append(stats['phi_pearson'].clamp(max=CLAMP).unsqueeze(-1))  # (B, 1)
                if self.d_corr > 0:
                    Psi = stats['Psi'] if 'Psi' in stats else stats['Psi_lap']  # (B, q, q)
                    std = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()  # (B, q)
                    psi_corr = (Psi / (std.unsqueeze(-1) * std.unsqueeze(-2))).clamp(-1, 1)
                    # Shrink toward identity: α = G_mom / (G_mom + C), C=5
                    # Reduces variance of the noisy MoM estimate without hurting ranking (R)
                    q_rfx = Psi.shape[-1]
                    G_mom = (
                        data['mask_m'].bool() & (data['ns'] > q_rfx + 1)
                    ).float().sum(-1)                                       # (B,)
                    alpha = (G_mom / (G_mom + 10.0)).view(-1, 1, 1)        # (B, 1, 1)
                    eye = torch.eye(q_rfx, dtype=Psi.dtype, device=Psi.device)
                    psi_corr = (psi_corr * alpha + eye * (1 - alpha)).clamp(-1 + 1e-6, 1 - 1e-6)
                    out.append(corrToUnconstrained(psi_corr))  # atanh space matches target
        return torch.cat(out, dim=-1)

    def _localContext(
        self,
        local_summary: torch.Tensor,
        global_params: torch.Tensor,
        data: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        b, m, _ = local_summary.shape
        ndim = global_params.dim()
        if ndim == 3:  # samples
            s = global_params.shape[-2]
            global_params_exp = global_params.unsqueeze(1).expand(b, m, s, -1)
            local_summary_exp = local_summary.unsqueeze(2).expand(b, m, s, -1)
        elif ndim == 2:  # ground truth
            global_params_exp = global_params.unsqueeze(1).expand(b, m, -1)
            local_summary_exp = local_summary
        else:
            raise ValueError(f'global_params has {ndim} dims, expected 2 or 3')
        ctx = torch.cat([local_summary_exp, global_params_exp], dim=-1)
        if data is not None and self.analytical_context and self.likelihood_family == 0:
            ctx = torch.cat([ctx, self._analyticalBLUPContext(data, global_params)], dim=-1)
        return ctx

    def _analyticalBLUPContext(
        self,
        data: dict[str, torch.Tensor],
        global_params: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to analyticalBLUPContext after extracting constrained params.

        global_params: (B, d_g) ground-truth or (B, S, d_g) samples — sigma in softplus space.
        Returns: (B, m, 3*q) or (B, m, S, 3*q).
        """
        has_s = global_params.dim() == 3
        gp = global_params.unsqueeze(1) if not has_s else global_params  # (B, S, d_g)

        d, q = self.d_ffx, self.d_rfx
        q_var = q + 1  # sigma_rfx (q) + sigma_eps (1); lf==0 always has sigma_eps
        beta = gp[..., :d]
        sigma_rfx = constrainSigma(gp[..., d : d + q])
        sigma_eps = constrainSigma(gp[..., d + q : d + q + 1]).squeeze(-1)
        z_corr = gp[..., d + q_var : d + q_var + self.d_corr] if self.d_corr > 0 else None

        ctx = analyticalBLUPContext(data, beta, sigma_rfx, sigma_eps, z_corr, clamp=CLAMP)
        return ctx.squeeze(2) if not has_s else ctx

    def _preprocess(self, targets: torch.Tensor, local: bool = False) -> torch.Tensor:
        """constrain target parameters"""
        if local:
            return targets
        targets = targets.clone()
        d = self.d_ffx
        q_var = self.d_rfx + (1 if self.has_sigma_eps else 0)

        # project sigma block R+ → R
        sigmas = targets[..., d : d + q_var]
        targets[..., d : d + q_var] = unconstrainSigma(sigmas)
        # project corr lower triangle (-1,1) → R
        if self.d_corr > 0:
            r_corr = targets[..., d + q_var : d + q_var + self.d_corr]
            targets[..., d + q_var : d + q_var + self.d_corr] = corrLowerToUnconstrained(r_corr, self.d_rfx)
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
            log_sigmas = samples[..., d : d + q_var].clone()
            log_det = logDetJacobian(log_sigmas)
            log_prob_g = proposed['global']['log_prob'] - log_det.sum(dim=-1)
            sigmas = constrainSigma(log_sigmas)
            samples_out = [samples[..., :d], sigmas]
            if d_corr > 0:
                z_corr = samples[..., d + q_var : d + q_var + d_corr]
                log_prob_g = log_prob_g - logDetJacobianCorr(z_corr, q)
                L_corr = unconstrainedToCholesky(z_corr, q)
                r_corr = corrToLower(L_corr @ L_corr.mT)
                samples_out.append(r_corr)
            proposed['global']['log_prob'] = log_prob_g
            proposed['global']['samples'] = torch.cat(samples_out, dim=-1)
        return Proposal(proposed, has_sigma_eps=self.has_sigma_eps, d_corr=d_corr)

    def summarize(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._inputs(data)
        summary_l_raw = self.summarizer_l(inputs, mask=data['mask_n'])
        if self.analytical_context:
            stats = self._dataStatistics(data)  # compute once, share across both contexts
        else:
            stats = None
        # Global summarizer sees REML BLUPs per group (best available at this point).
        summary_l_for_g = self._addMetadata(summary_l_raw, data, local=True, stats=stats)
        summary_g = self.summarizer_g(summary_l_for_g, mask=data['mask_m'])
        summary_g = self._addMetadata(summary_g, data, local=False, stats=stats)
        # Local context gets BLUP-free summary; analytical BLUPs are injected in _localContext
        # from the current global params to stay consistent with the conditioning sigma.
        if self.analytical_context and self.likelihood_family == 0:
            summary_l = self._addMetadata(summary_l_raw, data, local=True, stats=None)
        else:
            summary_l = summary_l_for_g
        return summary_g, summary_l

    def forward(
        self,
        data: dict[str, torch.Tensor],
        summaries: tuple[torch.Tensor, torch.Tensor] | None = None,
        ancestral_rate: float = 0.0,
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

        # local posterior — with curriculum, sometimes condition on sampled globals
        targets_l = self._targets(data, local=True)
        targets_l = self._preprocess(targets_l, local=True)
        mask_l = self._masks(data, local=True)
        if ancestral_rate > 0.0 and torch.rand(1).item() < ancestral_rate:
            with torch.no_grad():
                samples_g, _ = self.posterior_g.sample(  # type: ignore
                    1, context=summary_g, mask=mask_g
                )
            context_g = samples_g.squeeze(-2)
        else:
            context_g = targets_g
        context_l = self._localContext(summary_l, context_g, data)
        log_probs['local'] = self.posterior_l.logProb(  # type: ignore
            targets_l, context=context_l, mask=mask_l
        )
        return log_probs

    def backward(
        self,
        data: dict[str, torch.Tensor],
        summaries: tuple[torch.Tensor, torch.Tensor] | None = None,
        n_samples: int = 1,
        detach_global: bool = False,
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
        if detach_global:
            with torch.no_grad():
                samples_g, log_prob_g = self.posterior_g.sample(  # type: ignore
                    n_samples, context=summary_g, mask=mask_g
                )
            samples_g = samples_g.detach()
        else:
            samples_g, log_prob_g = self.posterior_g.sample(  # type: ignore
                n_samples, context=summary_g, mask=mask_g
            )
        proposed['global'] = {'samples': samples_g, 'log_prob': log_prob_g}

        # local posterior
        mask_l = self._masks(data, local=True)
        b, m, q = mask_l.shape
        mask_l = mask_l.unsqueeze(-2).expand(b, m, n_samples, q)
        context_l = self._localContext(summary_l, samples_g, data)
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
