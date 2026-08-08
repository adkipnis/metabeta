"""
posthoc/warmnuts.py — Warm-started NUTS correction for flow posteriors.

Design
------
A proposal — preferably the MB-IMH posterior (flow draws refined by Independence
MH, posthoc/metropolis.py), otherwise raw flow draws — provides three warm-start
ingredients for PyMC's NUTS sampler:

1. Chain start points (initvals), back-transformed to the non-centred
   parameterisation.
2. The initial diagonal mass matrix: per-coordinate mean/variance of the
   proposal draws mapped to PyMC's unconstrained space — the same estimator
   PyMC's 'adapt_diag' init builds from early tuning, but available at step 0.
   Adaptation stays on, so an off proposal variance is corrected during tune.
3. For correlated rfx, the LKJCholeskyCov start value itself (see below).

Unlike importance sampling, NUTS targets the exact posterior p(θ|y) without
weights, and the warm start cuts the tuning budget needed to reach the typical
set with an adapted metric.

Warm-start mechanics
--------------------
For each dataset at batch index b, n_chains diverse samples are selected — at
interior quantiles of the global log density when the proposal carries one, and
evenly spaced along the sample dimension otherwise (IMH output stores no log
density and repeats states on rejection; even spacing lands in distinct chain
blocks, and exact-duplicate picks are nudged forward).  Each sample is
back-transformed:

  Independent rfx (eta_rfx == 0 or q == 1):
      z_j  = rfx_j / σ_rfx_j    →  '{1|i,x1|i,...}_offset'
      σ_rfx_j                    →  '{1|i,x1|i,...}_sigma'

  Correlated rfx (eta_rfx > 0, q >= 2):
      Σ_rfx = D @ R @ D  (D = diag(σ_rfx), R = corr_rfx from the proposal)
      chol  = lower_cholesky(Σ_rfx)
      z     = rfx @ chol⁻ᵀ                    →  '_rfx_offset'
      packed lower triangle of chol            →  '_lkj_rfx'
      (previously the LKJCholeskyCov was left at PyMC defaults, so the offsets
      z — computed from the proposal's chol — were inconsistent with the chol
      the chain actually started at)

The resulting dict list is passed as `initvals` to pm.sample.

NB: pm.sample(nuts_kwargs={'max_treedepth': ...}) is *silently ignored* by
PyMC ≥ 5 (**kwargs there are step_kwargs keyed by stepper name, and unknown
keys inside the NUTS constructor vanish into its **kwargs chain) — the runs
before 2026-08 therefore all used the default tree depth 10.  max_treedepth is
now passed to the NUTS constructor directly.

Output
------
WarmNuts.__call__ returns a Proposal with b=1 and n_chains * draws samples.
runWarmNuts refines the flow proposal with IMH first (mode='marginal' for
Normal, 'laplace' for GLMMs), then stacks per-dataset proposals along the
batch dimension.
"""

import argparse

import arviz as az
import numpy as np
import torch
from torch import Tensor

from metabeta.models.approximator import Approximator
from metabeta.utils.constants import hasSigmaEps
from metabeta.utils.pymc import buildPymc, extractAll
from metabeta.utils.results import Proposal


class WarmNuts:
    def __init__(
        self,
        ds: dict[str, np.ndarray],
        n_chains: int = 4,
        tune: int = 500,
        draws: int = 500,
        seed: int = 42,
        target_accept: float = 0.9,
        max_treedepth: int = 12,
        warm_mass: bool = True,
        mass_draws: int = 64,
    ) -> None:
        """
        Parameters
        ----------
        ds : dict
            Single unpadded dataset (output of Fitter._getSingle / unpad).
        n_chains : int
            Number of independent NUTS chains.
        tune : int
            NUTS tuning steps (burn-in; discarded by PyMC).
        draws : int
            Posterior draws per chain.
        seed : int
            Random seed passed to pm.sample.
        target_accept : float
            Target acceptance rate for step-size adaptation.  0.9 is
            recommended for posteriors with complex geometry (default 0.9).
        max_treedepth : int
            NUTS maximum tree depth (default 12; PyMC default 10).  Higher
            values allow the sampler to take longer trajectories through
            difficult posteriors at the cost of more gradient evaluations.
        warm_mass : bool
            Initialise the diagonal mass matrix from the proposal draws'
            unconstrained-space variance instead of the unit metric
            (default True).  Adaptation during tune stays on either way.
        mass_draws : int
            Number of proposal draws used to estimate the mass matrix.
        """
        self.ds = ds
        self.n_chains = n_chains
        self.tune = tune
        self.draws = draws
        self.seed = seed
        self.target_accept = target_accept
        self.max_treedepth = max_treedepth
        self.warm_mass = warm_mass
        self.mass_draws = mass_draws

        self.d = int(ds['d'])
        self.q = int(ds['q'])
        self.m = int(ds['m'])
        self.correlated = float(ds.get('eta_rfx', 0)) > 0 and self.q >= 2
        self.has_sigma_eps = hasSigmaEps(int(ds.get('likelihood_family', 0)))

        # Build PyMC model once; reused across __call__ invocations.
        self.model = buildPymc(ds)
        # Transformed (unconstrained) defaults: template for mass-matrix points —
        # fixes coordinate order, shapes, and dtypes.
        self._ip = self.model.initial_point(random_seed=seed)
        self._value_names = {
            rv.name: self.model.rvs_to_values[rv].name for rv in self.model.free_RVs
        }

    # ------------------------------------------------------------------
    # Init-value construction
    # ------------------------------------------------------------------

    def _prep(self, proposal: Proposal, b_idx: int) -> tuple[Tensor, Tensor, Tensor | None]:
        """Slice per-dataset tensors once.

        proposal.corr_rfx is recomputed from the constrained corr dims on every
        property access when no cache is attached (e.g. IMH output) — hoist it
        here instead of touching the property per draw.
        """
        # full flow layout of this proposal, used by _ivFromSample to locate
        # sigma_rfx/sigma_eps in the global sample vector
        self._proposal_d = proposal.d
        self._proposal_q = proposal.q
        sg = proposal.samples_g[b_idx].cpu()   # (n_s, D_g)
        sl = proposal.samples_l[b_idx].cpu()   # (m_batch, n_s, q_batch)
        corr_all = proposal.corr_rfx           # (b, n_s, q, q) or None
        corr_b = corr_all[b_idx].cpu() if corr_all is not None else None
        return sg, sl, corr_b

    def _selectIndices(self, proposal: Proposal, b_idx: int, sg: Tensor) -> list[int]:
        """Select n_chains diverse start indices from the proposal.

        Quantiles of the global log density when available; evenly spaced
        otherwise.  IMH output stores an all-zero log_prob and duplicates
        states on rejection, so a constant/non-finite log density triggers the
        even-spacing fallback and exact-duplicate picks are nudged forward.
        """
        C = self.n_chains
        n_s = sg.shape[0]

        lp = None
        if 'log_prob' in proposal.data['global']:
            lp = proposal.log_prob_g[b_idx].cpu()
            if not torch.isfinite(lp).all() or (lp.max() - lp.min()) < 1e-9:
                lp = None
        if lp is not None:
            sorted_idx = torch.argsort(lp)
            qs = torch.linspace(0, 1, C + 2)[1:-1]  # C interior quantiles
            pick = (qs * n_s).long().clamp(0, n_s - 1)
            indices = sorted_idx[pick].tolist()
        else:
            indices = torch.linspace(0, n_s - 1, C).long().tolist()

        out: list[int] = []
        for idx in indices:
            j = int(idx)
            while any(torch.equal(sg[j], sg[k]) for k in out) and j + 1 < n_s:
                j += 1
            out.append(j)
        return out

    def _ivFromSample(self, sg: Tensor, sl: Tensor, corr_b: Tensor | None, s_idx: int) -> dict:
        """Back-transform one proposal draw to a PyMC initval dict (constrained space)."""
        d, q, m = self.d, self.q, self.m
        # The flow always outputs d_ffx fixed effects and d_rfx sigma values; the
        # full layout (p_d, p_q) locates sigma_rfx/sigma_eps, this dataset only
        # uses the first d/q entries.
        p_d = self._proposal_d
        p_q = self._proposal_q

        iv: dict = {}
        ffx_i = sg[s_idx, :d].numpy()
        for j in range(d):
            iv['Intercept' if j == 0 else f'x{j}'] = ffx_i[j]

        # sigma_eps (Normal only; PyMC applies the log-transform internally)
        if self.has_sigma_eps:
            s_eps = float(sg[s_idx, p_d + p_q].item())
            iv['sigma'] = max(s_eps, 1e-6)

        sr_i = sg[s_idx, p_d : p_d + q].numpy().clip(1e-6)   # (q,)
        rfx_i = sl[:m, s_idx, :q].numpy()                    # (m, q)

        if self.correlated:
            # Build Cholesky of Σ_rfx = D @ R @ D
            if corr_b is not None:
                R = corr_b[s_idx, :q, :q].numpy()
            else:
                R = np.eye(q, dtype=np.float32)
            D_mat = np.diag(sr_i)
            Sigma = D_mat @ R @ D_mat + 1e-6 * np.eye(q)
            chol = np.linalg.cholesky(Sigma)   # lower triangular
            # rfx = z @ chol.T  →  z = solve(chol, rfx.T).T
            iv['_rfx_offset'] = np.linalg.solve(chol, rfx_i.T).T   # (m, q)
            # warm-start the LKJCholeskyCov at the same chol the offsets assume
            iv['_lkj_rfx'] = chol[np.tril_indices(q)]              # (q(q+1)/2,)
        else:
            for j in range(q):
                s_name = '1|i_sigma' if j == 0 else f'x{j}|i_sigma'
                o_name = '1|i_offset' if j == 0 else f'x{j}|i_offset'
                iv[s_name] = float(sr_i[j])
                iv[o_name] = rfx_i[:, j] / (sr_i[j] + 1e-12)   # (m,)

        return iv

    def _initVals(self, proposal: Proposal, b_idx: int) -> list[dict]:
        """Select n_chains diverse start points from proposal and back-transform."""
        sg, sl, corr_b = self._prep(proposal, b_idx)
        indices = self._selectIndices(proposal, b_idx, sg)
        return [self._ivFromSample(sg, sl, corr_b, i) for i in indices]

    # ------------------------------------------------------------------
    # Mass-matrix warm start
    # ------------------------------------------------------------------

    def _transformedPoint(self, iv: dict) -> dict[str, np.ndarray]:
        """Map a constrained initval dict to PyMC's transformed (unconstrained) space.

        buildPymc only produces identity-, log-, and cholesky-packed-transformed
        free RVs, so the value transforms are applied numerically by value-var
        suffix instead of compiling per-draw pytensor graphs.  Entries missing
        from iv keep the model's transformed default.
        """
        point = dict(self._ip)
        for rv_name, value in iv.items():
            vname = self._value_names.get(rv_name)
            if vname is None:
                continue
            v = np.asarray(value, dtype=np.float64)
            if vname.endswith('_log__'):
                v = np.log(np.clip(v, 1e-12, None))
            elif vname.endswith('_cholesky-cov-packed__'):
                v = v.copy()
                diag_idx = [i * (i + 3) // 2 for i in range(self.q)]
                v[diag_idx] = np.log(np.clip(v[diag_idx], 1e-12, None))
            template = point[vname]
            point[vname] = v.reshape(template.shape).astype(template.dtype)
        return point

    def _warmPotential(self, sg: Tensor, sl: Tensor, corr_b: Tensor | None):
        """QuadPotentialDiagAdapt initialised at the proposal draws' mean/variance.

        Mirrors what PyMC's 'adapt_diag' init estimates from early tuning
        (initial_weight=10, adaptation on), but is available at step 0.
        Coordinates the proposal leaves constant fall back to the unit metric.
        """
        from pymc.blocking import DictToArrayBijection
        from pymc.step_methods.hmc.quadpotential import QuadPotentialDiagAdapt

        n_s = sg.shape[0]
        picks = np.unique(np.linspace(0, n_s - 1, min(self.mass_draws, n_s)).astype(int))
        flats = []
        for s_idx in picks:
            point = self._transformedPoint(self._ivFromSample(sg, sl, corr_b, int(s_idx)))
            flats.append(DictToArrayBijection.map(point).data)
        arr = np.stack(flats)   # (K, n_flat)

        mean = arr.mean(0)
        var = arr.var(0)
        mean[~np.isfinite(mean)] = 0.0
        var[~np.isfinite(var) | (var < 1e-10)] = 1.0
        return QuadPotentialDiagAdapt(arr.shape[1], mean, var, 10)

    # ------------------------------------------------------------------
    # Trace → Proposal conversion
    # ------------------------------------------------------------------

    def _traceToProposal(self, trace: az.InferenceData) -> Proposal:
        """Convert a PyMC NUTS trace to a Proposal with b=1."""
        d, q = self.d, self.q
        out = extractAll(trace, self.ds, d, q, 'wn')

        # Shapes from extractAll:
        #   wn_ffx:       (d, n_s)
        #   wn_sigma_rfx: (q, n_s)
        #   wn_sigma_eps: (1, n_s)   — if Normal
        #   wn_rfx:       (q, m, n_s)
        #   wn_corr_rfx:  (1, n_s, q, q)

        # arviz returns float64 arrays; cast to float32 for compatibility with the
        # rest of the pipeline (model tensors, evaluation, etc.).
        def _f32(a) -> torch.Tensor:
            return torch.as_tensor(a).float()

        ffx = _f32(out['wn_ffx']).T                           # (n_s, d)
        sigma_rfx = _f32(out['wn_sigma_rfx']).T               # (n_s, q)
        n_s = ffx.shape[0]
        parts = [ffx, sigma_rfx]

        if self.has_sigma_eps:
            sigma_eps = _f32(out['wn_sigma_eps']).squeeze(0)  # (n_s,)
            parts.append(sigma_eps.unsqueeze(-1))              # (n_s, 1)

        samples_g = torch.cat(parts, dim=-1).unsqueeze(0)               # (1, n_s, D)
        # (q, m, n_s) → permute(1, 2, 0) → (m, n_s, q) → unsqueeze(0) → (1, m, n_s, q)
        samples_l = _f32(out['wn_rfx']).permute(1, 2, 0).unsqueeze(0)

        proposed = {
            'global': {
                'samples': samples_g,
                'log_prob': torch.zeros(1, n_s),  # dummy
            },
            'local': {
                'samples': samples_l,
                'log_prob': torch.zeros(1, self.m, n_s),  # dummy
            },
        }

        # extractAll always stores corr_rfx (identity for non-correlated datasets)
        corr_rfx = _f32(out['wn_corr_rfx'])   # (1, n_s, q, q)
        proposal = Proposal(proposed, has_sigma_eps=self.has_sigma_eps, corr_rfx=corr_rfx)
        return proposal

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, proposal: Proposal, b_idx: int = 0) -> tuple['Proposal', dict]:
        """Run warm-started NUTS for the dataset at batch index b_idx.

        Parameters
        ----------
        proposal : Proposal
            Proposal used to seed chains and mass matrix (samples at b_idx) —
            MB-IMH output preferred, raw flow draws also work.  Must be in the
            *standardized* space that buildPymc models.
        b_idx : int
            Dataset index within the batch.

        Returns
        -------
        proposal_out : Proposal with n_chains * draws samples and b=1.
        diag : dict with keys 'n_divergences', 'max_rhat', 'min_ess',
            'min_ess_t', and 'reff'.
        """
        sg, sl, corr_b = self._prep(proposal, b_idx)
        indices = self._selectIndices(proposal, b_idx, sg)
        initvals = [self._ivFromSample(sg, sl, corr_b, i) for i in indices]

        import pymc as pm

        with self.model:
            if self.warm_mass:
                # explicit step: pm.sample would otherwise rebuild the potential
                # from the unit metric via init_nuts
                step = pm.NUTS(
                    potential=self._warmPotential(sg, sl, corr_b),
                    target_accept=self.target_accept,
                    max_treedepth=self.max_treedepth,
                )
                trace = pm.sample(
                    tune=self.tune,
                    draws=self.draws,
                    chains=self.n_chains,
                    step=step,
                    initvals=initvals,
                    random_seed=self.seed,
                    return_inferencedata=True,
                    progressbar=False,
                )
            else:
                trace = pm.sample(
                    tune=self.tune,
                    draws=self.draws,
                    chains=self.n_chains,
                    initvals=initvals,
                    target_accept=self.target_accept,
                    max_treedepth=self.max_treedepth,
                    random_seed=self.seed,
                    return_inferencedata=True,
                    progressbar=False,
                )
        n_divs = int(trace.sample_stats['diverging'].values.sum())
        n_draws_total = self.n_chains * self.draws
        try:
            df = az.summary(trace, kind='diagnostics')
            max_rhat = float(df['r_hat'].max())
            min_ess = float(df['ess_bulk'].min())
            min_ess_t = float(df['ess_tail'].min())
            reff = float(df['ess_bulk'].mean() / n_draws_total)
        except Exception:
            max_rhat = float('nan')
            min_ess = float('nan')
            min_ess_t = float('nan')
            reff = 1.0
        diag = {
            'n_divergences': n_divs,
            'max_rhat': max_rhat,
            'min_ess': min_ess,
            'min_ess_t': min_ess_t,
            'reff': reff,
        }
        proposal = self._traceToProposal(trace)
        proposal.reff = reff
        return proposal, diag


# ---------------------------------------------------------------------------
# Batch-stacking helper
# ---------------------------------------------------------------------------


def _stackProposals(
    proposals: list[Proposal],
    target_d: int | None = None,
    target_q: int | None = None,
) -> Proposal:
    """Stack per-dataset proposals (b=1 each) into a batch proposal along dim 0.

    Unlike concatProposalsBatch (which concatenates along the sample dim),
    this stacks along the batch dim, padding d/q/m dims as needed.

    WarmNuts proposals have actual d and q (from unpadded individual datasets),
    which may differ across datasets in a mixed-d/q collection.  samples_g is
    rebuilt as [ffx_padded(target_d), sigma_rfx_padded(target_q), sigma_eps(1)]
    so all batch entries share the same D_g dimension.

    Parameters
    ----------
    target_d : int or None
        Target fixed-effects dimension (≥ max actual d). Defaults to max actual d.
    target_q : int or None
        Target random-effects dimension (≥ max actual q). Defaults to max actual q.
    """
    has_sigma_eps = proposals[0].has_sigma_eps
    n_s = proposals[0].n_samples
    max_d = target_d if target_d is not None else max(p.d for p in proposals)
    max_q = target_q if target_q is not None else max(p.q for p in proposals)
    max_m = max(p.samples_l.shape[1] for p in proposals)

    # Rebuild samples_g with uniform [max_d, max_q, (1)] layout.
    sg_list, lg_list = [], []
    for p in proposals:
        ffx = p.ffx           # (1, n_s, d_i)
        srfx = p.sigma_rfx    # (1, n_s, q_i)
        parts: list[torch.Tensor] = []
        # pad ffx to max_d
        if p.d < max_d:
            parts.append(torch.cat([ffx, ffx.new_zeros(1, n_s, max_d - p.d)], dim=-1))
        else:
            parts.append(ffx)
        # pad sigma_rfx to max_q
        if p.q < max_q:
            parts.append(torch.cat([srfx, srfx.new_zeros(1, n_s, max_q - p.q)], dim=-1))
        else:
            parts.append(srfx)
        if has_sigma_eps:
            parts.append(p.sigma_eps.unsqueeze(-1))  # (1, n_s) → (1, n_s, 1)
        sg_list.append(torch.cat(parts, dim=-1))   # (1, n_s, max_d+max_q+1)
        lg_list.append(p.log_prob_g)               # (1, n_s)

    samples_g = torch.cat(sg_list, dim=0)    # (B, n_s, D)
    log_prob_g = torch.cat(lg_list, dim=0)   # (B, n_s)

    # Pad m and q dims for locals.
    sl_list, lp_list = [], []
    for p in proposals:
        m_i = p.samples_l.shape[1]
        q_i = p.samples_l.shape[3]
        sl = p.samples_l   # (1, m_i, n_s, q_i)
        lp = p.log_prob_l  # (1, m_i, n_s)
        if m_i < max_m:
            sl = torch.cat([sl, sl.new_zeros(1, max_m - m_i, n_s, q_i)], dim=1)
            lp = torch.cat([lp, lp.new_zeros(1, max_m - m_i, n_s)], dim=1)
        if q_i < max_q:
            sl = torch.cat([sl, sl.new_zeros(1, max_m, n_s, max_q - q_i)], dim=-1)
        sl_list.append(sl)
        lp_list.append(lp)

    proposed = {
        'global': {'samples': samples_g, 'log_prob': log_prob_g},
        'local': {
            'samples': torch.cat(sl_list, dim=0),  # (B, max_m, n_s, max_q)
            'log_prob': torch.cat(lp_list, dim=0),  # (B, max_m, n_s)
        },
    }
    merged = Proposal(proposed, has_sigma_eps=has_sigma_eps)
    # Propagate reff: use the mean across per-dataset proposals.
    merged.reff = float(np.mean([p.reff for p in proposals]))

    # Stack corr_rfx — all WarmNuts proposals always have it (identity for non-correlated)
    corr_rfx_list = [p.corr_rfx for p in proposals]
    if all(c is not None for c in corr_rfx_list):
        stacked = []
        for c in corr_rfx_list:
            q_i = c.shape[-1]  # type: ignore[union-attr]
            if q_i < max_q:
                eye = torch.eye(max_q, dtype=c.dtype)
                c_pad = eye.unsqueeze(0).unsqueeze(0).expand(1, n_s, -1, -1).clone()
                c_pad[:, :, :q_i, :q_i] = c
                stacked.append(c_pad)
            else:
                stacked.append(c)
        merged._corr_rfx = torch.cat(stacked, dim=0)   # (B, n_s, max_q, max_q)

    return merged


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


def runWarmNuts(
    model: Approximator,
    data: dict[str, torch.Tensor],
    ds_list: list[dict[str, np.ndarray]],
    cfg: argparse.Namespace,
) -> Proposal:
    """Draw a flow proposal, refine it with IMH, then warm-start NUTS per dataset.

    Operates dataset-by-dataset (NUTS is not batched).  The flow runs once on
    the full batch; the IMH pass (MB-IMH: mode='marginal' for Normal,
    'laplace' for GLMMs) then supplies the start points and mass matrix.

    cfg fields
    ----------
    n_chains       : int   — IMH chains (default 4)
    n_steps        : int   — IMH steps per chain (default 250)
    imh_burnin     : int   — IMH burnin steps (default 25)
    wn_chains      : int   — NUTS chains per dataset (default 4)
    wn_tune        : int   — NUTS tuning steps (default 500)
    wn_draws       : int   — NUTS draws per chain (default 500)
    wn_target_accept : float — target acceptance rate (default 0.9)
    wn_max_treedepth : int   — NUTS max tree depth (default 12)
    wn_warm_mass   : bool  — mass-matrix warm start (default True)
    likelihood_family : int
    rescale        : bool
    seed           : int
    """
    from metabeta.posthoc.metropolis import MetropolisSampler

    lf = getattr(cfg, 'likelihood_family', 0)
    imh_chains = getattr(cfg, 'n_chains', 4)
    imh_steps = getattr(cfg, 'n_steps', 250)
    imh_burnin = getattr(cfg, 'imh_burnin', 25)
    n_chains = getattr(cfg, 'wn_chains', 4)
    tune = getattr(cfg, 'wn_tune', 500)
    draws = getattr(cfg, 'wn_draws', 500)
    seed = getattr(cfg, 'seed', 42)
    target_accept = getattr(cfg, 'wn_target_accept', 0.9)
    max_treedepth = getattr(cfg, 'wn_max_treedepth', 12)
    warm_mass = getattr(cfg, 'wn_warm_mass', True)

    # Everything up to the stacked output stays in standardized space —
    # buildPymc (via WarmNuts) models standardized ds['y']/ds['X'] (from
    # col.raw / Fitter._getSingle), and the IMH weights must be computed on the
    # same data the chain start points describe.  Rescale afterward if set.
    proposal = model.estimate(data, n_samples=imh_chains * imh_steps)
    sampler = MetropolisSampler(
        data,
        n_chains=imh_chains,
        n_steps=imh_steps,
        burnin=imh_burnin,
        mode='marginal' if lf == 0 else 'laplace',
        likelihood_family=lf,
    )
    refined, _ = sampler(proposal)

    proposals = []
    for b, ds in enumerate(ds_list):
        wn_proposal, _ = WarmNuts(
            ds,
            n_chains=n_chains,
            tune=tune,
            draws=draws,
            seed=seed,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            warm_mass=warm_mass,
        )(refined, b_idx=b)
        proposals.append(wn_proposal)
    merged = _stackProposals(proposals)
    if cfg.rescale:
        merged.rescale(data['sd_y'])
    return merged
