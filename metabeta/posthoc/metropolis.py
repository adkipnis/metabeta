"""
posthoc/metropolis.py — Independence Metropolis-Hastings (IMH) correction for flow posteriors.

Design
------
The flow q(θ) approximates p(θ|y) but can be systematically off (e.g. the sigma_eps/sigma_rfx
ridge in Normal models).  IMH uses the flow as proposal in a Markov chain, accepting/rejecting
with ratio w(θ') / w(θ).  Unlike IS, degenerate proposals are corrected by rejection rather
than by collapsed weights, and the chain is guaranteed to target the correct posterior.

Three modes differ in how rfx (local params) are handled:

  'global'
      log_w = log p(y|θ_g, θ_l) + log p(θ_g) − log q_g
      rfx enter the likelihood but carry no IS correction.  When a global proposal is accepted
      the paired flow rfx from the same draw travels with it.  Works for all likelihoods.

  'marginal'  [default for Normal; requires likelihood_family == 0]
      log_w = log p_marginal(y|θ_g) + log p(θ_g) − log q_g
      rfx are integrated out analytically via the Normal-Normal conjugate marginal, so the
      chain operates only in the low-dimensional global space.  After acceptance, fresh rfx
      are drawn from the exact conditional posterior p(rfx | θ_g, y) (Rao-Blackwellised).
      Best mixing; theoretically optimal for Normal.

  'joint'
      log_w = log p(y|θ_g,θ_l) + log p(θ_l|θ_g) + log p(θ_g) − log q_g − log q_l
      Full joint target; rfx are subject to acceptance alongside globals.  Correct for GLMMs
      but degenerates with many groups (high rfx dimension).

  'laplace'  [GLMMs only; requires likelihood_family != 0]
      log_w = log p̂_laplace(y|θ_g) + log p(θ_g) − log q_g
      The GLMM analog of 'marginal': rfx are integrated out via the Laplace (nAGQ=1)
      approximation (posthoc/laplace_glmm.py), so the chain operates in the global space;
      after acceptance, fresh rfx are drawn from the Laplace-Gaussian conditional
      N(b*, H⁻¹) — gathered from the pool pass by the accepted samples' pool indices,
      since every accepted state is a pool member whose modes/Hessians are already known.
      The proposal's rfx only seed the per-group Newton mode search, so at inference the
      local flow is skipped and the analytical rfx estimate seeds it instead
      (Approximator.estimate(local=False)); accuracy is identical, and the local flow —
      the dominant CPU cost at production pool sizes, ~linear in the group count — is gone.
      Mirrors the recipe that fixed the huge-Normal regime — added because isLaplace's
      PSIS guardrail falls back on 13–50% of large/huge GLMM datasets
      (2026-07-29 ablation), and rejection-based correction has no fallback mode.
      Targets the same Laplace pseudo-posterior as isLaplace (shares its O(Laplace) bias).

  'laplace' with n_inner = K > 0  ['imhPM'; the GLMM default since 2026-09-26]
      log p̂_laplace is replaced by its unbiased IS² estimate (logMarginalLikelihoodIS2), so
      the chain is pseudo-marginal and targets the exact posterior; each kept state's rfx get
      one iterated-SIR move (refreshRfxIS2, reusing the pool's Laplace factors), since a
      rejected step would otherwise repeat the rfx vector. Matches NUTS in all eight GLMM
      ablation regimes where the Laplace chain lags (experiments/posthoc/is2.md).

Chain mechanics
---------------
A pool of s = n_chains × n_steps proposals is drawn upfront.  All log weights are computed in
one vectorised pass.  The MH loop is vectorised over (b, n_chains) and iterates as a Python
loop over n_steps — typically 50–250 steps.  The first `burnin` steps are discarded; the
remaining n_chains × (n_steps − burnin) samples are returned as a Proposal.

Empirical comparison
--------------------
Call MetropolisSampler with each mode in turn and evaluate the resulting Proposal via
Evaluator.summary() / plotComparison to choose the best mode for a given dataset type.
The diagnostics dict returned by __call__ contains per-chain acceptance rates as a quick
quality indicator before running the full evaluation.

Findings (2026-07 posthoc ablation, 128 validation datasets per family, small models)
-------------------------------------------------------------------------------------
Every IMH mode was measurably *worse* than raw flow samples on recovery, calibration, and
LOO-NLL. Root causes (the first two are fixed in this module; the third is open):

1. Argmax chain init (fixed). Chains were initialised at the pool-wide argmax of the
   log-weights of the very pool they then iterate over, so every later proposal had
   log_alpha = lw_prop − lw_argmax ≤ 0 by construction; at low acceptance a chain sat at
   that single point through the whole post-burnin phase. This produced the severe
   under-dispersion signature (RFX joint ECE ≈ −0.65, median LOO Pareto k of −1 to −3.5
   for 'joint' on Bernoulli/Poisson, acceptance 4–6%). Chains now initialise at t=0 of
   their block — a fair draw from the proposal.
2. Inconsistent marginal target (fixed). The 'marginal' branch computed its own weight
   with a diagonal-Σ_rfx marginal likelihood and *no* LKJ prior on z_corr, while
   `_sampleRfxConditional` conditioned rfx on those very z_corr dims — the chain targeted
   one distribution and drew rfx from another. Explains Corr(RFX) R dropping 0.812 → 0.650
   and σ_eps miscalibration under 'marginal' on Normal. Weights are now delegated to
   ImportanceSampler.unnormalizedPosterior (single source of truth with IS), which
   includes the correlated marginal likelihood and the LKJ prior term.
3. 'joint' mode degenerates structurally: one accept/reject decision for the full
   (global + all-groups rfx) block means acceptance collapses as m grows — this is not
   fixable by init or target corrections; prefer 'marginal' (Normal) or 'global' with a
   conditional rfx redraw (non-Normal, see TODO).

Note that marginal-IMH and marginal-SNIS (ImportanceSampler(marginal=True)) use identical
log-weights; SNIS uses all samples with smooth weights instead of a rejection chain and is
generally preferable — keep IMH as a diagnostic baseline.

TODO
----
- 'global' mode never MH-corrects rfx — accepted global samples keep the flow's raw,
  uncorrected rfx draw from the same proposal. Superseded in practice by mode='laplace'
  (the non-Normal default), whose conditional redraw handles this; 'global' remains a
  diagnostic baseline.

Findings (2026-09, 512 test datasets)
-------------------------------------
The robustified Laplace mode search removed init-dependent absorbing states from
mode='laplace' (Poisson-large σ_rfx ECE −0.070 → −0.051, LOO-NLL unchanged at NUTS level).
The large/huge-regime FFX under-dispersion is the finite proposal pool — identical with the
exact marginal target on Normal-huge, and shrinking as (ā·s)^−0.6 in a pool-size sweep —
hence the acceptance-based pool-size suggestion below.
"""

import argparse
import time
from typing import Literal

import torch
from torch import Tensor

from metabeta.models.approximator import Approximator
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.posthoc.laplace_glmm import (
    LaplaceImportanceSampler,
    refreshRfxIS2,
    sampleRfxLaplace,
)
from metabeta.utils.constants import hasSigmaEps
from metabeta.utils.families import sampleRfxConditionalNormal
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.regularization import corrLowerToUnconstrained, unconstrainedToCholesky
from metabeta.utils.results import Proposal

Mode = Literal['global', 'marginal', 'joint', 'laplace']

# IS² draws per group of the pseudo-marginal chain (mode='laplace', n_inner > 0; 'imhPM'):
# sd(log p̂(y|θ)) ≈ 0.08 median / ≤ 0.5 q90 at K = 8 (experiments/posthoc/is2_tuning.py)
IS2_N_INNER = 8

# Effective-draw target from the pool-size sweep (FFX ECE ∝ (ā·s)^−0.6; at ā·s ≈ 700 the
# huge regime reaches small-regime calibration). Suggested pool sizes aim for it.
N_EFF_TARGET = 700
SUGGEST_MIN = 1_000
SUGGEST_MAX = 16_000


def suggestPoolSize(
    accept_rate: Tensor,  # (b, n_chains) — post-burnin acceptance per chain
    n_eff_target: int = N_EFF_TARGET,
) -> Tensor:
    """Per-dataset pool-size suggestion from the measured IMH acceptance.

    Returns the smallest s (rounded up to a multiple of 500, clamped to
    [SUGGEST_MIN, SUGGEST_MAX]) whose expected accepted-draw count ā·s reaches
    n_eff_target. Advisory only — inference always runs at the user-specified
    pool size; datasets with near-zero acceptance saturate at SUGGEST_MAX.
    Returns (b,) long.
    """
    a_bar = accept_rate.mean(dim=1).clamp(min=1e-3)  # (b,)
    s = (n_eff_target / a_bar).ceil()
    s = (s / 500.0).ceil() * 500.0
    return s.clamp(min=SUGGEST_MIN, max=SUGGEST_MAX).long()


class MetropolisSampler:
    def __init__(
        self,
        data: dict[str, Tensor],
        n_chains: int = 4,
        n_steps: int = 250,
        burnin: int = 25,
        mode: Mode = 'marginal',
        likelihood_family: int = 0,
        eps: float = 1e-12,
        n_eff_target: int | None = N_EFF_TARGET,  # None disables the pool-size suggestion
        n_inner: int = 0,  # 'laplace' only: IS² draws per group; > 0 makes the chain pseudo-marginal
    ) -> None:
        if n_inner > 0 and mode != 'laplace':
            raise ValueError("n_inner (pseudo-marginal IS² weights) requires mode='laplace'")
        if mode == 'marginal' and likelihood_family != 0:
            raise ValueError("mode='marginal' requires likelihood_family=0 (Normal)")
        if mode == 'laplace' and likelihood_family == 0:
            raise ValueError("mode='laplace' is for GLMMs; Normal has the exact 'marginal'")
        if burnin >= n_steps:
            raise ValueError('burnin must be < n_steps')

        self.n_chains = n_chains
        self.n_steps = n_steps
        self.burnin = burnin
        self.mode = mode
        self.likelihood_family = likelihood_family
        self.has_sigma_eps = hasSigmaEps(likelihood_family)
        self.eps = eps
        self.n_eff_target = n_eff_target

        # Delegate all weight computation to ImportanceSampler.unnormalizedPosterior —
        # single source of truth shared with SNIS. 'marginal' uses the (correlated)
        # marginal likelihood + LKJ prior; 'laplace' the Laplace-approximated marginal
        # (same weights as isLaplace); 'joint' needs full=True (rfx prior + local
        # log-prob).
        if mode == 'laplace':
            self._is: ImportanceSampler = LaplaceImportanceSampler(
                data, likelihood_family=likelihood_family, eps=eps, n_inner=n_inner
            )
        else:
            self._is = ImportanceSampler(
                data,
                full=(mode == 'joint'),
                marginal=(mode == 'marginal'),
                likelihood_family=likelihood_family,
                eps=eps,
            )

        # Data tensors for the Normal-Normal conditional (marginal mode).
        # These mirror what ImportanceSampler stores but are kept as direct references.
        self._X = data['X']           # (b, m, n, d)
        self._Z = data['Z']           # (b, m, n, q)
        self._y = data['y'].unsqueeze(-1)  # (b, m, n, 1)
        self._mask_n = data['mask_n'].unsqueeze(-1)   # (b, m, n, 1)
        self._mask_m = data['mask_m'].unsqueeze(-1)   # (b, m, 1)

    # ------------------------------------------------------------------
    # Log-weight computation
    # ------------------------------------------------------------------

    def _logWeights(self, proposal: Proposal) -> Tensor:
        """Compute unnormalised log IS weights (b, s) according to self.mode.

        Fully delegated to ImportanceSampler.unnormalizedPosterior: 'marginal' gets
        the correlated marginal likelihood + LKJ prior there (the old hand-rolled
        branch used a diagonal-Σ marginal and no z_corr prior — see Findings).
        """
        log_q_g = proposal.log_prob_g  # (b, s)
        ll, lp = self._is.unnormalizedPosterior(proposal)
        if self.mode == 'joint':
            log_q_l = proposal.log_prob_l   # (b, m, s)
            lq = log_q_g + (log_q_l * self._is.mask_m).sum(1)
            return ll + lp - lq
        return ll + lp - log_q_g

    # ------------------------------------------------------------------
    # MH chain
    # ------------------------------------------------------------------

    def _runChains(
        self,
        log_w: Tensor,  # (b, s)
        sg: Tensor,  # (b, s, D_g)
        sl: Tensor | None,  # (b, m, s, q) — None for marginal mode
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:
        """Run n_chains independent IMH chains, return (sg_out, sl_out, idx_out, accept_rate).

        Outputs:
            sg_out       (b, C*T_post, D_g)
            sl_out       (b, m, C*T_post, q)  or None
            idx_out      (b, C*T_post)  — pool index (into the s axis) of each kept state
            accept_rate  (b, C)  — fraction of proposals accepted after burnin
        """
        b, s, D_g = sg.shape
        C, T = self.n_chains, self.n_steps
        T_post = T - self.burnin

        # Reshape pool into (b, C, T, ...)
        sg_ct = sg.reshape(b, C, T, D_g)
        lw_ct = log_w.reshape(b, C, T)
        if sl is not None:
            m, q = sl.shape[1], sl.shape[-1]
            sl_ct = sl.permute(0, 2, 1, 3).reshape(b, C, T, m, q)

        # Initialise at each chain's first draw — a fair sample from the proposal.
        # Initialising at the pool-wide argmax (as done previously) guarantees
        # under-dispersion: every later proposal in the pool then has
        # log_alpha = lw − lw_max ≤ 0, so low-acceptance chains sit at that single
        # point through the whole post-burnin phase (see Findings in the module
        # docstring).
        cur_g = sg_ct[:, :, 0].clone()   # (b, C, D_g)
        cur_lw = lw_ct[:, :, 0].clone()  # (b, C)
        # pool index of (chain c, step t) is c*T + t — the reshape above is row-major
        pool_base = (torch.arange(C, device=sg.device) * T).view(1, C).expand(b, C)
        cur_idx = pool_base.clone()  # (b, C)
        if sl is not None:
            cur_l = sl_ct[:, :, 0].clone()   # (b, C, m, q)

        keep_g: list[Tensor] = []
        keep_l: list[Tensor] = []
        keep_idx: list[Tensor] = []
        keep_acc: list[Tensor] = []

        for t in range(1, T):
            prop_lw = lw_ct[:, :, t]   # (b, C)

            log_alpha = (prop_lw - cur_lw).clamp(max=0.0)
            accept = torch.rand_like(log_alpha).log() < log_alpha  # (b, C)

            cur_g = torch.where(accept.unsqueeze(-1), sg_ct[:, :, t], cur_g)
            cur_lw = torch.where(accept, prop_lw, cur_lw)
            cur_idx = torch.where(accept, pool_base + t, cur_idx)
            if sl is not None:
                cur_l = torch.where(accept[:, :, None, None], sl_ct[:, :, t], cur_l)

            if t >= self.burnin:
                keep_g.append(cur_g.clone())
                keep_idx.append(cur_idx.clone())
                if sl is not None:
                    keep_l.append(cur_l.clone())
                keep_acc.append(accept.float())

        # Stack post-burnin samples: (T_post, b, C, D_g) → (b, C*T_post, D_g)
        sg_out = torch.stack(keep_g, dim=0).permute(1, 2, 0, 3).reshape(b, C * T_post, D_g)

        sl_out = None
        if sl is not None:
            sl_out = (
                torch.stack(keep_l, dim=0)  # (T_post, b, C, m, q)
                .permute(1, 2, 0, 3, 4)  # (b, C, T_post, m, q)
                .reshape(b, C * T_post, m, q)
                .permute(0, 2, 1, 3)  # (b, m, C*T_post, q)
            )

        idx_out = torch.stack(keep_idx, dim=0).permute(1, 2, 0).reshape(b, C * T_post)
        accept_rate = torch.stack(keep_acc, dim=0).mean(0)  # (b, C)
        return sg_out, sl_out, idx_out, accept_rate

    # ------------------------------------------------------------------
    # Normal-Normal conditional rfx posterior
    # ------------------------------------------------------------------

    def _sampleRfxConditional(
        self,
        sg_out: Tensor,  # (b, s_out, D_g)
        d: int,
        q: int,
        d_corr: int,
    ) -> Tensor:
        """Draw rfx ~ p(rfx | θ_g, y); delegates to sampleRfxConditionalNormal.

        The corr dims of sg_out store *constrained* lower-triangle correlations
        (like Proposal.samples_g), so they are mapped back to unconstrained space
        before building the Cholesky factor. (The previous inline implementation
        applied unconstrainedToCholesky to the constrained values directly —
        approximately right for small correlations, wrong in general.)

        Returns (b, m, s_out, q).
        """
        ffx = sg_out[..., :d]                # (b, s_out, d)
        sigma_rfx = sg_out[..., d : d + q]   # (b, s_out, q)
        sigma_eps = sg_out[..., d + q]       # (b, s_out)
        L_corr = None
        if d_corr > 0:
            z_corr = corrLowerToUnconstrained(sg_out[..., -d_corr:], q)
            L_corr = unconstrainedToCholesky(z_corr, q)  # (b, s_out, q, q)
        return sampleRfxConditionalNormal(
            ffx,
            sigma_rfx,
            sigma_eps,
            self._y,
            self._X,
            self._Z,
            self._mask_n,
            self._mask_m,
            L_corr=L_corr,
        )

    def _sampleRfxLaplace(self, idx_out: Tensor) -> Tensor:
        """Laplace analog of _sampleRfxConditional (mode='laplace').

        Every kept chain state is a member of the proposal pool, and the pool pass
        (_logWeights → LaplaceImportanceSampler.unnormalizedPosterior) already computed its
        per-group Laplace modes b* and Hessian factors. Gathering them by ``idx_out`` and
        drawing rfx ~ N(b*, H⁻¹) is therefore exact and avoids a second full Newton pass,
        which used to cost as much as the pool pass itself. Returns (b, m, s_out, q).

        With IS² weights (n_inner > 0) the kept state's weight-selected inner draw is gathered
        instead: under the pseudo-marginal extended target it is an exact conditional draw.
        """
        if self._is.n_inner > 0:
            return self._gatherPool(self._is._rfx, idx_out)
        modes_sel = self._gatherPool(self._is._modes, idx_out)
        chol_sel = self._gatherPool(self._is._chol_H, idx_out)
        return sampleRfxLaplace(modes_sel, chol_sel, self._is.mask_m)

    @staticmethod
    def _gatherPool(t: Tensor, idx_out: Tensor) -> Tensor:
        """Per-group pool tensor (b, m, s, ...) → its kept states (b, m, s_out, ...)."""
        idx = idx_out[:, None, :].reshape(*idx_out.shape[:1], 1, -1, *([1] * (t.dim() - 3)))
        return torch.gather(t, 2, idx.expand(*t.shape[:2], idx_out.shape[1], *t.shape[3:]))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, proposal: Proposal) -> tuple[Proposal, dict]:
        """Run IMH chains on a pre-drawn flow proposal.

        Parameters
        ----------
        proposal : Proposal
            Output of model.estimate(data, n_samples=n_chains * n_steps).

        Returns
        -------
        proposal_out : Proposal
            Post-burnin samples; n_chains * (n_steps - burnin) samples per dataset.
        diagnostics : dict
            'accept_rate' (b, n_chains) — fraction of proposals accepted post-burnin.
            'suggested_n_samples' (b,) — advisory pool size reaching the
            sweep-calibrated effective-draw target at the measured acceptance
            (omitted when n_eff_target is None). Inference itself always runs at
            the user-specified pool size.
        """
        t0 = time.perf_counter()
        s_expected = self.n_chains * self.n_steps
        if proposal.n_samples != s_expected:
            raise ValueError(
                f'proposal has {proposal.n_samples} samples; '
                f'expected n_chains × n_steps = {s_expected}'
            )

        d, q = proposal.d, proposal.q
        d_corr = proposal.d_corr

        log_w = self._logWeights(proposal)   # (b, s)

        # Run chain — rfx travels with globals only for 'global' and 'joint'; 'marginal'
        # and 'laplace' redraw rfx from the conditional at the accepted globals
        sl_in = proposal.samples_l if self.mode in ('global', 'joint') else None
        sg_out, sl_out, idx_out, accept_rate = self._runChains(log_w, proposal.samples_g, sl_in)

        # Attach rfx
        if self.mode == 'marginal':
            sl_out = self._sampleRfxConditional(sg_out, d, q, d_corr)
        elif self.mode == 'laplace':
            sl_out = self._sampleRfxLaplace(idx_out)
        # 'global' and 'joint': sl_out already set by _runChains

        b, s_out = sg_out.shape[0], sg_out.shape[1]
        m = sl_out.shape[1]
        proposed = {
            'global': {'samples': sg_out, 'log_prob': sg_out.new_zeros(b, s_out)},
            'local': {'samples': sl_out, 'log_prob': sl_out.new_zeros(b, m, s_out)},
        }
        out = Proposal(proposed, has_sigma_eps=proposal.has_sigma_eps, d_corr=d_corr)
        if self._is.n_inner > 0:
            # pseudo-marginal chain: a rejected step repeats the state's rfx, which left the rfx
            # under-covered at low acceptance (large/huge ablation); an iterated-SIR move given
            # θ_g, which leaves p(rfx | θ_g, y) invariant, gives each kept state its own draw.
            # The pool pass already solved the Laplace factors of every kept state: reuse them.
            is_ = self._is
            laplace = tuple(self._gatherPool(t, idx_out) for t in (is_._modes, is_._chol_H))
            _, ffx, sigma_eps = is_._logPriorGlobals(out)
            out.data['local']['samples'] = refreshRfxIS2(
                ffx,
                out.sigma_rfx,
                sigma_eps,
                is_.y,
                is_.X,
                is_.Z,
                is_.mask_n,
                is_.mask_m,
                self.likelihood_family,
                out.rfx,
                is_.n_inner,
                L_corr=is_._getLCorr(out),
                laplace=laplace,
            )
        t1 = time.perf_counter()
        out.tpd = (proposal.tpd or 0.0) + (t1 - t0)
        diagnostics = {'accept_rate': accept_rate}
        if self.n_eff_target is not None:
            diagnostics['suggested_n_samples'] = suggestPoolSize(accept_rate, self.n_eff_target)
        return out, diagnostics


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


def runIMH(
    model: Approximator,
    data: dict[str, torch.Tensor],
    cfg: argparse.Namespace,
) -> tuple[Proposal, dict]:
    """Draw a flow proposal and correct it with IMH.

    cfg fields
    ----------
    n_chains       : int  — number of independent chains (default 4)
    n_steps        : int  — steps per chain including burnin (default 250)
    imh_burnin     : int  — burnin steps to discard (default 25)
    imh_mode       : str  — 'global' | 'marginal' | 'joint' | 'laplace'
                     defaults to 'marginal' for Normal, 'laplace' otherwise
    imh_n_eff_target : int | None — effective-draw target for the advisory per-dataset
                     pool-size suggestion returned in the diagnostics (default 700,
                     the sweep-calibrated value; None disables). Inference always
                     runs at n_chains × n_steps regardless.
    rescale        : bool
    likelihood_family : int

    'marginal' and 'laplace' redraw the rfx from their conditional at the accepted globals,
    so the flow's local posterior is skipped for them (Approximator.estimate(local=False)).
    """
    lf = getattr(cfg, 'likelihood_family', 0)
    n_chains = getattr(cfg, 'n_chains', 4)
    n_steps = getattr(cfg, 'n_steps', 250)
    burnin = getattr(cfg, 'imh_burnin', 25)
    n_eff_target = getattr(cfg, 'imh_n_eff_target', N_EFF_TARGET)
    default_mode = 'marginal' if lf == 0 else 'laplace'
    mode: Mode = getattr(cfg, 'imh_mode', default_mode)

    proposal = model.estimate(data, n_samples=n_chains * n_steps, local=mode in ('global', 'joint'))

    if cfg.rescale:
        proposal.rescale(data['sd_y'])
        data = rescaleData(data)

    sampler = MetropolisSampler(
        data,
        n_chains=n_chains,
        n_steps=n_steps,
        burnin=burnin,
        mode=mode,
        likelihood_family=lf,
        n_eff_target=n_eff_target,
    )
    return sampler(proposal)
