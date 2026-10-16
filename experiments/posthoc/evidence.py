"""Evidence check: is the mean marginal IS weight a usable estimate of log p(D)?

The paper claims that, because the flow density is normalized and prior and
marginal likelihood are evaluated exactly, the mean importance weight over the proposal
pool is an unbiased estimate of the marginal likelihood (Gaussian outcomes), so Bayes
factors cost a single batched forward pass. This script tests that claim on the Gaussian
oracle sets against a bridge-sampling reference.

Per dataset (standardized space — the space the flow density lives in):
  IS         : log p(D) ≈ logsumexp(log_w) - log S from ImportanceSampler(marginal=True),
               for pool prefixes S in --pool-sizes (flow draws are i.i.d.).
  bridgeNuts : Meng-Wong iterative bridge sampling on the NUTS reference draws (level 2 of
               the ladder, test.nuts2.npz, thinned to N_REF_DRAWS), warped-Gaussian proposal
               fitted on one half of the draws and evaluated on the other, with the halves
               swapped as a noise floor. Each row records whether the reference converged
               (utils/evaluation.nutsConverged); the tables keep the converged rows only.
               The unnormalized target reuses ImportanceSampler.unnormalizedPosterior
               (marginal likelihood + priors) in unconstrained coordinates.
  bridgeImh  : the same bridge on MB-IMH draws (flow + independence MH, exact marginal
               target) — validates the IMH-bridge that serves as the reference for the
               reduced model in the nested comparison, where no NUTS draws exist.
  nested     : for datasets with q >= 2, the same data is also fit without the random
               slope (random intercept only, slope kept as a fixed effect). ln BF from IS
               vs. bridge reference; Jeffreys category agreement.
  rho check  : for correlated datasets, posterior mean of the first rfx correlation under
               raw flow / IS / IMH vs NUTS.

GLMMs (--family bernoulli/poisson): no closed-form marginal exists, so
  IS         : the unbiased IS² evidence (LaplaceImportanceSampler(n_inner=K)), next to the
               biased Laplace evidence (n_inner=0) and sd(log p̂(y|θ)) from two IS² passes;
  bridge*    : the same bridges with the marginal likelihood from adaptive Gauss-Hermite
               quadrature (logMarginalLikelihoodAGQ, nodes per dim from agqNodes), a
               deterministic reference independent of the IS² draws;
  IMH        : pseudo-marginal IMH (IS² weights) for bridgeImh and the reduced model;
  fidelity   : posterior means/sds of β and σ_rfx under raw flow, Laplace-IMH and
               pseudo-marginal IMH vs NUTS — does the pseudo-marginal chain target the
               exact posterior where the Laplace chain does not?

--t-only adds the estimator of Api.sample(evidence=True) to an existing CSV without NUTS or
bridges: flow + IMH draws as above plus the paper's random-walk moves where the IMH min bulk ESS
is below 150 (posterior_eval.MOVES_RW, the Api's ESS flag), on --device, then IS from a Student-t
fitted to them (posthoc/evidence.tEvidence), per row and nested reduced fit. Its T_COLUMNS go to
t/{stem}[_shard].csv; --summarize-only joins them onto {stem}.csv by idx.

Outputs: {out_dir}/{prefix}{size}_{split}_n{n_ds}.csv (per-dataset rows; prefix '' for Normal,
'{family}_k{K}_' otherwise), .md (summary tables) and .png (diagnostic figure);
--summarize-only additionally writes evidence_normal.tex / evidence_discrete.tex (converged
references only) and evidence_paper.md (the numbers quoted in the paper), with the t estimator
where the CSVs carry it.

Run from the repo root:
    uv run python experiments/posthoc/evidence.py --sizes small --n-datasets 32
    uv run python experiments/posthoc/evidence.py --sizes small medium --n-datasets 512 --nested 128
    # after all sizes ran (possibly on different nodes): rebuild summaries + the combined table
    uv run python experiments/posthoc/evidence.py --sizes small medium large huge --summarize-only
    uv run python experiments/posthoc/evidence.py --family bernoulli --sizes small --n-datasets 32
    # parallel shards (one process each), then merge + summarize
    uv run python experiments/posthoc/evidence.py --family bernoulli --sizes small --n-datasets 512 --shard 3 --n-shards 16
    uv run python experiments/posthoc/evidence.py --family bernoulli --sizes small --n-datasets 512 --summarize-only
    # t estimator on the rows of an existing CSV (sharded alike), joined by --summarize-only
    uv run python experiments/posthoc/evidence.py --sizes small --n-datasets 512 --t-only --device cuda --shard 0 --n-shards 4
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import distributions as D

from metabeta.utils.experiments import DATA_DIR, REPO_ROOT, experimentResultsPath

sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from build_ckpt import BEST_SEEDS, _ckpt_dir  # noqa: E402

from metabeta.models.approximator import Approximator  # noqa: E402
from metabeta.posthoc.evidence import isSummary, tEvidence  # noqa: E402
from metabeta.posthoc.importance import ImportanceSampler  # noqa: E402
from metabeta.posthoc.laplace_glmm import (  # noqa: E402
    LaplaceImportanceSampler,
    logMarginalLikelihoodAGQ,
)
from metabeta.posthoc.metropolis import (  # noqa: E402
    IS2_N_INNER,
    LocalMoves,
    MetropolisSampler,
    bulkEss,
)
from metabeta.utils.config import ApproximatorConfig  # noqa: E402
from metabeta.utils.constants import hasSigmaEps  # noqa: E402
from metabeta.utils.psis import psislw  # noqa: E402
from metabeta.utils.dataloader import Collection, collateGrouped, toDevice  # noqa: E402
from metabeta.utils.evaluation import NUTS_DIAG_KEYS, nutsConverged  # noqa: E402
from metabeta.utils.fits import REFERENCE_TAG, loadFits  # noqa: E402
from metabeta.utils.families import logProbCorrRfx  # noqa: E402
from metabeta.utils.posterior_eval import MOVES_RW  # noqa: E402
from metabeta.utils.preprocessing import logJacobianStandardization  # noqa: E402
from metabeta.utils.regularization import (  # noqa: E402
    corrToLower,
    corrToUnconstrained,
    unconstrainedToCholesky,
)
from metabeta.utils.results import Proposal  # noqa: E402

IMH_CHAINS = 4
IMH_BURNIN = 25
N_REF_DRAWS = 4000  # reference draws per dataset: level 2 thinned like the composite NUTS fit
SIZES = ('small', 'medium', 'large', 'huge')
FAMILIES = {'normal': 0, 'bernoulli': 1, 'poisson': 2}
TARGET_CHUNK = 1000  # bridge-target draws per AGQ pass (bounds the (m, n, s) intermediates)
JEFFREYS_EDGES = np.log([3.0, 10.0, 30.0, 100.0])  # ln BF thresholds (Jeffreys, natural log)
# test.npz keys that collateStats packs into batch['stats']
ANALYTICAL_STATS = (
    'beta_est',
    'sigma_rfx_est',
    'blup_est',
    'blup_var',
    'sigma_eps_est',
    'phi_pearson',
    'Psi',
)
COLOR = '#3B6FB6'
# CSVs written while the legacy 'z' prior coordinates were still reported carry an `_r` infix
LEGACY_COLUMNS = {'logev_is_r_': 'logev_is_', 'k_r_': 'k_', 'eff_r_': 'eff_', '_is_r': '_is'}
# --t-only; ess/moved: runImh's min IMH bulk ESS and whether the moves ran
T_COLUMNS = ('logev_t', 'k_t', 'eff_t', 'logev_t_red', 'k_t_red', 'logbf_t')
T_COLUMNS += ('ess_t', 'moved_t', 'ess_t_red', 'moved_t_red')


# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--family', default='normal', choices=list(FAMILIES))
    p.add_argument('--sizes', nargs='+', default=['small'], choices=SIZES)
    p.add_argument('--n-inner', type=int, default=IS2_N_INNER, help='IS² draws per group (GLMMs)')
    p.add_argument('--split', default='test', choices=['test'], help='only the test partition carries NUTS draws (test.nuts.npz)')
    p.add_argument('--prefix', default='latest', help='checkpoint prefix (latest = the checkpoint of the paper tables)')
    p.add_argument('--n-datasets', type=int, default=32, help='datasets per size (first n of the split)')
    p.add_argument('--pool-sizes', nargs='+', type=int, default=[1000, 2000, 4000], help='IS pool prefixes; the largest is drawn')
    p.add_argument('--n-bridge', type=int, default=2000, help='proposal draws per bridge run (posterior draws: half of the available)')
    p.add_argument('--nested', type=int, default=16, help='max number of q>=2 datasets for the nested comparison (0 disables)')
    p.add_argument('--imh-bridge', type=int, default=64, help='run the full-model IMH bridge (a cross-check of the reduced-model reference) on the first n datasets only; nested datasets always get it')
    p.add_argument('--shard', type=int, default=0, help='this process handles datasets i with i %% n_shards == shard')
    p.add_argument('--n-shards', type=int, default=1, help='split the datasets over this many processes; --summarize-only merges the shard CSVs')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu', help='device for flow sampling; everything else runs on cpu in float64 (--t-only: on this device)')
    p.add_argument('--out-dir', type=Path, default=experimentResultsPath('evidence'))
    p.add_argument('--summarize-only', action='store_true', help='regenerate md/png/tex from existing CSVs without recomputing')
    p.add_argument('--t-only', action='store_true', help='add the t-proposal evidence to the rows of the existing CSV (no NUTS, no bridge)')
    return p.parse_args()
# fmt: on


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def loadModel(ckpt: Path) -> tuple[Approximator, int]:
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = Approximator(ApproximatorConfig(**payload['model_cfg']))
    model.load_state_dict(payload['model_state'])
    model.eval()
    return model, payload['epoch']


def loadNuts(npz_path: Path, n_ds: int) -> dict[str, np.ndarray]:
    """Global draws (standardized space) and diagnostics of the NUTS reference (REFERENCE_TAG)
    for the first n_ds datasets, under ``nuts_*`` keys.

    Draws are thinned to N_REF_DRAWS with the composite fit's stride (fit.py, _composite),
    which keeps every chain, so the two bridge halves are still chains 0-1 and 2-3.
    ``nuts_converged`` is the convergence criterion of the paper. Every key is read once;
    the rfx draws (GBs) are never touched.
    """
    tag = REFERENCE_TAG
    draws = ('ffx', 'sigma_rfx', 'sigma_eps', 'corr_rfx')
    keys = [f'{tag}_{k}' for k in (*draws, *NUTS_DIAG_KEYS)]
    fits = {k: v[:n_ds] for k, v in loadFits(npz_path, tag, keys=keys).items()}
    out = {'nuts_converged': nutsConverged(fits, tag)}
    for k, v in fits.items():
        field = k[len(tag) + 1 :]
        if field in draws:
            axis = 2 if field == 'corr_rfx' else -1  # corr: (b, 1, s, q, q), else (b, ..., s)
            v = v.take(np.arange(0, v.shape[axis], v.shape[axis] // N_REF_DRAWS), axis=axis)
        out[f'nuts_{field}'] = v
    return out


def loadCsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for old, new in LEGACY_COLUMNS.items():
        df.columns = [c.replace(old, new) for c in df.columns]
    return df


def toDouble(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        k: v.double() if torch.is_tensor(v) and v.is_floating_point() else v
        for k, v in batch.items()
    }


def proposalDouble(p: Proposal, n_samples: int | None = None) -> Proposal:
    """float64 copy of a flow proposal, optionally trimmed to its first n_samples draws."""
    s = p.n_samples if n_samples is None else n_samples

    def cut(t: torch.Tensor, dim: int) -> torch.Tensor:
        return t.narrow(dim, 0, s).double().contiguous()

    proposed = {
        'global': {'samples': cut(p.samples_g, 1), 'log_prob': cut(p.log_prob_g, 1)},
        'local': {'samples': cut(p.samples_l, 2), 'log_prob': cut(p.log_prob_l, 2)},
    }
    return Proposal(proposed, has_sigma_eps=p.has_sigma_eps, d_corr=p.d_corr)


def reducedItem(item: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Same data without the random slope(s): random intercept only (q=1), slopes stay fixed."""
    out = dict(item)
    out['q'] = np.array(1, dtype=item['q'].dtype)
    out['Z'] = item['Z'].copy()
    out['Z'][:, 1:] = 0.0
    for key in ('tau_rfx', 'sigma_rfx'):
        out[key] = item[key].copy()
        out[key][1:] = 0.0
    out['rfx'] = item['rfx'].copy()
    out['rfx'][:, 1:] = 0.0
    out['eta_rfx'] = np.zeros_like(item['eta_rfx'])
    out['corr_rfx'] = np.eye(item['corr_rfx'].shape[-1], dtype=item['corr_rfx'].dtype)
    # the precomputed analytical stats belong to the full model; without them the flow
    # computes them from the reduced data (Approximator._resolveStats)
    for key in ANALYTICAL_STATS:
        out.pop(key, None)
    return out


# ---------------------------------------------------------------------------
# Unnormalized posterior over unconstrained globals (one dataset) + bridge sampling
# ---------------------------------------------------------------------------


def agqNodes(q: int) -> int:
    """Gauss-Hermite nodes per rfx dim: ≤ ~150 tensor-product nodes, ≥ 3 per dim; at q = 2 the
    12-node rule is exact to ~1e-7 nats per group (tests/utils/test_is2.py)."""
    return {1: 20, 2: 12, 3: 5, 4: 4}.get(q, 3)


class Target:
    """log p(y | θ_g) p(θ_g) |J| over u = (β, log σ_rfx[, log σ_eps][, z_corr]) for one dataset.

    The marginal likelihood is exact for Normal (ImportanceSampler.unnormalizedPosterior) and
    adaptive Gauss-Hermite quadrature for GLMMs; β/σ priors come from the same sampler on a
    padded constrained Proposal. The LKJ prior is added as a density over z in the dataset's
    own dimension q_i, so the target is independent of the flow's coordinate convention.
    """

    def __init__(
        self, batch64: dict[str, torch.Tensor], d_corr_model: int, likelihood_family: int
    ) -> None:
        self.lf = likelihood_family
        self.has_eps = hasSigmaEps(likelihood_family)
        self.is_ = ImportanceSampler(
            batch64, marginal=self.lf == 0, corr_prior=False, likelihood_family=self.lf
        )
        self.d_max = batch64['X'].shape[-1]
        self.q_max = batch64['Z'].shape[-1]
        self.m = batch64['X'].shape[1]
        self.d_corr_max = d_corr_model
        self.d_i = int(batch64['mask_d'].sum())
        self.q_i = int(batch64['mask_q'].sum())
        self.eta = batch64['eta_rfx'].double()  # (1,)
        self.corr = bool(self.eta.item() > 0) and self.q_i >= 2
        self.d_corr_i = self.q_i * (self.q_i - 1) // 2 if self.corr else 0
        # blocks of u
        self.sl_ffx = slice(0, self.d_i)
        self.sl_sigma = slice(self.d_i, self.d_i + self.q_i + int(self.has_eps))
        self.sl_corr = slice(self.sl_sigma.stop, self.sl_sigma.stop + self.d_corr_i)
        self.dim = self.sl_corr.stop
        self.D_g = self.d_max + self.q_max + int(self.has_eps) + self.d_corr_max

    def toProposal(self, u: torch.Tensor) -> Proposal:
        s = u.shape[0]
        g = u.new_zeros(s, self.D_g)
        g[:, : self.d_i] = u[:, self.sl_ffx]
        sigmas = u[:, self.sl_sigma].exp()
        g[:, self.d_max : self.d_max + self.q_i] = sigmas[:, : self.q_i]
        if self.has_eps:
            g[:, self.d_max + self.q_max] = sigmas[:, -1]
        if self.corr:
            L = unconstrainedToCholesky(u[:, self.sl_corr], self.q_i)
            off = self.d_max + self.q_max + int(self.has_eps)
            g[:, off : off + self.d_corr_i] = corrToLower(L @ L.mT)
        proposed = {
            'global': {'samples': g.unsqueeze(0), 'log_prob': g.new_zeros(1, s)},
            'local': {
                'samples': g.new_zeros(1, self.m, s, self.q_max),
                'log_prob': g.new_zeros(1, self.m, s),
            },
        }
        return Proposal(proposed, has_sigma_eps=self.has_eps, d_corr=self.d_corr_max)

    def _logJoint(self, p: Proposal) -> torch.Tensor:
        """log p(y | θ_g) + log p(θ_g) without the LKJ term, (s,)."""
        if self.lf == 0:
            ll, lp = self.is_.unnormalizedPosterior(p)
            return (ll + lp)[0]
        is_ = self.is_
        lp, ffx, sigma_eps = is_._logPriorGlobals(p)
        ll = logMarginalLikelihoodAGQ(
            ffx,
            p.sigma_rfx,
            sigma_eps,
            is_.y,
            is_.X,
            is_.Z,
            is_.mask_n,
            is_.mask_m,
            self.lf,
            n_nodes=agqNodes(self.q_i),
            L_corr=is_._getLCorr(p),
        )
        return (ll + lp)[0]

    def logProb(self, u: torch.Tensor) -> torch.Tensor:
        lj = torch.cat([self._logJoint(self.toProposal(c)) for c in u.split(TARGET_CHUNK)])
        out = lj + u[:, self.sl_sigma].sum(-1)  # |dσ/d log σ|
        if self.corr:
            z = u.new_zeros(1, u.shape[0], self.d_corr_max)
            z[0, :, : self.d_corr_i] = u[:, self.sl_corr]
            q_active = torch.tensor([self.q_i])
            out = out + logProbCorrRfx(z, self.q_max, self.eta, q_active=q_active)[0]
        return torch.nan_to_num(out, nan=-math.inf, posinf=-math.inf)

    def fromConstrained(
        self,
        ffx: torch.Tensor,
        sigma_rfx: torch.Tensor,
        sigma_eps: torch.Tensor | None,
        corr: torch.Tensor | None,
    ) -> torch.Tensor:
        """(s, d_i), (s, q_i), (s,) or None, (s, q, q) constrained draws → (s, dim) unconstrained."""
        parts = [
            ffx[:, : self.d_i].double(),
            sigma_rfx[:, : self.q_i].double().clamp_min(1e-8).log(),
        ]
        if self.has_eps:
            parts.append(sigma_eps.double().clamp_min(1e-8).log().unsqueeze(-1))
        if self.corr:
            parts.append(corrToUnconstrained(corr[:, : self.q_i, : self.q_i].double()))
        return torch.cat(parts, -1)

    def fromProposal(self, p: Proposal) -> torch.Tensor:
        corr = p.corr_rfx[0] if self.corr else None
        sigma_eps = p.sigma_eps[0] if self.has_eps else None
        return self.fromConstrained(p.ffx[0], p.sigma_rfx[0], sigma_eps, corr)

    def fromNuts(self, nuts: dict[str, np.ndarray], i: int) -> torch.Tensor:
        ffx = torch.as_tensor(nuts['nuts_ffx'][i]).T  # (S, d_max)
        sigma_rfx = torch.as_tensor(nuts['nuts_sigma_rfx'][i]).T  # (S, q_max)
        sigma_eps = torch.as_tensor(nuts['nuts_sigma_eps'][i, 0]) if self.has_eps else None  # (S,)
        corr = torch.as_tensor(nuts['nuts_corr_rfx'][i, 0]) if self.corr else None  # (S, q, q)
        return self.fromConstrained(ffx, sigma_rfx, sigma_eps, corr)


def _bridgeOnce(
    log_target,
    u_fit: torch.Tensor,
    u_est: torch.Tensor,
    n_prop: int,
    gen: torch.Generator,
    max_iter: int = 5000,
    tol: float = 1e-10,
) -> tuple[float, int]:
    """Meng-Wong optimal bridge with a Gaussian proposal fitted on u_fit, estimated on u_est."""
    mean = u_fit.mean(0)
    cov = torch.cov(u_fit.T) + 1e-6 * torch.eye(u_fit.shape[1], dtype=u_fit.dtype)
    g = D.MultivariateNormal(mean, scale_tril=torch.linalg.cholesky(cov))
    eps = torch.randn(n_prop, u_fit.shape[1], generator=gen, dtype=u_fit.dtype)
    u_prop = mean + eps @ g.scale_tril.T
    l1 = log_target(u_est) - g.log_prob(u_est)  # posterior draws
    l2 = log_target(u_prop) - g.log_prob(u_prop)  # proposal draws
    n1, n2 = l1.shape[0], l2.shape[0]
    ls1, ls2 = math.log(n1 / (n1 + n2)), math.log(n2 / (n1 + n2))
    log_r = (torch.logsumexp(l2, 0) - math.log(n2)).item()  # plain IS from g as init
    for it in range(1, max_iter + 1):
        num = torch.logsumexp(l2 - torch.logaddexp(ls1 + l2, torch.tensor(ls2 + log_r)), 0)
        den = torch.logsumexp(-torch.logaddexp(ls1 + l1, torch.tensor(ls2 + log_r)), 0)
        new = (num - math.log(n2) - den + math.log(n1)).item()
        converged = abs(new - log_r) < tol
        log_r = new
        if converged:
            break
    return log_r, it


def bridge(log_target, u_post: torch.Tensor, n_prop: int, gen: torch.Generator) -> dict:
    """Two-half bridge estimate: fit/estimate halves and their swap (noise floor)."""
    half = u_post.shape[0] // 2
    a, b = u_post[:half], u_post[half : 2 * half]
    ev_ab, it_ab = _bridgeOnce(log_target, a, b, n_prop, gen)
    ev_ba, it_ba = _bridgeOnce(log_target, b, a, n_prop, gen)
    return {'log_ev': ev_ab, 'log_ev_swap': ev_ba, 'n_iter': max(it_ab, it_ba)}


# ---------------------------------------------------------------------------
# Per-dataset pieces
# ---------------------------------------------------------------------------


def isEvidence(
    proposal: Proposal,
    batch64: dict[str, torch.Tensor],
    pool_sizes: list[int],
    lf: int,
    n_inner: int,
) -> dict:
    """IS log-evidence, PSIS k and efficiency per pool prefix; 'weights' = PSIS weights at S_max.

    GLMMs use the unbiased IS² weights and add, at S_max, the (biased) Laplace evidence and
    sd_loglik = sd(log p̂(y | θ)) under the PSIS weights, from two independent IS² passes —
    the IS² noise Tran et al. (arXiv:1309.3339) tune K by.
    """
    if lf == 0:
        sampler = ImportanceSampler(batch64, marginal=True, corr_prior=True, pareto=True)
    else:
        sampler = LaplaceImportanceSampler(
            batch64, n_inner=n_inner, corr_prior=True, pareto=True, likelihood_family=lf
        )
    out = sampler(proposalDouble(proposal))
    lw = out.is_results['log_w_raw'][0]  # (S,)
    res = {'weights': out.is_results['weights'][0]}
    for s in pool_sizes:
        lw_s = lw[:s]
        lw_np, k = psislw(lw_s.unsqueeze(0).numpy())
        w = torch.softmax(torch.as_tensor(lw_np[0]), -1)
        res[f'logev_is_s{s}'] = (torch.logsumexp(lw_s, 0) - math.log(s)).item()
        res[f'k_s{s}'] = float(k[0])
        res[f'eff_s{s}'] = float(1.0 / (s * (w**2).sum()))
    if lf != 0:
        s_max = max(pool_sizes)
        laplace = LaplaceImportanceSampler(
            batch64, corr_prior=True, pareto=False, constrain=False, likelihood_family=lf
        )
        res[f'logev_laplace_s{s_max}'] = laplace(proposalDouble(proposal)).log_evidence.item()
        ll_a, _ = sampler.unnormalizedPosterior(proposalDouble(proposal))
        ll_b, _ = sampler.unnormalizedPosterior(proposalDouble(proposal))
        half_sq = (ll_a - ll_b)[0].square() / 2  # unbiased for Var(log p̂) per draw
        res['sd_loglik'] = float((res['weights'] * half_sq).sum().sqrt())
    return res


def poolEvidence(
    batch64: dict[str, torch.Tensor], flow: Proposal, lf: int, n_inner: int
) -> dict[str, np.ndarray]:
    """isEvidence batched over datasets on the batch's device, for the full pool only and without
    its Laplace extras: log-evidence, PSIS k and efficiency per dataset, (b,) each."""
    flow = proposalDouble(flow)
    if lf == 0:
        sampler = ImportanceSampler(batch64, marginal=True, corr_prior=True)
    else:
        sampler = LaplaceImportanceSampler(
            batch64, n_inner=n_inner, corr_prior=True, likelihood_family=lf
        )
    ll, lp = sampler.unnormalizedPosterior(flow)
    return isSummary((ll + lp - flow.log_prob_g).cpu())


def runImh(
    proposal: Proposal,
    batch64: dict[str, torch.Tensor],
    lf: int,
    n_inner: int = 0,
    moves: LocalMoves | None = None,
) -> tuple[Proposal, dict]:
    """Marginal IMH (Normal) or Laplace IMH, pseudo-marginal when n_inner > 0 (GLMMs), followed by
    the random-walk moves (if given) where their gate fires; for one dataset, its mean acceptance,
    the IMH chains' min bulk ESS over the globals (the gate's statistic) and whether moves ran."""
    n_steps = proposal.n_samples // IMH_CHAINS
    sampler = MetropolisSampler(
        batch64,
        n_chains=IMH_CHAINS,
        n_steps=n_steps,
        burnin=IMH_BURNIN,
        mode='marginal' if lf == 0 else 'laplace',
        likelihood_family=lf,
        n_eff_target=None,
        n_inner=n_inner,
        moves=moves,
    )
    p_imh, diag = sampler(proposalDouble(proposal, IMH_CHAINS * n_steps))
    ess = diag['ess'] if moves else bulkEss(p_imh.samples_g, IMH_CHAINS)
    return p_imh, {
        'accept': float(diag['accept_rate'].mean()),
        'ess': float(np.nanmin(ess.numpy())),
        'moved': bool(moves and diag['moved'][0]),
    }


def fidelity(p: Proposal, nuts: dict[str, np.ndarray], i: int, target: Target) -> dict:
    """Mean |Δ posterior mean| / sd_NUTS and mean |log sd ratio| of β and σ_rfx vs NUTS."""
    blocks = {
        'ffx': (p.ffx[0, :, : target.d_i], nuts['nuts_ffx'][i, : target.d_i].T),
        'sig': (p.sigma_rfx[0, :, : target.q_i], nuts['nuts_sigma_rfx'][i, : target.q_i].T),
    }
    out = {}
    for name, (a, b) in blocks.items():
        a, b = a.double(), torch.as_tensor(b).double()
        sd_b = b.std(0)
        out[f'z_{name}'] = float(((a.mean(0) - b.mean(0)).abs() / sd_b).mean())
        out[f'lsd_{name}'] = float((a.std(0) / sd_b).log().abs().mean())
    return out


def fitDataset(
    model: Approximator,
    item: dict[str, np.ndarray],
    args: argparse.Namespace,
    pool_sizes: list[int],
    gen: torch.Generator,
    seed: int,
    laplace_imh: bool = False,
    imh_bridge: bool = True,
) -> dict:
    """Flow proposal, IS evidence per pool prefix, IMH refinement and the bridge on its draws.

    For GLMMs the IMH is pseudo-marginal (exact target, so its bridge is a valid reference);
    laplace_imh additionally runs the Laplace IMH on the same pool for the fidelity check.
    The bridge dominates the cost (AGQ target for GLMMs); imh_bridge=False skips it.
    """
    lf = FAMILIES[args.family]
    n_inner = args.n_inner if lf != 0 else 0
    batch = collateGrouped([item])
    batch64 = toDouble(batch)
    target = Target(batch64, model.d_corr, lf)
    torch.manual_seed(seed)
    with torch.no_grad():
        proposal = model.estimate(toDevice(dict(batch), args.device), n_samples=max(pool_sizes))
    proposal.to('cpu')
    p_imh, imh = runImh(proposal, batch64, lf, n_inner)
    fit = {
        'batch': batch,
        'target': target,
        'proposal': proposal,
        'is': isEvidence(proposal, batch64, pool_sizes, lf, n_inner),
        'p_imh': p_imh,
        'imh_accept': imh['accept'],
    }
    if imh_bridge:
        fit['bridge_imh'] = bridge(target.logProb, target.fromProposal(p_imh), args.n_bridge, gen)
    if laplace_imh:
        fit['p_imh_laplace'], imh_laplace = runImh(proposal, batch64, lf)
        fit['imh_accept_laplace'] = imh_laplace['accept']
    return fit


def tFit(
    model: Approximator, item: dict[str, np.ndarray], args: argparse.Namespace, seed: int
) -> dict[str, float]:
    """tEvidence's log_ev, k and eff on fitDataset's flow + IMH draws and the paper's gated
    random-walk moves, run on --device, with as many t draws as the flow pool; log_ev on the
    standardized scale of the bridges. Also runImh's ESS and moves flag."""
    lf = FAMILIES[args.family]
    s_max = max(args.pool_sizes)
    batch = toDevice(collateGrouped([item]), args.device)
    batch64 = toDouble(batch)
    torch.manual_seed(seed)
    with torch.no_grad():
        proposal = model.estimate(batch, n_samples=s_max)
    p_imh, imh = runImh(proposal, batch64, lf, args.n_inner if lf != 0 else 0, MOVES_RW)
    gen = torch.Generator(device=args.device).manual_seed(seed)
    res = tEvidence(batch64, p_imh, lf, s_max, gen)
    if p_imh.has_sigma_eps:  # tEvidence moved it to the raw response scale
        res['log_ev'] -= logJacobianStandardization(batch64).cpu().numpy()
    return {k: float(v[0]) for k, v in res.items()} | {'ess': imh['ess'], 'moved': imh['moved']}


def jeffreys(log_bf: float) -> float:
    """Signed Jeffreys category of ln BF: 0 anecdotal … ±4 decisive; NaN for a non-finite BF."""
    if not np.isfinite(log_bf):
        return np.nan
    return float(np.sign(log_bf) * np.searchsorted(JEFFREYS_EDGES, abs(log_bf), side='right'))


def categoryAgreement(a: pd.Series, b: pd.Series) -> float:
    """Share of rows whose Jeffreys categories agree; a non-finite estimate never agrees."""
    ca, cb = a.map(jeffreys), b.map(jeffreys)
    return float(((ca == cb) & ca.notna() & cb.notna()).mean())


def nestedRows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['logbf_ref'].notna()] if 'logbf_ref' in df else df.iloc[0:0]


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _absStats(delta: pd.Series) -> dict:
    a = delta.abs()
    return {
        'n': int(a.notna().sum()),
        'non-finite': int((~np.isfinite(delta)).sum()),
        'median |Δ|': a.median(),
        'q90 |Δ|': a.quantile(0.9),
        'frac |Δ|<0.5': (a < 0.5).mean(),
        'frac |Δ|<1': (a < 1.0).mean(),
        'median Δ': delta.median(),
    }


def _table(rows: dict, floatfmt: str = '.3f') -> str:
    return pd.DataFrame(rows).T.to_markdown(floatfmt=floatfmt)


def summarize(df: pd.DataFrame, pool_sizes: list[int], label: str) -> str:
    s_max = max(pool_sizes)
    ref = df['logev_bridge_nuts']
    d_max = df[f'logev_is_s{s_max}'] - ref
    k_max = df[f'k_s{s_max}']
    lines = [f'# {label}: IS log-evidence vs bridge sampling', '']

    lines += ['## Reference noise floor (nats)', '']
    lines.append(
        _table(
            {
                'bridgeNuts vs swapped halves': _absStats(ref - df['logev_bridge_nuts_swap']),
                'bridgeImh vs bridgeNuts': _absStats((df['logev_bridge_imh'] - ref).dropna()),
            }
        )
    )

    lines += ['', '## IS − bridgeNuts by pool size', '']
    rows = {f'S={s}': _absStats(df[f'logev_is_s{s}'] - ref) for s in pool_sizes}
    if f'logev_laplace_s{s_max}' in df:
        rows[f'Laplace, S={s_max}'] = _absStats(df[f'logev_laplace_s{s_max}'] - ref)
    lines.append(_table(rows))
    if 'sd_loglik' in df:
        sd = df['sd_loglik']
        lines += [
            '',
            f'IS² noise sd(log p̂(y|θ)) under the posterior: median {sd.median():.3f}, '
            f'q90 {sd.quantile(0.9):.3f}, max {sd.max():.3f}',
        ]

    lines += ['', f'## By PSIS k (S={s_max})', '']
    bins = (
        ('k ≤ 0.5', k_max <= 0.5),
        ('0.5 < k ≤ 0.7', (k_max > 0.5) & (k_max <= 0.7)),
        ('k > 0.7', k_max > 0.7),
    )
    lines.append(_table({name: _absStats(d_max[sel]) for name, sel in bins}))
    rho_k = d_max.abs().corr(k_max, method='spearman')
    rho_a = d_max.abs().corr(df['imh_accept'], method='spearman')
    lines += ['', f'Spearman(|Δ|, k) = {rho_k:.2f}, Spearman(|Δ|, IMH acceptance) = {rho_a:.2f}']

    corr = df[df['corr_active']]
    if len(corr):
        lines += ['', f'## Correlated datasets (n={len(corr)}), S={s_max}', '']
        lines.append(_table({'IS': _absStats(d_max[corr.index])}))
        lines += ['', 'Posterior mean of the first rfx correlation vs NUTS:', '']
        rows = {
            name: {
                'mean |Δρ|': (corr[col] - corr['rho_nuts']).abs().mean(),
                'mean Δρ': (corr[col] - corr['rho_nuts']).mean(),
            }
            for name, col in (('raw flow', 'rho_raw'), ('IS', 'rho_is'), ('IMH', 'rho_imh'))
        }
        lines.append(_table(rows, floatfmt='.4f'))

    if 'z_ffx_imhPM' in df:
        lines += [
            '',
            '## Posterior fidelity vs NUTS (mean over parameters, median over datasets)',
            '',
        ]
        rows = {}
        for tag in ('raw', 'imhLaplace', 'imhPM'):
            rows[tag] = {
                f'{stat} {block}': df[f'{stat}_{block}_{tag}'].median()
                for block in ('ffx', 'sig')
                for stat in ('z', 'lsd')
            }
        lines.append(_table(rows))
        lines += [
            '',
            'z = |Δ posterior mean| / sd_NUTS, lsd = |log sd ratio|; NUTS Monte Carlo floor of z '
            f'≈ 1/√ESS, median min-ESS {df["nuts_min_ess"].median():.0f}. '
            f'Paired: imhPM closer than imhLaplace in z sig on '
            f'{(df["z_sig_imhPM"] < df["z_sig_imhLaplace"]).mean():.0%} of datasets.',
        ]

    nested = nestedRows(df)
    if len(nested):
        lines += ['', f'## Nested comparison: random slope vs intercept-only (n={len(nested)})', '']
        rows = {}
        for name, col in (('IS', 'logbf_is'), ('bridgeImh (both)', 'logbf_ref_imh')):
            rows[name] = _absStats(nested[col] - nested['logbf_ref'])
            rows[name]['same Jeffreys cat.'] = categoryAgreement(nested[col], nested['logbf_ref'])
        lines.append(_table(rows))
        cats = nested['logbf_ref'].map(jeffreys).dropna().astype(int).value_counts().sort_index()
        lines += [
            '',
            'Reference ln BF categories (signed Jeffreys, + favours the random slope): '
            + ', '.join(f'{k}: {v}' for k, v in cats.items()),
        ]
    lines.append('')
    return '\n'.join(lines)


def plot(df: pd.DataFrame, pool_sizes: list[int], path: Path, label: str) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    s_max = max(pool_sizes)
    ref = df['logev_bridge_nuts']
    nested = nestedRows(df)
    n_ax = 3 if len(nested) else 2
    fig, axes = plt.subplots(1, n_ax, figsize=(4.2 * n_ax, 3.6))
    dots = dict(s=14, alpha=0.8, color=COLOR, edgecolor='none')

    ax = axes[0]
    ax.scatter(df[f'k_s{s_max}'], df[f'logev_is_s{s_max}'] - ref, **dots)
    ax.axhline(0, color='0.6', lw=0.8)
    ax.axvline(0.7, color='0.6', lw=0.8, ls=':')
    ax.set_xlabel('PSIS k')
    ax.set_ylabel(f'log p̂(D) IS (S={s_max}) − bridge (nats)')

    ax = axes[1]
    d = np.stack([(df[f'logev_is_s{s}'] - ref).abs().to_numpy() for s in pool_sizes], 1)
    for row in d:
        ax.plot(pool_sizes, row, color=COLOR, alpha=0.15, lw=0.8)
    ax.plot(pool_sizes, np.nanmedian(d, 0), color=COLOR, lw=2, marker='o', label='median')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(pool_sizes)
    ax.set_xticklabels([str(s) for s in pool_sizes])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('pool size S')
    ax.set_ylabel('|Δ log p̂(D)| (nats)')
    ax.legend(frameon=False)

    if len(nested):
        ax = axes[2]
        lim = max(nested[['logbf_ref', 'logbf_is']].abs().max()) * 1.05
        for e in np.concatenate([-JEFFREYS_EDGES, JEFFREYS_EDGES]):
            ax.axhline(e, color='0.85', lw=0.6)
            ax.axvline(e, color='0.85', lw=0.6)
        ax.plot([-lim, lim], [-lim, lim], color='0.6', lw=0.8)
        ax.scatter(nested['logbf_ref'], nested['logbf_is'], **dots)
        ax.set_xscale('symlog', linthresh=5.0)
        ax.set_yscale('symlog', linthresh=5.0)
        ticks = [-1000, -100, -10, 0, 10, 100, 1000]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel('ln BF (bridge reference)')
        ax.set_ylabel('ln BF (IS)')

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(label, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def convergedRows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose NUTS reference converged (all rows for CSVs without the column)."""
    return df[df['nuts_converged'].astype(bool)] if 'nuts_converged' in df else df


def writeOutputs(df: pd.DataFrame, pool_sizes: list[int], out_dir: Path, stem: str, label: str):
    """md summary and figure over the datasets whose NUTS reference converged."""
    out_dir.mkdir(parents=True, exist_ok=True)
    conv = convergedRows(df)
    label += f', converged reference {len(conv)}/{len(df)}'
    md = summarize(conv.reset_index(drop=True), pool_sizes, label)
    (out_dir / f'{stem}.md').write_text(md)
    plot(conv.reset_index(drop=True), pool_sizes, out_dir / f'{stem}.png', label)
    print(md)


def latestCsvs(out_dir: Path, family: str, split: str) -> dict[str, Path]:
    """size -> newest merged CSV of one family in out_dir (K = IS2_N_INNER for GLMMs)."""
    prefix = '' if family == 'normal' else f'{family}_k{IS2_N_INNER}_'
    csvs = {
        size: sorted(
            p for p in out_dir.glob(f'{prefix}{size}_{split}_n*.csv') if '_shard' not in p.stem
        )
        for size in SIZES
    }
    return {size: paths[-1] for size, paths in csvs.items() if paths}


def _texRows(df: pd.DataFrame, pool_sizes: list[int], head: str, blank: str) -> list[str]:
    """Table rows of one family and size, converged rows only: the flow pool at the largest S, and
    the t proposal (S cell '$t$') where the CSV carries it."""
    df = convergedRows(df)
    ref = df['logev_bridge_nuts']
    nested = nestedRows(df)
    s_max = max(pool_sizes)
    rows = [(s_max, f'logev_is_s{s_max}', f'k_s{s_max}', 'logbf_is')]
    if 'logev_t' in df:
        rows.append(('$t$', 'logev_t', 'k_t', 'logbf_t'))
    lines = []
    for s, logev, k, logbf in rows:
        d = (df[logev] - ref).abs()
        frac_k = (df[k] > 0.7).mean()
        nested_cells = ' & & '
        if len(nested):
            agree = categoryAgreement(nested[logbf], nested['logbf_ref'])
            bf = (nested[logbf] - nested['logbf_ref']).abs().median()
            nested_cells = f'{len(nested)} & ${agree:.2f}$ & ${bf:.3f}$'
        lines.append(
            f'{head} & {s} & ${d.median():.3f}$ & ${d.quantile(0.9):.3f}$ & '
            f'${frac_k:.2f}$ & {nested_cells} \\\\'
        )
        head = blank
    return lines


_TEX_COLUMNS = (
    r'$\#\mathrm{ds}$ & $\#\mathrm{corr}$ & $S$ & $\mathrm{med}\,|\Delta|$ & '
    r'$q_{90}\,|\Delta|$ & $\mathrm{frac}\,\hat k>0.7$ & $\#\mathrm{nested}$ & '
    r'$\mathrm{BF\ cat.\ agree}$ & $\mathrm{med}\,|\Delta\ln \mathrm{BF}|$ \\'
)


def writeTex(out_dir: Path, split: str, pool_sizes: list[int]) -> Path | None:
    """LaTeX table over all Normal sizes with a CSV in out_dir."""
    csvs = latestCsvs(out_dir, 'normal', split)
    if not csvs:
        return None
    lines = [
        r'\begin{tabular}{lrr r cc c r cc}',
        r'    \toprule',
        r'    $\mathrm{regime}$ & ' + _TEX_COLUMNS,
        r'    \midrule',
    ]
    for size, path in csvs.items():
        df = convergedRows(loadCsv(path))
        head = f'    \\texttt{{{size}}} & {len(df)} & {int(df["corr_active"].sum())}'
        lines += _texRows(df, pool_sizes, head, '     &  & ')
        lines.append(r'    \midrule')
    lines[-1] = r'    \bottomrule'
    lines.append(r'\end{tabular}')
    path = out_dir / 'evidence_normal.tex'
    path.write_text('\n'.join(lines) + '\n')
    return path


def writeTexDiscrete(out_dir: Path, split: str, pool_sizes: list[int]) -> Path | None:
    """LaTeX table over the Bernoulli and Poisson sizes in out_dir."""
    blocks = {f: latestCsvs(out_dir, f, split) for f in ('bernoulli', 'poisson')}
    if not any(blocks.values()):
        return None
    lines = [
        r'\begin{tabular}{llrr r cc c r cc}',
        r'    \toprule',
        r'    $\mathrm{family}$ & $\mathrm{regime}$ & ' + _TEX_COLUMNS,
        r'    \midrule',
    ]
    for family, csvs in blocks.items():
        if not csvs:
            continue
        name = family.capitalize()
        for size, path in csvs.items():
            df = convergedRows(loadCsv(path))
            head = f'    {name} & \\texttt{{{size}}} & {len(df)} & {int(df["corr_active"].sum())}'
            lines += _texRows(df, pool_sizes, head, '     &  &  &  ')
            lines.append(r'    \midrule')
            name = ''
        lines[-1] = r'    \midrule\midrule'
    lines[-1] = r'    \bottomrule'
    lines.append(r'\end{tabular}')
    path = out_dir / 'evidence_discrete.tex'
    path.write_text('\n'.join(lines) + '\n')
    return path


def _nearEdge(log_bf: pd.Series, tol: float = 0.2) -> pd.Series:
    """Reference ln BF within tol nats of a Jeffreys category boundary."""
    edges = np.concatenate([-JEFFREYS_EDGES, JEFFREYS_EDGES])
    return pd.Series(
        np.abs(log_bf.to_numpy()[:, None] - edges[None]).min(1) < tol, index=log_bf.index
    )


def _nestedStats(nested: pd.DataFrame, logbf: str, k: str, k_red: str, tag: str) -> dict:
    """Jeffreys agreement of one estimator's ln BF with the reference, and its disagreements
    flagged by k̂ > 0.7 in the full or the reduced fit (fullk: the full fit only) or with a
    reference ln BF near a category edge."""
    ca, cb = nested[logbf].map(jeffreys), nested['logbf_ref'].map(jeffreys)
    miss = ~((ca == cb) & ca.notna() & cb.notna())
    full_k, edge = nested[k] > 0.7, _nearEdge(nested['logbf_ref'])
    flag = full_k | (nested[k_red] > 0.7)
    return {
        f'agree {tag}': 1 - miss.mean(),
        f'med |ΔlnBF| {tag}': (nested[logbf] - nested['logbf_ref']).abs().median(),
        f'#disagree {tag}': int(miss.sum()),
        f'#disagree {tag} flagged|edge': int((miss & (flag | edge)).sum()),
        f'#disagree {tag} fullk|edge': int((miss & (full_k | edge)).sum()),
    }


def _pooledAgreement(frames: list[pd.DataFrame]) -> dict:
    """Nested Jeffreys agreement over the nested rows of several CSVs; t where all carry it."""
    nested = pd.concat([nestedRows(df) for df in frames])
    out = {
        'nested': len(nested),
        'agree IS': categoryAgreement(nested['logbf_is'], nested['logbf_ref']),
    }
    if all('logbf_t' in df for df in frames):
        out['agree t'] = categoryAgreement(nested['logbf_t'], nested['logbf_ref'])
    return out


def writePaperStats(out_dir: Path, split: str, pool_sizes: list[int]) -> Path | None:
    """The numbers the paper quotes, per family and size over converged rows,
    plus pooled counts; evidence_paper.md."""
    s_min, s_max = min(pool_sizes), max(pool_sizes)
    rows, pooled, frames = {}, {}, {}
    for family in FAMILIES:
        for size, path in latestCsvs(out_dir, family, split).items():
            full = loadCsv(path)
            df = convergedRows(full)
            frames[(family, size)] = (full, df)
            ref = df['logev_bridge_nuts']
            d_min = (df[f'logev_is_s{s_min}'] - ref).abs()
            d = (df[f'logev_is_s{s_max}'] - ref).abs()
            k = df[f'k_s{s_max}']
            nested = nestedRows(df)
            r = {
                'converged': f'{len(df)}/{len(full)}',
                'swap floor med': (ref - df['logev_bridge_nuts_swap']).abs().median(),
                f'med S={s_min}': d_min.median(),
                f'med S={s_max}': d.median(),
                f'q90 S={s_max}': d.quantile(0.9),
                'shrink': 1 - d.median() / d_min.median(),
                'frac k>0.7': (k > 0.7).mean(),
                'med k<=0.5': d[k <= 0.5].median(),
                'med 0.5<k<=0.7': d[(k > 0.5) & (k <= 0.7)].median(),
                'med k>0.7': d[k > 0.7].median(),
                '#|Δ|>1': int((d > 1).sum()),
                '#|Δ|>1 & k>0.7': int(((d > 1) & (k > 0.7)).sum()),
            }
            if f'logev_laplace_s{s_max}' in df:
                lap = (df[f'logev_laplace_s{s_max}'] - ref).abs()
                r['Laplace med'] = lap.median()
                r['Laplace frac>0.3'] = (lap > 0.3).mean()
            if 'logev_t' in df:
                d_t, k_t = (df['logev_t'] - ref).abs(), df['k_t']
                r |= {
                    'med t': d_t.median(),
                    'q90 t': d_t.quantile(0.9),
                    'frac k_t>0.7': (k_t > 0.7).mean(),
                    '#|Δt|>1': int((d_t > 1).sum()),
                    '#|Δt|>1 & k_t>0.7': int(((d_t > 1) & (k_t > 0.7)).sum()),
                }
            if len(nested):
                r['nested'] = len(nested)
                r['agree bridgeImh'] = categoryAgreement(
                    nested['logbf_ref_imh'], nested['logbf_ref']
                )
                r |= _nestedStats(nested, 'logbf_is', f'k_s{s_max}', 'k_red', 'IS')
                if 'logbf_t' in nested:
                    r |= _nestedStats(nested, 'logbf_t', 'k_t', 'k_t_red', 't')
            rows[f'{family} {size}'] = r
        fam = [v for (f, _), v in frames.items() if f == family]
        if not fam:
            continue
        for tag, idx in (('converged', 1), ('all', 0)):
            pooled[f'{family} ({tag})'] = _pooledAgreement([v[idx] for v in fam])
    if not rows:
        return None
    for tag, idx in (('converged', 1), ('all', 0)):
        glmm = [v[idx] for (f, _), v in frames.items() if f != 'normal']
        if glmm:
            pooled[f'bernoulli+poisson ({tag})'] = _pooledAgreement(glmm)
    lines = [
        '# Evidence check: numbers quoted in the paper',
        '',
        f'Reference: {REFERENCE_TAG} thinned to {N_REF_DRAWS} draws; rows restricted to '
        'nutsConverged unless marked (all).',
        '',
        _table(rows),
        '',
        '## Pooled nested agreement',
        '',
        _table(pooled),
        '',
    ]
    path = out_dir / 'evidence_paper.md'
    path.write_text('\n'.join(lines))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def stemPrefix(args: argparse.Namespace) -> str:
    """'' for Normal (keeps the published CSV names), '{family}_k{K}_' for GLMMs."""
    return '' if args.family == 'normal' else f'{args.family}_k{args.n_inner}_'


def runSize(size: str, args: argparse.Namespace) -> None:
    lf = FAMILIES[args.family]
    seed = BEST_SEEDS[(args.family, size)]
    ckpt = _ckpt_dir(args.family, size, seed) / f'{args.prefix}.pt'
    data_dir = DATA_DIR / f'{size}-{args.family[0]}-sampled'
    npz_path = data_dir / f'{args.split}.npz'
    label = f'{args.family.capitalize()} ({size}), {args.split}, n={args.n_datasets}'
    if lf != 0:
        label += f', IS² K={args.n_inner}'
    print(f'\n{"#" * 70}\n#  {label}\n{"#" * 70}')

    model, epoch = loadModel(ckpt)
    model.to(args.device)
    col = Collection(npz_path, permute=False)
    n_ds = min(args.n_datasets, len(col))
    nuts = loadNuts(npz_path, n_ds)
    print(f'model epoch={epoch}  d_ffx={model.d_ffx}  d_rfx={model.d_rfx}  datasets={n_ds}')

    pool_sizes = sorted(args.pool_sizes)
    s_max = pool_sizes[-1]
    # nested datasets: the first args.nested with q >= 2, fixed before sharding
    nested = set([i for i in range(n_ds) if int(col[i]['q']) >= 2][: args.nested])
    rows = []
    t0 = time.perf_counter()
    for i in range(args.shard, n_ds, args.n_shards):
        t_i = time.perf_counter()
        item = col[i]
        # per-dataset bridge RNG, so a sharded run reproduces the unsharded one
        gen = torch.Generator().manual_seed(args.seed * 100_003 + i)
        fit = fitDataset(
            model,
            item,
            args,
            pool_sizes,
            gen,
            seed=args.seed * 100_003 + i,
            laplace_imh=lf != 0,
            imh_bridge=i < args.imh_bridge or i in nested,
        )
        batch, target = fit['batch'], fit['target']
        ess = nuts['nuts_ess'][i]
        row = {
            'idx': i,
            'size': size,
            'd': target.d_i,
            'q': target.q_i,
            'm': int(batch['m']),
            'n': int(batch['n']),
            'eta': float(batch['eta_rfx']),
            'corr_active': target.corr,
            'sd_y': float(batch['sd_y']),
            'log_jac_std': float(logJacobianStandardization(batch)[0]),
            'nuts_max_rhat': float(np.nanmax(nuts['nuts_rhat'][i])),
            'nuts_min_ess': float(np.nanmin(np.where(ess > 0, ess, np.nan))),
            'nuts_div': int(nuts['nuts_divergences'][i].sum()),
            'nuts_converged': bool(nuts['nuts_converged'][i]),
            'imh_accept': fit['imh_accept'],
            **{k: v for k, v in fit['is'].items() if k != 'weights'},
        }
        if lf != 0:
            row['imh_accept_laplace'] = fit['imh_accept_laplace']
            for tag, p in (
                ('raw', fit['proposal']),
                ('imhLaplace', fit['p_imh_laplace']),
                ('imhPM', fit['p_imh']),
            ):
                row.update({f'{k}_{tag}': v for k, v in fidelity(p, nuts, i, target).items()})

        # bridge references: NUTS draws (primary) and the IMH draws (validates the IMH bridge)
        br = bridge(target.logProb, target.fromNuts(nuts, i), args.n_bridge, gen)
        row['logev_bridge_nuts'] = br['log_ev']
        row['logev_bridge_nuts_swap'] = br['log_ev_swap']
        row['bridge_nuts_iter'] = br['n_iter']
        if 'bridge_imh' in fit:
            row['logev_bridge_imh'] = fit['bridge_imh']['log_ev']
            row['logev_bridge_imh_swap'] = fit['bridge_imh']['log_ev_swap']

        # posterior mean of the first rfx correlation under raw flow / IS / IMH vs NUTS
        if target.corr:
            rho_flow = fit['proposal'].samples_g[0, :, -model.d_corr].double()
            row['rho_nuts'] = float(nuts['nuts_corr_rfx'][i, 0, :, 1, 0].mean())
            row['rho_raw'] = float(rho_flow.mean())
            row['rho_is'] = float((fit['is']['weights'] * rho_flow).sum())
            row['rho_imh'] = float(fit['p_imh'].corr_rfx[0, :, 1, 0].mean())

        # nested comparison: drop the random slope; the IMH bridge is the reduced model's reference
        if i in nested:
            red = fitDataset(
                model, reducedItem(item), args, [s_max], gen, seed=args.seed * 100_003 + i + 50_000
            )
            row['logev_is_red'] = red['is'][f'logev_is_s{s_max}']
            row['k_red'] = red['is'][f'k_s{s_max}']
            row['imh_accept_red'] = red['imh_accept']
            row['logev_bridge_imh_red'] = red['bridge_imh']['log_ev']
            row['logev_bridge_imh_red_swap'] = red['bridge_imh']['log_ev_swap']
            row['logbf_is'] = row[f'logev_is_s{s_max}'] - row['logev_is_red']
            row['logbf_ref'] = row['logev_bridge_nuts'] - row['logev_bridge_imh_red']
            row['logbf_ref_imh'] = row['logev_bridge_imh'] - row['logev_bridge_imh_red']

        rows.append(row)
        print(
            f'ds={i:3d} d={target.d_i} q={target.q_i} m={row["m"]:3d} n={row["n"]:4d} corr={int(target.corr)}  '
            f'bridge={row["logev_bridge_nuts"]:9.2f} (swap Δ={row["logev_bridge_nuts"] - row["logev_bridge_nuts_swap"]:+.3f}, '
            f'imh Δ={row.get("logev_bridge_imh", np.nan) - row["logev_bridge_nuts"]:+.3f})  '
            f'IS Δ={row[f"logev_is_s{s_max}"] - row["logev_bridge_nuts"]:+.3f}  '
            f'k={row[f"k_s{s_max}"]:.2f} acc={row["imh_accept"]:.2f}'
            + (
                f'  lnBF ref={row["logbf_ref"]:+.2f} is={row["logbf_is"]:+.2f}'
                if 'logbf_ref' in row
                else ''
            )
            + f'  [{time.perf_counter() - t_i:.1f}s]',
            flush=True,
        )

    df = pd.DataFrame(rows)
    stem = f'{stemPrefix(args)}{size}_{args.split}_n{n_ds}'
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.n_shards > 1:
        stem += f'_shard{args.shard}of{args.n_shards}'
        df.to_csv(args.out_dir / f'{stem}.csv', index=False)
        print(f'[saved] {args.out_dir / stem}.csv  ({time.perf_counter() - t0:.0f}s)')
        return
    df.to_csv(args.out_dir / f'{stem}.csv', index=False)
    writeOutputs(df, pool_sizes, args.out_dir, stem, label)
    print(f'[saved] {args.out_dir / stem}.{{csv,md,png}}  ({time.perf_counter() - t0:.0f}s)')


def runSizeT(size: str, args: argparse.Namespace) -> None:
    """--t-only: tFit on the rows of the size's CSV, and on the reduced model where the CSV has a
    nested reference, with runSize's seeds; written to t/ for tJoin."""
    ckpt = _ckpt_dir(args.family, size, BEST_SEEDS[(args.family, size)]) / f'{args.prefix}.pt'
    col = Collection(DATA_DIR / f'{size}-{args.family[0]}-sampled' / f'{args.split}.npz')
    stem = f'{stemPrefix(args)}{size}_{args.split}_n{min(args.n_datasets, len(col))}'
    ref = pd.read_csv(args.out_dir / f'{stem}.csv')
    model, _ = loadModel(ckpt)
    model.to(args.device)
    rows = []
    t0 = time.perf_counter()
    for r in ref[ref['idx'] % args.n_shards == args.shard].itertuples():
        i = int(r.idx)
        fit = tFit(model, col[i], args, seed=args.seed * 100_003 + i)
        row = {'idx': i, 'logev_t': fit['log_ev'], 'k_t': fit['k'], 'eff_t': fit['eff']}
        row |= {'ess_t': fit['ess'], 'moved_t': fit['moved']}
        msg = f'ds={i:3d}  t Δ={fit["log_ev"] - r.logev_bridge_nuts:+.3f}  k={fit["k"]:.2f}'
        msg += f'  ess={fit["ess"]:.0f}{" moved" if fit["moved"] else ""}'
        if pd.notna(r.logbf_ref):
            red = tFit(model, reducedItem(col[i]), args, seed=args.seed * 100_003 + i + 50_000)
            row |= {'logev_t_red': red['log_ev'], 'k_t_red': red['k']}
            row |= {'ess_t_red': red['ess'], 'moved_t_red': red['moved']}
            row['logbf_t'] = fit['log_ev'] - red['log_ev']
            msg += f'  lnBF ref={r.logbf_ref:+.2f} t={row["logbf_t"]:+.2f}'
        rows.append(row)
        print(msg, flush=True)
    shard = '' if args.n_shards == 1 else f'_shard{args.shard}of{args.n_shards}'
    path = args.out_dir / 't' / f'{stem}{shard}.csv'
    path.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=['idx', *T_COLUMNS]).to_csv(path, index=False)
    print(f'[saved] {path}  ({time.perf_counter() - t0:.0f}s)')


def tJoin(out_dir: Path, stem: str) -> None:
    """Join t/{stem}.csv onto {stem}.csv by idx, replacing earlier t columns; the CSV's own
    columns are read and written at round-trip precision and checked to come back unchanged."""
    t_path, path = out_dir / 't' / f'{stem}.csv', out_dir / f'{stem}.csv'
    if not t_path.exists():
        return
    df = pd.read_csv(path, float_precision='round_trip')
    df = df.drop(columns=list(T_COLUMNS), errors='ignore')
    t = pd.read_csv(t_path, float_precision='round_trip')
    df.merge(t, on='idx', how='left', validate='one_to_one').to_csv(path, index=False)
    back = pd.read_csv(path, float_precision='round_trip')[df.columns]
    pd.testing.assert_frame_equal(back, df, check_exact=True)
    print(f'[joined] {t_path} -> {path.name}')


def mergeShards(size: str, args: argparse.Namespace, directory: Path) -> None:
    """Concatenate complete sets of shard CSVs in directory into their unsharded CSV there
    (incomplete: skipped)."""
    shards: dict[tuple[str, int], list[Path]] = {}
    for path in directory.glob(f'{stemPrefix(args)}{size}_{args.split}_n*_shard*of*.csv'):
        base, tag = path.stem.rsplit('_shard', 1)
        shards.setdefault((base, int(tag.split('of')[1])), []).append(path)
    for (base, n_shards), paths in shards.items():
        if len(paths) < n_shards:
            print(f'[skip] {base}: {len(paths)}/{n_shards} shards present')
            continue
        # plain read_csv: loadCsv's legacy renames would also hit fresh columns (logev_is_red)
        df = pd.concat([pd.read_csv(p) for p in paths]).sort_values('idx').reset_index(drop=True)
        df.to_csv(directory / f'{base}.csv', index=False)
        print(f'[merged] {n_shards} shards -> {base}.csv')


def resummarize(size: str, args: argparse.Namespace) -> None:
    mergeShards(size, args, args.out_dir)
    mergeShards(size, args, args.out_dir / 't')
    paths = sorted(
        p
        for p in args.out_dir.glob(f'{stemPrefix(args)}{size}_{args.split}_n*.csv')
        if '_shard' not in p.stem
    )
    if not paths:
        print(f'[skip] no CSV for {size} in {args.out_dir}')
        return
    tJoin(args.out_dir, paths[-1].stem)
    df = loadCsv(paths[-1])
    label = f'{args.family.capitalize()} ({size}), {args.split}, n={len(df)}'
    writeOutputs(df, sorted(args.pool_sizes), args.out_dir, paths[-1].stem, label)


def main() -> None:
    args = setup()
    for size in args.sizes:
        if args.summarize_only:
            resummarize(size, args)
        else:
            (runSizeT if args.t_only else runSize)(size, args)
    # the combined tables span all sizes with a CSV in out_dir; assemble them only in the
    # summarize-only pass so parallel per-size runs do not race on the same file
    if args.summarize_only:
        pool_sizes = sorted(args.pool_sizes)
        for write in (writeTex, writeTexDiscrete, writePaperStats):
            path = write(args.out_dir, args.split, pool_sizes)
            if path is not None:
                print(f'[saved] {path}')


if __name__ == '__main__':
    main()
