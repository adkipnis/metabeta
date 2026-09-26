"""Evidence check: is the mean marginal IS weight a usable estimate of log p(D)?

The paper's outlook claims that, because the flow density is normalized and prior and
marginal likelihood are evaluated exactly, the mean importance weight over the proposal
pool is an unbiased estimate of the marginal likelihood (Gaussian outcomes), so Bayes
factors cost a single batched forward pass. This script tests that claim on the Gaussian
oracle sets against a bridge-sampling reference.

Per dataset (standardized space — the space the flow density lives in):
  IS         : log p(D) ≈ logsumexp(log_w) - log S from ImportanceSampler(marginal=True),
               for pool prefixes S in --pool-sizes (flow draws are i.i.d.).
  bridgeNuts : Meng-Wong iterative bridge sampling on the cached NUTS draws
               (test.fit.npz), warped-Gaussian proposal fitted on one half of the draws
               and evaluated on the other, with the halves swapped as a noise floor.
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

Outputs: {out_dir}/{prefix}{size}_{split}_n{n_ds}.csv (per-dataset rows; prefix '' for Normal,
'{family}_k{K}_' otherwise), .md (summary tables) and .png (diagnostic figure);
--summarize-only additionally writes evidence_normal.tex.

Run from the repo root:
    uv run python experiments/posthoc/evidence.py --sizes small --n-datasets 32
    uv run python experiments/posthoc/evidence.py --sizes small medium --n-datasets 512 --nested 128
    # after all sizes ran (possibly on different nodes): rebuild summaries + the combined table
    uv run python experiments/posthoc/evidence.py --sizes small medium large huge --summarize-only
    uv run python experiments/posthoc/evidence.py --family bernoulli --sizes small --n-datasets 32
    # parallel shards (one process each), then merge + summarize
    uv run python experiments/posthoc/evidence.py --family bernoulli --sizes small --n-datasets 512 --shard 3 --n-shards 16
    uv run python experiments/posthoc/evidence.py --family bernoulli --sizes small --n-datasets 512 --summarize-only
"""

import argparse
import math
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import torch
from torch import distributions as D

from metabeta.utils.experiments import DATA_DIR, REPO_ROOT, experimentResultsPath

sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from build_ckpt import BEST_SEEDS, _ckpt_dir  # noqa: E402

from metabeta.models.approximator import Approximator  # noqa: E402
from metabeta.posthoc.importance import ImportanceSampler  # noqa: E402
from metabeta.posthoc.laplace_glmm import (  # noqa: E402
    LaplaceImportanceSampler,
    logMarginalLikelihoodAGQ,
)
from metabeta.posthoc.metropolis import MetropolisSampler  # noqa: E402
from metabeta.utils.config import ApproximatorConfig  # noqa: E402
from metabeta.utils.constants import hasSigmaEps  # noqa: E402
from metabeta.utils.dataloader import Collection, collateGrouped, toDevice  # noqa: E402
from metabeta.utils.families import logProbCorrRfx  # noqa: E402
from metabeta.utils.preprocessing import logJacobianStandardization  # noqa: E402
from metabeta.utils.regularization import (  # noqa: E402
    corrToLower,
    corrToUnconstrained,
    unconstrainedToCholesky,
)
from metabeta.utils.results import Proposal  # noqa: E402

IMH_CHAINS = 4
IMH_BURNIN = 25
SIZES = ('small', 'medium', 'large', 'huge')
FAMILIES = {'normal': 0, 'bernoulli': 1, 'poisson': 2}
TARGET_CHUNK = 1000  # bridge-target draws per AGQ pass (bounds the (m, n, s) intermediates)
JEFFREYS_EDGES = np.log([3.0, 10.0, 30.0, 100.0])  # ln BF thresholds (Jeffreys, natural log)
COLOR = '#3B6FB6'
# CSVs written while the legacy 'z' prior coordinates were still reported carry an `_r` infix
LEGACY_COLUMNS = {'logev_is_r_': 'logev_is_', 'k_r_': 'k_', 'eff_r_': 'eff_', '_is_r': '_is'}


# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--family', default='normal', choices=list(FAMILIES))
    p.add_argument('--sizes', nargs='+', default=['small'], choices=SIZES)
    p.add_argument('--n-inner', type=int, default=8, help='IS² draws per group (GLMMs)')
    p.add_argument('--split', default='test', choices=['test'], help='only test.fit.npz carries NUTS draws')
    p.add_argument('--prefix', default='latest', help='checkpoint prefix (latest = the checkpoint of the paper tables)')
    p.add_argument('--n-datasets', type=int, default=32, help='datasets per size (first n of the split)')
    p.add_argument('--pool-sizes', nargs='+', type=int, default=[1000, 2000, 4000], help='IS pool prefixes; the largest is drawn')
    p.add_argument('--n-bridge', type=int, default=2000, help='proposal draws per bridge run (posterior draws: half of the available)')
    p.add_argument('--nested', type=int, default=16, help='max number of q>=2 datasets for the nested comparison (0 disables)')
    p.add_argument('--imh-bridge', type=int, default=64, help='run the full-model IMH bridge (a cross-check of the reduced-model reference) on the first n datasets only; nested datasets always get it')
    p.add_argument('--shard', type=int, default=0, help='this process handles datasets i with i %% n_shards == shard')
    p.add_argument('--n-shards', type=int, default=1, help='split the datasets over this many processes; --summarize-only merges the shard CSVs')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu', help='device for flow sampling; everything else runs on cpu in float64')
    p.add_argument('--out-dir', type=Path, default=experimentResultsPath('evidence'))
    p.add_argument('--summarize-only', action='store_true', help='regenerate md/png/tex from existing CSVs without recomputing')
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
    """Global NUTS draws (standardized space) and diagnostics for the first n_ds datasets.

    Each NpzFile member decompresses in full on access, so every key is read once; the
    rfx draws (GBs) are never touched.
    """
    keys = (
        'nuts_ffx',
        'nuts_sigma_rfx',
        'nuts_sigma_eps',
        'nuts_corr_rfx',
        'nuts_rhat',
        'nuts_ess',
        'nuts_divergences',
    )
    with np.load(npz_path, allow_pickle=True) as data:
        return {k: np.asarray(data[k][:n_ds]) for k in keys if k in data.files}


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
        lp, ffx, sigma_eps = self.is_._logPriorGlobals(p)
        is_ = self.is_
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
        lw_np, k = az.psislw(lw_s.unsqueeze(0).numpy())
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


def runImh(
    proposal: Proposal, batch64: dict[str, torch.Tensor], lf: int, n_inner: int = 0
) -> tuple[Proposal, float]:
    """Marginal IMH (Normal) or Laplace IMH, pseudo-marginal when n_inner > 0 (GLMMs)."""
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
    )
    p_imh, diag = sampler(proposalDouble(proposal, IMH_CHAINS * n_steps))
    return p_imh, float(diag['accept_rate'].mean())


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
    p_imh, accept = runImh(proposal, batch64, lf, n_inner)
    fit = {
        'batch': batch,
        'target': target,
        'proposal': proposal,
        'is': isEvidence(proposal, batch64, pool_sizes, lf, n_inner),
        'p_imh': p_imh,
        'imh_accept': accept,
    }
    if imh_bridge:
        fit['bridge_imh'] = bridge(target.logProb, target.fromProposal(p_imh), args.n_bridge, gen)
    if laplace_imh:
        fit['p_imh_laplace'], fit['imh_accept_laplace'] = runImh(proposal, batch64, lf)
    return fit


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


def writeOutputs(df: pd.DataFrame, pool_sizes: list[int], out_dir: Path, stem: str, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    md = summarize(df, pool_sizes, label)
    (out_dir / f'{stem}.md').write_text(md)
    plot(df, pool_sizes, out_dir / f'{stem}.png', label)
    print(md)


def writeTex(out_dir: Path, split: str, pool_sizes: list[int]) -> Path | None:
    """Appendix table over all sizes with a CSV in out_dir."""
    csvs = {size: sorted(out_dir.glob(f'{size}_{split}_n*.csv')) for size in SIZES}
    csvs = {size: paths[-1] for size, paths in csvs.items() if paths}
    if not csvs:
        return None
    lines = [
        r'\begin{tabular}{lrr r cc c r cc}',
        r'    \toprule',
        r'    $\mathrm{regime}$ & $\#\mathrm{ds}$ & $\#\mathrm{corr}$ & $S$ & '
        r'$\mathrm{med}\,|\Delta\log p(\mathcal{D})|$ & $q_{90}\,|\Delta\log p(\mathcal{D})|$ & '
        r'$\mathrm{frac}\,k>0.7$ & $\#\mathrm{nested}$ & $\mathrm{BF\ cat.\ agree}$ & '
        r'$\mathrm{med}\,|\Delta\ln \mathrm{BF}|$ \\',
        r'    \midrule',
    ]
    s_min, s_max = min(pool_sizes), max(pool_sizes)
    for size, path in csvs.items():
        df = loadCsv(path)
        ref = df['logev_bridge_nuts']
        nested = nestedRows(df)
        head = f'    \\texttt{{{size}}} & {len(df)} & {int(df["corr_active"].sum())}'
        for s in (s_min, s_max):
            if f'logev_is_s{s}' not in df:
                continue
            d = (df[f'logev_is_s{s}'] - ref).abs()
            frac_k = (df[f'k_s{s}'] > 0.7).mean()
            nested_cells = ' & & '
            if s == s_max and len(nested):
                agree = categoryAgreement(nested['logbf_is'], nested['logbf_ref'])
                bf = (nested['logbf_is'] - nested['logbf_ref']).abs().median()
                nested_cells = f'{len(nested)} & ${agree:.2f}$ & ${bf:.3f}$'
            lines.append(
                f'{head} & {s} & ${d.median():.3f}$ & ${d.quantile(0.9):.3f}$ & '
                f'${frac_k:.2f}$ & {nested_cells} \\\\'
            )
            head = '     &  & '
        lines.append(r'    \midrule')
    lines[-1] = r'    \bottomrule'
    lines.append(r'\end{tabular}')
    path = out_dir / 'evidence_normal.tex'
    path.write_text('\n'.join(lines) + '\n')
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
    npz_path = data_dir / f'{args.split}.fit.npz'
    label = f'{args.family.capitalize()} ({size}), {args.split}, n={args.n_datasets}'
    if lf != 0:
        label += f', IS² K={args.n_inner}'
    print(f'\n{"#" * 70}\n#  {label}\n{"#" * 70}')

    model, epoch = loadModel(ckpt)
    model.to(args.device)
    col = Collection(npz_path, permute=False, exclude_prefixes=('nuts_', 'advi_', 'laplace_'))
    n_ds = min(args.n_datasets, len(col))
    nuts = loadNuts(npz_path, n_ds)
    print(f'model epoch={epoch}  d_ffx={model.d_ffx}  d_rfx={model.d_rfx}  datasets={n_ds}')

    pool_sizes = sorted(args.pool_sizes)
    s_max = pool_sizes[-1]
    # nested datasets: the first args.nested with q >= 2, fixed before sharding
    nested = {i for i in range(n_ds) if int(col[i]['q']) >= 2}
    nested = set(sorted(nested)[: args.nested])
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


def mergeShards(size: str, args: argparse.Namespace) -> None:
    """Concatenate complete sets of shard CSVs into their unsharded CSV (incomplete: skipped)."""
    shards: dict[tuple[str, int], list[Path]] = {}
    for path in args.out_dir.glob(f'{stemPrefix(args)}{size}_{args.split}_n*_shard*of*.csv'):
        base, tag = path.stem.rsplit('_shard', 1)
        shards.setdefault((base, int(tag.split('of')[1])), []).append(path)
    for (base, n_shards), paths in shards.items():
        if len(paths) < n_shards:
            print(f'[skip] {base}: {len(paths)}/{n_shards} shards present')
            continue
        # plain read_csv: loadCsv's legacy renames would also hit fresh columns (logev_is_red)
        df = pd.concat([pd.read_csv(p) for p in paths]).sort_values('idx').reset_index(drop=True)
        df.to_csv(args.out_dir / f'{base}.csv', index=False)
        print(f'[merged] {n_shards} shards -> {base}.csv')


def resummarize(size: str, args: argparse.Namespace) -> None:
    mergeShards(size, args)
    paths = sorted(
        p
        for p in args.out_dir.glob(f'{stemPrefix(args)}{size}_{args.split}_n*.csv')
        if '_shard' not in p.stem
    )
    if not paths:
        print(f'[skip] no CSV for {size} in {args.out_dir}')
        return
    df = loadCsv(paths[-1])
    label = f'{args.family.capitalize()} ({size}), {args.split}, n={len(df)}'
    writeOutputs(df, sorted(args.pool_sizes), args.out_dir, paths[-1].stem, label)


def main() -> None:
    args = setup()
    for size in args.sizes:
        (resummarize if args.summarize_only else runSize)(size, args)
    # the combined table spans all sizes with a CSV in out_dir; assemble it only in the
    # summarize-only pass so parallel per-size runs do not race on the same file
    if args.summarize_only and args.family == 'normal':
        tex = writeTex(args.out_dir, args.split, sorted(args.pool_sizes))
        if tex is not None:
            print(f'[saved] {tex}')


if __name__ == '__main__':
    main()
