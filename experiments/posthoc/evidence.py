"""Evidence check: is the mean marginal IS weight a usable estimate of log p(D)?

The paper's outlook claims that, because the flow density is normalized and prior and
marginal likelihood are evaluated exactly, the mean importance weight over the proposal
pool is an unbiased estimate of the marginal likelihood (Gaussian outcomes), so Bayes
factors cost a single batched forward pass. This script tests that claim on the Gaussian
oracle sets against a bridge-sampling reference.

Per dataset (standardized space — the space the flow density lives in):
  IS         : log p(D) ≈ logsumexp(log_w) - log S from ImportanceSampler(marginal=True),
               for pool prefixes S in --pool-sizes (flow draws are i.i.d.), in both LKJ
               prior coordinate conventions ('z' legacy, 'r' consistent — see the
               "Correlation coordinates" section of posthoc/importance.py).
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
               raw flow / IS('z') / IS('r') vs NUTS — quantifies the coordinate tilt.

Outputs: {out_dir}/{size}_{split}_n{n_ds}.csv (per-dataset rows), .md (summary tables)
and .png (diagnostic figure).

Run from the repo root:
    uv run python experiments/posthoc/evidence.py --sizes small --n-datasets 32
    uv run python experiments/posthoc/evidence.py --sizes small medium --n-datasets 512 --nested 128
    # after all sizes ran (possibly on different nodes): rebuild summaries + the combined table
    uv run python experiments/posthoc/evidence.py --sizes small medium large huge --summarize-only
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
from metabeta.posthoc.metropolis import MetropolisSampler  # noqa: E402
from metabeta.utils.config import ApproximatorConfig  # noqa: E402
from metabeta.utils.dataloader import Collection, collateGrouped  # noqa: E402
from metabeta.utils.families import logProbCorrRfx  # noqa: E402
from metabeta.utils.preprocessing import logJacobianStandardization  # noqa: E402
from metabeta.utils.regularization import (  # noqa: E402
    corrToLower,
    corrToUnconstrained,
    unconstrainedToCholesky,
)
from metabeta.utils.results import Proposal  # noqa: E402

COORDS = ('z', 'r')
IMH_CHAINS = 4
IMH_BURNIN = 25
SIZES = ('small', 'medium', 'large', 'huge')
JEFFREYS_EDGES = np.log([3.0, 10.0, 30.0, 100.0])  # ln BF thresholds (Jeffreys, natural log)


# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sizes', nargs='+', default=['small'], choices=['small', 'medium', 'large', 'huge'])
    p.add_argument('--split', default='test', choices=['test'], help='only test.fit.npz carries NUTS draws')
    p.add_argument('--prefix', default='best', help='checkpoint prefix')
    p.add_argument('--n-datasets', type=int, default=32, help='datasets per size (first n of the split)')
    p.add_argument('--pool-sizes', nargs='+', type=int, default=[1000, 2000, 4000], help='IS pool prefixes; the largest is drawn')
    p.add_argument('--n-bridge', type=int, default=2000, help='proposal draws per bridge run (posterior draws: half of the available)')
    p.add_argument('--nested', type=int, default=16, help='max number of q>=2 datasets for the nested comparison (0 disables)')
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
        return {k: np.asarray(data[k][:n_ds]) for k in keys}


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
        'global': {
            'samples': cut(p.samples_g, 1),
            'log_prob': cut(p.log_prob_g, 1),
        },
        'local': {
            'samples': cut(p.samples_l, 2),
            'log_prob': cut(p.log_prob_l, 2),
        },
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


class Target:
    """log p(y | θ_g) p(θ_g) |J| over u = (β, log σ_rfx, log σ_eps[, z_corr]) for one dataset.

    Reuses ImportanceSampler.unnormalizedPosterior (marginal likelihood, β/σ priors) on
    a padded constrained Proposal; the LKJ prior is added as a density over z in the
    dataset's own dimension q_i, so the target is independent of the flow's coordinate
    convention (the 'z'/'r' question only concerns the flow's log q).
    """

    def __init__(self, batch64: dict[str, torch.Tensor], d_corr_model: int) -> None:
        self.is_ = ImportanceSampler(batch64, marginal=True, corr_prior=False, likelihood_family=0)
        self.d_max = batch64['X'].shape[-1]
        self.q_max = batch64['Z'].shape[-1]
        self.m = batch64['X'].shape[1]
        self.d_corr_max = d_corr_model
        self.d_i = int(batch64['mask_d'].sum())
        self.q_i = int(batch64['mask_q'].sum())
        self.eta = batch64['eta_rfx'].double()  # (1,)
        self.corr = bool(self.eta.item() > 0) and self.q_i >= 2
        self.d_corr_i = self.q_i * (self.q_i - 1) // 2 if self.corr else 0
        self.dim = self.d_i + self.q_i + 1 + self.d_corr_i
        self.D_g = self.d_max + self.q_max + 1 + self.d_corr_max

    def toProposal(self, u: torch.Tensor) -> Proposal:
        s = u.shape[0]
        g = u.new_zeros(s, self.D_g)
        g[:, : self.d_i] = u[:, : self.d_i]
        g[:, self.d_max : self.d_max + self.q_i] = u[:, self.d_i : self.d_i + self.q_i].exp()
        g[:, self.d_max + self.q_max] = u[:, self.d_i + self.q_i].exp()
        if self.corr:
            z = u[:, self.d_i + self.q_i + 1 :]
            L = unconstrainedToCholesky(z, self.q_i)
            off = self.d_max + self.q_max + 1
            g[:, off : off + self.d_corr_i] = corrToLower(L @ L.mT)
        proposed = {
            'global': {'samples': g.unsqueeze(0), 'log_prob': g.new_zeros(1, s)},
            'local': {
                'samples': g.new_zeros(1, self.m, s, self.q_max),
                'log_prob': g.new_zeros(1, self.m, s),
            },
        }
        return Proposal(proposed, has_sigma_eps=True, d_corr=self.d_corr_max)

    def logProb(self, u: torch.Tensor) -> torch.Tensor:
        ll, lp = self.is_.unnormalizedPosterior(self.toProposal(u))
        out = (ll + lp)[0]
        out = out + u[:, self.d_i : self.d_i + self.q_i + 1].sum(-1)  # |dσ/d log σ|
        if self.corr:
            z = u.new_zeros(1, u.shape[0], self.d_corr_max)
            z[0, :, : self.d_corr_i] = u[:, self.d_i + self.q_i + 1 :]
            out = (
                out + logProbCorrRfx(z, self.q_max, self.eta, q_active=torch.tensor([self.q_i]))[0]
            )
        return torch.nan_to_num(out, nan=-math.inf, posinf=-math.inf)

    def fromConstrained(
        self,
        ffx: torch.Tensor,
        sigma_rfx: torch.Tensor,
        sigma_eps: torch.Tensor,
        corr: torch.Tensor | None,
    ) -> torch.Tensor:
        """(s, d_i), (s, q_i), (s,), (s, q, q) constrained draws → (s, dim) unconstrained."""
        parts = [
            ffx[:, : self.d_i].double(),
            sigma_rfx[:, : self.q_i].double().clamp_min(1e-8).log(),
            sigma_eps.double().clamp_min(1e-8).log().unsqueeze(-1),
        ]
        if self.corr:
            parts.append(corrToUnconstrained(corr[:, : self.q_i, : self.q_i].double()))
        return torch.cat(parts, -1)


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
    proposal: Proposal, batch64: dict[str, torch.Tensor], pool_sizes: list[int], coords: str
) -> dict:
    sampler = ImportanceSampler(
        batch64, marginal=True, corr_prior=True, pareto=True, corr_prior_coords=coords
    )
    out = sampler(proposalDouble(proposal))
    lw = out.is_results['log_w_raw'][0]  # (S,)
    res = {'weights': out.is_results['weights'][0]}
    for s in pool_sizes:
        lw_s = lw[:s]
        lw_np, k = az.psislw(lw_s.unsqueeze(0).numpy())
        w = torch.softmax(torch.as_tensor(lw_np[0]), -1)
        res[f'logev_is_{coords}_s{s}'] = (torch.logsumexp(lw_s, 0) - math.log(s)).item()
        res[f'k_{coords}_s{s}'] = float(k[0])
        res[f'eff_{coords}_s{s}'] = float(1.0 / (s * (w**2).sum()))
    return res


def runImh(proposal: Proposal, batch64: dict[str, torch.Tensor]) -> tuple[Proposal, float]:
    n_steps = proposal.n_samples // IMH_CHAINS
    sampler = MetropolisSampler(
        batch64,
        n_chains=IMH_CHAINS,
        n_steps=n_steps,
        burnin=IMH_BURNIN,
        mode='marginal',
        likelihood_family=0,
        n_eff_target=None,
        corr_prior_coords='r',
    )
    p_imh, diag = sampler(proposalDouble(proposal, IMH_CHAINS * n_steps))
    return p_imh, float(diag['accept_rate'].mean())


def imhUnconstrained(target: Target, p_imh: Proposal) -> torch.Tensor:
    corr = p_imh.corr_rfx[0] if target.corr else None
    return target.fromConstrained(p_imh.ffx[0], p_imh.sigma_rfx[0], p_imh.sigma_eps[0], corr)


def nutsUnconstrained(target: Target, nuts: dict[str, np.ndarray], i: int) -> torch.Tensor:
    ffx = torch.as_tensor(nuts['nuts_ffx'][i]).T  # (S, d_max)
    sigma_rfx = torch.as_tensor(nuts['nuts_sigma_rfx'][i]).T  # (S, q_max)
    sigma_eps = torch.as_tensor(nuts['nuts_sigma_eps'][i, 0])  # (S,)
    corr = torch.as_tensor(nuts['nuts_corr_rfx'][i, 0]) if target.corr else None  # (S, q, q)
    return target.fromConstrained(ffx, sigma_rfx, sigma_eps, corr)


def jeffreys(log_bf: float) -> float:
    """Signed Jeffreys category of ln BF: 0 anecdotal … ±4 decisive; NaN for a non-finite BF."""
    if not np.isfinite(log_bf):
        return np.nan
    return float(np.sign(log_bf) * np.searchsorted(JEFFREYS_EDGES, abs(log_bf), side='right'))


def categoryAgreement(a: pd.Series, b: pd.Series) -> float:
    """Share of rows whose Jeffreys categories agree; a non-finite estimate never agrees."""
    ca, cb = a.map(jeffreys), b.map(jeffreys)
    return float(((ca == cb) & ca.notna() & cb.notna()).mean())


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


def summarize(df: pd.DataFrame, pool_sizes: list[int], label: str) -> str:
    lines = [f'# {label}: IS log-evidence vs bridge sampling', '']
    ref = df['logev_bridge_nuts']
    lines.append('## Reference noise floor (nats)')
    lines.append('')
    lines.append(
        pd.DataFrame(
            {
                'bridgeNuts vs swapped halves': _absStats(ref - df['logev_bridge_nuts_swap']),
                'bridgeImh vs bridgeNuts': _absStats(df['logev_bridge_imh'] - ref),
            }
        ).T.to_markdown(floatfmt='.3f')
    )
    lines += ['', '## IS − bridgeNuts by coordinates and pool size', '']
    rows = {}
    for c in COORDS:
        for s in pool_sizes:
            rows[f"IS('{c}') S={s}"] = _absStats(df[f'logev_is_{c}_s{s}'] - ref)
    lines.append(pd.DataFrame(rows).T.to_markdown(floatfmt='.3f'))
    s_max = max(pool_sizes)
    lines += ['', f'## By PSIS k (S={s_max})', '']
    rows = {}
    for c in COORDS:
        k = df[f'k_{c}_s{s_max}']
        d = df[f'logev_is_{c}_s{s_max}'] - ref
        for name, sel in (
            ('k ≤ 0.5', k <= 0.5),
            ('0.5 < k ≤ 0.7', (k > 0.5) & (k <= 0.7)),
            ('k > 0.7', k > 0.7),
        ):
            rows[f"IS('{c}') {name}"] = _absStats(d[sel])
    lines.append(pd.DataFrame(rows).T.to_markdown(floatfmt='.3f'))
    for c in COORDS:
        d = (df[f'logev_is_{c}_s{s_max}'] - ref).abs()
        rho_k = d.corr(df[f'k_{c}_s{s_max}'], method='spearman')
        rho_a = d.corr(df['imh_accept'], method='spearman')
        lines.append('')
        lines.append(
            f"IS('{c}') S={s_max}: Spearman(|Δ|, k) = {rho_k:.2f}, Spearman(|Δ|, IMH acceptance) = {rho_a:.2f}"
        )
    corr = df[df['corr_active']]
    if len(corr):
        lines += ['', f'## Correlated datasets (n={len(corr)}), S={s_max}', '']
        rows = {
            f"IS('{c}')": _absStats(corr[f'logev_is_{c}_s{s_max}'] - corr['logev_bridge_nuts'])
            for c in COORDS
        }
        lines.append(pd.DataFrame(rows).T.to_markdown(floatfmt='.3f'))
        lines += [
            '',
            'Posterior mean of the first rfx correlation vs NUTS (mean |E[ρ] − E_NUTS[ρ]|):',
            '',
        ]
        rows = {
            name: {
                'mean |Δρ|': (corr[col] - corr['rho_nuts']).abs().mean(),
                'mean Δρ': (corr[col] - corr['rho_nuts']).mean(),
            }
            for name, col in (
                ('raw flow', 'rho_raw'),
                ("IS('z')", 'rho_is_z'),
                ("IS('r')", 'rho_is_r'),
                ('IMH', 'rho_imh'),
            )
        }
        lines.append(pd.DataFrame(rows).T.to_markdown(floatfmt='.4f'))
    nested = df[df['logbf_ref'].notna()] if 'logbf_ref' in df else df.iloc[0:0]
    if len(nested):
        lines += ['', f'## Nested comparison: random slope vs intercept-only (n={len(nested)})', '']
        rows = {}
        for name, col in (
            ("IS('z')", 'logbf_is_z'),
            ("IS('r')", 'logbf_is_r'),
            ('bridgeImh (both)', 'logbf_ref_imh'),
        ):
            st = _absStats(nested[col] - nested['logbf_ref'])
            st['same Jeffreys cat.'] = categoryAgreement(nested[col], nested['logbf_ref'])
            rows[name] = st
        lines.append(pd.DataFrame(rows).T.to_markdown(floatfmt='.3f'))
        lines.append('')
        cats = nested['logbf_ref'].map(jeffreys).dropna().astype(int).value_counts().sort_index()
        lines.append(
            'Reference ln BF categories (signed Jeffreys, + favours the random slope): '
            + ', '.join(f'{k}: {v}' for k, v in cats.items())
        )
    lines.append('')
    return '\n'.join(lines)


def plot(df: pd.DataFrame, pool_sizes: list[int], path: Path, label: str) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'z': '#D9822B', 'r': '#3B6FB6'}
    s_max = max(pool_sizes)
    ref = df['logev_bridge_nuts']
    has_nested = 'logbf_ref' in df and df['logbf_ref'].notna().any()
    fig, axes = plt.subplots(
        1, 3 if has_nested else 2, figsize=(4.2 * (3 if has_nested else 2), 3.6)
    )
    ax = axes[0]
    for c in COORDS:
        ax.scatter(
            df[f'k_{c}_s{s_max}'],
            df[f'logev_is_{c}_s{s_max}'] - ref,
            s=14,
            alpha=0.8,
            color=colors[c],
            edgecolor='none',
            label=f"IS('{c}')",
        )
    ax.axhline(0, color='0.6', lw=0.8)
    ax.axvline(0.7, color='0.6', lw=0.8, ls=':')
    ax.set_xlabel('PSIS k')
    ax.set_ylabel(f'log p̂(D) IS (S={s_max}) − bridge (nats)')
    ax.legend(frameon=False)
    ax = axes[1]
    for c in COORDS:
        d = np.stack([(df[f'logev_is_{c}_s{s}'] - ref).abs().to_numpy() for s in pool_sizes], 1)
        for row in d:
            ax.plot(pool_sizes, row, color=colors[c], alpha=0.15, lw=0.8)
        ax.plot(
            pool_sizes,
            np.nanmedian(d, 0),
            color=colors[c],
            lw=2,
            marker='o',
            label=f"IS('{c}') median",
        )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(pool_sizes)
    ax.set_xticklabels([str(s) for s in pool_sizes])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('pool size S')
    ax.set_ylabel('|Δ log p̂(D)| (nats)')
    ax.legend(frameon=False)
    if has_nested:
        ax = axes[2]
        nested = df[df['logbf_ref'].notna()]
        lim = max(nested[['logbf_ref', 'logbf_is_r', 'logbf_is_z']].abs().max()) * 1.05
        for e in np.concatenate([-JEFFREYS_EDGES, JEFFREYS_EDGES]):
            ax.axhline(e, color='0.85', lw=0.6)
            ax.axvline(e, color='0.85', lw=0.6)
        ax.plot([-lim, lim], [-lim, lim], color='0.6', lw=0.8)
        for c in COORDS:
            ax.scatter(
                nested['logbf_ref'],
                nested[f'logbf_is_{c}'],
                s=14,
                alpha=0.8,
                color=colors[c],
                edgecolor='none',
                label=f"IS('{c}')",
            )
        ax.set_xscale('symlog', linthresh=5.0)
        ax.set_yscale('symlog', linthresh=5.0)
        ticks = [-1000, -100, -10, 0, 10, 100, 1000]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel('ln BF (bridge reference)')
        ax.set_ylabel('ln BF (IS)')
        ax.legend(frameon=False)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(label, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def runSize(size: str, args: argparse.Namespace) -> None:
    seed = BEST_SEEDS[('normal', size)]
    ckpt = _ckpt_dir('normal', size, seed) / f'{args.prefix}.pt'
    data_dir = DATA_DIR / f'{size}-n-sampled'
    npz_path = data_dir / f'{args.split}.fit.npz'
    label = f'Normal ({size}), {args.split}, n={args.n_datasets}'
    print(f'\n{"#" * 70}\n#  {label}\n{"#" * 70}')

    model, epoch = loadModel(ckpt)
    model.to(args.device)
    d_corr_model = model.d_corr
    col = Collection(npz_path, permute=False, exclude_prefixes=('nuts_', 'advi_', 'laplace_'))
    n_ds = min(args.n_datasets, len(col))
    nuts = loadNuts(npz_path, n_ds)
    print(f'model epoch={epoch}  d_ffx={model.d_ffx}  d_rfx={model.d_rfx}  datasets={n_ds}')

    pool_sizes = sorted(args.pool_sizes)
    s_max = pool_sizes[-1]
    gen = torch.Generator().manual_seed(args.seed)
    rows = []
    n_nested = 0
    t0 = time.perf_counter()
    for i in range(n_ds):
        t_i = time.perf_counter()
        item = col[i]
        batch = collateGrouped([item])
        batch64 = toDouble(batch)
        target = Target(batch64, d_corr_model)
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
            'nuts_min_ess': float(
                np.nanmin(np.where(nuts['nuts_ess'][i] > 0, nuts['nuts_ess'][i], np.nan))
            ),
            'nuts_div': int(nuts['nuts_divergences'][i].sum()),
        }

        # flow proposal (standardized space) and IS evidence in both coordinate conventions
        torch.manual_seed(args.seed * 100_003 + i)
        with torch.no_grad():
            proposal = model.estimate(
                {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()},
                n_samples=s_max,
            )
        proposal.to('cpu')
        weights = {}
        for c in COORDS:
            res = isEvidence(proposal, batch64, pool_sizes, c)
            weights[c] = res.pop('weights')
            row.update(res)

        # IMH (exact marginal target, 'r' coordinates): acceptance + posterior draws
        p_imh, accept = runImh(proposal, batch64)
        row['imh_accept'] = accept

        # bridge references
        br = bridge(target.logProb, nutsUnconstrained(target, nuts, i), args.n_bridge, gen)
        row.update(
            {
                'logev_bridge_nuts': br['log_ev'],
                'logev_bridge_nuts_swap': br['log_ev_swap'],
                'bridge_nuts_iter': br['n_iter'],
            }
        )
        br = bridge(target.logProb, imhUnconstrained(target, p_imh), args.n_bridge, gen)
        row.update({'logev_bridge_imh': br['log_ev'], 'logev_bridge_imh_swap': br['log_ev_swap']})

        # correlation tilt check (first rfx correlation)
        if target.corr:
            rho_flow = proposal.samples_g[0, :, -d_corr_model].double()
            row['rho_nuts'] = float(nuts['nuts_corr_rfx'][i, 0, :, 1, 0].mean())
            row['rho_raw'] = float(rho_flow.mean())
            row['rho_is_z'] = float((weights['z'] * rho_flow).sum())
            row['rho_is_r'] = float((weights['r'] * rho_flow).sum())
            row['rho_imh'] = float(p_imh.corr_rfx[0, :, 1, 0].mean())

        # nested comparison: drop the random slope
        if target.q_i >= 2 and n_nested < args.nested:
            n_nested += 1
            batch_r = collateGrouped([reducedItem(item)])
            batch64_r = toDouble(batch_r)
            torch.manual_seed(args.seed * 100_003 + i + 50_000)
            with torch.no_grad():
                proposal_r = model.estimate(
                    {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch_r.items()},
                    n_samples=s_max,
                )
            proposal_r.to('cpu')
            res_r = isEvidence(proposal_r, batch64_r, [s_max], 'r')
            row['logev_is_red'] = res_r[f'logev_is_r_s{s_max}']
            row['k_red'] = res_r[f'k_r_s{s_max}']
            p_imh_r, row['imh_accept_red'] = runImh(proposal_r, batch64_r)
            target_r = Target(batch64_r, d_corr_model)
            br = bridge(target_r.logProb, imhUnconstrained(target_r, p_imh_r), args.n_bridge, gen)
            row['logev_bridge_imh_red'] = br['log_ev']
            row['logev_bridge_imh_red_swap'] = br['log_ev_swap']
            for c in COORDS:
                row[f'logbf_is_{c}'] = row[f'logev_is_{c}_s{s_max}'] - row['logev_is_red']
            row['logbf_ref'] = row['logev_bridge_nuts'] - row['logev_bridge_imh_red']
            row['logbf_ref_imh'] = row['logev_bridge_imh'] - row['logev_bridge_imh_red']

        rows.append(row)
        d_z = row[f'logev_is_z_s{s_max}'] - row['logev_bridge_nuts']
        d_r = row[f'logev_is_r_s{s_max}'] - row['logev_bridge_nuts']
        print(
            f'ds={i:3d} d={target.d_i} q={target.q_i} m={row["m"]:3d} n={row["n"]:4d} corr={int(target.corr)}  '
            f'bridge={row["logev_bridge_nuts"]:9.2f} (swap Δ={row["logev_bridge_nuts"] - row["logev_bridge_nuts_swap"]:+.3f}, '
            f'imh Δ={row["logev_bridge_imh"] - row["logev_bridge_nuts"]:+.3f})  '
            f"IS Δz={d_z:+.3f} Δr={d_r:+.3f}  k={row[f'k_r_s{s_max}']:.2f} acc={accept:.2f}"
            + (
                f'  lnBF ref={row["logbf_ref"]:+.2f} is={row["logbf_is_r"]:+.2f}'
                if 'logbf_ref' in row
                else ''
            )
            + f'  [{time.perf_counter() - t_i:.1f}s]',
            flush=True,
        )

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{size}_{args.split}_n{n_ds}'
    df.to_csv(args.out_dir / f'{stem}.csv', index=False)
    md = summarize(df, pool_sizes, label)
    (args.out_dir / f'{stem}.md').write_text(md)
    plot(df, pool_sizes, args.out_dir / f'{stem}.png', label)
    print(md)
    print(f'[saved] {args.out_dir / stem}.{{csv,md,png}}  ({time.perf_counter() - t0:.0f}s)')


def writeTex(out_dir: Path, split: str, pool_sizes: list[int]) -> Path | None:
    """Appendix table over all sizes with a CSV in out_dir (consistent 'r' coordinates)."""
    csvs = {size: sorted(out_dir.glob(f'{size}_{split}_n*.csv')) for size in SIZES}
    csvs = {size: paths[-1] for size, paths in csvs.items() if paths}
    if not csvs:
        return None
    lines = [
        r'\begin{tabular}{lrr r cc c r cc}',
        r'    \toprule',
        r'    \mathrm{regime} & \#\mathrm{ds} & \#\mathrm{corr} & S & '
        r'\mathrm{med}\,|\Delta\log p(\mathcal{D})| & q_{90}\,|\Delta\log p(\mathcal{D})| & '
        r'\mathrm{frac}\,k>0.7 & \#\mathrm{nested} & \mathrm{BF\ cat.\ agree} & '
        r'\mathrm{med}\,|\Delta\ln \mathrm{BF}| \\',
        r'    \midrule',
    ]
    for size, path in csvs.items():
        df = pd.read_csv(path)
        ref = df['logev_bridge_nuts']
        nested = df[df['logbf_ref'].notna()] if 'logbf_ref' in df else df.iloc[0:0]
        first = True
        for s in (min(pool_sizes), max(pool_sizes)):
            col = f'logev_is_r_s{s}'
            if col not in df:
                continue
            d = (df[col] - ref).abs()
            frac_k = (df[f'k_r_s{s}'] > 0.7).mean()
            if s == max(pool_sizes) and len(nested):
                agree = categoryAgreement(nested['logbf_is_r'], nested['logbf_ref'])
                bf = (nested['logbf_is_r'] - nested['logbf_ref']).abs().median()
                nested_cells = f'{len(nested)} & ${agree:.2f}$ & ${bf:.3f}$'
            else:
                nested_cells = ' & & '
            head = (
                f'    \\texttt{{{size}}} & {len(df)} & {int(df["corr_active"].sum())}'
                if first
                else '     &  & '
            )
            lines.append(
                f'{head} & {s} & ${d.median():.3f}$ & ${d.quantile(0.9):.3f}$ & '
                f'${frac_k:.2f}$ & {nested_cells} \\\\'
            )
            first = False
        lines.append(r'    \midrule')
    lines[-1] = r'    \bottomrule'
    lines.append(r'\end{tabular}')
    path = out_dir / f'evidence_normal.tex'
    path.write_text('\n'.join(lines) + '\n')
    return path


def resummarize(size: str, args: argparse.Namespace) -> None:
    paths = sorted(args.out_dir.glob(f'{size}_{args.split}_n*.csv'))
    if not paths:
        print(f'[skip] no CSV for {size} in {args.out_dir}')
        return
    df = pd.read_csv(paths[-1])
    stem = paths[-1].stem
    label = f'Normal ({size}), {args.split}, n={len(df)}'
    pool_sizes = sorted(args.pool_sizes)
    md = summarize(df, pool_sizes, label)
    (args.out_dir / f'{stem}.md').write_text(md)
    plot(df, pool_sizes, args.out_dir / f'{stem}.png', label)
    print(md)


def main() -> None:
    args = setup()
    for size in args.sizes:
        if args.summarize_only:
            resummarize(size, args)
        else:
            runSize(size, args)
    # the combined table spans all sizes with a CSV in out_dir; assemble it only in the
    # summarize-only pass so parallel per-size runs do not race on the same file
    if args.summarize_only:
        tex = writeTex(args.out_dir, args.split, sorted(args.pool_sizes))
        if tex is not None:
            print(f'[saved] {tex}')


if __name__ == '__main__':
    main()
