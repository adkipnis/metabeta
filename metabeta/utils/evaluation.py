from typing import Literal, Sequence
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from metabeta.utils.dataloader import toDevice
from metabeta.utils.regularization import corrLowerToCorr


Proposed = dict[str, dict[str, torch.Tensor]]
Source = Literal['global', 'local']


def getMasks(
    data: dict[str, torch.Tensor], has_sigma_eps: bool = True
) -> dict[str, torch.Tensor | None]:
    out = {}
    out['ffx'] = data['mask_d']
    out['sigma_rfx'] = data['mask_q']
    out['rfx'] = data['mask_mq']
    if has_sigma_eps:
        out['sigma_eps'] = None
        out['sigmas'] = torch.nn.functional.pad(data['mask_q'], (0, 1), value=True)
    else:
        out['sigmas'] = data['mask_q']
    out['global'] = torch.cat([out['ffx'], out['sigmas']], dim=-1)
    return out


def getNames(source: str, num: int, has_sigma_eps: bool = True) -> list[str]:
    if source == 'ffx':
        return [rf'$\beta_{{{i}}}$' for i in range(num)]
    elif source == 'sigmas':
        names = [rf'$\sigma_{i}$' for i in range(num)]
        if has_sigma_eps:
            names.append(r'$\sigma_\epsilon$')
        return names
    elif source == 'rfx':
        return [rf'$\alpha_{{{i}}}$' for i in range(num)]
    else:
        raise ValueError(f'source {source} unknown.')


def getCorrRfxNames(q: int) -> list[str]:
    """Names for the lower-triangle correlation pairs: ρ_{01}, ρ_{02}, ρ_{12}, ..."""
    return [rf'$\rho_{{{j}{i}}}$' for i in range(q) for j in range(i)]


def getAllNames(d: int, q: int, has_sigma_eps: bool = True) -> dict[str, list[str]]:
    return {
        'ffx': getNames('ffx', d),
        'sigmas': getNames('sigmas', q, has_sigma_eps),
        'rfx': getNames('rfx', q),
    }


def joinSigmas(data: dict[str, torch.Tensor]) -> torch.Tensor:
    parts = [data['sigma_rfx']]
    if 'sigma_eps' in data:
        parts.append(data['sigma_eps'].unsqueeze(-1))
    return torch.cat(parts, dim=-1)


def joinGlobals(data: dict[str, torch.Tensor]) -> torch.Tensor:
    parts = [data['ffx'], data['sigma_rfx']]
    if 'sigma_eps' in data:
        parts.append(data['sigma_eps'].unsqueeze(-1))
    return torch.cat(parts, dim=-1)


class Proposal:
    def __init__(
        self,
        proposed: dict[str, dict[str, torch.Tensor]],
        has_sigma_eps: bool = True,
        d_corr: int = 0,
        corr_rfx: torch.Tensor | None = None,
    ) -> None:
        self.data = proposed
        self.has_sigma_eps = has_sigma_eps
        self.d_corr = d_corr
        self._corr_rfx = corr_rfx
        self.q = proposed['local']['samples'].shape[-1]
        D = proposed['global']['samples'].shape[-1]
        self.d = D - self.q - (1 if has_sigma_eps else 0) - d_corr
        self.is_results = {}
        self.tpd: float | None = None
        self.reff: float = 1.0  # relative ESS; 1.0 for i.i.d. flow samples, <1 for MCMC
        self.debug_stats: dict | None = None  # DEBUG: Jacobian diagnostics from _postprocess

    @property
    def n_samples(self) -> int:
        return self.samples_g.shape[-2]

    def __repr__(self) -> str:
        return f'ProposalPosterior(n_samples={self.n_samples})'

    def to(self, device: str | torch.device) -> None:
        self.data['global'] = toDevice(self.data['global'], device)
        self.data['local'] = toDevice(self.data['local'], device)
        if self._corr_rfx is not None:
            self._corr_rfx = self._corr_rfx.to(device)
        self.is_results = toDevice(self.is_results, device)

    @property
    def samples_g(self) -> torch.Tensor:
        return self.data['global']['samples']

    @property
    def samples_l(self) -> torch.Tensor:
        return self.data['local']['samples']

    @property
    def log_prob_g(self) -> torch.Tensor:
        return self.data['global']['log_prob']

    @property
    def log_prob_l(self) -> torch.Tensor:
        return self.data['local']['log_prob']

    @property
    def ffx(self) -> torch.Tensor:
        return self.samples_g[..., : self.d]

    @property
    def sigma_rfx(self) -> torch.Tensor:
        return self.samples_g[..., self.d : self.d + self.q]

    @property
    def sigma_eps(self) -> torch.Tensor:
        if not self.has_sigma_eps:
            raise AttributeError('sigma_eps not available for this likelihood')
        return self.samples_g[..., self.d + self.q]

    @property
    def sigmas(self) -> torch.Tensor:
        end = self.d + self.q + (1 if self.has_sigma_eps else 0)
        return self.samples_g[..., self.d : end]

    @property
    def corr_rfx(self) -> torch.Tensor | None:
        """Correlation matrix, shape (..., q, q).

        For flow proposals: reconstructed from unconstrained global samples.
        For NUTS/ADVI proposals: stored directly from the trace.
        Returns None when neither source is available.
        """
        if self._corr_rfx is not None:
            return self._corr_rfx
        if self.d_corr == 0:
            return None
        return corrLowerToCorr(self.samples_g[..., -self.d_corr :], self.q)

    @property
    def rfx(self) -> torch.Tensor:
        return self.samples_l

    @property
    def parameters(self) -> tuple[torch.Tensor, ...]:
        if self.has_sigma_eps:
            return self.ffx, self.sigma_rfx, self.sigma_eps, self.rfx
        return self.ffx, self.sigma_rfx, self.rfx

    @property
    def log_probs(self) -> tuple[torch.Tensor, ...]:
        return self.log_prob_g, self.log_prob_l

    @property
    def weights(self) -> torch.Tensor | None:
        return self.is_results.get('weights', None)

    @property
    def efficiency(self) -> torch.Tensor | None:
        return self.is_results.get('sample_efficiency')

    @property
    def pareto_k(self) -> torch.Tensor | None:
        return self.is_results.get('pareto_k')

    def partition(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        out['ffx'] = x[..., : self.d]
        out['sigma_rfx'] = x[..., self.d : self.d + self.q]
        if self.has_sigma_eps:
            out['sigma_eps'] = x[..., self.d + self.q]
        return out

    def slice_b(self, start: int, end: int) -> 'Proposal':
        """Return a view of this proposal restricted to datasets [start:end]."""
        new_data = {
            src: {k: v[start:end] for k, v in inner.items()} for src, inner in self.data.items()
        }
        corr = self._corr_rfx[start:end] if self._corr_rfx is not None else None
        sliced = Proposal(
            new_data, has_sigma_eps=self.has_sigma_eps, d_corr=self.d_corr, corr_rfx=corr
        )
        sliced.reff = self.reff
        sliced.is_results = {
            k: v[start:end] if torch.is_tensor(v) else v for k, v in self.is_results.items()
        }
        return sliced

    def rescale(self, scale: torch.Tensor) -> None:
        samples_l = self.data['local']['samples']
        scale_l = scale.view(-1, *([1] * (samples_l.ndim - 1)))
        self.data['local']['samples'] = samples_l * scale_l

        samples_g = self.data['global']['samples']
        scale_g = scale.view(-1, *([1] * (samples_g.ndim - 1)))
        if self.d_corr > 0:
            non_corr = samples_g[..., : -self.d_corr] * scale_g
            self.data['global']['samples'] = torch.cat(
                [non_corr, samples_g[..., -self.d_corr :]], dim=-1
            )
        else:
            self.data['global']['samples'] = samples_g * scale_g

    def subset(self, idx: torch.Tensor) -> None:
        b, s = idx.shape
        for source in ('global', 'local'):
            # samples
            samples = self.data[source]['samples']
            m, d = samples.shape[1], samples.shape[-1]
            index = idx.unsqueeze(-1).expand(b, s, d)
            if source == 'local':
                index = index.unsqueeze(1).expand(b, m, s, d)
            re_samples = torch.gather(samples, dim=-2, index=index)
            self.data[source]['samples'] = re_samples

            # log probs
            log_prob = self.data[source]['log_prob']
            index = idx.clone()
            if source == 'local':
                index = index.unsqueeze(1).expand(b, m, s)
            re_log_prob = torch.gather(log_prob, dim=-1, index=index)
            self.data[source]['log_prob'] = re_log_prob


def joinProposals(proposals: list[Proposal]) -> Proposal:
    has_sigma_eps = [p.has_sigma_eps for p in proposals]
    assert len(set(has_sigma_eps)) == 1, 'proposals disagree whether they have sigma_eps'
    d_corrs = [p.d_corr for p in proposals]
    assert len(set(d_corrs)) == 1, 'proposals disagree on d_corr'
    samples_g = torch.cat([p.samples_g for p in proposals], dim=-2)
    samples_l = torch.cat([p.samples_l for p in proposals], dim=-2)
    log_prob_g = torch.cat([p.log_prob_g for p in proposals], dim=-1)
    log_prob_l = torch.cat([p.log_prob_l for p in proposals], dim=-1)
    proposed = {
        'global': {'samples': samples_g, 'log_prob': log_prob_g},
        'local': {'samples': samples_l, 'log_prob': log_prob_l},
    }
    corr = None
    corr_parts = [p._corr_rfx for p in proposals]
    if all(part is not None for part in corr_parts):
        corr = torch.cat(corr_parts, dim=-3)  # type: ignore[arg-type]
    return Proposal(proposed, has_sigma_eps=has_sigma_eps[0], d_corr=d_corrs[0], corr_rfx=corr)


def concatProposalsBatch(proposals: list[Proposal]) -> Proposal:
    assert len(proposals) > 0, 'at least one proposal is required'
    has_sigma_eps = proposals[0].has_sigma_eps
    max_m = max(proposal.samples_l.shape[1] for proposal in proposals)

    samples_l_padded = []
    log_prob_l_padded = []
    for proposal in proposals:
        samples_l = proposal.samples_l
        log_prob_l = proposal.log_prob_l
        b, m, s, q = samples_l.shape
        if m < max_m:
            pad_samples = samples_l.new_zeros((b, max_m, s, q))
            pad_samples[:, :m] = samples_l
            samples_l = pad_samples

            pad_log_prob = log_prob_l.new_zeros((b, max_m, s))
            pad_log_prob[:, :m] = log_prob_l
            log_prob_l = pad_log_prob

        samples_l_padded.append(samples_l)
        log_prob_l_padded.append(log_prob_l)

    proposed = {
        'global': {
            'samples': torch.cat([proposal.samples_g for proposal in proposals], dim=0),
            'log_prob': torch.cat([proposal.log_prob_g for proposal in proposals], dim=0),
        },
        'local': {
            'samples': torch.cat(samples_l_padded, dim=0),
            'log_prob': torch.cat(log_prob_l_padded, dim=0),
        },
    }
    corr = None
    corr_parts = [proposal._corr_rfx for proposal in proposals]
    if all(part is not None for part in corr_parts):
        corr = torch.cat(corr_parts, dim=0)  # type: ignore[arg-type]
    merged = Proposal(
        proposed, has_sigma_eps=has_sigma_eps, d_corr=proposals[0].d_corr, corr_rfx=corr
    )

    common_keys: set[str] | None = None
    for proposal in proposals:
        keys = set(proposal.is_results.keys())
        if common_keys is None:
            common_keys = keys
        else:
            common_keys &= keys

    if common_keys:
        for key in common_keys:
            values = [proposal.is_results[key] for proposal in proposals]
            if all(torch.is_tensor(value) for value in values):
                try:
                    merged.is_results[key] = torch.cat(values, dim=0)
                except RuntimeError:
                    continue

    # DEBUG: average per-batch Jacobian diagnostics across minibatches
    debug_list = [p.debug_stats for p in proposals if p.debug_stats is not None]
    if debug_list:
        keys = debug_list[0].keys()
        merged.debug_stats = {k: sum(d[k] for d in debug_list) / len(debug_list) for k in keys}

    return merged


def weightedQuantile(
    x: torch.Tensor,
    w: torch.Tensor,
    q: float | Sequence[float] | torch.Tensor = 0.5,
) -> torch.Tensor:
    if not isinstance(q, torch.Tensor):
        q_t = torch.tensor(q, dtype=x.dtype, device=x.device)
    else:
        q_t = q
    w = w.unsqueeze(-1)
    if x.dim() == 4:
        w = w.unsqueeze(1)
    # sort samples and weights
    x_sorted, order = torch.sort(x, dim=-2)
    w_sorted = torch.gather(w.expand_as(x), dim=-2, index=order)

    # search sorted
    x_last = x_sorted.movedim(-2, -1)
    cdf = torch.cumsum(w_sorted, dim=-2).movedim(-2, -1).contiguous()
    t = q_t * torch.ones_like(cdf[..., 0:1])
    idx = torch.searchsorted(cdf, t, right=False).clamp(max=x_last.shape[-1] - 1)
    out = torch.gather(x_last, dim=-1, index=idx).squeeze(-1)
    return out


@dataclass
class PerDatasetMetrics:
    """Per-dataset tensors (shape B or n_alphas×B). Safe to subset by a boolean mask."""

    posterior_nll: torch.Tensor                   # (B,)
    loo_nll: torch.Tensor | None = None           # (B,)
    loo_pareto_k: torch.Tensor | None = None      # (B,)
    pp_fit: torch.Tensor | None = None            # (B,)  R² | AUC | deviance
    pp_cov_coverage: torch.Tensor | None = None   # (n_alphas, B)
    pp_cov_width: torch.Tensor | None = None      # (n_alphas, B)
    sample_efficiency: torch.Tensor | None = None   # (B,)
    pareto_k: torch.Tensor | None = None          # (B,)
    prior_nll: torch.Tensor | None = None         # (B,)  only when compute_prior=True

    @property
    def mnll(self) -> float:
        return float(self.posterior_nll.median())

    @property
    def mloonll(self) -> float | None:
        return float(self.loo_nll.median()) if self.loo_nll is not None else None

    @property
    def mloo_k(self) -> float | None:
        return float(self.loo_pareto_k.nanmedian()) if self.loo_pareto_k is not None else None

    @property
    def mfit(self) -> float | None:
        return float(self.pp_fit.nanmedian()) if self.pp_fit is not None else None

    @property
    def meff(self) -> float | None:
        return (
            float(self.sample_efficiency.median()) if self.sample_efficiency is not None else None
        )

    @property
    def mk(self) -> float | None:
        return float(self.pareto_k.median()) if self.pareto_k is not None else None

    @property
    def pp_eace(self) -> float | None:
        if self.pp_cov_coverage is None:
            return None
        from metabeta.evaluation.intervals import ALPHAS

        n = min(len(ALPHAS), self.pp_cov_coverage.shape[0])
        nominals = self.pp_cov_coverage.new_tensor([1.0 - a for a in ALPHAS[:n]])
        err = (self.pp_cov_coverage[:n] - nominals.unsqueeze(-1)).abs()  # (n, B)
        return float(err.mean(dim=0).median())

    @property
    def pp_width_90(self) -> float | None:
        if self.pp_cov_width is None:
            return None
        from metabeta.evaluation.intervals import ALPHAS

        try:
            idx = ALPHAS.index(0.1)
        except ValueError:
            idx = 0
        return float(self.pp_cov_width[idx].median())

    def subset(self, mask: np.ndarray) -> 'PerDatasetMetrics':
        idx = torch.from_numpy(mask)
        s1 = lambda t: None if t is None else t[idx]       # noqa: E731
        s2 = lambda t: None if t is None else t[:, idx]    # noqa: E731
        return PerDatasetMetrics(
            posterior_nll=self.posterior_nll[idx],
            loo_nll=s1(self.loo_nll),
            loo_pareto_k=s1(self.loo_pareto_k),
            pp_fit=s1(self.pp_fit),
            pp_cov_coverage=s2(self.pp_cov_coverage),
            pp_cov_width=s2(self.pp_cov_width),
            sample_efficiency=s1(self.sample_efficiency),
            pareto_k=s1(self.pareto_k),
            prior_nll=s1(self.prior_nll),
        )


@dataclass
class AggregatedMetrics:
    """Metrics already aggregated over datasets. Not meaningfully subsettable."""

    corr: dict[str, torch.Tensor]
    nrmse: dict[str, torch.Tensor]
    coverage: dict[float, dict[str, torch.Tensor]]
    ece: dict[str, torch.Tensor]    # mean |coverage_error| over alphas, per param dim
    eace: dict[str, torch.Tensor]   # mean abs|coverage_error| over alphas
    lcr: dict[str, torch.Tensor]    # mean log-coverage-ratio over alphas
    abs_lcr: dict[str, torch.Tensor]
    estimates: dict[str, torch.Tensor]
    rfx_joint_ece: float | None = None
    rfx_joint_eace: float | None = None


_SUMMARY_VERSION = 1


@dataclass
class EvaluationSummary:
    per_dataset: PerDatasetMetrics
    aggregated: AggregatedMetrics
    tpd: float | None = None

    def subset(self, mask: np.ndarray) -> 'EvaluationSummary':
        """Return a new summary with per-dataset fields filtered by a boolean mask.

        Aggregated fields (coverage, corr, nrmse, ece, …) are carried over unchanged
        — they reflect the original population and are not re-aggregated.
        """
        return EvaluationSummary(
            per_dataset=self.per_dataset.subset(mask),
            aggregated=self.aggregated,
            tpd=self.tpd,
        )

    def save(self, path: Path | str) -> None:
        pd, ag = self.per_dataset, self.aggregated
        torch.save(
            {
                '_version': _SUMMARY_VERSION,
                # PerDatasetMetrics
                'posterior_nll': pd.posterior_nll,
                'loo_nll': pd.loo_nll,
                'loo_pareto_k': pd.loo_pareto_k,
                'pp_fit': pd.pp_fit,
                'pp_cov_coverage': pd.pp_cov_coverage,
                'pp_cov_width': pd.pp_cov_width,
                'sample_efficiency': pd.sample_efficiency,
                'pareto_k': pd.pareto_k,
                'prior_nll': pd.prior_nll,
                # AggregatedMetrics
                'corr': ag.corr,
                'nrmse': ag.nrmse,
                'coverage': ag.coverage,
                'ece': ag.ece,
                'eace': ag.eace,
                'lcr': ag.lcr,
                'abs_lcr': ag.abs_lcr,
                'estimates': ag.estimates,
                'rfx_joint_ece': ag.rfx_joint_ece,
                'rfx_joint_eace': ag.rfx_joint_eace,
                # top-level
                'tpd': self.tpd,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> 'EvaluationSummary':
        d = torch.load(path, weights_only=False)
        v = d.get('_version')
        if v != _SUMMARY_VERSION:
            raise ValueError(f'unsupported summary version: {v}')
        per_dataset = PerDatasetMetrics(
            posterior_nll=d['posterior_nll'],
            loo_nll=d.get('loo_nll'),
            loo_pareto_k=d.get('loo_pareto_k'),
            pp_fit=d.get('pp_fit'),
            pp_cov_coverage=d.get('pp_cov_coverage'),
            pp_cov_width=d.get('pp_cov_width'),
            sample_efficiency=d.get('sample_efficiency'),
            pareto_k=d.get('pareto_k'),
            prior_nll=d.get('prior_nll'),
        )
        aggregated = AggregatedMetrics(
            corr=d['corr'],
            nrmse=d['nrmse'],
            coverage=d['coverage'],
            ece=d['ece'],
            eace=d['eace'],
            lcr=d['lcr'],
            abs_lcr=d['abs_lcr'],
            estimates=d['estimates'],
            rfx_joint_ece=d.get('rfx_joint_ece'),
            rfx_joint_eace=d.get('rfx_joint_eace'),
        )
        return cls(per_dataset=per_dataset, aggregated=aggregated, tpd=d.get('tpd'))


_NUTS_THRESHOLDS = {
    # Intended for posterior shape comparisons: NUTS must be a reliable reference.
    'strict': dict(div_rate=0.005, rhat=1.01, ess=400, td=0.05),
    # Intended for LOO-NLL comparisons: removes only directionally-biased failures
    # (high divergence rate → posterior too concentrated → LOO-NLL too optimistic).
    # Keeps ~92% of datasets on small-n-sampled.
    'liberal': dict(div_rate=0.020, rhat=1.05, ess=200, td=0.20),
}


def nutsConvergeMask(
    batch: dict[str, torch.Tensor],
    mode: str = 'strict',
) -> np.ndarray | None:
    """Boolean mask (shape b) that is True for NUTS-converged datasets.

    Args:
        batch: Dataset batch containing nuts_* diagnostic arrays.
        mode:  'strict' (default) — for posterior shape comparisons, requires
                   div_rate ≤ 0.5%, rhat ≤ 1.01, ESS ≥ 400, td_sat ≤ 5%.
               'liberal' — for LOO-NLL comparisons, removes only clear failures:
                   div_rate ≤ 2%, rhat ≤ 1.05, ESS ≥ 200, td_sat ≤ 20%.
    """
    if 'nuts_divergences' not in batch:
        return None

    thr = _NUTS_THRESHOLDS[mode]

    def _field(key: str) -> np.ndarray | None:
        return batch[key].numpy().astype(np.float64) if key in batch else None

    def _param_stat(arr: np.ndarray | None, fn) -> np.ndarray | None:
        if arr is None:
            return None
        a = arr.copy()
        a[a <= 0] = np.nan
        return fn(a, axis=-1)

    div_arr = batch['nuts_divergences'].numpy()                    # (b, chains)
    total_div = div_arr.sum(-1)                                    # (b,)
    n_chains = div_arr.shape[-1]
    n_draws = int(batch['nuts_draws'].item()) if 'nuts_draws' in batch else 1000
    total_samples = n_chains * n_draws

    max_rhat = _param_stat(_field('nuts_rhat'), np.nanmax)
    min_ess = _param_stat(_field('nuts_ess'), np.nanmin)
    min_ess_tail = _param_stat(_field('nuts_ess_tail'), np.nanmin)
    treedepth = _field('nuts_max_treedepth')
    mean_treedepth_sat = treedepth.mean(-1) if treedepth is not None else None

    b = len(total_div)
    f_rhat = (max_rhat > thr['rhat']) if max_rhat is not None else np.zeros(b, bool)
    f_div = (total_div / total_samples) > thr['div_rate']
    f_tree = (
        (mean_treedepth_sat > thr['td']) if mean_treedepth_sat is not None else np.zeros(b, bool)
    )
    f_ess = (min_ess < thr['ess']) if min_ess is not None else np.zeros(b, bool)
    f_ess_tail = (min_ess_tail < thr['ess']) if min_ess_tail is not None else np.zeros(b, bool)
    return ~(f_rhat | f_div | f_tree | f_ess | f_ess_tail)


def subsetProposal(proposal: 'Proposal', mask: np.ndarray) -> 'Proposal':
    """Return a new Proposal restricted to datasets selected by a boolean mask."""
    idx = torch.from_numpy(mask)
    new_data = {src: {k: v[idx] for k, v in inner.items()} for src, inner in proposal.data.items()}
    corr = proposal._corr_rfx[idx] if proposal._corr_rfx is not None else None
    sub = Proposal(
        new_data,
        has_sigma_eps=proposal.has_sigma_eps,
        d_corr=proposal.d_corr,
        corr_rfx=corr,
    )
    sub.reff = proposal.reff
    sub.tpd = proposal.tpd
    sub.is_results = {
        k: v[idx] if torch.is_tensor(v) else v for k, v in proposal.is_results.items()
    }
    return sub


def makePriorProposal(result: object, n_samples: int = 1000) -> Proposal:
    """Sample from the prior distributions stored in result, in original-scale units."""
    pp, si = result.prior_params, result.scale_info  # type: ignore[attr-defined]
    sd_y = si.y_std
    d = len(result.param_names['ffx'])  # type: ignore[attr-defined]
    q = len(result.param_names['sigma_rfx'])  # type: ignore[attr-defined]

    tau_f = torch.tensor(pp['tau_ffx'][0, :d], dtype=torch.float32) * sd_y
    nu_f = torch.tensor(pp['nu_ffx'][0, :d], dtype=torch.float32) * sd_y
    tau_r = torch.tensor(pp['tau_rfx'][0, :q], dtype=torch.float32) * sd_y

    ffx = torch.distributions.Normal(nu_f, tau_f).sample((n_samples,))
    srfx = torch.distributions.HalfNormal(tau_r).sample((n_samples,))
    seps = torch.distributions.HalfNormal(torch.tensor([2.5 * sd_y])).sample((n_samples,))

    ffx_orig = si.to_original_scale(ffx)
    samples_g = torch.cat([ffx_orig, srfx, seps], dim=-1).unsqueeze(0)  # (1, S, d+q+1)
    return Proposal(
        proposed={
            'global': {'samples': samples_g, 'log_prob': torch.zeros(1, n_samples)},
            'local': {
                'samples': torch.zeros(1, 1, n_samples, q),
                'log_prob': torch.zeros(1, 1, n_samples),
            },
        },
        has_sigma_eps=True,
    )


def makePriorPDFs(
    result: object, n_grid: int = 500
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute analytical prior PDFs in original-scale units, one per global parameter.

    Returns (x_grid, density) pairs ordered as: ffx[0..d-1], sigma_rfx[0..q-1], sigma_eps.
    The transform to original scale is applied analytically so the curves are exact
    (no sampling noise), matching the distributions described in posteriorSummary.
    """
    from scipy.stats import norm, halfnorm, t as student_t
    from metabeta.utils.constants import FFX_FAMILIES, SIGMA_FAMILIES, STUDENT_DF

    pp, si = result.prior_params, result.scale_info  # type: ignore[attr-defined]
    sd_y = float(si.y_std)
    d = len(result.param_names['ffx'])  # type: ignore[attr-defined]
    q = len(result.param_names['sigma_rfx'])  # type: ignore[attr-defined]

    tau_f = np.asarray(pp['tau_ffx'][0, :d], dtype=float) * sd_y
    nu_f = np.asarray(pp['nu_ffx'][0, :d], dtype=float) * sd_y
    tau_r = np.asarray(pp['tau_rfx'][0, :q], dtype=float) * sd_y
    tau_e = float(np.asarray(pp['tau_eps']).ravel()[0]) * sd_y

    fam_ffx_id = int(pp['family_ffx'][0]) if 'family_ffx' in pp else 0
    fam_rfx_id = int(pp['family_sigma_rfx'][0]) if 'family_sigma_rfx' in pp else 0
    fam_eps_id = int(np.asarray(pp['family_sigma_eps']).ravel()[0]) if 'family_sigma_eps' in pp else 0
    fam_ffx = FFX_FAMILIES[fam_ffx_id] if fam_ffx_id < len(FFX_FAMILIES) else 'normal'
    fam_rfx = SIGMA_FAMILIES[fam_rfx_id] if fam_rfx_id < len(SIGMA_FAMILIES) else 'halfnormal'
    fam_eps = SIGMA_FAMILIES[fam_eps_id] if fam_eps_id < len(SIGMA_FAMILIES) else 'halfstudent'

    x_stds = np.asarray(si.x_stds, dtype=float)
    x_means = np.asarray(si.x_means, dtype=float)

    def _sym_pdf(mu: float, sigma: float, fam: str) -> tuple[np.ndarray, np.ndarray]:
        span = max(sigma * 6, 1e-9)
        xg = np.linspace(mu - span, mu + span, n_grid)
        if fam == 'normal':
            return xg, norm.pdf(xg, loc=mu, scale=sigma)
        return xg, student_t.pdf(xg, df=STUDENT_DF, loc=mu, scale=sigma)

    def _half_pdf(tau: float, fam: str) -> tuple[np.ndarray, np.ndarray]:
        span = max(tau * 6, 1e-9)
        xg = np.linspace(0.0, span, n_grid)
        if fam == 'halfnormal':
            return xg, halfnorm.pdf(xg, scale=tau)
        return xg, np.where(xg >= 0, 2 * student_t.pdf(xg, df=STUDENT_DF, loc=0, scale=tau), 0.0)

    pdfs: list[tuple[np.ndarray, np.ndarray]] = []

    # FFX: back-transform to original scale analytically.
    # After to_original_scale (linear map):
    #   slope_orig[j] = ffx[j] / x_stds[j]   for j ≥ 1
    #   intercept_orig = ffx[0] + y_mean − Σ_{j≥1}(ffx[j] / x_stds[j] * x_means[j])
    # → slope_orig[j] ~ N(nu_f[j]/x_stds[j], tau_f[j]/x_stds[j])
    # → intercept_orig is Gaussian with mean/var computed below
    mu_int = float(nu_f[0] + si.y_mean - sum(nu_f[j] / x_stds[j] * x_means[j] for j in range(1, d)))
    var_int = float(tau_f[0] ** 2 + sum((tau_f[j] * x_means[j] / x_stds[j]) ** 2 for j in range(1, d)))
    pdfs.append(_sym_pdf(mu_int, float(np.sqrt(var_int)), fam_ffx))

    for j in range(1, d):
        pdfs.append(_sym_pdf(float(nu_f[j] / x_stds[j]), float(tau_f[j] / x_stds[j]), fam_ffx))

    # sigma_rfx
    for j in range(q):
        pdfs.append(_half_pdf(float(tau_r[j]), fam_rfx))

    # sigma_eps
    if result.proposal.has_sigma_eps:  # type: ignore[attr-defined]
        pdfs.append(_half_pdf(tau_e, fam_eps))

    return pdfs


def dictMean(td: dict[str, torch.Tensor]) -> float:
    values = list(td.values())
    if not values:
        return float('nan')
    for i, v in enumerate(values):
        if v.dim() == 0:
            values[i] = v.unsqueeze(0)
    cat = torch.cat(values)
    if cat.numel() == 0 or torch.isnan(cat).all():
        return float('nan')
    return torch.nanmean(cat).item()
