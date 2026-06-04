from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from metabeta.utils import results as _results


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


def subsetProposal(proposal: _results.Proposal, mask: np.ndarray) -> _results.Proposal:
    """Return a new Proposal restricted to datasets selected by a boolean mask."""
    idx = torch.from_numpy(mask)
    new_data = {src: {k: v[idx] for k, v in inner.items()} for src, inner in proposal.data.items()}
    corr = proposal._corr_rfx[idx] if proposal._corr_rfx is not None else None
    sub = _results.Proposal(
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


def makePriorProposal(result: object, n_samples: int = 1000) -> _results.Proposal:
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
    return _results.Proposal(
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
    result: object, n_grid: int = 500, batch_index: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute analytical prior PDFs in original-scale units, one per global parameter.

    Returns (x_grid, density) pairs ordered as: ffx[0..d-1], sigma_rfx[0..q-1], sigma_eps.
    The transform to original scale is applied analytically so the curves are exact
    (no sampling noise), matching the distributions described in posteriorSummary.

    batch_index selects which prior variant to use when prior_params has multiple rows
    (e.g. from a batched-priors sample() call).
    """
    from scipy.stats import norm, halfnorm, t as student_t
    from metabeta.utils.constants import FFX_FAMILIES, SIGMA_FAMILIES, STUDENT_DF

    pp, si = result.prior_params, result.scale_info  # type: ignore[attr-defined]
    sd_y = float(si.y_std)
    d = len(result.param_names['ffx'])  # type: ignore[attr-defined]
    q = len(result.param_names['sigma_rfx'])  # type: ignore[attr-defined]
    bi = batch_index

    tau_f = np.asarray(pp['tau_ffx'][bi, :d], dtype=float) * sd_y
    nu_f = np.asarray(pp['nu_ffx'][bi, :d], dtype=float) * sd_y
    tau_r = np.asarray(pp['tau_rfx'][bi, :q], dtype=float) * sd_y

    fam_ffx_id = int(pp['family_ffx'][bi]) if 'family_ffx' in pp else 0
    fam_rfx_id = int(pp['family_sigma_rfx'][bi]) if 'family_sigma_rfx' in pp else 0
    fam_ffx = FFX_FAMILIES[fam_ffx_id] if fam_ffx_id < len(FFX_FAMILIES) else 'normal'
    fam_rfx = SIGMA_FAMILIES[fam_rfx_id] if fam_rfx_id < len(SIGMA_FAMILIES) else 'halfnormal'

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
    var_int = float(
        tau_f[0] ** 2 + sum((tau_f[j] * x_means[j] / x_stds[j]) ** 2 for j in range(1, d))
    )
    pdfs.append(_sym_pdf(mu_int, float(np.sqrt(var_int)), fam_ffx))

    for j in range(1, d):
        pdfs.append(_sym_pdf(float(nu_f[j] / x_stds[j]), float(tau_f[j] / x_stds[j]), fam_ffx))

    # sigma_rfx
    for j in range(q):
        pdfs.append(_half_pdf(float(tau_r[j]), fam_rfx))

    # sigma_eps
    if result.proposal.has_sigma_eps:  # type: ignore[attr-defined]
        tau_e = float(np.asarray(pp['tau_eps']).ravel()[bi]) * sd_y
        fam_eps_id = (
            int(np.asarray(pp['family_sigma_eps']).ravel()[bi]) if 'family_sigma_eps' in pp else 0
        )
        fam_eps = SIGMA_FAMILIES[fam_eps_id] if fam_eps_id < len(SIGMA_FAMILIES) else 'halfstudent'
        pdfs.append(_half_pdf(tau_e, fam_eps))

    # LKJ marginals for correlation parameters
    # Under LKJ(eta), each off-diagonal rho has marginal (1+rho)/2 ~ Beta(alpha, alpha)
    # where alpha = eta + (q-2)/2.  Jacobian of rho = 2x-1 gives p(rho) = p_Beta(x) / 2.
    d_corr = result.proposal.d_corr  # type: ignore[attr-defined]
    if d_corr > 0 and 'eta_rfx' in pp:
        from scipy.stats import beta as beta_dist

        eta = float(np.asarray(pp['eta_rfx']).ravel()[bi])
        alpha = eta + (q - 2) / 2
        xg = np.linspace(-1.0, 1.0, n_grid)
        pdf_lkj = beta_dist.pdf((1 + xg) / 2, alpha, alpha) / 2
        for _ in range(d_corr):
            pdfs.append((xg, pdf_lkj))

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
