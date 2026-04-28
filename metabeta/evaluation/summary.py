import time
import logging
import numpy as np
import torch
from tabulate import tabulate

from metabeta.posthoc.conformal import Calibrator
from metabeta.utils.evaluation import Proposal, EvaluationSummary, dictMean
from metabeta.evaluation.point import getPointEstimates, getRMSE, getCorrelation
from metabeta.evaluation.intervals import (
    ALPHAS,
    getCoverageErrors,
    getCoverages,
    getCredibleIntervals,
)
from metabeta.evaluation.predictive import (
    getPriorSamples,
    getPosteriorPredictive,
    posteriorPredictiveAUC,
    posteriorPredictiveCoverage,
    posteriorPredictiveDeviance,
    posteriorPredictiveNLL,
    posteriorPredictiveR2,
    psisLooNLL,
)

logger = logging.getLogger(__name__)

# Parameters excluded from the Average row (structural, not estimable like free params).
_AVG_EXCLUDE = frozenset({'corr_rfx'})
EST_TYPE = 'mean'


def _dictMeanExcl(td: dict) -> float:
    return dictMean({k: v for k, v in td.items() if k not in _AVG_EXCLUDE})


def _t(label: str, t0: float) -> float:
    """Log elapsed time since t0 and return current time."""
    logger.debug('  %-30s %.2fs', label, time.perf_counter() - t0)
    return time.perf_counter()


def _sliceData(data: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    return {k: v[start:end] if torch.is_tensor(v) else v for k, v in data.items()}


def getSummary(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
    calibrator: Calibrator | None = None,
    likelihood_family: int = 0,
    compute_prior: bool = False,
    dataset_chunk_size: int = 16,
) -> EvaluationSummary:
    out = {}
    t0 = time.perf_counter()
    logger.debug('getSummary: s=%d', proposal.n_samples)

    out['estimates'] = est = getPointEstimates(proposal, EST_TYPE)
    out['nrmse'] = getRMSE(est, data, normalize=True)
    out['corr'] = getCorrelation(est, data)
    t0 = _t('point estimates', t0)

    ci_dicts = getCredibleIntervals(proposal)
    t0 = _t('credible intervals (quantiles)', t0)
    if calibrator is not None:
        ci_dicts = calibrator.apply(ci_dicts)
    out['credible_intervals'] = ci_dicts
    out['coverage'] = cvrg_dicts = getCoverages(ci_dicts, data)
    out['coverage_error'] = getCoverageErrors(cvrg_dicts, log_ratio=False)
    out['log_coverage_ratio'] = getCoverageErrors(cvrg_dicts, log_ratio=True)
    t0 = _t('coverage errors', t0)
    out['rfx_joint_ece'], out['rfx_joint_eace'] = _rfxJointCalibration(proposal, data)

    # prior predictive fit (optional — costs two (b,m,n,s) materializations)
    if compute_prior:
        prior_samples = getPriorSamples(data, proposal.n_samples, likelihood_family)
        pp_0 = getPosteriorPredictive(prior_samples, data, likelihood_family)
        out['prior_nll'] = posteriorPredictiveNLL(pp_0, data)
        t0 = _t('prior NLL', t0)

    # Posterior predictive quantities, chunked over datasets to bound peak memory.
    # (b, m, n, s) tensors can be several GB for MCMC (s≈4000); process chunk datasets at a time.
    b = proposal.samples_g.shape[0]
    chunk = min(dataset_chunk_size, b)
    logger.debug('  %-30s b=%d chunk=%d', 'predictive (chunked)', b, chunk)
    posterior_nlls, loo_nlls, loo_ks, pp_fits = [], [], [], []
    pp_covs, pp_widths = [], []
    t_pp = t_logp = t_nll = t_loo = t_fit = t_cov = 0.0

    for start in range(0, b, chunk):
        end = min(start + chunk, b)
        p_c = proposal.slice_b(start, end)
        d_c = _sliceData(data, start, end)

        tc = time.perf_counter()
        pp_c = getPosteriorPredictive(p_c, d_c, likelihood_family)
        t_pp += time.perf_counter() - tc

        tc = time.perf_counter()
        log_p_c = pp_c.log_prob(d_c['y'].unsqueeze(-1))   # (chunk, m, n, s)
        t_logp += time.perf_counter() - tc

        tc = time.perf_counter()
        posterior_nlls.append(posteriorPredictiveNLL(pp_c, d_c, w=p_c.weights, log_p=log_p_c))
        t_nll += time.perf_counter() - tc

        tc = time.perf_counter()
        loo_nll_c, loo_k_c = psisLooNLL(pp_c, d_c, w=p_c.weights, reff=p_c.reff, log_p=log_p_c)
        del log_p_c   # free before next chunk
        t_loo += time.perf_counter() - tc
        loo_nlls.append(loo_nll_c)
        loo_ks.append(loo_k_c)

        tc = time.perf_counter()
        if likelihood_family == 0:
            pp_fits.append(posteriorPredictiveR2(pp_c, d_c, w=p_c.weights))
        elif likelihood_family == 1:
            pp_fits.append(posteriorPredictiveAUC(pp_c, d_c, w=p_c.weights))
        elif likelihood_family == 2:
            pp_fits.append(posteriorPredictiveDeviance(pp_c, d_c, w=p_c.weights))
        t_fit += time.perf_counter() - tc

        tc = time.perf_counter()
        cov_c, wid_c = posteriorPredictiveCoverage(pp_c, d_c, w=p_c.weights)
        pp_covs.append(cov_c)
        pp_widths.append(wid_c)
        t_cov += time.perf_counter() - tc

    logger.debug('  %-30s %.2fs', 'linear predictor (eta)', t_pp)
    logger.debug('  %-30s %.2fs', 'log_prob', t_logp)
    logger.debug('  %-30s %.2fs', 'posterior NLL', t_nll)
    logger.debug('  %-30s %.2fs', 'PSIS-LOO NLL', t_loo)
    logger.debug('  %-30s %.2fs', 'pp fit (R²/AUC/deviance)', t_fit)
    logger.debug('  %-30s %.2fs', 'predictive coverage/width', t_cov)

    out['posterior_nll'] = torch.cat(posterior_nlls)
    out['loo_nll'] = torch.cat(loo_nlls)
    out['loo_pareto_k'] = torch.cat(loo_ks)
    out['pp_fit'] = torch.cat(pp_fits) if pp_fits else None
    out['pp_cov_coverage'] = torch.cat(pp_covs, dim=-1)   # (n_alphas, b)
    out['pp_cov_width'] = torch.cat(pp_widths, dim=-1)     # (n_alphas, b)
    out['sample_efficiency'] = proposal.efficiency
    out['pareto_k'] = proposal.pareto_k
    out['tpd'] = proposal.tpd

    return EvaluationSummary(**out)


def _rfxJointCalibration(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
) -> tuple[float | None, float | None]:
    """Joint calibration for random effects via Mahalanobis fractional ranks."""
    rfx_samples = proposal.rfx
    if rfx_samples.dim() != 4:
        return None, None

    mask_q = data['mask_q']
    mask_m = data['mask_m']
    truth = data['rfx']
    weights = proposal.weights
    n_samples = rfx_samples.shape[-2]

    ranks: list[float] = []
    eye_cache: dict[int, torch.Tensor] = {}

    for i in range(rfx_samples.shape[0]):
        q_i = int(mask_q[i].sum().item())
        if q_i < 2:
            continue

        if q_i not in eye_cache:
            eye_cache[q_i] = torch.eye(q_i, device=rfx_samples.device, dtype=rfx_samples.dtype)
        eye = eye_cache[q_i]

        w_i = None
        if weights is not None:
            w_i = weights[i]
            w_i = w_i / w_i.sum().clamp_min(1e-12)

        for j in range(rfx_samples.shape[1]):
            if not bool(mask_m[i, j]):
                continue

            samp = rfx_samples[i, j, :, :q_i]  # (s, q_i)
            mu = samp.mean(0)
            centered = samp - mu
            if n_samples > 1:
                cov = centered.mT @ centered / (n_samples - 1)
            else:
                cov = centered.mT @ centered
            cov = cov + 1e-10 * eye

            try:
                cov_inv = torch.linalg.pinv(cov)
            except RuntimeError:
                continue

            d2_samples = (centered @ cov_inv * centered).sum(-1)
            delta = truth[i, j, :q_i] - mu
            d2_truth = (delta @ cov_inv @ delta).item()
            inside = (d2_samples <= d2_truth).float()
            if w_i is None:
                rank = float(inside.mean().item())
            else:
                rank = float((inside * w_i).sum().item())
            if np.isfinite(rank):
                ranks.append(rank)

    if not ranks:
        return None, None

    r = np.asarray(ranks, dtype=np.float64)
    ece = 0.0
    eace = 0.0
    for alpha in ALPHAS:
        nominal = 1.0 - alpha
        covered = (np.abs(r - 0.5) <= nominal / 2.0).astype(np.float64)
        err = float(covered.mean() - nominal)
        ece += err
        eace += abs(err)
    return ece / len(ALPHAS), eace / len(ALPHAS)


def summaryTable(s: EvaluationSummary, likelihood_family: int = 0) -> str:
    long_table = longTable(s.corr, s.nrmse, s.ece, s.eace)
    fit_labels = {0: 'Median pp R²', 1: 'Median pp AUC', 2: 'Median pp Deviance'}
    fit_label = fit_labels.get(likelihood_family, 'Median pp R²')
    flat_table = flatTable(
        s.tpd,
        s.mloonll,
        s.mfit,
        fit_label,
        s.meff,
        s.mk,
        s.mloo_k,
        s.rfx_joint_ece,
        s.rfx_joint_eace,
        s.pp_eace,
        s.pp_width_90,
    )
    return long_table + '\n' + flat_table


def longTable(
    corr: dict[str, torch.Tensor],
    nrmse: dict[str, torch.Tensor],
    ece: dict[str, torch.Tensor],
    eace: dict[str, torch.Tensor],
) -> str:
    names = {
        'ffx': 'FFX',
        'sigma_rfx': 'Sigma(RFX)',
        'sigma_eps': 'Sigma(Eps)',
        'corr_rfx': 'Corr(RFX)',
        'rfx': 'RFX',
    }

    def to_float(v: str | torch.Tensor) -> str | float:
        return torch.mean(v).item() if isinstance(v, torch.Tensor) else v

    keys = [k for k in names if k in corr]
    rows = [
        [to_float(x) for x in [names[k], corr[k], nrmse[k], ece.get(k), eace.get(k)]] for k in keys
    ]
    rows.append(
        [
            'Average',
            _dictMeanExcl(corr),
            _dictMeanExcl(nrmse),
            _dictMeanExcl(ece),
            _dictMeanExcl(eace),
        ]
    )
    return (
        '\n'
        + tabulate(
            rows,
            headers=['', 'R', 'NRMSE', 'ECE', 'EACE'],
            floatfmt='.3f',
            tablefmt='simple',
            missingval='-',
        )
        + '\n'
    )


def flatTable(
    tpd: float | None = None,
    mloonll: float | None = None,
    mfit: float | None = None,
    fit_label: str = 'Median pp R²',
    meff: float | None = None,
    mk: float | None = None,
    mloo_k: float | None = None,
    rfx_joint_ece: float | None = None,
    rfx_joint_eace: float | None = None,
    pp_eace: float | None = None,
    pp_width_90: float | None = None,
) -> str:
    # in-sample ppNLL omitted (optimistic; use LOO NLL instead)
    rows = [
        row
        for row in [
            ['Estimation time / ds [s]', tpd],
            ['RFX Joint ECE', rfx_joint_ece],
            ['RFX Joint EACE', rfx_joint_eace],
            ['Pred. EACE', pp_eace],
            ['Median 90% pred. width', pp_width_90],
            ['Median LOO-NLL', mloonll],
            ['Median LOO Pareto k', mloo_k],
            [fit_label, mfit],
        ]
        if row[1] is not None
    ]
    result = tabulate(rows, floatfmt='.3f', tablefmt='simple')
    return f'{result}\n'
