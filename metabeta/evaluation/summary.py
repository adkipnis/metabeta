import time
import logging
import torch
from tabulate import tabulate

from metabeta.posthoc.conformal import Calibrator
from metabeta.utils.evaluation import Proposal, EvaluationSummary, dictMean

logger = logging.getLogger(__name__)

# Parameters excluded from the Average row (structural, not estimable like free params).
_AVG_EXCLUDE = frozenset({'corr_rfx'})


def _dictMeanExcl(td: dict) -> float:
    return dictMean({k: v for k, v in td.items() if k not in _AVG_EXCLUDE})


def _t(label: str, t0: float) -> float:
    """Log elapsed time since t0 and return current time."""
    logger.debug('  %-30s %.2fs', label, time.perf_counter() - t0)
    return time.perf_counter()


from metabeta.evaluation.point import (
    getPointEstimates,
    getRMSE,
    getCorrelation,
)
from metabeta.evaluation.intervals import (
    getCoverageErrors,
    getCoverages,
    getCredibleIntervals,
)
from metabeta.evaluation.predictive import (
    getPriorSamples,
    getPosteriorPredictive,
    posteriorPredictiveAUC,
    posteriorPredictiveDeviance,
    posteriorPredictiveNLL,
    posteriorPredictiveR2,
    psisLooNLL,
)



EST_TYPE = 'mean'


def _sliceData(
    data: dict[str, torch.Tensor], start: int, end: int
) -> dict[str, torch.Tensor]:
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

    # parameter recovery
    out['estimates'] = est = getPointEstimates(proposal, EST_TYPE)
    out['nrmse'] = getRMSE(est, data, normalize=True)
    out['corr'] = getCorrelation(est, data)
    t0 = _t('point estimates', t0)

    # coverage and calibration
    ci_dicts = getCredibleIntervals(proposal)
    t0 = _t('credible intervals (quantiles)', t0)
    if calibrator is not None:
        ci_dicts = calibrator.apply(ci_dicts)
    out['credible_intervals'] = ci_dicts
    out['coverage'] = cvrg_dicts = getCoverages(ci_dicts, data)
    out['coverage_error'] = getCoverageErrors(cvrg_dicts, log_ratio=False)
    out['log_coverage_ratio'] = getCoverageErrors(cvrg_dicts, log_ratio=True)
    t0 = _t('coverage errors', t0)

    # prior predictive fit (optional — skipped by default, costs two (b,m,n,s) computations)
    if compute_prior:
        prior_samples = getPriorSamples(data, proposal.n_samples, likelihood_family)
        pp_0 = getPosteriorPredictive(prior_samples, data, likelihood_family)
        out['prior_nll'] = posteriorPredictiveNLL(pp_0, data)
        t0 = _t('prior NLL', t0)

    # Posterior predictive quantities, chunked over datasets to bound peak memory.
    # Each (b, m, n, s) tensor is O(b * m * n * s); for MCMC s≈4000 this can be several GB.
    # Processing dataset_chunk_size datasets at a time keeps the working set manageable.
    b = proposal.samples_g.shape[0]
    chunk = min(dataset_chunk_size, b)
    logger.debug('  %-30s b=%d chunk=%d', 'predictive (chunked)', b, chunk)
    posterior_nlls, loo_nlls, loo_ks, pp_fits = [], [], [], []
    t_pp = t_logp = t_nll = t_loo = t_fit = 0.0

    for start in range(0, b, chunk):
        end = min(start + chunk, b)
        p_c = proposal.slice_b(start, end)
        d_c = _sliceData(data, start, end)

        _t0 = time.perf_counter()
        pp_c = getPosteriorPredictive(p_c, d_c, likelihood_family)
        _t1 = time.perf_counter(); t_pp += _t1 - _t0

        log_p_c = pp_c.log_prob(d_c['y'].unsqueeze(-1))   # (chunk, m, n, s)
        _t2 = time.perf_counter(); t_logp += _t2 - _t1

        posterior_nlls.append(
            posteriorPredictiveNLL(pp_c, d_c, w=p_c.weights, log_p=log_p_c)
        )
        _t3 = time.perf_counter(); t_nll += _t3 - _t2

        loo_nll_c, loo_k_c = psisLooNLL(
            pp_c, d_c, w=p_c.weights, reff=p_c.reff, log_p=log_p_c
        )
        del log_p_c   # free (chunk, m, n, s) before next chunk
        _t4 = time.perf_counter(); t_loo += _t4 - _t3
        loo_nlls.append(loo_nll_c)
        loo_ks.append(loo_k_c)

        if likelihood_family == 0:   # normal
            pp_fits.append(posteriorPredictiveR2(pp_c, d_c, w=p_c.weights))
        elif likelihood_family == 1:   # bernoulli
            pp_fits.append(posteriorPredictiveAUC(pp_c, d_c, w=p_c.weights))
        elif likelihood_family == 2:   # poisson
            pp_fits.append(posteriorPredictiveDeviance(pp_c, d_c, w=p_c.weights))
        _t5 = time.perf_counter(); t_fit += _t5 - _t4

    logger.debug('  %-30s %.2fs', 'linear predictor (eta)', t_pp)
    logger.debug('  %-30s %.2fs', 'log_prob', t_logp)
    logger.debug('  %-30s %.2fs', 'posterior NLL', t_nll)
    logger.debug('  %-30s %.2fs', 'PSIS-LOO NLL', t_loo)
    logger.debug('  %-30s %.2fs', 'pp fit (R²/AUC/deviance)', t_fit)

    out['posterior_nll'] = torch.cat(posterior_nlls)
    out['loo_nll'] = torch.cat(loo_nlls)
    out['loo_pareto_k'] = torch.cat(loo_ks)
    out['pp_fit'] = torch.cat(pp_fits) if pp_fits else None

    # visualize single datasets
    # gt = torch.cat([data['ffx'][0], data['sigma_rfx'][0], data['sigma_eps'][0:1]]).numpy()
    # plotParameters(proposal, index=0, truth=gt, prior=prior_samples)
    # plotPPC(pp, data)
    # plotPPD(pp, data, pp_prior=pp_0)

    # importance sampling
    out['sample_efficiency'] = proposal.efficiency
    out['pareto_k'] = proposal.pareto_k

    # time per dataset
    out['tpd'] = proposal.tpd

    summary = EvaluationSummary(**out)
    return summary


def summaryTable(s: EvaluationSummary, likelihood_family: int = 0) -> str:
    long_table = longTable(s.corr, s.nrmse, s.ece, s.eace)
    fit_labels = {0: 'Median pp R²', 1: 'Median pp AUC', 2: 'Median pp Deviance'}
    fit_label = fit_labels.get(likelihood_family, 'Median pp R²')
    flat_table = flatTable(s.tpd, s.mloonll, s.mfit, fit_label, s.meff, s.mk, s.mloo_k)
    return long_table + '\n' + flat_table


def longTable(
    corr: dict[str, torch.Tensor],  # Pearson correlation
    nrmse: dict[str, torch.Tensor],  # normalized root mean square error
    ece: dict[str, torch.Tensor],  # expected coverage error (signed)
    eace: dict[str, torch.Tensor],  # absolute ECE (abs per dataset, then avg)
) -> str:
    def process(row: list[str | torch.Tensor]) -> list[str | float]:
        out = []
        for e in row:
            e = torch.mean(e).item() if isinstance(e, torch.Tensor) else e
            out.append(e)
        return out

    names = {
        'ffx': 'FFX',
        'sigma_rfx': 'Sigma(RFX)',
        'sigma_eps': 'Sigma(Eps)',
        'corr_rfx': 'Corr(RFX)',
        'rfx': 'RFX',
    }
    keys = [k for k in names if k in corr]
    rows = [[names[k], corr[k], nrmse[k], ece.get(k), eace.get(k)] for k in keys]
    rows = [process(row) for row in rows]
    # Average row: exclude corr_rfx from all averages.
    rows += [['Average', _dictMeanExcl(corr), _dictMeanExcl(nrmse),
              _dictMeanExcl(ece), _dictMeanExcl(eace)]]
    results = tabulate(
        rows,
        headers=['', 'R', 'NRMSE', 'ECE', 'EACE'],
        floatfmt='.3f',
        tablefmt='simple',
        missingval='-',
    )
    return f'\n{results}\n'


def flatTable(
    tpd: float | None = None,
    mloonll: float | None = None,
    mfit: float | None = None,
    fit_label: str = 'Median pp R²',
    meff: float | None = None,
    mk: float | None = None,
    mloo_k: float | None = None,
) -> str:
    rows = []
    results = ''
    if tpd is not None:
        rows += [['Estimation time / ds [s]', tpd]]
    # in-sample ppNLL omitted from table (optimistic; use LOO NLL instead)
    if mloonll is not None:
        rows += [['Median LOO-NLL', mloonll]]
    if mfit is not None:
        rows += [[fit_label, mfit]]
    if meff is not None:
        rows += [['Median IS Efficiency', meff]]
    if mk is not None:
        rows += [['Median IS Pareto k', mk]]
    if mloo_k is not None:
        rows += [['Median LOO Pareto k', mloo_k]]
    if rows:
        results = tabulate(rows, floatfmt='.3f', tablefmt='simple')
    return f'{results}\n'
