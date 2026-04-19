import torch
from tabulate import tabulate

from metabeta.posthoc.conformal import Calibrator
from metabeta.utils.evaluation import Proposal, EvaluationSummary, dictMean

# Parameters excluded from the Average row (structural, not estimable like free params).
_AVG_EXCLUDE = frozenset({'corr_rfx'})


def _dictMeanExcl(td: dict) -> float:
    return dictMean({k: v for k, v in td.items() if k not in _AVG_EXCLUDE})
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


def getSummary(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
    calibrator: Calibrator | None = None,
    likelihood_family: int = 0,
) -> EvaluationSummary:
    out = {}

    # paramter recovery
    out['estimates'] = est = getPointEstimates(proposal, EST_TYPE)
    out['nrmse'] = getRMSE(est, data, normalize=True)
    out['corr'] = getCorrelation(est, data)

    # coverage and calibration
    ci_dicts = getCredibleIntervals(proposal)
    if calibrator is not None:
        ci_dicts = calibrator.apply(ci_dicts)
    out['credible_intervals'] = ci_dicts
    out['coverage'] = cvrg_dicts = getCoverages(ci_dicts, data)
    out['coverage_error'] = getCoverageErrors(cvrg_dicts, log_ratio=False)
    out['log_coverage_ratio'] = getCoverageErrors(cvrg_dicts, log_ratio=True)

    # prior predictive fit
    prior_samples = getPriorSamples(data, proposal.n_samples, likelihood_family)
    pp_0 = getPosteriorPredictive(prior_samples, data, likelihood_family)
    out['prior_nll'] = posteriorPredictiveNLL(pp_0, data)

    # posterior predictive fit
    pp = getPosteriorPredictive(proposal, data, likelihood_family)
    out['posterior_nll'] = posteriorPredictiveNLL(pp, data, w=proposal.weights)  # in-sample

    # PSIS-LOO NLL: automatically uses loo_is when IS weights are available (w=proposal.weights),
    # falling back to loo_raw (w=None) when no IS was done.
    out['loo_nll'], out['loo_pareto_k'] = psisLooNLL(pp, data, w=proposal.weights, reff=proposal.reff)
    if likelihood_family == 0:  # normal
        out['pp_fit'] = posteriorPredictiveR2(pp, data, w=proposal.weights)
    elif likelihood_family == 1:  # bernoulli
        out['pp_fit'] = posteriorPredictiveAUC(pp, data, w=proposal.weights)
    elif likelihood_family == 2:  # poisson
        out['pp_fit'] = posteriorPredictiveDeviance(pp, data, w=proposal.weights)

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
