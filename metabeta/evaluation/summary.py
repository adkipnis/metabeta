import torch
from tabulate import tabulate

from metabeta.posthoc.conformal import Calibrator
from metabeta.utils.evaluation import Proposal, EvaluationSummary, dictMean
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
    posteriorPredictiveNLL,
)
from metabeta.evaluation.correlation import evaluateCorrelation, summarizeCorrelation


EST_TYPE = 'mean'


def getSummary(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
    calibrator: Calibrator | None = None,
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

    # rfx correlation
    if proposal.rfx.shape[-1] >= 2:
        corr_results = evaluateCorrelation(proposal.rfx, data)
        out['rfx_corr'] = summarizeCorrelation(corr_results, data['mask_q'].sum(-1))

    # prior predictive fit
    prior_samples = getPriorSamples(data, proposal.n_samples)
    pp_0 = getPosteriorPredictive(prior_samples, data)
    out['prior_nll'] = posteriorPredictiveNLL(pp_0, data)

    # posterior predictive fit
    pp = getPosteriorPredictive(proposal, data)
    out['posterior_nll'] = posteriorPredictiveNLL(pp, data, w=proposal.weights)

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


def summaryTable(s: EvaluationSummary) -> str:
    long_table = longTable(s.corr, s.nrmse, s.ece, s.lcr)
    flat_table = flatTable(s.tpd, s.mnll, s.meff, s.mk, s.rfx_corr)
    return long_table + '\n' + flat_table


def longTable(
    corr: dict[str, torch.Tensor],  # Pearson correlation
    nrmse: dict[str, torch.Tensor],  # normalized root mean square error
    ece: dict[str, torch.Tensor],  # expeced coverage error
    lcr: dict[str, torch.Tensor],  # log coverage ratio
) -> str:
    def process(row: list[str | torch.Tensor]) -> list[str | float]:
        out = []
        for e in row:
            e = torch.mean(e).item() if isinstance(e, torch.Tensor) else e
            out.append(e)
        return out

    keys = ('ffx', 'sigma_rfx', 'sigma_eps', 'rfx')
    names = {
        'ffx': 'FFX',
        'sigma_rfx': 'Sigma(RFX)',
        'sigma_eps': 'Sigma(Eps)',
        'rfx': 'RFX',
    }
    rows = [[names[k], corr[k], nrmse[k], ece[k], lcr[k]] for k in keys]
    rows = [process(row) for row in rows]
    rows += [['Average', dictMean(corr), dictMean(nrmse), dictMean(ece), dictMean(lcr)]]
    results = tabulate(
        rows,
        headers=['', 'R', 'NRMSE', 'ECE', 'LCR'],
        floatfmt='.3f',
        tablefmt='simple',
    )
    return f'\n{results}\n'


def flatTable(
    tpd: float | None = None,
    mnll: float | None = None,
    meff: float | None = None,
    mk: float | None = None,
    rfx_corr: dict[str, float] | None = None,
) -> str:
    rows = []
    results = ''
    if tpd is not None:
        rows += [['Estimation time / ds [s]', tpd]]
    if mnll is not None:
        rows += [['Median ppNLL', mnll]]
    if meff is not None:
        rows += [['Median IS Efficency', meff]]
    if mk is not None:
        rows += [['Median Pareto k', mk]]
    if rfx_corr is not None:
        rows += [['Median RFX Corr MAE', rfx_corr.get('mae_all', float('nan'))]]
        if 'detection_rate' in rfx_corr:
            rows += [['RFX Corr Detection Rate', rfx_corr['detection_rate']]]
        if 'false_positive_rate' in rfx_corr:
            rows += [['RFX Corr FP Rate', rfx_corr['false_positive_rate']]]
    if rows:
        results = tabulate(rows, floatfmt='.3f', tablefmt='simple')
    return f'{results}\n'
