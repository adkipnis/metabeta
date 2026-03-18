import torch
from tabulate import tabulate

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
from metabeta.plot.predictive import plotPPC, plotPPD


EST_TYPE = 'mean'


def getSummary(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
) -> EvaluationSummary:
    out = {}

    # point-based stats
    out['estimates'] = est = getPointEstimates(proposal, EST_TYPE)
    out['nrmse'] = getRMSE(est, data, normalize=True)
    out['corr'] = getCorrelation(est, data)

    # inteval-based stats
    out['credible_intervals'] = ci_dicts = getCredibleIntervals(proposal)
    out['coverage'] = cvrg_dicts = getCoverages(ci_dicts, data)
    out['coverage_error'] = getCoverageErrors(cvrg_dicts, log_ratio=False)
    out['log_coverage_ratio'] = getCoverageErrors(cvrg_dicts, log_ratio=True)

    # prior predictive fit
    prior_samples = getPriorSamples(data, proposal.n_samples)
    pp_0 = getPosteriorPredictive(prior_samples, data)
    out['prior_nll'] = posteriorPredictiveNLL(pp_0, data)

    # posterior predictive fit
    pp = getPosteriorPredictive(proposal, data)
    out['posterior_nll'] = posteriorPredictiveNLL(pp, data, w=proposal.weights)

    plotPPC(pp, data)
    plotPPD(pp, data, pp_prior=pp_0)

    # importance sampling
    out['sample_efficiency'] = proposal.efficiency
    out['pareto_k'] = proposal.pareto_k

    summary = EvaluationSummary(**out)
    return summary


def summaryTable(s: EvaluationSummary) -> str:
    long_table = longTable(s.corr, s.nrmse, s.ece, s.lcr)
    flat_table = flatTable(s.tpd, s.mnll, s.meff, s.mk)
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
    if rows:
        results = tabulate(rows, floatfmt='.3f', tablefmt='simple')
    return f'{results}\n'
