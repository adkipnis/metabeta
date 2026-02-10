import torch
from tabulate import tabulate
import numpy as np

from metabeta.utils.evaluation import Proposed
from metabeta.evaluation.moments import (
    sampleLoc,
    sampleRMSE,
    sampleCorrelation,
)
from metabeta.evaluation.intervals import expectedCoverageError
from metabeta.evaluation.predictive import (
    posteriorPredictiveNLL,
    samplePosteriorPredictive,
)


def dictMean(data: dict[str, float]) -> float:
    values = list(data.values())
    return float(np.mean(values))


def dependentSummary(
    proposed: Proposed,
    data: dict[str, torch.Tensor],
) -> str:
    # moment-based stats
    sample_loc = sampleLoc(proposed, 'median')
    rmse = sampleRMSE(sample_loc, data)
    corr = sampleCorrelation(sample_loc, data)

    # inteval-based stats
    mce = expectedCoverageError(proposed, data)

    # print summary
    return longTable(corr, rmse, mce)


def longTable(
    corr: dict[str, float],  # Pearson correlation
    rmse: dict[str, float],  # root mean square error
    mce: dict[str, float],  # mean coverage error
) -> str:
    keys = ('ffx', 'sigma_rfx', 'sigma_eps', 'rfx')
    names = {
        'ffx': 'FFX',
        'sigma_rfx': 'Sigma(RFX)',
        'sigma_eps': 'Sigma(Eps)',
        'rfx': 'RFX',
    }
    rows = [[names[k], corr[k], rmse[k], mce[k]] for k in keys]
    rows += [['Average', dictMean(corr), dictMean(rmse), dictMean(mce)]]
    results = tabulate(
        rows,
        headers=['', 'R', 'RMSE', 'MCE'],
        floatfmt='.3f',
        tablefmt='simple',
    )
    return f'\n{results}\n'


def wideTable(
    corr: dict[str, float],  # Pearson correlation
    rmse: dict[str, float],  # root mean square error
    mce: dict[str, float],  # mean coverage error
) -> str:
    keys = ('ffx', 'sigma_rfx', 'sigma_eps', 'rfx')

    rows = [
        ['R'] + [corr[k] for k in keys] + [dictMean(corr)],
        ['RMSE'] + [rmse[k] for k in keys] + [dictMean(rmse)],
        ['MCE'] + [mce[k] for k in keys] + [dictMean(mce)],
    ]
    results = tabulate(
        rows,
        headers=['', 'FFX', 'Sigma(RFX)', 'Sigma(Eps)', 'RFX', 'All'],
        floatfmt='.3f',
        tablefmt='simple',
    )
    return f'\n{results}'


def flatSummary(
    proposed: Proposed,
    data: dict[str, torch.Tensor],
    time: float,
) -> str:
    # posterior predictive
    pp = samplePosteriorPredictive(proposed, data)
    nll = posteriorPredictiveNLL(pp, data)
    mnll = nll.mean(-1).median().item()

    # flat table
    rows = [
        ['Median NLL', mnll],
        ['time [s]', time],
    ]
    results = tabulate(rows, floatfmt='.3f', tablefmt='simple')
    return f'{results}\n'
