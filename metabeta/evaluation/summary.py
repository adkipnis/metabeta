from pathlib import Path
import torch
from torch.nn import functional as F
from tabulate import tabulate
import numpy as np

from metabeta.utils.evaluation import Proposal, getNames, joinSigmas
from metabeta.evaluation.moments import (
    sampleLoc,
    sampleRMSE,
    sampleCorrelation,
)
from metabeta.evaluation.intervals import expectedCoverageError
from metabeta.evaluation.predictive import (
    getPosteriorPredictive,
    posteriorPredictiveNLL,
)
from metabeta.plot import plot
from matplotlib import pyplot as plt


def dictMean(data: dict[str, float]) -> float:
    values = list(data.values())
    return float(np.mean(values))


def recoveryPlot(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
    plot_dir: Path,
    epoch: int | None = None,
) -> None:
    targets = []
    estimates = []
    masks = []
    names = []
    metrics = []

    # prepare stats
    stats = {}
    locs = sampleLoc(proposal, 'mean')
    stats['corr'] = sampleCorrelation(locs, data)
    stats['rmse'] = sampleRMSE(locs, data)

    # fixed effects
    d = data['ffx'].shape[-1]
    targets.append(data['ffx'])
    estimates.append(locs['ffx'])
    masks.append(data['mask_d'])
    names.append(getNames('ffx', d))
    metrics.append(
        {
            'r': stats['corr']['ffx'],
            'RMSE': stats['rmse']['ffx'],
        }
    )

    # variance parameters
    q = data['sigma_rfx'].shape[-1]
    targets.append(joinSigmas(data))
    estimates.append(joinSigmas(locs))
    masks.append(F.pad(data['mask_q'], (0, 1), value=True))
    names.append(getNames('sigmas', q))
    rmse = q / (q + 1) * stats['rmse']['sigma_rfx'] + 1 / (q + 1) * stats['rmse']['sigma_eps']
    r = q / (q + 1) * stats['corr']['sigma_rfx'] + 1 / (q + 1) * stats['corr']['sigma_eps']
    metrics.append({'r': r, 'RMSE': rmse})

    # random effects
    targets.append(data['rfx'].view(-1, q))
    estimates.append(locs['rfx'].view(-1, q))
    mask = data['mask_mq']
    masks.append(mask.view(-1, q))
    names.append(getNames('rfx', q))
    metrics.append(
        {
            'r': stats['corr']['rfx'],
            'RMSE': stats['rmse']['rfx'],
        }
    )

    # figure
    fig, axs = plt.subplots(figsize=(6 * 3, 6), ncols=3, dpi=300)
    plot.groupedRecovery(
        axs,
        targets=targets,
        estimates=estimates,
        masks=masks,
        metrics=metrics,
        names=names,
        titles=['Fixed Effects', 'Variances', 'Random Effects'],
    )
    for ax in axs.flat:
        ax.set_box_aspect(1)
    fig.tight_layout()

    # store
    fname = plot_dir / 'parameter_recovery_latest.png'
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.15)
    if epoch is not None:
        fname_e = plot_dir / f'parameter_recovery_e{epoch}.png'
        plt.savefig(fname_e, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def dependentSummary(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
) -> str:
    # moment-based stats
    sample_loc = sampleLoc(proposal, 'mean')
    rmse = sampleRMSE(sample_loc, data)
    corr = sampleCorrelation(sample_loc, data)

    # inteval-based stats
    mce = expectedCoverageError(proposal, data)

    # # recovery plot
    # if plot_this:
    #     recoveryPlot(sample_loc, data, stats=dict(rmse=rmse, corr=corr), plot_dir)

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
    proposal: Proposal,
    data: dict[str, torch.Tensor],
    time: float,
    is_eff: float | None = None,
) -> str:
    # posterior predictive
    pp = getPosteriorPredictive(proposal, data)
    nll = posteriorPredictiveNLL(pp, data)
    mnll = nll.mean(-1).median().item()

    # flat table
    rows = [
        ['Median NLL', mnll],
        ['time / ds [s]', time],
    ]
    if is_eff is not None:
        rows += [['IS Efficency', is_eff]]
    results = tabulate(rows, floatfmt='.3f', tablefmt='simple')
    return f'{results}\n'
