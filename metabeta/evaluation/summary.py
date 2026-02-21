from pathlib import Path
import torch
from torch.nn import functional as F
from tabulate import tabulate
import numpy as np

from metabeta.utils.evaluation import Proposal, getNames, joinSigmas
from metabeta.evaluation.point import (
    getPointEstimates,
    getRMSE,
    getCorrelation,
)
from metabeta.evaluation.intervals import analyzeCoverage
from metabeta.evaluation.predictive import (
    getPosteriorPredictive,
    posteriorPredictiveNLL,
)
from metabeta.plot import plot
from matplotlib import pyplot as plt


EST_TYPE = 'mean'

def tensorMean(data: dict[str, torch.Tensor]) -> dict[str, float]:
    out = {}
    for k, v in data.items():
        out[k] = v.mean().item()
    return out


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
    est = getPointEstimates(proposal, EST_TYPE)
    stats['corr'] = getCorrelation(est, data)
    stats['nrmse'] = getRMSE(est, data, normalize=True)

    # fixed effects
    d = data['ffx'].shape[-1]
    targets.append(data['ffx'])
    estimates.append(est['ffx'])
    masks.append(data['mask_d'])
    names.append(getNames('ffx', d))
    metrics.append(
        {
            'r': stats['corr']['ffx'],
            'NRMSE': stats['nrmse']['ffx'],
        }
    )

    # variance parameters
    q = data['sigma_rfx'].shape[-1]
    targets.append(joinSigmas(data))
    estimates.append(joinSigmas(est))
    masks.append(F.pad(data['mask_q'], (0, 1), value=True))
    names.append(getNames('sigmas', q))
    nrmse = q / (q + 1) * stats['nrmse']['sigma_rfx'] + 1 / (q + 1) * stats['nrmse']['sigma_eps']
    r = q / (q + 1) * stats['corr']['sigma_rfx'] + 1 / (q + 1) * stats['corr']['sigma_eps']
    metrics.append({'r': r, 'NRMSE': nrmse})

    # random effects
    targets.append(data['rfx'].view(-1, q))
    estimates.append(est['rfx'].view(-1, q))
    mask = data['mask_mq']
    masks.append(mask.view(-1, q))
    names.append(getNames('rfx', q))
    metrics.append(
        {
            'r': stats['corr']['rfx'],
            'NRMSE': stats['nrmse']['rfx'],
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
    plt.show()
    plt.close(fig)


def dependentSummary(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
) -> str:
    # point-based stats
    est = getPointEstimates(proposal, EST_TYPE)
    nrmse = getRMSE(est, data, normalize=True)
    corr = getCorrelation(est, data)

    # inteval-based stats
    cvge_results = analyzeCoverage(proposal, data)
    ece = tensorMean(cvge_results['error']['mean'])
    lcr = tensorMean(cvge_results['log_ratio']['mean'])

    # print summary
    return longTable(corr, nrmse, lcr)


def longTable(
    corr: dict[str, float],  # Pearson correlation
    nrmse: dict[str, float],  # normalized root mean square error
    lcr: dict[str, float],  # log coverage ratio
) -> str:
    keys = ('ffx', 'sigma_rfx', 'sigma_eps', 'rfx')
    names = {
        'ffx': 'FFX',
        'sigma_rfx': 'Sigma(RFX)',
        'sigma_eps': 'Sigma(Eps)',
        'rfx': 'RFX',
    }
    rows = [[names[k], corr[k], nrmse[k], lcr[k]] for k in keys]
    rows += [['Average', dictMean(corr), dictMean(nrmse), dictMean(lcr)]]
    results = tabulate(
        rows,
        headers=['', 'R', 'NRMSE', 'LCR'],
        floatfmt='.3f',
        tablefmt='simple',
    )
    return f'\n{results}\n'


def flatSummary(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
    tpd: float,
) -> str:
    # posterior predictive
    pp = getPosteriorPredictive(proposal, data)
    nll = posteriorPredictiveNLL(pp, data, w=proposal.weights) # nll per sample
    mnll = nll.median().item()

    # flat table
    rows = [
        ['Posterior Predictive NLL', mnll],
        ['NPE time / ds [s]', tpd],
    ]
    eff = proposal.mean_efficiency
    if eff is not None:
        rows += [['IS Efficency', eff]]
    results = tabulate(rows, floatfmt='.3f', tablefmt='simple')
    return f'{results}\n'
