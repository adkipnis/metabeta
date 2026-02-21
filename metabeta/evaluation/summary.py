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

Summary = dict[str, dict[str, float | torch.Tensor]]
EST_TYPE = 'mean'

def tensorMean(data: dict[str, torch.Tensor]) -> dict[str, float]:
    out = {}
    for k, v in data.items():
        out[k] = v.mean().item()
    return out


def dictMean(data: dict[str, float]) -> float:
    values = list(data.values())
    return float(np.mean(values))


def getSummary(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
) -> Summary:
    out = {}

    # point-based stats
    est = getPointEstimates(proposal, EST_TYPE)
    out['est'] = est
    out['nrmse'] = getRMSE(est, data, normalize=True)
    out['corr'] = getCorrelation(est, data)

    # inteval-based stats
    cvrg_results = analyzeCoverage(proposal, data)
    out['coverage'] = cvrg_results['coverage']
    out['ece'] = tensorMean(cvrg_results['error']['mean'])
    out['lcr'] = tensorMean(cvrg_results['log_ratio']['mean'])

    # posterior predictive
    pp = getPosteriorPredictive(proposal, data)
    nll = posteriorPredictiveNLL(pp, data, w=proposal.weights) # nll per sample
    out['mnll'] = nll.median().item()
    out['eff'] = proposal.mean_efficiency
    return out


def longTable(
    corr: dict[str, float],  # Pearson correlation
    nrmse: dict[str, float],  # normalized root mean square error
    ece: dict[str, float],  # expeced coverage error
    lcr: dict[str, float],  # log coverage ratio
) -> str:
    keys = ('ffx', 'sigma_rfx', 'sigma_eps', 'rfx')
    names = {
        'ffx': 'FFX',
        'sigma_rfx': 'Sigma(RFX)',
        'sigma_eps': 'Sigma(Eps)',
        'rfx': 'RFX',
    }
    rows = [[names[k], corr[k], nrmse[k], ece[k], lcr[k]] for k in keys]
    rows += [['Average', dictMean(corr), dictMean(nrmse), dictMean(ece), dictMean(lcr)]]
    results = tabulate(
        rows,
        headers=['', 'R', 'NRMSE', 'ECE', 'LCR'],
        floatfmt='.3f',
        tablefmt='simple',
    )
    return f'\n{results}\n'


def flatTable(
    mnll: float,
    tpd: float,
    eff: float | None = None,
) -> str:

    # flat table
    rows = [
        ['Posterior Predictive NLL', mnll],
        ['Estimation time / ds [s]', tpd],
    ]
    if eff is not None:
        rows += [['IS Efficency', eff]]
    results = tabulate(rows, floatfmt='.3f', tablefmt='simple')
    return f'{results}\n'


def summaryTable(summary: Summary) -> str:
    # results per parameter class
    corr = summary['corr']
    nrmse = summary['nrmse']
    ece = summary['ece']
    lcr = summary['lcr']
    long_table = longTable(corr, nrmse, ece, lcr) # type: ignore

    # general results
    mnll = summary['mnll']
    tpd = summary['tpd']
    eff = summary['eff']
    flat_table = flatTable(mnll, tpd, eff) # type: ignore
    return long_table + '\n' + flat_table


def coveragePlot(
    summary: Summary,
    proposal: Proposal,
    plot_dir: Path,
    epoch: int | None = None,
) -> None:
    cvrg = summary['coverage']
    names = (
        getNames('ffx', proposal.d) +
        getNames('sigmas', proposal.q) +
        getNames('rfx', proposal.q)
        )
    stats = {
        'ECE': 100 * dictMean(summary['ece']), # type: ignore
        'LCR': 100 * dictMean(summary['lcr']), # type: ignore
        }
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    plot.coverage(ax, cvrg, names, stats) # type: ignore
    ax.set_box_aspect(1)

    # store
    fname = plot_dir / 'coverage_latest.png'
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.15)
    if epoch is not None:
        fname_e = plot_dir / f'coverage_e{epoch}.png'
        plt.savefig(fname_e, bbox_inches='tight', pad_inches=0.15)
    plt.show()
    plt.close(fig)


def recoveryPlot(
    summary: Summary,
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
    est: dict[str, torch.Tensor] = summary['est'] # type: ignore
    stats = {
        k: v for k, v in summary.items()
        if k in ['corr', 'nrmse']
    }

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
