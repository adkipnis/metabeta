"""
Compare IS variants for the Normal likelihood across models and data splits:
  - Raw:         no IS, unweighted proposal
  - Conditional: IS with conditional log-likelihood (current)
  - Marginal:    IS with marginal log-likelihood (rfx integrated out analytically)

Reports NRMSE and LCR per parameter type on both validation and test sets.
"""

import sys
import copy
from pathlib import Path

import torch
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metabeta.evaluation.evaluate import Evaluator, setup as eval_setup
from metabeta.evaluation.point import getPointEstimates, getRMSE
from metabeta.evaluation.intervals import getCredibleIntervals, getCoverages, getCoverageErrors
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.evaluation import concatProposalsBatch, dictMean
from metabeta.utils.preprocessing import rescaleData
import argparse


CONFIGS = [
    'small-n-mixed', 'small-n-sampled',
    'mid-n-mixed',   'mid-n-sampled',
    'medium-n-mixed', 'medium-n-sampled',
]
EST_TYPE = 'mean'
PARAM_ORDER = ['ffx', 'sigma_rfx', 'rfx']
PARAM_LABELS = {'ffx': 'FFX', 'sigma_rfx': 'Sig(RFX)', 'rfx': 'RFX', 'Average': 'Avg'}


def sample_raw(ev: Evaluator, dl: Dataloader, cfg: argparse.Namespace):
    proposals = []
    with torch.inference_mode():
        for batch in dl:
            batch = toDevice(batch, ev.device)
            p = ev.model.estimate(batch, n_samples=cfg.n_samples)
            if cfg.rescale:
                p.rescale(batch['sd_y'])
            p.to('cpu')
            proposals.append(p)
    return concatProposalsBatch(proposals)


def apply_is(proposal, data_is: dict, marginal: bool):
    p = copy.deepcopy(proposal)
    sampler = ImportanceSampler(data_is, sir=False, likelihood_family=0, marginal=marginal)
    return sampler(p)


def avgOverAlpha(nested: dict) -> dict[str, torch.Tensor]:
    """Average dict[alpha, dict[param, tensor]] -> dict[param, tensor]."""
    alphas = list(nested.keys())
    params = list(nested[alphas[0]].keys())
    return {
        p: torch.stack([nested[a][p] for a in alphas]).mean(0)
        for p in params
    }


def metrics(proposal, data: dict) -> dict[str, dict[str, float]]:
    ests = getPointEstimates(proposal, EST_TYPE)
    nrmse_d = getRMSE(ests, data, normalize=True)
    ci_dicts = getCredibleIntervals(proposal)
    cvrg = getCoverages(ci_dicts, data)
    lcr_nested = getCoverageErrors(cvrg, log_ratio=True)
    lcr_d = avgOverAlpha(lcr_nested)

    out = {}
    for k in PARAM_ORDER:
        if k not in nrmse_d:
            continue
        out[k] = {
            'NRMSE': nrmse_d[k].mean().item(),
            'LCR':   lcr_d[k].mean().item(),
        }
    return out


def run_split(ev: Evaluator, dl: Dataloader, cfg: argparse.Namespace, split: str) -> None:
    full_batch = dl.fullBatch()
    full_batch_cpu = toDevice(full_batch, 'cpu')
    data_is = rescaleData(full_batch_cpu) if cfg.rescale else full_batch_cpu

    proposal_raw  = sample_raw(ev, dl, cfg)
    prop_cond = apply_is(proposal_raw, data_is, marginal=False)
    prop_marg = apply_is(proposal_raw, data_is, marginal=True)

    m_raw  = metrics(proposal_raw, data_is)
    m_cond = metrics(prop_cond,    data_is)
    m_marg = metrics(prop_marg,    data_is)

    eff_cond = prop_cond.efficiency.median().item()
    eff_marg = prop_marg.efficiency.median().item()

    print(f'  [{split}]  IS eff — cond: {eff_cond:.3f}  marg: {eff_marg:.3f}')

    rows = []
    for k in PARAM_ORDER:
        if k not in m_raw:
            continue
        label = PARAM_LABELS[k]
        rows.append([
            label,
            f'{m_raw[k]["NRMSE"]:.4f}',
            f'{m_raw[k]["LCR"]:.4f}',
            f'{m_cond[k]["NRMSE"]:.4f}',
            f'{m_cond[k]["LCR"]:.4f}',
            f'{m_marg[k]["NRMSE"]:.4f}',
            f'{m_marg[k]["LCR"]:.4f}',
        ])

    # averages
    def avg(m, key):
        vals = [m[k][key] for k in PARAM_ORDER if k in m]
        return sum(vals) / len(vals) if vals else float('nan')

    rows.append([
        'Avg',
        f'{avg(m_raw,  "NRMSE"):.4f}', f'{avg(m_raw,  "LCR"):.4f}',
        f'{avg(m_cond, "NRMSE"):.4f}', f'{avg(m_cond, "LCR"):.4f}',
        f'{avg(m_marg, "NRMSE"):.4f}', f'{avg(m_marg, "LCR"):.4f}',
    ])

    headers = ['', 'Raw NRMSE', 'Raw LCR', 'Cond NRMSE', 'Cond LCR', 'Marg NRMSE', 'Marg LCR']
    print(tabulate(rows, headers=headers, tablefmt='simple'))


def run_config(name: str) -> None:
    sys.argv = ['is_marginal.py', '--name', name]
    cfg = eval_setup()
    cfg.plot = False
    cfg.save_tables = False
    cfg.k = 0

    ev = Evaluator(cfg)

    print(f'\n{"=" * 68}')
    print(f'Config: {name}')
    print(f'{"=" * 68}')

    run_split(ev, ev.dl_valid, cfg, 'valid')
    run_split(ev, ev.dl_test,  cfg, 'test ')


if __name__ == '__main__':
    for cfg_name in CONFIGS:
        try:
            run_config(cfg_name)
        except Exception as e:
            import traceback
            print(f'\n[{cfg_name}] ERROR: {e}')
            traceback.print_exc()
