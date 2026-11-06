"""IMH safeguards of MB on one whole real dataset beyond the checkpoints' routing limits.

The Api refuses datasets outside a checkpoint's limits (groups, rows per group, total rows).
This script calls the network directly, as ``real_posterior.py`` does, on a set of
``experiments/simulation/full_real_sets.py``: the raw flow, then the family's default
refinement (presets.yaml ``posthoc``), run as in ``posterior_eval._refineChunk`` but keeping
the IMH diagnostics that the evaluation discards. Acceptance, minimum ESS and the k-hat of the
pool weights are flagged with the Api's thresholds. Writes the diagnostics and the global
draws of both posteriors to ``--out``.

Usage from repo root:
    uv run python experiments/evaluation/full_real_diagnostics.py --data_id large-n-full \
        --checkpoint metabeta/outputs/checkpoints/<name> --out experiments/results/full-real/x.npz
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from metabeta.models.api import IMH_ACCEPT_WARN, IMH_ESS_WARN, PSIS_K_THRESHOLD
from metabeta.posthoc.metropolis import IS2_N_INNER, MetropolisSampler
from metabeta.utils.dataloader import Collection, collateGrouped, trimBatchPadding
from metabeta.utils.experiments import DATA_DIR
from metabeta.utils.posterior_eval import (
    IMH_BURNIN,
    IMH_MOVES,
    IMH_N_CHAINS,
    _sliceSamples,
    loadModel,
    posthocDefaults,
    sampleMB,
)
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.psis import psislw
from metabeta.utils.sampling import setSeed


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_id', type=str, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--prefix', type=str, default='latest')
    parser.add_argument('--n_samples', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()
# fmt: on


def main() -> None:
    cfg = setup()
    setSeed(cfg.seed)
    device = torch.device('cpu')
    model, model_cfg = loadModel(cfg.checkpoint, cfg.prefix, device)
    lf = model_cfg.likelihood_family
    col = Collection(DATA_DIR / cfg.data_id / 'test.npz', max_d=model_cfg.max_d, max_q=model_cfg.max_q)
    batch = collateGrouped([col[0]])

    t0 = time.perf_counter()
    proposal, _ = sampleMB(model, batch, cfg.n_samples, batch_size=1, device=device, warmup=False)
    t_flow = time.perf_counter() - t0

    # refine in the rescaled space, as real_posterior.evaluateReal does
    proposal.rescale(batch['sd_y'])
    batch = rescaleData(batch)
    (method,) = posthocDefaults(lf)
    base, moves = IMH_MOVES[method]
    chunk = trimBatchPadding(batch)
    n_steps = cfg.n_samples // IMH_N_CHAINS
    pool = _sliceSamples(
        proposal.slice_b(0, 1).resizeGroups(chunk['mask_n'].shape[1]), IMH_N_CHAINS * n_steps
    )
    t0 = time.perf_counter()
    sampler = MetropolisSampler(
        chunk,
        n_chains=IMH_N_CHAINS,
        n_steps=n_steps,
        burnin=IMH_BURNIN,
        mode='marginal' if lf == 0 else 'laplace',
        likelihood_family=lf,
        n_inner=IS2_N_INNER if base == 'imhPM' else 0,
        moves=moves,
    )
    refined, diag = sampler(pool)
    t_refine = time.perf_counter() - t0

    accept = float(diag['accept_rate'].mean())
    ess_min = float(np.nanmin(diag['ess'].numpy()))
    k_pool = float(psislw(diag['log_w'].cpu().double().numpy())[1][0])
    info = {
        'data_id': cfg.data_id,
        'checkpoint': cfg.checkpoint.name,
        'prefix': cfg.prefix,
        'n_samples': cfg.n_samples,
        'method': method,
        'flow_s': t_flow,
        'refine_s': t_refine,
        'accept_rate': accept,
        'accept_flag': accept < IMH_ACCEPT_WARN,
        'ess_min': ess_min,
        'ess_flag': ess_min < IMH_ESS_WARN,
        'moves_ran': bool(diag['moved'][0]),
        'move_accept': float(np.nanmean(diag['move_accept'].numpy())),
        'psis_k_pool': k_pool,
        'psis_k_flag': k_pool > PSIS_K_THRESHOLD,
    }
    print(json.dumps(info), flush=True)
    draws = {'ffx': 'ffx', 'sigma_rfx': 'sigma_rfx'}
    if lf == 0:
        draws['sigma_eps'] = 'sigma_eps'
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.out,
        info=json.dumps(info),
        **{f'flow_{k}': getattr(proposal, a).numpy() for k, a in draws.items()},
        **{f'imh_{k}': getattr(refined, a).numpy() for k, a in draws.items()},
    )


if __name__ == '__main__':
    main()
