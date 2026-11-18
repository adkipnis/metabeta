"""IMH safeguards of MB on one whole real dataset beyond the checkpoints' training sizes.

This script calls the network directly, as ``real_posterior.py`` does, on a set of
``experiments/simulation/full_real_sets.py``: the raw flow, then the family's default
refinement (presets.yaml ``posthoc``, ``posterior_eval.refineProposal``), whose IMH stores its
diagnostics. Acceptance, minimum ESS and the k-hat of the pool weights are flagged with the
Api's thresholds. Writes the diagnostics and the global draws of both posteriors to ``--out``.

``--source`` runs the same diagnostics on every subsample of that source dataset in a
real-world test set, within the limits, to tell a hard dataset from too many groups or rows;
``--out`` then gets one JSON line per subsample and no draws.

Usage from repo root:
    uv run python experiments/evaluation/full_real_diagnostics.py --data_id large-n-full \
        --checkpoint metabeta/outputs/checkpoints/<name> --out experiments/results/full-real/x.npz
    uv run python experiments/evaluation/full_real_diagnostics.py --data_id large-n-real \
        --source insteval__grp_group --checkpoint metabeta/outputs/checkpoints/<name> \
        --out experiments/results/full-real/x.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from metabeta.models.api import PSIS_K_THRESHOLD
from metabeta.posthoc.metropolis import IMH_ACCEPT_WARN, IMH_ESS_WARN
from metabeta.utils.dataloader import Collection, collateGrouped
from metabeta.utils.experiments import DATA_DIR
from metabeta.utils.posterior_eval import (
    imhDiagnostics,
    loadModel,
    posthocDefaults,
    refineProposal,
    sampleMB,
)
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.results import Proposal
from metabeta.utils.sampling import setSeed


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_id', type=str, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--prefix', type=str, default='latest')
    parser.add_argument('--n_samples', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()
# fmt: on


def safeguards(refined: Proposal, method: str) -> list[dict]:
    """Per dataset, the IMH diagnostics ``method``'s refinement stored and the Api's flags."""
    d = imhDiagnostics(refined, method, ('accept_rate', 'ess_min', 'moved', 'pool_k'))
    return [
        {
            'method': method,
            'accept_rate': float(d['accept_rate'][i]),
            'accept_flag': bool(d['accept_rate'][i] < IMH_ACCEPT_WARN),
            'ess_min': float(d['ess_min'][i]),
            'ess_flag': bool(d['ess_min'][i] < IMH_ESS_WARN),
            'moves_ran': bool(d['moved'][i]),
            'psis_k_pool': float(d['pool_k'][i]),
            'psis_k_flag': bool(d['pool_k'][i] > PSIS_K_THRESHOLD),
        }
        for i in range(len(d['moved']))
    ]


def diagnose(
    model, lf: int, col: Collection, i: int, n_samples: int
) -> tuple[dict, Proposal, Proposal]:
    """Raw flow, then the default refinement, on dataset i: the diagnostics and both posteriors."""
    batch = collateGrouped([col[i]])

    t0 = time.perf_counter()
    cpu = torch.device('cpu')
    proposal, _ = sampleMB(model, batch, n_samples, batch_size=1, device=cpu, warmup=False)
    t_flow = time.perf_counter() - t0

    # refine in the rescaled space, as real_posterior.evaluateReal does
    proposal.rescale(batch['sd_y'])
    (method,) = posthocDefaults(lf)
    t0 = time.perf_counter()
    refined = refineProposal(method, proposal, rescaleData(batch), lf, batch_size=1)
    t_refine = time.perf_counter() - t0
    (info,) = safeguards(refined, method)
    return {**info, 'flow_s': t_flow, 'refine_s': t_refine}, proposal, refined


def main() -> None:
    cfg = setup()
    setSeed(cfg.seed)
    model, model_cfg = loadModel(cfg.checkpoint, cfg.prefix, torch.device('cpu'))
    lf = model_cfg.likelihood_family
    path = DATA_DIR / cfg.data_id / 'test.npz'
    col = Collection(path, max_d=model_cfg.max_d, max_q=model_cfg.max_q)
    run = {
        'data_id': cfg.data_id,
        'checkpoint': cfg.checkpoint.name,
        'prefix': cfg.prefix,
        'n_samples': cfg.n_samples,
    }
    cfg.out.parent.mkdir(parents=True, exist_ok=True)

    if cfg.source is not None:
        data = np.load(path, allow_pickle=True)
        sources = [str(np.asarray(s).item()) for s in data['source']]
        indices = [i for i, s in enumerate(sources) if s == cfg.source]
        assert indices, f'no dataset of {cfg.data_id} comes from {cfg.source}'
        with cfg.out.open('w') as f:
            for i in indices:
                info, _, _ = diagnose(model, lf, col, i, cfg.n_samples)
                dims = {k: int(data[k][i]) for k in ('m', 'n', 'd', 'q')}
                line = json.dumps({**run, 'source': cfg.source, 'index': i, **dims, **info})
                print(line, flush=True)
                f.write(line + '\n')
        return

    info, proposal, refined = diagnose(model, lf, col, 0, cfg.n_samples)
    info = {**run, **info}
    print(json.dumps(info), flush=True)
    draws = {'ffx': 'ffx', 'sigma_rfx': 'sigma_rfx'}
    if lf == 0:
        draws['sigma_eps'] = 'sigma_eps'
    np.savez_compressed(
        cfg.out,
        info=json.dumps(info),
        **{f'flow_{k}': getattr(proposal, a).numpy() for k, a in draws.items()},
        **{f'imh_{k}': getattr(refined, a).numpy() for k, a in draws.items()},
    )


if __name__ == '__main__':
    main()
