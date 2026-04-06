"""
RFX correlation recovery: metabeta vs NUTS vs ADVI.

For a given evaluation config (= trained model):
    1. Load model from checkpoint
    2. Load pre-fitted test set (.fit.npz)
    3. Sample metabeta (MB) posteriors
    4. Read pre-fitted NUTS and ADVI posteriors from the test file
    5. Plot rfx correlation recovery for all three methods in one figure

NUTS/ADVI fits must already exist in the .fit.npz test file
(produced by metabeta/simulation/fit.py).

Usage (from experiments/):
    uv run python rfx_corr.py
    uv run python rfx_corr.py --configs small-b-mixed small-b-sampled
    uv run python rfx_corr.py --show
    uv run python rfx_corr.py --n_sim 500
"""

import argparse
import time
import yaml
from pathlib import Path

import torch
from tqdm import tqdm

from metabeta.models.approximator import Approximator
from metabeta.utils.config import assimilateConfig, loadDataConfig, modelFromYaml
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.evaluation import Proposal, concatProposalsBatch
from metabeta.utils.io import datasetFilename, runName, setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.sampling import setSeed
from metabeta.evaluation.correlation import evaluateCorrelation, mergeCorrelationResults
from metabeta.plotting import plotRfxCorrelationFromResults

DIR = Path(__file__).resolve().parent
METABETA = DIR / '..' / 'metabeta'
EVAL_CFG_DIR = METABETA / 'evaluation' / 'configs'
OUT_DIR = DIR / 'results' / 'rfx_corr'

DEFAULT_CONFIGS = ['mid-n-mixed', 'medium-n-mixed', 'big-n-mixed']


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='RFX correlation recovery: MB vs NUTS vs ADVI.')
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS, help='evaluation config names')
    parser.add_argument('--n_sim', type=int, default=2000, help='simulations for empirical envelope')
    parser.add_argument('--show', action='store_true', help='display figures interactively')
    parser.add_argument('--verbosity', type=int, default=1, help='0=warnings | 1=info | 2=debug')
    return parser.parse_args()
# fmt: on


def loadEvalConfig(name: str, **overrides) -> argparse.Namespace:
    path = EVAL_CFG_DIR / f'{name}.yaml'
    assert path.exists(), f'eval config not found: {path}'
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg['name'] = name
    cfg.update(overrides)
    return argparse.Namespace(**cfg)


def initModel(cfg: argparse.Namespace, device: torch.device) -> tuple[Approximator, dict, str]:
    data_cfg = loadDataConfig(cfg.data_id)
    assimilateConfig(cfg, data_cfg)

    model_cfg_path = METABETA / 'models' / 'configs' / f'{cfg.model_id}.yaml'
    model_cfg = modelFromYaml(
        model_cfg_path,
        d_ffx=cfg.max_d,
        d_rfx=cfg.max_q,
        likelihood_family=getattr(cfg, 'likelihood_family', 0),
    )
    model = Approximator(model_cfg).to(device)
    model.eval()

    run = runName(vars(cfg))
    ckpt_dir = METABETA / 'outputs' / 'checkpoints' / run
    path = ckpt_dir / f'{cfg.prefix}.pt'
    assert path.exists(), f'checkpoint not found: {path}'
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload['model_state'])

    if cfg.compile and device.type != 'mps':
        model.compile()

    return model, data_cfg, run


def getTestDataloader(cfg: argparse.Namespace, batch_size: int | None = None) -> Dataloader:
    d_tag_valid = getattr(cfg, 'd_tag_valid', cfg.d_tag)
    data_cfg_valid = loadDataConfig(d_tag_valid)
    data_fname = datasetFilename('test')
    data_path = METABETA / 'outputs' / 'data' / data_cfg_valid['data_id'] / data_fname
    data_path = data_path.with_suffix('.fit.npz')
    assert data_path.exists(), f'fitted test data not found: {data_path}'
    sortish = batch_size is not None
    return Dataloader(data_path, batch_size=batch_size, sortish=sortish)


@torch.inference_mode()
def sampleMB(
    model: Approximator,
    cfg: argparse.Namespace,
    dl: Dataloader,
    device: torch.device,
) -> Proposal:
    proposals = []
    n_datasets = 0
    t0 = time.perf_counter()
    for batch in tqdm(dl, desc='  MB'):
        batch = toDevice(batch, device)
        proposal = model.estimate(batch, n_samples=cfg.n_samples)
        if cfg.rescale:
            proposal.rescale(batch['sd_y'])
        proposal.to('cpu')
        proposals.append(proposal)
        n_datasets += batch['X'].shape[0]
    t1 = time.perf_counter()

    merged = concatProposalsBatch(proposals)
    merged.tpd = (t1 - t0) / max(n_datasets, 1)
    return merged


def fit2proposal(batch: dict[str, torch.Tensor], prefix: str, rescale: bool) -> Proposal:
    ffx = batch[f'{prefix}_ffx']
    sigma_rfx = batch[f'{prefix}_sigma_rfx']
    samples_g = [ffx, sigma_rfx]
    if f'{prefix}_sigma_eps' in batch:
        samples_g.append(batch[f'{prefix}_sigma_eps'].unsqueeze(-1))
        has_sigma_eps = True
    else:
        has_sigma_eps = False
    proposed = {
        'global': {'samples': torch.cat(samples_g, dim=-1)},
        'local': {'samples': batch[f'{prefix}_rfx']},
    }
    proposal = Proposal(proposed, has_sigma_eps=has_sigma_eps)
    if rescale:
        proposal.rescale(batch['sd_y'])
    proposal.tpd = batch[f'{prefix}_duration'].mean().item()
    return proposal


METHODS = ['MB', 'NUTS', 'ADVI']


def run(configs: list[str], n_sim: int, show: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # accumulate results per method across all configs
    results_by_method: dict[str, list[dict]] = {m: [] for m in METHODS}
    n_skipped = 0

    for config_name in configs:
        print(f"\n{'=' * 60}")
        print(f'Config: {config_name}')
        print(f"{'=' * 60}")

        cfg = loadEvalConfig(config_name)
        setSeed(cfg.seed)
        device = setDevice(cfg.device)

        model, _, _ = initModel(cfg, device)

        dl_test = getTestDataloader(cfg, batch_size=8)
        full_batch = dl_test.fullBatch()

        # skip configs with no q >= 2 datasets
        qs = full_batch['mask_q'].sum(dim=-1).long()  # (b,)
        n_valid = int((qs >= 2).sum())
        if n_valid == 0:
            print(f'  Skipping: no datasets with q >= 2.')
            n_skipped += 1
            continue
        print(f'  {n_valid}/{len(qs)} datasets have q >= 2')

        proposal_mb = sampleMB(model, cfg, dl_test, device)
        proposal_nuts = fit2proposal(full_batch, prefix='nuts', rescale=cfg.rescale)
        proposal_advi = fit2proposal(full_batch, prefix='advi', rescale=cfg.rescale)

        batch = rescaleData(full_batch) if cfg.rescale else full_batch

        for label, proposal in zip(METHODS, [proposal_mb, proposal_nuts, proposal_advi]):
            results_by_method[label].append(evaluateCorrelation(proposal.rfx, batch, n_sim=n_sim))

    if len(configs) == n_skipped:
        print('\nNo valid configs to plot.')
        return

    merged = [mergeCorrelationResults(results_by_method[m]) for m in METHODS]
    plotRfxCorrelationFromResults(
        merged,
        labels=METHODS,
        plot_dir=OUT_DIR,
        show=show,
    )
    print(f'\n  Plot saved to {OUT_DIR}')


if __name__ == '__main__':
    args = setup()
    setupLogging(args.verbosity)
    run(args.configs, args.n_sim, args.show)
    print('\nDone.')
