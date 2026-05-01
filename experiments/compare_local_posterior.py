"""
experiments/gaussian_ceiling.py — Gaussian local posterior ceiling experiment.

Ablation for likelihood_family == 0: replaces the learned local flow with the
exact analytical Gaussian conditional posterior p(b_i | y_i, β, σ_rfx, σ_ε),
then evaluates RFX recovery under three global-conditioning scenarios:

  GaussLoc(true)  — condition on TRUE globals          [noise ceiling]
  GaussLoc(NUTS)  — condition on NUTS global samples
  GaussLoc(MB)    — condition on MB global samples

Compared against the standard baselines:
  MB              — full MetaBeta posterior (learned global + local flow)
  NUTS            — NUTS reference posterior

Key questions:
  GaussLoc(true) vs NUTS / MB  — how much better can we do with perfect globals?
  GaussLoc(MB)   vs MB         — is the learned local flow the bottleneck?
  GaussLoc(NUTS) vs NUTS       — does the analytical form help given good globals?

Usage (from metabeta/experiments/):
  uv run python gaussian_ceiling.py --checkpoint ../outputs/checkpoints/<run>
  uv run python gaussian_ceiling.py --checkpoint ../outputs/checkpoints/<run> --prefix best
"""

import argparse
import logging
from pathlib import Path

import torch
from tabulate import tabulate
from tqdm import tqdm

from metabeta.utils.logger import setupLogging
from metabeta.utils.io import setDevice, datasetFilename
from metabeta.utils.sampling import setSeed
from metabeta.utils.config import modelFromYaml, assimilateConfig, loadDataConfig
from metabeta.utils.templates import loadConfigFromCheckpoint
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.evaluation import Proposal, concatProposalsBatch, dictMean
from metabeta.models.approximator import Approximator
from metabeta.evaluation.summary import getSummary
from metabeta.posthoc.gaussian_local import gaussianCeiling, gaussianHybrid

logger = logging.getLogger('gaussian_ceiling')


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='Gaussian local posterior ceiling experiment',
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--prefix', type=str, default='latest',
                        help='Checkpoint prefix: best or latest')
    parser.add_argument('--data_id_valid', type=str,
                        help='Override validation/test data id')
    parser.add_argument('--n_samples', type=int, default=512,
                        help='Posterior samples per dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Datasets per minibatch for MB sampling')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--verbosity', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    # fmt: on
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    cfg_dict = loadConfigFromCheckpoint(checkpoint_path)
    cfg_dict['_checkpoint_dir'] = str(checkpoint_path)
    cfg_dict['_checkpoint_prefix'] = getattr(args, 'prefix', 'latest')

    for k in ('n_samples', 'batch_size', 'device', 'verbosity', 'seed', 'data_id_valid'):
        if hasattr(args, k):
            cfg_dict[k] = getattr(args, k)

    cfg_dict.setdefault('batch_size', 8)
    cfg_dict.setdefault('n_samples', 512)
    cfg_dict.setdefault('rescale', True)

    return argparse.Namespace(**cfg_dict)


# ---------------------------------------------------------------------------
# Data + model loading
# ---------------------------------------------------------------------------


def _loadData(cfg: argparse.Namespace, dir: Path) -> tuple[Dataloader, Dataloader]:
    data_cfg_train = loadDataConfig(cfg.data_id)
    assimilateConfig(cfg, data_cfg_train)

    data_cfg_valid = loadDataConfig(cfg.data_id_valid)
    data_id = data_cfg_valid['data_id']

    def _dl(partition: str) -> Dataloader:
        fname = datasetFilename(partition)
        path = dir / '..' / 'metabeta' / 'outputs' / 'data' / data_id / fname
        if partition == 'test':
            path = path.with_suffix('.fit.npz')
        path = path.resolve()
        assert path.exists(), f'data not found: {path}'
        return Dataloader(path, batch_size=cfg.batch_size, sortish=True)

    return _dl('valid'), _dl('test')


def _loadModel(cfg: argparse.Namespace, dir: Path, device: torch.device) -> Approximator:
    model_cfg_path = dir / '..' / 'metabeta' / 'configs' / 'models' / f'{cfg.model_id}.yaml'
    model_cfg = modelFromYaml(
        model_cfg_path.resolve(),
        d_ffx=cfg.max_d,
        d_rfx=cfg.max_q,
        likelihood_family=cfg.likelihood_family,
    )
    model = Approximator(model_cfg).to(device)
    model.eval()

    ckpt_dir = Path(cfg._checkpoint_dir)
    prefix = cfg._checkpoint_prefix
    path = ckpt_dir / f'{prefix}.pt'
    assert path.exists(), f'checkpoint not found: {path}'
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload['model_state'])
    logger.info('Loaded checkpoint: %s', path)
    return model


# ---------------------------------------------------------------------------
# Proposal helpers
# ---------------------------------------------------------------------------


def _batchToProposal(
    batch: dict[str, torch.Tensor],
    prefix: str,
    rescale: bool,
) -> Proposal:
    """Reconstruct a Proposal from stored NUTS/ADVI samples in the batch."""
    ffx = batch[f'{prefix}_ffx']
    sigma_rfx = batch[f'{prefix}_sigma_rfx']
    parts_g = [ffx, sigma_rfx]
    has_sigma_eps = f'{prefix}_sigma_eps' in batch
    if has_sigma_eps:
        parts_g.append(batch[f'{prefix}_sigma_eps'].unsqueeze(-1))
    global_samples = torch.cat(parts_g, dim=-1)
    corr_rfx = batch.get(f'{prefix}_corr_rfx', None)
    proposed = {
        'global': {'samples': global_samples, 'log_prob': torch.zeros(global_samples.shape[:2])},
        'local': {
            'samples': batch[f'{prefix}_rfx'],
            'log_prob': torch.zeros(batch[f'{prefix}_rfx'].shape[:-1]),
        },
    }
    proposal = Proposal(proposed, has_sigma_eps=has_sigma_eps, corr_rfx=corr_rfx)
    if rescale:
        proposal.rescale(batch['sd_y'])
    return proposal


@torch.inference_mode()
def _sampleMB(
    model: Approximator,
    dl: Dataloader,
    n_samples: int,
    device: torch.device,
    rescale: bool,
) -> Proposal:
    proposals = []
    for batch in tqdm(dl, desc='  MB sampling'):
        batch = toDevice(batch, device)
        p = model.estimate(batch, n_samples=n_samples)
        if rescale:
            p.rescale(batch['sd_y'])
        p.to('cpu')
        proposals.append(p)
    return concatProposalsBatch(proposals)


def _sliceBatch(batch: dict, start: int, end: int, B: int) -> dict:
    return {
        k: v[start:end] if isinstance(v, torch.Tensor) and v.shape[0] == B else v
        for k, v in batch.items()
    }


def _sliceProposal(p: Proposal, start: int, end: int) -> Proposal:
    g, l = p.data['global'], p.data['local']
    mini_data = {
        'global': {'samples': g['samples'][start:end], 'log_prob': g['log_prob'][start:end]},
        'local': {'samples': l['samples'][start:end], 'log_prob': l['log_prob'][start:end]},
    }
    corr_rfx = p._corr_rfx[start:end] if p._corr_rfx is not None else None
    return Proposal(mini_data, has_sigma_eps=p.has_sigma_eps, d_corr=p.d_corr, corr_rfx=corr_rfx)


@torch.inference_mode()
def _analyticalCeilingBatched(
    batch: dict,
    d_ffx: int,
    d_rfx: int,
    n_samples: int,
    batch_size: int,
) -> Proposal:
    B = batch['y'].shape[0]
    proposals = []
    for start in tqdm(range(0, B, batch_size), desc='  GaussLoc(true)'):
        end = min(start + batch_size, B)
        proposals.append(gaussianCeiling(_sliceBatch(batch, start, end, B), d_ffx, d_rfx, n_samples))
    return concatProposalsBatch(proposals)


@torch.inference_mode()
def _analyticalHybridBatched(
    global_proposal: Proposal,
    batch: dict,
    batch_size: int,
    label: str = '',
) -> Proposal:
    B = batch['y'].shape[0]
    proposals = []
    for start in tqdm(range(0, B, batch_size), desc=f'  GaussLoc({label})'):
        end = min(start + batch_size, B)
        proposals.append(
            gaussianHybrid(_sliceProposal(global_proposal, start, end), _sliceBatch(batch, start, end, B))
        )
    return concatProposalsBatch(proposals)


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------


_EXCL = frozenset({'corr_rfx'})


def _meanExcl(d: dict) -> float:
    """Mean across parameter types, excluding corr_rfx (matches evaluate.py summaryTable)."""
    return dictMean({k: v for k, v in d.items() if k not in _EXCL})


def _makeRow(label: str, summary, fit_label: str) -> dict:
    nrmse_rfx = summary.nrmse.get('rfx', None)
    return {
        'Method': label,
        'NRMSE': _meanExcl(summary.nrmse),
        'NRMSE_rfx': float(nrmse_rfx.mean()) if nrmse_rfx is not None else None,
        'R': _meanExcl(summary.corr),
        'ECE': _meanExcl(summary.ece),
        'RFX_joint_ECE': summary.rfx_joint_ece,
        fit_label: summary.mfit,
        'ppNLL': summary.mnll,
        'LOO-NLL': (float(summary.loo_nll.median()) if summary.loo_nll is not None else None),
        'Pareto_k': summary.mk,
    }


def _printTable(rows: list[dict]) -> None:
    headers = list(rows[0].keys())
    table = [
        [
            r[h] if r[h] is None else f'{r[h]:.4f}' if isinstance(r[h], float) else r[h]
            for h in headers
        ]
        for r in rows
    ]
    print()
    print(tabulate(table, headers=headers, tablefmt='simple', stralign='right'))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)

    if cfg.likelihood_family != 0:
        raise SystemExit(
            f'gaussian_ceiling only supports likelihood_family=0 (Gaussian); '
            f'got family={cfg.likelihood_family}'
        )

    device = setDevice(cfg.device)
    dir = Path(__file__).resolve().parent

    _, dl_test = _loadData(cfg, dir)
    model = _loadModel(cfg, dir, device)

    fit_label = 'ppR2'

    full_batch = dl_test.fullBatch()
    batch_rescaled = rescaleData(full_batch) if cfg.rescale else full_batch

    # --- standard proposals
    proposal_mb = _sampleMB(model, dl_test, cfg.n_samples, device, cfg.rescale)
    proposal_nuts = _batchToProposal(full_batch, 'nuts', cfg.rescale)
    proposal_advi = _batchToProposal(full_batch, 'advi', cfg.rescale)

    # --- Gaussian analytical proposals (batched to avoid OOM on large datasets)
    logger.info('Computing Gaussian analytical proposals...')
    proposal_ceil = _analyticalCeilingBatched(
        batch_rescaled, cfg.max_d, cfg.max_q, cfg.n_samples, cfg.batch_size
    )
    proposal_gl_nuts = _analyticalHybridBatched(proposal_nuts, batch_rescaled, cfg.batch_size, 'NUTS')
    proposal_gl_mb = _analyticalHybridBatched(proposal_mb, batch_rescaled, cfg.batch_size, 'MB')

    # --- summaries
    lf = cfg.likelihood_family
    rows = []
    for label, proposal in [
        ('MB', proposal_mb),
        ('NUTS', proposal_nuts),
        ('ADVI', proposal_advi),
        ('GaussLoc(true)', proposal_ceil),
        ('GaussLoc(NUTS)', proposal_gl_nuts),
        ('GaussLoc(MB)', proposal_gl_mb),
    ]:
        logger.info('  Summarising %s...', label)
        summary = getSummary(proposal, batch_rescaled, likelihood_family=lf)
        rows.append(_makeRow(label, summary, fit_label))

    _printTable(rows)


if __name__ == '__main__':
    main()
