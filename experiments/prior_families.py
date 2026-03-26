"""
Tests whether a trained model actually incorporates prior family information
into its posteriors, rather than ignoring the family encoding.

Approach:
    1. Load a trained model and a batch of test data
    2. Fix the random seed and dataset
    3. For each parameter group (ffx, sigma_rfx, sigma_eps), swap the family
       index while keeping everything else identical
    4. Compare posterior mean and SD under each family
    5. Verify that heavier-tailed families (StudentT, HalfStudentT) produce
       wider posteriors than lighter ones (Normal, HalfNormal)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from metabeta.models.approximator import Approximator
from metabeta.utils.dataloader import Dataloader
from metabeta.utils.config import dataFromYaml, modelFromYaml
from metabeta.utils.io import runName
from metabeta.utils.families import FFX_FAMILIES, SIGMA_FAMILIES
from metabeta.utils.plot import PALETTE, niceify


DIR = Path(__file__).resolve().parent
ROOT = DIR / '..'
CKPT_DIR = ROOT / 'metabeta' / 'outputs' / 'checkpoints'
SEED = 42


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prior family sensitivity experiment.')
    parser.add_argument('--d_tag', type=str, default='toy', help='data config tag')
    parser.add_argument('--m_tag', type=str, default='toy', help='model config tag')
    parser.add_argument('--seed', type=int, default=SEED, help='training seed (for checkpoint lookup)')
    parser.add_argument('--r_tag', type=str, default=None, help='run tag (for checkpoint lookup)')
    parser.add_argument('--n_samples', type=int, default=512, help='posterior samples')
    parser.add_argument('--prefix', type=str, default='latest', help='checkpoint prefix [latest, best]')
    parser.add_argument('--valid', action='store_true', help='use validation set instead of test set')
    parser.add_argument('--delta', action='store_true', help='plot delta variance instead of absolute')
    return parser.parse_args()
# fmt: on


def loadData(cfg: argparse.Namespace) -> dict[str, torch.Tensor]:
    """Load a single batch from test (default) or validation set."""
    data_cfg_path = ROOT / 'metabeta' / 'simulation' / 'configs' / f'{cfg.d_tag}.yaml'
    partition = 'valid' if cfg.valid else 'test'
    data_fname = dataFromYaml(data_cfg_path, partition)
    data_path = ROOT / 'metabeta' / 'outputs' / 'data' / data_fname
    assert data_path.exists(), f'{data_path} not found'
    dl = Dataloader(data_path)
    return next(iter(dl))


def loadModel(cfg: argparse.Namespace, d: int, q: int) -> Approximator:
    """Load model from latest (or best) checkpoint."""
    model_cfg_path = ROOT / 'metabeta' / 'models' / 'configs' / f'{cfg.m_tag}.yaml'
    model_cfg = modelFromYaml(model_cfg_path, d, q)
    model = Approximator(model_cfg)

    # find checkpoint
    run = runName(vars(cfg))
    ckpt_path = CKPT_DIR / run / f'{cfg.prefix}.pt'
    assert ckpt_path.exists(), f'checkpoint not found: {ckpt_path}'
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(payload['model_state'])
    epoch = payload.get('epoch', '?')
    print(f'  loaded {cfg.prefix} checkpoint (epoch {epoch}) from {ckpt_path}')

    model.eval()
    return model


def resetRng(model: Approximator) -> None:
    """Reset base distribution RNGs for reproducible sampling."""
    model.posterior_g.base_dist.base.rng = np.random.default_rng(SEED)
    model.posterior_l.base_dist.base.rng = np.random.default_rng(SEED)


@torch.inference_mode()
def getPosteriorStats(
    model: Approximator,
    batch: dict[str, torch.Tensor],
    n_samples: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run inference and return posterior mean and SD per parameter group (rescaled)."""
    resetRng(model)
    proposal = model.estimate(batch, n_samples=n_samples)
    proposal.rescale(batch['sd_y'])
    return {
        'ffx': {
            'mean': proposal.ffx.mean(dim=-2),  # (b, d)
            'sd': proposal.ffx.std(dim=-2),  # (b, d)
        },
        'sigma_rfx': {
            'mean': proposal.sigma_rfx.mean(dim=-2),  # (b, q)
            'sd': proposal.sigma_rfx.std(dim=-2),  # (b, q)
        },
        'sigma_eps': {
            'mean': proposal.sigma_eps.mean(dim=-1),  # (b,)
            'sd': proposal.sigma_eps.std(dim=-1),  # (b,)
        },
    }


def setBatchFamily(
    batch: dict[str, torch.Tensor],
    key: str,
    value: int,
) -> dict[str, torch.Tensor]:
    """Clone batch and set all family indices for one group to a fixed value."""
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    out[key] = torch.full_like(out[key], value)
    return out


# ---- Step 1: check that posterior changes when family changes ----


def checkPosteriorChanges(
    model: Approximator,
    batch: dict[str, torch.Tensor],
    n_samples: int,
) -> None:
    """For each parameter group, verify that swapping the family index
    changes the posterior mean and SD."""
    print('\n--- Step 1: Does the posterior change when the family changes? ---')

    groups = [
        ('family_ffx', FFX_FAMILIES, 'ffx'),
        ('family_sigma_rfx', SIGMA_FAMILIES, 'sigma_rfx'),
        ('family_sigma_eps', SIGMA_FAMILIES, 'sigma_eps'),
    ]

    for family_key, families, param_key in groups:
        print(f'\n  [{param_key}] comparing {len(families)} families:')
        stats_per_family = {}
        for i, name in enumerate(families):
            b = setBatchFamily(batch, family_key, i)
            stats_per_family[name] = getPosteriorStats(model, b, n_samples)

        # compare all pairs
        names = list(stats_per_family.keys())
        for j in range(len(names)):
            for k in range(j + 1, len(names)):
                s_j = stats_per_family[names[j]][param_key]
                s_k = stats_per_family[names[k]][param_key]
                mean_diff = (s_j['mean'] - s_k['mean']).abs().mean().item()
                sd_diff = (s_j['sd'] - s_k['sd']).abs().mean().item()
                mean_changed = not torch.allclose(s_j['mean'], s_k['mean'], atol=1e-6)
                sd_changed = not torch.allclose(s_j['sd'], s_k['sd'], atol=1e-6)
                status = 'PASS' if (mean_changed and sd_changed) else 'FAIL'
                print(
                    f'    {names[j]} vs {names[k]}: '
                    f'mean diff={mean_diff:.6f}, sd diff={sd_diff:.6f} [{status}]'
                )


# ---- Step 2: check direction (heavier tails → wider posteriors) ----
