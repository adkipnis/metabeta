from pathlib import Path
from typing import Any

import numpy as np
import torch

from metabeta.utils.results import Proposal

# v2: caches written before the 2026-07-28 CouplingFlow.sample() log-det sign fix carry
# corrupted log_prob_g/log_prob_l; bumping the version rejects them everywhere at load.
POSTERIOR_SAMPLE_CACHE_VERSION = 2


def posteriorSampleCacheName(
    partition: str,
    method: str,
    checkpoint_name: str,
    checkpoint_prefix: str,
    n_samples: int,
    seed: int,
    k: int = 0,
) -> str:
    return (
        f'{partition}.{method}.{checkpoint_name}_{checkpoint_prefix}'
        f'_s{n_samples}_seed{seed}_k{k}.npz'
    )


def saveProposalCache(
    path: Path | str,
    proposal: Proposal,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    arrays: dict[str, np.ndarray] = {
        '_version': np.array(POSTERIOR_SAMPLE_CACHE_VERSION, dtype=np.int64),
        'samples_g': proposal.samples_g.detach().cpu().numpy(),
        'samples_l': proposal.samples_l.detach().cpu().numpy(),
        'has_sigma_eps': np.array(proposal.has_sigma_eps, dtype=np.bool_),
        'd_corr': np.array(proposal.d_corr, dtype=np.int64),
        'tpd': np.array(np.nan if proposal.tpd is None else proposal.tpd, dtype=np.float64),
        'reff': np.array(proposal.reff, dtype=np.float64),
    }
    for source, out_key in (('global', 'log_prob_g'), ('local', 'log_prob_l')):
        value = proposal.data.get(source, {}).get('log_prob')
        if value is not None:
            arrays[out_key] = value.detach().cpu().numpy()
    if proposal._corr_rfx is not None:
        arrays['corr_rfx'] = proposal._corr_rfx.detach().cpu().numpy()
    for key, value in proposal.is_results.items():
        if torch.is_tensor(value):
            arrays[f'is_{key}'] = value.detach().cpu().numpy()
    if metadata is not None:
        for key, value in metadata.items():
            arrays[f'meta_{key}'] = np.array(value)

    np.savez_compressed(path, **arrays)
    return path


def loadProposalCache(path: Path | str) -> tuple[Proposal, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as raw:
        version = int(raw['_version'])
        if version != POSTERIOR_SAMPLE_CACHE_VERSION:
            raise ValueError(f'unsupported posterior sample cache version: {version}')

        proposed = {
            'global': {'samples': torch.as_tensor(raw['samples_g'])},
            'local': {'samples': torch.as_tensor(raw['samples_l'])},
        }
        if 'log_prob_g' in raw.files:
            proposed['global']['log_prob'] = torch.as_tensor(raw['log_prob_g'])
        if 'log_prob_l' in raw.files:
            proposed['local']['log_prob'] = torch.as_tensor(raw['log_prob_l'])
        corr_rfx = torch.as_tensor(raw['corr_rfx']) if 'corr_rfx' in raw.files else None
        proposal = Proposal(
            proposed,
            has_sigma_eps=bool(raw['has_sigma_eps']),
            d_corr=int(raw['d_corr']),
            corr_rfx=corr_rfx,
        )
        tpd = float(raw['tpd'])
        proposal.tpd = None if np.isnan(tpd) else tpd
        proposal.reff = float(raw['reff']) if 'reff' in raw.files else 1.0
        proposal.is_results = {
            key[3:]: torch.as_tensor(raw[key]) for key in raw.files if key.startswith('is_')
        }
        for key in raw.files:
            if not key.startswith('meta_'):
                continue
            value = raw[key]
            metadata[key[5:]] = value.item() if value.shape == () else value

    return proposal, metadata
