from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from metabeta.models.router import joinCheckpoints


def _write_checkpoint(path: Path, *, max_d: int, max_q: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'trainer_cfg': {
                'model_id': 'large',
                'seed': seed,
                'max_d': max_d,
                'max_q': max_q,
                'likelihood_family': 1,
                'min_bg_df': 4,
                'min_within_df': 2,
            },
            'data_cfg': {
                'data_id': f'test-d{max_d}-q{max_q}',
                'min_d': 1,
                'max_d': max_d,
                'min_q': 1,
                'max_q': max_q,
                'min_m': 5,
                'max_m': 200,
                'min_n': 5,
                'max_n': 150,
                'max_n_total': 3000,
                'shape_profile': 'standard',
                'likelihood_family': 1,
                'min_bg_df': 4,
                'min_within_df': 2,
            },
            'model_cfg': {
                'd_ffx': max_d,
                'd_rfx': max_q,
                'likelihood_family': 1,
            },
            'model_state': OrderedDict(
                {
                    'weight': torch.full((2, 2), float(max_d)),
                    'bias': torch.tensor([float(max_q)]),
                }
            ),
            'optimizer_state': {'state': {'unused': torch.tensor([1.0])}},
            'rng_torch': torch.get_rng_state(),
            'rng_python': ('not', 'needed'),
        },
        path,
    )


def test_join_checkpoints_writes_weights_only_joint_payload(tmp_path: Path):
    small_path = tmp_path / 'small.pt'
    medium_path = tmp_path / 'medium.pt'
    _write_checkpoint(small_path, max_d=4, max_q=2, seed=1)
    _write_checkpoint(medium_path, max_d=8, max_q=3, seed=1)

    output_path = joinCheckpoints(
        {'small': small_path, 'medium': medium_path},
        tmp_path / 'joint.pt',
    )

    payload = torch.load(output_path, map_location='cpu', weights_only=True)
    assert payload['_version'] == 1
    assert set(payload) == {'_version', 'created_at', 'submodels'}
    assert [submodel['id'] for submodel in payload['submodels']] == ['small', 'medium']

    small, medium = payload['submodels']
    assert small['routing']['max_d'] == 4
    assert small['routing']['max_q'] == 2
    assert small['routing']['max_n_total'] == 3000
    assert small['routing']['min_bg_df'] == 4
    assert small['routing']['min_within_df'] == 2
    assert medium['routing']['max_d'] == 8
    assert medium['routing']['max_q'] == 3
    assert torch.equal(small['model_state']['weight'], torch.full((2, 2), 4.0))

    for submodel in payload['submodels']:
        assert 'optimizer_state' not in submodel
        assert 'rng_torch' not in submodel
        assert 'rng_python' not in submodel


def test_join_checkpoints_resolves_directory_prefixes(tmp_path: Path):
    small_dir = tmp_path / 'small'
    medium_dir = tmp_path / 'medium'
    _write_checkpoint(small_dir / 'best.pt', max_d=4, max_q=2, seed=1)
    _write_checkpoint(medium_dir / 'latest.pt', max_d=8, max_q=3, seed=1)

    output_path = joinCheckpoints(
        [small_dir, medium_dir],
        tmp_path / 'joint.pt',
        ids=['small', 'medium'],
        prefixes={'small': 'best', 'medium': 'latest'},
    )

    payload = torch.load(output_path, map_location='cpu', weights_only=True)
    assert [submodel['source'] for submodel in payload['submodels']] == [
        str(small_dir / 'best.pt'),
        str(medium_dir / 'latest.pt'),
    ]


def test_join_checkpoints_uses_default_name_in_output_directory(tmp_path: Path):
    small_path = tmp_path / 'small.pt'
    medium_path = tmp_path / 'medium.pt'
    _write_checkpoint(small_path, max_d=4, max_q=2, seed=1)
    _write_checkpoint(medium_path, max_d=8, max_q=3, seed=1)

    output_path = joinCheckpoints(
        {'small': small_path, 'medium': medium_path},
        tmp_path,
    )

    assert output_path == tmp_path / 'joint_bernoulli_v1.pt'
    assert output_path.exists()


def test_join_checkpoints_rejects_missing_model_state(tmp_path: Path):
    checkpoint_path = tmp_path / 'bad.pt'
    torch.save({'optimizer_state': {}}, checkpoint_path)

    with pytest.raises(KeyError, match='model_state'):
        joinCheckpoints({'bad': checkpoint_path}, tmp_path / 'joint.pt')
