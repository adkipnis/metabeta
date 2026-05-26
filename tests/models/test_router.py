from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from metabeta.models.router import Router, joinCheckpoints


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


def _write_joint_checkpoint(path: Path) -> None:
    torch.save(
        {
            '_version': 1,
            'submodels': [
                {
                    'id': 'small',
                    'routing': {
                        'likelihood_family': 1,
                        'min_d': 1,
                        'max_d': 4,
                        'min_q': 1,
                        'max_q': 2,
                        'min_m': 5,
                        'max_m': 200,
                        'min_n': 5,
                        'max_n': 150,
                        'max_n_total': 3000,
                        'min_bg_df': 4,
                        'min_within_df': 2,
                    },
                    'model_cfg': {'d_ffx': 4, 'd_rfx': 2},
                    'model_state': {},
                },
                {
                    'id': 'medium',
                    'routing': {
                        'likelihood_family': 1,
                        'min_d': 5,
                        'max_d': 8,
                        'min_q': 1,
                        'max_q': 3,
                        'min_m': 5,
                        'max_m': 300,
                        'min_n': 5,
                        'max_n': 250,
                        'max_n_total': 8000,
                        'min_bg_df': 4,
                        'min_within_df': 2,
                    },
                    'model_cfg': {'d_ffx': 8, 'd_rfx': 3},
                    'model_state': {},
                },
            ],
        },
        path,
    )


def _batch(*, d: int, q: int, m: int = 12, n_i: int = 10, family: int = 1):
    max_d, max_q = max(8, d), max(3, q)
    ns = torch.zeros((1, m), dtype=torch.int64)
    ns[:, :] = n_i
    mask_d = torch.arange(max_d).unsqueeze(0) < d
    mask_q = torch.arange(max_q).unsqueeze(0) < q
    mask_m = ns > 0
    mask_n = torch.ones((1, m, n_i), dtype=torch.bool)
    return {
        'X': torch.zeros((1, m, n_i, max_d)),
        'Z': torch.zeros((1, m, n_i, max_q)),
        'y': torch.zeros((1, m, n_i)),
        'ns': ns,
        'm': torch.tensor([m]),
        'n': torch.tensor([m * n_i]),
        'mask_d': mask_d,
        'mask_q': mask_q,
        'mask_n': mask_n,
        'mask_m': mask_m,
        'likelihood_family': torch.tensor([family]),
    }


def test_router_selects_smallest_compatible_submodel(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_joint_checkpoint(joint_path)
    router = Router(joint_path)

    assert router.route(_batch(d=4, q=2)) == ['small']
    assert router.route(_batch(d=5, q=3)) == ['medium']


def test_router_rejects_dataset_outside_training_dimensions(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_joint_checkpoint(joint_path)
    router = Router(joint_path)

    with pytest.raises(ValueError, match='outside every routed submodel'):
        router.route(_batch(d=9, q=3))


def test_router_checks_between_and_within_group_degrees_of_freedom(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_joint_checkpoint(joint_path)
    router = Router(joint_path)

    with pytest.raises(ValueError, match='min_bg_df'):
        router.route(_batch(d=4, q=2, m=6, n_i=10))

    with pytest.raises(ValueError, match='min_within_df'):
        router.route(_batch(d=4, q=2, m=12, n_i=3))
