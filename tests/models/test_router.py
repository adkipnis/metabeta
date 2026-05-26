from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from metabeta.models.router import Router
from metabeta.utils.dataloader import Dataloader
from metabeta.utils.router import joinCheckpoints


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


def _write_math_joint_checkpoint(path: Path) -> None:
    torch.save(
        {
            '_version': 1,
            'submodels': [
                {
                    'id': 'math-small',
                    'routing': {
                        'likelihood_family': 0,
                        'min_d': 1,
                        'max_d': 3,
                        'min_q': 1,
                        'max_q': 2,
                        'min_m': 5,
                        'max_m': 300,
                        'min_n': 5,
                        'max_n': 200,
                        'max_n_total': 10_000,
                        'min_bg_df': 0,
                        'min_within_df': 0,
                    },
                    'model_cfg': {'d_ffx': 3, 'd_rfx': 2, 'likelihood_family': 0},
                    'model_state': {},
                },
                {
                    'id': 'math-large',
                    'routing': {
                        'likelihood_family': 0,
                        'min_d': 1,
                        'max_d': 6,
                        'min_q': 1,
                        'max_q': 3,
                        'min_m': 5,
                        'max_m': 300,
                        'min_n': 5,
                        'max_n': 200,
                        'max_n_total': 10_000,
                        'min_bg_df': 0,
                        'min_within_df': 0,
                    },
                    'model_cfg': {'d_ffx': 6, 'd_rfx': 3, 'likelihood_family': 0},
                    'model_state': {},
                },
                {
                    'id': 'math-q5',
                    'routing': {
                        'likelihood_family': 0,
                        'min_d': 1,
                        'max_d': 6,
                        'min_q': 4,
                        'max_q': 5,
                        'min_m': 5,
                        'max_m': 300,
                        'min_n': 5,
                        'max_n': 200,
                        'max_n_total': 10_000,
                        'min_bg_df': 0,
                        'min_within_df': 0,
                    },
                    'model_cfg': {'d_ffx': 6, 'd_rfx': 5, 'likelihood_family': 0},
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

    routes, _ = router._routeBatch(_batch(d=4, q=2))
    assert routes == ['small']

    routes, _ = router._routeBatch(_batch(d=5, q=3))
    assert routes == ['medium']


def test_router_rejects_dataset_outside_training_dimensions(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_joint_checkpoint(joint_path)
    router = Router(joint_path)

    with pytest.warns(RuntimeWarning, match='incompatible'):
        with pytest.raises(ValueError, match='outside every routed submodel'):
            router._routeBatch(_batch(d=9, q=3))


def test_router_checks_between_and_within_group_degrees_of_freedom(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_joint_checkpoint(joint_path)
    router = Router(joint_path)

    with pytest.warns(RuntimeWarning, match='incompatible'):
        with pytest.raises(ValueError, match='min_bg_df'):
            router._routeBatch(_batch(d=4, q=2, m=6, n_i=10))

    with pytest.warns(RuntimeWarning, match='incompatible'):
        with pytest.raises(ValueError, match='min_within_df'):
            router._routeBatch(_batch(d=4, q=2, m=12, n_i=3))


def test_router_prepares_preprocessed_numpy_dict_with_formula_and_default_priors(
    tmp_path: Path,
):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)

    with np.load(
        Path('metabeta/datasets/preprocessed/test/math__grp_group.npz'),
        allow_pickle=True,
    ) as raw:
        preprocessed = dict(raw)

    batch = router.prepareData(
        preprocessed,
        formula='y ~ meanses + ses + (1 + ses | group)',
    )

    assert router.route(batch) == ['math-small']
    assert batch['X'].shape == (1, 160, 67, 3)
    assert batch['Z'].shape == (1, 160, 67, 2)
    assert torch.equal(batch['mask_d'], torch.tensor([[True, True, True]]))
    assert torch.equal(batch['mask_q'], torch.tensor([[True, True]]))
    assert batch['likelihood_family'].item() == 0
    assert batch['tau_ffx'].shape == (1, 3)
    assert batch['tau_rfx'].shape == (1, 2)

    first = preprocessed['X'][0]
    assert torch.allclose(
        batch['X'][0, 0, 0, :3],
        torch.tensor([1.0, first[1], first[0]], dtype=batch['X'].dtype),
    )
    assert torch.allclose(
        batch['Z'][0, 0, 0, :2],
        torch.tensor([1.0, first[0]], dtype=batch['Z'].dtype),
    )


def test_router_accepts_canonical_fit_style_priors(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)

    with np.load(
        Path('metabeta/datasets/preprocessed/test/math__grp_group.npz'),
        allow_pickle=True,
    ) as raw:
        preprocessed = dict(raw)

    batch = router.prepareData(
        preprocessed,
        formula='y ~ meanses + ses + (1 + ses | group)',
        priors={
            'nu_ffx': np.array([1.0, 2.0, 3.0]),
            'tau_ffx': np.array([3.0, 2.0, 1.0]),
            'family_ffx': 0,
            'tau_rfx': np.array([1.5, 0.5]),
            'family_sigma_rfx': 0,
            'tau_eps': 1.2,
            'family_sigma_eps': 1,
            'eta_rfx': 0.0,
        },
    )

    assert router.route(batch) == ['math-small']
    assert torch.allclose(batch['nu_ffx'], torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.allclose(batch['tau_ffx'], torch.tensor([[3.0, 2.0, 1.0]]))
    assert torch.allclose(batch['tau_rfx'], torch.tensor([[1.5, 0.5]]))
    assert torch.allclose(batch['tau_eps'], torch.tensor([1.2]))
    assert torch.equal(batch['family_ffx'], torch.tensor([0]))
    assert torch.equal(batch['family_sigma_eps'], torch.tensor([1]))
    assert torch.allclose(batch['eta_rfx'], torch.tensor([0.0]))


def test_router_expands_multiple_named_term_priors_per_dataset(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)

    with np.load(
        Path('metabeta/datasets/preprocessed/test/math__grp_group.npz'),
        allow_pickle=True,
    ) as raw:
        preprocessed = dict(raw)

    batch = router.prepareData(
        preprocessed,
        formula='y ~ meanses + ses + (1 + ses | group)',
        priors={
            'weak': {
                'fixed': {
                    'Intercept': {'mu': 1.0, 'sigma': 3.0},
                    'ses': {'sigma': 0.5},
                },
                'random_sd': {'ses': {'sigma': 0.7}},
                'sigma_eps': {'sigma': 1.2},
                'corr_rfx': {'eta': 0.0},
            },
            'tight': {
                'fixed': {'meanses': {'sigma': 0.1}},
                'random_sd': {'Intercept': {'sigma': 0.3}},
            },
        },
    )

    assert router.route(batch) == ['math-small', 'math-small']
    assert batch['X'].shape[0] == 2
    assert torch.allclose(batch['nu_ffx'][0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(batch['tau_ffx'][0], torch.tensor([3.0, 2.5, 0.5]))
    assert torch.allclose(batch['tau_rfx'][0], torch.tensor([2.5, 0.7]))
    assert torch.allclose(batch['tau_eps'][0], torch.tensor(1.2))
    assert torch.allclose(batch['eta_rfx'][0], torch.tensor(0.0))
    assert torch.allclose(batch['tau_ffx'][1], torch.tensor([2.5, 0.1, 2.5]))
    assert torch.allclose(batch['tau_rfx'][1], torch.tensor([0.3, 2.5]))
    assert torch.equal(batch['_router_prior_index'], torch.tensor([0, 1]))
    assert torch.equal(batch['_router_source_index'], torch.tensor([0, 0]))
    assert batch['_router_prior_name'] == ['weak', 'tight']

    _, validation = router._routeBatch(batch)
    assert [row['prior_name'] for row in validation] == ['weak', 'tight']
    assert [row['prior_index'] for row in validation] == [0, 1]


def test_router_rejects_q_above_formula_limit(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)

    with np.load(
        Path('metabeta/datasets/preprocessed/test/math__grp_group.npz'),
        allow_pickle=True,
    ) as raw:
        preprocessed = dict(raw)

    with pytest.raises(ValueError, match='q must be <= 5'):
        router.prepareData(preprocessed, formula='y ~ meanses + ses', q=6)


def test_router_rejects_multiple_random_effect_blocks(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)

    with np.load(
        Path('metabeta/datasets/preprocessed/test/math__grp_group.npz'),
        allow_pickle=True,
    ) as raw:
        preprocessed = dict(raw)

    with pytest.raises(NotImplementedError, match='one random term'):
        router.prepareData(
            preprocessed,
            formula='y ~ meanses + ses + (1 | group) + (1 | school)',
        )


def test_router_rejects_missing_formula_term(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)

    with np.load(
        Path('metabeta/datasets/preprocessed/test/math__grp_group.npz'),
        allow_pickle=True,
    ) as raw:
        preprocessed = dict(raw)

    with pytest.raises(KeyError, match='formula term not found'):
        router.prepareData(preprocessed, formula='y ~ definitely_missing + (1 | group)')


def test_router_rejects_missing_prior_term(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)

    with np.load(
        Path('metabeta/datasets/preprocessed/test/math__grp_group.npz'),
        allow_pickle=True,
    ) as raw:
        preprocessed = dict(raw)

    with pytest.raises(KeyError, match='prior term not found'):
        router.prepareData(
            preprocessed,
            formula='y ~ meanses + ses + (1 + ses | group)',
            priors={'fixed': {'definitely_missing': {'sigma': 1.0}}},
        )


def test_router_rejects_malformed_canonical_prior_shapes(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)

    with np.load(
        Path('metabeta/datasets/preprocessed/test/math__grp_group.npz'),
        allow_pickle=True,
    ) as raw:
        preprocessed = dict(raw)

    with pytest.raises(ValueError, match='tau_ffx must have shape'):
        router.prepareData(
            preprocessed,
            formula='y ~ meanses + ses + (1 + ses | group)',
            priors={'tau_ffx': np.array([1.0, 2.0])},
        )


def test_router_rejects_per_term_family_mismatch(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)

    with np.load(
        Path('metabeta/datasets/preprocessed/test/math__grp_group.npz'),
        allow_pickle=True,
    ) as raw:
        preprocessed = dict(raw)

    with pytest.raises(ValueError, match='per-term fixed-effect prior families'):
        router.prepareData(
            preprocessed,
            formula='y ~ meanses + ses + (1 + ses | group)',
            priors={
                'fixed': {
                    'meanses': {'family': 'normal', 'sigma': 1.0},
                    'ses': {'family': 'student', 'sigma': 1.0},
                }
            },
        )


def test_router_rejects_sigma_eps_prior_for_non_gaussian(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_joint_checkpoint(joint_path)
    router = Router(joint_path)

    preprocessed = {
        'X': np.zeros((120, 2), dtype=float),
        'y': np.tile(np.array([0.0, 1.0]), 60),
        'groups': np.repeat(np.arange(12), 10),
        'columns': np.array(['x1', 'x2']),
        'd': np.array(3),
        'n': np.array(120),
        'ns': np.full(12, 10),
        'm': np.array(12),
        'y_type': np.array('binary'),
    }

    with pytest.raises(ValueError, match='sigma_eps prior is only valid'):
        with pytest.warns(RuntimeWarning, match='near-constant'):
            router.prepareData(
                preprocessed,
                formula='y ~ x1 + x2 + (1 + x1 | group)',
                priors={'sigma_eps': {'sigma': 1.0}},
            )


def test_router_rejects_empty_prior_list(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)

    with np.load(
        Path('metabeta/datasets/preprocessed/test/math__grp_group.npz'),
        allow_pickle=True,
    ) as raw:
        preprocessed = dict(raw)

    with pytest.raises(ValueError, match='priors sequence cannot be empty'):
        router.prepareData(
            preprocessed,
            formula='y ~ meanses + ses + (1 + ses | group)',
            priors=[],
        )


def test_router_prepares_parquet_through_preprocessor(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)

    parquet_path = Path('metabeta/datasets/from-r/parquet/math.parquet')
    batch = router.prepareData(
        parquet_path,
        formula='y ~ meanses + ses + minority + sex + (1 | group)',
        fit_preprocessor=True,
    )

    assert router.route(batch) == ['math-large']
    assert batch['X'].shape == (1, 160, 67, 6)
    assert batch['Z'].shape == (1, 160, 67, 3)
    assert batch['mask_d'].sum().item() == 5
    assert batch['mask_q'].sum().item() == 1
    assert batch['likelihood_family'].item() == 0


def test_router_supports_formula_random_effects_up_to_q5(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)

    parquet_path = Path('metabeta/datasets/from-r/parquet/math.parquet')
    batch = router.prepareData(
        parquet_path,
        formula='y ~ meanses + ses + minority + sex + (1 + ses + meanses + minority + sex | group)',
        fit_preprocessor=True,
    )

    assert router.route(batch) == ['math-q5']
    assert batch['Z'].shape == (1, 160, 67, 5)
    assert batch['mask_q'].sum().item() == 5


def test_router_rejects_tabular_input_without_preprocessor_or_fit_flag(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)
    df = pd.read_parquet('metabeta/datasets/from-r/parquet/math.parquet')

    with pytest.raises(ValueError, match='requires a fitted preprocessor'):
        router.prepareData(df, formula='y ~ ses + (1 | group)')


# ---------------------------------------------------------------------------
# Validation gap tests
# ---------------------------------------------------------------------------


def _full_batch(*, d: int, q: int, m: int = 12, n_i: int = 10, family: int = 1):
    """Like _batch but includes all keys required by _validateBatchFormat."""
    b = _batch(d=d, q=q, m=m, n_i=n_i, family=family)
    max_d, max_q = b['X'].shape[-1], b['Z'].shape[-1]
    b['nu_ffx'] = torch.zeros(1, max_d)
    b['tau_ffx'] = torch.ones(1, max_d) * 2.5
    b['tau_rfx'] = torch.ones(1, max_q) * 2.5
    b['eta_rfx'] = torch.zeros(1)
    b['family_ffx'] = torch.zeros(1, dtype=torch.long)
    b['family_sigma_rfx'] = torch.zeros(1, dtype=torch.long)
    b['sd_y'] = torch.ones(1)
    b['mask_mq'] = b['mask_m'].unsqueeze(-1) & b['mask_q'].unsqueeze(-2)
    q_val = max_q
    b['mask_corr'] = (
        torch.stack(
            [b['mask_q'][..., i] & b['mask_q'][..., j] for i in range(1, q_val) for j in range(i)],
            dim=-1,
        )
        if q_val >= 2
        else b['mask_q'].new_zeros(1, 0)
    )
    return b


def test_router_validates_mask_d_width(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_joint_checkpoint(joint_path)
    router = Router(joint_path)

    bad = _full_batch(d=4, q=2)
    bad['mask_d'] = bad['mask_d'][..., :3]  # wrong width (3 instead of 8)

    with pytest.raises(ValueError, match='mask_d width'):
        router._validateBatchFormat(bad)


def test_router_validates_mask_q_width(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_joint_checkpoint(joint_path)
    router = Router(joint_path)

    bad = _full_batch(d=4, q=2)
    bad['mask_q'] = bad['mask_q'][..., :2]  # wrong width (2 instead of 3)

    with pytest.raises(ValueError, match='mask_q width'):
        router._validateBatchFormat(bad)


def test_router_validates_mask_m_active_count(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_joint_checkpoint(joint_path)
    router = Router(joint_path)

    bad = _full_batch(d=4, q=2, m=12)
    # Report m=12 but mask_m has only 10 active groups
    mask_m = bad['mask_m'].clone()
    mask_m[0, 10] = False
    mask_m[0, 11] = False
    bad['mask_m'] = mask_m

    with pytest.raises(ValueError, match='mask_m active count'):
        router._validateBatchFormat(bad)


def test_router_validates_stats_beta_est_dim(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_joint_checkpoint(joint_path)
    router = Router(joint_path)

    bad = _full_batch(d=4, q=2)
    bad['stats'] = {'beta_est': torch.zeros(1, 6)}  # wrong d (6 instead of 8)

    with pytest.raises(ValueError, match='stats.beta_est'):
        router._validateBatchFormat(bad)


def test_router_validates_stats_sigma_rfx_est_dim(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_joint_checkpoint(joint_path)
    router = Router(joint_path)

    bad = _full_batch(d=4, q=2)
    bad['stats'] = {'sigma_rfx_est': torch.zeros(1, 5)}  # wrong q (5 instead of 3)

    with pytest.raises(ValueError, match='stats.sigma_rfx_est'):
        router._validateBatchFormat(bad)


# ---------------------------------------------------------------------------
# Integration tests (require local checkpoint and data files)
# ---------------------------------------------------------------------------

_JOINT_CHECKPOINT = Path('metabeta/outputs/checkpoints/joint_bernoulli_v1.pt')
_SMALL_B_SAMPLED = Path('metabeta/outputs/data/small-b-sampled/test.npz')
_MEDIUM_B_SAMPLED = Path('metabeta/outputs/data/medium-b-sampled/test.npz')

_HAS_REAL_DATA = (
    _JOINT_CHECKPOINT.exists() and _SMALL_B_SAMPLED.exists() and _MEDIUM_B_SAMPLED.exists()
)


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_REAL_DATA, reason='local checkpoint + data not present')
def test_router_sample_small_partition_returns_valid_proposal_shapes():
    from metabeta.utils.dataloader import Dataloader

    router = Router(_JOINT_CHECKPOINT, batch_size=4)
    result = router.sample(_SMALL_B_SAMPLED, n_samples=10)

    assert result.proposal is not None
    B = result.proposal.samples_g.shape[0]
    assert B == 4
    assert result.proposal.samples_g.shape[-2] == 10  # n_samples
    assert all(r == 'small' for r in result.routes)
    assert len(result.validation) == B


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_REAL_DATA, reason='local checkpoint + data not present')
def test_router_sample_medium_partition_routes_correctly():
    router = Router(_JOINT_CHECKPOINT, batch_size=4)
    result = router.sample(_MEDIUM_B_SAMPLED, n_samples=10)

    assert result.proposal is not None
    assert all(r == 'medium' for r in result.routes)


# ---------------------------------------------------------------------------
# Dataloader input handling
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_REAL_DATA, reason='local checkpoint + data not present')
def test_router_npz_path_wraps_in_dataloader():
    router = Router(_JOINT_CHECKPOINT, batch_size=4)
    batch = router.prepareData(_SMALL_B_SAMPLED)
    assert isinstance(batch, dict)
    assert batch['X'].dim() == 4


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_REAL_DATA, reason='local checkpoint + data not present')
def test_router_existing_dataloader_is_consumed_directly():
    loader = Dataloader(_SMALL_B_SAMPLED, batch_size=4)
    router = Router(_JOINT_CHECKPOINT, batch_size=4)
    batch = router.prepareData(loader)
    assert isinstance(batch, dict)
    assert batch['X'].dim() == 4


# ---------------------------------------------------------------------------
# New validation checks
# ---------------------------------------------------------------------------


def test_router_validates_active_group_ns_positive(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_joint_checkpoint(joint_path)
    router = Router(joint_path)

    bad = _full_batch(d=4, q=2, m=12)
    # Active group with ns=0 (mask_m says active but ns is zero)
    ns = bad['ns'].clone()
    ns[0, 5] = 0
    bad['ns'] = ns
    bad['n'] = bad['ns'][bad['mask_m']].sum().unsqueeze(0)

    with pytest.raises(ValueError, match='active groups must have ns > 0'):
        router._validateBatchFormat(bad)


def test_router_validates_inactive_group_ns_zero(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_joint_checkpoint(joint_path)
    router = Router(joint_path)

    # Build from scratch: 10 active groups + 2 padding slots
    m_active = 10
    m_pad = 12
    n_i = 10
    max_d, max_q = 8, 3
    ns = torch.zeros(1, m_pad, dtype=torch.int64)
    ns[0, :m_active] = n_i
    ns[0, m_active] = 5  # padding group with non-zero ns (should be 0)
    mask_m = torch.zeros(1, m_pad, dtype=torch.bool)
    mask_m[0, :m_active] = True  # mark only the first 10 as active
    mask_q = torch.arange(max_q).unsqueeze(0) < 2
    bad = {
        'X': torch.zeros((1, m_pad, n_i, max_d)),
        'Z': torch.zeros((1, m_pad, n_i, max_q)),
        'y': torch.zeros((1, m_pad, n_i)),
        'ns': ns,
        'm': torch.tensor([m_active]),
        'n': torch.tensor([m_active * n_i]),
        'mask_d': torch.arange(max_d).unsqueeze(0) < 4,
        'mask_q': mask_q,
        'mask_n': torch.ones((1, m_pad, n_i), dtype=torch.bool),
        'mask_m': mask_m,
        'mask_mq': mask_m.unsqueeze(-1) & mask_q.unsqueeze(-2),
        'mask_corr': mask_q.new_zeros(1, 1),
        'likelihood_family': torch.tensor([1]),
        'nu_ffx': torch.zeros(1, max_d),
        'tau_ffx': torch.ones(1, max_d) * 2.5,
        'tau_rfx': torch.ones(1, max_q) * 2.5,
        'eta_rfx': torch.zeros(1),
        'family_ffx': torch.zeros(1, dtype=torch.long),
        'family_sigma_rfx': torch.zeros(1, dtype=torch.long),
        'sd_y': torch.ones(1),
    }

    with pytest.raises(ValueError, match='inactive groups must have ns == 0'):
        router._validateBatchFormat(bad)


def test_router_validates_mask_mq_agreement(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_joint_checkpoint(joint_path)
    router = Router(joint_path)

    bad = _full_batch(d=4, q=2)
    # Corrupt mask_mq so it disagrees with mask_m & mask_q
    bad['mask_mq'] = bad['mask_mq'].clone()
    bad['mask_mq'][0, 0, 0] = not bad['mask_mq'][0, 0, 0].item()

    with pytest.raises(ValueError, match='mask_mq must equal'):
        router._validateBatchFormat(bad)


def test_router_validates_stats_non_finite(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_joint_checkpoint(joint_path)
    router = Router(joint_path)

    bad = _full_batch(d=4, q=2)
    bad['stats'] = {'beta_est': torch.full((1, 8), float('nan'))}

    with pytest.raises(ValueError, match='stats.beta_est contains non-finite'):
        router._validateBatchFormat(bad)


def test_router_warns_unstandardized_continuous_y(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)

    bad = _full_batch(d=3, q=2, m=12, family=0)
    bad['likelihood_family'] = torch.tensor([0])
    bad['sd_y'] = torch.ones(1)
    # y with large mean → unstandardized
    bad['y'] = bad['y'] + 100.0

    with pytest.warns(RuntimeWarning, match='unstandardized'):
        router._validateBatchFormat(bad)


def test_router_warns_near_constant_columns(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)

    n = 120
    near_const = np.full(n, 5.0)
    near_const[0] = 5.001  # std ≈ 0.0001, well below threshold
    preprocessed = {
        'X': np.column_stack([np.random.randn(n), near_const]),
        'y': np.random.randn(n),
        'groups': np.repeat(np.arange(12), 10),
        'columns': np.array(['x1', 'x_const']),
        'd': np.array(2),
        'n': np.array(n),
        'ns': np.full(12, 10, dtype=np.int64),
        'm': np.array(12),
        'y_type': np.array('continuous'),
    }

    with pytest.warns(RuntimeWarning, match='near-constant'):
        router.prepareData(preprocessed, formula='y ~ x1 + x_const + (1 | group)')


def test_router_warns_high_correlation_columns(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_math_joint_checkpoint(joint_path)
    router = Router(joint_path)

    n = 120
    x1 = np.random.randn(n)
    x2 = x1 + np.random.randn(n) * 0.001  # near-perfect correlation
    preprocessed = {
        'X': np.column_stack([x1, x2]),
        'y': np.random.randn(n),
        'groups': np.repeat(np.arange(12), 10),
        'columns': np.array(['x1', 'x2']),
        'd': np.array(2),
        'n': np.array(n),
        'ns': np.full(12, 10, dtype=np.int64),
        'm': np.array(12),
        'y_type': np.array('continuous'),
    }

    with pytest.warns(RuntimeWarning, match='highly correlated'):
        router.prepareData(preprocessed, formula='y ~ x1 + x2 + (1 | group)')


# ---------------------------------------------------------------------------
# Real-model inference tests (tiny Approximator, no fixture data required)
# ---------------------------------------------------------------------------


def _write_tiny_approximator_checkpoint(
    path: Path,
    *,
    max_d: int,
    max_q: int,
    likelihood_family: int = 1,
) -> None:
    from metabeta.models.approximator import Approximator
    from metabeta.utils.config import (
        ApproximatorConfig,
        PosteriorConfig,
        SummarizerConfig,
    )

    scfg = SummarizerConfig(d_model=8, d_ff=16, d_output=8, n_blocks=1, dropout=0.0)
    pcfg = PosteriorConfig(n_blocks=1)
    model_cfg = ApproximatorConfig(
        d_ffx=max_d,
        d_rfx=max_q,
        likelihood_family=likelihood_family,
        summarizer_l=scfg,
        summarizer_g=scfg,
        posterior_l=pcfg,
        posterior_g=pcfg,
        posterior_correlation=max_q >= 2,
        analytical_refinement='none',
        analytical_local_at_inference=False,
    )
    model = Approximator(model_cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            '_version': 1,
            'submodels': [
                {
                    'id': 'tiny',
                    'routing': {
                        'likelihood_family': likelihood_family,
                        'min_d': 1,
                        'max_d': max_d,
                        'min_q': 1,
                        'max_q': max_q,
                        'min_m': 5,
                        'max_m': 200,
                        'min_n': 5,
                        'max_n': 150,
                        'max_n_total': 3000,
                        'min_bg_df': 0,
                        'min_within_df': 0,
                    },
                    'model_cfg': model_cfg.to_dict(),
                    'model_state': model.state_dict(),
                },
            ],
        },
        path,
    )


def _exact_batch(
    *,
    d: int,
    q: int,
    m: int = 12,
    n_i: int = 10,
    family: int = 1,
    with_params: bool = False,
):
    """Create a batch padded to exactly (d, q) — for tiny-model inference tests."""
    ns = torch.full((1, m), n_i, dtype=torch.int64)
    mask_d = torch.ones(1, d, dtype=torch.bool)
    mask_q = torch.ones(1, q, dtype=torch.bool)
    mask_m = ns > 0
    mask_n = torch.ones((1, m, n_i), dtype=torch.bool)
    b: dict = {
        'X': torch.zeros((1, m, n_i, d)),
        'Z': torch.zeros((1, m, n_i, q)),
        'y': torch.zeros((1, m, n_i)),
        'ns': ns,
        'm': torch.tensor([m]),
        'n': torch.tensor([m * n_i]),
        'mask_d': mask_d,
        'mask_q': mask_q,
        'mask_n': mask_n,
        'mask_m': mask_m,
        'likelihood_family': torch.tensor([family]),
        'nu_ffx': torch.zeros(1, d),
        'tau_ffx': torch.ones(1, d) * 2.5,
        'tau_rfx': torch.ones(1, q) * 2.5,
        'eta_rfx': torch.zeros(1),
        'family_ffx': torch.zeros(1, dtype=torch.long),
        'family_sigma_rfx': torch.zeros(1, dtype=torch.long),
        'sd_y': torch.ones(1),
        'mask_mq': mask_m.unsqueeze(-1) & mask_q.unsqueeze(-2),
    }
    if q >= 2:
        b['mask_corr'] = torch.stack(
            [mask_q[..., i] & mask_q[..., j] for i in range(1, q) for j in range(i)], dim=-1
        )
    else:
        b['mask_corr'] = mask_q.new_zeros(1, 0)
    if with_params:
        b['ffx'] = torch.zeros(1, d)
        b['sigma_rfx'] = torch.ones(1, q) * 0.5
        b['rfx'] = torch.zeros(1, m, q)
        b['corr_rfx'] = torch.eye(q).unsqueeze(0)
    return b


def _full_batch_with_params(*, d: int, q: int, m: int = 12, n_i: int = 10, family: int = 1):
    """Like _full_batch but includes placeholder parameter tensors for log_prob."""
    b = _full_batch(d=d, q=q, m=m, n_i=n_i, family=family)
    max_d, max_q = b['X'].shape[-1], b['Z'].shape[-1]
    b['ffx'] = torch.zeros(1, max_d)
    b['sigma_rfx'] = torch.ones(1, max_q) * 0.5
    b['rfx'] = torch.zeros(1, m, max_q)
    b['corr_rfx'] = torch.eye(max_q).unsqueeze(0)
    return b


def test_router_sample_returns_valid_proposal(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_tiny_approximator_checkpoint(joint_path, max_d=4, max_q=2)
    router = Router(joint_path)

    batch = _exact_batch(d=4, q=2)
    result = router.sample(batch, n_samples=5)

    assert result.proposal is not None
    assert result.routes == ['tiny']
    assert len(result.validation) == 1
    assert result.proposal.samples_g.shape[0] == 1
    assert result.proposal.n_samples == 5
    assert result.diagnostics is None


def test_router_sample_with_precomputed_stats_runs(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_tiny_approximator_checkpoint(joint_path, max_d=4, max_q=2)
    router = Router(joint_path)

    batch = _exact_batch(d=4, q=2)
    batch['stats'] = {
        'beta_est': torch.zeros(1, 4),
        'sigma_rfx_est': torch.ones(1, 2) * 0.5,
    }
    result = router.sample(batch, n_samples=3)

    assert result.proposal is not None
    assert result.proposal.n_samples == 3


def test_router_sample_returns_diagnostics_when_requested(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_tiny_approximator_checkpoint(joint_path, max_d=4, max_q=2)
    router = Router(joint_path)

    batch = _exact_batch(d=4, q=2)
    result = router.sample(batch, n_samples=10, diagnostics=True)

    assert result.diagnostics is not None
    assert 'ppc_nll' in result.diagnostics
    assert 'param_summary' in result.diagnostics
    assert 'ffx' in result.diagnostics['param_summary']
    assert 'sigma_rfx' in result.diagnostics['param_summary']
    assert result.diagnostics['ppc_nll'].shape == (1,)


def test_router_log_prob_accepts_valid_parameters(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_tiny_approximator_checkpoint(joint_path, max_d=4, max_q=2)
    router = Router(joint_path)

    batch = _exact_batch(d=4, q=2, with_params=True)
    result = router.log_prob(batch)

    assert result.proposal is None
    assert result.routes == ['tiny']
    assert result.log_probs is not None
    assert 'global' in result.log_probs
    assert 'local' in result.log_probs
    assert result.log_probs['global'].shape[0] == 1


def test_router_log_prob_rejects_missing_parameter_keys(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_tiny_approximator_checkpoint(joint_path, max_d=4, max_q=2)
    router = Router(joint_path)

    batch = _exact_batch(d=4, q=2)  # no ffx/sigma_rfx/rfx keys

    with pytest.raises(KeyError, match='missing parameter keys'):
        router.log_prob(batch)


def test_router_log_prob_rejects_malformed_ffx_shape(tmp_path: Path):
    joint_path = tmp_path / 'joint.pt'
    _write_tiny_approximator_checkpoint(joint_path, max_d=4, max_q=2)
    router = Router(joint_path)

    batch = _exact_batch(d=4, q=2, with_params=True)
    batch['ffx'] = torch.zeros(1, 3)  # wrong d (3 instead of 4)

    with pytest.raises(ValueError, match='ffx must have shape'):
        router.log_prob(batch)
