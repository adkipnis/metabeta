"""Tests for metabeta/datasets/preprocess.py."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metabeta.datasets.preprocessor import (
    DataPreprocessor,
    _corrFilter,
    _dropHeavyColumns,
    coerceTargetToNumeric,
    detectGroupCandidates,
    detectMulticlassY,
    detectYType,
    lumpCategories,
    preprocess,
)
from metabeta.utils.preprocessing import NumericTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n=100, seed=0) -> pd.DataFrame:
    """Minimal well-behaved DataFrame with numeric + categorical predictors."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            'x1': rng.normal(size=n),
            'x2': rng.uniform(0, 10, size=n),
            'cat': rng.choice(['a', 'b', 'c'], size=n),
            'binary': rng.integers(0, 2, size=n).astype(float),
            'y': rng.normal(size=n),
        }
    )
    return df


def _make_grouped_df(n=200, m=10, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(m), n // m)
    df = pd.DataFrame(
        {
            'x1': rng.normal(size=n),
            'group': group,
            'y': rng.normal(size=n),
        }
    )
    return df


# ---------------------------------------------------------------------------
# coerceTargetToNumeric
# ---------------------------------------------------------------------------


def test_coerce_numeric_float():
    y = pd.Series([1.0, 2.5, 3.0])
    arr, is_bin = coerceTargetToNumeric(y)
    assert arr.dtype == float
    assert not is_bin


def test_coerce_binary_01():
    y = pd.Series([0.0, 1.0, 0.0, 1.0])
    arr, is_bin = coerceTargetToNumeric(y)
    assert is_bin


def test_coerce_binary_yes_no():
    y = pd.Series(['yes', 'no', 'yes'])
    arr, is_bin = coerceTargetToNumeric(y)
    assert is_bin
    assert set(arr.tolist()) == {0.0, 1.0}


def test_coerce_empty_raises():
    with pytest.raises(ValueError):
        coerceTargetToNumeric(pd.Series([], dtype=float))


# ---------------------------------------------------------------------------
# detectYType
# ---------------------------------------------------------------------------


def test_detect_y_type_binary():
    y = np.array([0.0, 1.0, 0.0])
    assert detectYType(y, y_is_binary=True) == 'binary'


def test_detect_y_type_count():
    y = np.array([0.0, 3.0, 7.0, 12.0])
    assert detectYType(y, y_is_binary=False) == 'count'


def test_detect_y_type_offset_integers_are_continuous():
    # Integer-valued y whose minimum is > 0 should be treated as continuous
    # (discretised measurements such as ratings, scores, diameters), not count.
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # ratings
    assert detectYType(y, y_is_binary=False) == 'continuous'
    y = np.array([18.0, 20.0, 23.0, 27.0])  # diameters
    assert detectYType(y, y_is_binary=False) == 'continuous'


def test_detect_y_type_continuous():
    y = np.array([1.5, -0.3, 2.7])
    assert detectYType(y, y_is_binary=False) == 'continuous'


def test_detect_y_type_multiclass():
    y = np.array([0.0, 1.0, 2.0, 0.0])
    assert detectYType(y, y_is_binary=False, is_multiclass=True) == 'multiclass'


# ---------------------------------------------------------------------------
# detectMulticlassY
# ---------------------------------------------------------------------------


def test_detect_multiclass_string_labels():
    y = pd.Series(['class_0', 'class_1', 'class_2', 'class_0'])
    assert detectMulticlassY(y)


def test_detect_multiclass_arbitrary_strings():
    y = pd.Series(['cat', 'dog', 'bird', 'cat'])
    assert detectMulticlassY(y)


def test_detect_multiclass_binary_not_multiclass():
    y = pd.Series(['yes', 'no', 'yes'])
    assert not detectMulticlassY(y)


def test_detect_multiclass_numeric_not_multiclass():
    y = pd.Series([0, 1, 2, 0])
    assert not detectMulticlassY(y)


def test_detect_multiclass_numeric_strings_not_multiclass():
    # "0"/"1"/"2" as strings are parseable → not detected as multiclass
    y = pd.Series(['0', '1', '2', '0'])
    assert not detectMulticlassY(y)


# ---------------------------------------------------------------------------
# _dropHeavyColumns
# ---------------------------------------------------------------------------


def test_drop_heavy_column():
    df = pd.DataFrame({'a': np.nan * np.ones(10), 'b': np.ones(10), 'c': np.ones(10)})
    result = _dropHeavyColumns(df, col_miss_threshold=0.25)
    assert 'a' not in result.columns


def test_drop_heavy_column_below_threshold_kept():
    # 25% missing is exactly at threshold (not above) — column is retained
    df = pd.DataFrame({'a': [1.0, np.nan, 1.0, 1.0], 'b': np.ones(4)})
    result = _dropHeavyColumns(df, col_miss_threshold=0.25)
    assert 'a' in result.columns
    assert result.isnull().any().any()  # missing value remains; imputation handles it


# ---------------------------------------------------------------------------
# NumericTransformer — conditional log1p
# ---------------------------------------------------------------------------


def test_log1p_applied_to_skewed_count():
    rng = np.random.default_rng(0)
    # highly right-skewed count column: many zeros, a few large values
    skewed = np.concatenate([np.zeros(80), rng.integers(10, 500, size=20)]).astype(float)
    x = skewed.reshape(-1, 1)
    nt = NumericTransformer()
    nt.fit(x)
    assert nt.is_log1p_[0], 'expected log1p flag for skewed count column'


def test_log1p_not_applied_to_low_skew_count():
    # uniform count column: low skew, should NOT be log1p-transformed
    x = np.arange(100, dtype=float).reshape(-1, 1)
    nt = NumericTransformer()
    nt.fit(x)
    assert not nt.is_log1p_[0], 'expected no log1p for low-skew count column'


# ---------------------------------------------------------------------------
# lumpCategories
# ---------------------------------------------------------------------------


def test_lump_adds_other():
    s = pd.Series(['a'] * 40 + ['b'] * 40 + ['c'] * 10 + ['d'] * 5 + ['e'] * 5)
    s.name = 'col'
    lumped, lump_levels, has_other = lumpCategories(s, max_categories=2, min_prevalence=0.05)
    assert has_other
    assert 'other' in lumped.values
    assert {'c', 'd', 'e'} == lump_levels


def test_lump_drops_when_pool_too_small():
    s = pd.Series(['a'] * 49 + ['b'] * 49 + ['c'] * 2)
    s.name = 'col'
    lumped, lump_levels, has_other = lumpCategories(s, max_categories=2, min_prevalence=0.05)
    assert not has_other
    assert lumped.isnull().sum() == 2  # 'c' rows become NaN


def test_lump_no_op_when_within_limit():
    s = pd.Series(['a', 'b', 'c'] * 10)
    s.name = 'col'
    lumped, lump_levels, has_other = lumpCategories(s, max_categories=5, min_prevalence=0.05)
    assert not lump_levels
    assert not has_other
    assert lumped.equals(s)


# ---------------------------------------------------------------------------
# _corrFilter
# ---------------------------------------------------------------------------


def test_corr_filter_drops_redundant_column():
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    df = pd.DataFrame({'a': x, 'b': x + 1e-10 * rng.normal(size=100), 'c': rng.normal(size=100)})
    result, dropped = _corrFilter(df, threshold=0.95)
    assert len(dropped) == 1
    assert dropped.issubset({'a', 'b'})
    assert 'c' in result.columns


def test_corr_filter_no_drop_when_below_threshold():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'a': rng.normal(size=100), 'b': rng.normal(size=100)})
    result, dropped = _corrFilter(df, threshold=0.95)
    assert len(dropped) == 0
    assert result.shape == df.shape


# ---------------------------------------------------------------------------
# detectGroupCandidates
# ---------------------------------------------------------------------------


def test_detect_group_candidates_finds_id_column():
    rng = np.random.default_rng(0)
    n = 200
    group = np.repeat(np.arange(20), 10)
    df = pd.DataFrame({'group_id': group, 'x': rng.normal(size=n)})
    candidates = detectGroupCandidates(df)
    assert any(c.name == 'group_id' for c in candidates)


def test_detect_group_candidates_ignores_blacklist():
    rng = np.random.default_rng(0)
    n = 200
    group = np.repeat(np.arange(20), 10)
    # 'year' is in BLACKLIST
    df = pd.DataFrame({'year': group, 'x': rng.normal(size=n)})
    candidates = detectGroupCandidates(df)
    assert all(c.name != 'year' for c in candidates)


# ---------------------------------------------------------------------------
# Imputation — group-aware
# ---------------------------------------------------------------------------


def test_imputation_fills_missing_no_row_drop():
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(size=n)
    x[::10] = np.nan  # 10% missing
    df = pd.DataFrame({'x': x, 'y': rng.normal(size=n)})
    result = DataPreprocessor().fit_transform(df)
    # no rows dropped despite missing values
    assert result['n'] == n
    assert np.isfinite(result['X']).all()


def test_imputation_group_aware():
    rng = np.random.default_rng(0)
    n, m = 100, 5
    groups = np.repeat(np.arange(m), n // m)
    # group 0 has x values ~10, group 1 has x ~0 — within-group imputation
    # should fill group-0 missing values near 10, not the global mean
    x = np.where(groups == 0, 10.0, 0.0) + rng.normal(scale=0.1, size=n)
    x[0] = np.nan  # missing in group 0
    df = pd.DataFrame({'x': x, 'group': groups, 'y': rng.normal(size=n)})
    prep = DataPreprocessor(group_name='group')
    result = prep.fit_transform(df)
    # imputed value for row 0 (group 0) should come from within-group median (~10),
    # not the global median (~5); after z-standardisation it should be clearly
    # positive (group-0 values are well above the global mean)
    col_idx = result['columns'].tolist().index('x')
    row_0_in_result = result['X'][0, col_idx]  # group is sorted; group 0 is first
    assert row_0_in_result > 0, 'group-aware imputation should yield above-mean value for group 0'


# ---------------------------------------------------------------------------
# High-cardinality categorical routing
# ---------------------------------------------------------------------------


def test_hi_card_cat_routed_as_random_effect():
    rng = np.random.default_rng(0)
    n, m = 200, 20  # 20 levels > max_categories=10 → should become random effect
    group = np.repeat(np.arange(m), n // m).astype(str)
    df = pd.DataFrame({
        'school': group,
        'x': rng.normal(size=n),
        'y': rng.normal(size=n),
    })
    result = DataPreprocessor().fit_transform(df)
    # 'school' should be the grouping variable, not a dummy column
    assert result['groups'] is not None
    assert result['m'] == m
    assert not any('school' in c for c in result['columns'].tolist())


# ---------------------------------------------------------------------------
# DataPreprocessor — round-trip
# ---------------------------------------------------------------------------


def test_preprocessor_fit_transform_output_schema():
    df = _make_df()
    result = DataPreprocessor().fit_transform(df)
    assert 'X' in result
    assert 'y' in result
    assert 'columns' in result
    assert 'd' in result
    assert result['d'] == result['X'].shape[1] + 1
    assert result['n'] == result['X'].shape[0]
    assert result['y_type'] == 'continuous'


def test_preprocessor_binary_not_standardised():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({'bin': rng.integers(0, 2, n).astype(float), 'y': rng.normal(size=n)})
    result = DataPreprocessor().fit_transform(df)
    col_idx = result['columns'].tolist().index('bin')
    col = result['X'][:, col_idx]
    assert set(np.unique(col)).issubset({0.0, 1.0})


def test_preprocessor_count_y_not_standardised():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({'x': rng.normal(size=n), 'y': rng.integers(0, 20, size=n).astype(float)})
    result = DataPreprocessor().fit_transform(df)
    assert result['y_type'] == 'count'
    assert result['y'].dtype == np.int64


def test_preprocessor_multiclass_y():
    rng = np.random.default_rng(0)
    n = 120
    classes = np.tile(['class_0', 'class_1', 'class_2', 'class_3'], n // 4)
    df = pd.DataFrame({'x': rng.normal(size=n), 'y': classes})
    result = DataPreprocessor().fit_transform(df)
    assert result['y_type'] == 'multiclass'
    assert result['y'].dtype == np.int64
    assert set(result['y'].tolist()) == {0, 1, 2, 3}


def test_preprocessor_treatment_contrasts():
    rng = np.random.default_rng(0)
    n = 120
    cat = np.tile(['a', 'b', 'c'], n // 3)
    df = pd.DataFrame({'cat': cat, 'y': rng.normal(size=n)})
    result = DataPreprocessor().fit_transform(df)
    # 3 levels → 2 dummies (drop='first')
    assert result['X'].shape[1] == 2


def test_preprocessor_grouped_output():
    df = _make_grouped_df()
    result = DataPreprocessor(group_name='group').fit_transform(df)
    assert result['groups'] is not None
    assert result['m'] == 10
    assert result['ns'] is not None
    assert result['ns'].sum() == result['n']


def test_preprocessor_corr_filter_applied():
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    df = pd.DataFrame({'a': x, 'b': x * 1.001, 'y': rng.normal(size=100)})
    result = DataPreprocessor().fit_transform(df)
    # one of {a, b} should be dropped
    assert result['X'].shape[1] == 1


# ---------------------------------------------------------------------------
# DataPreprocessor — statefulness
# ---------------------------------------------------------------------------


def test_preprocessor_transform_same_column_order():
    rng = np.random.default_rng(0)
    n = 150
    df_train = pd.DataFrame(
        {
            'x1': rng.normal(size=n),
            'cat': rng.choice(['a', 'b', 'c'], size=n),
            'y': rng.normal(size=n),
        }
    )
    df_test = df_train.sample(50, random_state=1).reset_index(drop=True)

    prep = DataPreprocessor()
    prep.fit(df_train)
    out_train = prep.transform(df_train)
    out_test = prep.transform(df_test)

    assert list(out_test['columns']) == list(out_train['columns'])


def test_preprocessor_transform_handle_unknown_category():
    rng = np.random.default_rng(0)
    n = 100
    df_train = pd.DataFrame({'cat': ['a', 'b'] * 50, 'y': rng.normal(size=n)})
    df_test = pd.DataFrame({'cat': ['a', 'z'], 'y': rng.normal(size=2)})

    prep = DataPreprocessor()
    prep.fit(df_train)
    # 'z' is unseen — handle_unknown='ignore' should zero it out, not raise
    out = prep.transform(df_test)
    assert out['X'].shape[0] == 2


# ---------------------------------------------------------------------------
# DataPreprocessor — save / load
# ---------------------------------------------------------------------------


def test_preprocessor_save_load_roundtrip():
    df = _make_df()
    prep = DataPreprocessor()
    out_before = prep.fit_transform(df)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'prep.pkl'
        prep.save(path)
        prep2 = DataPreprocessor.load(path)

    out_after = prep2.transform(df)
    np.testing.assert_array_equal(out_before['columns'], out_after['columns'])
    np.testing.assert_allclose(out_before['X'], out_after['X'], rtol=1e-6)


# ---------------------------------------------------------------------------
# preprocess shim
# ---------------------------------------------------------------------------


def test_preprocess_shim_matches_preprocessor():
    df = _make_df(seed=42)
    out_shim = preprocess(df.copy())
    out_class = DataPreprocessor().fit_transform(df.copy())
    np.testing.assert_array_equal(out_shim['columns'], out_class['columns'])
    np.testing.assert_allclose(out_shim['X'], out_class['X'], rtol=1e-6)
