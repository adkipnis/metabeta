import logging
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from metabeta.utils.preprocessing import (
    NumericTransformer,
    checkCountLike,
)

logger = logging.getLogger(__name__)

BLACKLIST = 'year age height size n_ num_ number max min attempts begin end name'.split(' ')
MAX_GROUP_CANDIDATES = 3


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupCandidate:
    name: str
    n_groups: int
    frac_unique: float
    avg_obs_per_group: float
    min_obs_per_group: int
    max_obs_per_group: int
    imbalance: float
    score: float


def categorical(df: pd.DataFrame) -> pd.Index:
    return df.select_dtypes(include=['object', 'category', 'string']).columns


def numerical(df: pd.DataFrame) -> pd.Index:
    return df.select_dtypes(include=[np.number]).columns


# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------


def _sentinelReplace(df: pd.DataFrame, sentinels: list[float]) -> pd.DataFrame:
    for s in sentinels:
        df = df.replace(s, np.nan)
    return df


def _dropHeavyColumns(df: pd.DataFrame, col_miss_threshold: float = 0.25) -> pd.DataFrame:
    """Drop columns whose fraction of missing values exceeds col_miss_threshold."""
    miss_frac = df.isnull().mean()
    heavy = miss_frac[miss_frac > col_miss_threshold].index.tolist()
    if heavy:
        logger.warning(f'Dropping {heavy} (>{col_miss_threshold * 100:.0f}% missing).')
        df = df.drop(columns=heavy)
    return df


def _computeWinsorBounds(
    df: pd.DataFrame, threshold: float = 4.0
) -> dict[str, tuple[float, float]]:
    """Compute per-column winsorization bounds (mean ± threshold * std) on numeric columns."""
    bounds: dict[str, tuple[float, float]] = {}
    for col in numerical(df).tolist():
        vals = df[col].dropna().to_numpy(dtype=float)
        if len(vals) < 2:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if std < 1e-12:
            continue
        bounds[col] = (mean - threshold * std, mean + threshold * std)
    return bounds


def _applyWinsor(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Clip numeric columns to pre-computed winsorization bounds."""
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def _computeImputeStats(
    df: pd.DataFrame,
    groups: np.ndarray | None,
    numeric_cols: list[str],
    cat_cols: list[str],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Compute group-aware imputation statistics (median for numeric, mode for categorical).

    For each column, stores a global statistic and, when groups are available,
    per-group statistics. Missing values in a group fall back to the global value.
    """
    num_stats: dict[str, dict] = {}
    for col in numeric_cols:
        global_val = float(df[col].median()) if df[col].notna().any() else 0.0
        by_group: dict[int, float] = {}
        if groups is not None:
            for g in np.unique(groups):
                vals = df.loc[groups == g, col].dropna()
                by_group[int(g)] = float(vals.median()) if len(vals) > 0 else global_val
        num_stats[col] = {'global': global_val, 'by_group': by_group}

    cat_stats: dict[str, dict] = {}
    for col in cat_cols:
        mode_s = df[col].mode()
        global_val_c = str(mode_s.iloc[0]) if len(mode_s) > 0 else ''
        by_group_c: dict[int, str] = {}
        if groups is not None:
            for g in np.unique(groups):
                mode_g = df.loc[groups == g, col].mode()
                by_group_c[int(g)] = str(mode_g.iloc[0]) if len(mode_g) > 0 else global_val_c
        cat_stats[col] = {'global': global_val_c, 'by_group': by_group_c}

    return num_stats, cat_stats


def _applyImputation(
    df: pd.DataFrame,
    groups: np.ndarray | None,
    num_stats: dict[str, dict],
    cat_stats: dict[str, dict],
) -> pd.DataFrame:
    """Fill missing values using stored group-aware statistics."""
    for col, stats in num_stats.items():
        if col not in df.columns:
            continue
        missing_idx = df.index[df[col].isnull()].to_numpy()
        if len(missing_idx) == 0:
            continue
        if groups is not None and stats['by_group']:
            fill = [stats['by_group'].get(int(groups[i]), stats['global']) for i in missing_idx]
        else:
            fill = [stats['global']] * len(missing_idx)
        df.loc[missing_idx, col] = fill

    for col, stats in cat_stats.items():
        if col not in df.columns:
            continue
        missing_idx = df.index[df[col].isnull()].to_numpy()
        if len(missing_idx) == 0:
            continue
        if groups is not None and stats['by_group']:
            fill = [stats['by_group'].get(int(groups[i]), stats['global']) for i in missing_idx]
        else:
            fill = [stats['global']] * len(missing_idx)
        df.loc[missing_idx, col] = fill

    return df


# ---------------------------------------------------------------------------
# Target coercion & type detection
# ---------------------------------------------------------------------------


def coerceTargetToNumeric(y: pd.Series) -> tuple[np.ndarray, bool]:
    if len(y) == 0:
        raise ValueError('Target y is empty.')

    if pd.api.types.is_numeric_dtype(y):
        arr = y.to_numpy(dtype=float)
        uniq = np.unique(arr[np.isfinite(arr)])
        is_binary = len(uniq) == 2 and set(uniq.tolist()).issubset({0.0, 1.0})
        return arr, is_binary

    y_str = y.astype('string').str.strip()
    non_missing = y_str.dropna()
    unique = sorted(non_missing.unique().tolist())

    if len(unique) == 2:
        lower = [u.lower() for u in unique]
        if set(lower) == {'n', 'y'}:
            mapper = {unique[lower.index('n')]: 0.0, unique[lower.index('y')]: 1.0}
        elif set(lower) == {'no', 'yes'}:
            mapper = {unique[lower.index('no')]: 0.0, unique[lower.index('yes')]: 1.0}
        elif set(lower) == {'false', 'true'}:
            mapper = {unique[lower.index('false')]: 0.0, unique[lower.index('true')]: 1.0}
        else:
            mapper = {unique[0]: 0.0, unique[1]: 1.0}
        mapped = y_str.map(mapper).to_numpy(dtype=float)
        return mapped, True

    parsed = pd.to_numeric(y_str, errors='coerce')
    if parsed.notna().sum() == non_missing.shape[0]:
        arr = parsed.to_numpy(dtype=float)
        uniq = np.unique(arr[np.isfinite(arr)])
        is_binary = len(uniq) == 2 and set(uniq.tolist()).issubset({0.0, 1.0})
        return arr, is_binary

    sample = ', '.join(map(str, unique[:8]))
    raise ValueError(f'Cannot convert target y to numeric. dtype={y.dtype}, sample=[{sample}]')


def detectYType(y: np.ndarray, y_is_binary: bool, is_multiclass: bool = False) -> str:
    """Return 'binary', 'count', 'continuous', or 'multiclass'."""
    if y_is_binary:
        return 'binary'
    if is_multiclass:
        return 'multiclass'
    finite = y[np.isfinite(y)]
    if len(finite) == 0:
        return 'continuous'
    is_count = checkCountLike(finite.reshape(-1, 1), axis=0)[0]
    # Integer arrays whose minimum is above 0 are more likely to be discretised
    # measurements (scores, ratings, diameters) than genuine event counts; only
    # classify as 'count' when values start at zero.
    if is_count and finite.min() > 0:
        is_count = False
    return 'count' if is_count else 'continuous'


def detectMulticlassY(y: pd.Series) -> bool:
    """Return True if y is a non-numeric string column with more than 2 unique values.

    Used to identify multiclass classification targets (e.g. PMLB datasets where
    class labels are stored as ``"class_0"``, ``"class_1"``, ... strings).
    """
    if not (
        pd.api.types.is_object_dtype(y)
        or pd.api.types.is_string_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
    ):
        return False
    y_str = y.astype('string').str.strip()
    non_missing = y_str.dropna()
    if non_missing.nunique() <= 2:
        return False
    parsed = pd.to_numeric(non_missing, errors='coerce')
    return parsed.notna().sum() < len(non_missing)


# ---------------------------------------------------------------------------
# Grouping variable detection
# ---------------------------------------------------------------------------


def detectGroupCandidates(
    df: pd.DataFrame,
    min_groups: int = 5,
    max_frac_unique: float = 0.5,
    min_obs_per_group: int = 2,
    max_groups_cap: int = 200,
    min_frac_singleton: float = 0.05,
    max_frac_singleton: float = 0.35,
) -> list[GroupCandidate]:
    assert 0.0 < max_frac_unique < 1.0

    if len(df) == 0:
        return []

    eligible: list[str] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            vals = s.dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            if not np.allclose(vals, np.round(vals), atol=1e-12, rtol=0):
                continue
        elif not (
            pd.api.types.is_integer_dtype(s)
            or pd.api.types.is_bool_dtype(s)
            or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
        ):
            continue
        eligible.append(col)

    if not eligible:
        return []

    _bl = set(BLACKLIST)
    eligible = [
        col
        for col in eligible
        if not any(tok in _bl for tok in re.split(r'[\s_]+', col.lower()))
    ]
    if not eligible:
        return []

    n = len(df)
    max_groups = min(max_groups_cap, int(max_frac_unique * n))
    candidates: list[GroupCandidate] = []
    for col in eligible:
        counts = df[col].value_counts(dropna=True)
        if len(counts) == 0:
            continue

        n_groups = int(len(counts))
        if n_groups < min_groups or n_groups > max_groups:
            continue

        min_count = int(counts.min())
        if min_count < min_obs_per_group:
            continue

        max_count = int(counts.max())
        avg_count = float(counts.mean())
        frac_unique = float(n_groups / n)
        imbalance = float(max_count / max(min_count, 1))
        frac_singleton = float(np.mean(counts == 1))

        score = 0.0
        score -= abs(np.log10(max(avg_count, 1e-12)) - np.log10(20.0))
        if 8 <= n_groups <= 150:
            score += 0.5
        score -= 0.1 * max(0.0, np.log10(max(imbalance, 1.0)))
        score -= 1.0 * abs(frac_singleton - 0.15)

        lower = col.lower()
        if re.search(r'group|subject|school|cluster|site|center|class|id', lower):
            score += 0.5
        if re.search(r'train|test|valid|fold|split', lower):
            score -= 1.0
        if frac_singleton < min_frac_singleton or frac_singleton > max_frac_singleton:
            score -= 0.75

        candidates.append(
            GroupCandidate(
                name=col,
                n_groups=n_groups,
                frac_unique=frac_unique,
                avg_obs_per_group=avg_count,
                min_obs_per_group=min_count,
                max_obs_per_group=max_count,
                imbalance=imbalance,
                score=score,
            )
        )

    return sorted(candidates, key=lambda c: (-c.score, c.name))


# ---------------------------------------------------------------------------
# Correlation filter
# ---------------------------------------------------------------------------


def _corrFilter(
    df: pd.DataFrame, threshold: float = 0.95
) -> tuple[pd.DataFrame, set[str]]:
    """Drop one column from each highly correlated pair (|r| > threshold).

    Greedy: at each step, remove the column with the higher mean absolute
    correlation to all remaining columns.
    """
    num_cols = numerical(df).tolist()
    if len(num_cols) < 2:
        return df, set()

    corr = df[num_cols].corr().abs()
    corr_vals = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_vals, 0.0)
    corr = pd.DataFrame(corr_vals, index=corr.index, columns=corr.columns)

    active = list(num_cols)
    dropped: set[str] = set()

    while True:
        sub = corr.loc[active, active]
        upper_vals = sub.values[np.triu(np.ones(sub.shape, dtype=bool), k=1)]
        if len(upper_vals) == 0 or upper_vals.max() <= threshold:
            break
        upper = sub.where(np.triu(np.ones(sub.shape, dtype=bool), k=1))
        j = upper.max(axis=0).idxmax()
        i = upper[j].idxmax()
        others = [c for c in active if c != i and c != j]
        mean_i = sub.loc[i, others].mean() if others else 0.0
        mean_j = sub.loc[j, others].mean() if others else 0.0
        drop = i if mean_i >= mean_j else j
        active.remove(drop)
        dropped.add(drop)

    if dropped:
        logger.warning(f'Dropping correlated columns {sorted(dropped)}.')
    return df.drop(columns=list(dropped)), dropped


# ---------------------------------------------------------------------------
# Categorical lumping  (fct_lump_n style)
# ---------------------------------------------------------------------------


def lumpCategories(
    series: pd.Series,
    max_categories: int = 10,
    min_prevalence: float = 0.05,
) -> tuple[pd.Series, set, bool]:
    """Pool rare levels into 'other' when a column has > max_categories levels.

    Returns (lumped_series, lump_levels, has_other).
    - lump_levels: original level names that were pooled.
    - has_other: True if the pooled mass >= min_prevalence ('other' was added);
      False if the rare observations are set to NaN (to be dropped).
    """
    # Categorical dtype doesn't allow assigning new values; convert to object
    if hasattr(series, 'cat'):
        series = series.astype(object)

    counts = series.value_counts(dropna=True)
    if len(counts) <= max_categories:
        return series, set(), False

    keep = set(counts.index[:max_categories])
    rare_mask = ~series.isin(keep) & series.notna()
    lump_levels = set(series[rare_mask].unique())
    lump_frac = rare_mask.mean()

    series = series.copy()
    if lump_frac >= min_prevalence:
        series[rare_mask] = 'other'
        logger.warning(
            f'Column "{series.name}": pooled {len(lump_levels)} rare levels '
            f'({lump_frac * 100:.1f}%) into "other".'
        )
        return series, lump_levels, True
    else:
        series[rare_mask] = np.nan
        logger.warning(
            f'Column "{series.name}": dropped {len(lump_levels)} rare levels '
            f'({lump_frac * 100:.1f}%); pool too small for "other".'
        )
        return series, lump_levels, False


# ---------------------------------------------------------------------------
# DataPreprocessor
# ---------------------------------------------------------------------------


class DataPreprocessor:
    """Stateful preprocessor for LMM/GLMM data.

    Call ``fit_transform`` (simulation path) to fit all statistics and return
    cleaned, encoded data in one pass.  Call ``fit`` then ``transform``
    (deployment path) to replay the same pipeline on new data without re-fitting.

    Missing values are handled consistently in both paths via group-aware
    median/mode imputation — there is no listwise deletion.  Extreme numeric
    values are winsorized (not deleted) so that no rows are ever removed after
    the initial heavy-column drop and non-finite-y filter.

    The output dict schema matches the ``.npz`` files expected by the emulator:
    ``X`` (n x d-1, no intercept), ``y``, ``groups``, ``columns``, ``d``,
    ``n``, ``ns``, ``m``, ``y_type``.
    """

    _version = '1'

    def __init__(
        self,
        group_name: str = '',
        col_miss_threshold: float = 0.25,
        constant_threshold: float = 0.95,
        outlier_threshold: float = 4.0,
        corr_threshold: float = 0.95,
        max_categories: int = 10,
        min_prevalence: float = 0.05,
        sentinels: list[float] | None = None,
    ):
        self.group_name = group_name
        self.col_miss_threshold = col_miss_threshold
        self.constant_threshold = constant_threshold
        self.outlier_threshold = outlier_threshold
        self.corr_threshold = corr_threshold
        self.max_categories = max_categories
        self.min_prevalence = min_prevalence
        self.sentinels: list[float] = sentinels if sentinels is not None else [-999]
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        self._fit_core(df)
        return self

    def fit_transform(self, df: pd.DataFrame) -> dict:
        return self._fit_core(df)

    def transform(self, df: pd.DataFrame) -> dict:
        """Apply the fitted pipeline to new data (deployment path).

        No rows are ever dropped. Missing values are filled using the
        group-aware statistics fitted during ``fit``/``fit_transform``.
        Winsorization bounds are applied identically to the training path.
        """
        if not self._fitted:
            raise RuntimeError('DataPreprocessor must be fitted before transform.')
        df = df.copy()
        df.columns = df.columns.str.lower()
        df = df.reset_index(drop=True)
        df = _sentinelReplace(df, self.sentinels)

        # --- extract y if present ---
        y: np.ndarray | None = None
        if 'y' in df.columns:
            y_raw = df.pop('y')
            if self._y_type == 'multiclass':
                codes, _ = pd.factorize(y_raw)
                y = codes.astype(float)
            else:
                y, _ = coerceTargetToNumeric(y_raw)

        # --- group extraction ---
        groups: np.ndarray | None = None
        group_name = self._group_name_fitted
        if group_name and group_name in df.columns:
            df = df.sort_values(by=group_name)
            sort_idx = df.index.values
            if y is not None and len(y) == len(df):
                y = y[sort_idx]
            df = df.reset_index(drop=True)
            groups_raw = df.pop(group_name)
            groups, _ = pd.factorize(groups_raw)

        df = df.reset_index(drop=True)

        # --- imputation (consistent with training path) ---
        df = _applyImputation(df, groups, self._num_impute_stats, self._cat_impute_stats)

        # --- apply lump maps ---
        for col, lump_levels in self._lump_maps.items():
            if col not in df.columns or not lump_levels:
                continue
            mask = df[col].isin(lump_levels)
            if self._lump_has_other.get(col, False):
                df[col] = df[col].where(~mask, 'other')
            else:
                df[col] = df[col].where(~mask, np.nan)

        # --- drop fitted-dropped columns ---
        drop_present = [c for c in self._dropped_cols if c in df.columns]
        if drop_present:
            df = df.drop(columns=drop_present)

        # --- winsorize (consistent with training path) ---
        df = _applyWinsor(df, self._winsor_bounds)

        # --- numeric transform ---
        x_parts: list[np.ndarray] = []
        if self._numeric_cols and self._num_transformer is not None:
            cols_present = [c for c in self._numeric_cols if c in df.columns]
            x_num = df[cols_present].to_numpy(dtype=float)
            x_num = self._num_transformer.transform(x_num)
            x_parts.append(x_num)

        # --- categorical encode ---
        if self._categorical_cols and self._encoder is not None:
            cols_present = [c for c in self._categorical_cols if c in df.columns]
            x_cat = self._encoder.transform(df[cols_present]).astype(float)
            x_parts.append(x_cat)

        X = np.concatenate(x_parts, axis=1) if x_parts else np.empty((len(df), 0))

        # --- standardise y ---
        if y is not None:
            if self._y_type == 'continuous':
                y = (y - self._y_mean) / self._y_std
            elif self._y_type in ('count', 'multiclass'):
                y = y.astype(np.int64)

        # --- group summary ---
        ns: np.ndarray | None = None
        m: int | None = None
        if groups is not None:
            _, ns = np.unique(groups, return_counts=True)
            m = int(len(ns))

        n, d_minus_1 = X.shape
        return {
            'X': X.astype(np.float64),
            'y': y,
            'groups': groups,
            'columns': np.array(self._output_columns, dtype=str),
            'd': d_minus_1 + 1,
            'n': n,
            'ns': ns,
            'm': m,
            'y_type': self._y_type,
        }

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> 'DataPreprocessor':
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise ValueError(f'Expected DataPreprocessor, got {type(obj)}')
        return obj

    # ------------------------------------------------------------------
    # Core fit logic (single pass)
    # ------------------------------------------------------------------

    def _fit_core(self, df: pd.DataFrame) -> dict:
        """Fit all statistics and transform training data in one pass.

        Sets all ``_fitted`` state on self. Returns the output dict.
        """
        df = df.copy()
        df.columns = df.columns.str.lower()
        df = df.reset_index(drop=True)
        df = _sentinelReplace(df, self.sentinels)

        # --- pop y so it doesn't influence column-level decisions ---
        if 'y' not in df.columns:
            raise ValueError("DataFrame must contain a 'y' column.")
        y_col = df.pop('y')

        # --- drop heavily-missing columns (only columns are removed here) ---
        df = _dropHeavyColumns(df, self.col_miss_threshold)

        # --- coerce and detect y type ---
        # When the entire target column is missing (e.g. UCI datasets with no
        # designated target), skip y processing and keep all rows.  The dataset
        # can still serve as a design-matrix source for ds_type='sampled'.
        y_all_missing = y_col.isna().all()
        if y_all_missing:
            logger.warning('Target y is entirely missing; dataset will be saved without y.')
            self._y_type = 'unobserved'
            self._y_mean = 0.0
            self._y_std = 1.0
            y = None
        else:
            y_is_multiclass = detectMulticlassY(y_col)
            if y_is_multiclass:
                codes, _ = pd.factorize(y_col)
                y = codes.astype(float)
                y_is_binary = False
            else:
                y, y_is_binary = coerceTargetToNumeric(y_col)
            self._y_type = detectYType(y, y_is_binary, y_is_multiclass)

            # remove rows with non-finite y (listwise on the target is non-negotiable)
            y_valid = np.isfinite(y)
            if not np.all(y_valid):
                n_bad = int((~y_valid).sum())
                logger.warning(f'Removing {n_bad} rows with non-finite y.')
                keep = np.where(y_valid)[0]
                df = df.iloc[keep].reset_index(drop=True)
                y = y[keep]

        # --- group detection and extraction ---
        # Group detection runs before imputation so that group membership can
        # inform within-group imputation statistics.
        group_name = self.group_name
        if not group_name:
            candidates = detectGroupCandidates(df)
            if candidates and candidates[0].score > 0:
                group_name = candidates[0].name
                logger.info(
                    f'Auto-detected grouping variable "{group_name}" '
                    f'(simulation path only; specify group_name explicitly for deployment).'
                )

        # High-cardinality categoricals (> max_categories levels) are more appropriately
        # modelled as random effects than as lumped fixed-effect dummies.  If no group
        # variable has been found yet, route the highest-cardinality categorical column
        # as the grouping factor, provided it has ≥ 2 observations per level on average.
        if not group_name:
            hi_card = [
                c for c in categorical(df).tolist() if df[c].nunique() > self.max_categories
            ]
            if hi_card:
                best_hi = max(hi_card, key=lambda c: df[c].nunique())
                if len(df) / df[best_hi].nunique() >= 2:
                    group_name = best_hi
                    logger.info(
                        f'Routing high-cardinality categorical "{group_name}" '
                        f'({df[group_name].nunique()} levels) as random effect.'
                    )

        self._group_name_fitted = group_name

        groups: np.ndarray | None = None
        if group_name and group_name in df.columns:
            df = df.sort_values(by=group_name)
            sort_idx = df.index.values
            if y is not None:
                y = y[sort_idx]
            df = df.reset_index(drop=True)
            groups_raw = df.pop(group_name)
            groups, _ = pd.factorize(groups_raw)

        # --- consistent group-aware imputation ---
        # Fit on training data; apply in transform() too — no listwise deletion.
        # Within-group median/mode is used when grouping is available; falls back
        # to the global statistic for groups where the column is entirely missing.
        num_cols_for_imp = numerical(df).tolist()
        cat_cols_for_imp = categorical(df).tolist()
        self._num_impute_stats, self._cat_impute_stats = _computeImputeStats(
            df, groups, num_cols_for_imp, cat_cols_for_imp
        )
        df = _applyImputation(df, groups, self._num_impute_stats, self._cat_impute_stats)

        # --- winsorize numeric columns ---
        # Extreme values are clipped to [mean ± outlier_threshold * std] rather than
        # deleted, so no rows are lost and the same bounds can be applied in transform().
        num_cols_now = numerical(df).tolist()
        if num_cols_now:
            self._winsor_bounds = _computeWinsorBounds(
                df[num_cols_now], threshold=self.outlier_threshold
            )
            df = _applyWinsor(df, self._winsor_bounds)
        else:
            self._winsor_bounds: dict[str, tuple[float, float]] = {}

        # --- near-constant column removal ---
        num_cols_now = numerical(df).tolist()
        bad_const = [
            c
            for c in num_cols_now
            if df[c].value_counts(normalize=True).max() > self.constant_threshold
        ]
        if bad_const:
            logger.warning(f'Dropping near-constant columns {bad_const}.')
        dropped: set[str] = set(bad_const)
        df = df.drop(columns=bad_const)

        # --- correlation filter ---
        df, corr_dropped = _corrFilter(df, threshold=self.corr_threshold)
        dropped |= corr_dropped
        self._dropped_cols = dropped

        # --- categorical lumping ---
        cat_cols = categorical(df).tolist()
        self._lump_maps: dict[str, set] = {}
        self._lump_has_other: dict[str, bool] = {}
        for col in cat_cols:
            lumped, lump_levels, has_other = lumpCategories(
                df[col], self.max_categories, self.min_prevalence
            )
            df[col] = lumped
            self._lump_maps[col] = lump_levels
            self._lump_has_other[col] = has_other

        # drop rows where rare categories became NaN (pool too small for "other")
        if cat_cols:
            still_missing = df[cat_cols].isnull().any(axis=1)
            if still_missing.any():
                keep = np.where(~still_missing)[0]
                df = df.iloc[keep].reset_index(drop=True)
                if y is not None:
                    y = y[keep]
                if groups is not None:
                    groups = groups[keep]
                    groups, _ = pd.factorize(pd.Series(groups))

        # --- classify columns ---
        self._numeric_cols: list[str] = numerical(df).tolist()
        self._categorical_cols: list[str] = categorical(df).tolist()

        # --- fit and apply numeric transformer ---
        if self._numeric_cols:
            x_num = df[self._numeric_cols].to_numpy(dtype=float)
            self._num_transformer: NumericTransformer | None = NumericTransformer(
                exclude_binary=True,
                transform_counts=True,
            )
            x_num = self._num_transformer.fitTransform(x_num)
        else:
            self._num_transformer = None
            x_num = np.empty((len(df), 0))

        # --- fit and apply categorical encoder ---
        if self._categorical_cols:
            # Drop the most frequent level as the reference category, matching
            # R's contr.treatment convention of using the largest group as baseline.
            modal_refs = [
                df[col].value_counts().index[0] for col in self._categorical_cols
            ]
            self._encoder: OneHotEncoder | None = OneHotEncoder(
                drop=modal_refs,
                sparse_output=False,
                handle_unknown='ignore',
            )
            x_cat = self._encoder.fit_transform(df[self._categorical_cols]).astype(float)
        else:
            self._encoder = None
            x_cat = np.empty((len(df), 0))

        # --- output column names ---
        dummy_cols: list[str] = []
        if self._encoder is not None:
            dummy_cols = self._encoder.get_feature_names_out(self._categorical_cols).tolist()
        self._output_columns: list[str] = self._numeric_cols + dummy_cols

        # --- combine X ---
        parts = [p for p in [x_num, x_cat] if p.shape[1] > 0]
        X = np.concatenate(parts, axis=1) if parts else np.empty((len(df), 0))

        # --- y standardisation ---
        if self._y_type == 'unobserved':
            y_out = None
        elif self._y_type == 'continuous':
            self._y_mean = float(np.nanmean(y))
            self._y_std = float(max(float(np.nanstd(y)), 1e-6))
            y_out = (y - self._y_mean) / self._y_std
        elif self._y_type in ('count', 'multiclass'):
            self._y_mean = 0.0
            self._y_std = 1.0
            y_out = y.astype(np.int64)
        else:  # binary
            self._y_mean = 0.0
            self._y_std = 1.0
            y_out = y.astype(float)

        # --- group summary ---
        ns: np.ndarray | None = None
        m: int | None = None
        if groups is not None:
            _, ns = np.unique(groups, return_counts=True)
            m = int(len(ns))

        self._fitted = True

        n, d_minus_1 = X.shape
        return {
            'X': X.astype(np.float64),
            'y': y_out,
            'groups': groups,
            'columns': np.array(self._output_columns, dtype=str),
            'd': d_minus_1 + 1,
            'n': n,
            'ns': ns,
            'm': m,
            'y_type': self._y_type,
        }


def preprocess(
    df: pd.DataFrame,
    group_name: str = '',
    col_miss_threshold: float = 0.25,
    constant_threshold: float = 0.95,
    outlier_threshold: float = 4.0,
) -> dict:
    return DataPreprocessor(
        group_name=group_name,
        col_miss_threshold=col_miss_threshold,
        constant_threshold=constant_threshold,
        outlier_threshold=outlier_threshold,
    ).fit_transform(df)


def preprocessAllGroups(
    df: pd.DataFrame,
    col_miss_threshold: float = 0.25,
    constant_threshold: float = 0.95,
    outlier_threshold: float = 4.0,
) -> dict[str, dict]:
    df = df.copy()
    df.columns = df.columns.str.lower()

    df_no_y = df.drop(columns=['y'], errors='ignore')
    candidates = detectGroupCandidates(df_no_y)
    if len(candidates) > MAX_GROUP_CANDIDATES:
        logger.warning(
            f'{len(candidates)} group candidates; keeping top {MAX_GROUP_CANDIDATES}.'
        )
        candidates = candidates[:MAX_GROUP_CANDIDATES]

    out: dict[str, dict] = {}
    if not candidates:
        out[''] = DataPreprocessor(
            col_miss_threshold=col_miss_threshold,
            constant_threshold=constant_threshold,
            outlier_threshold=outlier_threshold,
        ).fit_transform(df)
        return out

    for cand in candidates:
        try:
            data = DataPreprocessor(
                group_name=cand.name,
                col_miss_threshold=col_miss_threshold,
                constant_threshold=constant_threshold,
                outlier_threshold=outlier_threshold,
            ).fit_transform(df)
            out[cand.name] = data
        except ValueError as e:
            logger.error(f'Group "{cand.name}" skipped: {e}')
    return out
