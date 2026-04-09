"""
Fetch tabular datasets from UCI Machine Learning Repository.

Follows patterns from pmlb_fetcher.py:
- Cache dataset catalog (CSV) and metadata (JSON) locally
- Filter by size (50 <= n <= 1M)
- Rename target column to 'y' (None if dataset has no designated target)
- Encode multiclass targets as strings
- Apply categorical type annotations from UCI metadata
- Save to parquet format

Directory structure:
    ucimlr/
    ├── ucimlr_fetcher.py         # This script
    ├── ucimlr_catalog.csv        # Cached catalog of available datasets
    ├── ucimlr_metadata.json      # Cached per-dataset metadata
    └── parquet/
        ├── <id>_<name>.parquet

Usage:
    cd metabeta/datasets/ucimlr
    uv run python ucimlr_fetcher.py
"""

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from ucimlrepo import fetch_ucirepo

MIN_ROWS = 50
MAX_ROWS = 1_000_000
RATE_LIMIT_DELAY = 0.5
CATALOG_API_URL = 'https://archive.ics.uci.edu/api/datasets/list'
DATASET_API_URL = 'https://archive.ics.uci.edu/api/dataset'


def fetchDatasetMetadata(dataset_id: int) -> dict[str, Any]:
    """Fetch dataset-level metadata without downloading data."""
    response = requests.get(DATASET_API_URL, params={'id': dataset_id}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if payload.get('status') != 200 or 'data' not in payload:
        raise ValueError(f'Metadata API error for id={dataset_id}: {payload.get("message", "unknown")}')
    d = payload['data']
    return {
        'id': dataset_id,
        'name': d.get('name'),
        'uci_id': d.get('uci_id'),
        'abstract': d.get('abstract'),
        'area': d.get('area'),
        'task': d.get('task'),
        'characteristics': d.get('characteristics'),
        'num_instances': d.get('num_instances'),
        'num_features': d.get('num_features'),
        'year': d.get('year_of_dataset_creation'),
        'intro_paper': d.get('intro_paper'),
        'additional_info': d.get('additional_info'),
        'data_url': d.get('data_url'),
        'variables': d.get('variables'),
    }


def main():
    out_dir = Path('parquet')
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_cache = Path('ucimlr_catalog.csv')
    metadata_cache = Path('ucimlr_metadata.json')

    # Fetch and cache Python-compatible dataset catalog
    print('Fetching Python-compatible dataset catalog...')
    resp = requests.get(CATALOG_API_URL, params={'filter': 'python'}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get('status') != 200 or 'data' not in payload:
        raise ValueError(f'Catalog API error: {payload.get("message", "unknown")}')
    catalog_df = pd.DataFrame(payload['data'])[['id', 'name']].copy()
    catalog_df['id'] = catalog_df['id'].astype(int)
    catalog_df.to_csv(catalog_cache, index=False)
    print(f'{len(catalog_df)} Python-compatible datasets available.\n')

    # Load or initialise metadata cache
    if metadata_cache.exists():
        with open(metadata_cache) as f:
            metadata: dict[int, dict[str, Any]] = {int(k): v for k, v in json.load(f).items()}
        print(f'Loaded cached metadata for {len(metadata)} datasets.\n')
    else:
        metadata = {}

    n_saved = n_skipped = n_errors = 0

    for _, row in catalog_df.iterrows():
        dataset_id = int(row['id'])
        dataset_name = row['name']
        out_path = out_dir / f'{dataset_id}_{dataset_name}.parquet'

        if out_path.exists():
            n_saved += 1
            continue

        try:
            # Fetch metadata first to check size before downloading data
            if dataset_id not in metadata:
                metadata[dataset_id] = fetchDatasetMetadata(dataset_id)
                time.sleep(RATE_LIMIT_DELAY)

            n_instances = metadata[dataset_id].get('num_instances')
            if n_instances is None:
                print(f'[{dataset_id}] {dataset_name}: unknown size, skipping.')
                n_skipped += 1
                continue
            n_instances = int(n_instances)
            if not (MIN_ROWS <= n_instances <= MAX_ROWS):
                print(f'[{dataset_id}] {dataset_name}: n={n_instances:,} out of range, skipping.')
                n_skipped += 1
                continue

            print(f'[{dataset_id}] Fetching {dataset_name} (n={n_instances:,})...')
            data = fetch_ucirepo(id=dataset_id)

            metadata[dataset_id].update({
                'name': getattr(data.metadata, 'name', dataset_name),
                'uci_id': getattr(data.metadata, 'uci_id', dataset_id),
                'abstract': getattr(data.metadata, 'abstract', None),
                'area': getattr(data.metadata, 'area', None),
                'task': getattr(data.metadata, 'task', None),
                'characteristics': getattr(data.metadata, 'characteristics', None),
                'num_instances': getattr(data.metadata, 'num_instances', n_instances),
                'num_features': getattr(data.metadata, 'num_features', None),
                'year': getattr(data.metadata, 'year_of_dataset_creation', None),
                'intro_paper': getattr(data.metadata, 'intro_paper', None),
                'additional_info': getattr(data.metadata, 'additional_info', None),
                'variables': data.variables.to_dict('records') if data.variables is not None else None,
            })

            X = data.data.features
            y = data.data.targets

            df = X.copy()
            if y is None:
                print(f'  No target variable.')
                df['y'] = None
            else:
                if isinstance(y, pd.DataFrame):
                    if len(y.columns) > 1:
                        print(f'  Multiple targets, using first: {y.columns[0]}')
                    y = y.iloc[:, 0]
                df['y'] = y

                task_type = str(getattr(data.metadata, 'task', '')).lower()
                if 'classification' in task_type and df['y'].nunique() > 2:
                    df['y'] = 'class_' + df['y'].astype(str)
                    print(f'  Encoded {df["y"].nunique()}-class target as strings.')

            # Apply categorical type annotations from UCI metadata
            if data.variables is not None:
                cat_cols = data.variables[
                    data.variables['type'].isin(['Categorical', 'Binary'])
                ]['name'].tolist()
                for col in cat_cols:
                    if col in df.columns:
                        df[col] = df[col].astype('category')

            df.to_parquet(out_path)
            print(f'  Saved to {out_path}')
            n_saved += 1
            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            print(f'[{dataset_id}] {dataset_name}: ERROR - {e}')
            n_errors += 1

    with open(metadata_cache, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'\n{"=" * 60}')
    print(f'{n_saved} datasets saved, {n_skipped} skipped, {n_errors} errors.')


if __name__ == '__main__':
    main()
