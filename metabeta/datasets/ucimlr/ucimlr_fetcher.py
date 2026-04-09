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
