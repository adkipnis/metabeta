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


