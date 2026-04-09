import json
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from pmlb import dataset_names, fetch_data

# summary table
url = 'https://raw.githubusercontent.com/EpistasisLab/pmlb/master/pmlb/all_summary_stats.tsv'
stats_df = pd.read_csv(url, sep='\t')
stats_df.to_csv('pmlb_summary.csv')
summary = stats_df.set_index('dataset')

# init
raw_dir = Path('raw')
raw_dir.mkdir(parents=True, exist_ok=True)
out_dir = Path('parquet')
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Fetch and cache per-dataset feature-type metadata from PMLB GitHub.
# The packaged .tsv.gz files strip schema information, so integer-encoded
# nominal features are indistinguishable from true numeric features without
# this metadata.  The canonical source is:
#   https://raw.githubusercontent.com/EpistasisLab/pmlb/master/datasets/
#       <dataset>/metadata.yaml
# ---------------------------------------------------------------------------
metadata_cache = Path('pmlb_metadata.json')
if metadata_cache.exists():
    with open(metadata_cache) as f:
        metadata: dict = json.load(f)
    print(f'Loaded cached metadata for {len(metadata)} datasets.')
else:
    print('Fetching per-dataset metadata from PMLB GitHub...')
    metadata = {}
    base_url = 'https://raw.githubusercontent.com/EpistasisLab/pmlb/master/datasets'
    for i, name in enumerate(dataset_names):
        meta_url = f'{base_url}/{name}/metadata.yaml'
        try:
            r = requests.get(meta_url, timeout=15)
            if r.status_code == 200:
                metadata[name] = yaml.safe_load(r.text)
        except Exception as e:
            print(f'  [{name}] metadata fetch failed: {e}')
        if i % 50 == 49:
            print(f'  {i + 1}/{len(dataset_names)} fetched...')
            time.sleep(0.5)  # be polite to GitHub
    with open(metadata_cache, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'Cached metadata for {len(metadata)} datasets.')


def _apply_column_metadata(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Convert PMLB-typed categorical columns to pandas CategoricalDtype.

    The preprocessor's ``categorical()`` helper detects only object/category/string
    columns.  PMLB integer-encodes nominal features, so without this annotation they
    would be treated as numeric (z-standardised) instead of one-hot encoded.

    Only ``type: categorical`` features are converted; ``binary``, ``ordinal``, and
    ``continuous`` features are left as numeric so the pipeline handles them normally.
    """
    if name not in metadata:
        return df
    features = metadata[name].get('features', [])
    cat_cols = {f['name'] for f in features if f.get('type') == 'categorical'}
    for col in df.columns:
        if col in cat_cols:
            df[col] = df[col].astype('category')
    return df


# fetch all datasets
datasets = []
skipped_small = []
skipped_multiclass = []

for name in dataset_names:
    try:
        df = fetch_data(name, local_cache_dir=raw_dir)
        df = df.rename(columns={'target': 'y'})

        # skip tiny datasets (too few rows for meaningful group-level statistics)
        if len(df) < 50:
            skipped_small.append(name)
            print(f'Skipping {name} (n={len(df)} < 50)')
            continue

        fn = Path(out_dir, f'{name}.parquet')
        df.to_parquet(fn)
        print(f'Saved to {fn}')
        datasets += [df]
    except Exception as e:
        print(e)
print(f'\n{len(datasets)} datasets saved.')
