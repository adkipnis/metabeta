import json
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from pmlb import dataset_names, fetch_data

RATE_LIMIT_DELAY = 0.5


def applyColumnMetadata(df: pd.DataFrame, name: str, metadata: dict) -> pd.DataFrame:
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


def main():
    # Summary table
    url = 'https://raw.githubusercontent.com/EpistasisLab/pmlb/master/pmlb/all_summary_stats.tsv'
    stats_df = pd.read_csv(url, sep='\t')
    stats_df.to_csv('pmlb_summary.csv')
    summary = stats_df.set_index('dataset')

    raw_dir = Path('raw')
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path('parquet')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fetch and cache per-dataset feature-type metadata from PMLB GitHub.
    # The packaged .tsv.gz files strip schema information, so integer-encoded
    # nominal features are indistinguishable from true numeric features without
    # this metadata.  The canonical source is:
    #   https://raw.githubusercontent.com/EpistasisLab/pmlb/master/datasets/
    #       <dataset>/metadata.yaml
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
                time.sleep(RATE_LIMIT_DELAY)
        with open(metadata_cache, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f'Cached metadata for {len(metadata)} datasets.')

    n_saved = n_skipped = 0
    n_multiclass = n_binary_recoded = 0

    for name in dataset_names:
        try:
            df = fetch_data(name, local_cache_dir=raw_dir)
            df = df.rename(columns={'target': 'y'})

            if len(df) < 50:
                n_skipped += 1
                print(f'Skipping {name} (n={len(df)} < 50)')
                continue

            if name in summary.index:
                meta_row = summary.loc[name]
                task = meta_row['task']
                n_classes = int(meta_row.get('n_classes', 0))

                if task == 'classification':
                    if n_classes > 2:
                        # Encode multiclass targets as non-numeric strings so the
                        # preprocessor detects them as 'multiclass'.
                        df['y'] = 'class_' + df['y'].astype(str)
                        n_multiclass += 1
                    else:
                        # Recode binary targets to {0, 1} so the preprocessor
                        # detects them as binary rather than continuous.
                        classes = sorted(df['y'].dropna().unique().tolist(), key=str)
                        if list(classes) != [0, 1]:
                            df['y'] = df['y'].map({classes[0]: 0, classes[1]: 1})
                            n_binary_recoded += 1

            df = applyColumnMetadata(df, name, metadata)

            fn = Path(out_dir, f'{name}.parquet')
            df.to_parquet(fn)
            print(f'Saved to {fn}')
            n_saved += 1
        except Exception as e:
            print(e)

    print(f'\n{n_saved} datasets saved, {n_skipped} skipped (n < 50).')
    print(f'{n_multiclass} multiclass datasets re-encoded with string labels.')
    print(f'{n_binary_recoded} binary datasets recoded to {{0, 1}}.')


if __name__ == '__main__':
    main()
