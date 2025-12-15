from pathlib import Path
import pandas as pd
from pmlb import dataset_names, fetch_data

# summary table
url = "https://raw.githubusercontent.com/EpistasisLab/pmlb/master/pmlb/all_summary_stats.tsv"
stats_df = pd.read_csv(url, sep='\t')
stats_df.to_csv('pmlb_summary.csv')

# init
raw_dir = Path('raw')
raw_dir.mkdir(parents=True, exist_ok=True)
out_dir = Path('parquet')
out_dir.mkdir(parents=True, exist_ok=True)
datasets = []
candidate_cols = []

# fetch all datasets
for name in dataset_names:
    try:
        df = fetch_data(name, local_cache_dir=raw_dir)
        df = df.rename(columns={'target': 'y'})
        fn = Path(out_dir, f'{name}.parquet')
        df.to_parquet(fn)
        print(f'Saved to {fn}')
        datasets += [df]
    except Exception as e:
        print(e)
print(f'\n{len(datasets)} datasets saved.')
