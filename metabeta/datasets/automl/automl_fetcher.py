from pathlib import Path
import pandas as pd
import openml


def pullDataset(task_id: int):
    task = openml.tasks.get_task(task_id)
    dataset = openml.datasets.get_dataset(task.dataset_id)
    df, y, _, names = dataset.get_data(target=dataset.default_target_attribute)

    # check if columns match
    assert (df.columns == names).all(), 'column name mismatch'

    # convert sparse columns to dense before saving
    is_sparse = [isinstance(col.dtype, pd.SparseDtype) for _, col in df.items()]
    if any(is_sparse):
        df = pd.DataFrame(df.to_numpy(), columns=df.columns)
        y = pd.Series(y.to_numpy())

    # save to parquet
    df['y'] = y
    fn = Path('parquet', f'{dataset.name}.parquet')
    df.to_parquet(fn)
    print(f'Saved to {fn}')


def pullSuite(suite_id: int):
    suite = openml.study.get_suite(suite_id)
    for task_id in suite.tasks:
        pullDataset(task_id)


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # init
    Path('parquet').mkdir(parents=True, exist_ok=True)

    # download
    pullSuite(269)   # regression
    pullSuite(218)   # classification
