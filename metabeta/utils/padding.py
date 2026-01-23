import numpy as np


def maxShapes(batch: list[dict[str, np.ndarray]]) -> dict[str, tuple[int, ...]]:
    ''' for each array in dataset, get the maximal shape over the whole batch '''
    out = {}
    for dataset in batch:
        for key, array in dataset.items():
            if not isinstance(array, np.ndarray):
                raise ValueError('expected all entries to be arrays')
            shape = tuple(array.shape)
            if key not in out:
                out[key] = shape
                continue
            assert len(shape) == len(out[key]), 'ndim mismatch'
            # Expand out[key] tuple elementwise to max dimension
            out[key] = tuple(
                max(old_dim, new_dim)
                for old_dim, new_dim in zip(out[key], shape)
            )
    return out


def aggregate(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    ''' collate list of datasets to single batched dataset
        note: assumes consistency in keys and dtypes in batch '''
    out = {}
    max_shapes = maxShapes(batch)
    batch_size = len(batch)

    # init with zeros
    for key, shape in max_shapes.items():
        dtype = batch[0][key].dtype
        out[key] = np.zeros((batch_size, *shape), dtype=dtype)

    # fill with slicing
    for i, dataset in enumerate(batch):
        for key, dest in out.items():
            src = dataset[key]
            slc = (i, *tuple(slice(0, s) for s in src.shape))
            dest[slc] = src
    return out


def unpad(ds: dict[str, np.ndarray], sizes: dict[str, int]) -> dict[str, np.ndarray]:
    d, q, m, n = sizes['d'], sizes['q'], sizes['m'], sizes['n']

    # observations
    ds['y'] = ds['y'][:n]
    ds['X'] = ds['X'][:n, :d]
    ds['groups'] = ds['groups'][:n]
    ds['ns'] = ds['ns'][:m]

    # hyperparams
    ds['nu_ffx'] = ds['nu_ffx'][:d]
    ds['tau_ffx'] = ds['tau_ffx'][:d]
    ds['tau_rfx'] = ds['tau_rfx'][:q]

    # params
    ds['ffx'] = ds['ffx'][:d]
    ds['sigma_rfx'] = ds['sigma_rfx'][:q]
    ds['rfx'] = ds['rfx'][:m, :q]
    return ds

# def extractSingle(
#     batch: dict[str, np.ndarray],
#     idx: int,
# ) -> dict[str, np.ndarray]:
#     ''' extract a single dataset from batch and remove padding '''
#     ds = {k: v[idx] for k,v in batch.items()}
#     d, q, m, n = ds['d'], ds['q'], ds['m'], ds['n']

