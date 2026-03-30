import numpy as np


def maxShapes(batch: list[dict[str, np.ndarray]]) -> dict[str, tuple[int, ...]]:
    """for each array in dataset, get the maximal shape over the whole batch"""
    out = {}
    for dataset in batch:
        for key, array in dataset.items():
            if not isinstance(array, np.ndarray):
                raise ValueError('expected all entries to be arrays')
            shape = tuple(array.shape)
            if key not in out:
                out[key] = shape
                continue
            if len(shape) != len(out[key]):
                raise ValueError(f'ndim mismatch for key={key}')
            # Expand out[key] tuple elementwise to max dimension
            out[key] = tuple(max(old_dim, new_dim) for old_dim, new_dim in zip(out[key], shape))
    return out


def aggregate(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """collate list of datasets to single batched dataset
    note: assumes consistency in keys and dtypes in batch"""
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


def padToModel(
    ds: dict[str, np.ndarray],
    max_d: int,
    max_q: int,
) -> dict[str, np.ndarray]:
    """Pad a single (unpadded) dataset to a model's maximal d/q dimensions.

    Expects *actual*-sized arrays (after ``unpad``).  Zero-pads feature and
    random-effect dimensions so the dataset can be fed into a model whose
    architecture was built for ``max_d`` / ``max_q``.  Also creates ``Z``
    from the (now padded) ``X``.

    Fit-sample arrays (``nuts_*`` / ``advi_*``) are padded consistently.
    """
    d = int(ds['d'])
    q = int(ds['q'])

    # --- fixed-effects dimension (d → max_d) ---
    if d < max_d:
        n = ds['X'].shape[0]
        X = np.zeros((n, max_d), dtype=ds['X'].dtype)
        X[:, :d] = ds['X']
        ds['X'] = X

        for key in ('ffx', 'nu_ffx', 'tau_ffx'):
            padded = np.zeros(max_d, dtype=ds[key].dtype)
            padded[:d] = ds[key]
            ds[key] = padded

    # --- random-effects dimension (q → max_q) ---
    if q < max_q:
        for key in ('sigma_rfx', 'tau_rfx'):
            padded = np.zeros(max_q, dtype=ds[key].dtype)
            padded[:q] = ds[key]
            ds[key] = padded

        m = int(ds['m'])
        rfx = np.zeros((m, max_q), dtype=ds['rfx'].dtype)
        rfx[:, :q] = ds['rfx']
        ds['rfx'] = rfx

        corr = np.eye(max_q, dtype=ds['corr_rfx'].dtype)
        corr[:q, :q] = ds['corr_rfx']
        ds['corr_rfx'] = corr

    # Z is first max_q columns of (now padded) X
    ds['Z'] = ds['X'][:, :max_q].copy()

    # --- pad fit samples (nuts_* / advi_*) ---
    # Fit arrays may be stored at the file-wide max d/q (larger than this
    # dataset's actual d/q), so we trim first, then re-pad to model dims.
    for method in ('nuts', 'advi'):
        ffx_key = f'{method}_ffx'
        if ffx_key not in ds:
            continue
        s = ds[ffx_key].shape[-1]

        # trim then pad ffx: (d_file, s) → (d, s) → (max_d, s)
        ds[ffx_key] = ds[ffx_key][:d]
        if d < max_d:
            padded = np.zeros((max_d, s), dtype=ds[ffx_key].dtype)
            padded[:d] = ds[ffx_key]
            ds[ffx_key] = padded

        sig_key = f'{method}_sigma_rfx'
        if sig_key in ds:
            ds[sig_key] = ds[sig_key][:q]
            if q < max_q:
                padded = np.zeros((max_q, s), dtype=ds[sig_key].dtype)
                padded[:q] = ds[sig_key]
                ds[sig_key] = padded

        rfx_key = f'{method}_rfx'
        if rfx_key in ds:
            ds[rfx_key] = ds[rfx_key][:q]
            if q < max_q:
                m_fit = ds[rfx_key].shape[1]
                padded = np.zeros((max_q, m_fit, s), dtype=ds[rfx_key].dtype)
                padded[:q] = ds[rfx_key]
                ds[rfx_key] = padded

    return ds


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
    if 'corr_rfx' in ds:
        ds['corr_rfx'] = ds['corr_rfx'][:q, :q]
    return ds


# def extractSingle(
#     batch: dict[str, np.ndarray],
#     idx: int,
# ) -> dict[str, np.ndarray]:
#     ''' extract a single dataset from batch and remove padding '''
#     ds = {k: v[idx] for k,v in batch.items()}
#     d, q, m, n = ds['d'], ds['q'], ds['m'], ds['n']
