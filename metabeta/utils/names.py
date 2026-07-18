def datasetFilename(partition: str, epoch: int = 0) -> str:
    parts = [partition]
    if partition == 'train':
        parts.append(f'ep{epoch:05d}')
    return '_'.join(parts) + '.npz'


def runName(cfg: dict, prefix: str = '') -> str:
    parts = []
    if prefix:
        parts.append(prefix)
    parts += [
        f"data={cfg['data_id']}",
        f"model={cfg['model_id']}",
        f"seed={cfg['seed']}",
    ]
    if r_tag := cfg['r_tag']:
        parts.append(f'run={r_tag}')
    return '_'.join(parts)
