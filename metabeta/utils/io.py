import torch

def getDevice() -> str:
    if torch.cuda.is_available():
        return 'gpu'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def setDevice(device: str = ''):
    if not device:
        return torch.device(getDevice())
    elif device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def datasetFilename(cfg: dict, partition: str,  epoch: int = 0) -> str:
    # partition type and optionally epoch
    parts = [partition]
    if partition == 'train':
        parts.append(f'ep{epoch:04d}')

    # sizes
    parts += [
        f'd{cfg["max_d"]}',
        f'q{cfg["max_q"]}',
        f'm{cfg["min_m"]}-{cfg["max_m"]}',
        f'n{cfg["min_n"]}-{cfg["max_n"]}',
        cfg['ds_type'],
    ]

    # source of sampled data
    if cfg['ds_type'] == 'sampled':
        parts.append(cfg['source'])

    return '_'.join(parts) + '.npz'

def runName(cfg: dict, prefix: str = '') -> str:
    parts = []
    if prefix:
        parts.append(prefix)
    parts += [
        f'd{cfg["d_tag"]}',
        f'm{cfg["m_tag"]}',
        f's{cfg["seed"]}',
        ]
    if (r_tag := cfg['r_tag']):
        parts += [f'r{r_tag}']
    return '_'.join(parts)

