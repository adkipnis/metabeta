import torch
from metabeta.utils.families import LIKELIHOOD_FAMILIES


def getDevice() -> str:
    if torch.cuda.is_available():
        return 'gpu'
    return 'cpu'


def setDevice(device: str = ''):
    if not device:
        return torch.device(getDevice())
    elif device == 'mps':
        raise ValueError('MPS is not supported; use cpu or cuda')
    elif device == 'cuda':
        assert torch.cuda.is_available(), 'cuda is not available'
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def datasetFilename(partition: str, epoch: int = 0) -> str:
    """
    Generate dataset filename (without directory path).

    Dataset files are organized in subdirectories by data_id,
    so the filename only contains the partition and epoch info.

    Args:
        partition: One of 'train', 'valid', 'test'
        epoch: Training epoch number (only used for partition='train')

    Returns:
        Filename like 'train_ep0001.npz' or 'test.npz'
    """
    parts = [partition]
    if partition == 'train':
        parts.append(f'ep{epoch:04d}')
    return '_'.join(parts) + '.npz'


def runName(cfg: dict, prefix: str = '') -> str:
    """
    Generate run name for checkpoints.

    Args:
        cfg: Config dict containing data_id, model_id, seed, etc.
        prefix: Optional prefix for the run name

    Returns:
        Run name like 'tiny-n-toy_mtiny_s42' or 'prefix_tiny-n-toy_mtiny_s42'
    """
    parts = []
    if prefix:
        parts.append(prefix)

    # likelihood family
    likelihood_family = cfg.get('likelihood_family', 0)
    parts.append(LIKELIHOOD_FAMILIES[likelihood_family])

    parts += [
        f"d{cfg['data_id']}",
        f"m{cfg['model_id']}",
        f"s{cfg['seed']}",
    ]
    if r_tag := cfg['r_tag']:
        parts += [f'r{r_tag}']
    return '_'.join(parts)
