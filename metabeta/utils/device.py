import torch


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
