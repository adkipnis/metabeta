import torch

def getDevice() -> str:
    if torch.cuda.is_available():
        return 'gpu'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
