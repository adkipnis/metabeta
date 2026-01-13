import torch
from torch import nn
from metabeta.models.transformers import SetTransformer
from metabeta.models.normalizingflows import CouplingFlow

class Approximator(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.build()

