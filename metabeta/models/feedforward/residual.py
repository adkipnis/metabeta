import torch
from torch import nn
from torch.nn import functional as F
from metabeta.models.feedforward.utils import (
    getActivation, getInitializer,
    zeroInitializer, lastZeroInitializer, weightNormInitializer)

class ResidualBlock(nn.Module):
    ''' Residual Block with optional GLU:
        Linear -> Norm -> Activation -> Dropout -> Linear -> Norm -> (GLU ->) Residual Sum
    '''
    def __init__(
        self,
        d_hidden: int,
        d_context: int = 0,
        use_bias: bool = True,
        layer_norm: bool = False,
        pre_norm: bool = False,
        eps: float = 1e-3, # numerical stability in layer norm denominator
        activation: str = 'ReLU',
        dropout: float = 0.0,
        use_glu: bool = False,
        rscale: float = 0.1, # residual scale
        cscale: float = 0.1, # context scale
    ):
        super().__init__()
