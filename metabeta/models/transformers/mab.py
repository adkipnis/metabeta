import torch
from torch import nn
from metabeta.models.transformers import MHA
from metabeta.models.feedforward import TransformerFFN


class MAB(nn.Module):
    ''' Multihead-Attention Block '''
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int = 4,
        use_bias: bool = True,
        pre_norm: bool = True,
        activation: str = 'GELU',
        dropout: float = 0.01,
        eps: float = 1e-3,
    ):
        super().__init__()

        # Multihead Attention
