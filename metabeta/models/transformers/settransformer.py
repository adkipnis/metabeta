import torch
from torch import nn
from metabeta.models.transformers import MAB

class SetTransformer(nn.Module):
    ''' O(n²) Set Transformer:
        Linear -> Dropout -> [MAB] * n_blocks -> Pool (-> Linear -> Dropout)'''
    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_ff: int,
        d_output: int | None = None,
        n_heads: int = 4,
        n_blocks: int = 2,
        use_bias: bool = True,
        pre_norm: bool = True,
        activation: str = 'GELU',
        dropout: float = 0.01,
    ):
        super().__init__()

