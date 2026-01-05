import torch
from torch import nn

class MHA(nn.Module):
    ''' Multi-Head Attention wrapper
        automatically deals with inputs that are > 3D '''
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.01,
    ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout, batch_first=True)

