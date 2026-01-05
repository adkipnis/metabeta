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
        self.mha = MHA(d_model, n_heads, dropout, use_bias)

        # MLP
        self.mlp = TransformerFFN(
            d_input=d_model,
            d_hidden=d_ff,
            d_output=d_model,
            activation=activation,
            dropout=dropout,
            use_bias=use_bias,
        )

        # Layer Norms
        self.pre_norm = pre_norm
        self.norm0 = nn.LayerNorm(d_model, eps=eps)
        self.norm1 = nn.LayerNorm(d_model, eps=eps)


    def forward(self,
                x: torch.Tensor, # (batch, seq_len, d_model)
                mask: torch.Tensor | None = None):
        if self.pre_norm:
            x = x + self.mha(self.norm0(x), mask=mask)
            x = x + self.mlp(self.norm1(x))
        else:
            x = self.norm0(x + self.mha(x, mask=mask))
            x = self.norm1(x + self.mlp(x))
        return x


