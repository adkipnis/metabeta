from collections.abc import Iterable
import torch
import torch.nn as nn
from metabeta.models.feedforward import MLP


# -----------------------------------------------------------------------------
# utils
def make3d(x: torch.Tensor):
    shape = x.shape
    if x.dim() > 3:
        x = x.reshape(-1, *x.shape[-2:])
    return x, shape


def pool3d(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        x *= mask.unsqueeze(-1)
        counts = mask.sum(-1).view(-1, 1) + 1e-12
        x = x.sum(-2) / counts
    else:
        x = x.mean(-2)
    return x


def pool4d(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # mean pooling along samples
    if mask is not None:
        x *= mask.unsqueeze(-1).unsqueeze(-1)
        counts = mask.sum(-1).view(-1, 1, 1) + 1e-12
        x = x.sum(-3) / counts
    else:
        x = x.mean(-3)
    return x


# -----------------------------------------------------------------------------
# Multi-Head Attention
class MultiheadAttention(nn.Module):
    # wrapper for torch's mha
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.01,
        use_bias: bool = False,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # prepare mask
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask

        # set key if not provided
        if key is None:
            key = query

        # forward pass
        h, _ = self.mha(query, key, key, key_padding_mask=key_padding_mask)
        return h


# -----------------------------------------------------------------------------
# Multi-Head Attention Blocks
class BaseBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int | Iterable[int],
        n_heads: int = 4,
        activation: str = "GELU",
        dropout: float = 0.01,
        use_bias: bool = True,
        eps: float = 1e-3,
    ):
        super().__init__()

        # Multihead Attention
        self.att_samp = MultiheadAttention(d_model, n_heads, dropout, use_bias)

        # MLP
        self.mlp = MLP(
            d_input=d_model,
            d_hidden=d_hidden,
            d_output=d_model,
            activation=activation,
            dropout=dropout,
            use_bias=use_bias,
            weight_init=None,
            skip=False,
        )

        # Layer Norms
        self.norm0 = nn.LayerNorm(d_model, eps=eps, bias=use_bias)
        self.norm1 = nn.LayerNorm(d_model, eps=eps, bias=use_bias)

    def forward(
        self,
        h: torch.Tensor,
        masks: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class MultiheadAttentionBlock(BaseBlock):
