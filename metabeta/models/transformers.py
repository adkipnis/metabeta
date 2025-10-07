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
    # attention to samples after jointly embedding features
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
        super().__init__(d_model, d_hidden, n_heads, activation, dropout, use_bias, eps)

    def forward(self, h, masks=None):
        # x (b, n, emb)
        mask = None
        if masks is not None:
            mask = masks["mask"]

        # attend
        h = h + self.att_samp(h, mask=mask)
        h = self.norm0(h)

        # project out
        h = h + self.mlp(h)
        h = self.norm1(h)

        return h


class SampleAttentionBlock(BaseBlock):
    # attention to samples after separate embedding features
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
        super().__init__(d_model, d_hidden, n_heads, activation, dropout, use_bias, eps)

    def forward(self, h, masks=None):
        # h (b, n, d, emb)
        mask1 = None
        if masks is not None:
            mask1 = masks["mask1"]

        # attend samples
        h = h.transpose(-2, -3)  # make samples second-last dim
        h, shape = make3d(h)
        h = h + self.att_samp(h, mask=mask1)
        h = self.norm0(h)
        h = h.reshape(*shape)
        h = h.transpose(-2, -3)

        # project
        h = h + self.mlp(h)
        h = self.norm1(h)
        return h


class DualAttentionBlock(BaseBlock):
    # alternating attention to samples and features
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
        super().__init__(d_model, d_hidden, n_heads, activation, dropout, use_bias, eps)

        # additional modules
        self.att_feat = MultiheadAttention(d_model, n_heads, dropout, use_bias)
        self.norm2 = nn.LayerNorm(d_model, eps=eps, bias=use_bias)

    def forward(self, h, masks=None):
        mask0 = mask1 = None
        if masks is not None:
            mask0, mask1 = masks.values()

        # attentend features
        h, shape = make3d(h)
        if mask0 is not None:
            h[mask0] = h[mask0] + self.att_feat(h[mask0])
        else:
            h = h + self.att_feat(h)
        h = self.norm0(h)
        h = h.reshape(*shape).transpose(1, 2)

        # attend samples
        h, shape = make3d(h)
        h = h + self.att_samp(h, mask=mask1)
        h = self.norm1(h)
        h = h.reshape(*shape).transpose(1, 2)

        # project
        h = h + self.mlp(h)
        h = self.norm2(h)
        return h


class ElongatedAttentionBlock(BaseBlock):
