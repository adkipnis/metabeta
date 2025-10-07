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
    # simulatneous attention to samples and features
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
        mask0 = None
        if masks is not None:
            mask0 = masks["mask0"]

        h = h + self.att_samp(h, mask=mask0)
        h = self.norm0(h)

        # project
        h = h + self.mlp(h)
        h = self.norm1(h)
        return h


# -----------------------------------------------------------------------------
# Set Transformers


class BaseSetTransformer(nn.Module):
    # base class for set transformers with mean pooling along samples
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_output: int,
        d_input: int,
        depth: int = 2,
        n_heads: int = 4,
        n_blocks: int = 2,
        dropout: float = 0.01,
        activation: str = "GELU",
        use_bias: bool = True,
        eps: float = 1e-3,
        MAB: type[BaseBlock] = MultiheadAttentionBlock,
        **kwargs,
    ):
        super().__init__()

        # attention blocks
        blocks = []
        for _ in range(n_blocks):
            mab = MAB(
                d_model=d_model,
                d_hidden=(d_ff,) * depth,
                n_heads=n_heads,
                activation=activation,
                dropout=dropout,
                use_bias=use_bias,
                eps=eps,
            )
            blocks += [mab]
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Identity()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def getMasks(
        self,
        mask: torch.Tensor | None = None,
        shape: torch.Size | None = None,
    ) -> dict[str, torch.Tensor | None]:
        raise NotImplementedError

    def pool(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # reshape inputs
        x, shape = make3d(x)
        new_shape = x.shape
        if mask is not None:
            mask = mask.reshape(new_shape[:-1])

        # optional embedding
        x = self.embed(x)

        # prepare masks
        masks = self.getMasks(mask, new_shape)

        # attend
        for block in self.blocks:
            x = block(x, masks=masks)

        # overwrite mask-related nans
        x[x.isnan()] = 0.0

        # optionally unwind nd to n,d
        if new_shape[1] != x.shape[1]:
            x = x.reshape((*new_shape, -1))

        # pool across samples
        x = self.pool(x, mask)

        # reshape along features and project out
        x = x.reshape(*shape[:-2], -1)
        x = self.out(x)

        return x


class SetTransformer(BaseSetTransformer):
    # joint embedding, attention along samples
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_output: int,
        d_input: int,
        depth: int = 2,
        n_heads: int = 4,
        n_blocks: int = 2,
        dropout: float = 0.01,
        activation: str = "GELU",
        use_bias: bool = True,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__(d_model, d_ff, d_output, d_input,
                         depth, n_heads, n_blocks,
                         dropout, activation, use_bias, eps,
                         MAB=MultiheadAttentionBlock)

        # embedder
        self.emb = nn.Linear(d_input, d_model, bias=use_bias)

        # output projection
        if d_model != d_output:
            self.out = nn.Linear(d_model, d_output, bias=use_bias)

        # pooling
        self.pool = pool3d

    def embed(self, x):
        x = self.emb(x)
        return x

    def getMasks(self, mask=None, shape=None):
        return {"mask": mask}


class SampleTransformer(BaseSetTransformer):
    # separate embedding, attention along samples
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_output: int,
        d_input: int,
        depth: int = 2,
        n_heads: int = 4,
        n_blocks: int = 2,
        dropout: float = 0.01,
        activation: str = "GELU",
        use_bias: bool = True,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__(d_model, d_ff, d_output, d_input,
                         depth, n_heads, n_blocks,
                         dropout, activation, use_bias, eps,
                         MAB=SampleAttentionBlock)

        # projections
        self.emb = nn.Linear(1, d_model, bias=use_bias)
        self.out = nn.Linear(d_input * d_model, d_output, bias=use_bias)

        # pooling
        self.pool = pool4d

    def embed(self, x):
        # per feature embedding
        x = self.emb(x.unsqueeze(-1))
        return x

    def getMasks(self, mask=None, shape=None):
        # assumes mask (b, n) and no padding for d
        mask0 = mask1 = None
        if mask is not None:
            assert isinstance(shape, torch.Size)
            b, n, d = shape
            mask1 = mask.unsqueeze(1).expand(b, d, n).reshape(b * d, n)
        return dict(mask0=mask0, mask1=mask1)


class DualTransformer(BaseSetTransformer):
    # separate embedding, alternating attention along features and samples
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_output: int,
        d_input: int,
        depth: int = 2,
        n_heads: int = 4,
        n_blocks: int = 2,
        dropout: float = 0.01,
        activation: str = "GELU",
        use_bias: bool = True,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__(d_model, d_ff, d_output, d_input,
                         depth, n_heads, n_blocks,
                         dropout, activation, use_bias, eps,
                         MAB=DualAttentionBlock)

        # projections
        self.emb = nn.Linear(1, d_model, bias=use_bias)
        self.pos = torch.randn(1, 1, d_input, d_model)
        self.out = nn.Linear(d_input * d_model, d_output, bias=use_bias)

        # pooling
        self.pool = pool4d

    def embed(self, x):
        # per feature embedding + fixed positional embedding
        x = self.emb(x.unsqueeze(-1))
        x += self.pos.to(x.device)
        return x

    def getMasks(self, mask=None, shape=None):
        # assumes mask (b, n) and no padding for d
        mask0 = mask1 = None
        if mask is not None:
            assert isinstance(shape, torch.Size)
            b, n, d = shape
            mask0 = mask.reshape(b * n)
            mask1 = mask.unsqueeze(1).expand(b, d, n).reshape(b * d, n)
        return dict(mask0=mask0, mask1=mask1)


class SparseDualTransformer(BaseSetTransformer):
