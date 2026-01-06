import torch
from torch import nn
from metabeta.models.utils import zeroInitializer
from metabeta.models.feedforward import TransformerFFN


class MHA(nn.Module):
    ''' Multi-Head Attention wrapper '''
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.01,
        use_bias: bool = True,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout, bias=use_bias, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        if zero_init:
            zeroInitializer(self.mha.out_proj)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # prepare mask
        kpd = None
        if mask is not None:
            kpd = ~mask

        # prepare inputs
        if key is None:
            key = query
        if value is None:
            value = key

        # forward pass
        h, _ = self.mha(query, key, value,
                        key_padding_mask=kpd)
        return self.dropout(h)


class MAB(nn.Module):
    ''' Multihead-Attention Block:
        Norm -> MHA -> Residual -> Norm -> FF -> Residual '''
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
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'

        # Multihead Attention
        self.mha = MHA(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_bias=use_bias,
        )

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


    def forward(
        self,
        x: torch.Tensor, # (batch, seq_len, d_model)
        z: torch.Tensor | None = None, # (batch, seq_len, d_model)
        mask: torch.Tensor | None = None, # (batch, seq_len)
    ) -> torch.Tensor:
        if self.pre_norm:
            h = self.mha(self.norm0(x),
                         self.norm0(z) if z is not None else None,
                         mask=mask)
            x = x + h
            x = x + self.mlp(self.norm1(x))
        else:
            h = self.mha(x, z, mask=mask)
            x = self.norm0(x + h)
            x = self.norm1(x + self.mlp(x))
        return x


class ISAB(nn.Module):
    ''' Induced Self-Attention Block:
        MAB(I,X) -> MAB(X,H)'''
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_inducing: int, # number of induction points
        n_heads: int = 4,
        use_bias: bool = True,
        pre_norm: bool = True,
        activation: str = 'GELU',
        dropout: float = 0.01,
        eps: float = 1e-3,
    ):
        super().__init__()
        assert n_inducing > 0, 'ISAB requires at least one inducing point'
        mab_dict = dict(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            use_bias=use_bias,
            pre_norm=pre_norm,
            activation=activation,
            dropout=dropout,
            eps=eps)
        self.mab0 = MAB(**mab_dict) # type: ignore
        self.mab1 = MAB(**mab_dict) # type: ignore
        self._I = nn.Parameter(torch.empty(n_inducing, d_model))
        nn.init.normal_(self._I, 0.0, 0.02)
 
    def _getI(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-2]
        I = self._I
        return I.expand(*shape, *I.shape)

    def forward(
        self,
        x: torch.Tensor, # (batch, seq_len, d_model)
        mask: torch.Tensor | None = None, # (batch, seq_len)
    ) -> torch.Tensor:
        i = self._getI(x)
        h = self.mab0(i, x, mask=mask)
        h = self.mab1(x, h)
        return h


# --------------------------------------------------------
if __name__ == '__main__':

    # sizes
    d_model = 16
    d_ff = 64
    b, n = 8, 10

    # MHA
    model = MHA(d_model)
    torch.compile(model)
    model.eval()

    # test
    x = torch.randn(b, n, d_model)
    y = model(x)
    assert x.shape == y.shape

    # test qkv
    z = model(x, y)
    _ = model(x, y, z)

    # test mask
    mask = torch.ones(size=(b, n)).bool() # attend all entries
    y_ = model(x, mask=mask)
    assert torch.allclose(y, y_)

    # test mask with different values
    mask[..., 0] = False # don't attend the first entry in sequence
    y1 = model(x, mask=mask)
    x[~mask] = -99. # this entry won't be attended and thus should not matter
    y2 = model(x, mask=mask)
    assert torch.allclose(y1[mask], y2[mask], atol=1e-5)

    # MAB
    model = MAB(d_model, d_ff, pre_norm=False)
    torch.compile(model)
    x = torch.randn(b, n, d_model)
    y = model(x)

    # ISAB
    model = ISAB(d_model, d_ff, n_inducing=32)
    torch.compile(model)
    model.eval()
    y1 = model(x, mask=mask)
    x[~mask] = 0.
    y2 = model(x, mask=mask)
    assert torch.allclose(y1[mask], y2[mask], atol=1e-5)

