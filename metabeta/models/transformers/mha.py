import torch
from torch import nn
from metabeta.models.utils import zeroInitializer


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


# --------------------------------------------------------
if __name__ == '__main__':

    # sizes
    d_model = 16
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
    x[~mask] = -99 # this entry won't be attended and thus should not matter
    y2 = model(x, mask=mask)
    assert torch.allclose(y1[mask], y2[mask], atol=1e-5)

