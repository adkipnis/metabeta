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

        # deal with left-over dimensions
        shape = None
        if len(query.shape) > 3:
            shape = query.shape
            query = query.reshape(-1, *query.shape[-2:])
            key = key.reshape(-1, *key.shape[-2:])
            value = value.reshape(-1, *value.shape[-2:])
            if kpd is not None:
                kpd = kpd.reshape(-1, *kpd.shape[-1:])

        # forward pass
        h, _ = self.mha(query, key, value,
                        key_padding_mask=kpd)
 
        # optionally reshape outputs
        if shape is not None:
            h = h.view(shape)
        return h

