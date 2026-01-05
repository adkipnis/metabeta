import torch
from torch import nn
from metabeta.models.transformers import MAB, ISAB
from metabeta.models.utils import getInitializer

class SetEncoder(nn.Module):
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
        weight_init: tuple[str, str] | None = ('xavier', 'normal'),
        eps: float = 1e-3,
    ):
        super().__init__()

        # input projector
        self.proj_in = nn.Sequential(
                nn.Linear(d_input, d_model),
                nn.Dropout(dropout),
        )

        # pool token
        self.pool_token = torch.nn.Parameter(torch.randn(d_model) * 0.02)
 
        # attention blocks
        blocks = []
        for _ in range(n_blocks):
            mab = MAB(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                use_bias=use_bias,
                pre_norm=pre_norm,
                activation=activation,
                dropout=dropout,
                eps=eps,
            )
            blocks += [mab]
        self.blocks = nn.ModuleList(blocks)

        # posthoc norm
        self.norm = nn.LayerNorm(d_model, eps=eps)

        # optional output projector
        self.proj_out = None
        if d_output is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(d_model, d_output),
                nn.Dropout(dropout),
            )

        # optional weight initialization for the projectors
        if weight_init is not None:
            initializer = getInitializer(*weight_init)
            initializer(self.proj_in[0])
            if self.proj_out is not None:
                initializer(self.proj_out[0])

    def _reshape(
        self, x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # deal with sequences longer than 3d
        if x.dim() > 3:
            x = x.reshape(-1, *x.shape[-2:])
            if mask is not None:
                mask = mask.reshape(-1, *mask.shape[-1:])
        return x, mask

    def _insertToken(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor|None]:
        # prepend pool token along sequence dim and include in mask
        token = self.pool_token.unsqueeze(0).unsqueeze(0)
        token = token.expand(x.size(0), 1, -1)
        x = torch.cat([token, x], dim=-2)
        if mask is not None:
            token_mask = torch.ones_like(mask[:, 0:1])
            mask = torch.cat([token_mask, mask], dim=-1)
        return x, mask

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # optionally reshape inputs
        old_shape = x.shape
        x, mask = self._reshape(x, mask)

        # linear embedding
        x = self.proj_in(x)

        # insert token
        x, mask = self._insertToken(x, mask)

        # attend
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm(x)

        # pool by extracting token
        x = x[:, 0]
 
        # project out
        if self.proj_out is not None:
            x = self.proj_out(x)

        # get back original shape
        x = x.reshape(*old_shape[:-2], -1)
        return x


# --------------------------------------------------------
if __name__ == '__main__':

    # sizes
    d_model = 16
    d_ff = 64
    b, m, n, d = 8, 5, 10, 3

    # MHA
    model = SetEncoder(d, d_model, d_ff)
    # model = torch.compile(model)
    model.eval()
 
    # 3d case
    x = torch.randn(b, n, d)
    y = model(x)

    # 4d case
    x = torch.randn(b, m, n, d)
    y = model(x)

    # mask
    mask = torch.ones((b, m, n)).bool()
    mask[..., -1] = False
    y1 = model(x, mask=mask)
    x[~mask] = 99
    y2 = model(x, mask=mask)
    assert torch.allclose(y1, y2, atol=1e-5)

