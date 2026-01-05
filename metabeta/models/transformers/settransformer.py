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
            )
            blocks += [mab]
        self.blocks = nn.ModuleList(blocks)

        # optional output projector
        if d_output is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(d_model, d_output),
                nn.Dropout(dropout),
            )
        else:
            self.proj_out = nn.Identity()


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
