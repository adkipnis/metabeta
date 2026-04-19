import torch
from torch import nn

from metabeta.models.transformers import MAB, ISAB
from metabeta.utils.initializers import getInitializer
from metabeta.utils.activations import getActivation


class SetTransformer(nn.Module):
    """Set Transformer:
    Linear -> GELU -> Dropout -> [ISAB] * n_isab -> Pool token -> [MAB] * n_mab -> Norm -> Pool (-> Dropout -> Linear)
    ISAB blocks run on data-only (no pool token) to keep induced summaries clean."""

    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_ff: int,
        d_output: int | None = None,
        n_inducing: int = 32,  # for ISAB blocks
        n_heads: int = 4,
        n_blocks: int = 2,
        n_isab: int = 0,  # first n blocks are ISAB blocks instead of MAB
        use_bias: bool = True,
        pre_norm: bool = True,
        activation: str = 'GELU',
        dropout: float = 0.01,
        weight_init: tuple[str, str] | None = ('xavier', 'normal'),
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        assert n_blocks >= n_isab, 'n_isab must not be larger than n_blocks'
        self.n_isab = n_isab

        # input projector: GeGLU is a FFN-bottleneck activation and doesn't fit
        # a single embedding linear, so fall back to GELU for proj_in.
        proj_activation = 'GELU' if activation == 'GeGLU' else activation
        self.proj_in = nn.Sequential(
            nn.Linear(d_input, d_model),
            getActivation(proj_activation),
            nn.Dropout(dropout),
        )

        # pool token
        self.pool_token = torch.nn.Parameter(torch.randn(d_model) * 0.02)

        # configure blocks
        cfg = dict(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            use_bias=use_bias,
            pre_norm=pre_norm,
            activation=activation,
            dropout=dropout,
            eps=eps,
        )

        # build attention blocks
        blocks = []
        for _ in range(n_isab):
            blocks += [ISAB(**cfg, n_inducing=n_inducing)]   # type: ignore
        for _ in range(n_blocks - n_isab):
            blocks += [MAB(**cfg)]   # type: ignore
        self.blocks = nn.ModuleList(blocks)

        # posthoc norm
        self.norm = nn.LayerNorm(d_model, eps=eps)

        # optional output projector (dropout before linear to avoid noisy summaries)
        self.proj_out = None
        if d_output is not None:
            self.proj_out = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, d_output),
            )

        # optional weight initialization for the projectors
        if weight_init is not None:
            initializer = getInitializer(*weight_init)
            initializer(self.proj_in[0])
            if self.proj_out is not None:
                initializer(self.proj_out[1])

    def _reshape(
        self,
        x: torch.Tensor,
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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

        # ISAB blocks: operate on data only (pool token excluded)
        inv_mask = ~mask if mask is not None else None
        for block in self.blocks[: self.n_isab]:
            x = block(x, key_padding_mask=inv_mask)

        # insert pool token before MAB blocks
        x, mask = self._insertToken(x, mask)
        inv_mask = ~mask if mask is not None else None

        # MAB blocks: pool token participates as CLS token
        for block in self.blocks[self.n_isab :]:
            x = block(x, key_padding_mask=inv_mask)
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

    # ISAB/MAB set transformer
    model = SetTransformer(d, d_model, d_ff, n_blocks=4, n_isab=2)
    x = torch.randn(b, m, n, d)
    mask = torch.ones((b, m, n)).bool()
    mask[..., -1] = False
    out = model(x, mask=mask)
