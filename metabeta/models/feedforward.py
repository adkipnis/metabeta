from collections.abc import Iterable
import torch
from torch import nn
from torch.nn import functional as F


def initializer(method: str, distribution: str):
    methods = ["kaiming", "xavier", "lecun"]
    distributions = ["uniform", "normal"]
    assert method in methods and distribution in distributions, (
        "invalid weight initializer args"
    )
    if method == "kaiming":
        if distribution == "uniform":
            init = nn.init.kaiming_uniform_  # type: ignore
        elif distribution == "normal":
            init = nn.init.kaiming_normal_  # type: ignore
    elif method == "xavier":
        if distribution == "uniform":
            init = nn.init.xavier_uniform_  # type: ignore
        elif distribution == "normal":
            init = nn.init.xavier_normal_  # type: ignore
    elif method == "lecun":
        if distribution == "normal":

            def init(weight):
                fan_in = nn.init._calculate_correct_fan(weight, "fan_in")
                std = (1.0 / fan_in) ** 0.5
                return nn.init.normal_(weight, 0.0, std)
        elif distribution == "uniform":

            def init(weight):
                fan_in = nn.init._calculate_correct_fan(weight, "fan_in")
                limit = (3.0 / fan_in) ** 0.5
                return nn.init.uniform_(weight, -limit, limit)

    def initialize(layer):
        if isinstance(layer, nn.Linear):
            init(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    return initialize


class SLP(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_output: int,
        activation: str = "Mish",
        dropout: float = 0.01,
        norm: str | None = None,
        skip: bool = False,
        use_bias: bool = True,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__()

        layers = [nn.Linear(d_input, d_output, bias=use_bias)]
        if norm == "batch":
            layers += [nn.BatchNorm1d(d_output, eps=eps)]
        elif norm == "layer":
            layers += [nn.LayerNorm(d_output, eps=eps, bias=use_bias)]
        layers += [eval(f"nn.{activation}()")]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        self.layers = nn.Sequential(*layers)

        # optional skip connection
        self.skip = skip
        if self.skip:
            self.shortcut = nn.Linear(d_input, d_output)

    def forward(self, x):
        h = self.layers(x)
        if self.skip:
            h = h + self.shortcut(x)
        return h


class MLP(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_hidden: int | Iterable[int],
        d_output: int | None = None,
        activation: str = "Mish",
        dropout: float = 0.01,
        norm: str | None = None,
        skip: bool = True,
        weight_init: tuple[str, str] | None = ("lecun", "normal"),
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        if isinstance(d_hidden, int):
            d_hidden = [d_hidden]
        layers = []
        d_prev = d_input
        for d_next in d_hidden:
            slp = SLP(d_prev, d_next, activation, dropout, norm, skip, use_bias)
            layers += [slp]
            d_prev = d_next
        if d_output:
            layers += [nn.Linear(d_prev, d_output, bias=use_bias), nn.Dropout(dropout)]
        self.layers = nn.Sequential(*layers)
        if weight_init is not None:
            self.apply(initializer(*weight_init))

    def forward(self, x: torch.Tensor, context=None):
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        h = self.layers(x)
        return h


class ResidualBlock(nn.Module):
