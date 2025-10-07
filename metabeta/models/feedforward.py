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


