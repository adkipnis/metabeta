# implementation adapted from dingo https://github.com/dingo-gw/dingo
import torch
from torch import nn
from torchdiffeq import odeint
from metabeta.models.feedforward import MLP, ResidualNet
mse = nn.MSELoss(reduction='none')

