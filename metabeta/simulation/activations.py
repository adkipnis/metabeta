''' simplified version of SCM activations from https://github.com/soda-inria/tabicl'''
import numpy as np
import torch
from torch import nn
from torch import distributions as D
from torch.nn import functional as F
from metabeta.utils import logUniform

