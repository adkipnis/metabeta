from typing import List, Tuple
import torch
from torch import nn
from metabeta.models.normalizingflows.linear import Transform, Permute, LU, ActNorm
from metabeta.models.normalizingflows.couplingtransforms import Affine, RationalQuadratic
from metabeta.models.normalizingflows.distributions import DiagGaussian, DiagStudent


