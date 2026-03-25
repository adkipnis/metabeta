import math
import numpy as np
import torch
from scipy.stats import norm, t
from torch import distributions as D


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FFX_FAMILIES = ('normal', 'student')
SIGMA_FAMILIES = ('halfnormal', 'halfstudent')

STUDENT_DF = 4


