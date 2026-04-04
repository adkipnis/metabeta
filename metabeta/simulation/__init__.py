from .prior import Prior, hypersample, bambiDefaultPriors
from .synthesizer import Synthesizer, Scammer
from .emulator import Emulator, Subsampler
from .simulator import Simulator, simulate
from .generate import Generator
from .nonlinear import addNonlinearity, adjustParamsForNonlinearity, NONLINEAR_KINDS

__all__ = [
    'Prior',
    'hypersample',
    'bambiDefaultPriors',
    'Synthesizer',
    'Scammer',
    'Emulator',
    'Subsampler',  # predictors
    'Simulator',
    'simulate',  # outcomes
    'Generator',  # full datasets
    'addNonlinearity',  # structural perturbations
    'adjustParamsForNonlinearity',
    'NONLINEAR_KINDS',
]
