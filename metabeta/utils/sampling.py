import numpy as np

def logUniform(a: float,
               b: float,
               add: float = 0.0,
               round: bool = False) -> float|int:
    assert a > 0, 'lower bound must be positive'
    assert b > a, 'upper bound must be larger than lower bound'
    out = np.exp(np.random.uniform(np.log(a), np.log(b)))
    out += add
    if round:
        return int(np.round(out))
    return float(out)


