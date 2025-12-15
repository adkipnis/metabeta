from dataclasses import dataclass
import numpy as np

# --- standardization
def moments(x: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis, keepdims=True)
    std = x.std(axis, keepdims=True)
    return mean, std

def standardize(x: np.ndarray, axis: int) -> np.ndarray:
    mean, std = moments(x, axis)
    return (x - mean) / std

@dataclass
class Standardizer:
    axis: int

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mean, self.std = moments(x, self.axis)
        return (x - self.mean) / self.std

    def backward(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


# --- checkers
def checkBinary(x: np.ndarray) -> np.ndarray:
    binary = (x == 0) + (x == 1)
    return np.all(binary, axis=0)

def checkContinuous(x: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    diffs = np.abs(x - x.round())
    too_large = (diffs > tol)
    return np.all(too_large, axis=0)

def checkConstant(x: np.ndarray) -> np.ndarray:
    same = (x[0, :] == x)
    return np.all(same, axis=0)


# --- groups and counts
def sampleCounts(n: int, m: int, alpha: float = 10.) -> np.ndarray:
    p = np.random.dirichlet(np.ones(m) * alpha)
    ns = np.round(p * n).astype(int)
    diff = n - ns.sum()
    if diff > 0:
        idx = ns.argmin(0)
        ns[idx] += diff
    elif diff < 0:
        idx = ns.argmax(0)
        ns[idx] += diff
    if (ns < 1).any():
        print('non-positive counts found')
        return sampleCounts(n, m, alpha)
    return ns

def counts2groups(ns: np.ndarray) -> np.ndarray:
    m = len(ns)
    unique = np.arange(m)
    groups = np.repeat(unique, ns)
    return groups


