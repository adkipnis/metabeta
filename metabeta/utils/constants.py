FFX_FAMILIES = ('normal', 'student')
SIGMA_FAMILIES = ('halfnormal', 'halfstudent', 'exponential')

FFX_FAMILY_PROBS = (0.80, 0.20)
SIGMA_RFX_FAMILY_PROBS = (0.60, 0.30, 0.10)
SIGMA_EPS_FAMILY_PROBS = (0.50, 0.40, 0.10)

STUDENT_DF = 5

LIKELIHOOD_FAMILIES = ('normal', 'bernoulli', 'poisson')
LIKELIHOOD_HAS_SIGMA_EPS = (True, False, False)
LIKELIHOOD_BAMBI_FAMILY = ('gaussian', 'bernoulli', 'poisson')

FAMILY_NAMES = {0: 'n', 1: 'b', 2: 'p'}  # normal, bernoulli, poisson
FAMILY_NAMES_REVERSE = {'n': 0, 'b': 1, 'p': 2}


def hasSigmaEps(likelihood: int) -> bool:
    """Whether the likelihood family has a residual variance parameter."""
    return LIKELIHOOD_HAS_SIGMA_EPS[likelihood]


def bambiFamilyName(likelihood: int) -> str:
    """Bambi family string for the given likelihood index."""
    return LIKELIHOOD_BAMBI_FAMILY[likelihood]
