"""
Tests whether a trained model actually incorporates prior family information
into its posteriors, rather than ignoring the family encoding.

Approach:
    1. Load a trained model and a batch of test data
    2. Fix the random seed and dataset
    3. For each parameter group (ffx, sigma_rfx, sigma_eps), swap the family
       index while keeping everything else identical
    4. Compare posterior mean and SD under each family
    5. Verify that heavier-tailed families (StudentT, HalfStudentT) produce
       wider posteriors than lighter ones (Normal, HalfNormal)
"""

