import argparse

from metabeta.evaluation.evaluate import Evaluator


def test_evaluator_resolves_laplace_fit_model():
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.cfg = argparse.Namespace(models='LAPLACE')

    assert evaluator._resolveModels() == ['LAPLACE']
