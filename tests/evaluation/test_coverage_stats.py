import torch
import pytest

from metabeta.plotting.coverage import plotCoverage
from metabeta.utils.evaluation import Proposal, dictMeanExcl


def _make_proposal() -> Proposal:
    proposed = {
        'global': {
            'samples': torch.zeros((1, 2, 4)),
            'log_prob': torch.zeros((1, 2)),
        },
        'local': {
            'samples': torch.zeros((1, 1, 2, 2)),
            'log_prob': torch.zeros((1, 1, 2)),
        },
    }
    return Proposal(proposed, has_sigma_eps=False, d_corr=1)


def test_dict_mean_excl_ignores_corr_rfx() -> None:
    values = {
        'ffx': torch.tensor([0.1]),
        'sigma_rfx': torch.tensor([0.2]),
        'corr_rfx': torch.tensor([0.9]),
        'rfx': torch.tensor([0.3]),
    }

    assert dictMeanExcl(values) == pytest.approx(0.2)


def test_plot_coverage_stats_exclude_corr_rfx(monkeypatch) -> None:
    proposal = _make_proposal()
    summary = type('Summary', (), {})()
    summary.coverage = {
        0.1: {
            'ffx': torch.tensor([0.90]),
            'sigma_rfx': torch.tensor([0.80, 0.80]),
            'corr_rfx': torch.tensor([0.99]),
            'rfx': torch.tensor([0.70, 0.70]),
        }
    }
    summary.ece = {
        'ffx': torch.tensor([0.10]),
        'sigma_rfx': torch.tensor([0.20, 0.20]),
        'corr_rfx': torch.tensor([0.90]),
        'rfx': torch.tensor([0.30, 0.30]),
    }
    summary.eace = {
        'ffx': torch.tensor([0.11]),
        'sigma_rfx': torch.tensor([0.21, 0.21]),
        'corr_rfx': torch.tensor([0.91]),
        'rfx': torch.tensor([0.31, 0.31]),
    }

    captured: dict[str, float] = {}

    def fake_niceify(_ax, info):
        captured.update(info['stats'])

    monkeypatch.setattr('metabeta.plotting.coverage.niceify', fake_niceify)

    plotCoverage(summary, proposal, show=False, show_corr_rfx=False)

    assert captured['ECE'] == pytest.approx(22.0)
    assert captured['EACE'] == pytest.approx(23.0)
