"""Tests for the self-play analysis helpers."""

import numpy as np
import pandas as pd

from analyse_selfplay import category, cohen_kappa, icc_one_way, weighted_logistic


def test_icc_zero_when_lines_identical_in_expectation():
    rng = np.random.default_rng(0)
    rows = [{"position_id": p, "line_index": l, "score_white": rng.choice([0, 0.5, 1])}
            for p in range(40) for l in range(50) for _ in range(10)]
    assert abs(icc_one_way(pd.DataFrame(rows))) < 0.02


def test_icc_detects_line_effects():
    rng = np.random.default_rng(1)
    rows = []
    for p in range(40):
        for l in range(50):
            pw = rng.uniform(0.05, 0.6)          # each line has its own win rate
            rows += [{"position_id": p, "line_index": l, "score_white": float(rng.random() < pw)}
                     for _ in range(10)]
    assert icc_one_way(pd.DataFrame(rows)) > 0.05


def test_cohen_kappa():
    a = pd.Series(["x", "x", "y", "y"])
    assert cohen_kappa(a, a) == 1.0
    assert abs(cohen_kappa(a, pd.Series(["x", "y", "x", "y"]))) < 1e-9


def test_category_thresholds():
    r = pd.Series({"p_favours_white": 0.9, "p_favours_black": 0.0, "p_balanced": 0.1})
    assert category(r, 0.8) == "favours White"
    assert category(r, 0.95) == "inconclusive"


def test_weighted_logistic_recovers_curve():
    x = np.linspace(0, 100, 200)
    true = 1 / (1 + np.exp(-(-2.5 + 0.02 * x)))
    beta, se, predict = weighted_logistic(x, true, np.full_like(x, 300.0), degree=1)
    assert np.allclose(beta, [-2.5, 0.02], atol=1e-3)
    assert np.allclose(predict(x), true, atol=1e-4)
