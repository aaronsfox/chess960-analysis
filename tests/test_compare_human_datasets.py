"""Tests for the cross-dataset comparisons."""

import numpy as np

import compare_human_datasets as C


def test_cochran_q_null_and_spread():
    rng = np.random.default_rng(0)
    se = np.full(500, 10.0)
    q0 = C.cochran_q(5 + rng.normal(0, 10, 500), se)
    assert 0.85 < q0["Q_over_df"] < 1.15 and q0["p"] > 0.01
    q1 = C.cochran_q(5 + rng.normal(0, 10, 500) + rng.normal(0, 10, 500), se)
    assert q1["Q_over_df"] > 1.6 and q1["p"] < 1e-6


def test_full_transfer_benchmark_falls_with_noise():
    rng = np.random.default_rng(1)
    x = rng.normal(18, 12, 400)
    precise = C.rho_full_transfer(x, np.full(400, 2.0), rng)
    noisy = C.rho_full_transfer(x, np.full(400, 40.0), rng)
    assert precise > 0.95 and noisy < 0.4
