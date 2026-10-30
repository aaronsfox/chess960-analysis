"""Tests for the Lichess Chess960 analysis: simulated games with known
position effects, rating scale and engine slope."""

import numpy as np
import pandas as pd
import pytest
from scipy import special

import analyse_lichess960 as A

K = A.K


def simulate(n=400_000, tau=6.0, gamma=0.5, adv=8.0, beta=0.9, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(18, 12, 960)                      # engine self-play Delta_Elo
    true = adv + gamma * (x - x.mean()) + rng.normal(0, tau, 960)
    pos = rng.integers(0, 960, n)
    mean = rng.normal(1700, 250, n).clip(1200, 2600)
    gap = rng.normal(0, 150, n)
    we, be = (mean + gap / 2).round(), (mean - gap / 2).round()
    eta = (true[pos] + 0 * x[pos]) / K + beta * (we - be) * np.log(10) / 400
    p = special.expit(eta)
    u = rng.random(n)
    draw = rng.random(n) < 0.045
    score = np.where(draw, 0.5, (u < (p - 0.0225) / 0.955).astype(float))
    d = pd.DataFrame({"position_id": pos, "white_elo": we, "black_elo": be, "score_white": score,
                      "white_id": rng.integers(0, 50_000, n), "black_id": rng.integers(0, 50_000, n),
                      "time_forfeit": False, "tournament": False})
    eng = pd.DataFrame({"engine": "stockfish", "position_id": range(960), "delta_elo_median": x,
                        "delta_elo_sd": 0.0})
    eng = pd.concat([eng, eng.assign(engine="caissa")])
    eng = eng.pivot(index="position_id", columns="engine", values=["delta_elo_median", "delta_elo_sd"])
    return d, eng, true


@pytest.fixture(scope="module")
def sim():
    return simulate()


def test_rating_model_recovers_scale_and_advantage(sim):
    d, _, _ = sim
    rm = A.rating_model(d)
    assert rm["beta_lo"] < 0.9 < rm["beta_hi"]
    assert abs(rm["advantage_elo"] - 8.0) < 3


def test_stage2_recovers_slope_and_tau(sim):
    d, eng, true = sim
    rm = A.rating_model(d)
    pos = A.stage1(d, rm["_b"][1])
    assert np.median(pos.se_ratio_pair_cluster) == pytest.approx(1, abs=0.05)
    pos, meta, dec = A.stage2(pos, eng)
    sl = meta[(meta.model == "slope") & (meta.predictor == "stockfish")].iloc[0]
    assert sl.gamma_lo < 0.5 < sl.gamma_hi
    assert sl.tau_lo <= 6.0 <= sl.tau_hi
    assert len(dec) == 20 and dec.human_elo.is_monotonic_increasing is not None


def test_reml_tau_zero_when_no_spread():
    rng = np.random.default_rng(3)
    v = np.full(960, 100.0)
    y = 5 + rng.normal(0, 10, 960)
    r = A.reml(y, v, np.ones((960, 1)))
    assert r["tau"] < 3 and r["p_tau"] > 0.05 and r["tau_lo"] == 0.0


def test_game_level_and_bands(sim):
    d, eng, _ = sim
    inter, bands = A.game_level(d, eng)
    g = inter.set_index("term")
    assert g.loc["gamma", "lo"] < 0.5 < g.loc["gamma", "hi"]
    assert abs(g.loc["eta_gamma_per_100", "estimate"]) < 0.06
    assert list(bands.band) == A.BAND_LABELS


def test_filters_in_order():
    g = pd.DataFrame({
        "estimated_s": [300, 300, 300, 300, 300, 300, 300, 200, 600],
        "termination": ["Normal", "Abandoned", "Normal", "Normal", "Normal", "Normal", "Normal", "Normal", "Normal"],
        "plies": [40, 40, 4, 40, 40, 40, 40, 40, 40],
        "white_berserk": [False, False, False, True, False, False, False, False, False],
        "black_berserk": False,
        "white_elo": [1600, 1600, 1600, 1600, np.nan, 1600, 1100, 1600, 1600],
        "black_elo": [1600, 1600, 1600, 1600, 1600, 1600, 1100, 1600, 1600],
        "white_rating_diff": [5, 5, 5, 5, 5, 120, 5, 5, 5],
        "black_rating_diff": [-5] * 9,
        "white_title": "", "black_title": "",
    })
    kept, flow = A.apply_filters(g, 300, 480)
    assert list(kept.index) == [0]
    assert [n for _, n in flow] == [9, 7, 7, 6, 5, 4, 3, 2, 1]
