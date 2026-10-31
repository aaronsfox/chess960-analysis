"""Tests for the human vs engine Dirichlet comparison."""

import numpy as np
import pandas as pd

import compare_levels as C


def test_posteriors_and_summary():
    counts = pd.DataFrame({"source": ["lichess_blitz", "stockfish"], "position_id": [518, 518],
                           "games": [1000, 500], "white_wins": [500, 40], "draws": [50, 450],
                           "black_wins": [450, 10]})
    post = C.posteriors(counts, {"lichess_blitz": 1.0, "stockfish": 1.64})
    b = post.set_index("source")
    assert abs(b.loc["lichess_blitz", "decisive_mean"] - 0.95) < 0.01
    assert b.loc["stockfish", "delta_elo_lo"] < C.elo(0.53) < b.loc["stockfish", "delta_elo_hi"]
    # the design effect widens the engine interval relative to DEFF 1
    post1 = C.posteriors(counts.iloc[[1]], {"stockfish": 1.0})
    assert (b.loc["stockfish", "delta_elo_hi"] - b.loc["stockfish", "delta_elo_lo"]
            > post1.delta_elo_hi.iloc[0] - post1.delta_elo_lo.iloc[0])
    src = C.source_summary(post).set_index("source")
    assert src.loc["stockfish", "draws"] == 0.9 and abs(src.loc["stockfish", "decisive"] - 0.1) < 1e-12


def test_fill_ratings():
    import masters960 as M
    g = pd.DataFrame({"event": ["E", "E", "E"], "white": ["a", "b", "a"], "black": ["b", "a", "c"],
                      "white_elo": [2500, np.nan, np.nan], "black_elo": [2400, 2500, np.nan]})
    f = M.fill_ratings(g)
    assert f.white_elo.tolist()[:2] == [2500, 2400] and f.white_elo.iloc[2] == 2500
    assert np.isnan(f.black_elo.iloc[2])
