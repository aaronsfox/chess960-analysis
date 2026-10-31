"""Tests for the Freestyle Friday analysis."""

import numpy as np
import pandas as pd

import analyse_chesscom960 as A


def test_include_flow_and_ids():
    g = pd.DataFrame({"position_id": [5, 5, -1, 7], "score_white": [1.0, np.nan, 0.5, 0.0],
                      "plies": [40, 40, 40, 4], "white_elo": [2500.0] * 4, "black_elo": [2400.0] * 4,
                      "white_id": ["0a", "0b", "0c", "0d"], "black_id": ["1a", "1b", "1c", "1d"]})
    d, flow = A.include(g)
    assert len(d) == 1 and d.white_id.iloc[0] == 10
    assert flow.games.tolist() == [4, 3, 2, 1, 1]
