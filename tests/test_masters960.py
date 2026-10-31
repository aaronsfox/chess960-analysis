"""Tests for the master-level Chess960 parser."""

import io

import numpy as np
import pandas as pd

import masters960 as M
from lichess960 import iter_games

PGN = """[Event "Freestyle Paris KO"]
[Site "Paris"]
[Date "2025.04.12"]
[Round "1.1"]
[White "Carlsen, Magnus"]
[Black "Caruana, Fabiano"]
[Result "1-0"]
[WhiteElo "2833"]
[BlackElo "2795"]
[Variant "Chess960"]
[FEN "qbbrkrnn/pppppppp/8/8/8/8/PPPPPPPP/QBBRKRNN w KQkq - 0 1"]

1. d4 {[%clk 1:30:20]} d5 {[%clk 1:29:50]} 2. c4 {[%clk 1:29:00]} e6 3. Nc3 c6 4. e3 Nd6 5. b3 1-0

[Event "Freestyle Paris KO"]
[Round "1.5"]
[White "Caruana, Fabiano"]
[Black "Carlsen, Magnus"]
[Result "1/2-1/2"]
[WhiteElo "2795"]
[BlackElo "2833"]
[FEN "qbbrkrnn/pppppppp/8/8/8/8/PPPPPPPP/QBBRKRNN w KQkq - 0 1"]

1. d4 {[%clk 0:10:05]} d5 {[%clk 0:10:04]} 2. c4 e6 3. Nc3 c6 4. e3 Nd6 5. b3 (5. Nf3 Nf5) b6 1/2-1/2

[Event "Freestyle Paris KO Armageddon"]
[Round "1.7"]
[White "Caruana, Fabiano"]
[Black "Carlsen, Magnus"]
[Result "0-1"]
[WhiteElo "2795"]
[BlackElo "2833"]
[FEN "qbbrkrnn/pppppppp/8/8/8/8/PPPPPPPP/QBBRKRNN w KQkq - 0 1"]

1. d4 {[%clk 0:05:00]} d5 {[%clk 0:04:00]} 2. c4 e6 3. Nc3 c6 4. e3 Nd6 5. b3 b6 0-1

[Event "Freestyle Paris KO"]
[Round "2.1"]
[White "A"]
[Black "B"]
[Result "1-0"]
[WhiteElo "2700"]
[BlackElo "2650"]
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

1-0
"""

SRC = {"file": "freeparisko25.pgn", "event": "Paris", "year": 2025, "stage": "knockout",
       "base_minutes": 90}


def rows():
    return pd.DataFrame([M.game_record(h, mt, SRC) for h, mt in iter_games(io.StringIO(PGN))])


def test_records():
    g = rows()
    assert list(g.position_id) == [868, 868, 868, 518]
    assert list(g.tc_class) == ["classical", "rapid", "blitz", "classical"]
    assert list(g.armageddon) == [False, False, True, False]
    assert list(g.plies) == [9, 10, 10, 0]
    assert g.score_white.tolist() == [1.0, 0.5, 0.0, 1.0]


def test_variant_tag():
    assert M.is_960_variant("Chess960") and M.is_960_variant("chess 960")
    assert M.is_960_variant("Fischerandom") and M.is_960_variant("Fischer Random")
    assert not M.is_960_variant("") and not M.is_960_variant("Standard")
    assert rows().variant_960.tolist() == [True, False, False, False]


def test_include_flow():
    d, flow = M.include(rows())
    assert len(d) == 1 and d.iloc[0]["round"] == "1.5"
    # game 4 is the standard start with no Variant tag: not counted as Chess960
    assert flow.games.tolist() == [4, 3, 3, 3, 3, 2, 1, 1]


def test_advantage_offset():
    rng = np.random.default_rng(0)
    n = 4000
    we = rng.normal(2600, 100, n)
    be = rng.normal(2600, 100, n)
    p = 1 / (1 + 10 ** (-(we - be + 20) / 400))
    d = pd.DataFrame({"white_elo": we, "black_elo": be, "score_white": (rng.random(n) < p).astype(float)})
    a = M.advantage(d)
    assert a["advantage_lo"] < 20 < a["advantage_hi"]


def test_event_filter_and_zip(tmp_path):
    import zipfile
    other = PGN.replace("Freestyle Paris KO", "Some Open")
    buf = tmp_path / "w.zip"
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("twic.pgn", PGN + "\n" + other)
    body = M.pgn_from_zip(buf.read_bytes())
    data = tmp_path / "data"
    data.mkdir()
    (data / "weekly.pgn").write_bytes(body)
    src = pd.DataFrame([{**SRC, "file": "weekly.pgn", "event_filter": "Paris"}])
    g = M.parse(src, data, tmp_path / "out")
    assert len(g) == 4 and set(g.pgn_event.str.contains("Paris")) == {True}


def test_round_position_ok():
    g = pd.DataFrame({"source": "s", "round": [f"1.{i}" for i in range(12)] + ["2.1", "2.2"],
                      "position_id": [5] * 11 + [7] + [9, 8]})
    ok = M.round_position_ok(g)
    assert ok.tolist() == [True] * 11 + [False] + [True, True]
