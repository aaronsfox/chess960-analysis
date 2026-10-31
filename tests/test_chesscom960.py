"""Tests for the Chess.com Freestyle Friday parser."""

import pandas as pd

import chesscom960 as C

PGN = """[ApiRound "3"]
[ApiTournament "freestyle-friday-january-23-2026-6160617"]
[Event "Freestyle-Friday-January-23-2026"]
[Site "Chess.com"]
[Date "2026.01.23"]
[Round "-"]
[White "PlayerA"]
[Black "PlayerB"]
[Result "1-0"]
[SetUp "1"]
[FEN "qbbrkrnn/pppppppp/8/8/8/8/PPPPPPPP/QBBRKRNN w KQkq - 0 1"]
[WhiteElo "2850"]
[BlackElo "2790"]
[TimeControl "180+1"]
[Termination "PlayerA won by resignation"]
[Variant "Chess960"]
[Link "https://www.chess.com/game/live/1"]

1. d4 {[%clk 0:03:00.9]} 1... d5 {[%clk 0:02:58.1]} 2. c4 {[%clk 0:02:59]} 1-0

[ApiRound "3"]
[ApiTournament "freestyle-friday-january-23-2026-6160617"]
[Event "Freestyle-Friday-January-23-2026"]
[Date "2026.01.23"]
[White "PlayerC"]
[Black "PlayerD"]
[Result "1/2-1/2"]
[FEN "qbbrkrnn/pppppppp/8/8/8/8/PPPPPPPP/QBBRKRNN w KQkq - 0 1"]
[WhiteElo "2600"]
[BlackElo "2610"]
[TimeControl "180+1"]
[Termination "Game drawn by agreement"]
[Variant "Chess960"]
[Link "https://www.chess.com/game/live/2"]

1. e4 {[%clk 0:03:01]} 1... e5 {[%clk 0:03:00]} 1/2-1/2
"""


def test_parse_and_summary(tmp_path):
    d = tmp_path / "ff"
    d.mkdir()
    (d / "t.pgn").write_text(PGN)
    g = C.parse(d, tmp_path / "out")
    assert len(g) == 2 and set(g.position_id) == {868}
    assert g["round"].tolist() == ["3", "3"] and g.base_s.tolist() == [180, 180]
    assert g.score_white.tolist() == [1.0, 0.5] and g.plies.tolist() == [3, 2]
    assert g.white_id.iloc[0] != "PlayerA"
    text = C.summary_text(g)
    assert "Tournaments: 1" in text and "Positions per tournament round: median 1" in text


def test_date_from_id():
    assert C.date_from_id("freestyle-friday-june-20-2025-5734315") == "2025-06-20"
