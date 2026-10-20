"""Tests for the self-play game parsing and match settings."""

import io
from pathlib import Path

import chess.pgn
import pytest

from selfplay import (MatchSettings, end_reason, fastchess_command, game_row,
                      parse_move_comment, score_summary)
from utils import load_engine_config

PGN = """[Event "Fastchess Tournament"]
[Round "2"]
[White "stockfish_b"]
[Black "stockfish_a"]
[Result "1-0"]
[EngineWhiteName "Stockfish 19"]
[SetUp "1"]
[FEN "qbbrkrnn/pppppppp/8/8/8/8/PPPPPPPP/QBBRKRNN w FDfd - 0 1"]
[Variant "Chess960"]
[Termination "adjudication"]
[TimeControl "10+0.1"]

1. e4 {book} e5 {book} 2. d4 {book} exd4 {book} 3. Rxd4 {book} Nf6 {book}
4. Nf3 {+0.81/18 0.412s, n=210000, sd=25, nps=509000}
Ng6 {-M7/30 0.300s, n=100000, sd=31, nps=333333, White wins by adjudication} 1-0
"""

BOOK = {0: "e2e4 e7e5 d2d4 d7d5 c2c4 c7c6", 1: "e2e4 e7e5 d2d4 e5d4 d1d4 g8f6"}


def test_parse_move_comment():
    c = parse_move_comment("+0.28/13 0.249s, n=110878, sd=21, nps=441745")
    assert c == {"cp": 28, "mate": None, "depth": 13, "time": 0.249,
                 "nodes": 110878, "seldepth": 21}
    m = parse_move_comment("-M18/23 0.006s, n=8242, sd=19, nps=1030250, White wins by adjudication")
    assert m["mate"] == -18 and m["cp"] is None
    assert parse_move_comment("book") is None


def test_end_reason():
    assert end_reason("-M6/74 0.007s, n=41376, sd=7, nps=4597333, White wins by adjudication") \
        == "White wins by adjudication"
    assert end_reason("+0.10/12 0.050s, n=1000, sd=12, nps=20000") is None


def test_game_row_maps_line_and_result():
    game = chess.pgn.read_game(io.StringIO(PGN))
    row = game_row(game, "stockfish", BOOK)
    assert row["position_id"] == 868
    assert row["line_index"] == 1 and row["line_matched_by_round"]
    assert row["score_white"] == 1.0
    assert row["book_plies"] == 6 and row["plies"] == 8
    assert row["first_eval_white_cp"] == 81
    assert row["end_reason"] == "White wins by adjudication"


def test_game_row_falls_back_to_move_matching():
    game = chess.pgn.read_game(io.StringIO(PGN.replace('[Round "2"]', '[Round "1"]')))
    row = game_row(game, "stockfish", BOOK)
    assert row["line_index"] == 1 and not row["line_matched_by_round"]


def test_score_summary():
    import pandas as pd
    games = pd.DataFrame({"engine": "x", "position_id": 1, "score_white": [1, 0.5, 0.5, 0],
                          "plies": 100})
    s = score_summary(games, ["engine", "position_id"]).iloc[0]
    assert (s.white_wins, s.draws, s.black_wins) == (1, 2, 1)
    assert s.expected_score == 0.5 and abs(s.elo) < 1e-9


def test_settings_and_command():
    config = {"match": {"tc": "10+0.1", "games_per_line": 10, "fastchess": "tools/fc",
                        "draw": {"movenumber": 40}, "resign": {"twosided": True}},
              "uci_overrides": {"Hash": 64}}
    s = MatchSettings.from_config(config)
    cmd = fastchess_command(load_engine_config("stockfish"), s, Path("o.pgn"), Path("g.pgn"),
                            n_lines=50, concurrency=8)
    assert cmd[cmd.index("-rounds") + 1] == "250"
    assert "-repeat" in cmd and "option.Hash=64" in cmd
    assert "twosided=true" in cmd and "tc=10+0.1" in cmd
    with pytest.raises(ValueError):
        MatchSettings.from_config({**config, "match": {**config["match"], "games_per_line": 5}})
