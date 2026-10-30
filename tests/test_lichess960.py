"""Tests for the Lichess Chess960 parser."""

import io

from lichess960 import category, game_row, iter_games, month_range, position_from_fen

PGN = """[Event "Rated Chess960 game"]
[Site "https://lichess.org/AbCdEfGh"]
[Date "2026.09.01"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[UTCDate "2026.09.01"]
[UTCTime "10:00:00"]
[WhiteElo "1850"]
[BlackElo "1790"]
[WhiteRatingDiff "+5"]
[BlackRatingDiff "-5"]
[Variant "Chess960"]
[TimeControl "300+3"]
[Termination "Normal"]
[FEN "qbbrkrnn/pppppppp/8/8/8/8/PPPPPPPP/QBBRKRNN w KQkq - 0 1"]
[SetUp "1"]

1. d4 { [%clk 0:05:00] } 1... d5 { [%clk 0:02:30] } 2. c4 { [%clk 0:05:01] } 1-0

[Event "Rated Chess960 tournament https://lichess.org/tournament/x"]
[Site "https://lichess.org/IjKlMnOp"]
[White "carol"]
[Black "dave"]
[Result "1/2-1/2"]
[WhiteElo "2010"]
[BlackElo "2005"]
[Variant "Chess960"]
[TimeControl "60+0"]
[Termination "Normal"]
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

1. e4 { [%clk 0:01:00] } 1/2-1/2

[Event "Casual Chess960 game"]
[Site "https://lichess.org/QqQqQqQq"]
[White "erin"]
[Black "frank"]
[Result "0-1"]
[Variant "Chess960"]
[TimeControl "600+0"]
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

1. e4 0-1
"""


def test_iter_games_splits_three_games():
    games = list(iter_games(io.StringIO(PGN)))
    assert len(games) == 3
    assert games[0][0]["White"] == "alice"


def test_game_row_fields_and_filters():
    games = list(iter_games(io.StringIO(PGN)))
    row = game_row(*games[0], month="2026-09", min_seconds=180)
    assert row["position_id"] == 868 and row["score_white"] == 1.0
    assert row["estimated_s"] == 420 and row["category"] == "blitz"
    assert row["plies"] == 3 and not row["white_berserk"] and row["black_berserk"]
    assert row["white_id"] != "alice" and len(row["white_id"]) == 12
    assert row["game_id"] == "AbCdEfGh" and not row["tournament"]
    # bullet game below the minimum; casual game not rated
    assert game_row(*games[1], month="2026-09", min_seconds=180) is None
    assert game_row(*games[1], month="2026-09", min_seconds=0)["tournament"]
    assert game_row(*games[2], month="2026-09", min_seconds=0) is None


def test_helpers():
    assert month_range("2025-11:2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]
    assert category(420) == "blitz" and category(900) == "rapid" and category(2000) == "classical"
    assert position_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") == 518
