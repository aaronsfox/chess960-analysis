"""Tests for code/utils.py. Run with `python -m pytest` from the repo root."""

import math

import chess
import pytest

import utils


def test_960_distinct_positions():
    arrangements = {p.arrangement for p in utils.all_positions()}
    assert len(arrangements) == 960


def test_standard_position_is_518():
    pos = utils.get_position(utils.STANDARD_POSITION_ID)
    assert pos.arrangement == "RNBQKBNR"
    assert pos.board().board_fen() == chess.Board().board_fen()


@pytest.mark.parametrize("pos", utils.all_positions(), ids=lambda p: f"sp{p.id:03d}")
def test_chess960_rules(pos):
    a = pos.arrangement
    bishops = [i for i, piece in enumerate(a) if piece == "B"]
    assert bishops[0] % 2 != bishops[1] % 2, "bishops must be on opposite colours"
    rooks = [i for i, piece in enumerate(a) if piece == "R"]
    assert rooks[0] < a.index("K") < rooks[1], "king must be between the rooks"
    assert sorted(a) == sorted("RNBQKBNR")


def test_mirror_pairs():
    for pos_id in range(960):
        m = utils.mirror_id(pos_id)
        assert m != pos_id, "no position is its own mirror"
        assert utils.mirror_id(m) == pos_id
        assert utils.get_position(m).arrangement == utils.get_position(pos_id).arrangement[::-1]


def test_parse_position_spec(tmp_path):
    assert utils.parse_position_spec("0-3,518,2") == [0, 1, 2, 3, 518]
    assert len(utils.parse_position_spec("0-959")) == 960
    f = tmp_path / "ids.txt"
    f.write_text("# pilot\n518  # standard\n80\n")
    assert utils.parse_position_spec(f"@{f}") == [80, 518]
    with pytest.raises(ValueError):
        utils.parse_position_spec("960")


def test_expected_score_and_elo():
    assert utils.expected_score_from_wdl(100, 800, 100) == pytest.approx(0.5)
    assert utils.expected_score_from_wdl(200, 750, 50) == pytest.approx(0.575)
    assert utils.elo_from_expected(0.5) == pytest.approx(0.0)
    assert utils.elo_from_expected(0.64) == pytest.approx(100.0, abs=0.5)
    assert math.isnan(utils.elo_from_expected(math.nan))


def test_convergence_measures():
    values = [0.55, 0.53, 0.54, 0.54]
    assert utils.total_variation(values) == pytest.approx(0.03)
    assert utils.tail_drift(values, k=3) == pytest.approx(0.01)
    assert math.isnan(utils.tail_drift([0.5]))


def test_engine_manifests_load():
    keys = utils.list_engines()
    assert set(keys) == {"stockfish", "lc0", "caissa", "plentychess"}
    for key in keys:
        cfg = utils.load_engine_config(key)
        managed = {name.lower() for name in cfg.uci_options} & utils.MANAGED_OPTIONS
        assert not managed, f"{key} sets options python-chess manages: {managed}"
        assert cfg.uci_options.get("Threads") == 1, f"{key} must run single-threaded"
