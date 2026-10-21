"""Engine self-play from every Chess960 starting position (Part II).

Each engine plays itself from each position, starting every game from one of
that position's opening-book lines. Every line is played ``games_per_line``
times with the colours swapped in pairs, so each position gets the same
number of games (50 lines x 10 games = 500 by default). Games are played by
fastchess at a fixed time control (config/selfplay.toml).

Subcommands
-----------
run      Play the games for a set of positions with one engine. One fastchess
         match per position, written to raw/<engine>/spNNN.pgn.gz once the
         match is complete. Finished positions are skipped, so rerunning the
         same command resumes.
collect  Parse every finished PGN into tables:
           tables/selfplay_games.parquet      one row per game
           tables/selfplay_positions.parquet  per engine and position: W/D/L,
                                              White's expected score and Elo
           tables/selfplay_lines.parquet      the same per book line

Examples
--------
    python code/selfplay.py run --engine stockfish --positions 518,868 --concurrency 16
    python code/selfplay.py collect
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import chess
import chess.pgn
import pandas as pd

from utils import (
    CONFIG_DIR, REPO_ROOT, EngineConfig, EngineSetupError, elo_from_expected,
    get_position, load_engine_config, load_toml, parse_position_spec, platform_info,
    setup_logging, sha256_file, utc_now, verify_engine_files, write_parquet_atomic,
)

log = logging.getLogger("selfplay")

DEFAULT_CONFIG = CONFIG_DIR / "selfplay.toml"
RESULT_WHITE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


# ---------------------------------------------------------------------------
# Settings and paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchSettings:
    tc: str
    games_per_line: int
    fastchess: Path
    draw: dict[str, Any]
    resign: dict[str, Any]
    maxmoves: int
    tablebases: str
    tbpieces: int
    uci_overrides: dict[str, Any]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MatchSettings":
        m = config["match"]
        games = int(m["games_per_line"])
        if games < 2 or games % 2:
            raise ValueError("games_per_line must be even and at least 2")
        path = Path(m["fastchess"])
        return cls(
            tc=str(m["tc"]), games_per_line=games,
            fastchess=path if path.is_absolute() else REPO_ROOT / path,
            draw=dict(m.get("draw", {})), resign=dict(m.get("resign", {})),
            maxmoves=int(m.get("maxmoves", 0)), tablebases=str(m.get("tablebases", "")),
            tbpieces=int(m.get("tbpieces", 5)),
            uci_overrides=dict(config.get("uci_overrides", {})),
        )


def output_dir(config: dict[str, Any], override: str | None) -> Path:
    path = Path(override) if override else REPO_ROOT / config["output"]["dir"]
    return path if path.is_absolute() else REPO_ROOT / path


def book_path(config: dict[str, Any]) -> Path:
    path = Path(config["book"]["lines"])
    return path if path.is_absolute() else REPO_ROOT / path


def raw_path(out_dir: Path, engine_key: str, pos_id: int) -> Path:
    return out_dir / "raw" / engine_key / f"sp{pos_id:03d}.pgn.gz"


def load_book(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"opening book not found at {path} "
                                "(run opening_book_generator.py collect)")
    cols = ["position_id", "line_index", "start_fen", "moves_uci", "moves_san"]
    return pd.read_parquet(path, columns=cols).sort_values(["position_id", "line_index"])


# ---------------------------------------------------------------------------
# Running fastchess
# ---------------------------------------------------------------------------

def write_openings(lines: pd.DataFrame, path: Path) -> None:
    """One position's book lines as a PGN, in line_index order: fastchess plays
    them in this order (round r uses line (r - 1) mod n)."""
    with open(path, "w") as fh:
        for row in lines.itertuples():
            board = chess.Board(row.start_fen, chess960=True)
            game = chess.pgn.Game()
            game.setup(board)
            game.headers["Variant"] = "Chess960"
            game.headers["PositionId"] = str(row.position_id)
            game.headers["LineIndex"] = str(row.line_index)
            node = game
            for uci in row.moves_uci.split():
                node = node.add_variation(chess.Move.from_uci(uci))
            print(game, file=fh, end="\n\n")


def engine_args(cfg: EngineConfig, label: str, s: MatchSettings) -> list[str]:
    opts = {**cfg.uci_options, **s.uci_overrides}
    if cfg.network is not None and cfg.network_option:
        opts[cfg.network_option] = str(cfg.network.resolve())
    args = ["-engine", f"cmd={cfg.binary.resolve()}", f"name={label}",
            f"dir={cfg.binary.parent.resolve()}"]
    for name, value in opts.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        args.append(f"option.{name}={value}")
    return args


def fastchess_command(cfg: EngineConfig, s: MatchSettings, openings: Path, pgn: Path,
                      n_lines: int, concurrency: int, log_file: Path | None = None) -> list[str]:
    """Two copies of the same engine (<engine>_a and <engine>_b); every
    opening is played twice per round with colours swapped (-repeat)."""
    rounds = n_lines * s.games_per_line // 2
    cmd = [str(s.fastchess), "-variant", "fischerandom"]
    cmd += engine_args(cfg, f"{cfg.key}_a", s) + engine_args(cfg, f"{cfg.key}_b", s)
    cmd += ["-each", f"tc={s.tc}", "proto=uci",
            "-openings", f"file={openings}", "format=pgn", "order=sequential",
            "-rounds", str(rounds), "-repeat", "-concurrency", str(concurrency),
            "-recover", "-ratinginterval", "0", "-autosaveinterval", "0",
            "-pgnout", f"file={pgn}", "notation=san", "nodes=true", "seldepth=true"]
    if s.draw:
        cmd += ["-draw"] + [f"{k}={v}" for k, v in s.draw.items()]
    if s.resign:
        cmd += ["-resign"] + [f"{k}={str(v).lower()}" for k, v in s.resign.items()]
    if s.maxmoves:
        cmd += ["-maxmoves", str(s.maxmoves)]
    if s.tablebases:
        cmd += ["-tb", s.tablebases, "-tbpieces", str(s.tbpieces)]
    if log_file is not None:
        cmd += ["-log", f"file={log_file}", "level=info"]
    return cmd


def fastchess_version(binary: Path) -> str:
    try:
        out = subprocess.run([str(binary), "-version"], capture_output=True, text=True,
                             timeout=30)
        return out.stdout.strip() or out.stderr.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unknown ({exc})"


def count_games(pgn: Path) -> int:
    if not pgn.exists():
        return 0
    with open(pgn, errors="replace") as fh:
        return sum(1 for line in fh if line.startswith("[Result "))


def play_position(cfg: EngineConfig, s: MatchSettings, lines: pd.DataFrame, dest: Path,
                  concurrency: int, work_root: Path, progress_seconds: float = 300
                  ) -> dict[str, Any]:
    """Play one position's match and write it to ``dest`` (gzip) atomically."""
    n_lines = len(lines)
    expected = n_lines * s.games_per_line
    work = Path(tempfile.mkdtemp(prefix=f"sp{int(lines.position_id.iloc[0]):03d}_",
                                 dir=work_root))
    try:
        openings, pgn = work / "openings.pgn", work / "games.pgn"
        write_openings(lines, openings)
        cmd = fastchess_command(cfg, s, openings, pgn, n_lines, concurrency,
                                log_file=work / "fastchess.log")
        started = time.monotonic()
        # fastchess writes config.json in its working directory: keep it in work/.
        out_file = work / "output.txt"
        with open(out_file, "w") as out:
            proc = subprocess.Popen(cmd, cwd=work, stdout=out, stderr=subprocess.STDOUT)
            # Report progress every few minutes so a long match is visible in
            # the job log (games finished so far, from the PGN).
            while True:
                try:
                    proc.wait(timeout=progress_seconds)
                    break
                except subprocess.TimeoutExpired:
                    done = count_games(pgn)
                    elapsed = time.monotonic() - started
                    eta = elapsed / done * (expected - done) if done else math.nan
                    log.info("  sp%03d: %d/%d games after %.0f min (about %.0f min to go)",
                             int(lines.position_id.iloc[0]), done, expected, elapsed / 60,
                             eta / 60)
        seconds = time.monotonic() - started
        output = out_file.read_text(errors="replace")
        games = count_games(pgn)
        if proc.returncode != 0 or games != expected:
            tail = "\n".join(output.strip().splitlines()[-20:])
            status = (f"crashed (signal {-proc.returncode})" if proc.returncode < 0
                      else f"exit {proc.returncode}")
            # Keep fastchess's files (log, openings, config.json, any games)
            # for diagnosis; logs/ is not committed.
            keep = REPO_ROOT / "logs" / "selfplay_failed" / f"{cfg.key}_{work.name}"
            shutil.copytree(work, keep, dirs_exist_ok=True)
            (keep / "command.txt").write_text(" ".join(cmd) + "\n")
            raise RuntimeError(f"fastchess {status}, {games}/{expected} games "
                               f"(files kept in {keep})\n{tail}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        with open(pgn, "rb") as src, gzip.open(tmp, "wb", compresslevel=6) as out:
            shutil.copyfileobj(src, out)
        os.replace(tmp, dest)
        return {"games": games, "seconds": round(seconds, 1)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def cmd_run(args: argparse.Namespace) -> int:
    config = load_toml(Path(args.config))
    s = MatchSettings.from_config(config)
    out_dir = output_dir(config, args.output_dir)
    if args.engine not in config["match"]["engines"]:
        log.warning("%s is not in [match].engines (%s); running anyway",
                    args.engine, ", ".join(config["match"]["engines"]))
    if not s.fastchess.exists():
        log.error("fastchess not found at %s (run tools/fastchess/build_linux.sh)", s.fastchess)
        return 2
    cfg = load_engine_config(args.engine)
    try:
        fingerprint = verify_engine_files(cfg, strict=not args.allow_unlocked)
    except EngineSetupError as exc:
        log.error("%s", exc)
        return 2

    book = load_book(book_path(config))
    pos_ids = parse_position_spec(args.positions)
    todo = [p for p in pos_ids if args.overwrite or not raw_path(out_dir, cfg.key, p).exists()]
    concurrency = args.concurrency or len(os.sched_getaffinity(0))
    log.info("self-play %s %s | tc %s | %d games per line | concurrency %d | "
             "%d positions (%d already done)", cfg.name, cfg.version, s.tc, s.games_per_line,
             concurrency, len(todo), len(pos_ids) - len(todo))
    if not todo:
        return 0

    run_id = uuid.uuid4().hex[:12]
    meta = {
        "run_id": run_id, "engine": cfg.key, "engine_name": cfg.name,
        "engine_version": cfg.version, **fingerprint,
        "uci_options": {**cfg.uci_options, **s.uci_overrides},
        "fastchess_version": fastchess_version(s.fastchess),
        "fastchess_sha256": sha256_file(s.fastchess),
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(s).items()},
        "book": str(book_path(config).relative_to(REPO_ROOT)) if book_path(config).is_relative_to(REPO_ROOT) else str(book_path(config)),
        "book_sha256": sha256_file(book_path(config)),
        "concurrency": concurrency, "positions": todo, "started": utc_now(),
        "command_example": fastchess_command(cfg, s, Path("openings.pgn"), Path("games.pgn"),
                                             50, concurrency),
        **platform_info(),
    }
    meta_path = out_dir / "runs" / f"{run_id}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    work_root = Path(args.work_dir or os.environ.get("TMPDIR") or tempfile.gettempdir())
    work_root.mkdir(parents=True, exist_ok=True)
    failed: list[int] = []
    finished: dict[int, dict[str, Any]] = {}
    for n, pos_id in enumerate(todo, 1):
        lines = book[book.position_id == pos_id]
        if lines.empty:
            log.error("sp%03d: no book lines; skipping", pos_id)
            failed.append(pos_id)
            continue
        try:
            info = play_position(cfg, s, lines, raw_path(out_dir, cfg.key, pos_id),
                                 concurrency, work_root, progress_seconds=60 * args.progress_minutes)
        except RuntimeError as exc:
            log.error("sp%03d failed: %s", pos_id, exc)
            failed.append(pos_id)
            continue
        finished[pos_id] = info
        summary = summarise_pgn(raw_path(out_dir, cfg.key, pos_id))
        log.info("[%d/%d] sp%03d %s: %d games in %.0fs | W %d D %d L %d | White E %.3f",
                 n, len(todo), pos_id, get_position(pos_id).arrangement, info["games"],
                 info["seconds"], summary["white_wins"], summary["draws"],
                 summary["black_wins"], summary["expected_score"])

    meta.update({"finished": utc_now(), "position_results": finished, "failed": failed})
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    if failed:
        log.error("%d position(s) failed: %s (rerun to retry)", len(failed),
                  ",".join(map(str, failed)))
        return 1
    log.info("done")
    return 0


# ---------------------------------------------------------------------------
# Parsing games
# ---------------------------------------------------------------------------

# Engine move comments look like "+0.28/13 0.249s, n=110878, sd=21, nps=441745"
# (score in pawns from the mover's side, or +M5 / -M3 for mates, then depth).
SCORE_RE = re.compile(r"^(?P<score>[+-]?(?:M\d+|\d+(?:\.\d+)?))/(?P<depth>\d+)\s+(?P<time>[\d.]+)s")
KV_RE = re.compile(r"\b(n|sd|nps)=(\d+)")


def parse_move_comment(comment: str) -> dict[str, Any] | None:
    """Score (cp, mover's view), mate, depth, time and nodes from an engine
    move comment; None for book moves or comments without a score."""
    comment = comment.strip()
    m = SCORE_RE.match(comment)
    if not m:
        return None
    score = m.group("score")
    cp: int | None
    mate: int | None
    if "M" in score:
        n = int(score.lstrip("+-M"))
        mate, cp = (-n if score.startswith("-") else n), None
    else:
        mate, cp = None, int(round(float(score) * 100))
    kv = dict(KV_RE.findall(comment))
    return {"cp": cp, "mate": mate, "depth": int(m.group("depth")),
            "time": float(m.group("time")),
            "nodes": int(kv["n"]) if "n" in kv else None,
            "seldepth": int(kv["sd"]) if "sd" in kv else None}


def end_reason(comment: str) -> str | None:
    """The reason fastchess writes into the last move's comment, e.g.
    'Draw by adjudication' or 'White wins by adjudication: SyzygyTB'."""
    parts = [p.strip() for p in comment.split(",")]
    for part in reversed(parts):
        if part and not SCORE_RE.match(part) and not KV_RE.match(part) and part != "book":
            return part
    return None


def iter_games(path: Path) -> Iterator[chess.pgn.Game]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as fh:
        while (game := chess.pgn.read_game(fh)) is not None:
            yield game


def game_row(game: chess.pgn.Game, engine_key: str, book_moves: dict[int, str]
             ) -> dict[str, Any]:
    """One game as a row. ``book_moves`` maps line_index -> moves_uci for the
    game's position (book lines are matched by round, then checked)."""
    h = game.headers
    board = game.board()
    pos_id = get_position_id(h.get("FEN", ""))
    played_book: list[str] = []
    stats = {chess.WHITE: [], chess.BLACK: []}
    last_comment = ""
    plies = 0
    first_eval_white: int | None = None
    for node in game.mainline():
        mover = not node.board().turn  # side that just moved
        comment = node.comment or ""
        if comment.strip().startswith("book"):
            played_book.append(node.move.uci())
        else:
            parsed = parse_move_comment(comment)
            if parsed:
                stats[mover].append(parsed)
                if first_eval_white is None and parsed["cp"] is not None:
                    first_eval_white = parsed["cp"] if mover == chess.WHITE else -parsed["cp"]
        last_comment = comment
        plies += 1

    rnd = int(h.get("Round", "0") or 0)
    n_lines = len(book_moves)
    line_index = (rnd - 1) % n_lines if n_lines and rnd > 0 else -1
    book_str = " ".join(played_book)
    line_match = book_moves.get(line_index) == book_str
    if not line_match:
        matches = [i for i, mv in book_moves.items() if mv == book_str]
        line_index = matches[0] if matches else -1

    white_label = h.get("White", "")
    result = h.get("Result", "*")

    def mean(side: chess.Color, key: str) -> float:
        vals = [x[key] for x in stats[side] if x[key] is not None]
        return float(sum(vals) / len(vals)) if vals else math.nan

    return {
        "engine": engine_key,
        "position_id": pos_id,
        "line_index": line_index,
        "line_matched_by_round": line_match,
        "round": rnd,
        "white_player": white_label,
        "result": result,
        "score_white": RESULT_WHITE.get(result, math.nan),
        "termination": h.get("Termination", ""),
        "end_reason": end_reason(last_comment) or "",
        "plies": plies,
        "book_plies": len(played_book),
        "first_eval_white_cp": first_eval_white,
        "mean_depth_white": mean(chess.WHITE, "depth"),
        "mean_depth_black": mean(chess.BLACK, "depth"),
        "mean_nodes_white": mean(chess.WHITE, "nodes"),
        "mean_nodes_black": mean(chess.BLACK, "nodes"),
        "time_used_white": sum(x["time"] for x in stats[chess.WHITE]),
        "time_used_black": sum(x["time"] for x in stats[chess.BLACK]),
        "duration": h.get("GameDuration", ""),
        "start_time": h.get("GameStartTime", ""),
        "time_control": h.get("TimeControl", ""),
        "engine_id_name": h.get("EngineWhiteName", ""),
    }


_FEN_IDS: dict[str, int] | None = None


def get_position_id(fen: str) -> int:
    """Starting position id from a game's FEN header (its back rank)."""
    global _FEN_IDS
    if _FEN_IDS is None:
        _FEN_IDS = {get_position(i).arrangement: i for i in range(960)}
    rank = fen.split("/")[-1].split()[0] if fen else ""
    return _FEN_IDS.get(rank, -1)


def summarise_pgn(path: Path) -> dict[str, Any]:
    results = [g.headers.get("Result", "*") for g in iter_games(path)]
    w, d, l = results.count("1-0"), results.count("1/2-1/2"), results.count("0-1")
    n = w + d + l
    return {"white_wins": w, "draws": d, "black_wins": l,
            "expected_score": (w + d / 2) / n if n else math.nan}


def score_summary(games: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """W/D/L, White's expected score with a standard error, and Elo, by group.

    The standard error uses the per-game score variance; games are not quite
    independent (pairs share an opening line), so line-level summaries are
    also written for a cluster-robust analysis."""
    g = games.dropna(subset=["score_white"]).groupby(keys)
    out = g.agg(games=("score_white", "size"),
                white_wins=("score_white", lambda x: int((x == 1).sum())),
                draws=("score_white", lambda x: int((x == 0.5).sum())),
                black_wins=("score_white", lambda x: int((x == 0).sum())),
                expected_score=("score_white", "mean"),
                score_sd=("score_white", "std"),
                mean_plies=("plies", "mean")).reset_index()
    out["draw_rate"] = out["draws"] / out["games"]
    out["se"] = out["score_sd"] / out["games"].pow(0.5)
    out["elo"] = out["expected_score"].map(elo_from_expected)
    return out


def cmd_collect(args: argparse.Namespace) -> int:
    config = load_toml(Path(args.config))
    out_dir = output_dir(config, args.output_dir)
    book = load_book(book_path(config))
    book_by_pos = {pid: dict(zip(grp.line_index, grp.moves_uci))
                   for pid, grp in book.groupby("position_id")}
    rows: list[dict[str, Any]] = []
    files = sorted((out_dir / "raw").glob("*/sp*.pgn.gz"))
    if not files:
        log.error("no PGNs under %s", out_dir / "raw")
        return 1
    for n, path in enumerate(files, 1):
        engine_key = path.parent.name
        pos_id = int(path.name[2:5])
        for game in iter_games(path):
            rows.append(game_row(game, engine_key, book_by_pos.get(pos_id, {})))
        if n % 50 == 0:
            log.info("parsed %d/%d files", n, len(files))
    games = pd.DataFrame(rows)
    bad = games[(games.line_index < 0) | (games.position_id < 0)]
    if len(bad):
        log.warning("%d games could not be matched to a book line", len(bad))
    unexpected = games[~games.termination.isin(["normal", "adjudication"])]
    if len(unexpected):
        log.warning("%d games ended abnormally: %s", len(unexpected),
                    unexpected.termination.value_counts().to_dict())

    tables = out_dir / "tables"
    write_parquet_atomic(games, tables / "selfplay_games.parquet")
    positions = score_summary(games, ["engine", "position_id"])
    positions["arrangement"] = positions.position_id.map(lambda i: get_position(i).arrangement)
    write_parquet_atomic(positions, tables / "selfplay_positions.parquet")
    positions.to_csv(tables / "selfplay_positions.csv", index=False)
    lines = score_summary(games, ["engine", "position_id", "line_index"])
    write_parquet_atomic(lines, tables / "selfplay_lines.parquet")

    for engine_key, grp in positions.groupby("engine"):
        log.info("%s: %d positions, %d games | White E %.4f (SD across positions %.4f) | "
                 "draws %.1f%% | mean %.0f plies", engine_key, len(grp), int(grp.games.sum()),
                 grp.expected_score.mean(), grp.expected_score.std(),
                 100 * grp.draws.sum() / grp.games.sum(), grp.mean_plies.mean())
    log.info("wrote %s", tables)
    return 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--output-dir", help="override [output].dir")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    # --config, --output-dir and -v also work after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS)
    common.add_argument("--output-dir", default=argparse.SUPPRESS, help="override [output].dir")
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)

    r = sub.add_parser("run", parents=[common], help="play the games for some positions")
    r.add_argument("--engine", required=True, help="engine folder under engines/")
    r.add_argument("--positions", default="0-959", help="e.g. 0-9, 518,868 or @file")
    r.add_argument("--concurrency", type=int, default=0,
                   help="games at once (default: the CPUs available to this process)")
    r.add_argument("--work-dir", help="scratch folder for fastchess (default $TMPDIR)")
    r.add_argument("--progress-minutes", type=float, default=5,
                   help="log games finished every this many minutes (default 5)")
    r.add_argument("--overwrite", action="store_true", help="replay finished positions")
    r.add_argument("--allow-unlocked", action="store_true",
                   help="allow an engine with no lock entry (testing only)")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("collect", parents=[common], help="parse finished games into tables")
    c.set_defaults(func=cmd_collect)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
