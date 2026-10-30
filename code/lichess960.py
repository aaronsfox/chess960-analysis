"""Lichess rated Chess960 games: download, parse and summarise (human play).

Streams the monthly Lichess database exports of rated Chess960 games
(https://database.lichess.org/chess960/, CC0) and writes one row per game to
a Parquet file per month. Nothing is decompressed to disk: each file is
streamed from the web (or read from a local .pgn.zst) through a zstd
decompressor and parsed line by line.

Needs internet access (the cluster login node), or files already downloaded
with wget and passed with --pgn-dir.

Subcommands
-----------
fetch      Download and parse months, e.g. --months 2026-07:2026-09.
           Writes results/lichess960/raw/lichess960_YYYY-MM.parquet
           (finished months are skipped, so rerunning resumes).
summarise  Counts to decide the analysis: games by month and time-control
           category, games per starting position, rating levels and gaps,
           tournament and berserk shares, terminations. Prints them and
           writes results/lichess960/lichess960_summary.txt.

One row per game:
  month, game_id, utc_datetime, event, tournament (arena/swiss, else lobby),
  white_id, black_id (salted SHA-1 of the usernames, 12 hex characters: lets
  a model use player random effects without storing names), white_elo,
  black_elo, white_rating_diff, black_rating_diff, white_title, black_title,
  result, score_white (1, 0.5, 0), time_control, base_s, increment_s,
  estimated_s (base + 40 x increment, Lichess's definition), category
  (ultrabullet/bullet/blitz/rapid/classical), termination, fen, position_id
  (0-959 from the starting back rank), plies (from clock comments),
  white_berserk, black_berserk (first clock at most half the base time).

Filters applied while parsing: rated Chess960 games only, a known result,
and estimated duration at least --min-seconds (default 180, i.e. blitz and
slower; the category column allows any stricter cut later).

Examples
--------
    pip install zstandard                      # or have the `zstd` command available
    python code/lichess960.py fetch --months 2026-07:2026-09
    python code/lichess960.py summarise
    # or download first, then parse from disk
    wget -P data/lichess https://database.lichess.org/chess960/lichess_db_chess960_rated_2026-09.pgn.zst
    python code/lichess960.py fetch --months 2026-09 --pgn-dir data/lichess
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterator, TextIO

import pandas as pd

from utils import REPO_ROOT, get_position, setup_logging, write_parquet_atomic

log = logging.getLogger("lichess960")

URL = "https://database.lichess.org/chess960/lichess_db_chess960_rated_{month}.pgn.zst"
OUT_DIR = REPO_ROOT / "results" / "lichess960"
SALT = "chess960-fairness"           # fixed, so IDs are stable across months
RESULT_SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}

HEADER_RE = re.compile(r'^\[(\w+) "(.*)"\]\s*$')
CLK_RE = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def month_range(spec: str) -> list[str]:
    """'2026-07:2026-09' -> ['2026-07', '2026-08', '2026-09']; '2026-09' -> ['2026-09']."""
    first, _, last = spec.partition(":")
    last = last or first
    y, m = map(int, first.split("-"))
    y1, m1 = map(int, last.split("-"))
    out = []
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def category(estimated_s: float) -> str:
    """Lichess speed categories from estimated duration (base + 40 x inc)."""
    if estimated_s < 30:
        return "ultrabullet"
    if estimated_s < 180:
        return "bullet"
    if estimated_s < 480:
        return "blitz"
    if estimated_s < 1500:
        return "rapid"
    return "classical"


def parse_time_control(tc: str) -> tuple[float, float] | None:
    if "+" not in tc:
        return None            # "-" = correspondence / unlimited
    base, inc = tc.split("+", 1)
    try:
        return float(base), float(inc)
    except ValueError:
        return None


_ARRANGEMENTS: dict[str, int] | None = None


def position_from_fen(fen: str) -> int:
    global _ARRANGEMENTS
    if _ARRANGEMENTS is None:
        _ARRANGEMENTS = {get_position(i).arrangement: i for i in range(960)}
    try:
        rank1 = fen.split()[0].split("/")[-1]
    except IndexError:
        return -1
    return _ARRANGEMENTS.get(rank1, -1)


def player_id(name: str) -> str:
    return hashlib.sha1(f"{SALT}:{name.lower()}".encode()).hexdigest()[:12]


def to_int(value: str | None) -> float:
    try:
        return float(int(value))          # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def iter_games(stream: TextIO) -> Iterator[tuple[dict[str, str], str]]:
    """(headers, movetext) for each game in a PGN text stream."""
    headers: dict[str, str] = {}
    moves: list[str] = []
    in_moves = False
    for line in stream:
        line = line.rstrip("\n")
        if line.startswith("["):
            if in_moves:                       # a new game starts
                yield headers, " ".join(moves)
                headers, moves, in_moves = {}, [], False
            m = HEADER_RE.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
        elif line.strip():
            in_moves = True
            moves.append(line)
    if headers:
        yield headers, " ".join(moves)


def game_row(h: dict[str, str], movetext: str, month: str, min_seconds: float) -> dict | None:
    """One game as a row, or None if it fails the filters."""
    event = h.get("Event", "")
    if not event.lower().startswith("rated") or h.get("Variant", "").lower() != "chess960":
        return None
    result = h.get("Result", "*")
    if result not in RESULT_SCORE:
        return None
    tc = parse_time_control(h.get("TimeControl", "-"))
    if tc is None:
        return None
    base, inc = tc
    est = base + 40 * inc
    if est < min_seconds:
        return None
    clocks = CLK_RE.findall(movetext)
    secs = [int(a) * 3600 + int(b) * 60 + float(c) for a, b, c in clocks]
    white_berserk = bool(secs) and base > 0 and secs[0] <= base / 2 + 1
    black_berserk = len(secs) > 1 and base > 0 and secs[1] <= base / 2 + 1
    fen = h.get("FEN", "")
    site = h.get("Site", "")
    return {
        "month": month,
        "game_id": site.rsplit("/", 1)[-1],
        "utc_datetime": f"{h.get('UTCDate', '')} {h.get('UTCTime', '')}".strip(),
        "event": event,
        "tournament": ("tournament" in event.lower()) or ("swiss" in event.lower()),
        "white_id": player_id(h.get("White", "")),
        "black_id": player_id(h.get("Black", "")),
        "white_elo": to_int(h.get("WhiteElo")),
        "black_elo": to_int(h.get("BlackElo")),
        "white_rating_diff": to_int(h.get("WhiteRatingDiff")),
        "black_rating_diff": to_int(h.get("BlackRatingDiff")),
        "white_title": h.get("WhiteTitle", ""),
        "black_title": h.get("BlackTitle", ""),
        "result": result,
        "score_white": RESULT_SCORE[result],
        "time_control": h.get("TimeControl", ""),
        "base_s": base,
        "increment_s": inc,
        "estimated_s": est,
        "category": category(est),
        "termination": h.get("Termination", ""),
        "fen": fen,
        "position_id": position_from_fen(fen),
        "plies": len(secs),
        "white_berserk": white_berserk,
        "black_berserk": black_berserk,
    }


def open_zst_stream(source: str) -> tuple[TextIO, object]:
    """Text stream of a .pgn.zst from a URL or local path. Uses the
    `zstandard` Python package if installed, else the `zstd` command."""
    try:
        import zstandard  # type: ignore
        raw = urllib.request.urlopen(source) if source.startswith("http") else open(source, "rb")
        reader = zstandard.ZstdDecompressor().stream_reader(raw, read_size=1 << 20)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace"), raw
    except ImportError:
        if not shutil.which("zstd"):
            raise SystemExit("need the zstandard package (pip install zstandard) or the zstd command")
        if source.startswith("http"):
            cmd = f"curl -sSfL '{source}' | zstd -dc"
        else:
            cmd = f"zstd -dc '{source}'"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        return io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace"), proc


def fetch_month(month: str, out_dir: Path, min_seconds: float, local: str | None = None) -> Path:
    dest = out_dir / "raw" / f"lichess960_{month}.parquet"
    source = local or URL.format(month=month)
    log.info("%s: reading %s", month, source)
    stream, handle = open_zst_stream(source)
    rows, seen = [], 0
    for h, mt in iter_games(stream):
        seen += 1
        row = game_row(h, mt, month, min_seconds)
        if row is not None:
            rows.append(row)
        if seen % 500_000 == 0:
            log.info("  %s: %d games read, %d kept", month, seen, len(rows))
    if isinstance(handle, subprocess.Popen):
        handle.wait()
    df = pd.DataFrame(rows)
    write_parquet_atomic(df, dest)
    log.info("%s: %d games read, %d kept -> %s", month, seen, len(df), dest)
    return dest


def cmd_fetch(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    for month in month_range(args.months):
        dest = out_dir / "raw" / f"lichess960_{month}.parquet"
        if dest.exists() and not args.overwrite:
            log.info("%s: already done", month)
            continue
        local = None
        if args.pgn_dir:
            cand = Path(args.pgn_dir) / Path(URL.format(month=month)).name
            if cand.exists():
                local = str(cand)
            else:
                log.warning("%s: %s not found, downloading instead", month, cand)
        fetch_month(month, out_dir, args.min_seconds, local=local)
    return 0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summary_text(g: pd.DataFrame) -> str:
    L = ["Lichess rated Chess960: what we've got", "=" * 38, ""]
    L.append(f"Games: {len(g):,} from {g.month.nunique()} month(s) ({g.month.min()} to {g.month.max()}); "
             f"players: {pd.concat([g.white_id, g.black_id]).nunique():,}")
    L.append(f"Unrecognised starting positions: {(g.position_id < 0).sum()}")
    L += ["", "By month and category:",
          g.pivot_table(index="month", columns="category", values="game_id", aggfunc="size",
                        fill_value=0).to_string()]
    L += ["", "Most common time controls:", g.time_control.value_counts().head(12).to_string()]
    for label, sub in (("all kept", g), ("5 min+ (estimated >= 300 s)", g[g.estimated_s >= 300]),
                       ("rapid+ (>= 480 s)", g[g.estimated_s >= 480])):
        per = sub[sub.position_id >= 0].groupby("position_id").size().reindex(range(960), fill_value=0)
        L.append(f"\nGames per position, {label}: total {len(sub):,}; min {per.min()}, "
                 f"median {per.median():.0f}, max {per.max()}")
    mean_r = (g.white_elo + g.black_elo) / 2
    gap = g.white_elo - g.black_elo
    L += ["", "Ratings (mean of the two players): " + ", ".join(
        f"p{q}: {mean_r.quantile(q / 100):.0f}" for q in (5, 25, 50, 75, 95))]
    L.append("Rating gap White - Black: " + ", ".join(
        f"p{q}: {gap.quantile(q / 100):+.0f}" for q in (5, 25, 50, 75, 95)))
    bands = pd.cut(mean_r, [0, 1500, 1800, 2000, 2200, 4000],
                   labels=["<1500", "1500-1799", "1800-1999", "2000-2199", "2200+"])
    L.append("Games by mean-rating band: " + ", ".join(
        f"{k} {v:,}" for k, v in bands.value_counts().sort_index().items()))
    L += ["", f"Tournament games: {100 * g.tournament.mean():.1f}%; berserk (either side): "
          f"{100 * (g.white_berserk | g.black_berserk).mean():.1f}%",
          "Terminations: " + ", ".join(f"{k} {v:,}" for k, v in g.termination.value_counts().items()),
          "Results: White {:.1f}%, draw {:.1f}%, Black {:.1f}%; mean White score {:.4f}".format(
              100 * (g.score_white == 1).mean(), 100 * (g.score_white == 0.5).mean(),
              100 * (g.score_white == 0).mean(), g.score_white.mean())]
    L.append(f"Games with fewer than 10 plies: {(g.plies < 10).sum():,}; titled players in "
             f"{100 * ((g.white_title != '') | (g.black_title != '')).mean():.1f}% of games")
    return "\n".join(L) + "\n"


def cmd_summarise(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    files = sorted((out_dir / "raw").glob("lichess960_*.parquet"))
    if not files:
        log.error("no parsed months under %s", out_dir / "raw")
        return 1
    g = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    text = summary_text(g)
    (out_dir / "lichess960_summary.txt").write_text(text)
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", default=str(OUT_DIR))
    common.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", parents=[common], help="download and parse months")
    f.add_argument("--months", required=True, help="YYYY-MM or YYYY-MM:YYYY-MM")
    f.add_argument("--min-seconds", type=float, default=180,
                   help="keep games with base + 40 x increment >= this (default 180: blitz+)")
    f.add_argument("--overwrite", action="store_true")
    f.add_argument("--pgn-dir", help="read lichess_db_chess960_rated_YYYY-MM.pgn.zst from this "
                   "folder when present (e.g. downloaded with wget), instead of streaming")
    f.set_defaults(func=cmd_fetch)
    s = sub.add_parser("summarise", parents=[common], help="summary counts of the parsed games")
    s.set_defaults(func=cmd_summarise)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
