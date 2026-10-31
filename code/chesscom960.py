"""Chess.com Freestyle Friday: weekly Chess960 tournaments for titled players.

Freestyle Friday (from January 2025; back in October 2026 as qualifying for
the 2027 FIDE Freestyle World Championship) is an 11-round Swiss at 3+1 for
titled players on Chess.com. Games come from Chess.com's Published-Data API
(https://api.chess.com/pub/...), which is public and needs no login.

Subcommands
-----------
discover  Find the tournaments. Starting from the seed tournaments below, takes
          their most frequent players and reads those players' monthly game
          archives; every Chess960 game played in a tournament whose URL
          contains "freestyle-friday" names its tournament. Writes
          data/chesscom/freestyle_friday_tournaments.csv (new IDs can also be
          added there by hand).
fetch     For every tournament listed: its rounds and groups, and each group's
          games with PGN. One PGN file per tournament in
          data/chesscom/ff/<id>.pgn (finished tournaments are skipped).
parse     All tournament PGNs to results/chesscom960/chesscom960_games.parquet:
          tournament, date, round, hashed player IDs, Chess.com ratings,
          result, termination, time control, starting position (0-959 from the
          FEN), plies and first clocks.
summarise Counts: tournaments, games, players, games per position and per
          round, positions per round, ratings, results, terminations. Writes
          results/chesscom960/chesscom960_summary.txt.

Requests are made one at a time with a pause, and retried after a 429 (rate
limit) as the API asks. Everything fetched is cached under data/chesscom/
(ignored by git), so reruns resume.

    python code/chesscom960.py discover
    python code/chesscom960.py fetch
    python code/chesscom960.py parse
    python code/chesscom960.py summarise
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from lichess960 import CLK_RE, iter_games, position_from_fen
from utils import N_POSITIONS, REPO_ROOT, setup_logging

log = logging.getLogger("chesscom960")

API = "https://api.chess.com/pub"
DATA = REPO_ROOT / "data" / "chesscom"
CACHE = DATA / "cache"
PGN_DIR = DATA / "ff"
TOURNAMENTS = DATA / "freestyle_friday_tournaments.csv"
OUT = REPO_ROOT / "results" / "chesscom960"
USER_AGENT = "chess960-fairness research (github.com/aaronsfox/chess960-analysis)"
PAUSE_S = 0.4
SALT = "chess960-fairness"
NAME_RE = re.compile(r"freestyle-friday", re.I)
FIRST_MONTH = "2025-01"

# Tournaments found by web search (2026-10-31); discovery starts from these.
SEED_TOURNAMENTS = [
    "freestyle-friday-june-20-2025-5734315",
    "freestyle-friday-january-23-2026-6160617",
    "freestyle-friday-april-17-2026-6362201",
    "freestyle-friday-may-01-2026-6391977",
    "freestyle-friday-may-22-2026-6451887",
    "freestyle-friday-june-26-2026-6562133",
]
RESULTS = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


# ---------------------------------------------------------------------------
# API access
# ---------------------------------------------------------------------------

def get_json(url: str, cache: bool = True, retries: int = 5) -> dict | None:
    """GET a Published-Data API URL as JSON (None on 404), cached on disk."""
    key = CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".json")
    if cache and key.exists():
        return json.loads(key.read_text())
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            time.sleep(PAUSE_S)
            if cache:
                CACHE.mkdir(parents=True, exist_ok=True)
                key.write_text(json.dumps(data))
            return data
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code == 429:
                wait = 30 * (attempt + 1)
                print(f"    rate limited; waiting {wait} s")
                time.sleep(wait)
                continue
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return None


def tournament_id(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def month_list(first: str = FIRST_MONTH) -> list[str]:
    start = pd.Period(first, "M")
    end = pd.Period(pd.Timestamp.today(), "M")
    return [str(p) for p in pd.period_range(start, end, freq="M")]


# ---------------------------------------------------------------------------
# Discover, fetch
# ---------------------------------------------------------------------------

def discover(n_players: int = 60, out: Path = TOURNAMENTS) -> pd.DataFrame:
    """Tournament IDs from the archives of the seed tournaments' most frequent
    players."""
    known = set(SEED_TOURNAMENTS)
    if out.exists():
        known |= set(pd.read_csv(out).tournament_id)
    counts: Counter = Counter()
    print(f"Reading {len(SEED_TOURNAMENTS)} seed tournaments")
    for tid in SEED_TOURNAMENTS:
        t = get_json(f"{API}/tournament/{tid}")
        if not t:
            print(f"  {tid}: not found")
            continue
        players = [p["username"].lower() for p in t.get("players", []) if "username" in p]
        counts.update(players)
        print(f"  {tid}: {len(players)} players")
    regulars = [u for u, _ in counts.most_common(n_players)]
    months = month_list()
    print(f"\nReading {len(months)} monthly archives for {len(regulars)} regular players")
    found = Counter()
    for k, user in enumerate(regulars, 1):
        n_before = len(known)
        for m in months:
            y, mo = m.split("-")
            arch = get_json(f"{API}/player/{user}/games/{y}/{mo}")
            for g in (arch or {}).get("games", []):
                url = g.get("tournament", "")
                if url and NAME_RE.search(url) and g.get("rules") == "chess960":
                    tid = tournament_id(url)
                    known.add(tid)
                    found[tid] += 1
        print(f"  [{k:>2}/{len(regulars)}] {user}: {len(known) - n_before} new tournaments "
              f"(total {len(known)})")
    df = pd.DataFrame({"tournament_id": sorted(known)})
    df["date"] = df.tournament_id.map(date_from_id)
    df = df.sort_values("date")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} Freestyle Friday tournaments ({df.date.min()} to {df.date.max()}) -> {out}")
    return df


def date_from_id(tid: str) -> str:
    m = re.search(r"([a-z]+)-(\d{1,2})-(\d{4})", tid)
    if not m:
        return ""
    try:
        return str(pd.to_datetime(f"{m.group(1)} {m.group(2)} {m.group(3)}").date())
    except ValueError:
        return ""


def fetch(tournaments: Path = TOURNAMENTS, pgn_dir: Path = PGN_DIR) -> None:
    ids = pd.read_csv(tournaments).tournament_id.tolist()
    pgn_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(ids)} tournaments into {pgn_dir}")
    total = 0
    for k, tid in enumerate(ids, 1):
        dest = pgn_dir / f"{tid}.pgn"
        if dest.exists() and dest.stat().st_size > 0:
            n = dest.read_text(errors="replace").count("[Event ")
            print(f"[{k:>3}/{len(ids)}] {tid}: already here ({n} games)")
            total += n
            continue
        t = get_json(f"{API}/tournament/{tid}")
        if not t:
            print(f"[{k:>3}/{len(ids)}] {tid}: not found")
            continue
        if t.get("status") not in (None, "finished"):
            print(f"[{k:>3}/{len(ids)}] {tid}: status {t.get('status')}, skipped for now")
            continue
        chunks, n_games = [], 0
        for r_url in t.get("rounds", []):
            rnd = r_url.rstrip("/").split("/")[-1]
            r = get_json(r_url) or {}
            for g_url in r.get("groups", []):
                grp = get_json(g_url) or {}
                for g in grp.get("games", []):
                    pgn = g.get("pgn")
                    if not pgn:
                        continue
                    # Tag the API round number (live PGNs often carry "-" there).
                    chunks.append(f'[ApiRound "{rnd}"]\n[ApiTournament "{tid}"]\n' + pgn.strip())
                    n_games += 1
        dest.write_text("\n\n".join(chunks) + "\n")
        total += n_games
        print(f"[{k:>3}/{len(ids)}] {tid}: {len(t.get('players', []))} players, "
              f"{len(t.get('rounds', []))} rounds, {n_games} games")
    print(f"\nDone: {total:,} games in {pgn_dir}")


# ---------------------------------------------------------------------------
# Parse, summarise
# ---------------------------------------------------------------------------

def player_id(name: str) -> str:
    return hashlib.sha1(f"{SALT}:{name.lower()}".encode()).hexdigest()[:12]


def iter_tagged(fh):
    """iter_games, keeping the [ApiRound]/[ApiTournament] tags that precede
    each game's own headers (they are just more headers)."""
    yield from iter_games(fh)


def game_row(h: dict, movetext: str) -> dict:
    fen = h.get("FEN", "")
    clk = [int(a) * 3600 + int(b) * 60 + float(c) for a, b, c in CLK_RE.findall(movetext)]
    tc = h.get("TimeControl", "")
    base, inc = np.nan, np.nan
    m = re.match(r"(\d+)(?:\+(\d+))?$", tc)
    if m:
        base, inc = float(m.group(1)), float(m.group(2) or 0)

    def num(k):
        try:
            return float(h.get(k, ""))
        except ValueError:
            return np.nan
    return {"tournament_id": h.get("ApiTournament", ""), "round": h.get("ApiRound", ""),
            "date": h.get("Date", "").replace(".", "-"), "link": h.get("Link", ""),
            "white_id": player_id(h.get("White", "")), "black_id": player_id(h.get("Black", "")),
            "white_elo": num("WhiteElo"), "black_elo": num("BlackElo"),
            "result": h.get("Result", ""), "score_white": RESULTS.get(h.get("Result", ""), np.nan),
            "termination": h.get("Termination", ""), "time_control": tc, "base_s": base,
            "increment_s": inc, "variant": h.get("Variant", ""), "fen": fen,
            "position_id": position_from_fen(fen) if fen else -1, "plies": len(clk),
            "first_clock_white": clk[0] if len(clk) > 0 else np.nan,
            "first_clock_black": clk[1] if len(clk) > 1 else np.nan}


def parse(pgn_dir: Path = PGN_DIR, out: Path = OUT) -> pd.DataFrame:
    files = sorted(pgn_dir.glob("*.pgn"))
    if not files:
        raise SystemExit(f"no PGN files in {pgn_dir}; run `fetch` first")
    rows = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            rows += [game_row(h, mt) for h, mt in iter_tagged(fh)]
    g = pd.DataFrame(rows)
    before = len(g)
    g = g.drop_duplicates(subset=["link"]) if g.link.ne("").all() else g
    print(f"Parsed {before:,} games from {len(files)} tournaments; {before - len(g)} duplicates removed")
    print(f"Chess960 starting position identified: {(g.position_id >= 0).sum():,}")
    out.mkdir(parents=True, exist_ok=True)
    g.to_parquet(out / "chesscom960_games.parquet", index=False)
    print(f"Wrote {out / 'chesscom960_games.parquet'}")
    return g


def summary_text(g: pd.DataFrame) -> str:
    ok = g[g.position_id >= 0]
    per_pos = ok.groupby("position_id").size().reindex(range(N_POSITIONS), fill_value=0)
    pos_per_round = ok.groupby(["tournament_id", "round"]).position_id.nunique()
    share_top = ok.groupby(["tournament_id", "round"]).position_id.agg(
        lambda p: p.value_counts(normalize=True).iloc[0])
    s = ok.score_white.dropna()
    mean_r = (ok.white_elo + ok.black_elo) / 2
    gap = ok.white_elo - ok.black_elo
    L = ["Chess.com Freestyle Friday: what we've got", "=" * 42, "",
         f"Tournaments: {g.tournament_id.nunique()} ({g.date.min()} to {g.date.max()}); games: {len(g):,}; "
         f"players: {pd.concat([g.white_id, g.black_id]).nunique():,}",
         f"Chess960 position identified: {len(ok):,}; distinct positions: {ok.position_id.nunique()}",
         f"Games per position: min {per_pos.min()}, median {per_pos.median():.0f}, max {per_pos.max()}; "
         f"positions with >= 20 games: {(per_pos >= 20).sum()}",
         f"Positions per tournament round: median {pos_per_round.median():.0f} (1 = one drawn position "
         f"per round); share of a round on its most common position: median {share_top.median():.2f}",
         "Time controls: " + ", ".join(f"{k} {v:,}" for k, v in g.time_control.value_counts().head(5).items()),
         "Ratings (mean of the two players): " + ", ".join(
             f"p{q}: {mean_r.quantile(q / 100):.0f}" for q in (5, 25, 50, 75, 95)),
         "Rating gap White - Black: " + ", ".join(f"p{q}: {gap.quantile(q / 100):+.0f}" for q in (5, 25, 50, 75, 95)),
         f"Results: White {100 * (s == 1).mean():.1f}%, draw {100 * (s == 0.5).mean():.1f}%, "
         f"Black {100 * (s == 0).mean():.1f}%; mean White score {s.mean():.4f}",
         "Terminations: " + ", ".join(f"{k} {v:,}" for k, v in g.termination.str.replace(
             r"^.* (won|drawn|game abandoned)", r"\1", regex=True).value_counts().head(8).items()),
         f"Games with fewer than 10 plies: {(g.plies < 10).sum():,}"]
    games_per_t = g.groupby("tournament_id").size()
    L.append(f"Games per tournament: min {games_per_t.min()}, median {games_per_t.median():.0f}, "
             f"max {games_per_t.max()}")
    return "\n".join(L) + "\n"


def summarise(out: Path = OUT) -> None:
    g = pd.read_parquet(out / "chesscom960_games.parquet")
    text = summary_text(g)
    (out / "chesscom960_summary.txt").write_text(text)
    print(text)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover", parents=[common], help="find Freestyle Friday tournaments")
    d.add_argument("--players", type=int, default=60, help="regular players whose archives are read")
    for name, h in (("fetch", "download each tournament's games"), ("parse", "PGNs to a games table"),
                    ("summarise", "counts to scope the analysis")):
        sub.add_parser(name, parents=[common], help=h)
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    if args.command == "discover":
        discover(args.players)
    elif args.command == "fetch":
        fetch()
    elif args.command == "parse":
        parse()
    else:
        summarise()
    return 0


if __name__ == "__main__":
    sys.exit(main())
