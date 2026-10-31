"""Master-level over-the-board Chess960 games (human play case studies).

Games from elite Chess960 / Freestyle events, as PGN files from The Week in
Chess (TWIC). The files are listed in config/masters960_sources.csv (add rows
for events found later; `verified` marks URLs checked by hand). These events
usually draw one position per round for every board, so the data are many
games on a few dozen positions: case studies rather than a full ranking.

Subcommands
-----------
download  Fetch every listed file into data/masters/ (needs internet; skips
          files already there). Files can also be saved there by hand.
parse     Read every .pgn in data/masters/ and write
          results/masters960/masters960_games.csv: one row per game with the
          event (from the sources table), players, FIDE ratings, result,
          starting position (0-959 from the FEN), plies, clocks and the
          time-control class. Armageddon games (unequal clocks or named in the
          round/event) are flagged.
analyse   Inclusion: position identified, not Armageddon, at least 10 plies
          (drops forfeits), both ratings. Then:
            - events and positions tables (games, W/D/L, rating-adjusted
              human Delta_Elo with 95% CI next to the self-play values);
            - pooled fractional logits with the rating gap as an offset
              (FIDE Elo, so beta = 1): White's advantage at equal ratings and
              the slope on Stockfish / Caissa self-play Delta_Elo, overall,
              by time-control class and by rating level, position-clustered;
            - a comparison of White's advantage and the decisive rate across
              playing levels: Lichess bands, Lichess titled, masters, engines;
            - figures: case-study positions and the levels comparison.

    python code/masters960.py download
    python code/masters960.py parse
    python code/masters960.py analyse
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

from analyse_lichess960 import K, decisive_model, frac_logit, load_engines, per_position_decisive
from lichess960 import iter_games, position_from_fen
from plot_style import BAND_COLOUR, ENGINES, METRICS, MUTED_TEXT, apply_style, bold_tick_labels
from utils import REPO_ROOT, STANDARD_POSITION_ID, setup_logging

log = logging.getLogger("masters960")

SOURCES = REPO_ROOT / "config" / "masters960_sources.csv"
DATA = REPO_ROOT / "data" / "masters"
OUT = REPO_ROOT / "results" / "masters960"
FIGURES = REPO_ROOT / "results" / "figures"
LICHESS = REPO_ROOT / "results" / "lichess960" / "analysis"
SELFPLAY = REPO_ROOT / "results" / "selfplay" / "analysis" / "selfplay_posteriors.csv"

CLK_RE = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")
COMMENT_RE = re.compile(r"\{[^}]*\}")
VARIATION_RE = re.compile(r"\([^()]*\)")
MOVE_NO_RE = re.compile(r"\d+\.(\.\.)?")
RESULTS = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
MIN_PLIES = 10
ARMAGEDDON_RATIO = 0.25          # first clocks differing by more than this = unequal time
ARMAGEDDON_RATIO_BLITZ = 0.15    # ... in blitz-length games (Armageddon is e.g. 5 vs 4 min)
OWN_CLOCK_BELOW = 0.6            # first clock below this share of the event base: faster tiebreak
ROUND_POSITION_SHARE = 0.8       # a round where most games share one position drew that position
ROUND_MIN_GAMES = 10
MIN_CASE_GAMES = 20              # positions shown as case studies
RATING_LEVELS = [0, 2400, 2600, 4000]
RATING_LABELS = ["<2400", "2400-2599", "2600+"]
TC_ORDER = ["classical", "rapid", "blitz"]


# ---------------------------------------------------------------------------
# Download and parse
# ---------------------------------------------------------------------------

def load_sources(path: Path = SOURCES) -> pd.DataFrame:
    return pd.read_csv(path)


def download(sources: pd.DataFrame, out_dir: Path = DATA) -> None:
    """Fetch every file in the sources table. The URL comes from the table, so
    adding an event only needs a new row there."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(sources)
    print(f"Downloading {n} PGN files into {out_dir}")
    done, skipped, failed = [], [], []
    for k, r in enumerate(sources.itertuples(), 1):
        dest = out_dir / r.file
        tag = f"[{k:>2}/{n}] {r.file:<24}"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"{tag} already here ({dest.stat().st_size / 1024:,.0f} KB, "
                  f"{count_games(dest)} games), skipped")
            skipped.append(r.file)
            continue
        try:
            if r.url.startswith("https://lichess.org/broadcast/"):
                body = lichess_broadcast_pgn(r.url)
            else:
                body = fetch(r.url)
            if r.url.lower().endswith(".zip"):      # a TWIC weekly issue: keep its PGN
                body = pgn_from_zip(body)
            if b"[Event" not in body[:5000]:
                raise ValueError("response is not a PGN file (wrong URL or an error page?)")
            dest.write_bytes(body)
            print(f"{tag} downloaded {len(body) / 1024:,.0f} KB, {count_games(dest)} games "
                  f"({r.event} {r.year}, {r.stage})")
            done.append(r.file)
        except Exception as exc:          # noqa: BLE001 - report and carry on
            print(f"{tag} FAILED: {exc}")
            failed.append(r)
    print(f"\nDone: {len(done)} downloaded, {len(skipped)} already present, {len(failed)} failed.")
    if failed:
        print("Save these by hand into", out_dir, "(same file names), then run `parse`:")
        for r in failed:
            print(f"  {r.file}: {r.url}")


def fetch(url: str, accept: str | None = None) -> bytes:
    headers = {"User-Agent": "chess960-fairness research"}
    if accept:
        headers["Accept"] = accept
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as resp:
        return resp.read()


def lichess_broadcast_pgn(url: str) -> bytes:
    """All games of a Lichess broadcast, from the URL of any of its rounds
    (https://lichess.org/broadcast/<tour-slug>/<round-slug>/<round-id>[/<game-id>]).
    The round's tournament is exported with /api/broadcast/<tour-id>.pgn. With
    '#group' at the end of the URL, every tournament in the round's broadcast
    group is exported (e.g. both round-robin groups and the knockout)."""
    import json
    group = url.endswith("#group")
    parts = url.split("#")[0].rstrip("/").split("/broadcast/")[1].split("/")
    if len(parts) < 3:
        raise ValueError("expected a round URL: /broadcast/<tour-slug>/<round-slug>/<round-id>")
    tslug, rslug, round_id = parts[:3]
    info = json.loads(fetch(f"https://lichess.org/api/broadcast/{tslug}/{rslug}/{round_id}",
                            accept="application/json"))
    tours = [(info["tour"]["id"], info["tour"].get("name", ""))]
    if group:
        g = info.get("group") or {}
        if g.get("tours"):
            tours = [(t["id"], t.get("name", "")) for t in g["tours"]]
        else:
            print("    (no broadcast group found; exporting this tournament only)")
    chunks = []
    for tid, name in tours:
        pgn = fetch(f"https://lichess.org/api/broadcast/{tid}.pgn")
        print(f"    Lichess broadcast {tid} {name}: {pgn.count(b'[Event ')} games")
        chunks.append(pgn)
    return b"\n\n".join(chunks)


def pgn_from_zip(body: bytes) -> bytes:
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".pgn")]
        if not names:
            raise ValueError("zip has no .pgn file")
        return b"\n".join(z.read(n) for n in names)


def count_games(path: Path) -> int:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return sum(1 for line in fh if line.startswith("[Event "))


def tc_class(base_s: float) -> str:
    if base_s >= 3600:
        return "classical"
    if base_s >= 480:
        return "rapid"
    return "blitz"


def clocks(movetext: str) -> list[float]:
    return [int(h) * 3600 + int(m) * 60 + float(s) for h, m, s in CLK_RE.findall(movetext)]


def count_plies(movetext: str) -> int:
    t = COMMENT_RE.sub(" ", movetext)
    while VARIATION_RE.search(t):
        t = VARIATION_RE.sub(" ", t)
    t = MOVE_NO_RE.sub(" ", t)
    toks = [x for x in t.split() if x not in RESULTS and x != "*" and not x.startswith("$")]
    return len(toks)


VARIANT_960_RE = re.compile(r"960|fischer\s*-?\s*random|fischerandom|freestyle|9lx", re.I)


def is_960_variant(variant: str) -> bool:
    """Variant header naming Chess960 (spellings seen: Chess960, chess 960,
    Fischerandom, Fischer Random, Freestyle, 9LX)."""
    return bool(VARIANT_960_RE.search(variant or ""))


def game_record(h: dict[str, str], movetext: str, src: dict) -> dict:
    fen = h.get("FEN", "")
    clk = clocks(movetext)
    w0 = clk[0] if len(clk) > 0 else np.nan
    b0 = clk[1] if len(clk) > 1 else np.nan
    source_base = float(src.get("base_minutes", np.nan)) * 60
    first = np.nanmax([w0, b0]) if np.isfinite([w0, b0]).any() else np.nan
    # A game in a file whose first clocks are far below the event's base time
    # is a tiebreak at a faster control: classify it by its own clocks.
    # A game whose first clocks are far from the event's base time (a faster
    # tiebreak, or a classical game in a mixed file) is classed by its own clocks.
    base = (first if np.isfinite(first) and (first < OWN_CLOCK_BELOW * source_base
                                             or first > 1.5 * source_base)
            else source_base)
    text = f"{h.get('Event', '')} {h.get('Round', '')}".lower()
    limit = ARMAGEDDON_RATIO_BLITZ if np.isfinite(base) and base < 480 else ARMAGEDDON_RATIO
    unequal = (np.isfinite(w0) and np.isfinite(b0) and max(w0, b0) > 0
               and abs(w0 - b0) / max(w0, b0) > limit)
    def elo(k):
        try:
            v = float(h.get(k, ""))
            return v if v > 0 else np.nan
        except ValueError:
            return np.nan
    return {
        "source": src.get("file", ""), "event": src.get("event", h.get("Event", "")),
        "year": src.get("year", np.nan), "stage": src.get("stage", ""),
        "pgn_event": h.get("Event", ""), "site": h.get("Site", ""), "date": h.get("Date", ""),
        "round": h.get("Round", ""), "white": h.get("White", ""), "black": h.get("Black", ""),
        "white_elo": elo("WhiteElo"), "black_elo": elo("BlackElo"),
        "white_title": h.get("WhiteTitle", ""), "black_title": h.get("BlackTitle", ""),
        "result": h.get("Result", ""), "score_white": RESULTS.get(h.get("Result", ""), np.nan),
        "variant": h.get("Variant", ""), "variant_960": is_960_variant(h.get("Variant", "")),
        "fen": fen, "position_id": position_from_fen(fen) if fen else -1,
        "plies": count_plies(movetext), "first_clock_white": w0, "first_clock_black": b0,
        "base_s": base, "tc_class": tc_class(base) if np.isfinite(base) else "",
        "armageddon": bool("armageddon" in text or unequal),
    }


def parse(sources: pd.DataFrame, data_dir: Path = DATA, out: Path = OUT) -> pd.DataFrame:
    meta = {r["file"]: r for r in sources.to_dict("records")}
    files = sorted(data_dir.glob("*.pgn"))
    if not files:
        raise SystemExit(f"no PGN files in {data_dir}; run `download` first")
    print(f"Parsing {len(files)} PGN files from {data_dir}\n")
    print(f"{'file':<24} {'games':>6} {'variant':>7} {'960 pos':>7} {'positions':>9} {'classical':>9} "
          f"{'rapid':>6} {'blitz':>6} {'armag.':>6} {'no Elo':>6}  event")
    rows = []
    for f in files:
        src = meta.get(f.name)
        if src is None:
            print(f"{f.name:<24} not in {SOURCES.name}: event details unknown, add a row for it")
            src = {"file": f.name}
        with open(f, encoding="utf-8", errors="replace") as fh:
            flt = src.get("event_filter")
            flt = re.compile(flt, re.I) if isinstance(flt, str) and flt else None
            recs = [game_record(h, mt, src) for h, mt in iter_games(fh)
                    if flt is None or flt.search(h.get("Event", ""))]
        rows += recs
        r = pd.DataFrame(recs)
        if r.empty:
            print(f"{f.name:<24} {0:>6}  (no games found)")
            continue
        tc = r.tc_class.value_counts()
        print(f"{f.name:<24} {len(r):>6} {int(r.variant_960.sum()):>7} {(r.position_id >= 0).sum():>7} "
              f"{r.position_id[r.position_id >= 0].nunique():>9} {tc.get('classical', 0):>9} "
              f"{tc.get('rapid', 0):>6} {tc.get('blitz', 0):>6} {int(r.armageddon.sum()):>6} "
              f"{int((r.white_elo.isna() | r.black_elo.isna()).sum()):>6}  "
              f"{src.get('event', '?')} {src.get('year', '')}")
    g = pd.DataFrame(rows)
    before = len(g)
    g = g.drop_duplicates(subset=["white", "black", "date", "round", "result", "fen", "plies"])
    unknown = g[g.position_id < 0]
    no_tag = g[(g.position_id >= 0) & ~g.variant_960]
    tag_no_pos = g[g.variant_960 & (g.position_id < 0)]
    print(f"Variant tag says Chess960: {int(g.variant_960.sum()):,}; Chess960 FEN without the tag: "
          f"{len(no_tag)}; tag without a usable FEN: {len(tag_no_pos)}")
    if len(no_tag[no_tag.position_id == STANDARD_POSITION_ID]):
        print(f"  standard start (#518) without a 960 tag: "
              f"{int((no_tag.position_id == STANDARD_POSITION_ID).sum())} games (excluded as standard chess)")
    print(f"\nTotal: {before:,} games; {before - len(g)} duplicates removed; {len(g):,} kept")
    print(f"Starting position identified: {(g.position_id >= 0).sum():,}; distinct positions: "
          f"{g.position_id[g.position_id >= 0].nunique()}")
    if len(unknown):
        print(f"No Chess960 position for {len(unknown)} games (missing or unusual FEN), e.g. "
              f"{unknown.source.value_counts().head(3).to_dict()}")
    top = g[g.position_id >= 0].position_id.value_counts().head(10)
    print("Most-played positions: " + ", ".join(f"#{p} ({n})" for p, n in top.items()))
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "masters960_games.csv", index=False)
    print(f"Wrote {out / 'masters960_games.csv'}")
    return g


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def fill_ratings(g: pd.DataFrame) -> pd.DataFrame:
    """Fill a missing FIDE rating from the same player's other games in the
    same event (some rounds are published without ratings)."""
    g = g.copy()
    long = pd.concat([g[["event", "white", "white_elo"]].set_axis(["event", "player", "elo"], axis=1),
                      g[["event", "black", "black_elo"]].set_axis(["event", "player", "elo"], axis=1)])
    known = long.dropna().groupby(["event", "player"]).elo.median()
    for side in ("white", "black"):
        miss = g[f"{side}_elo"].isna()
        idx = pd.MultiIndex.from_frame(g.loc[miss, ["event", side]])
        g.loc[miss, f"{side}_elo"] = known.reindex(idx).to_numpy()
    return g


def round_position_ok(g: pd.DataFrame) -> pd.Series:
    """In events that draw one position per round, a game whose position
    differs from the one most of its round played is a broadcast error (a
    board still showing an earlier round's game). Applied to rounds with at
    least ROUND_MIN_GAMES games where one position covers at least
    ROUND_POSITION_SHARE of the identified games."""
    rnd = g["round"].astype(str).str.split(".").str[0]
    key = g.source.astype(str) + "|" + rnd
    ok = pd.Series(True, index=g.index)
    for _, sub in g[g.position_id >= 0].groupby(key[g.position_id >= 0]):
        if len(sub) < ROUND_MIN_GAMES:
            continue
        top = sub.position_id.value_counts()
        if top.iloc[0] / len(sub) >= ROUND_POSITION_SHARE:
            ok.loc[sub.index[sub.position_id != top.index[0]]] = False
    return ok


def include(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_missing = int((g.white_elo.isna() | g.black_elo.isna()).sum())
    g = fill_ratings(g)
    print(f"Ratings: {n_missing} games missing one; "
          f"{n_missing - int((g.white_elo.isna() | g.black_elo.isna()).sum())} filled from the same "
          "player's other games in the event")
    # A Chess960 game: the Variant header says so, or the FEN is a Chess960
    # start other than the standard one (a standard-chess game has no FEN or
    # the standard FEN and no 960 Variant tag).
    if "variant_960" not in g:            # games table from before the Variant check
        g["variant_960"] = False
    chess960 = g.variant_960.fillna(False).astype(bool) | (
        (g.position_id >= 0) & (g.position_id != STANDARD_POSITION_ID))
    steps = [("Chess960 game (Variant tag or Chess960 FEN)", chess960),
             ("starting position identified", g.position_id >= 0),
             ("position matches its round's drawn position", round_position_ok(g)),
             ("known result", g.score_white.notna()),
             ("not Armageddon", ~g.armageddon.astype(bool)),
             (f"at least {MIN_PLIES} plies (drops forfeits)", g.plies >= MIN_PLIES),
             ("both FIDE ratings", g.white_elo.notna() & g.black_elo.notna())]
    keep = pd.Series(True, index=g.index)
    flow = [{"criterion": "parsed", "games": len(g)}]
    for label, m in steps:
        keep &= m.fillna(False).astype(bool)
        flow.append({"criterion": label, "games": int(keep.sum())})
    flow = pd.DataFrame(flow)
    flow["removed"] = -flow.games.diff()
    return g[keep].copy(), flow


def rating_offset(d: pd.DataFrame) -> np.ndarray:
    """FIDE Elo gap on the logit scale, beta fixed at 1."""
    return (d.white_elo.to_numpy(float) - d.black_elo.to_numpy(float)) * np.log(10) / 400


def wdl(s: np.ndarray) -> dict:
    e = float(np.mean(s))
    return {"games": len(s), "white_wins": float(np.mean(s == 1)), "draws": float(np.mean(s == 0.5)),
            "black_wins": float(np.mean(s == 0)), "escore": e,
            "delta_elo_raw": 400 * np.log10(e / (1 - e)) if 0 < e < 1 else np.nan}


def advantage(d: pd.DataFrame, cluster: np.ndarray | None = None) -> dict:
    """White's advantage at equal FIDE ratings (Elo), robust or clustered SE."""
    s = d.score_white.to_numpy(float)
    if len(d) < 2 or s.min() == s.max():
        return {"advantage_elo": np.nan, "advantage_lo": np.nan, "advantage_hi": np.nan}
    b, V = frac_logit(np.ones((len(d), 1)), s, offset=rating_offset(d), cluster=cluster)
    se = float(np.sqrt(V[0, 0]))
    return {"advantage_elo": K * b[0], "advantage_se": K * se,
            "advantage_lo": K * (b[0] - 1.96 * se), "advantage_hi": K * (b[0] + 1.96 * se)}


def slope(d: pd.DataFrame, x: pd.Series, engine: str) -> dict:
    """Pooled slope of master outcomes on engine self-play Delta_Elo, with the
    rating gap as an offset, clustered by position."""
    xp = x.reindex(d.position_id).to_numpy()
    if len(np.unique(d.position_id)) < 3:
        return {}
    xc = (xp - np.mean(xp)) / K
    X = np.column_stack([np.ones(len(d)), xc])
    b, V = frac_logit(X, d.score_white.to_numpy(float), offset=rating_offset(d),
                      cluster=d.position_id.to_numpy())
    se = np.sqrt(np.diag(V))
    return {f"gamma_{engine}": b[1], f"gamma_{engine}_se": se[1],
            f"gamma_{engine}_lo": b[1] - 1.96 * se[1], f"gamma_{engine}_hi": b[1] + 1.96 * se[1],
            f"p_gamma_{engine}": 2 * stats.norm.sf(abs(b[1] / se[1]))}


def level(d: pd.DataFrame) -> pd.Series:
    m = (d.white_elo + d.black_elo) / 2
    return pd.cut(m, RATING_LEVELS, labels=RATING_LABELS, right=False)


def pooled(d: pd.DataFrame, eng: pd.DataFrame) -> pd.DataFrame:
    groups = [("all", "all", d)]
    groups += [("time control", t, d[d.tc_class == t]) for t in TC_ORDER]
    lv = level(d)
    groups += [("rating", r, d[lv == r]) for r in RATING_LABELS]
    rows = []
    for kind, name, sub in groups:
        if len(sub) == 0:
            continue
        row = {"group": kind, "level": name, "positions": sub.position_id.nunique(),
               "events": sub.event.nunique(), "mean_rating": float(((sub.white_elo + sub.black_elo) / 2).mean()),
               **wdl(sub.score_white.to_numpy(float)),
               **advantage(sub, sub.position_id.to_numpy()), **decisive_at_equal(sub)}
        for e in ("stockfish", "caissa"):
            row.update(slope(sub, eng[("delta_elo_median", e)], e))
        rows.append(row)
    return pd.DataFrame(rows)


def abs_gap(d: pd.DataFrame) -> np.ndarray:
    return np.abs(d.white_elo.to_numpy(float) - d.black_elo.to_numpy(float))


def decisive_at_equal(d: pd.DataFrame) -> dict:
    """Decisive rate between equally rated players (logit on |rating gap| and
    its square, position-clustered)."""
    s = d.score_white.to_numpy(float)
    dec = (s != 0.5).astype(float)
    m = decisive_model(abs_gap(d), dec, cluster=d.position_id.to_numpy())
    return {k: v for k, v in m.items() if not k.startswith("_")}


def positions_table(d: pd.DataFrame, eng: pd.DataFrame) -> pd.DataFrame:
    s = d.score_white.to_numpy(float)
    dec = (s != 0.5).astype(float)
    coef = decisive_model(abs_gap(d), dec, cluster=d.position_id.to_numpy())["_coef"]
    adj = per_position_decisive(d.position_id.to_numpy(), abs_gap(d), dec, coef)
    rows = []
    for p, sub in d.groupby("position_id"):
        r = {"position_id": p, "events": "; ".join(sorted(sub.event.unique())),
             "tc_classes": "; ".join(t for t in TC_ORDER if (sub.tc_class == t).any()),
             "mean_rating": float(((sub.white_elo + sub.black_elo) / 2).mean()),
             **wdl(sub.score_white.to_numpy(float)), **advantage(sub),
             "decisive_raw": float((sub.score_white != 0.5).mean()),
             **adj.loc[int(p)].to_dict()}
        for e in ("stockfish", "caissa"):
            r[f"{e}_selfplay_elo"] = eng.loc[p, ("delta_elo_median", e)]
            r[f"{e}_selfplay_sd"] = eng.loc[p, ("delta_elo_sd", e)]
        rows.append(r)
    return pd.DataFrame(rows).sort_values("games", ascending=False)


def events_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ev, yr, st), sub in d.groupby(["event", "year", "stage"], dropna=False):
        rows.append({"event": ev, "year": yr, "stage": st, "positions": sub.position_id.nunique(),
                     "mean_rating": float(((sub.white_elo + sub.black_elo) / 2).mean()),
                     "tc_classes": "; ".join(t for t in TC_ORDER if (sub.tc_class == t).any()),
                     **wdl(sub.score_white.to_numpy(float))})
    return pd.DataFrame(rows).sort_values(["year", "event", "stage"])


def levels_comparison(pool: pd.DataFrame, lichess_dir: Path = LICHESS,
                      selfplay: Path = SELFPLAY) -> pd.DataFrame:
    """White's advantage (at equal ratings for humans) and the decisive rate
    from club level to engines."""
    rows = []
    desc_p, bands_p, rm_p = (lichess_dir / f"lichess960_{k}.csv"
                             for k in ("descriptives", "rating_bands", "rating_model"))
    if desc_p.exists() and bands_p.exists() and rm_p.exists():
        desc, bands, rm = pd.read_csv(desc_p), pd.read_csv(bands_p), pd.read_csv(rm_p)
        for s in ("blitz", "rapid"):
            for _, b in bands[bands.set == s].iterrows():
                dr = desc[(desc.set == s) & (desc.band == b.band)]
                if pd.isna(b.get("advantage_elo")) or dr.empty:
                    continue
                rows.append({"source": f"Lichess {s}", "level": b.band, "games": b.games,
                             "advantage_elo": b.advantage_elo, "advantage_lo": b.advantage_lo,
                             "advantage_hi": b.advantage_hi, **dec_cols(dr.iloc[0])})
        t = rm[rm.set == "titled"]
        dt = desc[(desc.set == "titled") & (desc.band == "all")]
        if len(t) and len(dt):
            rows.append({"source": "Lichess titled", "level": "titled", "games": int(t.games.iloc[0]),
                         "advantage_elo": t.advantage_elo.iloc[0], "advantage_lo": t.advantage_lo.iloc[0],
                         "advantage_hi": t.advantage_hi.iloc[0], **dec_cols(dt.iloc[0])})
    cc = REPO_ROOT / "results" / "chesscom960" / "analysis"
    if (cc / "chesscom960_descriptives.csv").exists() and (cc / "chesscom960_rating_model.csv").exists():
        cd = pd.read_csv(cc / "chesscom960_descriptives.csv").iloc[0]
        cr = pd.read_csv(cc / "chesscom960_rating_model.csv").iloc[0]
        rows.append({"source": "Chess.com Freestyle Friday", "level": "titled", "games": int(cd.games),
                     "advantage_elo": cr.advantage_elo, "advantage_lo": cr.advantage_lo,
                     "advantage_hi": cr.advantage_hi, **dec_cols(cd)})
    for _, r in pool[pool.group.isin(["time control", "rating"])].iterrows():
        rows.append({"source": "Masters OTB" if r.group == "rating" else "Masters OTB (time control)",
                     "level": r.level, "games": r.games, "advantage_elo": r.advantage_elo,
                     "advantage_lo": r.advantage_lo, "advantage_hi": r.advantage_hi, **dec_cols(r)})
    if selfplay.exists():
        sp = pd.read_csv(selfplay)
        for e, g in sp.groupby("engine"):
            n = g.games.sum()
            e_score = (g.white_wins.sum() + 0.5 * g.draws.sum()) / n
            # mean over positions of the posterior median, with its spread
            rows.append({"source": "Engine self-play", "level": ENGINES[e].label, "games": int(n),
                         "advantage_elo": 400 * np.log10(e_score / (1 - e_score)),
                         "advantage_lo": np.nan, "advantage_hi": np.nan,
                         "decisive_raw": 1 - g.draws.sum() / n,
                         # engines play themselves: no rating gap to adjust for
                         "decisive": 1 - g.draws.sum() / n, "decisive_lo": np.nan, "decisive_hi": np.nan})
    return pd.DataFrame(rows)


def dec_cols(r) -> dict:
    """Decisive rate at equal ratings (with CI) and the raw rate, from a row
    with decisive_adj* (and decisive_raw or draws) columns."""
    raw = r["decisive_raw"] if "decisive_raw" in r and pd.notna(r["decisive_raw"]) else 1 - r["draws"]
    return {"decisive_raw": float(raw), "decisive": float(r["decisive_adj"]),
            "decisive_lo": float(r["decisive_adj_lo"]), "decisive_hi": float(r["decisive_adj_hi"])}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

TC_STYLE = {"classical": dict(marker="o", mfc="#111111", color="#111111"),
            "rapid": dict(marker="s", mfc="white", color="#111111"),
            "blitz": dict(marker="^", mfc="white", color="#777777")}


def save_fig(fig, name: str, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for p in paths:
        fig.savefig(p, metadata={"CreationDate": None} if p.suffix == ".pdf" else None)
    plt.close(fig)
    return paths


def fig_case_positions(d: pd.DataFrame, eng: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Positions with at least MIN_CASE_GAMES master games: rating-adjusted
    master Delta_Elo (95% CI) against Stockfish self-play Delta_Elo; marker by
    the time-control class with most games at that position."""
    plt = apply_style()
    pos = positions_table(d, eng)
    pos = pos[pos.games >= MIN_CASE_GAMES]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    if len(pos):
        main_tc = d.groupby("position_id").tc_class.agg(lambda s: s.value_counts().index[0])
        xlo = min(pos.stockfish_selfplay_elo.min(), 0) - 10
        xhi = pos.stockfish_selfplay_elo.max() + 10
        ax.plot([xlo, xhi], [xlo, xhi], color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)), zorder=1)
        ax.axhline(0, color=BAND_COLOUR, lw=0.8, zorder=0)
        for _, r in pos.iterrows():
            st = TC_STYLE.get(main_tc[r.position_id], TC_STYLE["rapid"])
            ax.errorbar(r.stockfish_selfplay_elo, r.advantage_elo,
                        yerr=[[r.advantage_elo - r.advantage_lo], [r.advantage_hi - r.advantage_elo]],
                        fmt=st["marker"], mfc=st["mfc"], color=st["color"], ms=4.5, elinewidth=0.8,
                        capsize=0, zorder=3)
        ax.set_xlim(xlo, xhi)
    ax.set_xlabel(f"Stockfish Self-play {METRICS['delta_elo'].bold_symbol}")
    ax.set_ylabel(f"Master {METRICS['delta_elo'].bold_symbol}")
    bold_tick_labels(ax)
    return save_fig(fig, "masters960_case_positions", out_dir)


def level_label(source: str, lev: str) -> str:
    if source == "Lichess titled":
        return "Lichess titled"
    if source.startswith("Lichess"):
        return f"Lichess {source.split()[-1]} {lev}"
    if source.startswith("Masters"):
        return f"Masters OTB {lev}"
    if source.startswith("Chess.com"):
        return "Chess.com Freestyle Friday"
    return f"{lev} self-play"


def fig_levels(levels: pd.DataFrame, out_dir: Path) -> list[Path]:
    """White's advantage (top) and decisive rate (bottom) from Lichess club level
    through titled players and over-the-board masters to engines."""
    plt = apply_style()
    # Masters by time control (rating-level groups are too thin at small gaps).
    order = levels[~levels.source.eq("Masters OTB") & levels.decisive.notna()].reset_index(drop=True)
    labels = [level_label(r.source, r.level) for r in order.itertuples()]
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True, gridspec_kw={"hspace": 0.12})
    colours = ["#111111" if s.startswith("Lichess") else ("#555555" if s.startswith(("Masters", "Chess.com")) else
               ENGINES[r.lower()].colour if r.lower() in ENGINES else "#999999")
               for s, r in zip(order.source, order.level)]
    ax = axes[0]
    ax.axhline(0, color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)))
    for i, r in order.iterrows():
        yerr = None if pd.isna(r.advantage_lo) else [[r.advantage_elo - r.advantage_lo], [r.advantage_hi - r.advantage_elo]]
        ax.errorbar(i, r.advantage_elo, yerr=yerr, fmt="o", ms=4.5, color=colours[i], elinewidth=1.0, capsize=0)
    ax.set_ylabel(f"White Advantage {METRICS['delta_elo'].bold_symbol}")
    axes[1].bar(x, 100 * order.decisive, color=colours, width=0.6)
    err = np.array([100 * (order.decisive - order.decisive_lo), 100 * (order.decisive_hi - order.decisive)])
    axes[1].errorbar(x, 100 * order.decisive, yerr=np.nan_to_num(err), fmt="none", ecolor="#999999",
                     elinewidth=0.8, capsize=0)
    axes[1].set_ylabel("Decisive at Equal Ratings (%)")
    axes[1].set_xticks(x, labels, fontsize=6.5, rotation=40, ha="right", rotation_mode="anchor")
    for a in axes:
        bold_tick_labels(a)
    return save_fig(fig, "masters960_levels", out_dir)


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------

def summary_text(flow, ev, pos, pool, levels) -> str:
    L = ["Master-level OTB Chess960: summary", "=" * 34, "", "Inclusion:"]
    L += [f"  {r.criterion}: {r.games:,}" for r in flow.itertuples()]
    L += ["", f"Events/stages: {len(ev)}; positions: {len(pos)}; games per position: median "
          f"{pos.games.median():.0f}, max {pos.games.max()}; positions with >= {MIN_CASE_GAMES} games: "
          f"{(pos.games >= MIN_CASE_GAMES).sum()}", "", "Pooled (rating gap as offset, position-clustered):"]
    for r in pool.itertuples():
        g = getattr(r, "gamma_stockfish", np.nan)
        L.append(f"  {r.group} / {r.level}: n {r.games:,}, positions {r.positions}; W/D/L "
                 f"{100 * r.white_wins:.1f}/{100 * r.draws:.1f}/{100 * r.black_wins:.1f}%; advantage "
                 f"{r.advantage_elo:+.1f} [{r.advantage_lo:+.1f}, {r.advantage_hi:+.1f}]; slope on Stockfish "
                 f"{g:.2f} [{getattr(r, 'gamma_stockfish_lo', np.nan):.2f}, "
                 f"{getattr(r, 'gamma_stockfish_hi', np.nan):.2f}]; decisive at equal ratings "
                 f"{100 * r.decisive_adj:.1f}% [{100 * r.decisive_adj_lo:.1f}, {100 * r.decisive_adj_hi:.1f}]")
    L += ["", f"Case-study positions (>= {MIN_CASE_GAMES} games):"]
    for r in pos[pos.games >= MIN_CASE_GAMES].itertuples():
        L.append(f"  #{r.position_id}: n {r.games}, {r.tc_classes}; W/D/L {100 * r.white_wins:.0f}/"
                 f"{100 * r.draws:.0f}/{100 * r.black_wins:.0f}%; master {r.advantage_elo:+.0f} "
                 f"[{r.advantage_lo:+.0f}, {r.advantage_hi:+.0f}]; Stockfish {r.stockfish_selfplay_elo:+.0f}, "
                 f"Caissa {r.caissa_selfplay_elo:+.0f}; decisive raw {100 * r.decisive_raw:.0f}%, at equal "
                 f"ratings {100 * r.decisive_adj:.0f}%  ({r.events})")
    if STANDARD_POSITION_ID in set(pos.position_id):
        L.append(f"#518 present: {int(pos.set_index('position_id').loc[STANDARD_POSITION_ID, 'games'])} games")
    return "\n".join(L) + "\n"


def analyse(out: Path = OUT, figures: Path = FIGURES) -> None:
    g = pd.read_csv(out / "masters960_games.csv")
    d, flow = include(g)
    print(f"Analysing {len(d):,} of {len(g):,} games after the inclusion criteria\n")
    eng = load_engines(SELFPLAY)
    ev, pos, pool = events_table(d), positions_table(d, eng), pooled(d, eng)
    levels = levels_comparison(pool)
    flow.to_csv(out / "masters960_flow.csv", index=False)
    ev.to_csv(out / "masters960_events.csv", index=False)
    pos.to_csv(out / "masters960_positions.csv", index=False)
    pool.to_csv(out / "masters960_pooled.csv", index=False)
    levels.to_csv(out / "masters960_levels.csv", index=False)
    text = summary_text(flow, ev, pos, pool, levels)
    (out / "masters960_summary.txt").write_text(text)
    print(text)
    for path in fig_case_positions(d, eng, figures) + fig_levels(levels, figures):
        if path.suffix == ".png":
            print(f"Figure: {path}")
    print(f"Tables: {out}/masters960_*.csv")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    for name, h in (("download", "fetch the listed PGN files into data/masters/"),
                    ("parse", "PGN files to a games table"), ("analyse", "tables, models and figures")):
        sub.add_parser(name, parents=[common], help=h)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    src = load_sources()
    if args.command == "download":
        download(src)
    elif args.command == "parse":
        parse(src)
    else:
        analyse()
    return 0


if __name__ == "__main__":
    sys.exit(main())
