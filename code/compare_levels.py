"""Lichess human play against engine self-play on the win/draw/loss simplex.

Applies the self-play outcome model to the Lichess games: for every source
and position, a Dirichlet posterior on (White wins, draws, Black wins) with
the Jeffreys prior, Dirichlet(1/2 + counts / DEFF). DEFF is the self-play
design effect for the engines and 1 for Lichess (rematch clustering changed
the Lichess SEs by under 1%). From 20,000 draws per position: E_score,
Delta_Elo and the decisive rate (1 - draw rate) with 95% credible intervals,
P(favours White) and P(within +/-5 Elo).

Sources: Stockfish and Caissa self-play; Lichess blitz and rapid (counts from
analyse_lichess960.py). The titled-player and master sets are too sparse per
position for this analysis (median 8 and 5 games). Lichess counts are raw:
positions are assigned at random, so rating gaps average out over a
position's games (the rating-adjusted estimates are in analyse_lichess960.py).

Outputs (results/compare_levels/):
  compare_posteriors.csv  every source x position
  compare_sources.csv     per source: games, W/D/L, decisive rate, pooled
                          Delta_Elo, median credible-interval width, counts of
                          positions favouring White / within +/-5 Elo
  compare_agreement.csv   Lichess set vs engine, across positions: Spearman rho
                          (bootstrap CI) for Delta_Elo and for the decisive rate
Figure (results/figures/):
  compare_ternary  top: every position's posterior mean per source in the full
                   triangle; bottom: 50%/95% highest-density regions for #868,
                   #518, #674, #226 by source

    python code/compare_levels.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyse_selfplay import (H, N_DRAWS, PRIOR, TERNARY_POSITIONS, balanced_strip, draw_rate_lines,
                              hpd_levels, interval, posterior_draws, rng_for, to_xy)
from plot_style import BAND_COLOUR, ENGINES, METRICS, MUTED_TEXT, apply_style, bold_tick_labels
from utils import REPO_ROOT, STANDARD_POSITION_ID, get_position, setup_logging

log = logging.getLogger("compare_levels")

SELFPLAY = REPO_ROOT / "results" / "selfplay" / "analysis"
LICHESS = REPO_ROOT / "results" / "lichess960" / "analysis" / "lichess960_positions.csv"
OUT = REPO_ROOT / "results" / "compare_levels"
FIGURES = REPO_ROOT / "results" / "figures"

BALANCED_ELO = 5.0
MIN_GAMES = 30                 # positions used in the agreement table
N_BOOT = 2000

# Source key -> label, colour, marker, filled
SOURCES = {
    "lichess_blitz": ("Lichess blitz", "#111111", "o", True),
    "lichess_rapid": ("Lichess rapid", "#8c8c8c", "o", True),
    "caissa": ("Caissa self-play", ENGINES["caissa"].colour, ENGINES["caissa"].marker, True),
    "stockfish": ("Stockfish self-play", ENGINES["stockfish"].colour, ENGINES["stockfish"].marker, True),
}
ENGINE_SOURCES = ("stockfish", "caissa")
HUMAN_SOURCES = ("lichess_blitz", "lichess_rapid")
ZOOM_SOURCES = ("lichess_blitz", "lichess_rapid", "caissa", "stockfish")


def elo(e):
    e = np.clip(e, 1e-9, 1 - 1e-9)
    return 400 * np.log10(e / (1 - e))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_counts(selfplay: Path = SELFPLAY, lichess: Path = LICHESS) -> tuple[pd.DataFrame, dict]:
    """One row per source x position: games, white_wins, draws, black_wins."""
    cols = ["source", "position_id", "games", "white_wins", "draws", "black_wins"]
    parts = []
    sp = pd.read_csv(selfplay / "selfplay_posteriors.csv")
    for e in ENGINE_SOURCES:
        g = sp[sp.engine == e].assign(source=e)
        parts.append(g[cols])
    q = pd.read_csv(selfplay / "selfplay_quality_checks.csv").set_index("engine")
    deff = {e: float(q.loc[e, "deff"]) for e in ENGINE_SOURCES}
    if lichess.exists():
        li = pd.read_csv(lichess)
        for s in ("blitz", "rapid"):
            g = li[li.set == s].assign(source=f"lichess_{s}")
            parts.append(g[cols])
    counts = pd.concat(parts, ignore_index=True)
    counts = counts[counts.games > 0]
    for s in counts.source.unique():
        deff.setdefault(s, 1.0)
    return counts, deff


# ---------------------------------------------------------------------------
# Posteriors and summaries
# ---------------------------------------------------------------------------

def posteriors(counts: pd.DataFrame, deff: dict) -> pd.DataFrame:
    lo_e = 1 / (1 + 10 ** (BALANCED_ELO / 400))
    hi_e = 1 / (1 + 10 ** (-BALANCED_ELO / 400))
    rows = []
    for r in counts.itertuples():
        p = posterior_draws(r.white_wins, r.draws, r.black_wins, deff[r.source],
                            rng_for("compare", r.source, int(r.position_id)))
        e = p[:, 0] + p[:, 1] / 2
        de = elo(e)
        rows.append({"source": r.source, "position_id": r.position_id, "games": r.games,
                     "white_wins": r.white_wins, "draws": r.draws, "black_wins": r.black_wins,
                     "deff": deff[r.source],
                     "escore_mean": e.mean(), "delta_elo_median": float(np.median(de)),
                     "delta_elo_lo": interval(de)[0], "delta_elo_hi": interval(de)[1],
                     "decisive_mean": 1 - p[:, 1].mean(), "decisive_lo": 1 - interval(p[:, 1])[1],
                     "decisive_hi": 1 - interval(p[:, 1])[0],
                     "p_favours_white": float((e > hi_e).mean()),
                     "p_balanced": float(((e >= lo_e) & (e <= hi_e)).mean()),
                     "p_favours_black": float((e < lo_e).mean())})
    return pd.DataFrame(rows)


def source_summary(post: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, g in post.groupby("source", sort=False):
        n = g.games.sum()
        e = (g.white_wins.sum() + 0.5 * g.draws.sum()) / n
        w = g.delta_elo_hi - g.delta_elo_lo
        rows.append({"source": s, "label": SOURCES[s][0], "positions": len(g), "games": int(n),
                     "games_per_position_median": float(g.games.median()),
                     "white_wins": g.white_wins.sum() / n, "draws": g.draws.sum() / n, "decisive": 1 - g.draws.sum() / n,
                     "black_wins": g.black_wins.sum() / n, "escore": e, "delta_elo_pooled": elo(e),
                     "delta_elo_median_of_positions": float(g.delta_elo_median.median()),
                     "ci_width_median": float(w.median()),
                     "positions_ci_above_0": int((g.delta_elo_lo > 0).sum()),
                     "positions_ci_below_0": int((g.delta_elo_hi < 0).sum()),
                     "positions_ci_within_5": int(((g.delta_elo_lo >= -BALANCED_ELO)
                                                   & (g.delta_elo_hi <= BALANCED_ELO)).sum())})
    return pd.DataFrame(rows)


def spearman_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    rho = stats.spearmanr(x, y).statistic
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    boot = [stats.spearmanr(x[i], y[i]).statistic for i in idx]
    return float(rho), float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def agreement(post: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HUMAN_SOURCES:
        hg = post[(post.source == h) & (post.games >= MIN_GAMES)].set_index("position_id")
        if len(hg) < 5:
            continue
        for e in ENGINE_SOURCES:
            eg = post[post.source == e].set_index("position_id").reindex(hg.index)
            rng = rng_for("agreement", h, e)
            r_elo = spearman_ci(eg.delta_elo_median.to_numpy(), hg.delta_elo_median.to_numpy(), rng)
            r_dec = spearman_ci(eg.decisive_mean.to_numpy(), hg.decisive_mean.to_numpy(), rng)
            rows.append({"human": h, "engine": e, "positions": len(hg),
                         "rho_delta_elo": r_elo[0], "rho_delta_elo_lo": r_elo[1], "rho_delta_elo_hi": r_elo[2],
                         "rho_decisive": r_dec[0], "rho_decisive_lo": r_dec[1], "rho_decisive_hi": r_dec[2]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def triangle(ax, labels: bool = True) -> None:
    ax.plot([0, 1, 0.5, 0], [0, 0, H, 0], color="#111111", lw=0.9, zorder=5)
    if labels:
        kw = dict(fontsize=8, fontweight="bold", color="#111111")
        ax.text(0.5, H + 0.03, "Draw", ha="center", va="bottom", **kw)
        ax.text(0.0, -0.04, "Black wins", ha="center", va="top", **kw)
        ax.text(1.0, -0.04, "White wins", ha="center", va="top", **kw)


def means_xy(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = g[["white_wins", "draws", "black_wins"]].to_numpy(float) / g.deff.to_numpy()[:, None] + PRIOR
    return to_xy(n / n.sum(axis=1, keepdims=True))


def contours(ax, r, colour: str, gx, gy, tri, key) -> None:
    p = posterior_draws(r.white_wins, r.draws, r.black_wins, r.deff, rng_for("compare", *key))
    sub = np.vstack(to_xy(p))[:, ::8]
    kde = stats.gaussian_kde(sub)
    z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    l50, l95 = hpd_levels(kde, sub)
    for layer in (ax.contourf(gx, gy, z, levels=[l95, z.max() + 1], colors=[colour], alpha=0.22, zorder=2),
                  ax.contour(gx, gy, z, levels=[l95], colors=[colour], linewidths=0.8, zorder=2),
                  ax.contour(gx, gy, z, levels=[l50], colors=[colour], linewidths=1.5, zorder=2)):
        layer.set_clip_path(tri)


def strip_panel(ax, post: pd.DataFrame, pos: int, sources, xlim=(0.3, 0.75), ylab: bool = False) -> None:
    """One position: highest-density regions per source in a vertical strip
    of the triangle covering every draw rate."""
    from matplotlib.patches import Polygon
    x0, x1 = xlim
    y0, y1 = 0.0, H * 0.98
    draw_rate_lines(ax)
    lo = 1 / (1 + 10 ** (BALANCED_ELO / 400))
    ax.axvspan(lo, 1 - lo, color=BAND_COLOUR, lw=0, zorder=0)
    tri = Polygon([[0, 0], [1, 0], [0.5, H]], closed=True, transform=ax.transData)
    ax.plot([0, 1, 0.5, 0], [0, 0, H, 0], color="#111111", lw=0.9, zorder=5)
    gx, gy = np.meshgrid(np.linspace(x0, x1, 180), np.linspace(y0, y1, 360))
    for s in sources:
        r = post[(post.source == s) & (post.position_id == pos)]
        if r.empty:
            continue
        contours(ax, r.iloc[0], SOURCES[s][1], gx, gy, tri, (s, int(pos)))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_title(f"#{pos}", fontsize=8, fontweight="bold")
    ax.set_xticks([0.4, 0.5, 0.6, 0.7] if x1 > 0.7 else [0.4, 0.5, 0.6])
    yt = np.arange(0, 1.01, 0.2)
    ax.set_yticks(H * yt, [f"{v * 100:.0f}" for v in yt] if ylab else [])
    if ylab:
        ax.set_ylabel("Draws (%)")
    ax.set_xlabel(METRICS["escore"].axis_label)
    bold_tick_labels(ax)


def fig_ternary(post: pd.DataFrame, out_dir: Path) -> list[Path]:
    plt = apply_style()
    fig = plt.figure(figsize=(7.0, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0], hspace=0.12, wspace=0.08)
    ax = fig.add_subplot(gs[0, :])
    draw_rate_lines(ax)
    balanced_strip(ax)
    for s in SOURCES:
        g = post[post.source == s]
        if g.empty:
            continue
        if s == "masters":
            g = g[g.games >= 10]
        if s == "lichess_titled":          # about 8 games per position: pooled marker only
            g_points = g.iloc[:0]
        else:
            g_points = g
        label, colour, marker, filled = SOURCES[s]
        x, y = means_xy(g_points)
        ax.scatter(x, y, s=5 if s not in ("masters", "lichess_titled") else 9, marker=marker,
                   facecolor=colour if filled else "none", edgecolor=colour, lw=0 if filled else 0.5,
                   alpha=0.35 if filled else 0.6, zorder=2)
        n = g[["white_wins", "draws", "black_wins"]].sum().to_numpy(float)
        px, py = to_xy((n / n.sum())[None, :])
        ax.scatter(px, py, s=60, marker=marker, facecolor=colour if filled else "white",
                   edgecolor="#111111" if filled else colour, lw=1.0, zorder=4)
    triangle(ax)
    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(-0.12, H + 0.09)
    ax.set_aspect("equal")
    ax.axis("off")
    for j, pos in enumerate(TERNARY_POSITIONS):
        strip_panel(fig.add_subplot(gs[1, j]), post, pos, ZOOM_SOURCES, ylab=(j == 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    return save(fig, "compare_ternary", out_dir)


def save(fig, name: str, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for p in paths:
        fig.savefig(p, metadata={"CreationDate": None} if p.suffix == ".pdf" else None)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------

def summary_text(src: pd.DataFrame, agree: pd.DataFrame, post: pd.DataFrame) -> str:
    L = ["Human play against engine self-play (Dirichlet posteriors)", "=" * 58, ""]
    for r in src.itertuples():
        L.append(f"{r.label:<20} {r.positions:>4} positions, {r.games:>9,} games (median {r.games_per_position_median:,.0f}"
                 f"/position); W/D/L {100 * r.white_wins:.1f}/{100 * r.draws:.1f}/{100 * r.black_wins:.1f}%; "
                 f"decisive {100 * r.decisive:.1f}%; "
                 f"pooled {r.delta_elo_pooled:+.1f} Elo; median CrI width {r.ci_width_median:.0f}; "
                 f"CrI > 0: {r.positions_ci_above_0}, < 0: {r.positions_ci_below_0}, within +/-5: "
                 f"{r.positions_ci_within_5}")
    L += ["", f"Agreement across positions with >= {MIN_GAMES} games (Spearman rho [95% bootstrap CI]):"]
    for r in agree.itertuples():
        L.append(f"  {SOURCES[r.human][0]} vs {SOURCES[r.engine][0]} ({r.positions} positions): "
                 f"Delta_Elo {r.rho_delta_elo:+.3f} [{r.rho_delta_elo_lo:+.3f}, {r.rho_delta_elo_hi:+.3f}]; "
                 f"decisive rate {r.rho_decisive:+.3f} [{r.rho_decisive_lo:+.3f}, {r.rho_decisive_hi:+.3f}]")
    L += ["", "Positions in the ternary panels (Delta_Elo median [95% CrI]; decisive %):"]
    for pos in TERNARY_POSITIONS:
        parts = []
        for s in SOURCES:
            r = post[(post.source == s) & (post.position_id == pos)]
            if len(r):
                r = r.iloc[0]
                parts.append(f"{SOURCES[s][0]} {r.delta_elo_median:+.0f} [{r.delta_elo_lo:+.0f}, "
                             f"{r.delta_elo_hi:+.0f}], {100 * r.decisive_mean:.0f}% (n {r.games})")
        L.append(f"  #{pos} {get_position(pos).arrangement}: " + "; ".join(parts))
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    counts, deff = load_counts()
    print("Sources: " + ", ".join(f"{SOURCES[s][0]} ({(counts.source == s).sum()} positions, "
                                  f"DEFF {deff[s]:.2f})" for s in SOURCES if s in deff))
    post = posteriors(counts, deff)
    src, agree = source_summary(post), agreement(post)
    OUT.mkdir(parents=True, exist_ok=True)
    post.to_csv(OUT / "compare_posteriors.csv", index=False)
    src.to_csv(OUT / "compare_sources.csv", index=False)
    agree.to_csv(OUT / "compare_agreement.csv", index=False)
    text = summary_text(src, agree, post)
    (OUT / "compare_summary.txt").write_text(text)
    print(text)
    for path in fig_ternary(post, FIGURES):
        if path.suffix == ".png":
            print(f"Figure: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
