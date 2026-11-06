"""Main-text figures for the human-play results (Part III).

H1  human_levels: White's advantage (top) and decisive rate (bottom) at equal
    ratings for every dataset and subgroup, with engine self-play.
H2  human_positions_ranked: per-position Delta_Elo ranked, with 95% CIs, for
    Lichess blitz, Lichess rapid and Freestyle Friday; #518 marked.
H3  human_vs_selfplay: per-position human Delta_Elo against Stockfish
    self-play Delta_Elo (Lichess blitz, rapid, Freestyle Friday, masters),
    with Spearman rho and the rho expected under full transfer.
(H4 is compare_ternary, from compare_levels.py.)

Inputs: results/compare_human/ (compare_human_datasets.py) and the position
tables of the three analyses.

    python code/figures_human.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from plot_style import BAND_COLOUR, ENGINES, METRICS, MUTED_TEXT, apply_style, bold_tick_labels
from utils import REPO_ROOT, STANDARD_POSITION_ID

CMP = REPO_ROOT / "results" / "compare_human"
FIG = REPO_ROOT / "results" / "figures"
LI_POS = REPO_ROOT / "results" / "lichess960" / "analysis" / "lichess960_positions.csv"
CC_POS = REPO_ROOT / "results" / "chesscom960" / "analysis" / "chesscom960_positions.csv"
MA_POS = REPO_ROOT / "results" / "masters960" / "masters960_positions.csv"
SELFPLAY = REPO_ROOT / "results" / "selfplay" / "analysis" / "selfplay_posteriors.csv"

# Dataset styles: human datasets in greys with distinct markers; engines in their colours.
STYLE = {
    "lichess_blitz": dict(label="Lichess blitz", color="#111111", marker="o", mfc="#111111"),
    "lichess_rapid": dict(label="Lichess rapid", color="#111111", marker="o", mfc="white"),
    "freestyle_friday": dict(label="Freestyle Friday", color="#555555", marker="s", mfc="#555555"),
    "masters": dict(label="Masters OTB", color="#555555", marker="D", mfc="white"),
}
DELTA = METRICS["delta_elo"].bold_symbol


def save(fig, name: str) -> list[Path]:
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    paths = [FIG / f"{name}.pdf", FIG / f"{name}.png"]
    for p in paths:
        fig.savefig(p, metadata={"CreationDate": None} if p.suffix == ".pdf" else None)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# H1: levels
# ---------------------------------------------------------------------------

GROUP_LABELS = {"all": "All", "lower third": "Lower", "middle third": "Middle", "upper third": "Upper",
                "classical": "Classical", "rapid": "Rapid", "2600+": "2600+"}


def fig_levels() -> list[Path]:
    """Each dataset is a block of columns: its subgroups, with the dataset's
    overall estimate as a horizontal line across the block. Engines at the
    right."""
    plt = apply_style()
    g = pd.read_csv(CMP / "groups_balance_decisive.csv")
    blocks = [("lichess_blitz", ["1200-1499", "1500-1799", "1800-1999", "2000+"]),
              ("lichess_rapid", ["1200-1499", "1500-1799", "1800-1999", "2000+"]),
              ("freestyle_friday", ["lower third", "middle third", "upper third"]),
              ("masters", ["classical", "rapid", "2600+"])]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True, gridspec_kw={"hspace": 0.12})
    x, ticks, tick_labels, block_mid = 0.0, [], [], []
    for ds, groups in blocks:
        st = STYLE[ds]
        start = x
        for grp in groups:
            r = g[(g.dataset == ds) & (g.group == grp)]
            if r.empty:
                continue
            r = r.iloc[0]
            for ax, (est, lo, hi, scale) in zip(axes, [
                    (r.advantage_elo, r.advantage_lo, r.advantage_hi, 1),
                    (r.decisive_adj, r.decisive_adj_lo, r.decisive_adj_hi, 100)]):
                if pd.isna(est):
                    continue
                ax.errorbar(x, scale * est, yerr=[[scale * (est - lo)], [scale * (hi - est)]], fmt=st["marker"],
                            color=st["color"], mfc=st["mfc"], ms=5, elinewidth=1.0, capsize=0, zorder=3)
            ticks.append(x)
            tick_labels.append(GROUP_LABELS.get(grp, grp))
            x += 1
        a = g[(g.dataset == ds) & (g.group == "all")].iloc[0]
        for ax, val in zip(axes, (a.advantage_elo, 100 * a.decisive_adj)):
            ax.hlines(val, start - 0.4, x - 0.6, color=st["color"], lw=0.9, ls=(0, (4, 2)), zorder=2)
        block_mid.append(((start + x - 1) / 2, st["label"]))
        x += 0.8
    for e in ("stockfish", "caissa"):
        r = g[g.dataset == f"{e}_selfplay"].iloc[0]
        es = ENGINES[e]
        axes[0].plot(x, r.advantage_elo, es.marker, color=es.colour, ms=6, zorder=3)
        axes[1].plot(x, 100 * r.decisive_adj, es.marker, color=es.colour, ms=6, zorder=3)
        ticks.append(x)
        tick_labels.append(es.label)
        x += 1
    block_mid.append((x - 1.5, "Engine self-play"))
    axes[0].axhline(0, color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)), zorder=0)
    axes[0].set_ylabel(f"White Advantage {DELTA}")
    axes[1].set_ylabel("Decisive at Equal Ratings (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_xticks(ticks, tick_labels, fontsize=6.5, rotation=40, ha="right", rotation_mode="anchor")
    for xm, lab in block_mid:
        axes[1].text(xm, -0.27, lab, transform=axes[1].get_xaxis_transform(), ha="center", va="top",
                     fontsize=7, fontweight="bold")
    for ax in axes:
        ax.set_xlim(-0.7, x - 0.3)
        bold_tick_labels(ax)
    return save(fig, "human_levels")


# ---------------------------------------------------------------------------
# H2: ranked positions
# ---------------------------------------------------------------------------

def positions() -> dict[str, pd.DataFrame]:
    li = pd.read_csv(LI_POS)
    cc = pd.read_csv(CC_POS)
    out = {"lichess_blitz": li[li.set == "blitz"], "lichess_rapid": li[li.set == "rapid"],
           "freestyle_friday": cc}
    return {k: v[v.human_elo.notna()].copy() for k, v in out.items()}


def fig_ranked() -> list[Path]:
    plt = apply_style()
    pos = positions()
    het = pd.read_csv(CMP / "positions_heterogeneity.csv").set_index("dataset")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True, gridspec_kw={"wspace": 0.08,
                             "width_ratios": [960, 960, 568]})
    for ax, (k, p) in zip(axes, pos.items()):
        g = p.sort_values("human_elo", ascending=False).reset_index(drop=True)
        rank = np.arange(1, len(g) + 1)
        st = STYLE[k]
        ax.vlines(rank, g.human_lo, g.human_hi, color="#9a9a9a", alpha=0.35, lw=0.5, zorder=1)
        ax.plot(rank, g.human_elo, color="#111111", lw=1.3, zorder=2)
        ax.axhline(het.loc[k, "pooled"], color="#111111", lw=0.8, ls=(0, (4, 2)), zorder=3)
        ax.axhline(0, color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)), zorder=0)
        hit = g.index[g.position_id == STANDARD_POSITION_ID]
        if len(hit):
            i = int(hit[0])
            ax.vlines(i + 1, g.loc[i, "human_lo"], g.loc[i, "human_hi"], color=ENGINES["caissa"].colour,
                      lw=1.3, zorder=4)
            ax.plot(i + 1, g.loc[i, "human_elo"], "o", color=ENGINES["caissa"].colour, ms=4, zorder=5)
        ax.set_xlim(0, len(g) + 1)
        ax.set_title(st["label"], fontsize=9, fontweight="bold")
        ax.set_xlabel("Position Rankings")
        bold_tick_labels(ax)
    axes[0].set_ylabel(f"White Advantage {DELTA}")
    axes[0].set_ylim(-150, 175)
    return save(fig, "human_positions_ranked")


# ---------------------------------------------------------------------------
# H3: human vs Stockfish
# ---------------------------------------------------------------------------

def deciles(x: np.ndarray, y: np.ndarray, se: np.ndarray) -> pd.DataFrame:
    """Precision-weighted mean human Delta_Elo within tenths of engine Delta_Elo."""
    q = pd.qcut(x, 10, labels=False)
    rows = []
    for k in range(10):
        m = q == k
        w = 1 / se[m] ** 2
        est = np.sum(w * y[m]) / np.sum(w)
        s = np.sqrt(1 / np.sum(w))
        rows.append((x[m].mean(), est, est - 1.96 * s, est + 1.96 * s))
    return pd.DataFrame(rows, columns=["x", "y", "lo", "hi"])


def fig_vs_selfplay(engine: str = "stockfish") -> list[Path]:
    plt = apply_style()
    from matplotlib.ticker import MultipleLocator
    sp = pd.read_csv(SELFPLAY)
    x_all = sp[sp.engine == engine].set_index("position_id").delta_elo_median
    tr = pd.read_csv(CMP / "transfer_spearman.csv")
    tr = tr[(tr.group == "all") & (tr.engine == engine)].set_index("dataset")
    pos = positions()
    ma = pd.read_csv(MA_POS)
    ma = ma[(ma.games >= 20) & ma.advantage_elo.notna()].rename(
        columns={"advantage_elo": "human_elo", "advantage_se": "human_se",
                 "advantage_lo": "human_lo", "advantage_hi": "human_hi"})
    pos["masters"] = ma
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.4), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.18, "wspace": 0.08})
    lim = (-30, 130)
    for ax, (k, p) in zip(axes.ravel(), pos.items()):
        x = x_all.reindex(p.position_id).to_numpy()
        y, se = p.human_elo.to_numpy(), p.human_se.to_numpy()
        ax.plot(lim, lim, color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)), zorder=1)
        ax.axhline(0, color=BAND_COLOUR, lw=0.8, zorder=0)
        if k == "masters":
            ax.errorbar(x, y, yerr=[y - p.human_lo, p.human_hi - y], fmt="D", ms=3.5, color="#555555",
                        mfc="white", elinewidth=0.7, capsize=0, zorder=3)
        else:
            ax.scatter(x, y, s=4, color="#222222", alpha=0.25, lw=0, zorder=2)
            dc = deciles(x, y, se)
            es = ENGINES[engine]
            ax.errorbar(dc.x, dc.y, yerr=[dc.y - dc.lo, dc.hi - dc.y], fmt=es.marker, ms=4, color=es.colour,
                        mec="white", mew=0.6, elinewidth=1.0, capsize=0, zorder=4)
        r = tr.loc[k]
        ax.text(0.04, 0.96, f"ρ = {r.rho:.2f} [{r.rho_lo:.2f}, {r.rho_hi:.2f}]\n"
                f"Full transfer ρ = {r.rho_if_full_transfer:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5, fontweight="bold", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.9))
        ax.set_title(STYLE[k]["label"], fontsize=9, fontweight="bold")
        ax.set_xlim(*lim)
        ax.set_ylim(-110, 150)
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_major_locator(MultipleLocator(50))
        bold_tick_labels(ax)
    for ax in axes[1]:
        ax.set_xlabel(f"{ENGINES[engine].label} Self-play {DELTA}")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Human Play {DELTA}")
    return save(fig, f"human_vs_selfplay_{engine}")


def main() -> int:
    for path in fig_levels() + fig_ranked() + fig_vs_selfplay("stockfish") + fig_vs_selfplay("caissa"):
        if path.suffix == ".png":
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
