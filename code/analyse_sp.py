"""Part I analysis: White's advantage in the 960 starting positions.

Three questions, from the evaluations collected by evaluate_sp.py
(results/evaluate_sp/tables/evaluate_sp_final.parquet, MultiPV 1):

1. Balance. White's expected score (E) and Elo-equivalent advantage (Delta)
   for each position at each engine's top budget: distribution, ranking, the
   practically balanced positions (|Delta| <= 5 Elo) and where standard chess
   (#518) sits. Stockfish, Caissa and Lc0.
2. Stability. How E and Delta change between successive tenfold increases in
   search budget. Convergence: the changes become progressively smaller.
   Stable: a change within +/-5 Elo. Each position's stabilisation budget is
   the lowest budget from which every further tenfold increase changed Delta
   by no more than 5 Elo; a position whose final step changed by more has not
   stabilised. Stockfish, Caissa and Lc0.
3. Agreement. Spearman rank correlations between engines' centipawn
   evaluations at the top budgets, with bootstrap intervals. All four engines
   (PlentyChess has no win/draw/loss output, so it appears only here).

Writes tables to results/evaluate_sp/analysis/ and figures to
results/figures/ (evaluate_sp_balance, evaluate_sp_stability, evaluate_sp_agreement; PDF and
PNG), plus a plain-text summary of the key numbers.

    python code/analyse_sp.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from utils import REPO_ROOT, STANDARD_POSITION_ID, elo_from_expected

TABLE = REPO_ROOT / "results" / "evaluate_sp" / "tables" / "evaluate_sp_final.parquet"
OUT_TABLES = REPO_ROOT / "results" / "evaluate_sp" / "analysis"
OUT_FIGURES = REPO_ROOT / "results" / "figures"

WDL_ENGINES = ["stockfish", "caissa", "lc0"]          # E and Delta available
ALL_ENGINES = ["stockfish", "caissa", "lc0", "plentychess"]
LABELS = {"stockfish": "Stockfish", "caissa": "Caissa", "lc0": "Lc0",
          "plentychess": "PlentyChess"}
# Fixed engine colours (validated categorical palette; identity is also
# carried by marker and line style, and every series is labelled).
COLOURS = {"stockfish": "#2a78d6", "caissa": "#eb6834", "lc0": "#1baf7a",
           "plentychess": "#eda100"}
MARKERS = {"stockfish": "o", "caissa": "s", "lc0": "^", "plentychess": "D"}
LINESTYLES = {"stockfish": "-", "caissa": "--", "lc0": "-.", "plentychess": ":"}

BALANCED_ELO = 5.0      # practically balanced: |Delta| <= 5 Elo
STABLE_ELO = 5.0        # stable: a tenfold step changes Delta by <= 5 Elo
N_BOOT = 2000
SEED = 960


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_evaluations(path: Path = TABLE) -> pd.DataFrame:
    """One row per engine, position and budget (MultiPV 1, best line)."""
    df = pd.read_parquet(path)
    df = df[(df.multipv_setting == 1) & (df.multipv_index == 1)].copy()
    df["elo"] = df["expected_score"].map(elo_from_expected)
    keep = ["engine", "position_id", "arrangement", "budget_nodes", "cp_white",
            "expected_score", "elo", "depth"]
    return df[keep].sort_values(["engine", "position_id", "budget_nodes"]).reset_index(drop=True)


def top_budget(df: pd.DataFrame) -> pd.DataFrame:
    """Each engine's evaluation at its largest budget."""
    return df.sort_values("budget_nodes").groupby(["engine", "position_id"]).tail(1)


def bootstrap_ci(values: np.ndarray, stat, n: int = N_BOOT, seed: int = SEED
                 ) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    boots = np.array([stat(values[i]) for i in idx])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------------------------------------------------------------------
# 1. Balance
# ---------------------------------------------------------------------------

def balance_ranking(top: pd.DataFrame) -> pd.DataFrame:
    """Per engine and position: E, Delta, rank (1 = largest advantage for
    White), percentile (share of positions with a smaller advantage) and
    whether it is practically balanced. PlentyChess is ranked on centipawns."""
    rows = []
    for engine, g in top.groupby("engine"):
        g = g.copy()
        value = g["elo"] if engine in WDL_ENGINES else g["cp_white"]
        g["rank"] = value.rank(ascending=False, method="min").astype(int)
        g["percentile"] = value.rank(method="average", pct=True) * 100
        g["balanced"] = (g["elo"].abs() <= BALANCED_ELO) if engine in WDL_ENGINES else np.nan
        g["favours_black"] = value < 0
        rows.append(g)
    out = pd.concat(rows)
    cols = ["engine", "position_id", "arrangement", "budget_nodes", "cp_white",
            "expected_score", "elo", "rank", "percentile", "balanced", "favours_black"]
    return out[cols].sort_values(["engine", "rank"]).reset_index(drop=True)


def balance_summary(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for engine in WDL_ENGINES:
        g = ranking[ranking.engine == engine]
        e, d = g["expected_score"].to_numpy(), g["elo"].to_numpy()
        std = g[g.position_id == STANDARD_POSITION_ID].iloc[0]
        lo, hi = bootstrap_ci(d, np.median)
        top_pos = g.nsmallest(1, "rank").iloc[0]
        rows.append({
            "engine": engine, "budget_nodes": int(g.budget_nodes.iloc[0]), "positions": len(g),
            "E_median": np.median(e), "E_q25": np.percentile(e, 25), "E_q75": np.percentile(e, 75),
            "E_p2.5": np.percentile(e, 2.5), "E_p97.5": np.percentile(e, 97.5),
            "E_min": e.min(), "E_max": e.max(),
            "elo_median": np.median(d), "elo_median_ci_low": lo, "elo_median_ci_high": hi,
            "elo_q25": np.percentile(d, 25), "elo_q75": np.percentile(d, 75),
            "elo_p2.5": np.percentile(d, 2.5), "elo_p97.5": np.percentile(d, 97.5),
            "elo_min": d.min(), "elo_max": d.max(), "elo_sd": d.std(ddof=1),
            "n_balanced": int(g["balanced"].sum()), "n_favour_black": int(g["favours_black"].sum()),
            "most_unbalanced": int(top_pos.position_id), "most_unbalanced_elo": top_pos.elo,
            "pos518_E": std.expected_score, "pos518_elo": std.elo,
            "pos518_rank": int(std["rank"]), "pos518_percentile": std.percentile,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Stability and convergence
# ---------------------------------------------------------------------------

def wide(df: pd.DataFrame, engine: str, value: str) -> pd.DataFrame:
    g = df[df.engine == engine].pivot(index="position_id", columns="budget_nodes", values=value)
    return g[sorted(g.columns)]


def stability_steps(df: pd.DataFrame) -> pd.DataFrame:
    """Per engine and tenfold step: size and direction of the changes in E and
    Delta, and the share of positions whose step is stable (<= 5 Elo)."""
    rows = []
    for engine in WDL_ENGINES:
        elo, e = wide(df, engine, "elo"), wide(df, engine, "expected_score")
        budgets = list(elo.columns)
        prev_median = np.nan
        for lo, hi in zip(budgets[:-1], budgets[1:]):
            d_elo, d_e = elo[hi] - elo[lo], e[hi] - e[lo]
            med = d_elo.abs().median()
            rows.append({
                "engine": engine, "from_budget": int(lo), "to_budget": int(hi),
                "abs_change_elo_median": med,
                "abs_change_elo_q25": d_elo.abs().quantile(0.25),
                "abs_change_elo_q75": d_elo.abs().quantile(0.75),
                "abs_change_elo_p90": d_elo.abs().quantile(0.90),
                "abs_change_E_median": d_e.abs().median(),
                "abs_change_E_q75": d_e.abs().quantile(0.75),
                "mean_change_elo": d_elo.mean(), "mean_change_E": d_e.mean(),
                "ratio_to_previous_step": med / prev_median if prev_median else np.nan,
                "pct_stable": 100 * (d_elo.abs() <= STABLE_ELO).mean(),
            })
            prev_median = med
    return pd.DataFrame(rows)


def stability_budgets(df: pd.DataFrame) -> pd.DataFrame:
    """Per engine and budget: the distribution of E and Delta across positions
    and the number of practically balanced positions."""
    rows = []
    for engine in WDL_ENGINES:
        elo, e = wide(df, engine, "elo"), wide(df, engine, "expected_score")
        prev_bal = None
        for b in elo.columns:
            bal = elo[b].abs() <= BALANCED_ELO
            rows.append({
                "engine": engine, "budget_nodes": int(b),
                "E_median": e[b].median(), "E_q25": e[b].quantile(0.25), "E_q75": e[b].quantile(0.75),
                "elo_median": elo[b].median(), "elo_q25": elo[b].quantile(0.25),
                "elo_q75": elo[b].quantile(0.75), "elo_p2.5": elo[b].quantile(0.025),
                "elo_p97.5": elo[b].quantile(0.975), "elo_sd": elo[b].std(),
                "n_balanced": int(bal.sum()),
                "n_balance_switched": int((bal != prev_bal).sum()) if prev_bal is not None else np.nan,
            })
            prev_bal = bal
    return pd.DataFrame(rows)


def stabilisation_budget(elo_row: np.ndarray, budgets: list[int], tol: float = STABLE_ELO
                         ) -> float:
    """Lowest budget from which every later tenfold step changes Delta by at
    most ``tol``; NaN if the final step itself exceeds it."""
    steps = np.abs(np.diff(elo_row))
    if steps[-1] > tol:
        return np.nan
    start = len(steps)                 # index of the step after the last unstable one
    for i in range(len(steps) - 1, -1, -1):
        if steps[i] > tol:
            break
        start = i
    return float(budgets[start])


def stability_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Per engine and position: the final step, whether it is stable, and the
    stabilisation budget."""
    rows = []
    for engine in WDL_ENGINES:
        elo, e = wide(df, engine, "elo"), wide(df, engine, "expected_score")
        budgets = [int(b) for b in elo.columns]
        for pos in elo.index:
            x = elo.loc[pos].to_numpy()
            rows.append({
                "engine": engine, "position_id": pos,
                "elo_top": x[-1], "E_top": e.loc[pos].iloc[-1],
                "final_step_elo": x[-1] - x[-2],
                "final_step_E": e.loc[pos].iloc[-1] - e.loc[pos].iloc[-2],
                "max_abs_step_elo": np.abs(np.diff(x)).max(),
                "stable_final_step": abs(x[-1] - x[-2]) <= STABLE_ELO,
                "stabilisation_budget": stabilisation_budget(x, budgets),
                "balanced_top": abs(x[-1]) <= BALANCED_ELO,
                "balanced_previous": abs(x[-2]) <= BALANCED_ELO,
            })
    return pd.DataFrame(rows)


def stabilisation_counts(pos: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative share of positions stabilised by each budget."""
    rows = []
    for engine in WDL_ENGINES:
        g = pos[pos.engine == engine]
        budgets = sorted(df[df.engine == engine].budget_nodes.unique())
        for b in budgets[:-1]:
            n = int((g.stabilisation_budget <= b).sum())
            rows.append({"engine": engine, "budget_nodes": int(b), "n_stabilised": n,
                         "pct_stabilised": 100 * n / len(g)})
        rows.append({"engine": engine, "budget_nodes": -1,
                     "n_stabilised": int(g.stabilisation_budget.isna().sum()),
                     "pct_stabilised": 100 * g.stabilisation_budget.isna().mean()})
    out = pd.DataFrame(rows)
    out["note"] = np.where(out.budget_nodes == -1, "not stabilised (final step > 5 Elo)", "")
    return out


# ---------------------------------------------------------------------------
# 3. Agreement
# ---------------------------------------------------------------------------

def agreement(top: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Spearman correlations of top-budget centipawn evaluations,
    with percentile bootstrap intervals over positions."""
    cp = top.pivot(index="position_id", columns="engine", values="cp_white")[ALL_ENGINES]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(cp), size=(N_BOOT, len(cp)))
    ranks = cp.rank()
    rows = []
    for a, b in itertools.combinations(ALL_ENGINES, 2):
        rho = stats.spearmanr(cp[a], cp[b]).statistic
        x, y = cp[a].to_numpy(), cp[b].to_numpy()
        boots = [stats.spearmanr(x[i], y[i]).statistic for i in idx]
        rows.append({"engine_a": a, "engine_b": b, "spearman": rho,
                     "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                     "positions": len(cp),
                     "median_abs_rank_difference": (ranks[a] - ranks[b]).abs().median()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e",
        "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b", "grid.color": "#e4e3df",
        "grid.linewidth": 0.5, "lines.linewidth": 1.6, "savefig.dpi": 300,
    })
    return plt


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")


def save(fig, name: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    return paths


def figure_balance(ranking: pd.DataFrame, out_dir: Path):
    """(a) Delta against rank for each engine, (b) distribution of Delta;
    the practically balanced band shaded and #518 marked."""
    plt = setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.3})
    for ax in (ax1, ax2):
        ax.grid(axis="y" if ax is ax1 else "x")
    ax1.axhspan(-BALANCED_ELO, BALANCED_ELO, color="#d9d8d3", alpha=0.6, lw=0, zorder=0)
    ax2.axvspan(-BALANCED_ELO, BALANCED_ELO, color="#d9d8d3", alpha=0.6, lw=0, zorder=0)
    bins = np.arange(-10, 95, 2.5)
    for engine in WDL_ENGINES:
        g = ranking[ranking.engine == engine].sort_values("rank")
        c = COLOURS[engine]
        ax1.plot(g["rank"], g["elo"], color=c, ls=LINESTYLES[engine], label=LABELS[engine])
        s = g[g.position_id == STANDARD_POSITION_ID].iloc[0]
        ax1.plot(s["rank"], s.elo, marker=MARKERS[engine], ms=7, color=c, mec="#fcfcfb", mew=1.2,
                 zorder=5)
        ax2.hist(g["elo"], bins=bins, histtype="step", color=c, ls=LINESTYLES[engine], lw=1.4,
                 label=LABELS[engine])
        ax2.axvline(s.elo, color=c, ls=LINESTYLES[engine], lw=0.9)
    ax1.text(0.97, 0.62, "Markers: #518", transform=ax1.transAxes, ha="right",
             fontsize=7, color="#52514e")
    ax1.text(20, 0, "±5 Elo", ha="left", va="center", fontsize=7, color="#52514e")
    ax1.set_xlabel("Rank (1 = largest advantage for White)")
    ax1.set_ylabel("White's advantage, Δ (Elo)")
    ax1.set_xlim(0, 961)
    ax1.legend(frameon=False, loc="upper right")
    ax2.set_xlabel("White's advantage, Δ (Elo)")
    ax2.set_ylabel("Positions")
    ax2.text(0.97, 0.93, "Vertical lines: #518", transform=ax2.transAxes, ha="right",
             fontsize=7, color="#52514e")
    ax2.legend(frameon=False, loc="center right")
    panel_letter(ax1, "a")
    panel_letter(ax2, "b")
    paths = save(fig, "evaluate_sp_balance", out_dir)
    plt.close(fig)
    return paths


def figure_stability(steps: pd.DataFrame, budgets: pd.DataFrame, counts: pd.DataFrame,
                     out_dir: Path):
    """(a) size of the change per tenfold step, (b) distribution of Delta at
    each budget, (c) share of positions stabilised by each budget."""
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.42})
    ax1, ax2, ax3 = axes
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(axis="y")
    ax1.axhline(STABLE_ELO, color="#8a8984", lw=0.8, ls=":")
    for engine in WDL_ENGINES:
        c, ls, mk = COLOURS[engine], LINESTYLES[engine], MARKERS[engine]
        s = steps[steps.engine == engine]
        ax1.fill_between(s.to_budget, s.abs_change_elo_q25, s.abs_change_elo_q75, color=c,
                         alpha=0.15, lw=0)
        ax1.plot(s.to_budget, s.abs_change_elo_median, color=c, ls=ls, marker=mk, ms=4,
                 label=LABELS[engine])
        b = budgets[budgets.engine == engine]
        ax2.fill_between(b.budget_nodes, b.elo_q25, b.elo_q75, color=c, alpha=0.15, lw=0)
        ax2.plot(b.budget_nodes, b.elo_median, color=c, ls=ls, marker=mk, ms=4)
        k = counts[(counts.engine == engine) & (counts.budget_nodes > 0)]
        ax3.plot(k.budget_nodes, k.pct_stabilised, color=c, ls=ls, marker=mk, ms=4)
    ax1.text(1.2e9, STABLE_ELO, "5 Elo", fontsize=7, color="#52514e", va="center")
    ax1.set_xlabel("Search budget after step (nodes)")
    ax1.set_ylabel("|Change in Δ| per tenfold step (Elo)")
    ax1.set_ylim(bottom=0)
    ax1.legend(frameon=False, loc="upper right")
    ax2.axhspan(-BALANCED_ELO, BALANCED_ELO, color="#d9d8d3", alpha=0.6, lw=0, zorder=0)
    ax2.set_xlabel("Search budget (nodes)")
    ax2.set_ylabel("Δ across positions (Elo)")
    ax3.set_xlabel("Search budget (nodes)")
    ax3.set_ylabel("Positions stable from this budget on (%)")
    ax3.set_ylim(0, 100)
    for ax, letter in zip(axes, "abc"):
        panel_letter(ax, letter)
    paths = save(fig, "evaluate_sp_stability", out_dir)
    plt.close(fig)
    return paths


def figure_agreement(agree: pd.DataFrame, out_dir: Path):
    """Lower-triangle matrix of Spearman correlations with 95% intervals."""
    plt = setup_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#2a78d6", "#0d3d7a"])
    n = len(ALL_ENGINES)
    m = np.full((n, n), np.nan)
    text = {}
    for r in agree.itertuples():
        i, j = ALL_ENGINES.index(r.engine_b), ALL_ENGINES.index(r.engine_a)
        m[i, j] = r.spearman
        text[(i, j)] = f"{r.spearman:.2f}\n[{r.ci_low:.2f}, {r.ci_high:.2f}]"
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
    for (i, j), t in text.items():
        ax.text(j, i, t, ha="center", va="center", fontsize=7,
                color="#ffffff" if m[i, j] > 0.6 else "#0b0b0b")
    ax.set_xticks(range(n), [LABELS[e] for e in ALL_ENGINES], rotation=30, ha="right")
    ax.set_yticks(range(n), [LABELS[e] for e in ALL_ENGINES])
    ax.set_xlim(-0.5, n - 1.5)
    ax.set_ylim(n - 0.5, 0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman's ρ (top-budget centipawns)")
    cb.outline.set_visible(False)
    paths = save(fig, "evaluate_sp_agreement", out_dir)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

def summary_text(bal: pd.DataFrame, steps: pd.DataFrame, budgets: pd.DataFrame,
                 counts: pd.DataFrame, agree: pd.DataFrame) -> str:
    lines = ["Part I: key numbers", "=" * 19, "", "1. Balance (top budget)"]
    for r in bal.itertuples():
        lines.append(
            f"  {LABELS[r.engine]} ({r.budget_nodes:.0e} nodes): E median {r.E_median:.3f} "
            f"(IQR {r.E_q25:.3f}-{r.E_q75:.3f}, range {r.E_min:.3f}-{r.E_max:.3f}); "
            f"Delta median {r.elo_median:+.1f} Elo (95% CI {r.elo_median_ci_low:+.1f} to "
            f"{r.elo_median_ci_high:+.1f}; IQR {r.elo_q25:+.1f} to {r.elo_q75:+.1f}; "
            f"range {r.elo_min:+.1f} to {r.elo_max:+.1f}); practically balanced "
            f"{r.n_balanced}/960; favour Black {r.n_favour_black}; most unbalanced "
            f"#{r.most_unbalanced} ({r.most_unbalanced_elo:+.1f} Elo); #518 E {r.pos518_E:.3f}, "
            f"Delta {r.pos518_elo:+.1f} Elo, rank {r.pos518_rank}/960, "
            f"{r.pos518_percentile:.0f}th percentile")
    lines += ["", "2. Stability (per tenfold step: median |change in Delta| [IQR], mean change, "
              "% of positions stable)"]
    for engine in WDL_ENGINES:
        lines.append(f"  {LABELS[engine]}")
        for r in steps[steps.engine == engine].itertuples():
            lines.append(f"    {r.from_budget:.0e} -> {r.to_budget:.0e}: "
                         f"{r.abs_change_elo_median:.1f} Elo [{r.abs_change_elo_q25:.1f}-"
                         f"{r.abs_change_elo_q75:.1f}] (E {r.abs_change_E_median:.4f}); mean "
                         f"{r.mean_change_elo:+.1f} Elo; {r.pct_stable:.0f}% stable")
        b = budgets[budgets.engine == engine]
        lines.append("    practically balanced by budget: " + ", ".join(
            f"{int(x.budget_nodes):.0e}: {x.n_balanced}" for x in b.itertuples()))
        k = counts[counts.engine == engine]
        not_stab = k[k.budget_nodes == -1].iloc[0]
        lines.append("    stabilised by budget: " + ", ".join(
            f"{int(x.budget_nodes):.0e}: {x.pct_stabilised:.0f}%"
            for x in k[k.budget_nodes > 0].itertuples())
            + f"; not stabilised: {not_stab.n_stabilised} ({not_stab.pct_stabilised:.0f}%)")
    lines += ["", "3. Agreement (Spearman, top-budget centipawns, 95% bootstrap CI)"]
    for r in agree.itertuples():
        lines.append(f"  {LABELS[r.engine_a]} - {LABELS[r.engine_b]}: {r.spearman:.2f} "
                     f"[{r.ci_low:.2f}, {r.ci_high:.2f}]")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default=str(TABLE))
    p.add_argument("--tables-dir", default=str(OUT_TABLES))
    p.add_argument("--figures-dir", default=str(OUT_FIGURES))
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    df = load_evaluations(Path(args.table))
    top = top_budget(df)
    ranking = balance_ranking(top)
    bal = balance_summary(ranking)
    steps = stability_steps(df)
    budgets = stability_budgets(df)
    pos = stability_positions(df)
    counts = stabilisation_counts(pos, df)
    agree = agreement(top)

    out = Path(args.tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "evaluate_sp_balance_summary.csv": bal,
        "evaluate_sp_balance_ranking.csv": ranking,
        "evaluate_sp_stability_steps.csv": steps,
        "evaluate_sp_stability_budgets.csv": budgets,
        "evaluate_sp_stability_positions.csv": pos,
        "evaluate_sp_stability_stabilised.csv": counts,
        "evaluate_sp_agreement_spearman.csv": agree,
    }
    for name, table in outputs.items():
        table.to_csv(out / name, index=False, float_format="%.6g")
    text = summary_text(bal, steps, budgets, counts, agree)
    (out / "evaluate_sp_summary.txt").write_text(text)
    print(text)
    if not args.no_figures:
        fig_dir = Path(args.figures_dir)
        for paths in (figure_balance(ranking, fig_dir),
                      figure_stability(steps, budgets, counts, fig_dir),
                      figure_agreement(agree, fig_dir)):
            print("wrote", ", ".join(str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT)
                                     else str(p) for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
