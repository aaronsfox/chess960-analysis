"""Analysis of the starting-position evaluations (evaluate_sp).

Three questions, from the evaluations collected by evaluate_sp.py
(results/evaluate_sp/tables/evaluate_sp_final.parquet, MultiPV 1). Metrics:
White's expected score, E_score (LaTeX $E_{\\text{score}}$), and its
Elo-equivalent advantage, Delta_Elo ($\\Delta_{\\text{Elo}}$).

1. Balance. E_score and Delta_Elo for each position at each engine's top
   budget: distribution, ranking, the practically balanced positions
   (|Delta_Elo| <= 5) and where standard chess (#518) sits. Stockfish,
   Caissa and Lc0.
2. Stability. How E_score and Delta_Elo change between successive tenfold
   increases in search budget. Convergence: the changes become progressively
   smaller. Stable: a change in Delta_Elo within +/-5. Each position's
   stabilisation budget is the lowest budget from which every further tenfold
   increase changed Delta_Elo by no more than 5; a position whose final step
   changed by more has not stabilised. Stockfish, Caissa and Lc0.
3. Agreement. Spearman rank correlations between engines' centipawn
   evaluations at the top budgets, with bootstrap intervals. All four engines
   (PlentyChess has no win/draw/loss output, so it appears only here).

Writes tables to results/evaluate_sp/analysis/ (evaluate_sp_*.csv and
evaluate_sp_summary.txt) and one figure per plot to results/figures/
(evaluate_sp_<plot>_<metric>.pdf/.png, metric = escore or delta_elo).
Figure style (fonts, engine colours, markers, line styles, metric labels)
comes from plot_style.py.

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

from plot_style import BAND_COLOUR, ENGINES, LATEX, METRICS, MUTED_TEXT, apply_style, bold_tick_labels
from utils import REPO_ROOT, STANDARD_POSITION_ID, elo_from_expected

TABLE = REPO_ROOT / "results" / "evaluate_sp" / "tables" / "evaluate_sp_final.parquet"
OUT_TABLES = REPO_ROOT / "results" / "evaluate_sp" / "analysis"
OUT_FIGURES = REPO_ROOT / "results" / "figures"

WDL_ENGINES = ["stockfish", "caissa", "lc0"]          # E_score and Delta_Elo available
ALL_ENGINES = ["stockfish", "caissa", "lc0", "plentychess"]

BALANCED_ELO = 5.0      # practically balanced: |Delta_Elo| <= 5
STABLE_ELO = 5.0        # stable: a tenfold step changes Delta_Elo by <= 5
N_BOOT = 2000
SEED = 960


def escore_from_elo(elo: float) -> float:
    return 1.0 / (1.0 + 10 ** (-elo / 400.0))


# E_score bounds of the practically balanced band (|Delta_Elo| <= 5), and the
# change in E_score equivalent to 5 Elo at an even position (for reference
# lines on E_score plots; stability itself is judged on Delta_Elo).
BALANCED_ESCORE = (escore_from_elo(-BALANCED_ELO), escore_from_elo(BALANCED_ELO))
STABLE_ESCORE_AT_EVEN = escore_from_elo(STABLE_ELO) - 0.5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_evaluations(path: Path = TABLE) -> pd.DataFrame:
    """One row per engine, position and budget (MultiPV 1, best line)."""
    df = pd.read_parquet(path)
    df = df[(df.multipv_setting == 1) & (df.multipv_index == 1)].copy()
    df["escore"] = df["expected_score"]
    df["delta_elo"] = df["expected_score"].map(elo_from_expected)
    keep = ["engine", "position_id", "arrangement", "budget_nodes", "cp_white",
            "escore", "delta_elo", "depth"]
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


def describe(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {f"{prefix}_median": np.median(values),
            f"{prefix}_q25": np.percentile(values, 25), f"{prefix}_q75": np.percentile(values, 75),
            f"{prefix}_p2.5": np.percentile(values, 2.5),
            f"{prefix}_p97.5": np.percentile(values, 97.5),
            f"{prefix}_min": values.min(), f"{prefix}_max": values.max(),
            f"{prefix}_sd": values.std(ddof=1)}


# ---------------------------------------------------------------------------
# 1. Balance
# ---------------------------------------------------------------------------

def balance_ranking(top: pd.DataFrame) -> pd.DataFrame:
    """Per engine and position: E_score, Delta_Elo, rank (1 = largest
    advantage for White), percentile (share of positions with a smaller
    advantage) and whether it is practically balanced. PlentyChess is ranked
    on centipawns."""
    rows = []
    for engine, g in top.groupby("engine"):
        g = g.copy()
        value = g["delta_elo"] if engine in WDL_ENGINES else g["cp_white"]
        g["rank"] = value.rank(ascending=False, method="min").astype(int)
        g["percentile"] = value.rank(method="average", pct=True) * 100
        g["balanced"] = (g["delta_elo"].abs() <= BALANCED_ELO) if engine in WDL_ENGINES else np.nan
        g["favours_black"] = value < 0
        rows.append(g)
    out = pd.concat(rows)
    cols = ["engine", "position_id", "arrangement", "budget_nodes", "cp_white",
            "escore", "delta_elo", "rank", "percentile", "balanced", "favours_black"]
    return out[cols].sort_values(["engine", "rank"]).reset_index(drop=True)


def balance_summary(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for engine in WDL_ENGINES:
        g = ranking[ranking.engine == engine]
        e, d = g["escore"].to_numpy(), g["delta_elo"].to_numpy()
        std = g[g.position_id == STANDARD_POSITION_ID].iloc[0]
        e_lo, e_hi = bootstrap_ci(e, np.median)
        d_lo, d_hi = bootstrap_ci(d, np.median)
        top_pos = g.nsmallest(1, "rank").iloc[0]
        rows.append({
            "engine": engine, "budget_nodes": int(g.budget_nodes.iloc[0]), "positions": len(g),
            **describe(e, "escore"), "escore_median_ci_low": e_lo, "escore_median_ci_high": e_hi,
            **describe(d, "delta_elo"), "delta_elo_median_ci_low": d_lo,
            "delta_elo_median_ci_high": d_hi,
            "n_balanced": int(g["balanced"].sum()), "n_favour_black": int(g["favours_black"].sum()),
            "most_unbalanced": int(top_pos.position_id),
            "most_unbalanced_escore": top_pos.escore, "most_unbalanced_delta_elo": top_pos.delta_elo,
            "pos518_escore": std.escore, "pos518_delta_elo": std.delta_elo,
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
    """Per engine and tenfold step: size and direction of the changes in
    E_score and Delta_Elo, and the share of positions whose step is stable."""
    rows = []
    for engine in WDL_ENGINES:
        d_wide, e_wide = wide(df, engine, "delta_elo"), wide(df, engine, "escore")
        budgets = list(d_wide.columns)
        prev = {"escore": np.nan, "delta_elo": np.nan}
        for lo, hi in zip(budgets[:-1], budgets[1:]):
            row = {"engine": engine, "from_budget": int(lo), "to_budget": int(hi)}
            for key, w in (("escore", e_wide), ("delta_elo", d_wide)):
                change = w[hi] - w[lo]
                med = change.abs().median()
                row.update({
                    f"abs_change_{key}_median": med,
                    f"abs_change_{key}_q25": change.abs().quantile(0.25),
                    f"abs_change_{key}_q75": change.abs().quantile(0.75),
                    f"abs_change_{key}_p90": change.abs().quantile(0.90),
                    f"mean_change_{key}": change.mean(),
                    f"ratio_to_previous_step_{key}": med / prev[key] if prev[key] else np.nan,
                })
                prev[key] = med
            row["pct_stable"] = 100 * ((d_wide[hi] - d_wide[lo]).abs() <= STABLE_ELO).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def stability_budgets(df: pd.DataFrame) -> pd.DataFrame:
    """Per engine and budget: the distribution of E_score and Delta_Elo across
    positions and the number of practically balanced positions."""
    rows = []
    for engine in WDL_ENGINES:
        d_wide, e_wide = wide(df, engine, "delta_elo"), wide(df, engine, "escore")
        prev_bal = None
        for b in d_wide.columns:
            bal = d_wide[b].abs() <= BALANCED_ELO
            rows.append({
                "engine": engine, "budget_nodes": int(b),
                **describe(e_wide[b].to_numpy(), "escore"),
                **describe(d_wide[b].to_numpy(), "delta_elo"),
                "n_balanced": int(bal.sum()),
                "n_balance_switched": int((bal != prev_bal).sum()) if prev_bal is not None else np.nan,
            })
            prev_bal = bal
    return pd.DataFrame(rows)


def stabilisation_budget(elo_row: np.ndarray, budgets: list[int], tol: float = STABLE_ELO
                         ) -> float:
    """Lowest budget from which every later tenfold step changes Delta_Elo by
    at most ``tol``; NaN if the final step itself exceeds it."""
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
        d_wide, e_wide = wide(df, engine, "delta_elo"), wide(df, engine, "escore")
        budgets = [int(b) for b in d_wide.columns]
        for pos in d_wide.index:
            x, e = d_wide.loc[pos].to_numpy(), e_wide.loc[pos].to_numpy()
            rows.append({
                "engine": engine, "position_id": pos,
                "escore_top": e[-1], "delta_elo_top": x[-1],
                "final_step_escore": e[-1] - e[-2], "final_step_delta_elo": x[-1] - x[-2],
                "max_abs_step_delta_elo": np.abs(np.diff(x)).max(),
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
# Figures: one plot per file, each for E_score and for Delta_Elo
# ---------------------------------------------------------------------------

FIG_SIZE = (3.5, 2.8)      # single-column width


def new_axes(plt, size=FIG_SIZE):
    fig, ax = plt.subplots(figsize=size)
    return fig, ax


def finish(fig, ax, name: str, out_dir: Path) -> list[Path]:
    bold_tick_labels(ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for p in paths:
        fig.savefig(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def balanced_band(metric: str) -> tuple[float, float]:
    return BALANCED_ESCORE if metric == "escore" else (-BALANCED_ELO, BALANCED_ELO)


def figure_ranked(ranking: pd.DataFrame, metric: str, out_dir: Path) -> list[Path]:
    """Each position's advantage against its rank, #518 marked."""
    plt = apply_style()
    m = METRICS[metric]
    fig, ax = new_axes(plt)
    lo, hi = balanced_band(metric)
    ax.axhspan(lo, hi, color=BAND_COLOUR, lw=0, zorder=0)
    for engine in WDL_ENGINES:
        s = ENGINES[engine]
        g = ranking[ranking.engine == engine].sort_values("rank")
        ax.plot(g["rank"], g[metric], color=s.colour, ls=s.linestyle, label=s.label, zorder=2)
        p = g[g.position_id == STANDARD_POSITION_ID].iloc[0]
        ax.plot(p["rank"], p[metric], marker=s.marker, ms=6, color=s.colour, mec="white",
                mew=1.0, ls="none", zorder=4)
    ax.text(15, (lo + hi) / 2, "Practically balanced", va="center", fontsize=7, color=MUTED_TEXT)
    ax.plot([], [], marker="o", color=MUTED_TEXT, ls="none", ms=5, label="#518")
    ax.set_xlim(0, 961)
    ax.set_xlabel("Rank (1 = largest advantage for White)")
    ax.set_ylabel(m.axis_label)
    ax.legend(loc="upper right")
    return finish(fig, ax, f"evaluate_sp_balance_ranked_{metric}", out_dir)


def figure_distribution(ranking: pd.DataFrame, metric: str, out_dir: Path) -> list[Path]:
    """Kernel density of the advantage across positions, #518 marked."""
    plt = apply_style()
    m = METRICS[metric]
    fig, ax = new_axes(plt)
    lo, hi = balanced_band(metric)
    ax.axvspan(lo, hi, color=BAND_COLOUR, lw=0, zorder=0)
    values = ranking[ranking.engine.isin(WDL_ENGINES)][metric]
    pad = 0.1 * (values.max() - values.min())
    grid = np.linspace(values.min() - pad, values.max() + pad, 600)
    for engine in WDL_ENGINES:
        s = ENGINES[engine]
        g = ranking[ranking.engine == engine]
        density = stats.gaussian_kde(g[metric].to_numpy())(grid)
        ax.fill_between(grid, density, color=s.colour, alpha=0.12, lw=0)
        ax.plot(grid, density, color=s.colour, ls=s.linestyle, label=s.label)
        p = g[g.position_id == STANDARD_POSITION_ID].iloc[0]
        y = stats.gaussian_kde(g[metric].to_numpy())(p[metric])[0]
        ax.plot(p[metric], y, marker=s.marker, ms=6, color=s.colour, mec="white", mew=1.0,
                ls="none", zorder=4)
    ax.plot([], [], marker="o", color=MUTED_TEXT, ls="none", ms=5, label="#518")
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel(m.axis_label)
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    return finish(fig, ax, f"evaluate_sp_balance_distribution_{metric}", out_dir)


def figure_step_change(steps: pd.DataFrame, metric: str, out_dir: Path) -> list[Path]:
    """Median absolute change per tenfold step (interquartile range shaded)."""
    plt = apply_style()
    m = METRICS[metric]
    fig, ax = new_axes(plt)
    ax.set_xscale("log")
    ref = STABLE_ELO if metric == "delta_elo" else STABLE_ESCORE_AT_EVEN
    ax.axhline(ref, color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)))
    ref_label = "5 Elo" if metric == "delta_elo" else "≈5 Elo"
    for engine in WDL_ENGINES:
        s = ENGINES[engine]
        g = steps[steps.engine == engine]
        ax.fill_between(g.to_budget, g[f"abs_change_{metric}_q25"], g[f"abs_change_{metric}_q75"],
                        color=s.colour, alpha=0.12, lw=0)
        ax.plot(g.to_budget, g[f"abs_change_{metric}_median"], color=s.colour, ls=s.linestyle,
                marker=s.marker, label=s.label)
    ax.text(1.4e9, ref, ref_label, va="center", fontsize=7, color=MUTED_TEXT)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Search budget after tenfold increase (nodes)")
    ax.set_ylabel(f"|Change in {m.bold_symbol}|" + (f" ({m.unit})" if m.unit else ""))
    ax.legend(loc="upper right")
    return finish(fig, ax, f"evaluate_sp_stability_step_change_{metric}", out_dir)


def figure_budgets(budgets: pd.DataFrame, metric: str, out_dir: Path) -> list[Path]:
    """Median advantage across positions at each budget (IQR shaded)."""
    plt = apply_style()
    m = METRICS[metric]
    fig, ax = new_axes(plt)
    ax.set_xscale("log")
    lo, hi = balanced_band(metric)
    ax.axhspan(lo, hi, color=BAND_COLOUR, lw=0, zorder=0)
    for engine in WDL_ENGINES:
        s = ENGINES[engine]
        g = budgets[budgets.engine == engine]
        ax.fill_between(g.budget_nodes, g[f"{metric}_q25"], g[f"{metric}_q75"], color=s.colour,
                        alpha=0.12, lw=0)
        ax.plot(g.budget_nodes, g[f"{metric}_median"], color=s.colour, ls=s.linestyle,
                marker=s.marker, label=s.label)
    ax.set_xlabel("Search budget (nodes)")
    ax.set_ylabel(m.axis_label)
    ax.legend(loc="upper right", ncol=3, columnspacing=1.0, handlelength=1.8)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.18 * (ymax - ymin))
    return finish(fig, ax, f"evaluate_sp_stability_budgets_{metric}", out_dir)


def figure_stabilised(counts: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Share of positions stable (every further step within 5 Elo) from each
    budget on. Defined on Delta_Elo only, so there is one version."""
    plt = apply_style()
    fig, ax = new_axes(plt)
    ax.set_xscale("log")
    for engine in WDL_ENGINES:
        s = ENGINES[engine]
        g = counts[(counts.engine == engine) & (counts.budget_nodes > 0)]
        ax.plot(g.budget_nodes, g.pct_stabilised, color=s.colour, ls=s.linestyle,
                marker=s.marker, label=s.label)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Search budget (nodes)")
    ax.set_ylabel(f"Positions stable from budget (%)")
    ax.legend(loc="upper left")
    return finish(fig, ax, "evaluate_sp_stability_stabilised_delta_elo", out_dir)


def figure_agreement(agree: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Lower-triangle matrix of Spearman correlations (centipawns) with 95%
    intervals. Rank-based, so one version covers E_score and Delta_Elo."""
    plt = apply_style()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#eef2f8", "#5b7fb3", "#104485"])
    n = len(ALL_ENGINES)
    mat = np.full((n, n), np.nan)
    text = {}
    for r in agree.itertuples():
        i, j = ALL_ENGINES.index(r.engine_b), ALL_ENGINES.index(r.engine_a)
        mat[i, j] = r.spearman
        text[(i, j)] = f"{r.spearman:.2f}\n[{r.ci_low:.2f}, {r.ci_high:.2f}]"
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
    for (i, j), t in text.items():
        ax.text(j, i, t, ha="center", va="center", fontsize=7, fontweight="bold",
                color="white" if mat[i, j] > 0.55 else "#111111")
    labels = [ENGINES[e].label for e in ALL_ENGINES]
    ax.set_xticks(range(n), labels, rotation=30, ha="right")
    ax.set_yticks(range(n), labels)
    ax.set_xlim(-0.5, n - 1.5)
    ax.set_ylim(n - 0.5, 0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman's ρ")
    cb.outline.set_visible(False)
    for label in cb.ax.get_yticklabels():
        label.set_fontweight("bold")
    return finish(fig, ax, "evaluate_sp_agreement_spearman", out_dir)


def all_figures(ranking, steps, budgets, counts, agree, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for metric in METRICS:
        paths += figure_ranked(ranking, metric, out_dir)
        paths += figure_distribution(ranking, metric, out_dir)
        paths += figure_step_change(steps, metric, out_dir)
        paths += figure_budgets(budgets, metric, out_dir)
    paths += figure_stabilised(counts, out_dir)
    paths += figure_agreement(agree, out_dir)
    return paths


# ---------------------------------------------------------------------------
# Summary text (metric names in LaTeX)
# ---------------------------------------------------------------------------

def summary_text(bal: pd.DataFrame, steps: pd.DataFrame, budgets: pd.DataFrame,
                 counts: pd.DataFrame, agree: pd.DataFrame) -> str:
    E, D = LATEX["escore"], LATEX["delta_elo"]
    label = {k: s.label for k, s in ENGINES.items()}
    lines = ["Starting-position evaluations (evaluate_sp): key numbers",
             "=" * 56, "", "1. Balance (top budget)"]
    for r in bal.itertuples():
        lines.append(
            f"  {label[r.engine]} ({r.budget_nodes:.0e} nodes): {E} median {r.escore_median:.3f} "
            f"(95% CI {r.escore_median_ci_low:.3f} to {r.escore_median_ci_high:.3f}; "
            f"IQR {r.escore_q25:.3f} to {r.escore_q75:.3f}; range {r.escore_min:.3f} to "
            f"{r.escore_max:.3f}); {D} median {r.delta_elo_median:+.1f} (95% CI "
            f"{r.delta_elo_median_ci_low:+.1f} to {r.delta_elo_median_ci_high:+.1f}; IQR "
            f"{r.delta_elo_q25:+.1f} to {r.delta_elo_q75:+.1f}; range {r.delta_elo_min:+.1f} to "
            f"{r.delta_elo_max:+.1f}); practically balanced ({D} within ±5) {r.n_balanced}/960; "
            f"favour Black {r.n_favour_black}; most unbalanced #{r.most_unbalanced} ({E} "
            f"{r.most_unbalanced_escore:.3f}, {D} {r.most_unbalanced_delta_elo:+.1f}); #518 {E} "
            f"{r.pos518_escore:.3f}, {D} {r.pos518_delta_elo:+.1f}, rank {r.pos518_rank}/960, "
            f"{r.pos518_percentile:.0f}th percentile")
    lines += ["", f"2. Stability (per tenfold step: median |change| in {E} and {D} [IQR], mean "
              f"change in {D}, % of positions stable)"]
    for engine in WDL_ENGINES:
        lines.append(f"  {label[engine]}")
        for r in steps[steps.engine == engine].itertuples():
            lines.append(f"    {r.from_budget:.0e} -> {r.to_budget:.0e}: {E} "
                         f"{r.abs_change_escore_median:.4f} [{r.abs_change_escore_q25:.4f} to "
                         f"{r.abs_change_escore_q75:.4f}]; {D} {r.abs_change_delta_elo_median:.1f} "
                         f"[{r.abs_change_delta_elo_q25:.1f} to {r.abs_change_delta_elo_q75:.1f}]; "
                         f"mean {r.mean_change_delta_elo:+.1f}; {r.pct_stable:.0f}% stable")
        b = budgets[budgets.engine == engine]
        lines.append("    practically balanced by budget: " + ", ".join(
            f"{int(x.budget_nodes):.0e}: {x.n_balanced}" for x in b.itertuples()))
        k = counts[counts.engine == engine]
        not_stab = k[k.budget_nodes == -1].iloc[0]
        lines.append("    stable from budget: " + ", ".join(
            f"{int(x.budget_nodes):.0e}: {x.pct_stabilised:.0f}%"
            for x in k[k.budget_nodes > 0].itertuples())
            + f"; not stabilised: {not_stab.n_stabilised} ({not_stab.pct_stabilised:.0f}%)")
    lines += ["", "3. Agreement (Spearman, top-budget centipawns, 95% bootstrap CI)"]
    for r in agree.itertuples():
        lines.append(f"  {label[r.engine_a]} - {label[r.engine_b]}: {r.spearman:.2f} "
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
        paths = all_figures(ranking, steps, budgets, counts, agree, Path(args.figures_dir))
        print(f"wrote {len(paths) // 2} figures (PDF and PNG) to {args.figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
