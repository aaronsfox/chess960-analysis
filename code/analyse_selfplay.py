"""Analysis of the self-play games (selfplay).

Follows the self-play analysis plan: quality checks, then three questions
for Stockfish and Caissa self-play from every starting position (500 games
per position: 50 book lines x 10 games). Metrics: White's expected score,
E_score ($E_{\\text{score}}$), and its Elo-equivalent advantage, Delta_Elo
($\\Delta_{\\text{Elo}}$).

Shared framework. Each position's win/draw/loss counts get a Dirichlet
posterior, Dirichlet(1/2 + (W, D, L) / DEFF): a Jeffreys prior, with counts
deflated by the design effect DEFF = 1 + (m - 1) ICC for correlation among
the m = 10 games sharing a book line (ICC pooled across positions, per
engine). Every derived quantity comes from Monte Carlo draws of these
posteriors. A Bayesian bootstrap over book lines is the sensitivity check.

1. Balance. (a) Self-play E_score and Delta_Elo regressed on each engine's
   top-budget evaluation of the same position (from evaluate_sp). (b) From
   the posteriors: E_score and Delta_Elo with 95% intervals; P(favours
   White) = P(Delta_Elo > 5), P(practically balanced) = P(|Delta_Elo| <= 5),
   P(favours Black); and the comparison with standard chess (#518):
   P(at least as fair as #518), P(more favourable to White than #518) and
   the difference from #518.
2. Agreement between Stockfish and Caissa games: Spearman's rho (observed
   and corrected for noise), Bland-Altman agreement, level and spread,
   classification agreement (Cohen's kappa) and overlap of extremes.
3. Decisiveness: decisive rate (1 - draw rate) per position, its relation to
   |Delta_Elo| (logistic regression weighted by the effective sample size),
   White's share of decisive games, how games end by advantage, and
   agreement between engines.

Positions are ranked rather than selected: selfplay_position_table.csv lists
all 960 with both engines' estimates, intervals and probabilities, ranked by
advantage, fairness, decisiveness and by P(at least as fair as and more
decisive than #518). Counts at probability thresholds 0.80 and 0.95 are
reported in the summary for transparency only.

Writes tables to results/selfplay/analysis/ (selfplay_*.csv and
selfplay_summary.txt) and figures to results/figures/ (selfplay_*.pdf/.png,
styled by plot_style.py).

    python code/analyse_selfplay.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from analyse_sp import load_evaluations, top_budget
from plot_style import BAND_COLOUR, ENGINES, LATEX, METRICS, MUTED_TEXT, apply_style, bold_tick_labels
from utils import REPO_ROOT, STANDARD_POSITION_ID

TABLES = REPO_ROOT / "results" / "selfplay" / "tables"
BOOK = REPO_ROOT / "results" / "opening_book" / "book" / "opening_book_lines.parquet"
OUT_TABLES = REPO_ROOT / "results" / "selfplay" / "analysis"
OUT_FIGURES = REPO_ROOT / "results" / "figures"

SELFPLAY_ENGINES = ["stockfish", "caissa"]
EVAL_ENGINES = ["stockfish", "caissa", "lc0", "plentychess"]
GAMES_PER_LINE = 10
BALANCED_ELO = 5.0
PRIOR = 0.5                    # Jeffreys prior, Dirichlet(1/2, 1/2, 1/2)
N_DRAWS = 20_000
N_BOOT_LINES = 2_000
N_BOOT = 2_000
THRESHOLDS = (0.80, 0.95)
SEED = 960


def elo(e):
    e = np.clip(e, 1e-9, 1 - 1e-9)
    return 400.0 * np.log10(e / (1.0 - e))


def rng_for(*keys) -> np.random.Generator:
    """Reproducible generator per engine/position (crc32 of string keys, so
    the draws don't depend on Python's per-process hash seed)."""
    return np.random.default_rng([SEED, *[k if isinstance(k, int) else zlib.crc32(str(k).encode())
                                          for k in keys]])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_games(tables: Path = TABLES) -> pd.DataFrame:
    return pd.read_parquet(tables / "selfplay_games.parquet")


def position_counts(games: pd.DataFrame) -> pd.DataFrame:
    g = games.groupby(["engine", "position_id"])
    out = g.agg(games=("score_white", "size"),
                white_wins=("score_white", lambda x: int((x == 1).sum())),
                draws=("score_white", lambda x: int((x == 0.5).sum())),
                black_wins=("score_white", lambda x: int((x == 0).sum())),
                mean_plies=("plies", "mean")).reset_index()
    return out


def line_counts(games: pd.DataFrame) -> pd.DataFrame:
    g = games.groupby(["engine", "position_id", "line_index"])
    return g.agg(w=("score_white", lambda x: int((x == 1).sum())),
                 d=("score_white", lambda x: int((x == 0.5).sum())),
                 l=("score_white", lambda x: int((x == 0).sum()))).reset_index()


# ---------------------------------------------------------------------------
# Quality checks and design effect
# ---------------------------------------------------------------------------

def icc_one_way(games: pd.DataFrame, value: str = "score_white", m: int = GAMES_PER_LINE) -> float:
    """Pooled one-way ANOVA ICC of games within book lines, within positions
    (position means removed first)."""
    g = games.copy()
    g["res"] = g[value] - g.groupby("position_id")[value].transform("mean")
    lm = g.groupby(["position_id", "line_index"])["res"].agg(["mean", "var"])
    n_pos = g.position_id.nunique()
    msb = m * (lm["mean"] ** 2).sum() / (len(lm) - n_pos)
    msw = lm["var"].mean()
    return float((msb - msw) / (msb + (m - 1) * msw))


def quality_checks(games: pd.DataFrame, book: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, deff = [], {}
    for engine, g in games.groupby("engine"):
        per_pos = g.groupby("position_id").size()
        icc = icc_one_way(g)
        deff[engine] = 1 + (GAMES_PER_LINE - 1) * icc
        # Book drift: within-position slope of game score on the line's drift,
        # and the ICC once that drift is accounted for.
        m = g.merge(book, on=["position_id", "line_index"])
        m["res"] = m.score_white - m.groupby("position_id").score_white.transform("mean")
        m["shift_c"] = m.shift_expected_check - m.groupby("position_id").shift_expected_check.transform("mean")
        fit = stats.linregress(m.shift_c, m.res)
        m["adj"] = m.score_white - fit.slope * m.shift_c
        icc_adj = icc_one_way(m, "adj")
        copies = g.groupby("white_player").score_white.mean()
        nm = (g.plies - g.book_plies) / 2
        nps = ((g.mean_nodes_white * np.ceil(nm) + g.mean_nodes_black * np.floor(nm))
               / (g.time_used_white + g.time_used_black))
        rows.append({
            "engine": engine, "positions": len(per_pos), "games": len(g),
            "games_per_position_min": int(per_pos.min()), "games_per_position_max": int(per_pos.max()),
            "unmatched_lines": int((g.line_index < 0).sum()),
            "losses_on_time": int((g.termination == "time forfeit").sum()),
            "abnormal_terminations": int((~g.termination.isin(["normal", "adjudication"])).sum()),
            "white_win_rate": (g.score_white == 1).mean(), "draw_rate": (g.score_white == 0.5).mean(),
            "black_win_rate": (g.score_white == 0).mean(), "mean_plies": g.plies.mean(),
            "copy_a_white_score": copies.iloc[0], "copy_b_white_score": copies.iloc[1],
            "icc": icc, "deff": deff[engine],
            "book_shift_slope": fit.slope, "book_shift_slope_se": fit.stderr, "book_shift_r": fit.rvalue,
            "icc_book_adjusted": icc_adj, "deff_book_adjusted": 1 + (GAMES_PER_LINE - 1) * icc_adj,
            "max_abs_position_mean_shift": float(book.groupby("position_id").shift_expected_check
                                                 .mean().abs().max()),
            "nodes_per_second_median": float(nps.median()),
        })
    return pd.DataFrame(rows), deff


# ---------------------------------------------------------------------------
# Dirichlet posteriors
# ---------------------------------------------------------------------------

def posterior_draws(w: float, d: float, l: float, deff: float, rng: np.random.Generator,
                    n: int = N_DRAWS) -> np.ndarray:
    alpha = PRIOR + np.array([w, d, l], dtype=float) / deff
    return rng.dirichlet(alpha, n)


def interval(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def posteriors(counts: pd.DataFrame, deff: dict[str, float]) -> pd.DataFrame:
    """Per engine and position: posterior summaries, probabilities and the
    comparison with #518."""
    rows = []
    for engine, g in counts.groupby("engine"):
        ref = g[g.position_id == STANDARD_POSITION_ID].iloc[0]
        ref_p = posterior_draws(ref.white_wins, ref.draws, ref.black_wins, deff[engine],
                                rng_for(engine, "ref"))
        ref_e = ref_p[:, 0] + ref_p[:, 1] / 2
        ref_elo, ref_dec = elo(ref_e), 1 - ref_p[:, 1]
        for r in g.itertuples():
            p = posterior_draws(r.white_wins, r.draws, r.black_wins, deff[engine],
                                rng_for(engine, int(r.position_id)))
            e = p[:, 0] + p[:, 1] / 2
            de = elo(e)
            dec = 1 - p[:, 1]
            share = p[:, 0] / (p[:, 0] + p[:, 2])
            e_lo, e_hi = interval(e)
            d_lo, d_hi = interval(de)
            dec_lo, dec_hi = interval(dec)
            sh_lo, sh_hi = interval(share)
            diff = de - ref_elo
            df_lo, df_hi = interval(diff)
            rows.append({
                "engine": engine, "position_id": int(r.position_id),
                "games": int(r.games), "white_wins": int(r.white_wins), "draws": int(r.draws),
                "black_wins": int(r.black_wins), "mean_plies": r.mean_plies,
                "escore_observed": (r.white_wins + r.draws / 2) / r.games,
                "escore_mean": e.mean(), "escore_sd": e.std(), "escore_lo": e_lo, "escore_hi": e_hi,
                "delta_elo_median": float(np.median(de)), "delta_elo_sd": de.std(),
                "delta_elo_lo": d_lo, "delta_elo_hi": d_hi,
                "p_favours_white": float((de > BALANCED_ELO).mean()),
                "p_balanced": float((np.abs(de) <= BALANCED_ELO).mean()),
                "p_favours_black": float((de < -BALANCED_ELO).mean()),
                "decisive_mean": dec.mean(), "decisive_lo": dec_lo, "decisive_hi": dec_hi,
                "white_share_decisive_mean": share.mean(), "white_share_decisive_lo": sh_lo,
                "white_share_decisive_hi": sh_hi,
                "diff_518_delta_elo_median": float(np.median(diff)), "diff_518_lo": df_lo,
                "diff_518_hi": df_hi,
                "p_at_least_as_fair_as_518": float((np.abs(de) <= np.abs(ref_elo)).mean()),
                "p_more_white_than_518": float((de > ref_elo).mean()),
                "p_more_decisive_than_518": float((dec > ref_dec).mean()),
                "p_fairer_and_more_decisive_than_518": float(((np.abs(de) <= np.abs(ref_elo))
                                                              & (dec > ref_dec)).mean()),
            })
    out = pd.DataFrame(rows)
    out["rank_delta_elo"] = out.groupby("engine").delta_elo_median.rank(ascending=False,
                                                                        method="min").astype(int)
    out["category_080"] = out.apply(lambda r: category(r, 0.80), axis=1)
    out["category_095"] = out.apply(lambda r: category(r, 0.95), axis=1)
    return out


def category(r, threshold: float) -> str:
    if r.p_favours_white >= threshold:
        return "favours White"
    if r.p_favours_black >= threshold:
        return "favours Black"
    if r.p_balanced >= threshold:
        return "practically balanced"
    return "inconclusive"


def bayesian_bootstrap(lines: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity check: posterior SD of E_score from a Bayesian bootstrap
    over book lines, against the design-effect Dirichlet."""
    rows = []
    for (engine, pos), g in lines.groupby(["engine", "position_id"]):
        n = g[["w", "d", "l"]].sum(axis=1).to_numpy(float)
        e_line = (g.w + g.d / 2).to_numpy(float) / n
        wts = rng_for(engine, int(pos), "bb").dirichlet(np.ones(len(g)), N_BOOT_LINES)
        e = wts @ e_line
        rows.append({"engine": engine, "position_id": int(pos), "escore_sd_bootstrap": e.std(),
                     "escore_lo_bootstrap": np.percentile(e, 2.5),
                     "escore_hi_bootstrap": np.percentile(e, 97.5)})
    bb = pd.DataFrame(rows).merge(post[["engine", "position_id", "escore_sd", "escore_mean"]],
                                  on=["engine", "position_id"])
    bb["sd_ratio_bootstrap_to_dirichlet"] = bb.escore_sd_bootstrap / bb.escore_sd
    return bb


# ---------------------------------------------------------------------------
# 1. Balance
# ---------------------------------------------------------------------------

def reliability(post: pd.DataFrame, engine: str) -> float:
    g = post[post.engine == engine]
    return float(1 - (g.escore_sd ** 2).mean() / g.escore_mean.var())


def balance_regression(post: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
    """Self-play outcome regressed on each engine's top-budget evaluation."""
    rows = []
    for sp in SELFPLAY_ENGINES:
        y = post[post.engine == sp].set_index("position_id")
        rel = reliability(post, sp)
        for ev in EVAL_ENGINES:
            x = top[top.engine == ev].set_index("position_id").reindex(y.index)
            for metric, xcol, ycol in (("escore", "escore", "escore_mean"),
                                       ("delta_elo", "delta_elo", "delta_elo_median"),
                                       ("centipawns", "cp_white", "escore_mean")):
                if ev == "plentychess" and metric != "centipawns":
                    continue
                if ev != "plentychess" and metric == "centipawns":
                    continue
                xv, yv = x[xcol].to_numpy(float), y[ycol].to_numpy(float)
                f = stats.linregress(xv, yv)
                rows.append({"selfplay_engine": sp, "evaluation_engine": ev, "metric": metric,
                             "slope": f.slope, "slope_se": f.stderr, "intercept": f.intercept,
                             "r2": f.rvalue ** 2, "pearson": f.rvalue,
                             "spearman": stats.spearmanr(xv, yv).statistic,
                             "selfplay_reliability": rel,
                             "pearson_corrected_for_selfplay_noise": f.rvalue / np.sqrt(rel)})
    return pd.DataFrame(rows)


def balance_summary(post: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for engine, g in post.groupby("engine"):
        ref = g[g.position_id == STANDARD_POSITION_ID].iloc[0]
        top_pos = g.nsmallest(1, "rank_delta_elo").iloc[0]
        row = {"engine": engine, "positions": len(g),
               "escore_median": g.escore_mean.median(), "escore_q25": g.escore_mean.quantile(.25),
               "escore_q75": g.escore_mean.quantile(.75), "escore_min": g.escore_mean.min(),
               "escore_max": g.escore_mean.max(),
               "delta_elo_median": g.delta_elo_median.median(),
               "delta_elo_q25": g.delta_elo_median.quantile(.25),
               "delta_elo_q75": g.delta_elo_median.quantile(.75),
               "delta_elo_min": g.delta_elo_median.min(), "delta_elo_max": g.delta_elo_median.max(),
               "delta_elo_true_sd": float(np.sqrt(max(g.delta_elo_median.var()
                                                       - (g.delta_elo_sd ** 2).mean(), 0))),
               "mean_interval_width_delta_elo": (g.delta_elo_hi - g.delta_elo_lo).mean(),
               "expected_n_favours_white": g.p_favours_white.sum(),
               "expected_n_balanced": g.p_balanced.sum(),
               "expected_n_favours_black": g.p_favours_black.sum(),
               "expected_n_at_least_as_fair_as_518": g.p_at_least_as_fair_as_518.sum(),
               "most_unbalanced": int(top_pos.position_id),
               "most_unbalanced_delta_elo": top_pos.delta_elo_median,
               "pos518_escore": ref.escore_mean, "pos518_escore_lo": ref.escore_lo,
               "pos518_escore_hi": ref.escore_hi, "pos518_delta_elo": ref.delta_elo_median,
               "pos518_delta_elo_lo": ref.delta_elo_lo, "pos518_delta_elo_hi": ref.delta_elo_hi,
               "pos518_p_favours_white": ref.p_favours_white, "pos518_rank": int(ref.rank_delta_elo)}
        for t in THRESHOLDS:
            k = f"{int(t * 100):03d}"
            row[f"n_favours_white_{k}"] = int((g.p_favours_white >= t).sum())
            row[f"n_balanced_{k}"] = int((g.p_balanced >= t).sum())
            row[f"n_favours_black_{k}"] = int((g.p_favours_black >= t).sum())
            row[f"n_at_least_as_fair_as_518_{k}"] = int((g.p_at_least_as_fair_as_518 >= t).sum())
            row[f"n_more_white_than_518_{k}"] = int((g.p_more_white_than_518 >= t).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Agreement between engines
# ---------------------------------------------------------------------------

def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    tab = pd.crosstab(a, b).reindex(index=cats, columns=cats, fill_value=0).to_numpy(float)
    n = tab.sum()
    po = np.trace(tab) / n
    pe = (tab.sum(0) * tab.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def agreement(post: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    a, b = SELFPLAY_ENGINES
    A = post[post.engine == a].set_index("position_id")
    B = post[post.engine == b].set_index("position_id").reindex(A.index)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(A), size=(N_BOOT, len(A)))
    rows = []
    for metric, col, sdcol in (("escore", "escore_mean", "escore_sd"),
                               ("delta_elo", "delta_elo_median", "delta_elo_sd")):
        x, y = A[col].to_numpy(), B[col].to_numpy()
        rho = stats.spearmanr(x, y).statistic
        boots = [stats.spearmanr(x[i], y[i]).statistic for i in idx]
        r = stats.pearsonr(x, y).statistic
        rel_a = 1 - (A[sdcol] ** 2).mean() / A[col].var()
        rel_b = 1 - (B[sdcol] ** 2).mean() / B[col].var()
        diff = y - x
        mean = (x + y) / 2
        ba = stats.linregress(mean, diff)
        rows.append({
            "metric": metric, "spearman": rho, "spearman_lo": np.percentile(boots, 2.5),
            "spearman_hi": np.percentile(boots, 97.5), "pearson": r,
            f"reliability_{a}": rel_a, f"reliability_{b}": rel_b,
            "spearman_corrected": rho / np.sqrt(rel_a * rel_b),
            "pearson_corrected": r / np.sqrt(rel_a * rel_b),
            f"median_{a}": np.median(x), f"median_{b}": np.median(y),
            f"true_sd_{a}": float(np.sqrt(max(A[col].var() - (A[sdcol] ** 2).mean(), 0))),
            f"true_sd_{b}": float(np.sqrt(max(B[col].var() - (B[sdcol] ** 2).mean(), 0))),
            "mean_difference": diff.mean(),
            "mean_difference_lo": diff.mean() - 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)),
            "mean_difference_hi": diff.mean() + 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)),
            "loa_low": diff.mean() - 1.96 * diff.std(ddof=1),
            "loa_high": diff.mean() + 1.96 * diff.std(ddof=1),
            "difference_on_mean_slope": ba.slope, "difference_on_mean_slope_se": ba.stderr,
            "difference_on_mean_p": ba.pvalue,
        })
    extra: dict[str, pd.DataFrame] = {}
    cls_rows = []
    for t in THRESHOLDS:
        k = f"category_{int(t * 100):03d}"
        tab = pd.crosstab(A[k].rename(a), B[k].rename(b))
        tab.index = [f"{a}: {i}" for i in tab.index]
        extra[f"crosstab_{int(t * 100):03d}"] = tab
        fair_a = A.p_at_least_as_fair_as_518 >= t
        fair_b = B.p_at_least_as_fair_as_518 >= t
        cls_rows.append({"threshold": t, "kappa_category": cohen_kappa(A[k], B[k]),
                         "percent_agreement": float((A[k] == B[k]).mean() * 100),
                         f"n_fair_518_{a}": int(fair_a.sum()), f"n_fair_518_{b}": int(fair_b.sum()),
                         "n_fair_518_both": int((fair_a & fair_b).sum()),
                         "kappa_fair_518": cohen_kappa(fair_a, fair_b)})
    for n in (10, 50):
        ta = set(A.nsmallest(n, "rank_delta_elo").index)
        tb = set(B.nsmallest(n, "rank_delta_elo").index)
        cls_rows.append({"threshold": np.nan, f"top{n}_overlap": len(ta & tb),
                         f"top{n}_common": " ".join(map(str, sorted(ta & tb)))})
    extra["classification"] = pd.DataFrame(cls_rows)
    return pd.DataFrame(rows), extra


# ---------------------------------------------------------------------------
# 3. Decisiveness
# ---------------------------------------------------------------------------

def weighted_logistic(x: np.ndarray, y: np.ndarray, n: np.ndarray, degree: int = 2):
    """Binomial logistic regression of proportion y on polynomial(x), with
    trials n (effective sample sizes). Returns coefficients, their SEs and
    a prediction function."""
    X = np.vander(x, degree + 1, increasing=True)

    def nll(beta):
        eta = X @ beta
        return -(n * (y * eta - np.logaddexp(0, eta))).sum()

    def grad(beta):
        p = 1 / (1 + np.exp(-(X @ beta)))
        return -X.T @ (n * (y - p))

    beta0 = np.zeros(X.shape[1])
    beta0[0] = np.log(y.mean() / (1 - y.mean()))
    res = optimize.minimize(nll, beta0, jac=grad, method="BFGS")
    p = 1 / (1 + np.exp(-(X @ res.x)))
    cov = np.linalg.inv((X * (n * p * (1 - p))[:, None]).T @ X)

    def predict(xn):
        return 1 / (1 + np.exp(-(np.vander(np.atleast_1d(xn), degree + 1, increasing=True) @ res.x)))

    return res.x, np.sqrt(np.diag(cov)), predict


def decisiveness(post: pd.DataFrame, deff: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fits, residual_rows, predictors = [], [], {}
    for engine, g in post.groupby("engine"):
        x = g.delta_elo_median.abs().to_numpy()
        y = 1 - g.draws.to_numpy() / g.games.to_numpy()
        n = g.games.to_numpy() / deff[engine]
        beta, se, predict = weighted_logistic(x, y, n)
        predictors[engine] = predict
        fitted = predict(x)
        resid = (y - fitted) / np.sqrt(fitted * (1 - fitted) / n)
        ref = g[g.position_id == STANDARD_POSITION_ID].iloc[0]
        fits.append({"engine": engine, "model": "logit(decisive) ~ |Delta_Elo| + |Delta_Elo|^2",
                     "b0": beta[0], "b0_se": se[0], "b1": beta[1], "b1_se": se[1],
                     "b2": beta[2], "b2_se": se[2],
                     "decisive_at_0": float(predict(0.0)[0]), "decisive_at_10": float(predict(10.0)[0]),
                     "decisive_at_25": float(predict(25.0)[0]), "decisive_at_50": float(predict(50.0)[0]),
                     "decisive_at_100": float(predict(100.0)[0]),
                     "decisive_median": float(np.median(y)), "decisive_min": float(y.min()),
                     "decisive_max": float(y.max()), "pos518_decisive": ref.decisive_mean,
                     "pos518_decisive_lo": ref.decisive_lo, "pos518_decisive_hi": ref.decisive_hi,
                     "pos518_decisive_rank": int((y > 1 - ref.draws / ref.games).sum() + 1),
                     "spearman_decisive_abs_delta": stats.spearmanr(x, y).statistic})
        residual_rows.append(pd.DataFrame({"engine": engine, "position_id": g.position_id.to_numpy(),
                                           "decisive_observed": y, "decisive_fitted": fitted,
                                           "decisive_residual_z": resid}))
    return pd.DataFrame(fits), pd.concat(residual_rows), predictors


def position_table(post: pd.DataFrame, resid: pd.DataFrame) -> pd.DataFrame:
    """Every position with both engines' estimates side by side and its
    rankings: advantage for White (1 = largest), fairness (1 = smallest
    |Delta_Elo|), decisiveness (1 = most decisive), and fair-and-decisive
    (1 = highest probability of being both at least as fair as and more
    decisive than #518). No thresholds: positions are ranked, not selected.
    The cross-engine rank uses the mean of the two engines' joint
    probabilities."""
    from utils import get_position
    m = post.merge(resid[["engine", "position_id", "decisive_residual_z"]],
                   on=["engine", "position_id"])
    keep = ["escore_mean", "escore_lo", "escore_hi", "delta_elo_median", "delta_elo_lo",
            "delta_elo_hi", "p_favours_white", "p_balanced", "p_favours_black",
            "decisive_mean", "decisive_lo", "decisive_hi", "white_share_decisive_mean",
            "p_at_least_as_fair_as_518", "p_more_white_than_518", "p_more_decisive_than_518",
            "p_fairer_and_more_decisive_than_518", "decisive_residual_z"]
    parts = []
    for engine, g in m.groupby("engine"):
        g = g.set_index("position_id")[keep].copy()
        g["rank_advantage"] = g.delta_elo_median.rank(ascending=False, method="min").astype(int)
        g["rank_fairness"] = g.delta_elo_median.abs().rank(method="min").astype(int)
        g["rank_decisive"] = g.decisive_mean.rank(ascending=False, method="min").astype(int)
        g["rank_fair_decisive"] = g.p_fairer_and_more_decisive_than_518.rank(
            ascending=False, method="min").astype(int)
        g.columns = [f"{engine}_{c}" for c in g.columns]
        parts.append(g)
    out = pd.concat(parts, axis=1).reset_index()
    out.insert(1, "arrangement", out.position_id.map(lambda i: get_position(i).arrangement))
    jp = out[[f"{e}_p_fairer_and_more_decisive_than_518" for e in SELFPLAY_ENGINES]].mean(axis=1)
    out["mean_p_fairer_and_more_decisive_than_518"] = jp
    out["rank_fair_decisive_both"] = jp.rank(ascending=False, method="min").astype(int)
    return out.sort_values("rank_fair_decisive_both").reset_index(drop=True)


def endings_by_advantage(games: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    def kind(reason: str) -> str:
        r = reason.lower()
        if "mates" in r:
            return "checkmate"
        if "wins by adjudication" in r:
            return "resignation (adjudicated win)"
        if "draw by adjudication" in r:
            return "adjudicated draw"
        if "repetition" in r:
            return "repetition"
        if "insufficient" in r:
            return "insufficient material"
        if "fifty" in r:
            return "fifty-move rule"
        if "stalemate" in r:
            return "stalemate"
        return "other"

    g = games[["engine", "position_id", "end_reason", "plies"]].copy()
    g["ending"] = g.end_reason.map(kind)
    adv = post[["engine", "position_id", "delta_elo_median"]].copy()
    adv["abs_delta_elo_band"] = pd.cut(adv.delta_elo_median.abs(), [0, 5, 10, 20, 40, 80, np.inf],
                                       include_lowest=True,
                                       labels=["0-5", "5-10", "10-20", "20-40", "40-80", "80+"])
    g = g.merge(adv, on=["engine", "position_id"])
    tab = (g.groupby(["engine", "abs_delta_elo_band"], observed=True).ending
           .value_counts(normalize=True).unstack(fill_value=0) * 100)
    plies = g.groupby(["engine", "abs_delta_elo_band"], observed=True).plies.mean().rename("mean_plies")
    npos = (adv.groupby(["engine", "abs_delta_elo_band"], observed=True).size().rename("positions"))
    return tab.join(plies).join(npos).reset_index()


def decisive_agreement(post: pd.DataFrame) -> dict:
    a, b = SELFPLAY_ENGINES
    A = post[post.engine == a].set_index("position_id")
    B = post[post.engine == b].set_index("position_id").reindex(A.index)
    return {"spearman_decisive": stats.spearmanr(A.decisive_mean, B.decisive_mean).statistic,
            "pearson_decisive": stats.pearsonr(A.decisive_mean, B.decisive_mean).statistic}


# ---------------------------------------------------------------------------
# Figures (one plot per file, no legends; style from plot_style.py)
# ---------------------------------------------------------------------------

FIG = (3.5, 2.8)


def finish(fig, ax, name: str, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    bold_tick_labels(ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for p in paths:
        fig.savefig(p, metadata={"CreationDate": None} if p.suffix == ".pdf" else None)
    plt.close(fig)
    return paths


def band(metric: str) -> tuple[float, float]:
    if metric == "escore":
        return (1 / (1 + 10 ** (BALANCED_ELO / 400)), 1 / (1 + 10 ** (-BALANCED_ELO / 400)))
    return (-BALANCED_ELO, BALANCED_ELO)


def cols(metric: str) -> tuple[str, str, str]:
    return (("escore_mean", "escore_lo", "escore_hi") if metric == "escore"
            else ("delta_elo_median", "delta_elo_lo", "delta_elo_hi"))


def fig_vs_evaluation(post, top, sp: str, metric: str, out_dir: Path) -> list[Path]:
    """Self-play outcome against the same engine's top-budget evaluation."""
    plt = apply_style()
    m = METRICS[metric]
    c, lo, hi = cols(metric)
    y = post[post.engine == sp].set_index("position_id")
    x = top[top.engine == sp].set_index("position_id").reindex(y.index)[metric]
    s = ENGINES[sp]
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    lim = (min(x.min(), y[lo].min()), max(x.max(), y[hi].max()))
    pad = 0.03 * (lim[1] - lim[0])
    lim = (lim[0] - pad, lim[1] + pad)
    ax.plot(lim, lim, color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.vlines(x, y[lo], y[hi], color=s.colour, alpha=0.12, lw=0.6, zorder=1)
    ax.scatter(x, y[c], s=8, color=s.colour, alpha=0.55, lw=0, marker=s.marker, zorder=2)
    ax.scatter(x[STANDARD_POSITION_ID], y.loc[STANDARD_POSITION_ID, c], s=46, facecolor="none",
               edgecolor="#111111", lw=1.2, zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Evaluation {m.bold_symbol}")
    ax.set_ylabel(f"Self-play {m.bold_symbol}")
    return finish(fig, ax, f"selfplay_balance_vs_evaluation_{sp}_{metric}", out_dir)


def fig_ranked(post, metric: str, out_dir: Path) -> list[Path]:
    """Posterior estimate against rank with 95% interval band; #518 marked."""
    plt = apply_style()
    m = METRICS[metric]
    c, lo, hi = cols(metric)
    fig, ax = plt.subplots(figsize=FIG)
    b0, b1 = band(metric)
    ax.axhspan(b0, b1, color=BAND_COLOUR, lw=0, zorder=0)
    for sp in SELFPLAY_ENGINES:
        s = ENGINES[sp]
        g = post[post.engine == sp].sort_values(c, ascending=False).reset_index(drop=True)
        rank = np.arange(1, len(g) + 1)
        ax.fill_between(rank, g[lo], g[hi], color=s.colour, alpha=0.15, lw=0)
        ax.plot(rank, g[c], color=s.colour, ls=s.linestyle)
        k = g.index[g.position_id == STANDARD_POSITION_ID][0]
        ax.plot(k + 1, g.loc[k, c], marker=s.marker, ms=6, color=s.colour, mec="white", mew=1.0,
                ls="none", zorder=4)
    ax.set_xlim(0, 961)
    ax.set_xlabel("Rank (1 = largest advantage for White)")
    ax.set_ylabel(m.axis_label)
    return finish(fig, ax, f"selfplay_balance_ranked_{metric}", out_dir)


def fig_prob_white(post, out_dir: Path) -> list[Path]:
    """P(favours White) against posterior Delta_Elo for both engines."""
    plt = apply_style()
    fig, ax = plt.subplots(figsize=FIG)
    ax.axvspan(-BALANCED_ELO, BALANCED_ELO, color=BAND_COLOUR, lw=0, zorder=0)
    for sp in SELFPLAY_ENGINES:
        s = ENGINES[sp]
        g = post[post.engine == sp]
        ax.scatter(g.delta_elo_median, g.p_favours_white, s=7, color=s.colour, alpha=0.5, lw=0,
                   marker=s.marker)
        r = g[g.position_id == STANDARD_POSITION_ID].iloc[0]
        ax.scatter(r.delta_elo_median, r.p_favours_white, s=46, facecolor="none",
                   edgecolor=s.colour, lw=1.3, zorder=3)
    for t in THRESHOLDS:
        ax.axhline(t, color=MUTED_TEXT, lw=0.7, ls=(0, (1, 2)))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(METRICS["delta_elo"].axis_label)
    ax.set_ylabel("P(favours White)")
    return finish(fig, ax, "selfplay_balance_prob_white", out_dir)


def fig_agreement_scatter(post, metric: str, out_dir: Path) -> list[Path]:
    plt = apply_style()
    m = METRICS[metric]
    c, lo, hi = cols(metric)
    a, b = SELFPLAY_ENGINES
    A = post[post.engine == a].set_index("position_id")
    B = post[post.engine == b].set_index("position_id").reindex(A.index)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    lim = (min(A[lo].min(), B[lo].min()), max(A[hi].max(), B[hi].max()))
    ax.plot(lim, lim, color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.errorbar(A[c], B[c], xerr=[A[c] - A[lo], A[hi] - A[c]], yerr=[B[c] - B[lo], B[hi] - B[c]],
                fmt="none", ecolor="#222222", alpha=0.07, lw=0.5, zorder=1)
    ax.scatter(A[c], B[c], s=8, color="#222222", alpha=0.45, lw=0, zorder=2)
    ax.scatter(A.loc[STANDARD_POSITION_ID, c], B.loc[STANDARD_POSITION_ID, c], s=46,
               facecolor="none", edgecolor="#111111", lw=1.2, zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{ENGINES[a].label} self-play {m.bold_symbol}")
    ax.set_ylabel(f"{ENGINES[b].label} self-play {m.bold_symbol}")
    return finish(fig, ax, f"selfplay_agreement_scatter_{metric}", out_dir)


def fig_bland_altman(post, agree, metric: str, out_dir: Path) -> list[Path]:
    plt = apply_style()
    m = METRICS[metric]
    c, _, _ = cols(metric)
    a, b = SELFPLAY_ENGINES
    A = post[post.engine == a].set_index("position_id")
    B = post[post.engine == b].set_index("position_id").reindex(A.index)
    mean, diff = (A[c] + B[c]) / 2, B[c] - A[c]
    r = agree[agree.metric == metric].iloc[0]
    fig, ax = plt.subplots(figsize=FIG)
    ax.scatter(mean, diff, s=7, color="#222222", alpha=0.4, lw=0)
    ax.axhline(0, color=MUTED_TEXT, lw=0.7)
    ax.axhline(r.mean_difference, color="#111111", lw=1.0)
    for v in (r.loa_low, r.loa_high):
        ax.axhline(v, color="#111111", lw=0.8, ls=(0, (3, 2)))
    ax.scatter(mean[STANDARD_POSITION_ID], diff[STANDARD_POSITION_ID], s=46, facecolor="none",
               edgecolor="#111111", lw=1.2, zorder=3)
    ax.set_xlabel(f"Mean of engines' {m.bold_symbol}")
    ax.set_ylabel(f"{ENGINES[b].label} − {ENGINES[a].label} {m.bold_symbol}")
    return finish(fig, ax, f"selfplay_agreement_bland_altman_{metric}", out_dir)


def fig_decisive(post, predictors, fair: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Decisive rate against |Delta_Elo| with the fitted curve per engine."""
    plt = apply_style()
    fig, ax = plt.subplots(figsize=FIG)
    for sp in SELFPLAY_ENGINES:
        s = ENGINES[sp]
        g = post[post.engine == sp]
        x = g.delta_elo_median.abs()
        y = 1 - g.draws / g.games
        ax.scatter(x, y * 100, s=7, color=s.colour, alpha=0.4, lw=0, marker=s.marker)
        grid = np.linspace(0, x.max(), 300)
        ax.plot(grid, predictors[sp](grid) * 100, color=s.colour, ls=s.linestyle, lw=1.6)
        r = g[g.position_id == STANDARD_POSITION_ID].iloc[0]
        ax.scatter(abs(r.delta_elo_median), (1 - r.draws / r.games) * 100, s=46, facecolor="none",
                   edgecolor=s.colour, lw=1.3, zorder=3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"|{METRICS['delta_elo'].bold_symbol}|")
    ax.set_ylabel("Decisive games (%)")
    return finish(fig, ax, "selfplay_decisive_vs_advantage", out_dir)


def fig_white_share(post, out_dir: Path) -> list[Path]:
    """White's share of decisive games against Delta_Elo."""
    plt = apply_style()
    fig, ax = plt.subplots(figsize=FIG)
    ax.axhline(50, color=MUTED_TEXT, lw=0.7, ls=(0, (1, 2)))
    for sp in SELFPLAY_ENGINES:
        s = ENGINES[sp]
        g = post[post.engine == sp]
        ax.scatter(g.delta_elo_median, g.white_share_decisive_mean * 100, s=7, color=s.colour,
                   alpha=0.4, lw=0, marker=s.marker)
    ax.set_ylim(0, 100)
    ax.set_xlabel(METRICS["delta_elo"].axis_label)
    ax.set_ylabel("White's share of decisive games (%)")
    return finish(fig, ax, "selfplay_decisive_white_share", out_dir)


# Ternary (win/draw/loss) plots. Corners: Black wins (0, 0), White wins (1, 0),
# draw (1/2, sqrt(3)/2). A point (p_W, p_D, p_L) maps to x = p_W + p_D/2, which
# is exactly White's expected score, and y = (sqrt(3)/2) p_D, so lines of equal
# advantage are vertical and height is the draw rate.
H = np.sqrt(3) / 2
TERNARY_POSITIONS = [868, STANDARD_POSITION_ID, 674, 226]   # unbalanced, standard,
#   fair and decisive, fair and drawish


def to_xy(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return p[:, 0] + p[:, 1] / 2, H * p[:, 1]


# The four-colour palette from plot_style.py, here identifying positions.
TERNARY_COLOURS = {868: ENGINES["stockfish"].colour, STANDARD_POSITION_ID: ENGINES["caissa"].colour,
                   674: ENGINES["lc0"].colour, 226: ENGINES["plentychess"].colour}
ZOOM = (0.45, 0.72, H * 0.55, H * 1.0)    # x (E_score) and y (draw rate 55-100%)


def draw_rate_lines(ax) -> None:
    """Horizontal lines of constant draw rate (20%, 40%, ...), the only grid."""
    for v in np.arange(0.2, 1.0, 0.2):
        ax.plot([v / 2, 1 - v / 2], [H * v, H * v], color="#e3e3e3", lw=0.6, zorder=0)


def balanced_strip(ax) -> None:
    lo, hi = band("escore")
    # The strip clipped to the triangle: at x the triangle's top is H(1 - |2x - 1|).
    ax.fill([lo, hi, hi, lo], [0, 0, H * (1 - 2 * (hi - 0.5)), H * (1 - 2 * (0.5 - lo))],
            color=BAND_COLOUR, lw=0, zorder=0)


def hpd_levels(kde, sample: np.ndarray, masses=(0.5, 0.95)) -> list[float]:
    dens = kde(sample)
    return [float(np.quantile(dens, 1 - m)) for m in masses]


def ternary_overview_panel(ax, post, sp: str) -> None:
    draw_rate_lines(ax)
    balanced_strip(ax)
    g = post[post.engine == sp]
    n = g[["white_wins", "draws", "black_wins"]].to_numpy(float) + PRIOR
    x, y = to_xy(n / n.sum(axis=1, keepdims=True))
    ax.scatter(x, y, s=4, color="#111111", alpha=0.45, lw=0, zorder=2)
    x0, x1, y0, y1 = ZOOM
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#111111", lw=0.7,
            ls=(0, (2, 2)), zorder=4)
    ax.plot([0, 1, 0.5, 0], [0, 0, H, 0], color="#111111", lw=0.9, zorder=3)
    kw = dict(fontsize=8, fontweight="bold", color="#111111")
    ax.text(0.5, H + 0.03, "Draw", ha="center", va="bottom", **kw)
    ax.text(0.0, -0.04, "Black wins", ha="center", va="top", **kw)
    ax.text(1.0, -0.04, "White wins", ha="center", va="top", **kw)
    ax.set_title(ENGINES[sp].label, fontsize=9, fontweight="bold", pad=14)
    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(-0.14, H + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")


def ternary_zoom_panel(ax, post, deff: dict[str, float], sp: str) -> None:
    from matplotlib.patches import Polygon
    x0, x1, y0, y1 = ZOOM
    draw_rate_lines(ax)
    lo, hi = band("escore")
    ax.axvspan(lo, hi, color=BAND_COLOUR, lw=0, zorder=0)
    tri = Polygon([[0, 0], [1, 0], [0.5, H]], closed=True, transform=ax.transData)
    ax.plot([0, 1, 0.5, 0], [0, 0, H, 0], color="#111111", lw=0.9, zorder=3)
    gx, gy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    g = post[post.engine == sp].set_index("position_id")
    for pos in TERNARY_POSITIONS:
        colour = TERNARY_COLOURS[pos]
        r = g.loc[pos]
        draws = posterior_draws(r.white_wins, r.draws, r.black_wins, deff[sp],
                                rng_for(sp, int(pos)), n=N_DRAWS)
        sub = np.vstack(to_xy(draws))[:, ::4]
        kde = stats.gaussian_kde(sub)
        z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        l50, l95 = hpd_levels(kde, sub)
        layers = [ax.contourf(gx, gy, z, levels=[l95, z.max() + 1], colors=[colour], alpha=0.22,
                              zorder=2),
                  ax.contour(gx, gy, z, levels=[l95], colors=[colour], linewidths=0.8, zorder=2),
                  ax.contour(gx, gy, z, levels=[l50], colors=[colour], linewidths=1.5, zorder=2)]
        for layer in layers:
            layer.set_clip_path(tri)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel(METRICS["escore"].axis_label)
    yt = np.arange(0.6, 1.01, 0.1)
    ax.set_yticks(H * yt, [f"{v * 100:.0f}" for v in yt])
    ax.set_ylabel("Draws (%)")
    bold_tick_labels(ax)


def fig_ternary(post, deff: dict[str, float], out_dir: Path) -> list[Path]:
    """2 x 2 panel: top row, every position's posterior mean in the full
    win/draw/loss triangle (Stockfish left, Caissa right); bottom row, the
    50% and 95% highest-density regions of the posterior for four contrasting
    positions in the draw-heavy corner (dashed box above). Position colours:
    TERNARY_COLOURS (named in the caption)."""
    plt = apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6),
                             gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.18,
                                          "wspace": 0.25})
    for j, sp in enumerate(SELFPLAY_ENGINES):
        ternary_overview_panel(axes[0, j], post, sp)
        ternary_zoom_panel(axes[1, j], post, deff, sp)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "selfplay_ternary.pdf", out_dir / "selfplay_ternary.png"]
    for path in paths:
        fig.savefig(path, metadata={"CreationDate": None} if path.suffix == ".pdf" else None)
    plt.close(fig)
    return paths


def save_fig(fig, name: str, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for path in paths:
        fig.savefig(path, metadata={"CreationDate": None} if path.suffix == ".pdf" else None)
    plt.close(fig)
    return paths


def fig_vs_evaluation_panel(post, top, reg, out_dir: Path) -> list[Path]:
    """Self-play Delta_Elo against the same engine's top-budget evaluation,
    Stockfish and Caissa side by side on common axes, in the style of the
    starting-position scatter matrix: one point per position, dotted line of
    equality, solid least-squares line, slope (SE) and R² in the corner."""
    plt = apply_style()
    from matplotlib.ticker import MultipleLocator
    xs, ys = {}, {}
    for sp in SELFPLAY_ENGINES:
        y = post[post.engine == sp].set_index("position_id")["delta_elo_median"]
        x = top[top.engine == sp].set_index("position_id").reindex(y.index)["delta_elo"]
        xs[sp], ys[sp] = x, y
    allv = pd.concat(list(xs.values()) + list(ys.values()))
    lo, hi = float(allv.min()) - 8, float(allv.max()) + 8
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.1})
    for ax, sp in zip(axes, SELFPLAY_ENGINES):
        x, y = xs[sp], ys[sp]
        r = reg[(reg.selfplay_engine == sp) & (reg.evaluation_engine == sp)
                & (reg.metric == "delta_elo")].iloc[0]
        ax.plot([lo, hi], [lo, hi], color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)), zorder=1)
        ax.scatter(x, y, s=7, color="#222222", alpha=0.35, lw=0, zorder=2)
        grid = np.array([lo, hi])
        ax.plot(grid, r.intercept + r.slope * grid, color="#111111", lw=1.1, zorder=3)
        ax.text(0.05, 0.95, f"Slope (SE) = {r.slope:.2f} ({r.slope_se:.2f})\nR$^2$ = {r.r2:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5, fontweight="bold",
                linespacing=1.35)
        ax.set_title(ENGINES[sp].label, fontsize=9, fontweight="bold")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.set_xlabel(f"Starting Position Evaluation {METRICS['delta_elo'].bold_symbol}")
        bold_tick_labels(ax)
    axes[0].set_ylabel(f"Self-play {METRICS['delta_elo'].bold_symbol}")
    return save_fig(fig, "selfplay_balance_vs_evaluation", out_dir)


def fig_ranked_panel(post, out_dir: Path) -> list[Path]:
    """Positions ranked by posterior Delta_Elo with each position's 95%
    credible interval as a vertical line, Stockfish and Caissa side by side
    on a common scale; practically balanced band shaded; #518 marked."""
    plt = apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, gridspec_kw={"wspace": 0.08})
    b0, b1 = band("delta_elo")
    for ax, sp in zip(axes, SELFPLAY_ENGINES):
        s = ENGINES[sp]
        g = post[post.engine == sp].sort_values("delta_elo_median", ascending=False).reset_index(drop=True)
        rank = np.arange(1, len(g) + 1)
        ax.axhspan(b0, b1, color=BAND_COLOUR, lw=0, zorder=0)
        ax.vlines(rank, g.delta_elo_lo, g.delta_elo_hi, color=s.colour, alpha=0.25, lw=0.6, zorder=1)
        ax.plot(rank, g.delta_elo_median, color=s.colour, lw=1.4, zorder=2)
        k = int(g.index[g.position_id == STANDARD_POSITION_ID][0])
        ax.vlines(k + 1, g.loc[k, "delta_elo_lo"], g.loc[k, "delta_elo_hi"], color="#111111",
                  lw=1.2, zorder=3)
        ax.plot(k + 1, g.loc[k, "delta_elo_median"], marker=s.marker, ms=6, color="#111111",
                mec="white", mew=0.9, ls="none", zorder=4)
        ax.set_title(s.label, fontsize=9, fontweight="bold")
        ax.set_xlim(0, len(g) + 1)
        ax.set_xlabel("Position Rankings")
        bold_tick_labels(ax)
    axes[0].set_ylabel(METRICS["delta_elo"].axis_label)
    return save_fig(fig, "selfplay_balance_ranked", out_dir)


def fig_agreement_panel(post, agree, out_dir: Path) -> list[Path]:
    """Agreement between Stockfish and Caissa self-play Delta_Elo (posterior
    medians): left, Caissa against Stockfish with the line of equality and
    Spearman's rho [95% CI]; right, Bland-Altman plot of the difference
    (Caissa - Stockfish) against the mean, with the mean difference, 95%
    limits of agreement and the fitted trend."""
    plt = apply_style()
    from matplotlib.ticker import MultipleLocator
    a, b = SELFPLAY_ENGINES
    A = post[post.engine == a].set_index("position_id")["delta_elo_median"]
    B = post[post.engine == b].set_index("position_id")["delta_elo_median"].reindex(A.index)
    r = agree[agree.metric == "delta_elo"].iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5), gridspec_kw={"wspace": 0.3})
    lo = float(min(A.min(), B.min())) - 8
    hi = float(max(A.max(), B.max())) + 8
    ax1.plot([lo, hi], [lo, hi], color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax1.scatter(A, B, s=7, color="#222222", alpha=0.35, lw=0, zorder=2)
    ax1.text(0.05, 0.95, f"ρ = {r.spearman:.2f}\n[{r.spearman_lo:.2f}, {r.spearman_hi:.2f}]",
             transform=ax1.transAxes, ha="left", va="top", fontsize=7.5, fontweight="bold",
             linespacing=1.35)
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)            # same range on both axes, so a square box keeps 1:1
    ax1.xaxis.set_major_locator(MultipleLocator(25))
    ax1.yaxis.set_major_locator(MultipleLocator(25))
    ax1.set_xlabel(f"{ENGINES[a].label} {METRICS['delta_elo'].bold_symbol}")
    ax1.set_ylabel(f"{ENGINES[b].label} {METRICS['delta_elo'].bold_symbol}")
    mean, diff = (A + B) / 2, B - A
    ax2.scatter(mean, diff, s=7, color="#222222", alpha=0.35, lw=0, zorder=2)
    ax2.axhline(0, color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ba_colour = ENGINES["plentychess"].colour        # magenta from the shared palette
    ax2.axhline(r.mean_difference, color=ba_colour, lw=1.3, zorder=3)
    for v in (r.loa_low, r.loa_high):
        ax2.axhline(v, color=ba_colour, lw=1.0, ls=(0, (4, 2)), zorder=3)
    grid = np.linspace(mean.min(), mean.max(), 50)
    fit = stats.linregress(mean, diff)
    ax2.plot(grid, fit.intercept + fit.slope * grid, color="#111111", lw=0.9, ls=(0, (1, 1.5)),
             zorder=3)
    ax2.text(0.97, 0.95, (f"Mean difference = {r.mean_difference:+.1f}\n"
             f"95% LoA = {r.loa_low:+.1f} to {r.loa_high:+.1f}").replace("-", "−"),
             transform=ax2.transAxes, ha="right", va="top", fontsize=7.5, fontweight="bold",
             linespacing=1.35)
    ax2.set_xlabel(f"Mean {METRICS['delta_elo'].bold_symbol}")
    ax2.set_ylabel(f"{ENGINES[b].label} − {ENGINES[a].label} {METRICS['delta_elo'].bold_symbol}")
    for ax in (ax1, ax2):
        ax.set_box_aspect(1)        # identical square panels, aligned x-axes
        bold_tick_labels(ax)
    return save_fig(fig, "selfplay_agreement", out_dir)


def all_figures(post, top, agree, predictors, fair, out_dir: Path, deff=None,
                reg=None) -> list[Path]:
    paths: list[Path] = []
    paths += fig_ranked_panel(post, out_dir)
    paths += fig_agreement_panel(post, agree, out_dir)
    if reg is not None:
        paths += fig_vs_evaluation_panel(post, top, reg, out_dir)
    for metric in METRICS:
        for sp in SELFPLAY_ENGINES:
            paths += fig_vs_evaluation(post, top, sp, metric, out_dir)
        paths += fig_ranked(post, metric, out_dir)
        paths += fig_agreement_scatter(post, metric, out_dir)
        paths += fig_bland_altman(post, agree, metric, out_dir)
    paths += fig_prob_white(post, out_dir)
    paths += fig_decisive(post, predictors, fair, out_dir)
    paths += fig_white_share(post, out_dir)
    if deff is not None:
        paths += fig_ternary(post, deff, out_dir)
    return paths


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

def summary_text(qc, reg, bal, agree, cls, dfit, fair, dagree, bb) -> str:
    E, D = LATEX["escore"], LATEX["delta_elo"]
    lab = {k: s.label for k, s in ENGINES.items()}
    L = ["Self-play (selfplay): key numbers", "=" * 33, "", "0. Quality checks and design effect"]
    for r in qc.itertuples():
        L.append(f"  {lab[r.engine]}: {r.positions} positions, {r.games} games "
                 f"({r.games_per_position_min}-{r.games_per_position_max} per position), unmatched "
                 f"lines {r.unmatched_lines}, losses on time {r.losses_on_time}; W/D/L "
                 f"{100 * r.white_win_rate:.1f}/{100 * r.draw_rate:.1f}/{100 * r.black_win_rate:.1f}%; "
                 f"copies as White {r.copy_a_white_score:.4f} vs {r.copy_b_white_score:.4f}; "
                 f"ICC {r.icc:.3f}, DEFF {r.deff:.2f} (book-adjusted {r.deff_book_adjusted:.2f}); "
                 f"line score on book drift slope {r.book_shift_slope:.2f} (r {r.book_shift_r:.2f}); "
                 f"max |position mean drift| {r.max_abs_position_mean_shift:.4f}")
    L += ["", "1a. Self-play against top-budget evaluations (Pearson r [corrected for self-play "
          "noise], slope)"]
    for r in reg.itertuples():
        L.append(f"  {lab[r.selfplay_engine]} games ~ {lab[r.evaluation_engine]} evaluation "
                 f"({r.metric}): r {r.pearson:.2f} [{r.pearson_corrected_for_selfplay_noise:.2f}], "
                 f"rho {r.spearman:.2f}, slope {r.slope:.3f} (SE {r.slope_se:.3f}), intercept "
                 f"{r.intercept:+.3f}, R2 {r.r2:.2f}")
    L += ["", "1b. Posterior balance"]
    for r in bal.itertuples():
        L.append(f"  {lab[r.engine]}: {E} median {r.escore_median:.4f} (IQR {r.escore_q25:.4f} to "
                 f"{r.escore_q75:.4f}); {D} median {r.delta_elo_median:+.1f} (IQR {r.delta_elo_q25:+.1f} "
                 f"to {r.delta_elo_q75:+.1f}; range {r.delta_elo_min:+.1f} to {r.delta_elo_max:+.1f}; "
                 f"true SD {r.delta_elo_true_sd:.1f}); mean 95% interval width "
                 f"{r.mean_interval_width_delta_elo:.1f}")
        L.append(f"    P >= 0.80: favours White {r.n_favours_white_080}, balanced {r.n_balanced_080}, "
                 f"favours Black {r.n_favours_black_080}, at least as fair as #518 "
                 f"{r.n_at_least_as_fair_as_518_080}, more favourable to White than #518 "
                 f"{r.n_more_white_than_518_080}")
        L.append(f"    P >= 0.95: favours White {r.n_favours_white_095}, balanced {r.n_balanced_095}, "
                 f"favours Black {r.n_favours_black_095}, at least as fair as #518 "
                 f"{r.n_at_least_as_fair_as_518_095}, more favourable to White than #518 "
                 f"{r.n_more_white_than_518_095}")
        L.append(f"    expected counts (sum of probabilities): favours White "
                 f"{r.expected_n_favours_white:.0f}, balanced {r.expected_n_balanced:.0f}, favours "
                 f"Black {r.expected_n_favours_black:.0f}, at least as fair as #518 "
                 f"{r.expected_n_at_least_as_fair_as_518:.0f}")
        L.append(f"    #518: {E} {r.pos518_escore:.4f} [{r.pos518_escore_lo:.4f}, {r.pos518_escore_hi:.4f}], "
                 f"{D} {r.pos518_delta_elo:+.1f} [{r.pos518_delta_elo_lo:+.1f}, {r.pos518_delta_elo_hi:+.1f}], "
                 f"P(favours White) {r.pos518_p_favours_white:.2f}, rank {r.pos518_rank}/960; most "
                 f"unbalanced #{r.most_unbalanced} ({D} {r.most_unbalanced_delta_elo:+.1f})")
    L += ["", "2. Agreement between Stockfish and Caissa games"]
    for r in agree.itertuples():
        L.append(f"  {r.metric}: Spearman {r.spearman:.2f} [{r.spearman_lo:.2f}, {r.spearman_hi:.2f}], "
                 f"corrected for noise {r.spearman_corrected:.2f} (reliabilities "
                 f"{r.reliability_stockfish:.2f}, {r.reliability_caissa:.2f}); mean difference "
                 f"(Caissa - Stockfish) {r.mean_difference:+.4g} [{r.mean_difference_lo:+.4g}, "
                 f"{r.mean_difference_hi:+.4g}], limits of agreement {r.loa_low:+.4g} to {r.loa_high:+.4g}; "
                 f"difference-on-mean slope {r.difference_on_mean_slope:+.3f} (p {r.difference_on_mean_p:.2g}); "
                 f"medians {r.median_stockfish:.4g} vs {r.median_caissa:.4g}; true SD "
                 f"{r.true_sd_stockfish:.4g} vs {r.true_sd_caissa:.4g}")
    for r in cls.itertuples():
        if not np.isnan(r.threshold):
            L.append(f"  categories at P >= {r.threshold:.2f}: kappa {r.kappa_category:.2f}, agreement "
                     f"{r.percent_agreement:.0f}%; at least as fair as #518: Stockfish "
                     f"{r.n_fair_518_stockfish:.0f}, Caissa {r.n_fair_518_caissa:.0f}, both "
                     f"{r.n_fair_518_both:.0f} (kappa {r.kappa_fair_518:.2f})")
    for n in (10, 50):
        row = cls[cls.get(f"top{n}_overlap").notna()] if f"top{n}_overlap" in cls else None
        if row is not None and len(row):
            rr = row.iloc[0]
            L.append(f"  top {n} most unbalanced in both: {int(rr[f'top{n}_overlap'])} "
                     f"({rr[f'top{n}_common']})")
    L += ["", "3. Decisiveness"]
    for r in dfit.itertuples():
        L.append(f"  {lab[r.engine]}: decisive median {100 * r.decisive_median:.1f}% (range "
                 f"{100 * r.decisive_min:.1f}-{100 * r.decisive_max:.1f}%); fitted at |{D}| = 0/10/25/50/100: "
                 f"{100 * r.decisive_at_0:.1f}/{100 * r.decisive_at_10:.1f}/{100 * r.decisive_at_25:.1f}/"
                 f"{100 * r.decisive_at_50:.1f}/{100 * r.decisive_at_100:.1f}%; Spearman with |{D}| "
                 f"{r.spearman_decisive_abs_delta:.2f}; #518 {100 * r.pos518_decisive:.1f}% "
                 f"[{100 * r.pos518_decisive_lo:.1f}, {100 * r.pos518_decisive_hi:.1f}], rank "
                 f"{r.pos518_decisive_rank}/960 for decisiveness")
    L.append(f"  agreement between engines on decisive rate: Spearman {dagree['spearman_decisive']:.2f}")
    L.append("  fair and decisive relative to #518, ranked by P(at least as fair AND more decisive "
             "than #518), mean of both engines (top 15):")
    for r in fair.head(15).itertuples():
        L.append(f"    {r.rank_fair_decisive_both:>3}. #{r.position_id} {r.arrangement}: mean P "
                 f"{r.mean_p_fairer_and_more_decisive_than_518:.2f} | Stockfish {D} "
                 f"{r.stockfish_delta_elo_median:+.1f}, decisive {100 * r.stockfish_decisive_mean:.1f}% "
                 f"(P {r.stockfish_p_fairer_and_more_decisive_than_518:.2f}) | Caissa {D} "
                 f"{r.caissa_delta_elo_median:+.1f}, decisive {100 * r.caissa_decisive_mean:.1f}% "
                 f"(P {r.caissa_p_fairer_and_more_decisive_than_518:.2f})")
    L += ["", "Sensitivity: Bayesian bootstrap over lines vs design-effect Dirichlet (ratio of "
          f"{E} posterior SDs)"]
    for engine, g in bb.groupby("engine"):
        L.append(f"  {lab[engine]}: median ratio {g.sd_ratio_bootstrap_to_dirichlet.median():.2f} "
                 f"(IQR {g.sd_ratio_bootstrap_to_dirichlet.quantile(.25):.2f}-"
                 f"{g.sd_ratio_bootstrap_to_dirichlet.quantile(.75):.2f})")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tables", default=str(TABLES), help="selfplay collect tables folder")
    p.add_argument("--tables-dir", default=str(OUT_TABLES))
    p.add_argument("--figures-dir", default=str(OUT_FIGURES))
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    games = load_games(Path(args.tables))
    book = pd.read_parquet(BOOK, columns=["position_id", "line_index", "shift_expected_check"])
    top = top_budget(load_evaluations())

    qc, deff = quality_checks(games, book)
    counts = position_counts(games)
    post = posteriors(counts, deff)
    bb = bayesian_bootstrap(line_counts(games), post)
    reg = balance_regression(post, top)
    bal = balance_summary(post)
    agree, extra = agreement(post)
    dfit, resid, predictors = decisiveness(post, deff)
    fair = position_table(post, resid)
    endings = endings_by_advantage(games, post)
    dagree = decisive_agreement(post)

    out = Path(args.tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "selfplay_quality_checks.csv": qc,
        "selfplay_posteriors.csv": post.merge(resid, on=["engine", "position_id"]),
        "selfplay_balance_regression.csv": reg,
        "selfplay_balance_summary.csv": bal,
        "selfplay_agreement.csv": agree,
        "selfplay_agreement_classification.csv": extra["classification"],
        "selfplay_decisive_fit.csv": dfit,
        "selfplay_position_table.csv": fair,
        "selfplay_endings_by_advantage.csv": endings,
        "selfplay_sensitivity_bootstrap.csv": bb,
    }
    for name, t in tables.items():
        t.to_csv(out / name, index=False, float_format="%.6g")
    for k, v in extra.items():
        if k.startswith("crosstab"):
            v.to_csv(out / f"selfplay_agreement_{k}.csv")
    text = summary_text(qc, reg, bal, agree, extra["classification"], dfit, fair, dagree, bb)
    (out / "selfplay_summary.txt").write_text(text)
    print(text)
    if not args.no_figures:
        paths = all_figures(post, top, agree, predictors, fair, Path(args.figures_dir), deff=deff,
                            reg=reg)
        print(f"wrote {len(paths) // 2} figures (PDF and PNG) to {args.figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
