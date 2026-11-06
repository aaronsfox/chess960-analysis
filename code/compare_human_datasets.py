"""Comparisons across the human-play datasets and with engine self-play.

Uses the outputs of analyse_lichess960.py, analyse_chesscom960.py and
masters960.py (and the prepared Lichess games and the Freestyle Friday games
for the rating-band analyses). Datasets: Lichess blitz, Lichess rapid,
Chess.com Freestyle Friday, master-level over-the-board games.

1. Balance across datasets: White's advantage at equal ratings (Delta_Elo)
   per dataset; pairwise differences (independent datasets: difference with
   SE sqrt(se1^2 + se2^2), 95% CI, two-sided p); Cochran's Q across datasets.
2. Positions:
   - within each dataset, Cochran's Q for heterogeneity of the per-position
     estimates (do positions differ beyond sampling error?) and the number of
     positions whose 95% CI excludes the pooled estimate (5% expected);
   - between datasets, Spearman rho of per-position Delta_Elo over the
     positions both datasets played (bootstrap 95% CI), with the rho
     expected if both reflected the engine-measured differences in full.
3. Transfer of engine imbalance: Spearman rho between per-position human
   Delta_Elo and Stockfish / Caissa self-play Delta_Elo, overall and within
   Lichess rating bands and Freestyle Friday rating thirds (per-position
   estimates refitted within each band), with bootstrap 95% CIs and the rho
   expected under full transfer: each position's human value simulated as
   its engine value plus noise with that position's SE.

4. Subgroups for the balance and decisiveness analyses, all from models
   without engine terms: White's advantage and the decisive rate at equal
   ratings per dataset and subgroup (Lichess rating bands, Freestyle Friday
   rating thirds, master time controls and 2600+), the Lichess mean-rating
   trend (K*delta per 100 rating points), and engine self-play pooled over
   all 960 positions.
5. Decisive rate by position: Spearman rho between per-position decisive
   rates at equal ratings (Lichess, Freestyle Friday) and self-play
   per-position decisive rates (posterior means), bootstrap 95% CIs.

Outputs in results/compare_human/.

    python code/compare_human_datasets.py
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyse_lichess960 import (BAND_LABELS, K, RATING_CENTRE, band_of, decisive_model, frac_logit,
                                rating_model, rating_x, stage1)
from utils import REPO_ROOT, setup_logging

log = logging.getLogger("compare_human_datasets")

LICHESS = REPO_ROOT / "results" / "lichess960"
CHESSCOM = REPO_ROOT / "results" / "chesscom960"
MASTERS = REPO_ROOT / "results" / "masters960"
SELFPLAY = REPO_ROOT / "results" / "selfplay" / "analysis" / "selfplay_posteriors.csv"
OUT = REPO_ROOT / "results" / "compare_human"

N_BOOT = 2000
N_SIM = 200
MIN_CASE = 20            # master positions used in rank comparisons
ENGINES = ("stockfish", "caissa")
DATASET_LABELS = {"lichess_blitz": "Lichess blitz", "lichess_rapid": "Lichess rapid",
                  "freestyle_friday": "Freestyle Friday", "masters": "Masters OTB"}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def balance_estimates() -> pd.DataFrame:
    rows = []
    li = pd.read_csv(LICHESS / "analysis" / "lichess960_rating_model.csv")
    for s in ("blitz", "rapid"):
        r = li[li.set == s].iloc[0]
        rows.append({"dataset": f"lichess_{s}", "games": int(r.games),
                     "advantage_elo": r.advantage_elo, "advantage_se": r.advantage_se})
    cc = pd.read_csv(CHESSCOM / "analysis" / "chesscom960_rating_model.csv").iloc[0]
    rows.append({"dataset": "freestyle_friday", "games": int(cc.games),
                 "advantage_elo": cc.advantage_elo, "advantage_se": cc.advantage_se})
    ma = pd.read_csv(MASTERS / "masters960_pooled.csv")
    m = ma[ma.group == "all"].iloc[0]
    rows.append({"dataset": "masters", "games": int(m.games),
                 "advantage_elo": m.advantage_elo, "advantage_se": m.advantage_se})
    return pd.DataFrame(rows)


def position_estimates() -> dict[str, pd.DataFrame]:
    """Per-position Delta_Elo (rating-adjusted) and SE for each dataset."""
    out = {}
    li = pd.read_csv(LICHESS / "analysis" / "lichess960_positions.csv")
    for s in ("blitz", "rapid"):
        p = li[(li.set == s) & li.human_elo.notna()]
        out[f"lichess_{s}"] = p[["position_id", "games", "human_elo", "human_se"]].rename(
            columns={"human_elo": "elo", "human_se": "se"})
    cc = pd.read_csv(CHESSCOM / "analysis" / "chesscom960_positions.csv")
    cc = cc[cc.human_elo.notna()]
    out["freestyle_friday"] = cc[["position_id", "games", "human_elo", "human_se"]].rename(
        columns={"human_elo": "elo", "human_se": "se"})
    ma = pd.read_csv(MASTERS / "masters960_positions.csv")
    ma = ma[(ma.games >= MIN_CASE) & ma.advantage_elo.notna()]
    out["masters"] = ma[["position_id", "games", "advantage_elo", "advantage_se"]].rename(
        columns={"advantage_elo": "elo", "advantage_se": "se"})
    return {k: v.set_index("position_id") for k, v in out.items()}


def engine_values() -> pd.DataFrame:
    sp = pd.read_csv(SELFPLAY)
    return sp.pivot(index="position_id", columns="engine", values="delta_elo_median")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cochran_q(y: np.ndarray, se: np.ndarray) -> dict:
    w = 1 / se ** 2
    mean = np.sum(w * y) / np.sum(w)
    q = float(np.sum(w * (y - mean) ** 2))
    df = len(y) - 1
    return {"k": len(y), "pooled": float(mean), "pooled_se": float(np.sqrt(1 / np.sum(w))),
            "Q": q, "df": df, "Q_over_df": q / df if df else np.nan,
            "p": float(stats.chi2.sf(q, df)) if df else np.nan,
            "I2": max(0.0, (q - df) / q) if q > 0 else 0.0}


def spearman_boot(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    rho = float(stats.spearmanr(x, y).statistic)
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    boot = np.array([stats.spearmanr(x[i], y[i]).statistic for i in idx])
    return rho, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def rho_full_transfer(x: np.ndarray, se: np.ndarray, rng: np.random.Generator,
                      se2: np.ndarray | None = None) -> float:
    """Mean Spearman rho expected if the human estimates were the engine
    values plus their own sampling noise. With se2, both sides are human
    datasets reflecting the engine values (each with its own noise)."""
    sims = []
    for _ in range(N_SIM):
        a = x + rng.normal(0, se)
        b = x if se2 is None else x + rng.normal(0, se2)
        sims.append(stats.spearmanr(b, a).statistic)
    return float(np.mean(sims))


def seed(*parts) -> np.random.Generator:
    import zlib
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


# ---------------------------------------------------------------------------
# 1. Balance across datasets
# ---------------------------------------------------------------------------

def balance_comparison(bal: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    b = bal.set_index("dataset")
    for d1, d2 in itertools.combinations(b.index, 2):
        diff = b.loc[d2, "advantage_elo"] - b.loc[d1, "advantage_elo"]
        se = float(np.hypot(b.loc[d1, "advantage_se"], b.loc[d2, "advantage_se"]))
        rows.append({"dataset_1": d1, "dataset_2": d2, "difference_elo": diff, "se": se,
                     "lo": diff - 1.96 * se, "hi": diff + 1.96 * se,
                     "p": float(2 * stats.norm.sf(abs(diff / se)))})
    q = cochran_q(bal.advantage_elo.to_numpy(), bal.advantage_se.to_numpy())
    return pd.DataFrame(rows), q


# ---------------------------------------------------------------------------
# 2. Positions within and between datasets
# ---------------------------------------------------------------------------

def heterogeneity(pos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for k, p in pos.items():
        q = cochran_q(p.elo.to_numpy(), p.se.to_numpy())
        lo, hi = p.elo - 1.96 * p.se, p.elo + 1.96 * p.se
        excl = int(((lo > q["pooled"]) | (hi < q["pooled"])).sum())
        rows.append({"dataset": k, **q, "ci_excluding_pooled": excl,
                     "expected_by_chance": 0.05 * len(p)})
    return pd.DataFrame(rows)


def between_datasets(pos: dict[str, pd.DataFrame], eng: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d1, d2 in itertools.combinations(pos, 2):
        common = pos[d1].index.intersection(pos[d2].index)
        if len(common) < 5:
            continue
        a, b = pos[d1].loc[common], pos[d2].loc[common]
        rng = seed("between", d1, d2)
        rho = spearman_boot(a.elo.to_numpy(), b.elo.to_numpy(), rng)
        x = eng.loc[common, "stockfish"].to_numpy()
        bench = rho_full_transfer(x, b.se.to_numpy(), rng, se2=a.se.to_numpy())
        rows.append({"dataset_1": d1, "dataset_2": d2, "positions": len(common),
                     "rho": rho[0], "rho_lo": rho[1], "rho_hi": rho[2],
                     "rho_if_both_reflect_stockfish": bench})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Transfer of engine imbalance (overall and within rating groups)
# ---------------------------------------------------------------------------

def transfer_row(dataset: str, group: str, p: pd.DataFrame, eng: pd.DataFrame) -> list[dict]:
    rows = []
    for e in ENGINES:
        x = eng.loc[p.index, e].to_numpy()
        rng = seed("transfer", dataset, group, e)
        rho = spearman_boot(x, p.elo.to_numpy(), rng)
        rows.append({"dataset": dataset, "group": group, "engine": e, "positions": len(p),
                     "median_games": float(p.games.median()), "median_se": float(p.se.median()),
                     "rho": rho[0], "rho_lo": rho[1], "rho_hi": rho[2],
                     "rho_if_full_transfer": rho_full_transfer(x, p.se.to_numpy(), rng)})
    return rows


def per_position(d: pd.DataFrame) -> pd.DataFrame:
    rm = rating_model(d)
    p = stage1(d, rm["_b"][1])
    p = p[p.human_elo.notna()]
    return p.set_index("position_id")[["games", "human_elo", "human_se"]].rename(
        columns={"human_elo": "elo", "human_se": "se"})


def lichess_band_positions() -> dict[tuple[str, str], pd.DataFrame]:
    out = {}
    for s in ("blitz", "rapid"):
        path = LICHESS / "prepared" / f"lichess960_{s}.parquet"
        if not path.exists():
            log.warning("%s not found: run analyse_lichess960.py prepare for the band analyses", path)
            continue
        d = pd.read_parquet(path)
        band = band_of((d.white_elo.astype(float) + d.black_elo.astype(float)) / 2)
        for lab in BAND_LABELS:
            sub = d[(band == lab).to_numpy()]
            if len(sub) >= 5000:
                out[(f"lichess_{s}", lab)] = per_position(sub)
    return out


def chesscom_third_positions() -> dict[tuple[str, str], pd.DataFrame]:
    from analyse_chesscom960 import include
    g = pd.read_parquet(CHESSCOM / "chesscom960_games.parquet")
    d, _ = include(g)
    mean_r = (d.white_elo + d.black_elo) / 2
    thirds = pd.qcut(mean_r, 3, labels=["lower third", "middle third", "upper third"])
    out = {}
    for t in thirds.cat.categories:
        sub = d[(thirds == t).to_numpy()]
        r = mean_r[(thirds == t).to_numpy()]
        out[("freestyle_friday", f"{t} ({r.min():.0f}-{r.max():.0f})")] = per_position(sub)
    return out


def transfer(pos: dict[str, pd.DataFrame], eng: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, p in pos.items():
        rows += transfer_row(k, "all", p, eng)
    for (k, group), p in {**lichess_band_positions(), **chesscom_third_positions()}.items():
        rows += transfer_row(k, group, p, eng)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Balance and decisiveness by dataset and subgroup (no engine terms)
# ---------------------------------------------------------------------------

def group_row(dataset: str, group: str, d: pd.DataFrame) -> dict:
    rm = rating_model(d)
    s = d.score_white.to_numpy(float)
    gap = np.abs(d.white_elo.to_numpy(float) - d.black_elo.to_numpy(float))
    dm = decisive_model(gap, (s != 0.5).astype(float), cluster=d.position_id.to_numpy())
    mean_r = (d.white_elo.astype(float) + d.black_elo.astype(float)) / 2
    return {"dataset": dataset, "group": group, "games": len(d),
            "rating_range": f"{mean_r.min():.0f}-{mean_r.max():.0f}",
            "advantage_elo": rm["advantage_elo"], "advantage_lo": rm["advantage_lo"],
            "advantage_hi": rm["advantage_hi"], "beta": rm["beta"],
            "decisive_raw": float((s != 0.5).mean()), "decisive_adj": dm["decisive_adj"],
            "decisive_adj_lo": dm["decisive_adj_lo"], "decisive_adj_hi": dm["decisive_adj_hi"]}


def mean_rating_trend(d: pd.DataFrame) -> dict:
    """Change in White's advantage (Delta_Elo) per 100 points of mean rating:
    logit E[s] = alpha + beta (ln10/400) D + delta (M - 1750)/100."""
    mc = ((d.white_elo.to_numpy(float) + d.black_elo.to_numpy(float)) / 2 - RATING_CENTRE) / 100
    X = np.column_stack([np.ones(len(d)), rating_x(d), mc])
    b, V = frac_logit(X, d.score_white.to_numpy(float), cluster=d.position_id.to_numpy())
    se = np.sqrt(np.diag(V))
    return {"advantage_at_1750": K * b[0], "delta_elo_per_100": K * b[2],
            "delta_lo": K * (b[2] - 1.96 * se[2]), "delta_hi": K * (b[2] + 1.96 * se[2]),
            "p": float(2 * stats.norm.sf(abs(b[2] / se[2])))}


def groups() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, trends = [], []
    for s in ("blitz", "rapid"):
        path = LICHESS / "prepared" / f"lichess960_{s}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path)
        rows.append(group_row(f"lichess_{s}", "all", d))
        band = band_of((d.white_elo.astype(float) + d.black_elo.astype(float)) / 2)
        for lab in BAND_LABELS:
            sub = d[(band == lab).to_numpy()]
            if len(sub) >= 1000:
                rows.append(group_row(f"lichess_{s}", lab, sub))
        trends.append({"dataset": f"lichess_{s}", **mean_rating_trend(d)})
    from analyse_chesscom960 import include
    d, _ = include(pd.read_parquet(CHESSCOM / "chesscom960_games.parquet"))
    rows.append(group_row("freestyle_friday", "all", d))
    mean_r = (d.white_elo + d.black_elo) / 2
    thirds = pd.qcut(mean_r, 3, labels=["lower third", "middle third", "upper third"])
    for t in thirds.cat.categories:
        rows.append(group_row("freestyle_friday", t, d[(thirds == t).to_numpy()]))
    ma = pd.read_csv(MASTERS / "masters960_pooled.csv")
    for _, r in ma[ma.group.isin(["all", "time control"]) | (ma.level == "2600+")].iterrows():
        rows.append({"dataset": "masters", "group": r.level, "games": int(r.games),
                     "advantage_elo": r.advantage_elo, "advantage_lo": r.advantage_lo,
                     "advantage_hi": r.advantage_hi, "beta": 1.0,
                     "decisive_raw": 1 - r.draws, "decisive_adj": r.decisive_adj,
                     "decisive_adj_lo": r.decisive_adj_lo, "decisive_adj_hi": r.decisive_adj_hi})
    sp = pd.read_csv(SELFPLAY)
    for e, g in sp.groupby("engine"):
        n = g.games.sum()
        e_score = (g.white_wins.sum() + 0.5 * g.draws.sum()) / n
        rows.append({"dataset": f"{e}_selfplay", "group": "all 960 positions", "games": int(n),
                     "advantage_elo": 400 * np.log10(e_score / (1 - e_score)),
                     "decisive_raw": 1 - g.draws.sum() / n, "decisive_adj": 1 - g.draws.sum() / n})
    return pd.DataFrame(rows), pd.DataFrame(trends)


def decisive_spearman() -> pd.DataFrame:
    sp = pd.read_csv(SELFPLAY).set_index(["engine", "position_id"])
    li = pd.read_csv(LICHESS / "analysis" / "lichess960_positions.csv")
    cc = pd.read_csv(CHESSCOM / "analysis" / "chesscom960_positions.csv")
    sets = {"lichess_blitz": li[li.set == "blitz"], "lichess_rapid": li[li.set == "rapid"],
            "freestyle_friday": cc[cc.games >= MIN_CASE]}
    rows = []
    for k, p in sets.items():
        p = p[p.decisive_adj.notna()]
        for e in ENGINES:
            x = sp.loc[e].reindex(p.position_id).decisive_mean.to_numpy()
            rho = spearman_boot(x, p.decisive_adj.to_numpy(), seed("decisive", k, e))
            rows.append({"dataset": k, "engine": e, "positions": len(p),
                         "rho": rho[0], "rho_lo": rho[1], "rho_hi": rho[2]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def summary_text(bal, pairs, q_bal, het, betw, trans, grp=None, trend=None, dec=None) -> str:
    lab = lambda k: DATASET_LABELS.get(k, k)
    L = ["Comparisons across human-play datasets", "=" * 38, "",
         "1. White's advantage at equal ratings (Delta_Elo [95% CI])"]
    for r in bal.itertuples():
        L.append(f"  {lab(r.dataset):<17} {r.advantage_elo:+6.1f} [{r.advantage_elo - 1.96 * r.advantage_se:+.1f}, "
                 f"{r.advantage_elo + 1.96 * r.advantage_se:+.1f}]  (n {r.games:,})")
    L.append(f"  Heterogeneity across datasets: Q {q_bal['Q']:.2f} on {q_bal['df']} df, p {q_bal['p']:.3g}")
    L.append("  Pairwise differences (second minus first):")
    for r in pairs.itertuples():
        L.append(f"    {lab(r.dataset_2)} - {lab(r.dataset_1)}: {r.difference_elo:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}], "
                 f"p {r.p:.3g}")
    L += ["", "2a. Do positions differ within a dataset? (Cochran's Q on per-position Delta_Elo)"]
    for r in het.itertuples():
        L.append(f"  {lab(r.dataset):<17} k {r.k}: Q {r.Q:.1f} on {r.df} df (Q/df {r.Q_over_df:.2f}), p {r.p:.3g}; "
                 f"CIs excluding the pooled {r.pooled:+.1f}: {r.ci_excluding_pooled} (expected {r.expected_by_chance:.0f})")
    L += ["", "2b. Do positions rank alike across datasets? (Spearman rho of per-position Delta_Elo)"]
    for r in betw.itertuples():
        L.append(f"  {lab(r.dataset_1)} vs {lab(r.dataset_2)} ({r.positions} positions): rho {r.rho:+.3f} "
                 f"[{r.rho_lo:+.3f}, {r.rho_hi:+.3f}]; expected if both reflected Stockfish: "
                 f"{r.rho_if_both_reflect_stockfish:.2f}")
    L += ["", "3. Transfer of engine imbalance (Spearman rho, human vs self-play Delta_Elo)"]
    for r in trans.itertuples():
        L.append(f"  {lab(r.dataset):<17} {r.group:<28} {r.engine:<9} k {r.positions:>3}: rho {r.rho:+.3f} "
                 f"[{r.rho_lo:+.3f}, {r.rho_hi:+.3f}]; expected if full transfer {r.rho_if_full_transfer:.2f}")
    if grp is not None:
        L += ["", "4. By dataset and subgroup (models without engine terms): White's advantage and decisive "
              "rate at equal ratings"]
        for r in grp.itertuples():
            adv = (f"{r.advantage_elo:+6.1f} [{r.advantage_lo:+.1f}, {r.advantage_hi:+.1f}]"
                   if pd.notna(getattr(r, "advantage_lo", np.nan)) else f"{r.advantage_elo:+6.1f}")
            dec_ = (f"{100 * r.decisive_adj:.1f}%" + (f" [{100 * r.decisive_adj_lo:.1f}, {100 * r.decisive_adj_hi:.1f}]"
                    if pd.notna(getattr(r, "decisive_adj_lo", np.nan)) else "")) if pd.notna(r.decisive_adj) else "n/a"
            L.append(f"  {lab(r.dataset):<20} {r.group:<18} n {r.games:>9,}: advantage {adv}; decisive {dec_}")
        for r in trend.itertuples():
            L.append(f"  {lab(r.dataset)}: change in White's advantage per 100 rating points "
                     f"{r.delta_elo_per_100:+.2f} [{r.delta_lo:+.2f}, {r.delta_hi:+.2f}] Elo (p {r.p:.3g})")
    if dec is not None:
        L += ["", "5. Decisive rate by position vs engine self-play (Spearman rho)"]
        for r in dec.itertuples():
            L.append(f"  {lab(r.dataset):<17} {r.engine:<9} k {r.positions}: rho {r.rho:+.3f} "
                     f"[{r.rho_lo:+.3f}, {r.rho_hi:+.3f}]")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    OUT.mkdir(parents=True, exist_ok=True)
    bal = balance_estimates()
    pairs, q_bal = balance_comparison(bal)
    pos = position_estimates()
    eng = engine_values()
    het = heterogeneity(pos)
    betw = between_datasets(pos, eng)
    trans = transfer(pos, eng)
    grp, trend = groups()
    dec = decisive_spearman()
    grp.to_csv(OUT / "groups_balance_decisive.csv", index=False)
    trend.to_csv(OUT / "lichess_mean_rating_trend.csv", index=False)
    dec.to_csv(OUT / "decisive_spearman.csv", index=False)
    bal.to_csv(OUT / "balance_by_dataset.csv", index=False)
    pairs.to_csv(OUT / "balance_pairwise.csv", index=False)
    pd.DataFrame([q_bal]).to_csv(OUT / "balance_heterogeneity.csv", index=False)
    het.to_csv(OUT / "positions_heterogeneity.csv", index=False)
    betw.to_csv(OUT / "positions_between_datasets.csv", index=False)
    trans.to_csv(OUT / "transfer_spearman.csv", index=False)
    text = summary_text(bal, pairs, q_bal, het, betw, trans, grp, trend, dec)
    (OUT / "compare_human_summary.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
