"""Chess.com Freestyle Friday (titled players, 3+1/3+2): analysis.

Same models as the Lichess analysis, fitted on Chess.com ratings (their own
scale, estimated from the data; never compared with Lichess ratings):

- inclusion: Chess960 position identified, known result, at least 10 plies;
- rating model: White's score on the rating gap (beta fitted) -> White's
  advantage at equal ratings;
- stage 1: each position's rating-adjusted Delta_Elo (SEs clustered on the
  player pair), and its decisive rate between equally rated players;
- stage 2: REML meta-regression of position Delta_Elo on Stockfish and
  Caissa self-play Delta_Elo (tau, slope, reliability-corrected slope);
- game-level pooled slope (position-clustered SEs), overall and by thirds of
  mean Chess.com rating;
- agreement with the engines across positions (Spearman rho): rating-adjusted
  Delta_Elo, and the decisive rate at equal ratings.

Rating-matched games (|gap| <= 100) are only 12% of these Swiss games, so the
rating adjustment here is model-based throughout.

    python code/analyse_chesscom960.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

from analyse_lichess960 import (K, MATCH_GAP, PREDICTORS, calibration, decisive_model, frac_logit,
                                load_engines, rating_model, rating_x, stage1, stage2)
from plot_style import BAND_COLOUR, ENGINES, METRICS, MUTED_TEXT, apply_style, bold_tick_labels
from utils import N_POSITIONS, REPO_ROOT, setup_logging

log = logging.getLogger("analyse_chesscom960")

BASE = REPO_ROOT / "results" / "chesscom960"
GAMES = BASE / "chesscom960_games.parquet"
OUT = BASE / "analysis"
FIGURES = REPO_ROOT / "results" / "figures"
SELFPLAY = REPO_ROOT / "results" / "selfplay" / "analysis" / "selfplay_posteriors.csv"
MIN_PLIES = 10
MIN_GAMES = 20          # positions used in the agreement table
N_BOOT = 2000


def include(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    steps = [("Chess960 position identified", g.position_id >= 0),
             ("known result", g.score_white.notna()),
             (f"at least {MIN_PLIES} plies", g.plies >= MIN_PLIES),
             ("both ratings", g.white_elo.notna() & g.black_elo.notna())]
    keep = pd.Series(True, index=g.index)
    flow = [{"criterion": "parsed", "games": len(g)}]
    for label, m in steps:
        keep &= m.fillna(False).astype(bool)
        flow.append({"criterion": label, "games": int(keep.sum())})
    flow = pd.DataFrame(flow)
    flow["removed"] = -flow.games.diff()
    d = g[keep].copy()
    for side in ("white", "black"):          # hashed IDs -> integers (stage 1 clusters on pairs)
        d[f"{side}_id"] = [int(x, 16) for x in d[f"{side}_id"]]
    d["position_id"] = d.position_id.astype(int)
    return d.reset_index(drop=True), flow


def descriptives(d: pd.DataFrame) -> dict:
    s = d.score_white.to_numpy(float)
    gap = np.abs(d.white_elo.to_numpy(float) - d.black_elo.to_numpy(float))
    e = s.mean()
    dm = decisive_model(gap, (s != 0.5).astype(float), cluster=d.position_id.to_numpy())
    return {"games": len(d), "tournaments": d.tournament_id.nunique(),
            "players": pd.unique(np.concatenate([d.white_id, d.black_id])).size,
            "positions": d.position_id.nunique(), "white_wins": float((s == 1).mean()),
            "draws": float((s == 0.5).mean()), "black_wins": float((s == 0).mean()),
            "escore": e, "delta_elo_raw": 400 * np.log10(e / (1 - e)),
            "decisive_raw": float((s != 0.5).mean()),
            **{k: v for k, v in dm.items() if not k.startswith("_")},
            "games_matched": int((gap <= MATCH_GAP).sum()),
            "mean_rating_median": float(((d.white_elo + d.black_elo) / 2).median())}


def pooled_slopes(d: pd.DataFrame, eng: pd.DataFrame) -> pd.DataFrame:
    """Game-level fractional logit: White's score on the rating gap (beta
    free) and the position's engine Delta_Elo / K, position-clustered;
    overall and by thirds of mean rating."""
    mean_r = (d.white_elo + d.black_elo) / 2
    thirds = pd.qcut(mean_r, 3, labels=["lower third", "middle third", "upper third"])
    rows = []
    for e in PREDICTORS:
        x = eng[("delta_elo_median", e)]
        for name, m in [("all", np.ones(len(d), bool))] + [(t, (thirds == t).to_numpy())
                                                            for t in thirds.cat.categories]:
            sub = d[m]
            xc = (x - x.mean()).reindex(sub.position_id).to_numpy() / K
            X = np.column_stack([np.ones(len(sub)), rating_x(sub), xc])
            b, V = frac_logit(X, sub.score_white.to_numpy(float), cluster=sub.position_id.to_numpy())
            se = np.sqrt(np.diag(V))
            r = mean_r[m]
            rows.append({"predictor": e, "group": name, "games": len(sub),
                         "rating_range": f"{r.min():.0f}-{r.max():.0f}",
                         "advantage_elo": K * b[0], "advantage_lo": K * (b[0] - 1.96 * se[0]),
                         "advantage_hi": K * (b[0] + 1.96 * se[0]), "beta": b[1], "beta_se": se[1],
                         "gamma": b[2], "gamma_se": se[2], "gamma_lo": b[2] - 1.96 * se[2],
                         "gamma_hi": b[2] + 1.96 * se[2],
                         "p_gamma": 2 * stats.norm.sf(abs(b[2] / se[2]))})
    return pd.DataFrame(rows)


def spearman_ci(x, y, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    rho = stats.spearmanr(x, y).statistic
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    boot = [stats.spearmanr(x[i], y[i]).statistic for i in idx]
    return float(rho), float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def agreement(pos: pd.DataFrame, selfplay: Path = SELFPLAY) -> pd.DataFrame:
    sp = pd.read_csv(selfplay).set_index(["engine", "position_id"])
    p = pos[(pos.games >= MIN_GAMES) & pos.human_elo.notna()].set_index("position_id")
    rows = []
    for k, e in enumerate(PREDICTORS):
        ee = sp.loc[e].reindex(p.index)
        r1 = spearman_ci(ee.delta_elo_median.to_numpy(), p.human_elo.to_numpy(), 10 + k)
        q = p.decisive_adj.notna().to_numpy()
        r2 = spearman_ci(ee.decisive_mean.to_numpy()[q], p.decisive_adj.to_numpy()[q], 20 + k)
        rows.append({"engine": e, "positions": len(p), "rho_delta_elo": r1[0], "rho_delta_elo_lo": r1[1],
                     "rho_delta_elo_hi": r1[2], "positions_decisive": int(q.sum()),
                     "rho_decisive": r2[0], "rho_decisive_lo": r2[1], "rho_decisive_hi": r2[2]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def fig_human_vs_selfplay(pos: pd.DataFrame, meta: pd.DataFrame, deciles: pd.DataFrame,
                          out_dir: Path) -> list[Path]:
    """Freestyle Friday position Delta_Elo (rating-adjusted, faint points)
    against Stockfish and Caissa self-play Delta_Elo; solid line: the
    meta-regression; dotted: equality; coloured: decile means with 95% CIs."""
    plt = apply_style()
    from matplotlib.ticker import MultipleLocator
    p = pos[pos.human_elo.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True, gridspec_kw={"wspace": 0.1})
    ylo, yhi = np.percentile(p.human_elo, [1, 99])
    for ax, e in zip(axes, PREDICTORS):
        r = meta[(meta.model == "slope") & (meta.predictor == e)].iloc[0]
        dc = deciles[deciles.predictor == e]
        x = p[f"{e}_selfplay_elo"]
        xlo, xhi = x.min() - 5, x.max() + 5
        g = np.array([xlo, xhi])
        ax.plot(g, g, color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)), zorder=1)
        ax.axhline(0, color=BAND_COLOUR, lw=0.8, zorder=0)
        ax.scatter(x, p.human_elo, s=5, color="#222222", alpha=0.3, lw=0, zorder=2)
        ax.plot(g, r.mu + r.gamma * (g - r.engine_mean), color="#111111", lw=1.1, zorder=3)
        st = ENGINES[e]
        ax.errorbar(dc.engine_mean, dc.human_elo, yerr=[dc.human_elo - dc.lo, dc.hi - dc.human_elo],
                    fmt=st.marker, ms=4, color=st.colour, mec="white", mew=0.6, elinewidth=1.0,
                    capsize=0, zorder=4)
        ax.text(0.05, 0.95, f"Slope (SE) = {r.gamma:.2f} ({r.gamma_se:.2f})", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5, fontweight="bold")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo - 10, yhi + 10)
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.set_xlabel(f"{ENGINES[e].label} Self-play {METRICS['delta_elo'].bold_symbol}")
        bold_tick_labels(ax)
    axes[0].set_ylabel(f"Freestyle Friday {METRICS['delta_elo'].bold_symbol}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "chesscom960_human_vs_selfplay.pdf", out_dir / "chesscom960_human_vs_selfplay.png"]
    for path in paths:
        fig.savefig(path, metadata={"CreationDate": None} if path.suffix == ".pdf" else None)
    plt.close(fig)
    return paths


def summary_text(flow, desc, rm, meta, slopes, agree, pos) -> str:
    L = ["Chess.com Freestyle Friday: analysis summary", "=" * 44, "", "Inclusion:"]
    L += [f"  {r.criterion}: {r.games:,}" for r in flow.itertuples()]
    L += ["", f"{desc['games']:,} games, {desc['tournaments']} tournaments, {desc['players']:,} players, "
          f"{desc['positions']} positions (median {pos[pos.games > 0].games.median():.0f} games each)",
          f"W/D/L {100 * desc['white_wins']:.1f} / {100 * desc['draws']:.1f} / {100 * desc['black_wins']:.1f}%; "
          f"raw Delta_Elo {desc['delta_elo_raw']:+.1f}",
          f"Decisive rate: raw {100 * desc['decisive_raw']:.1f}%; at equal ratings {100 * desc['decisive_adj']:.1f}% "
          f"[{100 * desc['decisive_adj_lo']:.1f}, {100 * desc['decisive_adj_hi']:.1f}]",
          f"Rating model: White advantage at equal ratings {rm['advantage_elo']:+.1f} "
          f"[{rm['advantage_lo']:+.1f}, {rm['advantage_hi']:+.1f}] Elo; beta {rm['beta']:.3f} "
          f"[{rm['beta_lo']:.3f}, {rm['beta_hi']:.3f}] (1 = Elo scale on Chess.com ratings)"]
    null = meta[meta.model == "null"].iloc[0]
    L.append(f"Positions: mean SE {null.mean_se:.1f} Elo; tau {null.tau:.1f} [{null.tau_lo:.1f}, "
             f"{null.tau_hi:.1f}] (p {null.p_tau:.3g}); positions used {int(null.positions_used)}")
    for _, m in meta[meta.model == "slope"].iterrows():
        L.append(f"  meta-regression on {m.predictor}: slope {m.gamma:.3f} [{m.gamma_lo:.3f}, {m.gamma_hi:.3f}] "
                 f"(p {m.p_gamma:.3g}); corrected {m.gamma_corrected:.3f}; rho {m.spearman_rho:.3f}")
    L.append("Game-level slopes (position-clustered):")
    for _, s in slopes.iterrows():
        L.append(f"  {s.predictor} / {s.group} (ratings {s.rating_range}, n {s.games:,}): advantage "
                 f"{s.advantage_elo:+.1f} [{s.advantage_lo:+.1f}, {s.advantage_hi:+.1f}]; slope {s.gamma:.3f} "
                 f"[{s.gamma_lo:.3f}, {s.gamma_hi:.3f}]")
    L.append(f"Agreement with the engines across positions with >= {MIN_GAMES} games (Spearman rho):")
    for _, a in agree.iterrows():
        L.append(f"  {a.engine}: Delta_Elo {a.rho_delta_elo:+.3f} [{a.rho_delta_elo_lo:+.3f}, "
                 f"{a.rho_delta_elo_hi:+.3f}]; decisive rate at equal ratings {a.rho_decisive:+.3f} "
                 f"[{a.rho_decisive_lo:+.3f}, {a.rho_decisive_hi:+.3f}] ({a.positions_decisive} positions)")
    for pid in (868, 879, 518):
        if pid in set(pos.position_id) and pos.set_index("position_id").loc[pid, "games"] > 0:
            r = pos.set_index("position_id").loc[pid]
            L.append(f"#{pid}: {int(r.games)} games; {r.human_elo:+.1f} [{r.human_lo:+.1f}, {r.human_hi:+.1f}]; "
                     f"Stockfish {r.stockfish_selfplay_elo:+.0f}")
    return "\n".join(L) + "\n"


def analyse(games: Path = GAMES, out: Path = OUT, figures: Path = FIGURES) -> None:
    g = pd.read_parquet(games)
    d, flow = include(g)
    eng = load_engines(SELFPLAY)
    rm = rating_model(d)
    pos = stage1(d, rm["_b"][1])
    pos, meta, dec = stage2(pos, eng)
    slopes = pooled_slopes(d, eng)
    desc = descriptives(d)
    agree = agreement(pos)
    out.mkdir(parents=True, exist_ok=True)
    flow.to_csv(out / "chesscom960_flow.csv", index=False)
    pd.DataFrame([desc]).to_csv(out / "chesscom960_descriptives.csv", index=False)
    pd.DataFrame([{k: v for k, v in rm.items() if not k.startswith("_")}]).to_csv(
        out / "chesscom960_rating_model.csv", index=False)
    calibration(d, rm["_b"]).to_csv(out / "chesscom960_rating_calibration.csv", index=False)
    pos.to_csv(out / "chesscom960_positions.csv", index=False)
    meta.to_csv(out / "chesscom960_meta_regression.csv", index=False)
    dec.to_csv(out / "chesscom960_deciles.csv", index=False)
    slopes.to_csv(out / "chesscom960_slopes.csv", index=False)
    agree.to_csv(out / "chesscom960_agreement.csv", index=False)
    text = summary_text(flow, desc, rm, meta, slopes, agree, pos)
    (out / "chesscom960_summary.txt").write_text(text)
    print(text)
    for path in fig_human_vs_selfplay(pos, meta, dec, figures):
        if path.suffix == ".png":
            print(f"Figure: {path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    analyse()
    return 0


if __name__ == "__main__":
    sys.exit(main())
