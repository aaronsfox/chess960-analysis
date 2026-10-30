"""Human play (Part III): rated Lichess Chess960 games against the engine
self-play results.

Subcommands
-----------
prepare  Combine the monthly tables from lichess960.py, apply the inclusion
         criteria separately for blitz (estimated 300-479 s) and rapid
         (>= 480 s), and write one compact table per set to
         results/lichess960/prepared/. Also writes the flow table (games
         removed at each step), a randomisation check (games per position
         against uniform) and the rating-change distribution used for the
         provisional-rating rule.
analyse  For each set:
           - descriptives overall and by rating band;
           - rating model: White score against the rating gap (is the Lichess
             scale the 400-point Elo scale?) and White's advantage at equal
             ratings;
           - stage 1: each position's rating-adjusted human Delta_Elo and SE;
           - stage 2: random-effects meta-regression over the 960 positions,
             with tau (true spread between positions) and the slope on
             engine self-play Delta_Elo (Stockfish primary, Caissa second);
           - game-level model with rating-band interaction, and fits by band.
         Writes tables to results/lichess960/analysis/ and figures to
         results/figures/.
all      prepare, then analyse.

Model
-----
White's score s (1, 1/2, 0) is modelled with a fractional logit, so the
fitted mean is the expected score E_score used throughout the paper. Logit
units become Elo by multiplying by K = 400 / ln 10 (about 173.7). With the
rating gap D = R_White - R_Black entered as D * ln10 / 400, its coefficient
beta is 1 when Lichess ratings follow the Elo scale, and the intercept times K
is White's advantage in Elo at equal ratings.

Stage 1 fixes beta from the set's rating model and fits one intercept per
position (robust SEs, so draws are handled correctly). Stage 2 fits
alpha_p = mu + gamma * (engine Delta_Elo_p - mean) + u_p, u_p ~ N(0, tau^2),
by REML with the stage-1 variances known. gamma is also reported corrected
for noise in the engine values (divided by their reliability, from the
self-play posterior SDs).

    python code/analyse_lichess960.py all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from plot_style import BAND_COLOUR, ENGINES, METRICS, MUTED_TEXT, apply_style, bold_tick_labels
from utils import N_POSITIONS, REPO_ROOT, STANDARD_POSITION_ID, setup_logging

log = logging.getLogger("analyse_lichess960")

BASE = REPO_ROOT / "results" / "lichess960"
RAW = BASE / "raw"
PREPARED = BASE / "prepared"
OUT_TABLES = BASE / "analysis"
OUT_FIGURES = REPO_ROOT / "results" / "figures"
SELFPLAY_POSTERIORS = REPO_ROOT / "results" / "selfplay" / "analysis" / "selfplay_posteriors.csv"

K = 400 / np.log(10)                 # logit -> Elo
SETS = {"blitz": (300, 480), "rapid": (480, np.inf)}   # estimated seconds, [lo, hi)
SET_LABELS = {"blitz": "Blitz", "rapid": "Rapid"}
BAD_TERMINATIONS = {"Abandoned", "Rules infraction"}
MIN_PLIES = 10
PROVISIONAL_CHANGE = 30              # |rating change| above this = provisional (high RD)
RATING_FLOOR = 1200
BAND_EDGES = [1200, 1500, 1800, 2000, np.inf]
BAND_LABELS = ["1200-1499", "1500-1799", "1800-1999", "2000+"]
RATING_CENTRE = 1750
PREDICTORS = ["stockfish", "caissa"]     # engine self-play Delta_Elo; Stockfish primary
CALIBRATION_BINS = np.arange(-400, 401, 50)

RAW_COLUMNS = ["month", "game_id", "position_id", "estimated_s", "white_elo", "black_elo",
               "white_rating_diff", "black_rating_diff", "white_berserk", "black_berserk",
               "termination", "score_white", "plies", "white_id", "black_id", "tournament"]


# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------

def band_of(mean_rating: pd.Series) -> pd.Series:
    return pd.cut(mean_rating, BAND_EDGES, labels=BAND_LABELS, right=False)


def hex_to_int(s: pd.Series) -> np.ndarray:
    """12-hex-character player IDs to int64 (48 bits), to save memory."""
    return np.array([int(x, 16) for x in s.astype(str)], dtype=np.int64)


def filter_steps(g: pd.DataFrame, lo: float, hi: float) -> list[tuple[str, pd.Series]]:
    """Inclusion criteria in order, each as (label, keep mask)."""
    big_change = (g.white_rating_diff.abs() > PROVISIONAL_CHANGE) | \
                 (g.black_rating_diff.abs() > PROVISIONAL_CHANGE)
    return [
        ("time control", (g.estimated_s >= lo) & (g.estimated_s < hi)),
        ("not abandoned / rules infraction", ~g.termination.isin(BAD_TERMINATIONS)),
        (f"at least {MIN_PLIES} plies", g.plies >= MIN_PLIES),
        ("no berserk", ~(g.white_berserk | g.black_berserk)),
        ("both ratings present", g.white_elo.notna() & g.black_elo.notna()),
        (f"no provisional rating (|change| <= {PROVISIONAL_CHANGE})", ~big_change.fillna(False)),
        (f"mean rating >= {RATING_FLOOR}", (g.white_elo + g.black_elo) / 2 >= RATING_FLOOR),
    ]


def apply_filters(g: pd.DataFrame, lo: float, hi: float) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    flow = [("parsed (rated, blitz and slower)", len(g))]
    keep = pd.Series(True, index=g.index)
    for label, mask in filter_steps(g, lo, hi):
        keep &= mask.fillna(False).astype(bool)
        flow.append((label, int(keep.sum())))
    return g[keep], flow


def compact_set(g: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "month": g.month.astype("category"),
        "position_id": g.position_id.astype(np.int16),
        "estimated_s": g.estimated_s.astype(np.int32),
        "white_elo": g.white_elo.astype(np.int16),
        "black_elo": g.black_elo.astype(np.int16),
        "score_white": g.score_white.astype(np.float32),
        "white_id": hex_to_int(g.white_id),
        "black_id": hex_to_int(g.black_id),
        "tournament": g.tournament.astype(bool),
        "time_forfeit": (g.termination == "Time forfeit").to_numpy(),
    })
    return out.reset_index(drop=True)


def randomisation_check(pos: pd.Series) -> dict:
    n = pos.value_counts().reindex(range(N_POSITIONS), fill_value=0)
    e = n.sum() / N_POSITIONS
    chi2 = float(((n - e) ** 2 / e).sum())
    z = (n - e) / np.sqrt(e * (1 - 1 / N_POSITIONS))
    return {"games": int(n.sum()), "expected_per_position": e, "min": int(n.min()),
            "max": int(n.max()), "chi2": chi2, "df": N_POSITIONS - 1,
            "p": float(stats.chi2.sf(chi2, N_POSITIONS - 1)),
            "z_518": float(z[STANDARD_POSITION_ID]), "max_abs_z": float(z.abs().max()),
            "position_max_abs_z": int(z.abs().idxmax())}


def prepare(raw_dir: Path = RAW, out_dir: Path = PREPARED, tables: Path = OUT_TABLES) -> None:
    files = sorted(raw_dir.glob("lichess960_*.parquet"))
    if not files:
        raise SystemExit(f"no monthly tables in {raw_dir}; run lichess960.py fetch first")
    parts = {s: [] for s in SETS}
    flows = {s: None for s in SETS}
    change_q = []
    seen: set[str] = set()
    n_dup = 0
    for f in files:
        g = pd.read_parquet(f, columns=RAW_COLUMNS)
        dup = g.game_id.isin(seen) | g.game_id.duplicated()
        n_dup += int(dup.sum())
        g = g[~dup]
        seen.update(g.game_id)
        for s, (lo, hi) in SETS.items():
            kept, flow = apply_filters(g, lo, hi)
            parts[s].append(compact_set(kept))
            counts = np.array([c for _, c in flow])
            flows[s] = (flow if flows[s] is None else
                        [(lab, c0 + c) for (lab, c0), c in zip(flows[s], counts)])
        tc = g[(g.estimated_s >= 300)]
        change_q.append(np.maximum(tc.white_rating_diff.abs(), tc.black_rating_diff.abs()).dropna()
                        .to_numpy(np.float32))
        log.info("%s: %d games", f.stem, len(g))
    out_dir.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    flow_rows, rand_rows = [], []
    for s in SETS:
        d = pd.concat(parts[s], ignore_index=True)
        d["month"] = d.month.astype(str).astype("category")
        d.to_parquet(out_dir / f"lichess960_{s}.parquet", index=False)
        prev = None
        for step, (label, n) in enumerate(flows[s]):
            flow_rows.append({"set": s, "step": step, "criterion": label, "games": n,
                              "removed": None if prev is None else prev - n})
            prev = n
        rand_rows.append({"set": s, **randomisation_check(d.position_id)})
        log.info("%s: %d games kept", s, len(d))
    pd.DataFrame(flow_rows).to_csv(tables / "lichess960_flow.csv", index=False)
    pd.DataFrame(rand_rows).to_csv(tables / "lichess960_randomisation.csv", index=False)
    ch = np.concatenate(change_q)
    q = {f"p{p}": float(np.percentile(ch, p)) for p in (50, 75, 90, 95, 99)}
    q.update({f"share_over_{t}": float((ch > t).mean()) for t in (20, 30, 50)})
    q["duplicates_removed"] = n_dup
    pd.DataFrame([q]).to_csv(tables / "lichess960_rating_change.csv", index=False)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def frac_logit(X: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None,
               cluster: np.ndarray | None = None, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Fractional logit by Newton-Raphson. Returns coefficients and their
    covariance: sandwich (robust) or, with `cluster` codes, cluster-robust."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = special.expit(X @ b + off)
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < tol:
            break
    p = special.expit(X @ b + off)
    H = X.T @ (X * (p * (1 - p))[:, None])
    U = X * (y - p)[:, None]
    if cluster is None:
        meat = U.T @ U
    else:
        codes, inv = np.unique(cluster, return_inverse=True)
        S = np.column_stack([np.bincount(inv, U[:, j], len(codes)) for j in range(U.shape[1])])
        g = len(codes)
        meat = S.T @ S * g / max(g - 1, 1)
    Hi = np.linalg.inv(H)
    return b, Hi @ meat @ Hi


def rating_x(d: pd.DataFrame) -> np.ndarray:
    return (d.white_elo.to_numpy(float) - d.black_elo.to_numpy(float)) * np.log(10) / 400


def rating_model(d: pd.DataFrame) -> dict:
    X = np.column_stack([np.ones(len(d)), rating_x(d)])
    b, V = frac_logit(X, d.score_white, cluster=d.position_id.to_numpy())
    se = np.sqrt(np.diag(V))
    return {"games": len(d), "advantage_elo": K * b[0], "advantage_se": K * se[0],
            "advantage_lo": K * (b[0] - 1.96 * se[0]), "advantage_hi": K * (b[0] + 1.96 * se[0]),
            "beta": b[1], "beta_se": se[1], "beta_lo": b[1] - 1.96 * se[1],
            "beta_hi": b[1] + 1.96 * se[1], "_b": b}


def calibration(d: pd.DataFrame, b: np.ndarray) -> pd.DataFrame:
    gap = d.white_elo.astype(float) - d.black_elo.astype(float)
    cut = pd.cut(gap, CALIBRATION_BINS)
    g = pd.DataFrame({"bin": cut, "gap": gap, "score": d.score_white.astype(float)}) \
        .groupby("bin", observed=True).agg(games=("score", "size"), gap_mean=("gap", "mean"),
                                           observed=("score", "mean"),
                                           sd=("score", "std")).reset_index()
    g["se"] = g.sd / np.sqrt(g.games)
    g["fitted"] = special.expit(b[0] + b[1] * g.gap_mean * np.log(10) / 400)
    g["elo_curve"] = special.expit(b[0] + g.gap_mean * np.log(10) / 400)
    g["bin"] = g.bin.astype(str)
    return g.drop(columns="sd")


def stage1(d: pd.DataFrame, beta: float) -> pd.DataFrame:
    """One intercept per position with the rating gap as an offset: White's
    human advantage at equal ratings, in Elo, with robust SE. Also SEs
    clustered on the White and on the Black player within each position, as
    a check that repeat players don't matter."""
    pos = d.position_id.to_numpy(np.int64)
    y = d.score_white.to_numpy(float)
    off = beta * rating_x(d)
    a = np.zeros(N_POSITIONS)
    for _ in range(50):
        p = special.expit(a[pos] + off)
        g = np.bincount(pos, y - p, N_POSITIONS)
        h = np.bincount(pos, p * (1 - p), N_POSITIONS)
        step = np.divide(g, h, out=np.zeros_like(g), where=h > 0)
        a += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = special.expit(a[pos] + off)
    r = y - p
    h = np.bincount(pos, p * (1 - p), N_POSITIONS)
    se = np.sqrt(np.bincount(pos, r ** 2, N_POSITIONS)) / h
    out = {"position_id": np.arange(N_POSITIONS), "games": np.bincount(pos, minlength=N_POSITIONS),
           "white_wins": np.bincount(pos, y == 1, N_POSITIONS).astype(int),
           "draws": np.bincount(pos, y == 0.5, N_POSITIONS).astype(int),
           "black_wins": np.bincount(pos, y == 0, N_POSITIONS).astype(int),
           "escore_observed": np.bincount(pos, y, N_POSITIONS) / np.bincount(pos, minlength=N_POSITIONS),
           "human_elo": K * a, "human_se": K * se}
    for side in ("white", "black"):
        s = pd.DataFrame({"pos": pos, "pl": d[f"{side}_id"].to_numpy(), "r": r}) \
            .groupby(["pos", "pl"], sort=False).r.sum()
        cl = np.sqrt((s ** 2).groupby(level=0).sum().reindex(range(N_POSITIONS), fill_value=0))
        out[f"se_ratio_{side}_cluster"] = cl.to_numpy() / h / se
    return pd.DataFrame(out)


def reml(y: np.ndarray, v: np.ndarray, X: np.ndarray) -> dict:
    """Random-effects meta-regression y = Xb + u + e, u ~ N(0, tau2),
    e ~ N(0, v) with v known; tau2 by REML. Includes the boundary LRT for
    tau2 > 0 and a profile-likelihood 95% CI for tau."""
    def fit(t2):
        w = 1 / (v + t2)
        A = X.T @ (X * w[:, None])
        b = np.linalg.solve(A, X.T @ (w * y))
        r = y - X @ b
        nll = 0.5 * (np.sum(np.log(v + t2)) + np.linalg.slogdet(A)[1] + np.sum(w * r ** 2))
        return nll, b, np.linalg.inv(A)

    upper = max(10 * np.var(y), 1.0)
    res = optimize.minimize_scalar(lambda t: fit(t)[0], bounds=(0, upper), method="bounded",
                                   options={"xatol": 1e-6})
    t2 = res.x if fit(res.x)[0] < fit(0.0)[0] else 0.0
    nll, b, cov = fit(t2)
    nll0 = fit(0.0)[0]
    stat = max(2 * (nll0 - nll), 0.0)
    p = 1.0 if stat < 1e-9 else 0.5 * stats.chi2.sf(stat, 1)
    crit = nll + stats.chi2.ppf(0.95, 1) / 2
    f = lambda t: fit(t)[0] - crit
    lo = 0.0 if f(0.0) <= 0 or t2 == 0 else optimize.brentq(f, 0.0, t2)
    hi = optimize.brentq(f, t2, upper) if f(upper) > 0 else upper
    return {"tau2": t2, "tau": np.sqrt(t2), "tau_lo": np.sqrt(lo), "tau_hi": np.sqrt(hi),
            "b": b, "cov": cov, "lrt": stat, "p_tau": p, "nll": nll}


def load_engines(path: Path = SELFPLAY_POSTERIORS) -> pd.DataFrame:
    post = pd.read_csv(path, usecols=["engine", "position_id", "delta_elo_median", "delta_elo_sd"])
    return post.pivot(index="position_id", columns="engine",
                      values=["delta_elo_median", "delta_elo_sd"])


def engine_reliability(x: np.ndarray, sd: np.ndarray) -> float:
    return float(1 - np.mean(sd ** 2) / np.var(x, ddof=1))


def stage2(pos: pd.DataFrame, eng: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Null model (tau, shrunken estimates, ranks) and one meta-regression
    per engine predictor; deciles of engine Delta_Elo for the figure."""
    y = pos.human_elo.to_numpy()
    v = pos.human_se.to_numpy() ** 2
    one = np.ones((len(y), 1))
    null = reml(y, v, one)
    mu, t2 = null["b"][0], null["tau2"]
    shrink = t2 / (t2 + v)
    pos = pos.copy()
    pos["shrunk_elo"] = mu + shrink * (y - mu)
    sd = np.sqrt(t2 * v / (t2 + v) + (1 - shrink) ** 2 * null["cov"][0, 0])
    pos["shrunk_lo"] = pos.shrunk_elo - 1.96 * sd
    pos["shrunk_hi"] = pos.shrunk_elo + 1.96 * sd
    pos["rank_shrunk"] = pos.shrunk_elo.rank(ascending=False, method="min").astype(int)
    human_rel = t2 / (t2 + np.mean(v))
    rows = [{"model": "null", "predictor": "", "mu": mu, "mu_se": np.sqrt(null["cov"][0, 0]),
             "tau": null["tau"], "tau_lo": null["tau_lo"], "tau_hi": null["tau_hi"],
             "lrt_tau": null["lrt"], "p_tau": null["p_tau"], "human_reliability": human_rel,
             "mean_se": float(np.sqrt(np.mean(v)))}]
    deciles = []
    for e in PREDICTORS:
        x = eng[("delta_elo_median", e)].reindex(pos.position_id).to_numpy()
        xsd = eng[("delta_elo_sd", e)].reindex(pos.position_id).to_numpy()
        pos[f"{e}_selfplay_elo"] = x
        xc = x - x.mean()
        fit = reml(y, v, np.column_stack([one, xc]))
        se = np.sqrt(np.diag(fit["cov"]))
        lam = engine_reliability(x, xsd)
        rho = stats.spearmanr(x, y).statistic
        denom = np.sqrt(lam * human_rel)
        rows.append({
            "model": "slope", "predictor": e, "mu": fit["b"][0], "mu_se": se[0],
            "gamma": fit["b"][1], "gamma_se": se[1], "gamma_lo": fit["b"][1] - 1.96 * se[1],
            "gamma_hi": fit["b"][1] + 1.96 * se[1],
            "p_gamma": 2 * stats.norm.sf(abs(fit["b"][1] / se[1])),
            "intercept_at_engine_zero": fit["b"][0] - fit["b"][1] * x.mean(),
            "tau": fit["tau"], "tau_lo": fit["tau_lo"], "tau_hi": fit["tau_hi"],
            "lrt_tau": fit["lrt"], "p_tau": fit["p_tau"],
            "tau2_explained": (1 - fit["tau2"] / t2) if t2 > 0 else np.nan,
            "engine_reliability": lam, "gamma_corrected": fit["b"][1] / lam,
            "gamma_corrected_lo": (fit["b"][1] - 1.96 * se[1]) / lam,
            "gamma_corrected_hi": (fit["b"][1] + 1.96 * se[1]) / lam,
            "spearman_rho": rho, "human_reliability": human_rel,
            "spearman_rho_corrected": rho / denom if human_rel > 0.05 else np.nan,
            "engine_mean": x.mean()})
        dec = pd.qcut(x, 10, labels=False)
        for k in range(10):
            m = dec == k
            w = 1 / (v[m] + fit["tau2"])
            est = np.sum(w * y[m]) / np.sum(w)
            s_ = np.sqrt(1 / np.sum(w))
            deciles.append({"predictor": e, "decile": k + 1, "positions": int(m.sum()),
                            "engine_mean": x[m].mean(), "human_elo": est,
                            "lo": est - 1.96 * s_, "hi": est + 1.96 * s_})
    return pos, pd.DataFrame(rows), pd.DataFrame(deciles)


def game_level(d: pd.DataFrame, eng: pd.DataFrame, beta_free: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rating dependence with the primary predictor (Stockfish): one model
    with a mean-rating interaction, and separate fits by rating band.
    Position-clustered SEs. The engine column is Delta_Elo / K, so its
    coefficient (gamma) is human Elo per engine Elo, and Elo effects are
    coefficient x K."""
    x = eng[("delta_elo_median", PREDICTORS[0])]
    xc = (x - x.mean()).reindex(d.position_id).to_numpy() / K
    mc = ((d.white_elo.to_numpy(float) + d.black_elo.to_numpy(float)) / 2 - RATING_CENTRE) / 100
    dx = rating_x(d)
    y = d.score_white.to_numpy(float)
    cl = d.position_id.to_numpy()
    X = np.column_stack([np.ones(len(d)), dx, xc, mc, xc * mc])
    b, V = frac_logit(X, y, cluster=cl)
    se = np.sqrt(np.diag(V))
    names = ["advantage_elo", "beta", "gamma", "delta_elo_per_100", "eta_gamma_per_100"]
    scale = [K, 1, 1, K, 1]
    inter = pd.DataFrame({"term": names, "estimate": b * scale, "se": se * scale})
    inter["lo"] = inter.estimate - 1.96 * inter.se
    inter["hi"] = inter.estimate + 1.96 * inter.se
    inter["p"] = 2 * stats.norm.sf(np.abs(inter.estimate / inter.se))
    rows = []
    band = band_of(pd.Series((d.white_elo.to_numpy(float) + d.black_elo.to_numpy(float)) / 2))
    for lab in BAND_LABELS:
        m = (band == lab).to_numpy()
        Xb = np.column_stack([np.ones(m.sum()), dx[m], xc[m]])
        bb, Vb = frac_logit(Xb, y[m], cluster=cl[m])
        sb = np.sqrt(np.diag(Vb))
        rows.append({"band": lab, "games": int(m.sum()),
                     "advantage_elo": K * bb[0], "advantage_lo": K * (bb[0] - 1.96 * sb[0]),
                     "advantage_hi": K * (bb[0] + 1.96 * sb[0]),
                     "gamma": bb[2], "gamma_se": sb[2], "gamma_lo": bb[2] - 1.96 * sb[2],
                     "gamma_hi": bb[2] + 1.96 * sb[2], "beta": bb[1], "beta_se": sb[1]})
    return inter, pd.DataFrame(rows)


def descriptives(d: pd.DataFrame) -> pd.DataFrame:
    band = band_of((d.white_elo.astype(float) + d.black_elo.astype(float)) / 2)
    rows = []
    for lab, m in [("all", np.ones(len(d), bool))] + [(b, (band == b).to_numpy()) for b in BAND_LABELS]:
        s = d.score_white.to_numpy(float)[m]
        e = s.mean()
        rows.append({"band": lab, "games": int(m.sum()), "white_wins": float((s == 1).mean()),
                     "draws": float((s == 0.5).mean()), "black_wins": float((s == 0).mean()),
                     "escore": e, "delta_elo_raw": 400 * np.log10(e / (1 - e)),
                     "mean_gap": float((d.white_elo.astype(float) - d.black_elo.astype(float))[m].mean()),
                     "time_forfeit": float(d.time_forfeit.to_numpy()[m].mean()),
                     "tournament": float(d.tournament.to_numpy()[m].mean()),
                     "players": int(pd.unique(np.concatenate([d.white_id.to_numpy()[m],
                                                             d.black_id.to_numpy()[m]])).size)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

SET_STYLE = {"blitz": dict(color="#111111", marker="o", mfc="#111111"),
             "rapid": dict(color="#777777", marker="s", mfc="white")}


def save_fig(fig, name: str, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for path in paths:
        fig.savefig(path, metadata={"CreationDate": None} if path.suffix == ".pdf" else None)
    plt.close(fig)
    return paths


def fig_human_vs_selfplay(positions: dict, meta: dict, deciles: dict, out_dir: Path) -> list[Path]:
    """Rows: Stockfish, Caissa self-play Delta_Elo. Columns: blitz, rapid.
    Faint points: each position's stage-1 human Delta_Elo; solid line: the
    meta-regression; dotted: equality; coloured: decile means with 95% CIs."""
    plt = apply_style()
    from matplotlib.ticker import MultipleLocator
    fig, axes = plt.subplots(len(PREDICTORS), len(SETS), figsize=(7.0, 3.4 * len(PREDICTORS)),
                             sharey=True, gridspec_kw={"wspace": 0.1, "hspace": 0.35})
    ally = pd.concat([p.human_elo for p in positions.values()])
    ylo, yhi = np.percentile(ally, [0.5, 99.5])
    ylo, yhi = ylo - 5, yhi + 5
    for i, e in enumerate(PREDICTORS):
        allx = pd.concat([p[f"{e}_selfplay_elo"] for p in positions.values()])
        xlo, xhi = allx.min() - 5, allx.max() + 5
        for j, s in enumerate(SETS):
            ax = axes[i, j]
            p = positions[s]
            r = meta[s][(meta[s].model == "slope") & (meta[s].predictor == e)].iloc[0]
            dc = deciles[s][deciles[s].predictor == e]
            g = np.array([xlo, xhi])
            ax.plot(g, g, color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)), zorder=1)
            ax.axhline(0, color=BAND_COLOUR, lw=0.8, zorder=0)
            ax.scatter(p[f"{e}_selfplay_elo"], p.human_elo, s=5, color="#222222", alpha=0.25,
                       lw=0, zorder=2)
            ax.plot(g, r.mu + r.gamma * (g - r.engine_mean), color="#111111", lw=1.1, zorder=3)
            st = ENGINES[e]
            ax.errorbar(dc.engine_mean, dc.human_elo, yerr=[dc.human_elo - dc.lo, dc.hi - dc.human_elo],
                        fmt=st.marker, ms=4, color=st.colour, mec="white", mew=0.6, elinewidth=1.0,
                        capsize=0, zorder=4)
            ax.text(0.05, 0.95, f"Slope (SE) = {r.gamma:.2f} ({r.gamma_se:.2f})",
                    transform=ax.transAxes, ha="left", va="top", fontsize=7.5, fontweight="bold")
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            ax.xaxis.set_major_locator(MultipleLocator(25))
            ax.yaxis.set_major_locator(MultipleLocator(25))
            if i == 0:
                ax.set_title(SET_LABELS[s], fontsize=9, fontweight="bold")
            ax.set_xlabel(f"{ENGINES[e].label} Self-play {METRICS['delta_elo'].bold_symbol}")
            if j == 0:
                ax.set_ylabel(f"Lichess {METRICS['delta_elo'].bold_symbol}")
            bold_tick_labels(ax)
    return save_fig(fig, "lichess960_human_vs_selfplay", out_dir)


def fig_bands(bands: dict, out_dir: Path) -> list[Path]:
    """White's advantage at equal ratings (left) and the slope on Stockfish
    self-play Delta_Elo (right) by rating band; blitz filled circles,
    rapid open squares; 95% CIs."""
    plt = apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"wspace": 0.3})
    xs = np.arange(len(BAND_LABELS))
    for k, s in enumerate(SETS):
        b = bands[s]
        st = SET_STYLE[s]
        off = (k - 0.5) * 0.18
        for ax, c in zip(axes, ("advantage", "gamma")):
            est = b[f"{c}_elo"] if c == "advantage" else b.gamma
            ax.errorbar(xs + off, est, yerr=[est - b[f"{c}_lo"], b[f"{c}_hi"] - est], fmt=st["marker"],
                        color=st["color"], mfc=st["mfc"], ms=5, elinewidth=1.0, capsize=0)
    axes[0].axhline(0, color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)))
    axes[1].axhline(0, color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)))
    axes[1].axhline(1, color=MUTED_TEXT, lw=0.8, ls=(0, (4, 2)))
    for ax in axes:
        ax.set_xticks(xs, BAND_LABELS)
        ax.set_xlim(-0.5, len(BAND_LABELS) - 0.5)
        ax.set_xlabel("Mean Rating")
        bold_tick_labels(ax)
    axes[0].set_ylabel(f"White Advantage {METRICS['delta_elo'].bold_symbol}")
    axes[1].set_ylabel("Slope on Stockfish Self-play")
    return save_fig(fig, "lichess960_rating_bands", out_dir)


def fig_calibration(cal: dict, out_dir: Path) -> list[Path]:
    """Observed White score by rating-gap bin with the fitted curve (solid)
    and the 400-point Elo curve (dotted)."""
    plt = apply_style()
    fig, axes = plt.subplots(1, len(SETS), figsize=(7.0, 3.0), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, s in zip(axes, SETS):
        c = cal[s]
        ax.plot(c.gap_mean, c.elo_curve, color=MUTED_TEXT, lw=0.9, ls=(0, (1, 2)))
        ax.plot(c.gap_mean, c.fitted, color="#111111", lw=1.1)
        ax.errorbar(c.gap_mean, c.observed, yerr=1.96 * c.se, fmt="o", ms=3.5, color="#111111",
                    elinewidth=0.8, capsize=0)
        ax.set_title(SET_LABELS[s], fontsize=9, fontweight="bold")
        ax.set_xlabel("Rating Gap (White - Black)")
        bold_tick_labels(ax)
    axes[0].set_ylabel(f"White {METRICS['escore'].bold_symbol}")
    return save_fig(fig, "lichess960_rating_calibration", out_dir)


def fig_ranked(positions: dict, out_dir: Path) -> list[Path]:
    """Positions ranked by shrunken human Delta_Elo with 95% intervals; #518 black."""
    plt = apply_style()
    fig, axes = plt.subplots(1, len(SETS), figsize=(7.0, 3.0), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, s in zip(axes, SETS):
        g = positions[s].sort_values("shrunk_elo", ascending=False).reset_index(drop=True)
        rank = np.arange(1, len(g) + 1)
        ax.axhline(0, color=MUTED_TEXT, lw=0.8, ls=(0, (1, 2)), zorder=0)
        ax.vlines(rank, g.shrunk_lo, g.shrunk_hi, color=SET_STYLE[s]["color"], alpha=0.25, lw=0.6)
        ax.plot(rank, g.shrunk_elo, color=SET_STYLE[s]["color"], lw=1.4)
        k = int(g.index[g.position_id == STANDARD_POSITION_ID][0])
        ax.vlines(k + 1, g.loc[k, "shrunk_lo"], g.loc[k, "shrunk_hi"], color="#111111", lw=1.2, zorder=3)
        ax.plot(k + 1, g.loc[k, "shrunk_elo"], "o", color="#111111", ms=4, zorder=4)
        ax.set_xlim(0, 961)
        ax.set_title(SET_LABELS[s], fontsize=9, fontweight="bold")
        ax.set_xlabel("Position Rankings")
        bold_tick_labels(ax)
    axes[0].set_ylabel(f"Lichess {METRICS['delta_elo'].bold_symbol}")
    return save_fig(fig, "lichess960_ranked", out_dir)


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------

def summary_text(res: dict) -> str:
    L = ["Lichess Chess960 human play: analysis summary", "=" * 46]
    for s in SETS:
        r = res[s]
        rm, desc = r["rating"], r["desc"]
        null = r["meta"][r["meta"].model == "null"].iloc[0]
        a = desc[desc.band == "all"].iloc[0]
        L += ["", f"{SET_LABELS[s]} ({len(r['data']):,} games, {a.players:,} players)",
              "-" * 40,
              f"W/D/L {100 * a.white_wins:.1f} / {100 * a.draws:.1f} / {100 * a.black_wins:.1f}%; "
              f"E_score {a.escore:.4f}; raw Delta_Elo {a.delta_elo_raw:+.1f}",
              f"Rating model: White advantage at equal ratings {rm['advantage_elo']:+.1f} "
              f"[{rm['advantage_lo']:+.1f}, {rm['advantage_hi']:+.1f}] Elo; rating-gap coefficient "
              f"beta {rm['beta']:.3f} [{rm['beta_lo']:.3f}, {rm['beta_hi']:.3f}] (1 = Elo scale)",
              f"Positions: mean SE {null.mean_se:.1f} Elo; tau {null.tau:.1f} [{null.tau_lo:.1f}, "
              f"{null.tau_hi:.1f}] Elo (LRT {null.lrt_tau:.2f}, p {null.p_tau:.3g}); "
              f"human reliability {null.human_reliability:.2f}"]
        for _, m in r["meta"][r["meta"].model == "slope"].iterrows():
            L.append(f"  on {m.predictor}: slope {m.gamma:.3f} [{m.gamma_lo:.3f}, {m.gamma_hi:.3f}] "
                     f"(p {m.p_gamma:.3g}); corrected {m.gamma_corrected:.3f}; "
                     f"intercept at engine 0 {m.intercept_at_engine_zero:+.1f}; residual tau "
                     f"{m.tau:.1f}; tau^2 explained {m.tau2_explained:.2f}; rho {m.spearman_rho:.3f}")
        p518 = r["positions"].set_index("position_id").loc[STANDARD_POSITION_ID]
        L.append(f"#518: {p518.games:,} games, raw {p518.human_elo:+.1f} (SE {p518.human_se:.1f}); "
                 f"shrunk {p518.shrunk_elo:+.1f} [{p518.shrunk_lo:+.1f}, {p518.shrunk_hi:+.1f}]; "
                 f"rank {p518.rank_shrunk}")
        sep = r["positions"]
        L.append(f"Shrunk intervals entirely above 0: {(sep.shrunk_lo > 0).sum()}; below 0: "
                 f"{(sep.shrunk_hi < 0).sum()}")
        L.append("Player clustering: median SE ratio White-clustered "
                 f"{np.nanmedian(sep.se_ratio_white_cluster):.3f}, Black-clustered "
                 f"{np.nanmedian(sep.se_ratio_black_cluster):.3f}")
        L.append("Game-level model (Stockfish, position-clustered SEs):")
        for _, t in r["inter"].iterrows():
            L.append(f"  {t.term}: {t.estimate:+.3f} [{t.lo:+.3f}, {t.hi:+.3f}] (p {t.p:.3g})")
        L.append("By rating band: advantage Elo [CI]; slope [CI]")
        for _, b in r["bands"].iterrows():
            L.append(f"  {b.band}: n {b.games:,}; {b.advantage_elo:+.1f} [{b.advantage_lo:+.1f}, "
                     f"{b.advantage_hi:+.1f}]; {b.gamma:.3f} [{b.gamma_lo:.3f}, {b.gamma_hi:.3f}]")
    return "\n".join(L) + "\n"


def analyse(prepared: Path = PREPARED, tables: Path = OUT_TABLES, figures: Path = OUT_FIGURES,
            engines_csv: Path = SELFPLAY_POSTERIORS) -> dict:
    eng = load_engines(engines_csv)
    tables.mkdir(parents=True, exist_ok=True)
    res = {}
    for s in SETS:
        d = pd.read_parquet(prepared / f"lichess960_{s}.parquet")
        log.info("%s: %d games", s, len(d))
        rm = rating_model(d)
        pos = stage1(d, rm["_b"][1])
        pos, meta, dec = stage2(pos, eng)
        inter, bands = game_level(d, eng)
        res[s] = {"data": d, "rating": rm, "desc": descriptives(d), "cal": calibration(d, rm["_b"]),
                  "positions": pos, "meta": meta, "deciles": dec, "inter": inter, "bands": bands}
    def stack(key):
        return pd.concat([res[s][key].assign(set=s) for s in SETS], ignore_index=True)
    stack("desc").to_csv(tables / "lichess960_descriptives.csv", index=False)
    stack("cal").to_csv(tables / "lichess960_rating_calibration.csv", index=False)
    pd.DataFrame([{k: v for k, v in res[s]["rating"].items() if not k.startswith("_")} | {"set": s}
                  for s in SETS]).to_csv(tables / "lichess960_rating_model.csv", index=False)
    stack("positions").to_csv(tables / "lichess960_positions.csv", index=False)
    stack("meta").to_csv(tables / "lichess960_meta_regression.csv", index=False)
    stack("deciles").to_csv(tables / "lichess960_deciles.csv", index=False)
    stack("inter").to_csv(tables / "lichess960_interaction.csv", index=False)
    stack("bands").to_csv(tables / "lichess960_rating_bands.csv", index=False)
    text = summary_text(res)
    (tables / "lichess960_summary.txt").write_text(text)
    print(text)
    pick = lambda key: {s: res[s][key] for s in SETS}
    fig_human_vs_selfplay(pick("positions"), pick("meta"), pick("deciles"), figures)
    fig_bands(pick("bands"), figures)
    fig_calibration(pick("cal"), figures)
    fig_ranked(pick("positions"), figures)
    return res


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    for name, h in (("prepare", "filter the monthly tables into blitz and rapid sets"),
                    ("analyse", "models, tables and figures"), ("all", "prepare then analyse")):
        sub.add_parser(name, parents=[common], help=h)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    if args.command in ("prepare", "all"):
        prepare()
    if args.command in ("analyse", "all"):
        analyse()
    return 0


if __name__ == "__main__":
    sys.exit(main())
