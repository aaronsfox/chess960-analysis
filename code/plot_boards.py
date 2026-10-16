"""Board diagrams of Chess960 starting positions for figures.

Draws a grid of labelled boards (file and rank coordinates, one caption per
board), optionally with the engine's best move as an arrow and its evaluation
in the caption, read from the evaluate_sp tables.

Examples
--------
    python code/plot_boards.py --positions 518,534,868,94 --out results/figures/boards
    python code/plot_boards.py --positions 518,868 --engine stockfish --cols 2 \\
        --labels "Standard chess" "Most unbalanced (Stockfish)"

Writes <out>.pdf (for the manuscript) and <out>.png (for viewing).
"""

from __future__ import annotations

import argparse
import io
import string
import sys
from pathlib import Path

import chess
import chess.svg
import pandas as pd

from utils import REPO_ROOT, elo_from_expected, get_position, parse_position_spec

DEFAULT_TABLE = REPO_ROOT / "results" / "evaluate_sp" / "tables" / "evaluate_sp_final.parquet"

# Muted print-friendly squares; the arrow is the only strong colour.
BOARD_COLORS = {
    "square light": "#ece6d8",
    "square dark": "#b4a58a",
    "margin": "#ffffff",
    "coord": "#333333",
}
ARROW_COLOR = "#1f6f5fcc"


def board_png(board: chess.Board, arrow: chess.Move | None, size: int, dpi_scale: float) -> bytes:
    import cairosvg

    arrows = []
    if arrow is not None:
        to_sq = arrow.to_square
        # Chess960 castling is stored as king-takes-rook; draw it to the king's
        # destination square instead so the arrow reads as castling.
        if board.is_castling(arrow):
            kingside = board.is_kingside_castling(arrow)
            to_sq = chess.square(6 if kingside else 2, chess.square_rank(arrow.from_square))
        arrows.append(chess.svg.Arrow(arrow.from_square, to_sq, color=ARROW_COLOR))
    svg = chess.svg.board(board, size=size, coordinates=True, arrows=arrows, colors=BOARD_COLORS)
    return cairosvg.svg2png(bytestring=svg.encode(), scale=dpi_scale)


def ordered_positions(spec: str) -> list[int]:
    """Position ids in the order given (parse_position_spec sorts them).
    Accepts "518,868,94", ranges like "0-3", or "@file"."""
    if spec.startswith("@"):
        return parse_position_spec(spec)
    ids: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if token:
            ids.extend(parse_position_spec(token))
    return ids


def evaluation_lookup(table: Path, engine: str) -> pd.DataFrame | None:
    if not table.exists():
        return None
    f = pd.read_parquet(table)
    f = f[(f.engine == engine) & (f.multipv_setting == 1) & (f.multipv_index == 1)]
    if f.empty:
        return None
    top = f.sort_values("budget_nodes").groupby("position_id").tail(1).set_index("position_id")
    top["rank"] = top["expected_score"].rank(ascending=False, method="min").astype(int)
    return top


def caption_for(pos_id: int, label: str | None, ev: pd.DataFrame | None, engine: str) -> str:
    pos = get_position(pos_id)
    head = f"#{pos_id}  {pos.arrangement}"
    if label:
        head += f"\n{label}"
    if ev is not None and pos_id in ev.index:
        r = ev.loc[pos_id]
        head += (f"\n{engine.capitalize()}: {int(r.cp_white):+d} cp, E = {r.expected_score:.3f}"
                 f" (rank {int(r['rank'])}/960)")
    return head


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--positions", required=True, help="e.g. 518,868,94 or @file")
    p.add_argument("--labels", nargs="*", help="one short label per position (optional)")
    p.add_argument("--engine", default="stockfish", help="engine for arrow and caption ('none' to omit)")
    p.add_argument("--table", default=str(DEFAULT_TABLE), help="evaluate_sp_final.parquet")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "figures" / "boards"))
    p.add_argument("--no-arrows", action="store_true")
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import image as mpimg

    ids = ordered_positions(args.positions)
    labels = args.labels or [None] * len(ids)
    if len(labels) != len(ids):
        p.error("--labels needs one label per position")
    ev = None if args.engine == "none" else evaluation_lookup(Path(args.table), args.engine)

    cols = min(args.cols, len(ids))
    rows = -(-len(ids) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.25 * rows), squeeze=False)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= len(ids):
            continue
        pos_id = ids[i]
        board = get_position(pos_id).board()
        arrow = None
        if ev is not None and not args.no_arrows and pos_id in ev.index:
            arrow = chess.Move.from_uci(ev.loc[pos_id, "bestmove"])
        png = board_png(board, arrow, size=400, dpi_scale=2.0)
        ax.imshow(mpimg.imread(io.BytesIO(png), format="png"))
        letter = string.ascii_lowercase[i]
        ax.set_title(f"({letter})", loc="left", fontsize=9, fontweight="bold", pad=2)
        ax.text(0.5, -0.03, caption_for(pos_id, labels[i], ev, args.engine),
                transform=ax.transAxes, ha="center", va="top", fontsize=7.2, linespacing=1.35)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.08, wspace=0.06, hspace=0.42)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
