# Chess960 fairness

How fair are the 960 Chess960 starting positions? This project measures each
position's White advantage with several chess engines, separates real
differences between positions from measurement noise, and tests whether those
differences show up in game outcomes.

The project has three parts:

- **Evaluating starting positions** (`code/evaluate_sp.py`, in progress).
  Evaluate all 960 positions with four engines across a ladder of node
  budgets. Primary variables: expected score, its Elo-equivalent, and
  convergence with search effort. Secondary: agreement and disagreement
  between engines.
- **Self-play outcomes** (planned). Test whether evaluated differences between
  positions appear in the results of engine self-play games, and how they
  change with engine strength.
- **Fairness in human play and tournaments** (planned). Compare the engine
  results with human games, and assess what position differences mean for
  tournament formats.

## Repository layout

```
engines/            one folder per engine: manifest, install notes, Linux build
                    script, bin/ (not committed)
  stockfish/  lc0/  caissa/  plentychess/
  engines.lock.json SHA-256 of every installed engine binary and network
config/             study settings (node budgets, output paths, position lists)
code/
  utils.py          shared helpers: positions, engine setup, metrics, IO
  setup_engines.py  check, fingerprint and smoke-test the engines
  evaluate_sp.py    evaluate starting positions and collect results
jobs/               SLURM job scripts (CPU engines; Lc0 on GPU)
tests/              unit tests (pytest)
results/            outputs (not committed)
```

## Design choices that keep the evaluations deterministic

- Every search is **single-threaded** and limited by **node count**, not time,
  so the result doesn't depend on hardware speed and a rerun gives the same answer.
- Each node budget is a **separate, fresh search** (`ucinewgame` first).
- Stockfish, Caissa and PlentyChess run on **CPU**; Lc0 runs on an
  **NVIDIA GPU** (fp32 CUDA backend), since it is far too slow on one CPU core.
  Node limits mean the hardware changes speed, not what a search covers.
- Engine files are **hash-checked** against `engines/engines.lock.json` before
  every production run, and each result row records the engine hash,
  platform, host and CPU.
- Develop and test on Windows; produce all final data on the Linux cluster, so
  builds from different platforms never mix in the results.

## Setup

Requires Python 3.11+. Use one of these.

**Conda** (`environment.yml`):

```bash
conda env create -f environment.yml
conda activate chess960
python -m pytest             # run the tests
```

**uv** (`pyproject.toml` and `uv.lock`, exact pinned versions):

```bash
uv sync                      # creates .venv with the locked dependencies
uv run python -m pytest
```

**pip**: `python -m venv .venv`, activate it, then
`pip install chess numpy pandas pyarrow pytest`.

On the cluster, use the same method as on your PC so both machines run the
same package versions.

Then install each engine as described in `engines/<name>/README.md`. On the
Linux cluster, each engine folder has a `build_linux.sh`; `engines/README.md`
lists them and their settings. Then:

```bash
python code/setup_engines.py check     # all engines present and configured?
python code/setup_engines.py smoke     # legal moves on all 960 positions?
python code/setup_engines.py record    # write hashes; commit engines.lock.json
```

## Evaluating starting positions (evaluate_sp.py)

```bash
# A quick test on three positions with small budgets
python code/evaluate_sp.py run --engine stockfish --positions 80,518,644 \
    --budgets 10000 100000 --allow-unlocked --output-dir results/test

# Full ladder for the pilot positions
python code/evaluate_sp.py run --engine stockfish --positions @config/pilot_positions.txt

# MultiPV = 5 at the top budget
python code/evaluate_sp.py run --engine stockfish --multipv 5

# Combine everything into tables in results/evaluate_sp/tables/
python code/evaluate_sp.py collect
```

`run` writes one Parquet file per position under `results/evaluate_sp/raw/` and skips
positions already done, so an interrupted run resumes where it stopped.
`collect` writes:

| File | Contents |
| --- | --- |
| `evaluate_sp_info_lines.parquet` | Every UCI info line from every search |
| `evaluate_sp_final.parquet` / `.csv` | One row per search and principal variation: the evaluation (score, WDL, expected score, Elo-equivalent), best move, and the search's last line in `last_*` columns |
| `evaluate_sp_convergence.csv` | Per engine and position: top-budget values, total variation and tail drift across budgets |

A search's evaluation is its last line with an **exact** score. When the node
limit stops a search partway through a depth, the last line is often an
aspiration-window bound (the true score is only known to be above or below
it), so it isn't used. `eval_source` is `last_exact` normally, or
`last_line_bound` if a search printed no exact line. The best move is always
the engine's own `bestmove`.

On the cluster (create `logs/` first):

```bash
mkdir -p logs
sbatch --export=ALL,ENGINE=stockfish jobs/evaluate_sp_array.slurm   # CPU engines
sbatch jobs/evaluate_sp_lc0_gpu.slurm                               # Lc0: 4 processes per GPU, 24 short tasks
```

Both scripts activate the `chess960` conda environment; add modules the
engines need at run time with `--export=ALL,ENGINE=...,MODULES="..."`.

## Self-play opening book (opening_book_generator.py)

Self-play games start from short opening lines so that games vary, as in
real play, where players choose between several good moves. The book holds
50 lines of 6 plies (three moves each) for every starting position, built
with Stockfish and shared by every self-play engine, so differences between
engines can't come from the openings. Settings are in
`config/opening_book.toml`.

How a position's lines are built:

1. At each ply, a MultiPV search lists the moves within 50 cp of the best
   move (the classic engine-analysis threshold for an inaccuracy, so no book
   move is even an inaccuracy). Lines are built by choosing uniformly among
   them (seeded random walks), so they spread over different first moves, as
   different players choose different openings.
2. Both sides face the same limits: each side's total loss over its three
   moves is capped at 100 cp, and White's and Black's totals differ by at
   most 50 cp.
3. Every candidate line's final position, and the starting position, are
   re-evaluated with a deeper search. Lines whose evaluation drifts more than
   50 cp from the starting position's are dropped.
4. The book takes lines from different first moves in turn, each time
   choosing the line that keeps the average drift in expected score (from
   Stockfish's WDL, the scale game results are on) closest to zero. Positions
   with fewer good lines than needed repeat lines, so every position has
   exactly 50.

Searches are single-threaded and node-limited and the random choices are
seeded per position, so the same engine build regenerates the identical book.

```bash
python code/opening_book_generator.py build --positions 518 --output-dir results/test_book
python code/opening_book_generator.py collect --output-dir results/test_book
```

On the cluster: `sbatch jobs/opening_book_array.slurm`, then `collect`. For the
30 pilot positions (`config/pilot_positions.txt`), one per task:
`sbatch --array=0-29 --export=ALL,POSITIONS=@config/pilot_positions.txt,POSITIONS_PER_TASK=1 jobs/opening_book_array.slurm`.
`collect` writes to `results/opening_book/book/`:

| File | Contents |
| --- | --- |
| `opening_book_lines.parquet` / `.csv` | Every line: moves (UCI and SAN), final position, each side's loss, and the evaluation shift (shallow and deeper check) |
| `opening_book_summary.csv` | Per position: distinct lines, first moves, mean shift, mean loss per side |

`build` also keeps every candidate line it checked (`raw/<engine>/pool/`),
with its evaluations and whether it was selected, so the selection can be
examined or redone without repeating any search.
| `opening_book.pgn` | The lines as games from each starting position, for fastchess |
| `opening_book.epd` | The lines' final positions, one per line |

## Figures

`code/plot_boards.py` draws labelled board diagrams of starting positions
(coordinates, a caption per board, and optionally an engine's best move as an
arrow with its evaluation and rank from the evaluate_sp tables):

```bash
python code/plot_boards.py --positions 518,534,868,94 \
    --labels "Standard chess" "Mirror of #518" "Most unbalanced" "Mirror of #868" \
    --out results/figures/boards
```

It writes a PDF for the manuscript and a PNG for viewing; positions are drawn
in the order given. The default style is black and white like chess books
(hatched dark squares, coordinates on the left and bottom); `--style colour`
gives tan squares instead. `--no-arrows` and `--no-letters` drop the move arrows
and the (a), (b), ... panel letters. Use `--engine caissa` (or lc0, plentychess) for another
engine's move and evaluation, or `--engine none` for plain boards.

## Positions

Positions use Scharnagl numbering 0–959, with #518 the standard starting
position (RNBQKBNR). No position is its own mirror image, so the 960 form 480
mirror pairs; `utils.mirror_id()` gives a position's partner.
