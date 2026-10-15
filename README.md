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
sbatch jobs/evaluate_sp_lc0_gpu.slurm                               # Lc0 on a GPU
```

Both scripts activate the `chess960` conda environment; add modules the
engines need at run time with `--export=ALL,ENGINE=...,MODULES="..."`.

## Positions

Positions use Scharnagl numbering 0–959, with #518 the standard starting
position (RNBQKBNR). No position is its own mirror image, so the 960 form 480
mirror pairs; `utils.mirror_id()` gives a position's partner.
