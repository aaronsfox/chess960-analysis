# Chess960 fairness

How fair are the 960 Chess960 starting positions? This project measures each
position's White advantage with several chess engines, separates real
differences between positions from measurement noise, and tests whether those
differences show up in game outcomes.

- **Aim 1 — starting-position evaluation** (in progress). Evaluate all 960
  positions with four engines across a ladder of node budgets. Primary
  variables: expected score, its Elo-equivalent, and convergence with search
  effort. Secondary: agreement and disagreement between engines.
- **Aim 2 — engine self-play outcomes.** Planned.
- **Aim 3 — practical fairness for human play and tournaments.** Planned.

## Repository layout

```
engines/            one folder per engine: manifest, install notes, bin/ (not committed)
  stockfish/  lc0/  caissa/  plentychess/
  engines.lock.json SHA-256 of every installed engine binary and network
config/             study settings (node budgets, output paths, position lists)
code/
  utils.py          shared helpers: positions, engine setup, metrics, IO
  setup_engines.py  check, fingerprint and smoke-test the engines
  aim1_evaluate.py  Aim 1: run evaluations and collect results
jobs/               SLURM job scripts for the cluster
tests/              unit tests (pytest)
results/            outputs (not committed)
```

## Design choices that keep Aim 1 deterministic

- Every search is **single-threaded** and limited by **node count**, not time,
  so the result doesn't depend on hardware speed and a rerun gives the same answer.
- Each node budget is a **separate, fresh search** (`ucinewgame` first).
- All engines, including Lc0, run on **CPU**.
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

Then install each engine as described in `engines/<name>/README.md`, and:

```bash
python code/setup_engines.py check     # all engines present and configured?
python code/setup_engines.py smoke     # legal moves on all 960 positions?
python code/setup_engines.py record    # write hashes; commit engines.lock.json
```

## Running Aim 1

```bash
# A quick test on three positions with small budgets
python code/aim1_evaluate.py run --engine stockfish --positions 80,518,644 \
    --budgets 10000 100000 --allow-unlocked --output-dir results/test

# Full ladder for the pilot positions
python code/aim1_evaluate.py run --engine stockfish --positions @config/pilot_positions.txt

# MultiPV = 5 at the top budget
python code/aim1_evaluate.py run --engine stockfish --multipv 5

# Combine everything into tables in results/aim1/tables/
python code/aim1_evaluate.py collect
```

`run` writes one Parquet file per position under `results/aim1/raw/` and skips
positions already done, so an interrupted run resumes where it stopped.
`collect` writes:

| File | Contents |
| --- | --- |
| `aim1_info_lines.parquet` | Every UCI info line from every search |
| `aim1_final.parquet` / `.csv` | One row per search and principal variation: final score, WDL, expected score, Elo-equivalent |
| `aim1_convergence.csv` | Per engine and position: top-budget values, total variation and tail drift across budgets |

On the cluster (create `logs/` first):

```bash
mkdir -p logs
sbatch --export=ALL,ENGINE=stockfish jobs/aim1_array.slurm
```

## Positions

Positions use Scharnagl numbering 0–959, with #518 the standard starting
position (RNBQKBNR). No position is its own mirror image, so the 960 form 480
mirror pairs; `utils.mirror_id()` gives a position's partner.
