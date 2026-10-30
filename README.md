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
  opening_book_generator.py  build the self-play opening book
  selfplay.py       engine self-play games with fastchess, and their tables
  analyse_selfplay.py  self-play analysis: balance, agreement, decisiveness
  analyse_sp.py     starting-position analysis: balance, stability, agreement
  plot_style.py     shared figure style: fonts, engine colours/markers/lines, metric labels
  plot_boards.py    board diagrams for figures
tools/fastchess/    build script for fastchess (the match runner); bin/ not committed
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
   Stockfish's WDL, the scale game results are on) closest to zero, then
   swaps lines in and out while that brings the average closer to zero,
   keeping every first move represented. Positions
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
| `opening_book.pgn` | The lines as games from each starting position, for fastchess |
| `opening_book.epd` | The lines' final positions, one per line |

`build` also keeps every candidate line it checked (`raw/<engine>/pool/`),
with its evaluations and whether it was selected, so the selection can be
examined or redone without repeating any search: `reselect` rebuilds every
position's selection from these saved lines in about a minute, then run
`collect` again.

## Self-play (selfplay.py)

Each engine plays itself from every starting position at 10+0.1, single
threaded with a 64 MB hash. Every game starts from one of the position's 50
book lines, and each line is played 10 times with the colours swapped in
pairs, so every position gets 500 games. Two copies of the same binary play
(`<engine>_a` and `<engine>_b`). Games use fastchess's draw adjudication
(from move 40, 8 moves within 10 cp of 0) and two-sided resignation (5 moves
beyond 10 pawns), and a move may overrun the clock by up to 100 ms before it
loses on time. All settings are in `config/selfplay.toml`. The 30-position
pilot (no time margin) is kept separately in `results/pilot/selfplay/`.

```bash
bash tools/fastchess/build_linux.sh          # once, on a login node
python code/selfplay.py run --engine stockfish --positions 518 --concurrency 4 \
    --output-dir results/test_selfplay
python code/selfplay.py collect --output-dir results/test_selfplay
```

`run` plays one fastchess match per position and writes it, gzipped, to
`raw/<engine>/spNNN.pgn.gz` only once all 500 games are done, so an
interrupted job loses at most the position in progress; rerunning skips
finished positions. Each run's settings, engine and fastchess hashes and the
book's hash go to `runs/<id>.json`.

On the cluster: `sbatch --export=ALL,ENGINE=stockfish jobs/selfplay_array.slurm`
(16 cores per task, one game per physical core), likewise for `caissa`, then
`collect`. Pilot on the 30 pilot positions, one per task:
`sbatch --array=0-29 --time=01:00:00 --export=ALL,ENGINE=stockfish,POSITIONS=@config/pilot_positions.txt,POSITIONS_PER_TASK=1 jobs/selfplay_array.slurm`.

`collect` writes to `results/selfplay/tables/`:

| File | Contents |
| --- | --- |
| `selfplay_games.parquet` | One row per game: position, book line, result (White's score), termination and reason, length, first engine evaluation, mean depth/nodes and time used per side |
| `selfplay_positions.parquet` / `.csv` | Per engine and position: W/D/L, draw rate, White's expected score with its standard error, Elo-equivalent advantage |
| `selfplay_lines.parquet` | The same per book line (for cluster-robust analyses: games sharing a line are not independent) |

The PGNs, tables and run records are all committed. The PGNs come to about
0.7 GB per engine (one ~0.7 MB file per position), so push one engine at a
time: GitHub refuses a single push over 2 GB.

## Starting-position analysis (analyse_sp.py)

```bash
python code/analyse_sp.py
```

Reads `results/evaluate_sp/tables/evaluate_sp_final.parquet` and answers the
three questions about the starting positions, using White's expected score
($E_{\text{score}}$) and its Elo-equivalent advantage ($\Delta_{\text{Elo}}$):

1. **Balance:** $E_{\text{score}}$ and $\Delta_{\text{Elo}}$ at each engine's top
   budget for Stockfish, Caissa and Lc0: distribution, ranking, practically
   balanced positions ($\Delta_{\text{Elo}}$ within ±5) and where #518 sits.
2. **Stability:** the change in $E_{\text{score}}$ and $\Delta_{\text{Elo}}$
   between successive tenfold budgets. Convergence means the changes shrink
   as the budget rises; a step is stable if it changes $\Delta_{\text{Elo}}$ by
   at most 5. Each position's stabilisation budget is the lowest budget from
   which every further step is stable (none if the final step is not).
3. **Agreement:** Spearman correlations between all four engines' top-budget
   centipawn evaluations, with 2,000-resample bootstrap intervals (PlentyChess
   has no win/draw/loss output, so it appears only here).

Tables go to `results/evaluate_sp/analysis/` (`evaluate_sp_*.csv`, plus
`evaluate_sp_summary.txt` with the key numbers and
`evaluate_sp_agreement_matrix.csv`, a lower-triangle table of ρ [95% CI]) and figures, one plot per
file as PDF and PNG, to `results/figures/`:

| Figure | Versions |
| --- | --- |
| `evaluate_sp_balance_ranked_<metric>` | advantage against rank, #518 marked |
| `evaluate_sp_balance_distribution_<metric>` | kernel density across positions, truncated to the observed range, #518 marked |
| `evaluate_sp_stability_step_change_<metric>` | median absolute change per tenfold step (IQR shaded) |
| `evaluate_sp_stability_budgets_<metric>` | median advantage at each budget (IQR shaded) |
| `evaluate_sp_stability_stabilised_delta_elo` | positions stable from each budget on |
| `evaluate_sp_agreement_spearman` | Spearman matrix with 95% intervals |
| `evaluate_sp_agreement_scatter_matrix` | lower-triangle grid of all six centipawn scatter plots, each with ρ [95% CI]; replaces the Spearman table |
| `evaluate_sp_agreement_scatter_<a>_<b>` | top-budget centipawns of engine a against b (all six pairs), #518 ringed, line of equality |

`<metric>` is `escore` or `delta_elo`. Stabilisation is defined on
$\Delta_{\text{Elo}}$ and Spearman correlations on ranks, so those two
figures have one version each.

Figures have no legends (engines and markers are explained in the captions). All figures take their style from `code/plot_style.py`: Arial (Liberation
Sans where Arial isn't installed), bold axis labels (7 pt) and tick labels (8 pt), inward ticks,
no gridlines, the metric labels, and each engine's colour, marker and line
style (Stockfish #104485, circle, solid; Caissa #f17f03, square, dashed;
Lc0 #418c47, triangle, dotted; PlentyChess #b63667, diamond, dash-dot).

## Self-play analysis (analyse_selfplay.py)

```bash
python code/analyse_selfplay.py
```

Reads the `selfplay.py collect` tables and the starting-position evaluations
and follows the self-play analysis plan for Stockfish and Caissa:

- **Quality checks and design effect:** games per position, line matching,
  losses on time, colour balance of the two engine copies, book drift, and
  the intraclass correlation (ICC) of games within book lines, giving the
  design effect DEFF = 1 + 9·ICC per engine.
- **Dirichlet posteriors:** each position's win/draw/loss counts get a
  Dirichlet(½ + (W, D, L)/DEFF) posterior. Monte Carlo draws give
  $E_{\text{score}}$ and $\Delta_{\text{Elo}}$ with 95% intervals,
  P(favours White), P(practically balanced), P(favours Black), decisive rate,
  White's share of decisive games, and comparisons with standard chess (#518).
  A Bayesian bootstrap over book lines is the sensitivity check.
- **1. Balance:** self-play outcomes regressed on each engine's top-budget
  evaluation (Pearson r also corrected for self-play noise), and the posterior
  summaries. Counts are given at probability thresholds 0.80 and 0.95.
- **2. Agreement** between Stockfish and Caissa games: Spearman's ρ (observed
  and corrected for noise), Bland–Altman, level and spread, Cohen's κ on the
  categories, overlap of the most unbalanced positions.
- **3. Decisiveness:** decisive rate against |$\Delta_{\text{Elo}}$| (logistic
  regression weighted by effective sample size), White's share of decisive
  games, and how games end.
- **Ranking, not selection:** `selfplay_position_table.csv` lists all 960
  positions with both engines' estimates, intervals and probabilities, ranked
  by advantage, fairness (|$\Delta_{\text{Elo}}$|), decisiveness and by
  P(at least as fair as and more decisive than #518).

Main-text figures: `selfplay_balance_vs_evaluation` (self-play against
evaluation $\Delta_{\text{Elo}}$, Stockfish and Caissa side by side, with the line of
equality, least-squares line and statistics) and `selfplay_balance_ranked`
(positions ranked by posterior $\Delta_{\text{Elo}}$ with each 95% credible interval,
both engines side by side) and `selfplay_agreement` (Caissa against Stockfish with
Spearman's ρ [95% CI], beside a Bland–Altman plot). Ternary (win/draw/loss) figure `selfplay_ternary` (2 x 2): the top row places
every position's posterior mean in the full triangle (corners Black wins,
White wins, draw), Stockfish left and Caissa right; the horizontal position
is White's expected score and the height the draw rate, so the practically
balanced band is a vertical strip and the grey lines mark draw rates. The
bottom row enlarges the dashed box and shows the 50% and 95%
highest-density regions of the posterior for four contrasting positions,
coloured with the four-colour palette: #868 (blue, unbalanced), #518
(orange, standard), #674 (green, fair and decisive) and #226 (magenta, fair
and drawish).

Tables go to `results/selfplay/analysis/` (`selfplay_*.csv` and
`selfplay_summary.txt`) and figures to `results/figures/` (`selfplay_*`),
in the `plot_style.py` style.

## Human play: Lichess Chess960 (lichess960.py)

Rated Chess960 games from the monthly Lichess database exports
(<https://database.lichess.org/chess960/>, CC0), parsed to one row per game
with the starting position (0–959), ratings, time control, result,
termination, tournament/berserk flags and hashed player IDs (no usernames are
stored).

```bash
python code/lichess960.py fetch --months 2026-07:2026-09   # streams and parses
python code/lichess960.py summarise                         # counts for scoping
```

- Kept while parsing: rated, Chess960, known result, and estimated duration
  (base + 40 × increment, Lichess's definition) of at least `--min-seconds`
  (default 180 s, i.e. blitz and slower). Categories: bullet < 180 s,
  blitz < 480 s, rapid < 1500 s, classical ≥ 1500 s.
- Output: `results/lichess960/raw/lichess960_YYYY-MM.parquet` (finished
  months are skipped; `--overwrite` redoes them) and
  `results/lichess960/lichess960_summary.txt`.
- Needs internet access, so run it on the login node, or download the
  `.pgn.zst` files with `wget` into `data/lichess/` (ignored by git) and pass
  `--pgn-dir data/lichess`.

## Human play analysis (analyse_lichess960.py)

```bash
python code/analyse_lichess960.py all      # or: prepare, then analyse
```

- `prepare` filters the monthly tables into two sets analysed separately:
  blitz (estimated 300–479 s) and rapid (≥ 480 s). In order, it drops
  bot accounts, abandoned games and rules infractions, games under 10 plies, berserk games,
  games missing a rating, games with a provisional player (|rating change|
  > 30) and games with mean rating below 1200. Writes
  `results/lichess960/prepared/` (regenerated, not committed), the flow
  table, a randomisation check and the rating-change distribution.
- A third set, `titled` (both players titled, 3 min+), gives an online
  master-level check; it is too sparse per position, so only the pooled
  game-level slope is reported for it.
- `analyse` fits, per set: a fractional-logit rating model (White's
  advantage at equal ratings; is the Lichess scale the Elo scale?); each
  position's rating-adjusted human ΔElo; a REML random-effects
  meta-regression across positions on Stockfish (primary) and Caissa
  self-play ΔElo (tau, slope, reliability-corrected slope); and a game-level
  model with a mean-rating interaction plus fits by rating band
  (1200–1499, 1500–1799, 1800–1999, 2000+).
- Tables: `results/lichess960/analysis/lichess960_*`. Figures:
  `lichess960_human_vs_selfplay`, `lichess960_rating_bands`,
  `lichess960_rating_calibration`, `lichess960_ranked`.

## Master-level games (masters960.py)

Over-the-board Chess960 / Freestyle events (Freestyle Grand Slam Tour, grenke
Freestyle Open, FIDE World Fischer Random Championship, Champions Showdown
9LX), as TWIC PGN files listed in `config/masters960_sources.csv`.

```bash
python code/masters960.py download   # into data/masters/ (ignored by git); or save files there by hand
python code/masters960.py parse      # -> results/masters960/masters960_games.csv
python code/masters960.py analyse    # tables, summary and figures
```

Events draw one position per round for every board, so the data are many
games on a few dozen positions (case studies). Armageddon games and forfeits
are excluded; time-control class (classical/rapid/blitz) comes from the event
and, for faster tiebreaks within a file, from the games' own clocks. Models use
the FIDE rating gap as an offset (beta = 1) with position-clustered SEs.
Figures: `masters960_case_positions` and `masters960_levels` (White's
advantage and draw rate from Lichess club level to engines).

## Human play against engine self-play (compare_levels.py)

```bash
python code/compare_levels.py
```

Applies the self-play Dirichlet model to the human games: for each source
(Stockfish and Caissa self-play, Lichess blitz/rapid/titled, master OTB) and
position, Dirichlet(1/2 + counts / DEFF) with DEFF from self-play for the
engines and 1 for humans. Writes `results/compare_levels/` (posteriors per
source and position, per-source summary, Spearman agreement with the engines
for Delta_Elo and the draw rate) and the figures `compare_ternary` (all
sources in the win/draw/loss triangle; HPD regions for #868, #518, #674, #226)
and `compare_ternary_masters` (the eight positions with most master games).
Needs the outputs of analyse_selfplay.py, analyse_lichess960.py and
masters960.py.

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

Boards after opening-book lines (`--lines position:line_index,...`), with the
moves in the caption, e.g. the opening-book example figure:

```bash
python code/plot_boards.py --lines 518:1,518:14,518:43,518:34,868:0,868:3,868:4,868:32 \
    --cols 4 --no-letters --out results/figures/selfplay_book_examples \
    --labels "Sicilian Defence: Closed (B23)" "Queen's Gambit Declined (D30)" \
             "Caro-Kann Defence (B10)" "English: Anglo-Indian, Slav (A15)" "" "" "" ""
```

## Positions

Positions use Scharnagl numbering 0–959, with #518 the standard starting
position (RNBQKBNR). No position is its own mirror image, so the 960 form 480
mirror pairs; `utils.mirror_id()` gives a position's partner.
