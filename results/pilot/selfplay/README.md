# Self-play pilot

Stockfish and Caissa self-play on the 30 pilot positions
(`config/pilot_positions.txt`), 500 games each (50 book lines x 10 games),
10+0.1, played 2026-10-22 on the Xeon 6240R nodes with 16 games at a time.

Same settings as the main run (`config/selfplay.toml`) except that there was
**no time margin**: a move that overran the clock by any amount lost on time.
Scheduling delays on a busy node cost Caissa 65 games on time (37 of them in
position 722) and Stockfish 2, all by 1-98 ms. The main run allows a 100 ms
margin, so these games are kept here as the pilot and the main run replays
all 960 positions, including these 30.

Pilot findings (used to settle the main run's design):

- Self-play E against each engine's Part I evaluation of the same position:
  Pearson 0.95 (Stockfish) and 0.88 (Caissa), slope 0.99.
- Draw rates 66-98% (Stockfish) and 60-91% (Caissa), falling as positions
  become less balanced.
- Games sharing a book line are correlated (ICC 0.08-0.10, design effect
  1.7-1.9), so the analysis treats line as a cluster.
- 876 s (Stockfish) and 943 s (Caissa) per position on 16 cores.

Rebuild the tables with
`python code/selfplay.py collect --output-dir results/pilot/selfplay`.
