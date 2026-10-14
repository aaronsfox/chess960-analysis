# Engines

Each engine has its own folder:

| Folder | Engine | Search | Training data |
| --- | --- | --- | --- |
| `stockfish/` | Stockfish 19 | Alpha-beta, NNUE | Largely Lc0 self-play data |
| `lc0/` | Lc0 v0.32.1 (CPU) | Neural MCTS | Lc0 self-play |
| `caissa/` | Caissa 1.25 | Alpha-beta, NNUE | Own self-play games |
| `plentychess/` | PlentyChess 7.0.0 | Alpha-beta, NNUE | Own self-generated positions |

In each folder:

- `engine.toml` is the manifest: version, binary path per platform, the UCI
  options every search uses, and any network file.
- `README.md` says how to install that engine on Windows and on the Linux cluster.
- `bin/` holds the binary and `networks/` any network file. Git ignores both,
  so binaries never enter the repository.

`engines.lock.json` records the SHA-256 of every binary and network, per
platform. Production runs refuse to start if the files on disk don't match it,
so every result can be traced to the exact engine build that produced it.

## Adding or upgrading an engine

1. Install the binary as the engine's README describes.
2. Update `version` in `engine.toml`.
3. Run `python code/setup_engines.py check --engine <name>` and
   `python code/setup_engines.py smoke --engine <name>`.
4. Run `python code/setup_engines.py record --engine <name>` and commit the
   updated `engines.lock.json`.

Don't upgrade an engine partway through a production run: results from two
builds would be mixed.
