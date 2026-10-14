# Engines

Each engine has its own folder:

| Folder | Engine | Search | Training data |
| --- | --- | --- | --- |
| `stockfish/` | Stockfish 19 | Alpha-beta, NNUE | Largely Lc0 self-play data |
| `lc0/` | Lc0 v0.32.1 (GPU) | Neural MCTS | Lc0 self-play |
| `caissa/` | Caissa 2.0 | Alpha-beta, NNUE | Own self-play games |
| `plentychess/` | PlentyChess 8.0.0 | Alpha-beta, NNUE | Own self-generated positions |

In each folder:

- `engine.toml` is the manifest: version, binary path per platform, the UCI
  options every search uses, and any network file.
- `README.md` says how to install that engine on Windows and on the Linux cluster.
- `bin/` holds the binary and `networks/` any network file. Git ignores both,
  so binaries never enter the repository.

`engines.lock.json` records the SHA-256 of every binary and network, per
platform. Production runs refuse to start if the files on disk don't match it,
so every result can be traced to the exact engine build that produced it.

## Building on the Linux cluster

Each engine folder has a `build_linux.sh` that downloads the pinned release's
source, builds it and installs the binary in that folder's `bin/`. Run them
from the repository root on a **login node**, since each build downloads
source code and network files:

```bash
module load GCC/10.2.0                    # C++ compiler; your cluster's module name
bash engines/stockfish/build_linux.sh     # only if the universal release binary fails
bash engines/caissa/build_linux.sh        # needed: no Linux release binaries
bash engines/plentychess/build_linux.sh   # needed if the release binary reports GLIBC errors
module load CUDA/12.2.2
LC0_CUDA_CC=80 bash engines/lc0/build_linux.sh   # GPU build for the A100; needs meson and ninja
```

| Engine | Default build | Override with |
| --- | --- | --- |
| Stockfish | `ARCH=x86-64-vnni512` (Cascade Lake), no PGO | `ARCH`, `SF_VERSION`, `PGO=1` |
| Caissa | `ARCH=avx512`, `CC=g++` | `ARCH`, `CAISSA_VERSION`, `CC` |
| PlentyChess | `ARCH=avx512`, `CXX=g++`, no PGO | `ARCH`, `PLENTY_VERSION`, `CXX`, `PGO=1` |
| Lc0 | CUDA for compute capability `LC0_CUDA_CC`, portable CPU code | `LC0_CUDA_CC`, `LC0_VERSION` |

Builds skip profile-guided optimisation by default, because that step runs
the new binary on the build machine, which may not support the target
instruction set. It affects speed only, never results.

After building, if an engine reports `GLIBCXX` errors when it runs, load the
same GCC module when running it (in job scripts, `MODULES="GCC/10.2.0"`).
Then check each engine and record its hashes:

```bash
python code/setup_engines.py check
python code/setup_engines.py smoke
python code/setup_engines.py record
```

Compare each engine's `bench` node count with your PC's build to confirm
both search identically.

## Adding or upgrading an engine

1. Install the binary as the engine's README describes.
2. Update `version` in `engine.toml`.
3. Run `python code/setup_engines.py check --engine <name>` and
   `python code/setup_engines.py smoke --engine <name>`.
4. Run `python code/setup_engines.py record --engine <name>` and commit the
   updated `engines.lock.json`.

Don't upgrade an engine partway through a production run: results from two
builds would be mixed.
