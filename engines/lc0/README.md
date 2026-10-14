# Lc0 (Leela Chess Zero)

Version pinned for this study: **v0.32.1**, run on **CPU**.

Lc0 needs two files: the engine binary and a neural network. Neither is
committed. Searches are limited by node count, so running on CPU changes only
how long a search takes, not its result. CPU backends also compute at full
(fp32) precision.

## Install the binary

The binary goes in `engines/lc0/bin/`.

**Windows.** Download `lc0-v0.32.1-windows-cpu-dnnl.zip` (or `-cpu-openblas`)
from the [releases page](https://github.com/LeelaChessZero/lc0/releases/tag/v0.32.1)
and put `lc0.exe` and its DLLs in `bin/`.

**Linux cluster.** There are no Linux release binaries, so build without
CUDA. `build_linux.sh` in this folder does this; it needs `meson`, `ninja`, a
C++17 compiler and OpenBLAS (often `module load openblas` on a cluster):

```bash
bash engines/lc0/build_linux.sh
```

## Install the network

Networks are listed at <https://lczero.org/play/networks/bestnets/>. Large
transformer networks are slow on CPU (perhaps tens to hundreds of nodes per
second per core), so the pilot measures speed for one or two candidates and
picks the network and top node budget to fit the compute allocation.

Save the chosen file in `engines/lc0/networks/` and set its name in
`engine.toml` under `[network] file`.

## Settings that keep Lc0 deterministic

`engine.toml` sets one search thread, `SmartPruningFactor = 0` (search exactly
the node budget rather than stopping early) and one BLAS thread through
environment variables. The pilot reruns a few positions to confirm results
reproduce exactly.

## Verify

```bash
python code/setup_engines.py check  --engine lc0
python code/setup_engines.py smoke  --engine lc0
python code/setup_engines.py record --engine lc0
```
