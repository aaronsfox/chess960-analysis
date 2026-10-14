# Stockfish

Version pinned for this study: **19** (check `engine.toml`).

## Install

The binary goes in `engines/stockfish/bin/` (git ignores that folder).

**Windows (development).** Download the Windows x86-64 AVX2 build from the
[Stockfish download page](https://stockfishchess.org/download/) or the
[GitHub releases](https://github.com/official-stockfish/Stockfish/releases),
unzip it and rename the executable to `bin/stockfish.exe`.

**Linux cluster (production data).** Building from the release tag gives a
binary tuned to the cluster's CPUs:

```bash
git clone --depth 1 --branch sf_19 https://github.com/official-stockfish/Stockfish.git /tmp/sf
cd /tmp/sf/src
make -j profile-build ARCH=x86-64-avx2      # or ARCH=native on the compute node type
cp stockfish <repo>/engines/stockfish/bin/stockfish
```

Check the tag name on the releases page if `sf_19` doesn't resolve.

## Verify

```bash
python code/setup_engines.py check  --engine stockfish
python code/setup_engines.py smoke  --engine stockfish
python code/setup_engines.py record --engine stockfish   # then commit engines/engines.lock.json
```

Stockfish's `bench` command prints a node count that is identical across
CPU architectures for the same source version. Running `./bin/stockfish bench`
on your PC and on the cluster and comparing the "Nodes searched" line is a
quick check that both builds search identically.
