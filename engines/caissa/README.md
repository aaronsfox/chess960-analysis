# Caissa

Version pinned for this study: **2.0** (release tag `2.0`, September 2026,
[release page](https://github.com/Witek902/Caissa/releases/tag/2.0)).

Caissa's networks are trained on its own self-play games, and it reports
win/draw/loss probabilities (`UCI_ShowWDL`). Version 2.0 moves to a
multilayer network (wider first layer, pairwise activation, output
subnetworks chosen by piece count), fixes several FEN-parsing issues and a
`hashfull` reporting bug, and removes the `UCI_AnalyseMode` option. This
study doesn't use that option.

## Install

The binary goes in `engines/caissa/bin/` (git ignores that folder).

**Windows (development).** Download the release build for your CPU from the
[2.0 release page](https://github.com/Witek902/Caissa/releases/tag/2.0) and
rename it to `bin/caissa.exe`. The release files are
`caissa-2.0-x64-<build>.exe` for `avx512icl`, `avx512`, `avx2-bmi2`, `avx2`,
`sse4-popcnt` and `sse2`. Most modern CPUs run `avx2-bmi2`; `avx512icl` needs
an Intel Ice Lake (or newer) or AMD Zen 4 CPU.

**Linux cluster (production data).** The 2.0 release has Windows builds only,
so build from source. Caissa's `src/makefile` needs only `make` and GCC 10+
(no CMake). It has a target per instruction set and downloads the network
file from the Caissa-Nets repository, embedding it in the binary, so run it
on a login node with internet access:

```bash
module load gcc                     # GCC 10+ for C++20; check `g++ --version`
git clone --depth 1 --branch 2.0 https://github.com/Witek902/Caissa.git /tmp/caissa-build
cd /tmp/caissa-build/src
make -j4 avx512                     # or bmi2; avoid avx512icl on Cascade Lake
ls -lt | head                       # the newest executable is the build
cp <built-executable> <repo>/engines/caissa/bin/caissa
chmod +x <repo>/engines/caissa/bin/caissa
```

Match the Windows release file to the Linux target: `avx512` with
`caissa-2.0-x64-avx512.exe`, or `bmi2` with `caissa-2.0-x64-avx2-bmi2.exe`.

If the network download fails, get the `.pnn` file named in the build error
from the Caissa-Nets releases, put it in `data/neuralNets/`, and run `make`
again. If you prefer CMake, `pip install cmake` in the project environment
installs it without admin rights.

## Use the same build type everywhere

Pick one instruction-set build (for example `bmi2`) that runs on both your PC
and the cluster's compute nodes, and use it on both. Different builds use
different vector code paths, which can change results slightly. Caissa's
`bench` command prints a node count: if it matches on both machines, the
builds search identically.

## Verify

```bash
python code/setup_engines.py check  --engine caissa
python code/setup_engines.py smoke  --engine caissa
python code/setup_engines.py record --engine caissa
```
