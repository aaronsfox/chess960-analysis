# PlentyChess

Version pinned for this study: **8.0.0** (release tag `b-v8.0.0`, June 2026,
[release page](https://github.com/Yoshie2000/PlentyChess/releases/tag/b-v8.0.0)).

PlentyChess's networks are trained on its own self-generated positions,
covering both standard chess and Chess960. Version 8.0.0 adds a threat-input
network architecture with pawn-pair inputs, more fractional-depth search, and
NUMA handling on Linux; its release notes report about +49 Elo over 7.0.0 in
Double Fischer Random testing. PlentyChess reports centipawn scores
but no win/draw/loss probabilities, and this study doesn't calibrate them.
Its centipawns are used for ranking the positions, rank agreement with the
other engines, and convergence across node budgets; expected score, Elo and
draw probability come from the engines that report WDL.

## Install

The binary goes in `engines/plentychess/bin/` (git ignores that folder).

**Windows (development).** Download the Windows build for your CPU from the
[8.0.0 release page](https://github.com/Yoshie2000/PlentyChess/releases/tag/b-v8.0.0)
(click "Show all assets") and rename it to `bin/plentychess.exe`.

**Linux cluster (production data).** Build from source with `build_linux.sh`.
The release's Linux binaries need a recent system C library (glibc 2.38) and
fail on clusters with older ones (`GLIBC_2.38 not found`). The build needs
`make`, a C++17 compiler and `curl`, which downloads the network, so run it
on a login node:

```bash
module load GCC/10.2.0                          # your cluster's module name
bash engines/plentychess/build_linux.sh         # ARCH=bmi2, CXX=g++ by default
```

`ARCH` takes `generic`, `ssse3`, `fma`, `avx2`, `bmi2`, `avx512` or
`avx512vbmi2`; use the same architecture as your Windows build. The default
build skips profile-guided optimisation, which only affects speed (`PGO=1`
adds it). If the build stops on `numa.h` or `-lnuma`, load the cluster's
`numactl` module. If the cluster's glibc is 2.38 or newer, the release file
`PlentyChess-8.0.0-linux-<arch>` also works.

## Use the same build type everywhere

Pick one architecture build (`bmi2` by default) that runs on both your PC
and the cluster's compute nodes, and use it on both. Different builds use
different vector code paths, which can change results slightly. Your Windows
release build and your GCC-built Linux binary also come from different
compilers, so compare `bench` node counts on both machines: if they match,
both search identically.

## Verify

```bash
python code/setup_engines.py check  --engine plentychess
python code/setup_engines.py smoke  --engine plentychess
python code/setup_engines.py record --engine plentychess
```
