# PlentyChess

Version pinned for this study: **8.0.0** (release tag `b-v8.0.0`, June 2026,
[release page](https://github.com/Yoshie2000/PlentyChess/releases/tag/b-v8.0.0)).

PlentyChess's networks are trained on its own self-generated positions,
covering both standard chess and Chess960. Version 8.0.0 adds a threat-input
network architecture with pawn-pair inputs, more fractional-depth search, and
NUMA handling on Linux; its release notes report about +49 Elo over 7.0.0 in
Double Fischer Random testing. Whether it reports win/draw/loss
probabilities is checked by `setup_engines.py check`; if not, expected score
is derived later from a fitted centipawn calibration.

## Install

The binary goes in `engines/plentychess/bin/` (git ignores that folder).

**Windows (development).** Download the Windows build for your CPU from the
[8.0.0 release page](https://github.com/Yoshie2000/PlentyChess/releases/tag/b-v8.0.0)
(click "Show all assets") and rename it to `bin/plentychess.exe`.

**Linux cluster (production data).** Download the Linux build for the compute
nodes' CPUs. The release offers `PlentyChess-8.0.0-linux-<arch>` builds for
`generic`, `ssse3`, `fma`, `avx2`, `bmi2`, `avx512` and `avx512vbmi2`. Then:

```bash
cp PlentyChess-8.0.0-linux-bmi2 <repo>/engines/plentychess/bin/plentychess
chmod +x <repo>/engines/plentychess/bin/plentychess
```

Or build from the release tag (the Makefile downloads and embeds the network):

```bash
git clone --depth 1 --branch b-v8.0.0 https://github.com/Yoshie2000/PlentyChess.git /tmp/plenty
cd /tmp/plenty
make -j profile-build EXE=plentychess arch=bmi2
cp plentychess <repo>/engines/plentychess/bin/plentychess
```

## Use the same build type everywhere

Pick one architecture build (for example `bmi2`) that runs on both your PC and
the cluster's compute nodes, and use it on both. Different builds use
different vector code paths, which can change results slightly.

## Verify

```bash
python code/setup_engines.py check  --engine plentychess
python code/setup_engines.py smoke  --engine plentychess
python code/setup_engines.py record --engine plentychess
```
