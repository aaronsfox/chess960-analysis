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
rename it to `bin/caissa.exe`. Builds are offered for `avx512icl`, `avx512`,
`bmi2`, `avx2` and `sse4-popcnt`/`sse2`. Most modern CPUs run `bmi2`;
`avx512icl` needs an Intel Ice Lake (or newer) or AMD Zen 4 CPU.

**Linux cluster (production data).** Use the matching Linux release build if
one is offered, or build from the `2.0` tag with CMake:

```bash
git clone --depth 1 --branch 2.0 https://github.com/Witek902/Caissa.git /tmp/caissa
cd /tmp/caissa && mkdir build && cd build
cmake -DTARGET_ARCH=x64-bmi2 -DCMAKE_BUILD_TYPE=Final ..
make -j
cp bin/caissa <repo>/engines/caissa/bin/caissa   # output path may differ; check Caissa's README
```

Since version 1.26, Caissa's network files (`.pnn`) are kept in a separate
Caissa-Nets repository and the build downloads them automatically. Release
binaries are meant to run as downloaded. If your build instead needs the
network set with `EvalFile`, put the file in `networks/` and uncomment the
`[network]` section in `engine.toml`, so the file's hash is recorded.

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
