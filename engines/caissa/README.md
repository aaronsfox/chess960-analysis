# Caissa

Version pinned for this study: **1.25** (check `engine.toml` and the
[releases page](https://github.com/Witek902/Caissa/releases) before pinning).

Caissa's networks are trained on its own self-play games, including Double
Fischer Random openings, and it reports win/draw/loss probabilities
(`UCI_ShowWDL`).

## Install

The binary goes in `engines/caissa/bin/` (git ignores that folder).

**Windows.** Download the release build matching your CPU (BMI2 suits most
modern CPUs) and rename it to `bin/caissa.exe`.

**Linux cluster.** Build from the release tag with CMake:

```bash
git clone --depth 1 --branch <release-tag> https://github.com/Witek902/Caissa.git /tmp/caissa
cd /tmp/caissa && mkdir build && cd build
cmake -DTARGET_ARCH=x64-bmi2 -DCMAKE_BUILD_TYPE=Final ..
make -j
cp bin/caissa <repo>/engines/caissa/bin/caissa   # path may differ; see Caissa's README
```

Caissa's README says to copy the network file from `data/neuralNets/` next to
the binary after compiling. If the engine needs it, put it in `bin/` as well;
if it must be set with `EvalFile`, uncomment the `[network]` section in
`engine.toml` so the file's hash is recorded.

## Verify

```bash
python code/setup_engines.py check  --engine caissa
python code/setup_engines.py smoke  --engine caissa
python code/setup_engines.py record --engine caissa
```
