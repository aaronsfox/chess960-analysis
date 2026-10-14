# PlentyChess

Version pinned for this study: **7.0.0** (check `engine.toml` and the
[releases page](https://github.com/Yoshie2000/PlentyChess/releases) before pinning).

PlentyChess's networks are trained on its own self-generated positions,
covering both standard chess and Chess960. Whether it reports win/draw/loss
probabilities is checked by `setup_engines.py check`; if not, expected score
is derived later from a fitted centipawn calibration.

## Install

The binary goes in `engines/plentychess/bin/` (git ignores that folder).

**Windows.** Download the release build matching your CPU and rename it to
`bin/plentychess.exe`.

**Linux cluster.** Build from the release tag (the Makefile downloads and
embeds the network automatically):

```bash
git clone --depth 1 --branch <release-tag> https://github.com/Yoshie2000/PlentyChess.git /tmp/plenty
cd /tmp/plenty
make -j profile-build EXE=plentychess
cp plentychess <repo>/engines/plentychess/bin/plentychess
```

## Verify

```bash
python code/setup_engines.py check  --engine plentychess
python code/setup_engines.py smoke  --engine plentychess
python code/setup_engines.py record --engine plentychess
```
