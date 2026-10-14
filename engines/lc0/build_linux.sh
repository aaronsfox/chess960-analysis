#!/usr/bin/env bash
# Build Lc0 for CPU (BLAS backend, no CUDA) and copy it into engines/lc0/bin/.
# Needs: git, meson, ninja, a C++17 compiler, OpenBLAS headers and library.
# On a cluster you may need e.g. `module load gcc openblas` first.
set -euo pipefail

VERSION="${LC0_VERSION:-v0.32.1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${TMPDIR:-/tmp}/lc0-build-${VERSION}"

rm -rf "$WORK"
git clone --depth 1 --branch "$VERSION" --recurse-submodules \
    https://github.com/LeelaChessZero/lc0.git "$WORK"
cd "$WORK"

# Lc0's build.sh passes extra arguments to meson. Turn off the GPU backends so
# meson doesn't look for CUDA; BLAS is on by default. If meson reports an
# unknown option, check `meson configure build/release` for this version's names.
./build.sh -Dcudnn=false -Dplain_cuda=false -Dopencl=false -Dgtest=false

mkdir -p "$HERE/bin"
cp build/release/lc0 "$HERE/bin/lc0"
echo "Installed $HERE/bin/lc0"
"$HERE/bin/lc0" --help | head -n 3 || true
