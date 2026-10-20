#!/usr/bin/env bash
# Build fastchess (the match runner for self-play) and install it as
# tools/fastchess/bin/fastchess.
#
#   module load GCC/10.2.0                      # C++17 compiler; your module name
#   bash tools/fastchess/build_linux.sh
#
# Settings (environment variables):
#   FASTCHESS_VERSION  release tag              (default v1.8.2-alpha)
#   CXX                C++ compiler              (default g++)
#
# `make build=release` produces a portable (x86-64), statically linked binary,
# so it runs on every node regardless of the cluster's system libraries.
# Needs git and make; run it on a login node.
set -euo pipefail

VERSION="${FASTCHESS_VERSION:-v1.8.2-alpha}"
COMPILER="${CXX:-g++}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${TMPDIR:-/tmp}/fastchess-build-${VERSION}"

echo "Building fastchess ${VERSION} with $($COMPILER --version | head -n 1)"
rm -rf "$WORK"
git clone --depth 1 --branch "$VERSION" https://github.com/Disservin/fastchess.git "$WORK"
cd "$WORK"
make -j4 build=release CXX="$COMPILER"

mkdir -p "$HERE/bin"
cp fastchess "$HERE/bin/fastchess"
chmod +x "$HERE/bin/fastchess"
echo "Installed $HERE/bin/fastchess"
"$HERE/bin/fastchess" -version
