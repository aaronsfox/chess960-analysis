#!/usr/bin/env bash
# Build PlentyChess from source and install it as
# engines/plentychess/bin/plentychess. The release's Linux binaries need
# glibc 2.38, which many clusters don't have.
#
#   module load GCC/10.2.0                      # your cluster's module name
#   bash engines/plentychess/build_linux.sh
#
# Settings (environment variables):
#   PLENTY_VERSION  release tag                 (default b-v8.0.0)
#   ARCH            generic, ssse3, fma, avx2, bmi2, avx512, avx512vbmi2
#                   (default bmi2; avoid avx512vbmi2 on Cascade Lake).
#                   Use the same architecture as your Windows build.
#   CXX             C++ compiler                (default g++; the makefile's
#                   own default is clang++)
#   PGO=1           profile-guided build: faster, but runs the new binary
#                   during the build, so the build machine must support ARCH
#
# Needs git, make, a C++17 compiler and curl (the build downloads
# PlentyChess's network), so run it on a login node.
set -euo pipefail

VERSION="${PLENTY_VERSION:-b-v8.0.0}"
ARCH="${ARCH:-bmi2}"
COMPILER="${CXX:-g++}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${TMPDIR:-/tmp}/plentychess-build-${VERSION}"

echo "Building PlentyChess ${VERSION} for ${ARCH} with $($COMPILER --version | head -n 1)"
rm -rf "$WORK"
git clone --depth 1 --branch "$VERSION" https://github.com/Yoshie2000/PlentyChess.git "$WORK"
cd "$WORK"

if [[ "${PGO:-0}" == "1" ]]; then
    make -j4 profile-build EXE=plentychess arch="$ARCH" CXX="$COMPILER"
else
    make -j4 EXE=plentychess arch="$ARCH" CXX="$COMPILER"
fi

mkdir -p "$HERE/bin"
cp plentychess "$HERE/bin/plentychess"
echo "Installed $HERE/bin/plentychess"
printf 'uci\nquit\n' | "$HERE/bin/plentychess" | grep -m 1 "id name" || true
