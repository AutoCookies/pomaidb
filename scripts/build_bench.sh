#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Build & run benchmarks in an isolated build directory.
#
# Usage:
#   ./scripts/build_bench.sh
#
# Env overrides:
#   BUILD_TYPE=Release|RelWithDebInfo|Debug    (default Release)
#   BUILD_DIR=<path>                           (default <repo>/build-bench)
#   BENCH_ARGS="..."                           (default: a reasonable preset)
#   NATIVE=1                                   (Release only, adds -march=native)
#
# Notes:
# - If benchmarks/CMakeLists.txt exists, we build benchmarks as a subproject.
# - Otherwise we fallback to building the root project target 'bench_pomai'.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-bench}"
NATIVE="${NATIVE:-0}"

# Default args; override freely
BENCH_ARGS="${BENCH_ARGS:---n 200000 --dim 512 --batch 512 --shards 4 --queries 2000 --topk 10 --fsync never}"

GREEN='\033[1;32m'
BLUE='\033[1;34m'
RED='\033[1;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[bench]${NC} $*"; }
info() { echo -e "${BLUE}[info]${NC} $*"; }
err()  { echo -e "${RED}[error]${NC} $*"; }

# Prefer Ninja if available
GENERATOR=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR=(-G Ninja)
fi

# Optional native flags
CXX_FLAGS_RELEASE=()
if [[ "${BUILD_TYPE}" == "Release" && "${NATIVE}" == "1" ]]; then
  CXX_FLAGS_RELEASE=(-DCMAKE_CXX_FLAGS_RELEASE=-O3\ -march=native)
fi

BENCH_SUBPROJECT="${REPO_ROOT}/benchmarks/CMakeLists.txt"
ROOT_CMAKE="${REPO_ROOT}/CMakeLists.txt"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

log "Repo root:  ${REPO_ROOT}"
log "Build dir:  ${BUILD_DIR}"
log "Build type: ${BUILD_TYPE}"
log "Native:     ${NATIVE}"

if [[ -f "${BENCH_SUBPROJECT}" ]]; then
  log "Detected benchmarks subproject: benchmarks/CMakeLists.txt"
  cmake -S "${REPO_ROOT}/benchmarks" -B "${BUILD_DIR}" \
    "${GENERATOR[@]}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    "${CXX_FLAGS_RELEASE[@]}" \
    -Wno-dev
else
  log "No benchmarks/CMakeLists.txt found. Falling back to root build (target bench_pomai)."
  if [[ ! -f "${ROOT_CMAKE}" ]]; then
    err "Root CMakeLists.txt not found at: ${ROOT_CMAKE}"
    exit 1
  fi
  cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
    "${GENERATOR[@]}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DPOMAI_BUILD_BENCHMARKS=ON \
    "${CXX_FLAGS_RELEASE[@]}" \
    -Wno-dev
fi

log "Building..."
cmake --build "${BUILD_DIR}" -j"$(nproc)"

BENCH_BIN="${BUILD_DIR}/bin/bench_pomai"
if [[ ! -x "${BENCH_BIN}" ]]; then
  # Some projects output to build root instead of bin — try fallback
  if [[ -x "${BUILD_DIR}/bench_pomai" ]]; then
    BENCH_BIN="${BUILD_DIR}/bench_pomai"
  else
    err "bench_pomai not found in ${BUILD_DIR}/bin (or build root)."
    err "Check your benchmark target name and output directory."
    exit 1
  fi
fi

log "Running: ${BENCH_BIN} ${BENCH_ARGS}"
exec "${BENCH_BIN}" ${BENCH_ARGS}
