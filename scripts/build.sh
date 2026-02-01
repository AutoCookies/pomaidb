#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Pomai build helper
#
# Usage:
#   ./scripts/build.sh                 # build (default Release)
#   ./scripts/build.sh clean           # remove build dir
#   ./scripts/build.sh test            # build + ctest
#   ./scripts/build.sh bench           # build + run bench_pomai (if exists)
#   ./scripts/build.sh run             # build + run pomai-server (if exists)
#   ./scripts/build.sh install         # build + install
#
# Presets:
#   ./scripts/build.sh asan            # RelWithDebInfo + ASAN+UBSAN
#   ./scripts/build.sh tsan            # Debug + TSAN
#
# Env overrides:
#   BUILD_TYPE=Release|Debug|RelWithDebInfo|MinSizeRel   (default Release)
#   BUILD_DIR=build                                      (default <repo>/build)
#   NATIVE=1                                             (Release only, adds -march=native)
#   PREFIX=/usr/local                                    (install prefix)
#   RUN_BIN=...                                          (override binary for action=run)
#   RUN_ARGS="..."                                       (args for action=run)
#   BENCH_BIN=...                                        (override benchmark binary)
#   BENCH_ARGS="..."                                     (args for action=bench)
# ==============================================================================

ACTION="${1:-build}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
NATIVE="${NATIVE:-0}"
PREFIX="${PREFIX:-/usr/local}"

RUN_ARGS="${RUN_ARGS:-}"
BENCH_ARGS="${BENCH_ARGS:-}"

# Default binary names (override via env if your target differs)
RUN_BIN="${RUN_BIN:-${BUILD_DIR}/bin/pomai-server}"
BENCH_BIN="${BENCH_BIN:-${BUILD_DIR}/bin/bench_pomai}"

GREEN='\033[1;32m'
BLUE='\033[1;34m'
RED='\033[1;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[pomai]${NC} $*"; }
info() { echo -e "${BLUE}[info]${NC} $*"; }
err()  { echo -e "${RED}[error]${NC} $*"; }

POMAI_CMAKE_OPTS=()

case "${ACTION}" in
  clean)
    log "Cleaning build directory: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
    log "Clean complete."
    exit 0
    ;;
  asan)
    ACTION="build"
    BUILD_TYPE="RelWithDebInfo"
    POMAI_CMAKE_OPTS+=(-DPOMAI_ENABLE_ASAN=ON -DPOMAI_ENABLE_UBSAN=ON)
    ;;
  tsan)
    ACTION="build"
    BUILD_TYPE="Debug"
    POMAI_CMAKE_OPTS+=(-DPOMAI_ENABLE_TSAN=ON)
    ;;
esac

# Prefer Ninja if available
GENERATOR=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR=(-G Ninja)
fi

log "Build Environment:"
info "  Repo root:  ${REPO_ROOT}"
info "  Build dir:  ${BUILD_DIR}"
info "  Build type: ${BUILD_TYPE}"
info "  Native:     ${NATIVE}"
if ((${#POMAI_CMAKE_OPTS[@]})); then
  info "  CMake opts: ${POMAI_CMAKE_OPTS[*]}"
fi

mkdir -p "${BUILD_DIR}"

log "Configuring CMake..."
CXX_FLAGS_RELEASE=()
if [[ "${BUILD_TYPE}" == "Release" && "${NATIVE}" == "1" ]]; then
  # NOTE: -march=native makes binaries non-portable.
  CXX_FLAGS_RELEASE=(-DCMAKE_CXX_FLAGS_RELEASE=-O3\ -march=native)
fi

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
  "${GENERATOR[@]}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  "${CXX_FLAGS_RELEASE[@]}" \
  "${POMAI_CMAKE_OPTS[@]}" \
  -Wno-dev

log "Compiling..."
cmake --build "${BUILD_DIR}" -j"$(nproc)"

case "${ACTION}" in
  test)
    log "Running tests (ctest)..."
    (cd "${BUILD_DIR}" && ctest --output-on-failure)
    ;;
  install)
    log "Installing to prefix: ${PREFIX}"
    cmake --install "${BUILD_DIR}"
    ;;
  run)
    if [[ ! -x "${RUN_BIN}" ]]; then
      err "Run binary not found: ${RUN_BIN}"
      err "Set RUN_BIN=/path/to/your/binary or ensure you have a server target."
      exit 1
    fi
    log "Running: ${RUN_BIN} ${RUN_ARGS}"
    exec "${RUN_BIN}" ${RUN_ARGS}
    ;;
  bench)
    if [[ ! -x "${BENCH_BIN}" ]]; then
      err "Benchmark binary not found: ${BENCH_BIN}"
      err "Tip: ensure your root CMake builds bench_pomai, or run scripts/build_bench.sh."
      exit 1
    fi
    log "Running: ${BENCH_BIN} ${BENCH_ARGS}"
    exec "${BENCH_BIN}" ${BENCH_ARGS}
    ;;
  build|*)
    log "Done."
    ;;
esac
