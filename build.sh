#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./build.sh                    # Build (default: Release)
#   ./build.sh clean              # Remove build dir
#   ./build.sh run                # Build then run server
#   ./build.sh bench              # Build then run benchmark (if target exists)
#   ./build.sh test               # Build then run ctest
#   ./build.sh install            # Build then install (DESTDIR/PREFIX supported)
#
# Presets:
#   ./build.sh asan               # RelWithDebInfo + ASAN+UBSAN
#   ./build.sh tsan               # Debug + TSAN
#
# Env:
#   BUILD_TYPE=Release|Debug|RelWithDebInfo|MinSizeRel  (default: Release)
#   NATIVE=1          -> enable -march=native (Release only; local machine)
#   BUILD_DIR=build   -> override build dir
#   SERVER_ARGS="..." -> extra args passed to pomai-server
#   PREFIX=/usr/local -> install prefix (default: /usr/local)

ACTION="${1:-build}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
NATIVE="${NATIVE:-0}"
SERVER_ARGS="${SERVER_ARGS:-}"
PREFIX="${PREFIX:-/usr/local}"

GREEN='\033[1;32m'
BLUE='\033[1;34m'
RED='\033[1;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[pomai]${NC} $1"; }
info() { echo -e "${BLUE}[info]${NC} $1"; }
err()  { echo -e "${RED}[error]${NC} $1"; }

# ---- Presets ----
POMAI_CMAKE_OPTS=()

case "${ACTION}" in
  clean)
    log "Cleaning build directory..."
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
info "  Root:       ${ROOT_DIR}"
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
  # NOTE: -march=native makes binaries non-portable across different CPUs.
  CXX_FLAGS_RELEASE=(-DCMAKE_CXX_FLAGS_RELEASE=-O3\ -march=native)
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  "${GENERATOR[@]}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  "${CXX_FLAGS_RELEASE[@]}" \
  "${POMAI_CMAKE_OPTS[@]}" \
  -Wno-dev

log "Compiling Sources..."
cmake --build "${BUILD_DIR}" -j"$(nproc)"

BIN_DIR="${BUILD_DIR}/bin"
SERVER_BIN="${BIN_DIR}/pomai-server"
BENCH_BIN="${BIN_DIR}/bench_embedded_upsert"

# Helper: check if a CMake target exists (only works reliably with Ninja/Makefiles)
cmake_target_exists() {
  local t="$1"
  # Try Ninja first
  if [[ -f "${BUILD_DIR}/build.ninja" ]]; then
    ninja -C "${BUILD_DIR}" -t targets all 2>/dev/null | grep -qE "^${t}:" && return 0 || return 1
  fi
  # Fallback: best-effort for Makefiles
  if [[ -f "${BUILD_DIR}/Makefile" ]]; then
    make -C "${BUILD_DIR}" -qp 2>/dev/null | grep -qE "^${t}:" && return 0 || return 1
  fi
  return 1
}

if [[ "${ACTION}" == "run" ]]; then
  if [[ ! -x "${SERVER_BIN}" ]]; then
    err "Server binary not found at: ${SERVER_BIN}"
    err "Tip: ensure POMAI_BUILD_SERVER=ON and target name is 'pomai-server'."
    exit 1
  fi
  log "🚀 Launching Pomai Server..."
  # No YAML/--config assumption. Use SERVER_ARGS for runtime flags.
  # Example:
  #   SERVER_ARGS="--host 0.0.0.0 --port 8080 --data ./data --shards 4 --dim 512" ./build.sh run
  exec "${SERVER_BIN}" ${SERVER_ARGS}

elif [[ "${ACTION}" == "bench" ]]; then
  if [[ -x "${BENCH_BIN}" ]]; then
    log "Running Embedded Core Benchmark..."
    exec "${BENCH_BIN}"
  fi

  # If you haven't implemented bench target yet, be explicit and helpful.
  err "Benchmark binary not found at: ${BENCH_BIN}"
  err "Tip: add a 'bench_embedded' target in CMake and output to build/bin/"
  exit 1

elif [[ "${ACTION}" == "test" ]]; then
  log "Running tests (ctest)..."
  (cd "${BUILD_DIR}" && ctest --output-on-failure)

elif [[ "${ACTION}" == "install" ]]; then
  log "Installing to prefix: ${PREFIX}"
  cmake --install "${BUILD_DIR}"

else
  log "Done."
fi
