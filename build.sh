#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./build.sh              # Release build (default)
#   ./build.sh Debug        # Debug build
#   ./build.sh Release run  # Build + run pomai-server
#
# Optional env:
#   CONFIG=path/to/pomai.yaml

BUILD_TYPE="${1:-Release}"
ACTION="${2:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
CONFIG="${CONFIG:-${ROOT_DIR}/config/pomai.yaml}"

echo "[pomai] Build type: ${BUILD_TYPE}"
echo "[pomai] Build dir : ${BUILD_DIR}"

mkdir -p "${BUILD_DIR}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
cmake --build "${BUILD_DIR}" -j"$(nproc)" --target pomai-server

echo "[pomai] Build done."
echo "[pomai] Binary:"
echo "  - ${BUILD_DIR}/pomai-server"

if [[ "${ACTION}" == "run" ]]; then
  echo "[pomai] Running server..."
  exec "${BUILD_DIR}/pomai-server" --config "${CONFIG}"
fi
