#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./build.sh              # Release build (default)
#   ./build.sh clean        # Clean build directory completely
#   ./build.sh run          # Build and run

ACTION="${1:-build}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
CONFIG="${ROOT_DIR}/config/pomai.yaml"

# Màu mè cho dễ nhìn
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[pomai]${NC} $1"
}

err() {
    echo -e "${RED}[error]${NC} $1"
}

if [[ "${ACTION}" == "clean" ]]; then
    log "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    exit 0
fi

log "Build Configuration:"
log "  Root: ${ROOT_DIR}"
log "  Mode: Release (Optimized)"

# Tạo thư mục build
mkdir -p "${BUILD_DIR}"

# Cấu hình CMake
# -DCMAKE_BUILD_TYPE=Release: Quan trọng để bật -O3
log "Configuring CMake..."
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -Wno-dev

# Compile
log "Compiling..."
# Sử dụng tất cả core CPU (-j)
if cmake --build "${BUILD_DIR}" -j"$(nproc)" --target pomai-server; then
    log "Build SUCCESS."
else
    err "Build FAILED."
    exit 1
fi

BINARY="${BUILD_DIR}/pomai-server"
log "Binary: ${BINARY}"

if [[ "${ACTION}" == "run" ]]; then
    if [[ ! -f "${CONFIG}" ]]; then
        err "Config file not found at ${CONFIG}"
        exit 1
    fi
    log "Running server..."
    exec "${BINARY}" --config "${CONFIG}"
fi