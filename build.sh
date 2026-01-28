#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./build.sh              # Build toàn bộ (Server + Benchmark)
#   ./build.sh clean        # Xóa sạch thư mục build
#   ./build.sh run          # Build xong chạy Server
#   ./build.sh bench        # Build xong chạy Benchmark Embedded

ACTION="${1:-build}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
CONFIG="${ROOT_DIR}/config/pomai.yaml"

# Màu mè chuyên nghiệp
GREEN='\033[1;32m'
BLUE='\033[1;34m'
RED='\033[1;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[pomai]${NC} $1"
}

info() {
    echo -e "${BLUE}[info]${NC} $1"
}

err() {
    echo -e "${RED}[error]${NC} $1"
}

# --- 1. CLEANUP ---
if [[ "${ACTION}" == "clean" ]]; then
    log "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    log "Clean complete."
    exit 0
fi

# --- 2. BUILD SYSTEM ---
log "Build Environment:"
info "  Root: ${ROOT_DIR}"
info "  Mode: Release (Optimized -O3 -march=native)"

mkdir -p "${BUILD_DIR}"

log "Configuring CMake..."
# -DCMAKE_BUILD_TYPE=Release: Bật tối ưu hóa CPU
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -Wno-dev

log "Compiling Sources..."
# Build TẤT CẢ các target (Core, Server, Bench)
# -j"$(nproc)": Sử dụng tối đa số core CPU để compile nhanh gấp 4-8 lần
if cmake --build "${BUILD_DIR}" -j"$(nproc)"; then
    log "Build SUCCESS."
else
    err "Build FAILED."
    exit 1
fi

SERVER_BIN="${BUILD_DIR}/pomai-server"
BENCH_BIN="${BUILD_DIR}/bench_embedded"

# --- 3. RUN ACTIONS ---

if [[ "${ACTION}" == "run" ]]; then
    # Chế độ chạy Server (Network Mode)
    if [[ ! -f "${SERVER_BIN}" ]]; then
        err "Server binary not found at: ${SERVER_BIN}"
        exit 1
    fi
    if [[ ! -f "${CONFIG}" ]]; then
        err "Config file missing at: ${CONFIG}"
        exit 1
    fi
    
    log "🚀 Launching Pomai Server..."
    exec "${SERVER_BIN}" --config "${CONFIG}"

elif [[ "${ACTION}" == "bench" ]]; then
    # Chế độ chạy Benchmark (Embedded Mode)
    if [[ ! -f "${BENCH_BIN}" ]]; then
        err "Benchmark binary not found at: ${BENCH_BIN}"
        exit 1
    fi

    log "Running Embedded Core Benchmark..."
    exec "${BENCH_BIN}"
fi