# PomaiDB — Hardware Simulation Lab (Multi-Stage)
# Stage 1: Builder | Stage 2: Minimal runtime (Edge device firmware simulation)
# C++20, single-threaded vector DB for constrained Edge/IoT.

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    g++-13 \
    libssl-dev \
    ninja-build \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Use g++-13 as default for full C++20 (std::format, etc.)
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100

WORKDIR /src

# Copy project (including .git for submodule resolution)
COPY . .

# Initialize palloc submodule (required for build)
RUN git submodule update --init third_party/palloc

# Build: Release, no tests; server + benchmarks
RUN mkdir -p build && cd build \
    && cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++ \
        -DPOMAI_BUILD_TESTS=OFF \
        -DPOMAI_BUILD_BENCH=ON \
        .. \
    && ninja -j$(nproc) \
        pomaidb_server \
        benchmark_a \
        ingestion_bench \
        comprehensive_bench \
        ci_perf_bench \
        rag_bench \
        bench_cbrs \
        pomai_inspect

# =============================================================================
# Stage 2: Runtime (Edge device — minimal image)
# =============================================================================
# Use ubuntu:24.04 (same as builder) for compatibility; -slim can be unavailable in some registries.
FROM ubuntu:24.04 AS runtime

# Minimal runtime: libc + OpenSSL (pomai links libcrypto)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binaries from builder
COPY --from=builder /src/build/pomaidb_server /usr/local/bin/
COPY --from=builder /src/build/benchmark_a /usr/local/bin/
COPY --from=builder /src/build/ingestion_bench /usr/local/bin/
COPY --from=builder /src/build/comprehensive_bench /usr/local/bin/
COPY --from=builder /src/build/ci_perf_bench /usr/local/bin/
COPY --from=builder /src/build/rag_bench /usr/local/bin/
COPY --from=builder /src/build/pomai_inspect /usr/local/bin/
COPY --from=builder /src/build/bin/bench_cbrs /usr/local/bin/

# Mount points: /data for DB files, /bench for report output
RUN mkdir -p /data /bench && chmod 777 /data /bench

WORKDIR /data

# Edge Gateway: HTTP (health, metrics, JSON ingest) + line ingest on one port (default 2406).
EXPOSE 2406/tcp

# Default: run network gateway (embedded use = link libpomai in your app instead).
# Split mode: -e POMAID_HTTP_PORT=8080 -e POMAID_INGEST_PORT=2406
ENV POMAID_DATA_PATH=/data/pomaidb
ENV POMAID_PORT=2406
CMD ["pomaidb_server"]
