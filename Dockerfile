FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    clang \
    git \
    python3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/pomaidb

# Copy full source tree so Docker builds are reproducible in CI/dev.
COPY . .

# Default to a release build that includes tests.
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPOMAI_BUILD_TESTS=ON \
    && cmake --build build --parallel

# Useful default for local container runs.
CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
