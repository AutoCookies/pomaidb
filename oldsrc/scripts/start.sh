#!/usr/bin/env bash
# POSIX shell script for Linux / macOS
set -euo pipefail

PORT="${PORT:-8080}"
DATA_DIR="${DATA_DIR:-./data}"
PERSISTENCE="${PERSISTENCE:-noop}" # options: noop, file, wal

mkdir -p "${DATA_DIR}"

# If binary doesn't exist, build it
if [ ! -x ./pomai-cache ]; then
  echo "Building binary..."
  go build -v -o pomai-cache ./cmd/server
fi

echo "Starting pomai-cache (port=${PORT}, data=${DATA_DIR}, persistence=${PERSISTENCE})"
export PORT DATA_DIR PERSISTENCE
exec ./pomai-cache