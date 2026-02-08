#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB_PATH="${POMAI_LIB:-$ROOT_DIR/build/libpomai_c.so}"

if [[ ! -f "$LIB_PATH" ]]; then
  echo "Missing shared library: $LIB_PATH" >&2
  exit 1
fi

"$ROOT_DIR/scripts/pomai-bench" recall --lib "$LIB_PATH"
"$ROOT_DIR/scripts/pomai-bench" mixed-load --lib "$LIB_PATH" --dim 256 --count 50000 --shards 2 --batch-size 512 --queries 500
