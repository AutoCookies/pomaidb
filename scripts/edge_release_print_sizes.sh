#!/usr/bin/env bash
# Print on-disk sizes of PomaiDB artifacts for release notes / EDGE_RELEASE.md baselines.
# Usage: from repo root after build:
#   ./scripts/edge_release_print_sizes.sh [build-dir]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-${ROOT}/build}"
echo "PomaiDB artifact sizes (bytes) under: ${BUILD}"
for name in libpomai_c.so libpomai_c.a libpomai.a; do
  f="${BUILD}/${name}"
  if [[ -f "$f" ]]; then
    wc -c <"$f" | awk -v p="$f" '{printf "%s  %d bytes\n", p, $1}'
  fi
done
