#!/usr/bin/env bash
set -euo pipefail

# Robust test runner for no-gtest tests.
# Usage:
#  - From repo root: ./tests/run_tests.sh
#  - Or: bash ./tests/run_tests.sh

# Resolve script directory and repo root reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

CXX=${CXX:-g++}
CXXFLAGS="-std=c++17 -O2 -I${REPO_ROOT} -pthread -Wall -Wextra"

# Extra compilation units needed by tests (objects providing non-inline symbols)
EXTRA_SRC="$REPO_ROOT/core/config.cc"

mkdir -p "$BUILD_DIR"

# Helper to compile a test if source exists; links in EXTRA_SRC so shared symbols are available.
compile_and_run() {
  local src="$1"   # relative to repo root, e.g. tests/test_map_harvest.cc
  local out="$2"   # output binary path
  if [ ! -f "$REPO_ROOT/$src" ]; then
    echo "Skipping $src (not found)"
    return 0
  fi
  echo "Compiling $src -> $out"
  "$CXX" $CXXFLAGS "$REPO_ROOT/$src" "$EXTRA_SRC" -o "$out"
  echo "Running $out"
  "$out"
}

echo "Repository root: $REPO_ROOT"
echo "Build dir: $BUILD_DIR"

# Run available no-gtest tests (cc variant)
compile_and_run "tests/test_arena.cc" "$BUILD_DIR/test_arena_no_gtest"
compile_and_run "tests/test_map_harvest.cc" "$BUILD_DIR/test_map_harvest_no_gtest"
compile_and_run "tests/test_indirect.cc" "$BUILD_DIR/test_indirect_no_gtest"
compile_and_run "tests/sanity_check.cc" "$BUILD_DIR/sanity_check"
compile_and_run "tests/engine_check.cc" "$BUILD_DIR/engine_check"

# Newly added tests
compile_and_run "tests/test_blob_freelist.cc" "$BUILD_DIR/test_blob_freelist"
compile_and_run "tests/test_stress_churn.cc" "$BUILD_DIR/test_stress_churn"

# -- Added HNSW layout unit test --
compile_and_run "tests/test_hnsw_layout.cc" "$BUILD_DIR/test_hnsw_layout"

# Fallback: if gtests exist but no no-gtest version, try compiling them (may fail without GTest)
if [ ! -f "$BUILD_DIR/test_arena_no_gtest" ] && [ -f "$REPO_ROOT/tests/test_arena.cc" ]; then
  echo "Found tests/test_arena.cc - attempting compile (may require gtest)"
  "$CXX" $CXXFLAGS "$REPO_ROOT/tests/test_arena.cc" "$EXTRA_SRC" -o "$BUILD_DIR/test_arena" || true
  "$BUILD_DIR/test_arena" || true
fi

echo "Done."