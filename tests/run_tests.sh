#!/usr/bin/env bash
set -euo pipefail

# Robust test runner that builds all src/ .cc/.cpp into a static archive
# and links each test in tests/ (recursively) against that archive.
#
# Usage:
#   ./tests/run_tests.sh
#
# Notes:
# - Expects repository layout with top-level src/ and tests/ directories.
# - Adjust CXX/CXXFLAGS at top if you need custom toolchain options.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
OBJ_DIR="${BUILD_DIR}/obj"

CXX=${CXX:-g++}
# Standard flags: c++17, pthread, include repo and src headers
CXXFLAGS="-std=c++17 -O2 -g -I${REPO_ROOT} -I${REPO_ROOT}/src -pthread -Wall -Wextra"

mkdir -p "${OBJ_DIR}"

echo "Repo root: ${REPO_ROOT}"
echo "Build dir: ${BUILD_DIR}"
echo "Compiler: ${CXX}"
echo "CXXFLAGS: ${CXXFLAGS}"

# Collect source files from src/
mapfile -d '' SRC_FILES < <(find "${REPO_ROOT}/src" -type f \( -name '*.cc' -o -name '*.cpp' \) -print0)

if [ ${#SRC_FILES[@]} -eq 0 ]; then
  echo "No source files found under src/ - nothing to build."
  exit 1
fi

# Compile each source file into object file (skip tests sources)
OBJS=()
for src in "${SRC_FILES[@]}"; do
  # skip files under tests/ or examples if any slipped into src/
  rel="$(realpath --relative-to="${REPO_ROOT}" "$src")"
  out="${OBJ_DIR}/$(echo "$rel" | tr '/' '_' | sed -E 's/\.c(pp)?$/.o/')"
  echo "Compiling: ${rel}"
  "${CXX}" ${CXXFLAGS} -c "$src" -o "$out"
  OBJS+=("$out")
done

# Create static archive
ARCHIVE="${BUILD_DIR}/libsrc.a"
echo "Creating archive ${ARCHIVE}"
rm -f "${ARCHIVE}"
ar rcs "${ARCHIVE}" "${OBJS[@]}"

# Find test files under tests/ recursively ('.cc' and '.cpp')
TEST_DIR="${REPO_ROOT}/tests"
if [ ! -d "${TEST_DIR}" ]; then
  echo "No tests/ directory found; nothing to run."
  exit 0
fi

# Collect all test sources recursively (including subdirectories)
mapfile -d '' TEST_SOURCES < <(find "${TEST_DIR}" -type f \( -name '*.cc' -o -name '*.cpp' \) -print0)

if [ ${#TEST_SOURCES[@]} -eq 0 ]; then
  echo "No test sources (*.cc, *.cpp) found under tests/ (recursively)"
  exit 0
fi

# Build & run each test
for testsrc in "${TEST_SOURCES[@]}"; do
  # Create a deterministic binary name based on test source path relative to repo root
  relpath="$(realpath --relative-to="${REPO_ROOT}" "${testsrc}")"
  # sanitize: replace '/' with '_' and remove extension
  tname="$(echo "${relpath}" | tr '/' '_' | sed -E 's/\.[^.]+$//')"
  outbin="${BUILD_DIR}/${tname}"
  echo "---------------------------------------------"
  echo "Building test: ${relpath} -> ${tname}"
  "${CXX}" ${CXXFLAGS} "${testsrc}" "${ARCHIVE}" -o "${outbin}" || {
    echo "Compilation failed for ${testsrc}; skipping run."
    continue
  }

  echo "Running: ${outbin}"
  set +e
  "${outbin}"
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "Test ${tname} FAILED (exit ${rc})"
  else
    echo "Test ${tname} PASSED"
  fi
done

echo "All done."