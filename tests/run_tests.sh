#!/usr/bin/env bash
set -euo pipefail

# Robust test runner that builds all src/ .cc/.cpp into a static archive
# and links each test in tests/ against that archive.
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

# Find test files under tests/ ('.cc' files)
TEST_DIR="${REPO_ROOT}/tests"
if [ ! -d "${TEST_DIR}" ]; then
  echo "No tests/ directory found; nothing to run."
  exit 0
fi

shopt -s nullglob
TEST_SOURCES=("${TEST_DIR}"/*.cc "${TEST_DIR}"/*.cpp)
if [ ${#TEST_SOURCES[@]} -eq 0 ]; then
  echo "No test sources (*.cc, *.cpp) found in tests/"
  exit 0
fi

# Build & run each test
for testsrc in "${TEST_SOURCES[@]}"; do
  tname="$(basename "${testsrc%.*}")"
  outbin="${BUILD_DIR}/${tname}"
  echo "---------------------------------------------"
  echo "Building test: ${tname}"
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