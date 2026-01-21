#!/usr/bin/env bash
# tests/build_tests.sh
#
# Build-only test helper for the Pomai repo.
# - Compiles all src/*.cc/.cpp into object files (incremental by timestamp)
# - Packs them into a static archive build/libsrc.a
# - Compiles every test source under tests/ (recursively) and emits executables
#   into build/tests/
#
# Usage:
#   ./tests/build_tests.sh            # default: builds with reasonable flags
#   JOBS=8 CXX=clang++ ./tests/build_tests.sh
#   EXTRA_LIBS="-lm -ldl" ./tests/build_tests.sh
#
# Notes:
# - This is intentionally simple and conservative: timestamp-based incremental
#   builds, no fancy dependency scanning. It's intended for CI/dev quick builds.
# - It does NOT run the tests.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Build layout
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build/tests_build}"
OBJ_DIR="${OBJ_DIR:-${BUILD_DIR}/obj}"
ARCHIVE="${ARCHIVE:-${BUILD_DIR}/libsrc.a}"
BIN_DIR="${BIN_DIR:-${BUILD_DIR}/tests}"

# Toolchain / flags (can be overridden from environment)
CXX="${CXX:-g++}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 1)}"
CXXFLAGS="${CXXFLAGS:--std=c++17 -O2 -g -I${REPO_ROOT} -I${REPO_ROOT}/src -pthread -Wall -Wextra}"
LDFLAGS="${LDFLAGS:-}"
EXTRA_LIBS="${EXTRA_LIBS:-}"

echo "Building tests (repo root: ${REPO_ROOT})"
echo "Compiler: ${CXX}"
echo "CXXFLAGS: ${CXXFLAGS}"
echo "Jobs: ${JOBS}"
echo "Build dir: ${BUILD_DIR}"
echo

mkdir -p "${OBJ_DIR}"
mkdir -p "${BIN_DIR}"

# Collect source files from src/ (exclude tests inside src/ if any)
mapfile -d '' SRC_FILES < <(find "${REPO_ROOT}/src" -type f \( -name '*.cc' -o -name '*.cpp' \) -print0)

if [ ${#SRC_FILES[@]} -eq 0 ]; then
  echo "No source files found under src/ - nothing to build."
  exit 1
fi

# Helper to compute object path for a source file
obj_path_for() {
  local src="$1"
  # produce a filesystem-safe object filename preserving directory structure
  local rel
  rel="$(realpath --relative-to="${REPO_ROOT}" "$src")"
  # replace / with _ and append .o
  local out="${OBJ_DIR}/$(echo "$rel" | tr '/' '_' | sed -E 's/\.[^.]+$/.o/')"
  echo "$out"
}

# Compile changed sources (timestamp check)
echo "Compiling source files..."
compile_count=0
for src in "${SRC_FILES[@]}"; do
  out="$(obj_path_for "$src")"
  out_dir="$(dirname "$out")"
  mkdir -p "$out_dir"
  if [ ! -f "$out" ] || [ "$src" -nt "$out" ]; then
    echo " CXX -> $(realpath --relative-to="${REPO_ROOT}" "$src")"
    "${CXX}" ${CXXFLAGS} -c "$src" -o "$out"
    compile_count=$((compile_count + 1))
  fi
done

echo "Compiled ${compile_count} source files (or reused up-to-date objects)."

# Build static archive
echo "Creating archive ${ARCHIVE}..."
# collect object files list
mapfile -d '' OBJ_LIST < <(find "${OBJ_DIR}" -type f -name '*.o' -print0)
if [ ${#OBJ_LIST[@]} -eq 0 ]; then
  echo "ERROR: no object files found in ${OBJ_DIR}"
  exit 1
fi

# create archive
rm -f "${ARCHIVE}"
ar rcs "${ARCHIVE}" "${OBJ_LIST[@]}"
echo "Archive created: ${ARCHIVE} (size: $(stat -c%s "${ARCHIVE}") bytes)"

# Find test sources recursively under tests/
TEST_DIR="${REPO_ROOT}/tests"
if [ ! -d "${TEST_DIR}" ]; then
  echo "No tests/ directory found; nothing to build."
  exit 0
fi

mapfile -d '' TEST_SOURCES < <(find "${TEST_DIR}" -type f \( -name '*.cc' -o -name '*.cpp' \) -print0)

if [ ${#TEST_SOURCES[@]} -eq 0 ]; then
  echo "No test sources (*.cc, *.cpp) found under tests/"
  exit 0
fi

# Compile each test source into an executable
echo
echo "Building tests (executables) into ${BIN_DIR} ..."
built=0
failed=0
for testsrc in "${TEST_SOURCES[@]}"; do
  rel="$(realpath --relative-to="${REPO_ROOT}" "${testsrc}")"
  tname="$(echo "${rel}" | tr '/' '_' | sed -E 's/\.[^.]+$//')"
  outbin="${BIN_DIR}/${tname}"
  echo " TEST -> ${rel} -> ${outbin}"
  mkdir -p "$(dirname "${outbin}")"
  set +e
  "${CXX}" ${CXXFLAGS} "${testsrc}" "${ARCHIVE}" ${LDFLAGS} -o "${outbin}" ${EXTRA_LIBS}
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "  [ERROR] build failed for ${rel}"
    failed=$((failed+1))
  else
    chmod +x "${outbin}"
    built=$((built+1))
  fi
done

echo
echo "Build summary: ${built} tests built, ${failed} failed."
if [ $failed -ne 0 ]; then
  echo "Some test builds failed. Inspect the output above."
  exit 2
fi

echo "All test binaries are available under: ${BIN_DIR}"
echo "To run a test: ${BIN_DIR}/<path_sanitized_test_name>"
exit 0