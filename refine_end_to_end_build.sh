#!/usr/bin/env bash
# refine_end_to_end_build.sh
#
# Build (incrementally) the project sources into build/libsrc.a and compile
# a single test (default: tests/refine/refine_end_to_end.cc).
#
# Usage:
#   ./refine_end_to_end_build.sh                 # build default test
#   ./refine_end_to_end_build.sh path/to/test.cc # build a specific test
#
# After the script finishes the test binary will be at:
#   build/<sanitized_test_path>
#
# To run with the debug prints added earlier:
#   POMAI_DEBUG_PREFILTER=1 ./build/tests_refine_refine_end_to_end
#
set -euo pipefail

# Allow override of CXX/CXXFLAGS from environment
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--std=c++17 -O2 -g -I$(pwd) -I$(pwd)/src -pthread -Wall -Wextra}"

TEST_SRC="${1:-tests/refine/refine_end_to_end.cc}"

REPO_ROOT="$(pwd)"
BUILD_DIR="${REPO_ROOT}/build"
OBJ_DIR="${BUILD_DIR}/obj"
ARCHIVE="${BUILD_DIR}/libsrc.a"

mkdir -p "${OBJ_DIR}"

echo "Repo root: ${REPO_ROOT}"
echo "Build dir: ${BUILD_DIR}"
echo "Compiler: ${CXX}"
echo "CXXFLAGS: ${CXXFLAGS}"
echo "Test source: ${TEST_SRC}"

# Find all project source files
mapfile -d '' SRC_FILES < <(find "${REPO_ROOT}/src" -type f \( -name '*.cc' -o -name '*.cpp' \) -print0)

if [ ${#SRC_FILES[@]} -eq 0 ]; then
  echo "No source files found under src/"
  exit 1
fi

# Compile only changed/absent object files (incremental)
for src in "${SRC_FILES[@]}"; do
  # create a stable object filename in build/obj
  rel="$(realpath --relative-to="${REPO_ROOT}" "$src")"
  out="${OBJ_DIR}/$(echo "$rel" | tr '/' '_' | sed -E 's/\.[^.]+$/.o/')"
  # compile if missing or if source is newer
  if [ ! -f "$out" ] || [ "$src" -nt "$out" ]; then
    echo "Compiling: ${rel} -> ${out}"
    "${CXX}" ${CXXFLAGS} -c "$src" -o "$out"
  else
    echo "Skipping (up-to-date): ${rel}"
  fi
done

# Create (or update) static archive
echo "Creating archive ${ARCHIVE}"
rm -f "${ARCHIVE}"
ar rcs "${ARCHIVE}" "${OBJ_DIR}"/*.o

# Prepare test output binary name (sanitize path -> replace / with _ and strip extension)
tname="$(echo "${TEST_SRC}" | tr '/' '_' | sed -E 's/\.[^.]+$//')"
outbin="${BUILD_DIR}/${tname}"

echo "Compiling test: ${TEST_SRC} -> ${outbin}"
"${CXX}" ${CXXFLAGS} "${TEST_SRC}" "${ARCHIVE}" -o "${outbin}"

echo "Build complete. Test binary: ${outbin}"
echo
echo "Run the test (with optional debug):"
echo "  POMAI_DEBUG_PREFILTER=1 ${outbin}"
echo "or just:"
echo "  ${outbin}"

# exit success
exit 0