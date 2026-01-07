#!/usr/bin/env bash
# build.sh - build libsrc.a and server binary (build/pomai_server)
#
# Usage:
#   ./build.sh            # builds libsrc.a and server (prefers src/main.cc, falls back to examples/main.cc)
#   ./build.sh clean      # remove build/ artifacts
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
SRC_DIR="${REPO_ROOT}/src"
EXAMPLES_DIR="${REPO_ROOT}/examples"
BUILD_DIR="${REPO_ROOT}/build"
OBJ_DIR="${BUILD_DIR}/obj"

CXX=${CXX:-g++}
CXXFLAGS="-std=c++17 -O2 -g -I${REPO_ROOT} -I${SRC_DIR} -pthread -Wall -Wextra"

mkdir -p "${BUILD_DIR}" "${OBJ_DIR}"

if [ "${1:-}" = "clean" ]; then
  echo "Cleaning ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
  exit 0
fi

echo "Compiler: ${CXX}"
echo "CXXFLAGS: ${CXXFLAGS}"

# Find source files under src/ (exclude tests/)
mapfile -d '' SRC_FILES < <(find "${SRC_DIR}" -type f \( -name '*.cc' -o -name '*.cpp' \) -not -path "${SRC_DIR}/tests/*" -print0)

if [ ${#SRC_FILES[@]} -eq 0 ]; then
  echo "No source files found under src/ - nothing to build."
  exit 1
fi

# Compile each source file into object file with deterministic sanitized name
OBJS=()
for src in "${SRC_FILES[@]}"; do
  rel="$(realpath --relative-to="${REPO_ROOT}" "$src")"
  out="${OBJ_DIR}/$(echo "$rel" | tr '/' '_' | sed -E 's/\.c(pp)?$/.o/')"
  echo "Compiling: ${rel} -> ${out}"
  mkdir -p "$(dirname "$out")"
  "${CXX}" ${CXXFLAGS} -c "$src" -o "$out"
  OBJS+=("$out")
done

# Create static archive
ARCHIVE="${BUILD_DIR}/libsrc.a"
echo "Creating archive ${ARCHIVE}"
rm -f "${ARCHIVE}"
ar rcs "${ARCHIVE}" "${OBJS[@]}"

# Determine which main source to link (prefer src/main.cc)
SERVER_SRC_CANDIDATES=("${SRC_DIR}/main.cc" "${EXAMPLES_DIR}/main.cc")
SERVER_SRC=""
for c in "${SERVER_SRC_CANDIDATES[@]}"; do
  if [ -f "${c}" ]; then
    SERVER_SRC="${c}"
    break
  fi
done

if [ -z "${SERVER_SRC}" ]; then
  echo "No main.cc found in src/ or examples/; built archive only: ${ARCHIVE}"
  exit 0
fi

OUT_BIN="${BUILD_DIR}/pomai_server"
echo "Linking server from ${SERVER_SRC} -> ${OUT_BIN}"
"${CXX}" ${CXXFLAGS} "${SERVER_SRC}" "${ARCHIVE}" -o "${OUT_BIN}"

echo "Build complete."
echo "  archive: ${ARCHIVE}"
echo "  server:  ${OUT_BIN}"