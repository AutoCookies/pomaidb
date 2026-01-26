#!/usr/bin/env bash
# build.sh - build libsrc.a and server binary (build/pomai_server)
# FIX: Added support for .c files (crc64.c, xxhash.c) using GCC
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
SRC_DIR="${REPO_ROOT}/src"
EXAMPLES_DIR="${REPO_ROOT}/examples"
BUILD_DIR="${REPO_ROOT}/build"
OBJ_DIR="${BUILD_DIR}/obj"

# Toolchain
CC=${CC:-gcc}
CXX=${CXX:-g++}

# Flags
# C++: C++17
CXXFLAGS="-std=c++17 -O2 -g -I${REPO_ROOT} -I${SRC_DIR} -pthread -Wall -Wextra"
# C: C99 or C11
CFLAGS="-std=c99 -O2 -g -I${REPO_ROOT} -I${SRC_DIR} -pthread -Wall -Wextra"

mkdir -p "${BUILD_DIR}" "${OBJ_DIR}"

if [ "${1:-}" = "clean" ]; then
  echo "Cleaning ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
  exit 0
fi

echo "CXX: ${CXX}"
echo "CC:  ${CC}"

# 1. Find C++ Sources (.cc, .cpp)
mapfile -d '' CPP_FILES < <(find "${SRC_DIR}" -type f \( -name '*.cc' -o -name '*.cpp' \) -not -path "${SRC_DIR}/tests/*" -print0)

# 2. Find C Sources (.c)
mapfile -d '' C_FILES < <(find "${SRC_DIR}" -type f \( -name '*.c' \) -not -path "${SRC_DIR}/tests/*" -print0)

if [ ${#CPP_FILES[@]} -eq 0 ] && [ ${#C_FILES[@]} -eq 0 ]; then
  echo "No source files found in ${SRC_DIR}."
  exit 1
fi

OBJS=()

# Compile C++
for src in "${CPP_FILES[@]}"; do
  # Determine relative path for unique object name
  rel="$(realpath --relative-to="${REPO_ROOT}" "$src")"
  # Flatten path: src/core/main.cc -> src_core_main.o
  obj_name="$(echo "$rel" | tr '/' '_' | sed -E 's/\.(cc|cpp)$/.o/')"
  out="${OBJ_DIR}/${obj_name}"
  
  # Only compile if needed (simple timestamp check could go here, but we rebuild for safety)
  echo "Compiling C++: ${rel} -> ${out}"
  "${CXX}" ${CXXFLAGS} -c "$src" -o "$out"
  OBJS+=("$out")
done

# Compile C (CRC64, XXHASH...)
for src in "${C_FILES[@]}"; do
  rel="$(realpath --relative-to="${REPO_ROOT}" "$src")"
  obj_name="$(echo "$rel" | tr '/' '_' | sed -E 's/\.c$/.o/')"
  out="${OBJ_DIR}/${obj_name}"
  
  echo "Compiling C  : ${rel} -> ${out}"
  "${CC}" ${CFLAGS} -c "$src" -o "$out"
  OBJS+=("$out")
done

# Create Archive
ARCHIVE="${BUILD_DIR}/libsrc.a"
echo "Creating archive ${ARCHIVE}"
rm -f "${ARCHIVE}"
ar rcs "${ARCHIVE}" "${OBJS[@]}"

# Link Server
SERVER_SRC_CANDIDATES=("${SRC_DIR}/main.cc" "${EXAMPLES_DIR}/main.cc")
SERVER_SRC=""
for c in "${SERVER_SRC_CANDIDATES[@]}"; do
  if [ -f "${c}" ]; then
    SERVER_SRC="${c}"
    break
  fi
done

if [ -n "${SERVER_SRC}" ]; then
  OUT_BIN="${BUILD_DIR}/pomai_server"
  echo "Linking server from ${SERVER_SRC} -> ${OUT_BIN}"
  # Note: Use CXX to link since we have C++ code
  "${CXX}" ${CXXFLAGS} "${SERVER_SRC}" "${ARCHIVE}" -o "${OUT_BIN}"
  echo "Server built: ${OUT_BIN}"
else
  echo "No main.cc found, skipped server build."
fi

# Link CLI (Optional)
CLI_SRC="${SRC_DIR}/pomai_cli.cc"
CLI_BIN="${BUILD_DIR}/pomai_cli"
if [ -f "$CLI_SRC" ]; then
  echo "Linking CLI from ${CLI_SRC} -> ${CLI_BIN}"
  "${CXX}" ${CXXFLAGS} "$CLI_SRC" "${ARCHIVE}" -o "${CLI_BIN}"
  echo "CLI built:    ${CLI_BIN}"
fi