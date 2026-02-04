#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TYPE="release"
COMPILER=""
GENERATOR=""
JOBS="$(nproc)"

usage() {
  echo "Usage: $0 [release|debug|asan|tsan] [--clang|--gcc]"
  echo
  echo "Examples:"
  echo "  ./scripts/build.sh"
  echo "  ./scripts/build.sh debug"
  echo "  ./scripts/build.sh asan --clang"
  echo "  ./scripts/build.sh tsan --clang"
  exit 1
}

# -------- parse args --------
for arg in "$@"; do
  case "$arg" in
    release|debug|asan|tsan)
      BUILD_TYPE="$arg"
      ;;
    --clang)
      COMPILER="clang"
      ;;
    --gcc)
      COMPILER="gcc"
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown arg: $arg"
      usage
      ;;
  esac
done

# -------- generator --------
if command -v ninja >/dev/null 2>&1; then
  GENERATOR="Ninja"
else
  GENERATOR="Unix Makefiles"
fi

# -------- compiler --------
CMAKE_COMPILER_ARGS=()
if [[ "$COMPILER" == "clang" ]]; then
  CMAKE_COMPILER_ARGS+=(
    -DCMAKE_CXX_COMPILER=clang++
    -DCMAKE_C_COMPILER=clang
  )
elif [[ "$COMPILER" == "gcc" ]]; then
  CMAKE_COMPILER_ARGS+=(
    -DCMAKE_CXX_COMPILER=g++
    -DCMAKE_C_COMPILER=gcc
  )
fi

# -------- build config --------
CMAKE_BUILD_TYPE="Release"
CMAKE_EXTRA_FLAGS=()

case "$BUILD_TYPE" in
  release)
    CMAKE_BUILD_TYPE="Release"
    ;;
  debug)
    CMAKE_BUILD_TYPE="Debug"
    ;;
  asan)
    CMAKE_BUILD_TYPE="RelWithDebInfo"
    CMAKE_EXTRA_FLAGS+=(
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
    )
    ;;
  tsan)
    CMAKE_BUILD_TYPE="RelWithDebInfo"
    CMAKE_EXTRA_FLAGS+=(
      -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer"
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"
    )
    ;;
  *)
    usage
    ;;
esac

BUILD_DIR="${ROOT_DIR}/build/${BUILD_TYPE}"

echo "== Pomai build =="
echo "  type      : ${BUILD_TYPE}"
echo "  generator : ${GENERATOR}"
echo "  compiler  : ${COMPILER:-default}"
echo "  build dir : ${BUILD_DIR}"
echo

mkdir -p "${BUILD_DIR}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -G "${GENERATOR}" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
  -DPOMAI_BUILD_TESTS=ON \
  "${CMAKE_COMPILER_ARGS[@]}" \
  "${CMAKE_EXTRA_FLAGS[@]}"

cmake --build "${BUILD_DIR}" -j"${JOBS}"

echo
echo "Build OK → ${BUILD_DIR}"
