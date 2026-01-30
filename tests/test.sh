#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-"$ROOT_DIR/build"}
JOBS=${JOBS:-"$(getconf _NPROCESSORS_ONLN || echo 4)"}

mkdir -p "$BUILD_DIR"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DPOMAI_BUILD_TESTS=ON
cmake --build "$BUILD_DIR" -j "$JOBS"

OUTPUT_FILE=$(mktemp)
set +e
ctest --output-on-failure -j "$JOBS" --test-dir "$BUILD_DIR" | tee "$OUTPUT_FILE"
CTEST_STATUS=${PIPESTATUS[0]}
set -e

passed=0
failed=0
failed_names=()

while IFS= read -r line; do
    if [[ $line =~ Test\ #[0-9]+:\ (.*)\ \.\.\.\ Passed ]]; then
        ((passed+=1))
    elif [[ $line =~ Test\ #[0-9]+:\ (.*)\ \.\.\.\ \*\*\*Failed ]]; then
        failed_names+=("${BASH_REMATCH[1]}")
        ((failed+=1))
    elif [[ $line =~ Test\ #[0-9]+:\ (.*)\ \.\.\.\ Failed ]]; then
        failed_names+=("${BASH_REMATCH[1]}")
        ((failed+=1))
    fi
done < "$OUTPUT_FILE"

GREEN="\033[0;32m"
RED="\033[0;31m"
NC="\033[0m"

echo -e "PASSED: ${GREEN}${passed}${NC}"
echo -e "FAILED: ${RED}${failed}${NC}"

if ((failed > 0)); then
    echo "Failed tests:"
    for name in "${failed_names[@]}"; do
        echo "  - ${name}"
    done
fi

rm -f "$OUTPUT_FILE"

if ((failed > 0)); then
    exit 1
fi

exit $CTEST_STATUS
