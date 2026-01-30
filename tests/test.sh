#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

# Build everything (use existing build script)
cd "${ROOT_DIR}"
./build.sh

WAL_UNIT="${BUILD_DIR}/wal_unit_tests"
WAL_WRITER="${BUILD_DIR}/wal_integration_writer"
WAL_INSPECT="${BUILD_DIR}/wal_replay_inspect"

if [[ ! -x "${WAL_UNIT}" ]]; then
    echo "Unit test binary not found: ${WAL_UNIT}"
    exit 2
fi

echo "=== Running WAL unit tests ==="
"${WAL_UNIT}"
echo "=== WAL unit tests completed ==="

echo "=== Running integration sync writer test (deterministic) ==="
TMPDIR=$(mktemp -d /tmp/pomai_integ.XXXXXX)
echo "Using temp dir: ${TMPDIR}"

# launch writer that writes 20 batches of 10 vectors with wait_durable=1 (sync)
"${WAL_WRITER}" "${TMPDIR}" 20 10 1 > "${TMPDIR}/writer.log" 2>&1 &
WR_PID=$!
# wait for writer to finish
wait ${WR_PID}

echo "Writer finished; now running replay inspect..."
"${WAL_INSPECT}" "${TMPDIR}" 8

# Check replay output contains expected vector count (20 * 10 = 200)
REPLAY_LINE=$( "${WAL_INSPECT}" "${TMPDIR}" 8 )
echo "${REPLAY_LINE}"
VECTORS=$(echo "${REPLAY_LINE}" | sed -n 's/.*vectors_applied=\([0-9]*\).*/\1/p')
if [[ "${VECTORS}" != "200" ]]; then
    echo "Integration sync test FAILED: expected 200 vectors, got ${VECTORS}"
    exit 3
fi
echo "Integration sync test PASSED"

# Optional non-deterministic kill-9 scenario: start writer async and kill mid-run
echo "=== Running optional kill-9 async test (informational) ==="
TMPDIR2=$(mktemp -d /tmp/pomai_integ_async.XXXXXX)
echo "Using temp dir: ${TMPDIR2}"
# run writer with wait_durable=0
"${WAL_WRITER}" "${TMPDIR2}" 100 10 0 > "${TMPDIR2}/writer.log" 2>&1 &
PID=$!
# sleep shortly then kill -9
sleep 0.25
echo "Killing writer (${PID}) with SIGKILL..."
kill -9 ${PID} || true
wait ${PID} 2>/dev/null || true

echo "After hard kill, replay stats:"
"${WAL_INSPECT}" "${TMPDIR2}" 8 || true

echo "All tests run. Note: kill-9 async test is informational and may be non-deterministic."
exit 0