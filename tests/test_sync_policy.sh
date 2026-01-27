#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

# Binaries
WRITER_BIN="${BUILD_DIR}/pomai_db_writer"
INSPECT_BIN="${BUILD_DIR}/wal_replay_inspect"

if [[ ! -x "${WRITER_BIN}" ]]; then
    echo "Writer binary not found: ${WRITER_BIN}"
    exit 2
fi
if [[ ! -x "${INSPECT_BIN}" ]]; then
    echo "Inspect binary not found: ${INSPECT_BIN}"
    exit 2
fi

# Parameters
NUM_BATCHES=20
BATCH_SIZE=10
DIM=8
KILL_AT_BATCH=10

run_and_kill() {
    local allow_sync="$1"
    local tmpdir
    tmpdir="$(mktemp -d /tmp/pomai_sync_test.XXXXXX)"
    echo "Using tmpdir: ${tmpdir}"

    # Launch writer in background, redirect output
    "${WRITER_BIN}" "${tmpdir}" "${allow_sync}" "${NUM_BATCHES}" "${BATCH_SIZE}" "${DIM}" > "${tmpdir}/writer.log" 2>&1 &
    local pid=$!

    echo "Writer pid=${pid}, waiting for batch ${KILL_AT_BATCH} marker..."

    # Wait for status file to contain the desired marker
    local status_file="${tmpdir}/writer.status"
    local found=0
    for i in {1..200}; do
        if [[ -f "${status_file}" ]]; then
            if grep -q "^batch ${KILL_AT_BATCH}$" "${status_file}"; then
                found=1
                break
            fi
        fi
        sleep 0.05
    done

    if [[ $found -ne 1 ]]; then
        echo "Timed out waiting for status marker. Dumping writer log:"
        tail -n +1 "${tmpdir}/writer.log" || true
        kill -9 ${pid} || true
        wait ${pid} 2>/dev/null || true
        exit 3
    fi

    echo "Marker seen; killing writer (SIGKILL)..."
    kill -9 ${pid} || true
    wait ${pid} 2>/dev/null || true

    echo "Running replay inspect on ${tmpdir} ..."
    "${INSPECT_BIN}" "${tmpdir}" "${DIM}"
    echo "Replay done for allow_sync=${allow_sync}"

    # Return tmpdir for manual inspection if needed
    echo "${tmpdir}"
}

echo "=== Test 1: allow_sync_on_append = 1 (should preserve batches up to kill point) ==="
td1=$(run_and_kill 1)
# parse vectors_applied from inspect
out1="$("${INSPECT_BIN}" "${td1}" "${DIM}" 2>/dev/null || true)"
echo "Inspect output: ${out1}"
vecs1=$(echo "${out1}" | sed -n 's/.*vectors_applied=\([0-9]*\).*/\1/p' || true)
echo "Vectors recovered (allow=1): ${vecs1}"

echo "=== Test 2: allow_sync_on_append = 0 (client sync requests denied; some loss expected) ==="
td2=$(run_and_kill 0)
out2="$("${INSPECT_BIN}" "${td2}" "${DIM}" 2>/dev/null || true)"
echo "Inspect output: ${out2}"
vecs2=$(echo "${out2}" | sed -n 's/.*vectors_applied=\([0-9]*\).*/\1/p' || true)
echo "Vectors recovered (allow=0): ${vecs2}"

echo
echo "Summary: allow=1 -> ${vecs1} vectors, allow=0 -> ${vecs2} vectors"
if [[ -z "${vecs1}" || -z "${vecs2}" ]]; then
    echo "Could not parse inspect outputs; check logs in ${td1} and ${td2}"
    exit 4
fi

if (( vecs1 >= KILL_AT_BATCH * BATCH_SIZE )); then
    echo "PASS: allow=1 recovered at least ${KILL_AT_BATCH} batches."
else
    echo "FAIL: allow=1 recovered fewer than expected (${vecs1} < $((KILL_AT_BATCH * BATCH_SIZE)))."
    exit 5
fi

if (( vecs2 < KILL_AT_BATCH * BATCH_SIZE )); then
    echo "PASS: allow=0 lost some data as expected (vectors ${vecs2} < ${KILL_AT_BATCH * BATCH_SIZE})."
else
    echo "WARNING: allow=0 recovered as many or more vectors (${vecs2}) than expected (${KILL_AT_BATCH * BATCH_SIZE}). This can happen on certain filesystems where kernel flushed buffers quickly."
    echo "Inspect writer logs: ${td2}/writer.log"
fi

echo "Tests complete. Temporary dirs: ${td1} ${td2}"
exit 0