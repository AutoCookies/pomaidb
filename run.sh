#!/usr/bin/env bash
# run.sh
# Launches build/pomai_server in the background and attaches a pomai_cli session.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
SERVER_BIN="${BUILD_DIR}/pomai_server"
CLI_BIN="${BUILD_DIR}/pomai_cli"

# Check server binary
if [ ! -x "$SERVER_BIN" ]; then
  echo "Error: $SERVER_BIN not found or not executable. Please build first with ./build.sh." >&2
  exit 1
fi

# Start server in background, log output to build/server.log
SERVER_LOG="${BUILD_DIR}/server.log"
echo "[run.sh] Starting Pomai server in background ..."
"$SERVER_BIN" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[run.sh] Pomai server PID: $SERVER_PID (logs: $SERVER_LOG)"

# Wait for server to be up (simple port wait, e.g. 7777)
if [ -n "${POMAI_PORT:-}" ]; then
  PORT_TO_CHECK="$POMAI_PORT"
elif [ -n "${PORT:-}" ]; then
  PORT_TO_CHECK="$PORT"
else
  PORT_TO_CHECK="7777"
fi
MAX_WAIT=10
wait_time=0
while ! nc -z 127.0.0.1 $PORT_TO_CHECK >/dev/null 2>&1; do
  if [ $wait_time -ge $MAX_WAIT ]; then
    echo "Error: Pomai server did not start on port $PORT_TO_CHECK within $MAX_WAIT seconds."
    kill $SERVER_PID
    exit 2
  fi
  sleep 1
  wait_time=$((wait_time + 1))
done

echo "[run.sh] Pomai server is up on port $PORT_TO_CHECK."

# Check cli binary
if [ ! -x "$CLI_BIN" ]; then
  echo "Error: $CLI_BIN not found or not executable. Please build first." >&2
  kill $SERVER_PID
  exit 1
fi

# Run CLI (in current shell/terminal, attached)
echo "[run.sh] Attaching pomai_cli ..."
"$CLI_BIN"

# When CLI exits, gracefully stop the server
echo "[run.sh] CLI exited. Stopping Pomai server (PID $SERVER_PID)."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true
echo "[run.sh] Bye!"
