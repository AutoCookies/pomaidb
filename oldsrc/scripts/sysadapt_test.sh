#!/usr/bin/env bash
set -euo pipefail

echo "=== SYSADAPT TEST (Linux) ==="
echo "Date: $(date)"
echo
echo "--- /sys/fs/cgroup/memory.max ---"
if [ -f /sys/fs/cgroup/memory.max ]; then
  cat /sys/fs/cgroup/memory.max || true
else
  echo "(not present)"
fi
echo
echo "--- /sys/fs/cgroup/memory/memory.limit_in_bytes ---"
if [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
  cat /sys/fs/cgroup/memory/memory.limit_in_bytes || true
else
  echo "(not present)"
fi
echo
echo "--- /proc/meminfo (MemTotal) ---"
grep -i MemTotal /proc/meminfo || true
echo
echo "Run server with verbose logs to observe SYSADAPT behavior:"
echo "  ./pomai-server"
echo
echo "Or force mem-limit sentinel:"
echo "  ./pomai-server --mem-limit=4GB"