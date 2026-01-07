#!/usr/bin/env bash
# Small example client using curl (Linux/macOS)
set -euo pipefail

HOST="${HOST:-http://localhost:8080}"
TENANT="${TENANT:-tenant1}"
KEY="${1:-example-key}"
FILE="${2:-}"

echo "PUT (string) -> $HOST/v1/cache/$KEY"
curl -v -X PUT -H "X-Tenant-ID: $TENANT" --data-binary "hello from cli at $(date)" "$HOST/v1/cache/$KEY?ttl=60"
echo

echo "GET -> $HOST/v1/cache/$KEY"
curl -v -H "X-Tenant-ID: $TENANT" "$HOST/v1/cache/$KEY" || true
echo

echo "INCR -> $HOST/v1/cache/counter/incr?delta=1"
curl -v -X POST -H "X-Tenant-ID: $TENANT" "$HOST/v1/cache/counter/incr?delta=1"
echo