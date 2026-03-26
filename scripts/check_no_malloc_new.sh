#!/usr/bin/env bash
# CI / policy check: PomaiDB owned code must not use raw malloc/new.
#
# Note: we intentionally exclude vendored AI/runtime stacks under src/ai/*
# and third_party/* because those upstream sources may legitimately use raw
# allocators. This policy is about PomaiDB "core/owned" code paths.
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

if ! command -v rg >/dev/null 2>&1; then
  echo "error: ripgrep (rg) is required"
  exit 2
fi

declare -a PATHS=(
  "src/api"
  "src/capi"
  "src/core"
  "src/storage"
  "src/table"
  "src/compute"
  "src/util"
  "benchmarks"
  "examples"
  "include/pomai"
  "include/palloc_page_pool.h"
)

RAW_ALLOC_PATTERN='^\s*(?!//)(?!/\*)(?!\*).*\b(malloc|calloc|realloc|free)\s*\('
OP_NEW_PATTERN='\bnew\s+[A-Za-z_:][A-Za-z0-9_:<>,]*\s*(\(|\{)'
OP_NEW_ARRAY_PATTERN='\bnew\s+[A-Za-z_:][A-Za-z0-9_:<>,]*\s*\['
GLOBAL_NEW_DELETE_PATTERN='::operator\s+(new|delete)\b'

FAIL=0
for p in "${PATHS[@]}"; do
  [[ -e "$p" ]] || continue

  if rg -n --pcre2 --glob '*.{c,cc,cpp,h,hpp}' "$RAW_ALLOC_PATTERN" "$p" >/dev/null; then
    echo "error: raw C allocator usage found in $p"
    rg -n --pcre2 --glob '*.{c,cc,cpp,h,hpp}' "$RAW_ALLOC_PATTERN" "$p" || true
    FAIL=1
  fi

  if rg -n --pcre2 --glob '*.{cc,cpp,h,hpp}' "$OP_NEW_PATTERN" "$p" >/dev/null; then
    echo "error: operator new usage found in $p (use make_unique or placement new with palloc_malloc_aligned)"
    rg -n --pcre2 --glob '*.{cc,cpp,h,hpp}' "$OP_NEW_PATTERN" "$p" || true
    FAIL=1
  fi

  if rg -n --pcre2 --glob '*.{cc,cpp,h,hpp}' "$OP_NEW_ARRAY_PATTERN" "$p" >/dev/null; then
    echo "error: operator new[] usage found in $p (avoid raw new[] allocations)"
    rg -n --pcre2 --glob '*.{cc,cpp,h,hpp}' "$OP_NEW_ARRAY_PATTERN" "$p" || true
    FAIL=1
  fi

  if rg -n --pcre2 --glob '*.{cc,cpp,h,hpp}' "$GLOBAL_NEW_DELETE_PATTERN" "$p" >/dev/null; then
    echo "error: explicit ::operator new/delete usage found in $p"
    rg -n --pcre2 --glob '*.{cc,cpp,h,hpp}' "$GLOBAL_NEW_DELETE_PATTERN" "$p" || true
    FAIL=1
  fi
done

if [[ $FAIL -ne 0 ]]; then
  echo "Policy: no raw malloc/calloc/realloc/free or operator new in owned PomaiDB paths."
  exit 1
fi

echo "check_no_malloc_new: ok (allocator-clean in owned code paths)"
