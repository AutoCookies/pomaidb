#!/usr/bin/env bash
# scripts/reorg_repo.sh
# Safe reorganization helper:
# - creates canonical dirs if missing
# - moves top-level module dirs into src/
# - moves tests into tests/unit or tests/integration if they look like unit/integration
# - moves oldsrc -> archive if present
#
# This script uses `git mv` when inside a git repository; otherwise falls back to `mv`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Repository root: $ROOT"

git_root=false
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_root=true
  echo "Git repo detected: using git mv where possible."
else
  echo "Not a git repo: using plain mv (no history preserved)."
fi

# ensure directories exist
mkdir -p src/ai src/core src/memory src/facade src/sharding apps tests/unit tests/integration scripts build archive include assets

# helper to move dir safely
move_dir() {
  local src=$1
  local dst=$2
  if [ -d "$src" ]; then
    echo "Moving $src -> $dst"
    if $git_root; then
      git mv -k "$src" "$dst" || mv -f "$src" "$dst"
    else
      mv -f "$src" "$dst"
    fi
  else
    echo "No $src (skipped)"
  fi
}

# Common top-level folders to move into src/
move_dir "ai" "src/ai"
move_dir "core" "src/core"
move_dir "memory" "src/memory"
move_dir "facade" "src/facade"
move_dir "sharding" "src/sharding"

# tests: try to detect existing test files and move
if [ -d "tests" ]; then
  echo "tests dir exists; moving into tests/unit by default"
  # move all files from tests root into tests/unit (preserves subdirs)
  if $git_root; then
    git mv -k tests/* tests/unit/ || true
  else
    mv -f tests/* tests/unit/ || true
  fi
fi

# oldsrc -> archive/
move_dir "oldsrc" "archive/oldsrc"

# If there's top-level C++ sources (e.g., .cc files), move to src/ (optional)
shopt -s nullglob
top_ccs=( *.cc *.cpp )
if [ ${#top_ccs[@]} -gt 0 ]; then
  echo "Moving top-level C/C++ sources into src/ (check these)"
  mkdir -p src/top
  for f in "${top_ccs[@]}"; do
    if $git_root; then git mv -k "$f" "src/top/" || mv -f "$f" "src/top/"; else mv -f "$f" "src/top/"; fi
  done
fi
shopt -u nullglob

echo "Reorganization finished. Please inspect changes, update CMakeLists and include paths as needed."
echo "Suggested next steps:"
echo "  - Open top-level CMakeLists.txt and per-module CMakeLists in src/*"
echo "  - Run: mkdir -p build && cd build && cmake .. && cmake --build ."