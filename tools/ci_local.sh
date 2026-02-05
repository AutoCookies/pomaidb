#!/bin/bash
# Local CI Pipeline for PomaiDB
# Runs all quality checks that would run in CI/CD

set -e

echo "=========================================="
echo " PomaiDB Local CI Pipeline"
echo "=========================================="
echo ""

FAILED_STAGES=""

# Stage 1: Build
echo "▶ Stage 1/5: Building..."
if cmake --build build -j$(nproc) 2>&1 | tail -5; then
    echo "✅ Build PASSED"
else
    echo "❌ Build FAILED"
    FAILED_STAGES="$FAILED_STAGES Build"
fi
echo ""

# Stage 2: Unit & Integration Tests
echo "▶ Stage 2/5: Running tests..."
if ctest --test-dir build --output-on-failure -j$(nproc) -L integ; then
    echo "✅ Tests PASSED"
else
    echo "❌ Tests FAILED"
    FAILED_STAGES="$FAILED_STAGES Tests"
fi
echo ""

# Stage 3: Performance Gate
echo "▶ Stage 3/5: Performance validation..."
if ./tools/perf_gate.sh --dataset=small; then
    echo "✅ Performance PASSED"
else
    echo "⚠️  Performance gate failed (non-blocking)"
fi
echo ""

# Stage 4: Crash Recovery
echo "▶ Stage 4/5: Crash recovery tests..."
if ./build/recovery_test; then
    echo "✅ Recovery tests PASSED"
else
    echo "❌ Recovery tests FAILED"
    FAILED_STAGES="$FAILED_STAGES Recovery"
fi
echo ""

# Stage 5: Tools Check
echo "▶ Stage 5/5: Verifying tools..."
if ./build/pomai_inspect --help > /dev/null 2>&1; then
    echo "✅ Tools check PASSED"
else
    echo "❌ Tools check FAILED"
    FAILED_STAGES="$FAILED_STAGES Tools"
fi
echo ""

# Summary
echo "=========================================="
if [ -z "$FAILED_STAGES" ]; then
    echo " ✅ ALL STAGES PASSED"
    echo "=========================================="
    exit 0
else
    echo " ❌ FAILED STAGES:$FAILED_STAGES"
    echo "=========================================="
    exit 1
fi
