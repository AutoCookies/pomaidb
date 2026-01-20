#!/bin/bash
# run_benchmark.sh

SERVER_BIN="./build/pomai_server"
DATA_DIR="./data"
VEC_COUNT=50000 # 50k vectors ~ 150MB WAL

# Màu mè
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== POMAI CHECKPOINT & STRESS BENCHMARK ===${NC}"

# 0. Kiểm tra file thực thi
if [ ! -f "$SERVER_BIN" ]; then
    echo -e "${RED}Lỗi: Không tìm thấy $SERVER_BIN${NC}"
    echo "Hãy chạy 'make' thủ công trước!"
    exit 1
fi

# 1. Clean (Xóa dữ liệu cũ để test sạch)
echo -e "${CYAN}[STEP 0] Cleaning old data...${NC}"
rm -rf "$DATA_DIR"
rm -f orbit.wal
# make -j4 > /dev/null  <-- ĐÃ TẮT BUILD THEO YÊU CẦU

# 2. Start Server
echo -e "${CYAN}[STEP 1] Starting Server...${NC}"
$SERVER_BIN > benchmark_server.log 2>&1 &
PID=$!
echo "Server PID: $PID"
sleep 2

# 3. Ingest Data & Trigger Checkpoint
echo -e "${CYAN}[STEP 2] Running Ingest & Checkpoint Benchmark...${NC}"
python3 tests/test_durability.py --mode ingest --count $VEC_COUNT

if [ $? -ne 0 ]; then
    echo -e "${RED}Ingest failed. Killing server.${NC}"
    kill -9 $PID
    exit 1
fi

# 4. Restart Test (Fast Recovery)
echo -e "${CYAN}[STEP 3] Killing Server & Restarting (Testing Fast Recovery)...${NC}"
kill -9 $PID
sleep 1

echo "Starting server again (Expect instant start due to empty WAL)..."
start_time=$(date +%s%N)
$SERVER_BIN > benchmark_recovery.log 2>&1 &
NEW_PID=$!

# Wait for port open (max 10s)
for i in {1..100}; do
    if nc -z localhost 7777; then
        break
    fi
    sleep 0.1
done

end_time=$(date +%s%N)
elapsed=$(( (end_time - start_time) / 1000000 ))

echo -e "${GREEN}Server recovered in approx ${elapsed} ms${NC}"

# 5. Verify Data
echo -e "${CYAN}[STEP 4] Verifying Data Integrity...${NC}"
python3 tests/test_durability.py --mode verify --count $VEC_COUNT

# Cleanup
kill -9 $NEW_PID
echo -e "${GREEN}=== BENCHMARK COMPLETE ===${NC}"