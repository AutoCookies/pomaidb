#!/bin/bash
# Simple fuzzing harness for PomaiDB
# Tests random operation sequences to find edge cases

set -e

DB_PATH="/tmp/pomai_fuzz_db"
ITERATIONS=1000
SEED=${RANDOM}

echo "=========================================="
echo " PomaiDB Fuzzing Harness"
echo "=========================================="
echo "Iterations: $ITERATIONS"
echo "Seed: $SEED"
echo "DB Path: $DB_PATH"
echo "=========================================="
echo ""

# Clean start
rm -rf "$DB_PATH"

# Build fuzzer if needed
if [ ! -f "./build/tests/fuzz_operations" ]; then
    echo "Building fuzzer..."
    cat > /tmp/fuzz_operations.cc << 'EOF'
#include "pomai/pomai.h"
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <db_path> <seed>\n";
        return 1;
    }
    
    std::string db_path = argv[1];
    uint32_t seed = std::stoul(argv[2]);
    std::mt19937 rng(seed);
    
    // Open DB
    pomai::DBOptions opt;
    opt.path = db_path;
    opt.dim = 16;
    opt.shard_count = 2;
    opt.fsync = pomai::FsyncPolicy::kNever;
    
    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opt, &db);
    if (!st.ok()) {
        std::cerr << "Open failed: " << st.message() << "\n";
        return 1;
    }
    
    pomai::MembraneSpec spec;
    spec.name = "fuzz";
    spec.dim = 16;
    spec.shard_count = 2;
    db->CreateMembrane(spec);
    db->OpenMembrane("fuzz");
    
    // Random operations
    std::uniform_int_distribution<int> op_dist(0, 4);
    std::uniform_int_distribution<uint64_t> id_dist(0, 999);
    
    for (int i = 0; i < 10000; ++i) {
        int op = op_dist(rng);
        uint64_t id = id_dist(rng);
        
        try {
            switch (op) {
                case 0: { // Put
                    std::vector<float> vec(16);
                    for (auto& v : vec) v = static_cast<float>(rng() % 100);
                    db->Put("fuzz", id, vec);
                    break;
                }
                case 1: { // Delete
                    db->Delete("fuzz", id);
                    break;
                }
                case 2: { // Get
                    std::vector<float> out;
                    db->Get("fuzz", id, &out);
                    break;
                }
                case 3: { // Search
                    std::vector<float> query(16, 1.0f);
                    pomai::SearchResult res;
                    db->Search("fuzz", query, 10, &res);
                    break;
                }
                case 4: { // Freeze
                    if (i % 1000 == 0) db->Freeze("fuzz");
                    break;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception on iteration " << i << ": " << e.what() << "\n";
            return 1;
        }
    }
    
    db->Close();
    std::cout << "Fuzzing complete: " << seed << " iterations OK\n";
    return 0;
}
EOF
    
    g++ -std=c++20 /tmp/fuzz_operations.cc -o ./build/fuzz_operations \
        -I./include -L./build -lpomai -lpthread || {
        echo "Build failed, trying with installed pomai..."
        exit 1
    }
fi

# Run fuzzer
echo "Running fuzzer..."
LD_LIBRARY_PATH=./build ./build/fuzz_operations "$DB_PATH" "$SEED"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo " ✅ FUZZING PASSED"
    echo "=========================================="
    exit 0
else
    echo ""
    echo "=========================================="
    echo " ❌ FUZZING FAILED"
    echo "=========================================="
    exit 1
fi
