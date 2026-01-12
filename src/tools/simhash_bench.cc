// src/tools/simhash_bench.cc
//
// Small microbenchmark for SimHash.
// Measures compute() throughput (vecs/sec, ns/vec) and memory bandwidth estimate.
//
// Usage:
//   g++ -std=c++20 -O3 -march=native src/tools/simhash_bench.cc src/ai/simhash.cc -I. -pthread -o build/simhash_bench
//   ./build/simhash_bench --dim 512 --bits 512 --reps 20000 --batch 16
//
// Notes:
//  - The benchmark pre-generates random float vectors and reuses them to avoid RNG cost in timing.
//  - Runs a warmup phase then timed phase. Reports ops/sec and estimated memory read GB/s.

#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <cassert>
#include <getopt.h>

#include "src/ai/simhash.h"

using namespace std::chrono;
using namespace pomai::ai;

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [--dim N] [--bits B] [--reps R] [--batch K] [--seed S]\n";
    std::cerr << "  --dim N     : vector dimensionality (default 512)\n";
    std::cerr << "  --bits B    : fingerprint bits (default 512)\n";
    std::cerr << "  --reps R    : total vectors processed in timed loop (default 20000)\n";
    std::cerr << "  --batch K   : batch size per compute call to amortize overhead (default 8)\n";
    std::cerr << "  --seed S    : rng seed (default 1234)\n";
}

int main(int argc, char** argv) {
    size_t dim = 512;
    size_t bits = 512;
    size_t reps = 20000;
    size_t batch = 8;
    uint64_t seed = 1234;

    static struct option long_opts[] = {
        {"dim", required_argument, nullptr, 'd'},
        {"bits", required_argument, nullptr, 'b'},
        {"reps", required_argument, nullptr, 'r'},
        {"batch", required_argument, nullptr, 'k'},
        {"seed", required_argument, nullptr, 's'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr,0,nullptr,0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:b:r:k:s:h", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'd': dim = static_cast<size_t>(std::stoul(optarg)); break;
            case 'b': bits = static_cast<size_t>(std::stoul(optarg)); break;
            case 'r': reps = static_cast<size_t>(std::stoul(optarg)); break;
            case 'k': batch = static_cast<size_t>(std::stoul(optarg)); break;
            case 's': seed = static_cast<uint64_t>(std::stoull(optarg)); break;
            case 'h':
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    std::cout << "SimHash micro-benchmark\n";
    std::cout << " dim=" << dim << " bits=" << bits << " reps=" << reps << " batch=" << batch << " seed=" << seed << "\n";

    // Prepare RNG and data
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    // Pre-generate N vectors (keep small set reused to avoid memory blowing)
    size_t pool = std::min<size_t>(1024, std::max<size_t>(256, batch * 16));
    std::vector<float> data(pool * dim);
    for (size_t i = 0; i < pool * dim; ++i) data[i] = ud(rng);

    // Create SimHash instance
    SimHash sh(dim, bits, 0xC0FFEE);
    size_t out_bytes = sh.bytes();
    std::vector<uint8_t> out(out_bytes * batch);

    // Warmup: run some computes to warm caches and JIT (if any)
    size_t warm = std::min<size_t>(reps / 10 + 100, 2000);
    for (size_t w = 0; w < warm; ++w) {
        const float* v = data.data() + ( (w % pool) * dim );
        sh.compute(v, out.data());
    }

    // Timed loop: measure compute() throughput on batches
    size_t total = 0;
    auto t0 = high_resolution_clock::now();
    while (total < reps) {
        size_t do_batch = std::min(batch, reps - total);
        for (size_t i = 0; i < do_batch; ++i) {
            const float* v = data.data() + ( ((total + i) % pool) * dim );
            sh.compute(v, out.data() + i * out_bytes);
        }
        total += do_batch;
    }
    auto t1 = high_resolution_clock::now();
    double sec = duration_cast<duration<double>>(t1 - t0).count();
    double ops = static_cast<double>(reps);
    double ops_per_sec = ops / sec;
    double ns_per_op = (sec * 1e9) / ops;

    // Estimate memory read: dim * 4 bytes per vector (reads for dot)
    double bytes_read_per_vec = static_cast<double>(dim) * sizeof(float);
    double gb_s = (ops_per_sec * bytes_read_per_vec) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "RESULT compute():\n";
    std::cout << "  total_vecs=" << reps << " time=" << sec << "s\n";
    std::cout << "  throughput=" << ops_per_sec << " vec/s\n";
    std::cout << "  ns/vec=" << ns_per_op << " ns\n";
    std::cout << "  approx mem read=" << gb_s << " GB/s\n";

    // Also test compute_words (64-bit word output) to see if behavior differs
    size_t words_needed = (bits + 63) / 64;
    std::vector<uint64_t> out_words(words_needed * batch);
    total = 0;
    t0 = high_resolution_clock::now();
    while (total < reps) {
        size_t do_batch = std::min(batch, reps - total);
        for (size_t i = 0; i < do_batch; ++i) {
            const float* v = data.data() + ( ((total + i) % pool) * dim );
            sh.compute_words(v, out_words.data() + i * words_needed, words_needed);
        }
        total += do_batch;
    }
    t1 = high_resolution_clock::now();
    sec = duration_cast<duration<double>>(t1 - t0).count();
    ops_per_sec = static_cast<double>(reps) / sec;
    ns_per_op = (sec * 1e9) / static_cast<double>(reps);
    gb_s = (ops_per_sec * bytes_read_per_vec) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "RESULT compute_words():\n";
    std::cout << "  total_vecs=" << reps << " time=" << sec << "s\n";
    std::cout << "  throughput=" << ops_per_sec << " vec/s\n";
    std::cout << "  ns/vec=" << ns_per_op << " ns\n";
    std::cout << "  approx mem read=" << gb_s << " GB/s\n";

    return 0;
}