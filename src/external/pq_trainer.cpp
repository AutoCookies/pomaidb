// src/external/pq_trainer.cpp
//
// Simple PQ trainer CLI (offline utility).
//
// Usage examples:
//   ./pq_trainer --in samples.bin --n 20000 --dim 768 --m 48 --k 256 --iters 20 --out pq_codebooks.bin
//   ./pq_trainer --random --n 20000 --dim 768 --m 48 --k 256 --iters 20 --out pq_codebooks.bin
//
// samples.bin is raw float32: N * dim floats (little-endian).
// The produced codebooks file is compatible with ProductQuantizer::load_codebooks()
//
// Notes:
//  - This is intentionally minimal and robust: it reads raw floats or generates
//    random samples, trains per-subquantizer k-means (ProductQuantizer::train)
//    and writes codebooks via ProductQuantizer::save_codebooks.
//  - For production you may extend this to accept .npy, CSV, or other formats.

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <cstring>
#include <cstdint>
#include <stdexcept>

#include "src/ai/pq.h"

static void print_usage(const char *prog)
{
    std::cerr << "PQ trainer\n";
    std::cerr << "Usage:\n";
    std::cerr << "  " << prog << " [--in samples.bin | --random] --n N --dim D --m M --k K --iters I --out codebooks.bin\n";
    std::cerr << "\nOptions:\n";
    std::cerr << "  --in FILE        raw float32 samples file (N * D floats)\n";
    std::cerr << "  --random         generate N random samples in [0,1]\n";
    std::cerr << "  --n N            number of training samples\n";
    std::cerr << "  --dim D          vector dimensionality\n";
    std::cerr << "  --m M            number of subquantizers\n";
    std::cerr << "  --k K            codebook size per sub (e.g. 256)\n";
    std::cerr << "  --iters I        max kmeans iterations per sub (default 20)\n";
    std::cerr << "  --out FILE       output codebooks file\n";
    std::cerr << "  --seed S         rng seed (optional)\n";
}

int main(int argc, char **argv)
{
    std::string in_file;
    bool use_random = false;
    size_t N = 0;
    size_t dim = 0;
    size_t m = 0;
    size_t k = 0;
    size_t iters = 20;
    std::string out_file;
    uint64_t seed = 123456789ULL;

    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--in" && i + 1 < argc)
        {
            in_file = argv[++i];
        }
        else if (a == "--random")
        {
            use_random = true;
        }
        else if (a == "--n" && i + 1 < argc)
        {
            N = static_cast<size_t>(std::stoull(argv[++i]));
        }
        else if (a == "--dim" && i + 1 < argc)
        {
            dim = static_cast<size_t>(std::stoull(argv[++i]));
        }
        else if (a == "--m" && i + 1 < argc)
        {
            m = static_cast<size_t>(std::stoull(argv[++i]));
        }
        else if (a == "--k" && i + 1 < argc)
        {
            k = static_cast<size_t>(std::stoull(argv[++i]));
        }
        else if (a == "--iters" && i + 1 < argc)
        {
            iters = static_cast<size_t>(std::stoull(argv[++i]));
        }
        else if (a == "--out" && i + 1 < argc)
        {
            out_file = argv[++i];
        }
        else if (a == "--seed" && i + 1 < argc)
        {
            seed = static_cast<uint64_t>(std::stoull(argv[++i]));
        }
        else
        {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (N == 0 || dim == 0 || m == 0 || k == 0 || out_file.empty() || (!use_random && in_file.empty()))
    {
        print_usage(argv[0]);
        return 1;
    }

    // allocate sample buffer
    std::vector<float> samples;
    samples.resize(static_cast<size_t>(N) * static_cast<size_t>(dim));

    if (use_random)
    {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> ud(0.0f, 1.0f);
        for (size_t i = 0; i < N * dim; ++i)
            samples[i] = ud(rng);
        std::cerr << "[pq_trainer] generated " << N << " random samples (dim=" << dim << ")\n";
    }
    else
    {
        // read raw file
        std::ifstream f(in_file, std::ios::binary);
        if (!f)
        {
            std::cerr << "Failed to open input file: " << in_file << "\n";
            return 1;
        }
        // determine file size
        f.seekg(0, std::ios::end);
        std::streamoff sz = f.tellg();
        f.seekg(0, std::ios::beg);
        const std::streamoff expected = static_cast<std::streamoff>(N) * static_cast<std::streamoff>(dim) * static_cast<std::streamoff>(sizeof(float));
        if (sz < expected)
        {
            std::cerr << "Input file too small: expected at least " << expected << " bytes but file is " << sz << "\n";
            return 1;
        }
        f.read(reinterpret_cast<char *>(samples.data()), expected);
        if (!f)
        {
            std::cerr << "Failed to read samples from " << in_file << "\n";
            return 1;
        }
        std::cerr << "[pq_trainer] loaded " << N << " samples from " << in_file << "\n";
    }

    try
    {
        pomai::ai::ProductQuantizer pq(dim, m, k);
        std::cerr << "[pq_trainer] training PQ: dim=" << dim << " m=" << m << " k=" << k << " n_samples=" << N << " iters=" << iters << "\n";
        pq.train(samples.data(), N, iters);
        std::cerr << "[pq_trainer] training complete; saving codebooks to " << out_file << "\n";
        if (!pq.save_codebooks(out_file))
        {
            std::cerr << "Failed to save codebooks to " << out_file << "\n";
            return 1;
        }
        std::cerr << "[pq_trainer] saved codebooks successfully\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception during PQ training: " << e.what() << "\n";
        return 1;
    }
    return 0;
}