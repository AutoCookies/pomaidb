#include "src/ai/zeroharmony_pack.h"
#include "src/core/cpu_kernels.h"
#include "src/core/config.h"

#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

using namespace pomai::ai;
using namespace pomai::config;

static void expect(bool cond, const char *msg)
{
    if (!cond)
    {
        std::cerr << "[FAIL] " << msg << "\n";
        std::abort();
    }
    else
    {
        std::cout << "[PASS] " << msg << "\n";
    }
}

static std::vector<float> rand_vec(size_t dim, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v(dim);
    for (size_t i = 0; i < dim; ++i) v[i] = nd(rng);
    return v;
}

int main()
{
    // Basic configuration
    ZeroHarmonyConfig cfg_default;
    cfg_default.use_half_nonzero = false;
    cfg_default.half_max_exact_abs = 1.0f; // default

    // 1) Zero-run behaviour: when vec == mean all deltas are zero and pack should be minimal.
    {
        size_t dim = 100;
        ZeroHarmonyPacker packer(cfg_default, dim);

        std::vector<float> mean(dim, 0.0f);
        std::vector<float> vec = mean; // identical

        auto packed = packer.pack_with_mean(vec.data(), mean);
        // Since entire vector is zero-deltas and dim < 255, expect a single zero-run pair: [0, dim]
        expect(packed.size() == 2, "Zero-run: packed size == 2 for full-zero delta vector");
        expect(packed[0] == 0 && packed[1] == static_cast<uint8_t>(dim), "Zero-run: encoded run tag+len correct");

        // Unpack and verify exact equality
        std::vector<float> out(dim, 0.0f);
        bool ok = packer.unpack_to(packed.data(), packed.size(), mean, out.data());
        expect(ok, "Zero-run: unpack succeeded");
        for (size_t i = 0; i < dim; ++i) expect(out[i] == mean[i], "Zero-run: unpacked value equals mean");
    }

    // 2) Round-trip random vectors (float32 path)
    {
        size_t dim = 64;
        ZeroHarmonyPacker packer(cfg_default, dim);

        auto mean = rand_vec(dim, 42);
        auto vec = rand_vec(dim, 99);

        auto packed = packer.pack_with_mean(vec.data(), mean);
        expect(!packed.empty(), "Roundtrip: packed non-empty");

        std::vector<float> out(dim, 0.0f);
        bool ok = packer.unpack_to(packed.data(), packed.size(), mean, out.data());
        expect(ok, "Roundtrip: unpack succeeded");

        // compute l2sq between original vec and unpacked (mean + deltas)
        float err = ::l2sq(vec.data(), out.data(), dim);
        expect(err < 1e-6f, "Roundtrip: reconstructed equals original (tiny L2 error)");
    }

    // 3) Half-precision path: ensure fp16-branch is exercised when enabled and exact
    {
        ZeroHarmonyConfig cfg = cfg_default;
        cfg.use_half_nonzero = true;
        cfg.half_max_exact_abs = 65504.0f; // large so typical values qualify

        size_t dim = 4;
        ZeroHarmonyPacker packer(cfg, dim);

        std::vector<float> mean(dim, 0.0f);
        std::vector<float> vec(dim, 0.0f);
        vec[0] = 1.0f;   // exactly representable in fp16
        vec[1] = 0.0f;
        vec[2] = 0.0f;
        vec[3] = 0.0f;

        // pack
        auto packed = packer.pack_with_mean(vec.data(), mean);
        expect(!packed.empty(), "Half-path: packed non-empty");

        // decode first few bytes to assert fp16 encoding used for first non-zero entry
        // The format for first element (non-zero) should be: [1, lowByte(h), highByte(h), ...]
        // There may be trailing zero-run pairs after.
        expect(packed[0] == 1u, "Half-path: first token is non-zero tag (1)");

        // compute expected fp16 encoding
        uint16_t expected_h = fp32_to_fp16(vec[0] - mean[0]);
        uint8_t lo = static_cast<uint8_t>(expected_h & 0xFF);
        uint8_t hi = static_cast<uint8_t>((expected_h >> 8) & 0xFF);

        expect(packed.size() >= 3, "Half-path: packed has enough bytes for fp16 value");
        expect(packed[1] == lo && packed[2] == hi, "Half-path: fp16 bytes match expected encoding");

        // Unpack and verify values
        std::vector<float> out(dim, 0.0f);
        bool ok = packer.unpack_to(packed.data(), packed.size(), mean, out.data());
        expect(ok, "Half-path: unpack succeeded");
        // delta was exactly representable -> exact equality
        expect(out[0] == vec[0], "Half-path: reconstructed first element equals original");
        for (size_t i = 1; i < dim; ++i) expect(out[i] == mean[i], "Half-path: other elements equal mean");
    }

    // 4) approx_dist correctness: distance between query and original should match unpack+distance
    {
        size_t dim = 32;
        ZeroHarmonyPacker packer(cfg_default, dim);
        auto mean = rand_vec(dim, 7);
        auto vec = rand_vec(dim, 123);
        auto query = rand_vec(dim, 321);

        auto packed = packer.pack_with_mean(vec.data(), mean);
        std::vector<float> out(dim);
        bool ok = packer.unpack_to(packed.data(), packed.size(), mean, out.data());
        expect(ok, "approx_dist: unpack succeeded for distance check");

        float d1 = ::l2sq(query.data(), out.data(), dim);
        float d2 = packer.approx_dist(query.data(), packed.data(), packed.size(), mean);
        expect(std::fabs(d1 - d2) < 1e-6f, "approx_dist: distance from packed equals distance to unpacked reconstruction");
    }

    std::cout << "All ZeroHarmonyPacker tests passed.\n";
    return 0;
}