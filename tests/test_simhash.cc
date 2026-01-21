#include "src/ai/simhash.h"
#include "src/core/cpu_kernels.h"

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>

using namespace pomai::ai;

static const char *ANSI_GREEN = "\033[32m";
static const char *ANSI_RED = "\033[31m";
static const char *ANSI_RESET = "\033[0m";

struct Runner
{
    void expect(bool cond, const char *name)
    {
        if (cond)
        {
            std::cout << ANSI_GREEN << "[PASS] " << ANSI_RESET << name << "\n";
            passed++;
        }
        else
        {
            std::cout << ANSI_RED << "[FAIL] " << ANSI_RESET << name << "\n";
            failed++;
        }
    }

    int summary()
    {
        std::cout << "\nResults: " << passed << " passed, " << failed << " failed.\n";
        return failed == 0 ? 0 : 1;
    }

    int passed = 0;
    int failed = 0;
};

static std::vector<uint64_t> bytes_to_words(const uint8_t *bytes, size_t bytes_len)
{
    size_t words = (bytes_len + 7) / 8;
    std::vector<uint64_t> out(words, 0);
    for (size_t i = 0; i < bytes_len; ++i)
    {
        size_t w = i / 8;
        size_t bit = (i % 8);
        if ((bytes[i] & 1) != 0) // but simhash sets whole bit (1<<bit), so test whole byte
            ;
        // We need to set bit i if bytes[i] has bit (i%8). To reconstruct exact words:
        if (bytes[i] != 0)
            out[w] |= (1ULL << (i % 64));
    }
    return out;
}

// Helper: convert simhash byte-array to words (matching compute_words layout)
static std::vector<uint64_t> bytes_to_words_exact(const uint8_t *bytes, size_t bytes_len)
{
    size_t word_count = (bytes_len + 7) / 8;
    std::vector<uint64_t> words(word_count, 0);
    for (size_t i = 0; i < bytes_len; ++i)
    {
        if ((bytes[i] & (1u << (i % 8))) != 0) // this is wrong; bit positions in SimHash: bit i placed at byte[i/8] bit (i%8)
        {
            // dead branch -- keep for clarity; we'll instead reconstruct by reading bits directly
        }
    }
    // Simpler: iterate bits and set if corresponding bit set in bytes
    std::fill(words.begin(), words.end(), 0ULL);
    for (size_t bit = 0; bit < bytes_len * 8; ++bit)
    {
        size_t bidx = bit / 8;
        if (bidx >= bytes_len)
            break;
        uint8_t bv = bytes[bidx];
        if (bv & (1u << (bit % 8)))
        {
            size_t w = bit / 64;
            size_t bw = bit % 64;
            if (w < words.size())
                words[w] |= (1ULL << bw);
        }
    }
    return words;
}

int main()
{
    pomai_init_cpu_kernels();
    Runner r;

    // 1) ctor validation (dim==0 and bits==0)
    try
    {
        pomai::config::FingerprintConfig cfg;
        cfg.fingerprint_bits = 128;
        SimHash sh(0, cfg, 12345);
        r.expect(false, "ctor rejects dim==0");
    }
    catch (...)
    {
        r.expect(true, "ctor rejects dim==0");
    }

    try
    {
        pomai::config::FingerprintConfig cfg;
        cfg.fingerprint_bits = 0;
        SimHash sh(16, cfg, 123);
        r.expect(false, "ctor rejects bits==0");
    }
    catch (...)
    {
        r.expect(true, "ctor rejects bits==0");
    }

    // 2) basic compute_vec / compute_words consistency & determinism
    {
        pomai::config::FingerprintConfig cfg;
        cfg.fingerprint_bits = 128;
        uint64_t seed = 42;
        SimHash sh(64, cfg, seed);

        std::vector<float> v(64, 0.0f);
        v[0] = 1.2345f;
        v[10] = -0.5f;
        v[63] = 0.25f;

        std::vector<uint8_t> bytes = sh.compute_vec(v.data());
        std::vector<uint64_t> words((sh.bytes() + 7) / 8);
        sh.compute_words(v.data(), words.data(), words.size());

        // Reconstruct words from bytes precisely and compare
        auto words_from_bytes = bytes_to_words_exact(bytes.data(), sh.bytes());
        bool same = (words.size() == words_from_bytes.size());
        if (same)
        {
            for (size_t i = 0; i < words.size(); ++i)
            {
                if (words[i] != words_from_bytes[i])
                {
                    same = false;
                    break;
                }
            }
        }
        r.expect(same, "compute_vec and compute_words produce consistent bit patterns");

        // Determinism: same seed and same input -> same result
        SimHash sh2(64, cfg, seed);
        auto b2 = sh2.compute_vec(v.data());
        bool det = (b2.size() == bytes.size() && std::memcmp(b2.data(), bytes.data(), bytes.size()) == 0);
        r.expect(det, "same seed and vector produce identical simhash");

        // Different seed likely produces different projection -> allow non-deterministic but check inequality usually
        SimHash sh3(64, cfg, seed + 1);
        auto b3 = sh3.compute_vec(v.data());
        bool different = !(b3.size() == bytes.size() && std::memcmp(b3.data(), bytes.data(), bytes.size()) == 0);
        r.expect(different, "different seed usually yields different simhash (best-effort)");
    }

    // 3) hamming_dist correctness on known patterns
    {
        // Build two byte arrays: all-zero and all-ones (0xFF)
        uint8_t a[16];
        uint8_t b[16];
        std::memset(a, 0, sizeof(a));
        std::memset(b, 0xFF, sizeof(b));
        uint32_t dist = SimHash::hamming_dist(a, b, sizeof(a));
        // each byte differs in 8 bits -> 16 * 8 = 128
        r.expect(dist == 16 * 8, "hamming_dist counts differing bits for full mismatch");

        // identical arrays -> 0
        uint32_t dist2 = SimHash::hamming_dist(a, a, sizeof(a));
        r.expect(dist2 == 0, "hamming_dist is zero for identical arrays");

        // partial difference
        uint8_t c[2] = {0x0F, 0xF0}; // first byte low 4 bits set, second byte high 4 bits set
        uint8_t d[2] = {0x00, 0x00};
        uint32_t d3 = SimHash::hamming_dist(c, d, 2);
        r.expect(d3 == 8, "hamming_dist partial difference (4+4) == 8");
    }

    // 4) bit-count and packing edge - ensure compute sets bits in expected range
    {
        pomai::config::FingerprintConfig cfg;
        cfg.fingerprint_bits = 100; // not multiple of 8
        SimHash sh(37, cfg, 999);

        std::vector<float> v(37, 0.1f);
        auto bytes = sh.compute_vec(v.data());
        // bytes length should equal ceil(bits/8)
        r.expect(bytes.size() == (sh.bits() + 7) / 8, "compute_vec returns expected byte length for non-byte-aligned bits");

        // hamming distance against itself should be 0
        uint32_t dd = SimHash::hamming_dist(bytes.data(), bytes.data(), bytes.size());
        r.expect(dd == 0, "self-hamming distance is zero for non-byte-aligned bits");
    }

    return r.summary();
}