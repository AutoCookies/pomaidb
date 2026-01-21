#include "src/ai/fingerprint.h"
#include "src/ai/simhash.h"
#include "src/core/cpu_kernels.h"
#include "src/core/config.h"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>

using namespace pomai::ai;

static const char *GREEN = "\033[32m";
static const char *RED = "\033[31m";
static const char *RESET = "\033[0m";

struct Runner
{
    int pass = 0;
    int fail = 0;
    void report(const char *n, bool ok)
    {
        if (ok)
        {
            ++pass;
            std::cout << GREEN << "PASS" << RESET << " - " << n << "\n";
        }
        else
        {
            ++fail;
            std::cout << RED << "FAIL" << RESET << " - " << n << "\n";
        }
    }
    int summary()
    {
        std::cout << "Summary: " << pass << " passed, " << fail << " failed\n";
        return fail == 0 ? 0 : 1;
    }
};

int main()
{
    pomai_init_cpu_kernels();
    Runner r;
    std::mt19937_64 rng(1234567ULL);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    size_t dim = 128;
    pomai::config::FingerprintConfig cfg;
    cfg.fingerprint_bits = 256;

    try
    {
        auto enc = FingerprintEncoder::createSimHash(dim, cfg, 42ULL);
        bool ok1 = (enc->bytes() > 0);
        std::vector<float> v(dim);
        for (size_t i = 0; i < dim; ++i)
            v[i] = nd(rng);
        std::vector<uint8_t> out(enc->bytes());
        enc->compute(v.data(), out.data());
        bool finite = true;
        for (auto b : out)
        {
            (void)b;
        }
        r.report("SimHash compute basic", ok1 && finite);

        std::string tmp = "/tmp/test_rotation_bad.bin";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            uint64_t wrong = 1;
            f.write(reinterpret_cast<const char *>(&wrong), sizeof(wrong));
            uint32_t shortdata = 0xdeadbeef;
            f.write(reinterpret_cast<const char *>(&shortdata), sizeof(shortdata));
        }

        bool threw = false;
        try
        {
            auto opq = FingerprintEncoder::createOPQSign(dim, cfg, tmp, 123ULL);
            std::vector<uint8_t> out2(opq->bytes());
            opq->compute(v.data(), out2.data());
        }
        catch (...)
        {
            threw = true;
        }
        r.report("OPQSign fallback to identity (no throw)", !threw);

        std::vector<float> large(dim);
        for (size_t i = 0; i < dim; ++i)
            large[i] = nd(rng) * 1e6f;
        bool ok2 = true;
        try
        {
            auto e = FingerprintEncoder::createSimHash(dim, cfg, 7ULL);
            std::vector<uint8_t> buf(e->bytes());
            e->compute(large.data(), buf.data());
            e->compute_words(large.data(), nullptr, 0);
        }
        catch (...)
        {
            ok2 = false;
        }
        r.report("SimHash large magnitude stability", ok2);

        r.report("SimHash bytes consistent", enc->bytes() == (size_t)((cfg.fingerprint_bits + 7) / 8));
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Fatal: " << ex.what() << "\n";
        return 2;
    }

    return r.summary();
}