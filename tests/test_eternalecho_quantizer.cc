#include "src/ai/eternalecho_quantizer.h"
#include "src/core/config.h"
#include "src/core/cpu_kernels.h"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <exception>
#include <iomanip>
#include <limits>
#include <functional>

using namespace pomai::ai;
using pomai::config::EternalEchoConfig;

static const char *GREEN = "\033[32m";
static const char *RED = "\033[31m";
static const char *RESET = "\033[0m";

struct TestRunner
{
    int passed = 0;
    int failed = 0;

    void report(const char *name, bool ok, const std::string &msg = {})
    {
        if (ok)
        {
            ++passed;
            std::cout << GREEN << "PASS" << RESET << " - " << name << "\n";
        }
        else
        {
            ++failed;
            std::cout << RED << "FAIL" << RESET << " - " << name << "\n";
            if (!msg.empty())
                std::cout << "       " << msg << "\n";
        }
    }

    int summary_and_exit()
    {
        std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";
        return (failed == 0) ? 0 : 1;
    }
};

static inline bool is_finite_vec(const std::vector<float> &v)
{
    for (float x : v)
        if (!std::isfinite(x))
            return false;
    return true;
}

static inline bool has_nan_or_inf(const float *v, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(v[i]))
            return true;
    return false;
}

static std::vector<uint8_t> serialize_code_bytes(const EchoCode &code, const EternalEchoConfig &cfg)
{
    std::vector<uint8_t> out;
    uint8_t depth = code.depth;
    out.push_back(depth);
    for (size_t k = 0; k < depth; ++k)
    {
        uint8_t q = 0;
        if (k < code.scales_q.size())
            q = code.scales_q[k];
        out.push_back(q);
    }
    for (size_t k = 0; k < depth; ++k)
    {
        if (k < code.sign_bytes.size())
        {
            const std::vector<uint8_t> &sb = code.sign_bytes[k];
            out.insert(out.end(), sb.begin(), sb.end());
        }
        else
        {
            // layer had no bytes -> nothing
        }
    }
    return out;
}

int main()
{
    TestRunner runner;

    pomai_init_cpu_kernels();

    // Deterministic RNG for tests
    std::mt19937_64 rng(123456789ULL);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(-1e3f, 1e3f);

    auto make_rand_vec = [&](size_t dim, bool gaussian = true, float scale = 1.0f)
    {
        std::vector<float> v(dim);
        if (gaussian)
        {
            for (size_t i = 0; i < dim; ++i)
                v[i] = nd(rng) * scale;
        }
        else
        {
            for (size_t i = 0; i < dim; ++i)
                v[i] = ud(rng) * scale;
        }
        return v;
    };

    // Test parameters
    const size_t dim = 64; // moderate dimension for speed
    EternalEchoConfig cfg; // defaults from config.h (quantize_scales=true, bits_per_layer set)

    try
    {
        // Test 1: Basic encode/decode produces finite outputs (random vectors)
        {
            EternalEchoQuantizer q(dim, cfg, 42ULL);
            bool ok = true;
            for (int t = 0; t < 32; ++t)
            {
                auto v = make_rand_vec(dim);
                if (has_nan_or_inf(v.data(), dim))
                {
                    ok = false;
                    break;
                }
                EchoCode code = q.encode(v.data());
                std::vector<float> out(dim);
                q.decode(code, out.data());
                if (!is_finite_vec(out))
                {
                    ok = false;
                    break;
                }
                // decoded vector should not contain NaN and distance should be finite and non-negative
                float d = ::pomai::core::kernels_internal::impl_l2sq(v.data(), out.data(), dim);
                if (!std::isfinite(d) || d < 0.0f)
                {
                    ok = false;
                    break;
                }
            }
            runner.report("Basic encode/decode finite (random vectors)", ok);
        }

        // Test 2: Zero vector -> encode/decode yields small residual (or exactly zero)
        {
            EternalEchoQuantizer q(dim, cfg, 123ULL);
            std::vector<float> v(dim, 0.0f);
            EchoCode code = q.encode(v.data());
            std::vector<float> out(dim, 0.0f);
            q.decode(code, out.data());
            bool ok = is_finite_vec(out);
            // measure norm
            float norm = 0.0f;
            for (size_t i = 0; i < dim; ++i)
                norm += out[i] * out[i];
            // Expect decode to reconstruct approx zero (norm small)
            ok = ok && (norm < 1e-6f);
            runner.report("Zero-vector encode/decode", ok, ok ? "" : "decoded not near-zero or non-finite");
        }

        // Test 3: Large magnitude vector should not produce NaNs
        {
            EternalEchoQuantizer q(dim, cfg, 999ULL);
            auto v = make_rand_vec(dim, true, 1e6f);
            bool ok = true;
            // ensure input not Inf/NaN
            if (has_nan_or_inf(v.data(), dim))
                ok = false;
            try
            {
                EchoCode code = q.encode(v.data());
                std::vector<float> out(dim);
                q.decode(code, out.data());
                ok = ok && is_finite_vec(out);
            }
            catch (const std::exception &e)
            {
                // encode may throw for pathological inputs; treat as failure for large values
                ok = false;
            }
            runner.report("Large-magnitude vector stability", ok);
        }

        // Test 4: NaN/Inf handling - should either throw or produce no NaNs downstream
        {
            EternalEchoQuantizer q(dim, cfg, 555ULL);
            std::vector<float> v = make_rand_vec(dim);
            v[dim / 3] = std::numeric_limits<float>::quiet_NaN();
            bool ok = false;
            try
            {
                EchoCode code = q.encode(v.data());
                // If encode returned, ensure decode produces finite values (preferred)
                std::vector<float> out(dim);
                q.decode(code, out.data());
                ok = is_finite_vec(out);
            }
            catch (const std::invalid_argument &)
            {
                ok = true; // acceptable
            }
            catch (const std::runtime_error &)
            {
                ok = true; // acceptable
            }
            catch (...)
            {
                ok = false;
            }
            runner.report("NaN input handling (throw or safe)", ok, ok ? "" : "encode produced invalid output or crashed");
        }

        // Test 5: approx_dist_code_bytes consistency vs decode-based approx_dist
        {
            EternalEchoQuantizer q(dim, cfg, 2021ULL);
            bool ok = true;
            for (int t = 0; t < 50; ++t)
            {
                auto base = make_rand_vec(dim);
                auto query = make_rand_vec(dim);
                EchoCode code = q.encode(base.data());
                std::vector<float> recon(dim);
                q.decode(code, recon.data());
                // exact distance
                float exact = ::l2sq(query.data(), recon.data(), dim);
                // produce qproj
                std::vector<std::vector<float>> qproj;
                q.project_query(query.data(), qproj);
                float qnorm2 = ::pomai_dot(query.data(), query.data(), dim);
                // serialize code to bytes
                std::vector<uint8_t> bytes = serialize_code_bytes(code, cfg);
                float approx = q.approx_dist_code_bytes(qproj, qnorm2, bytes.data(), bytes.size());
                if (!std::isfinite(approx) || approx < 0.0f)
                {
                    ok = false;
                    break;
                }
                // approx should be reasonably close to exact (within relative tolerance)
                float rel = (exact > 1e-6f) ? std::fabs(approx - exact) / (exact) : std::fabs(approx - exact);
                if (rel > 1e-1f)
                { // allow 10% relative error for compressed approximation
                    ok = false;
                    break;
                }
            }
            runner.report("approx_dist_code_bytes vs decode-based distance", ok, ok ? "" : "approx differs too much or non-finite");
        }

        // Test 6: Stress test many random vectors -> ensure no NaNs produced in encoding loop
        {
            EternalEchoQuantizer q(dim, cfg, 314159ULL);
            bool ok = true;
            for (int t = 0; t < 200; ++t)
            {
                auto v = make_rand_vec(dim);
                EchoCode code;
                try
                {
                    code = q.encode(v.data());
                }
                catch (...)
                {
                    ok = false;
                    break;
                }
                // verify sign_bytes sizes correspond to bits per layer
                uint8_t depth = code.depth;
                for (size_t k = 0; k < depth; ++k)
                {
                    uint32_t expect_b = cfg.bits_per_layer[k];
                    size_t expect_bytes = (expect_b + 7) / 8;
                    if (k >= code.sign_bytes.size())
                    {
                        ok = false;
                        break;
                    }
                    if (code.sign_bytes[k].size() != expect_bytes)
                    {
                        ok = false;
                        break;
                    }
                }
                if (!ok)
                    break;
            }
            runner.report("Stress encode loop (no crash / valid sign bytes)", ok);
        }

        // Test 7: layer_col_energy_ sanity (>0)
        {
            EternalEchoQuantizer q(dim, cfg, 777ULL);
            bool ok = true;
            for (float e : q.layer_col_energy())
            {
                if (!std::isfinite(e) || e <= 0.0f)
                {
                    ok = false;
                    break;
                }
            }
            runner.report("layer_col_energy sanity (>0)", ok);
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << RED << "FATAL ERROR" << RESET << " - Exception during tests: " << ex.what() << "\n";
        return 2;
    }
    catch (...)
    {
        std::cerr << RED << "FATAL ERROR" << RESET << " - Unknown exception during tests\n";
        return 2;
    }

    return runner.summary_and_exit();
}