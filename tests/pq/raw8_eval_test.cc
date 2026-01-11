// tests/pq/raw8_eval_test.cc
//
// Simple unit test for pq_approx_dist_batch_raw8 in src/ai/pq_eval.{h,cc}

#include "src/ai/pq_eval.h"
#include <vector>
#include <cassert>
#include <iostream>

int main()
{
    using namespace pomai::ai;

    // small synthetic example: m=4 subquantizers, k=256, nvecs=3
    size_t m = 4;
    size_t k = 256;
    size_t nvecs = 3;

    // tables: m*k floats: set tables[sub*k + c] = sub*1000 + c (distinct)
    std::vector<float> tables(m * k);
    for (size_t sub = 0; sub < m; ++sub)
    {
        for (size_t c = 0; c < k; ++c)
        {
            tables[sub * k + c] = static_cast<float>(sub * 1000 + c);
        }
    }

    // raw8 codes: for each vector, choose codes [0,10,20,30], [1,11,21,31], [2,12,22,32]
    std::vector<uint8_t> raw8(nvecs * m);
    for (size_t vi = 0; vi < nvecs; ++vi)
    {
        raw8[vi * m + 0] = static_cast<uint8_t>(vi + 0);
        raw8[vi * m + 1] = static_cast<uint8_t>(vi + 10);
        raw8[vi * m + 2] = static_cast<uint8_t>(vi + 20);
        raw8[vi * m + 3] = static_cast<uint8_t>(vi + 30);
    }

    std::vector<float> out(nvecs, 0.0f);
    pq_approx_dist_batch_raw8(tables.data(), m, k, raw8.data(), nvecs, out.data());

    // validate results: expected sum of the chosen table entries
    for (size_t vi = 0; vi < nvecs; ++vi)
    {
        float expect = 0.0f;
        expect += tables[0 * k + raw8[vi * m + 0]];
        expect += tables[1 * k + raw8[vi * m + 1]];
        expect += tables[2 * k + raw8[vi * m + 2]];
        expect += tables[3 * k + raw8[vi * m + 3]];
        if (out[vi] != expect)
        {
            std::cerr << "mismatch vi=" << vi << " got=" << out[vi] << " expected=" << expect << "\n";
            return 2;
        }
    }

    std::cout << "pq raw8 eval test OK\n";
    return 0;
}