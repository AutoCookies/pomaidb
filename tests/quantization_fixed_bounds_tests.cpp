#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <pomai/core/seed.h>

using namespace pomai;

static bool NearlyEqual(float a, float b, float eps = 1e-6f)
{
    return std::fabs(a - b) <= eps;
}

int main()
{
    Seed seed(2);
    seed.SetFixedBoundsAfterCount(3);

    std::vector<UpsertRequest> warm;
    for (int i = 0; i < 3; ++i)
    {
        UpsertRequest r;
        r.id = static_cast<Id>(i + 1);
        r.vec.data = {static_cast<float>(i), static_cast<float>(i)};
        warm.push_back(std::move(r));
    }
    seed.ApplyUpserts(warm);

    auto snap_before = seed.MakeSnapshot();
    std::vector<float> row_before(2);
    Seed::DequantizeRow(snap_before, 0, row_before.data());

    std::vector<UpsertRequest> drift;
    UpsertRequest r;
    r.id = 999;
    r.vec.data = {1000.0f, -1000.0f};
    drift.push_back(std::move(r));
    seed.ApplyUpserts(drift);

    auto snap_after = seed.MakeSnapshot();
    std::vector<float> row_after(2);
    Seed::DequantizeRow(snap_after, 0, row_after.data());

    if (!NearlyEqual(row_before[0], row_after[0]) || !NearlyEqual(row_before[1], row_after[1]))
    {
        std::cerr << "Quantization drift detected after fixed bounds.\n";
        return 1;
    }

    std::cout << "Fixed bounds quantization test PASS\n";
    return 0;
}
