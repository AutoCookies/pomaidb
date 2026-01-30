#include <catch2/catch.hpp>

#include <pomai/core/spatial_router.h>

#include <numeric>

namespace pomai
{
    struct SpatialRouterTestAccess
    {
        static std::size_t SelectBestIndex(const Vector *centroids,
                                           const std::size_t *indices,
                                           std::size_t count,
                                           const float *query,
                                           std::size_t dim)
        {
            return SpatialRouter::SelectBestIndex(centroids, indices, count, query, dim);
        }
    };
}

TEST_CASE("SpatialRouter SelectBestIndex handles counts above 256", "[core][spatial_router]")
{
    using pomai::SpatialRouterTestAccess;
    using pomai::Vector;

    const std::size_t dim = 1;
    const std::size_t counts[] = {257, 1024, 4096};

    for (std::size_t count : counts)
    {
        std::vector<Vector> centroids;
        centroids.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            Vector v;
            v.data.push_back(static_cast<float>(i));
            centroids.push_back(std::move(v));
        }

        std::vector<std::size_t> indices(count);
        std::iota(indices.begin(), indices.end(), 0);

        const float query_val = static_cast<float>(count - 1);
        std::size_t best = SpatialRouterTestAccess::SelectBestIndex(centroids.data(),
                                                                    indices.data(),
                                                                    count,
                                                                    &query_val,
                                                                    dim);

        REQUIRE(best == count - 1);
    }
}
