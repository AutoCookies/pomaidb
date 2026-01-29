#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <new>

#include "search_utils.h"
#include "seed.h"

namespace
{
    std::atomic<std::size_t> g_allocs{0};

    void *CountedAlloc(std::size_t sz)
    {
        g_allocs.fetch_add(1, std::memory_order_relaxed);
        if (void *p = std::malloc(sz))
            return p;
        throw std::bad_alloc();
    }
} // namespace

void *operator new(std::size_t sz)
{
    return CountedAlloc(sz);
}

void *operator new[](std::size_t sz)
{
    return CountedAlloc(sz);
}

void operator delete(void *p) noexcept
{
    std::free(p);
}

void operator delete[](void *p) noexcept
{
    std::free(p);
}

void operator delete(void *p, std::size_t) noexcept
{
    std::free(p);
}

void operator delete[](void *p, std::size_t) noexcept
{
    std::free(p);
}

static float L2Sqr(const float *a, const float *b, std::size_t dim)
{
    float d = 0.0f;
    for (std::size_t i = 0; i < dim; ++i)
    {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
}

int main()
{
    std::cout << "Rerank exact tests starting...\n";
    int failures = 0;

    const std::size_t dim = 8;
    const std::size_t n = 64;
    const std::size_t topk = 5;

    try
    {
        pomai::Seed seed(dim);
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<std::vector<float>> original(n, std::vector<float>(dim));
        for (std::size_t i = 0; i < n; ++i)
        {
            pomai::UpsertRequest req;
            req.id = static_cast<pomai::Id>(i);
            req.vec.data.resize(dim);
            for (std::size_t d = 0; d < dim; ++d)
            {
                float v = dist(rng);
                req.vec.data[d] = v;
                original[i][d] = v;
            }
            seed.ApplyUpserts({req});
        }

        auto snap = seed.MakeSnapshot();
        pomai::SearchRequest req;
        req.topk = topk;
        req.candidate_k = n;
        req.max_rerank_k = n;
        req.query.data = original[7];

        auto resp = pomai::Seed::SearchSnapshot(snap, req);

        std::vector<float> buf(dim);
        std::vector<std::pair<float, pomai::Id>> dists;
        dists.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            pomai::Seed::DequantizeRow(snap, i, buf.data());
            dists.push_back({L2Sqr(buf.data(), req.query.data.data(), dim), static_cast<pomai::Id>(i)});
        }
        std::sort(dists.begin(), dists.end(), [](const auto &a, const auto &b)
                  {
                      if (a.first == b.first)
                          return a.second < b.second;
                      return a.first < b.first;
                  });

        if (resp.items.size() != topk)
            throw std::runtime_error("unexpected topk size");
        for (std::size_t i = 0; i < topk; ++i)
        {
            if (resp.items[i].id != dists[i].second)
                throw std::runtime_error("rerank mismatch vs brute force");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exact rerank test failed: " << e.what() << "\n";
        ++failures;
    }

    try
    {
        std::vector<pomai::SearchResultItem> items = {
            {42, 1.0f},
            {7, 2.0f},
            {42, 1.5f},
            {9, 2.0f},
        };
        pomai::SortAndDedupeResults(items, 3);
        if (items.size() != 3)
            throw std::runtime_error("dedupe size mismatch");
        if (items[0].id != 7 || items[1].id != 9 || items[2].id != 42)
            throw std::runtime_error("dedupe ordering mismatch");
    }
    catch (const std::exception &e)
    {
        std::cerr << "Dedup test failed: " << e.what() << "\n";
        ++failures;
    }

    try
    {
        pomai::Seed seed(dim);
        std::mt19937 rng(456);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (std::size_t i = 0; i < n; ++i)
        {
            pomai::UpsertRequest req;
            req.id = static_cast<pomai::Id>(i);
            req.vec.data.resize(dim);
            for (std::size_t d = 0; d < dim; ++d)
                req.vec.data[d] = dist(rng);
            seed.ApplyUpserts({req});
        }
        auto snap = seed.MakeSnapshot();
        pomai::SearchRequest req;
        req.topk = topk;
        req.candidate_k = 32;
        req.max_rerank_k = 64;
        req.query.data.resize(dim, 0.5f);

        pomai::Seed::SearchSnapshot(snap, req);
        g_allocs.store(0, std::memory_order_relaxed);
        pomai::Seed::SearchSnapshot(snap, req);
        std::size_t allocs = g_allocs.load(std::memory_order_relaxed);
        if (allocs > 8)
            throw std::runtime_error("excess allocations during rerank");
    }
    catch (const std::exception &e)
    {
        std::cerr << "Allocation guard test failed: " << e.what() << "\n";
        ++failures;
    }

    if (failures == 0)
    {
        std::cout << "All rerank exact tests PASS\n";
        return 0;
    }

    std::cerr << failures << " rerank exact tests FAILED\n";
    return 1;
}
