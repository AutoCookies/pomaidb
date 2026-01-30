#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pomai/api/pomai_db.h>
#include <pomai/core/seed.h>

using namespace pomai;
namespace fs = std::filesystem;

static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_metadata_filter.XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char *res = mkdtemp(buf.data());
    if (!res)
        throw std::runtime_error("mkdtemp failed");
    return std::string(res);
}

static void RemoveDir(const std::string &d)
{
    std::error_code ec;
    fs::remove_all(d, ec);
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

static std::vector<Id> BruteForceFiltered(const std::vector<std::vector<float>> &vectors,
                                          const std::vector<Metadata> &meta,
                                          const Filter &filter,
                                          const std::vector<float> &query,
                                          std::size_t topk)
{
    struct Item
    {
        float d;
        Id id;
    };
    std::vector<Item> items;
    for (std::size_t i = 0; i < vectors.size(); ++i)
    {
        const auto &m = meta[i];
        if (filter.namespace_id && *filter.namespace_id != m.namespace_id)
            continue;
        bool match = true;
        if (!filter.exclude_tags.empty())
        {
            for (TagId t : m.tag_ids)
            {
                if (std::binary_search(filter.exclude_tags.begin(), filter.exclude_tags.end(), t))
                {
                    match = false;
                    break;
                }
            }
        }
        if (!match)
            continue;
        if (!filter.require_all_tags.empty())
        {
            for (TagId t : filter.require_all_tags)
            {
                if (!std::binary_search(m.tag_ids.begin(), m.tag_ids.end(), t))
                {
                    match = false;
                    break;
                }
            }
        }
        if (!match)
            continue;
        if (!filter.require_any_tags.empty())
        {
            bool any = false;
            for (TagId t : filter.require_any_tags)
            {
                if (std::binary_search(m.tag_ids.begin(), m.tag_ids.end(), t))
                {
                    any = true;
                    break;
                }
            }
            if (!any)
                continue;
        }
        items.push_back({L2Sqr(vectors[i].data(), query.data(), query.size()), static_cast<Id>(i)});
    }
    std::sort(items.begin(), items.end(), [](const auto &a, const auto &b)
              { return a.d < b.d; });
    std::vector<Id> out;
    for (std::size_t i = 0; i < std::min(topk, items.size()); ++i)
        out.push_back(items[i].id);
    return out;
}

int main()
{
    std::cout << "Metadata filter tests starting...\n";
    int failures = 0;

    try
    {
        const std::size_t dim = 4;
        DbOptions opt;
        opt.dim = dim;
        opt.shards = 1;
        opt.wal_dir = MakeTempDir();
        opt.centroids_load_mode = MembraneRouter::CentroidsLoadMode::None;

        PomaiDB db(opt);
        db.Start();

        std::vector<UpsertRequest> batch;
        batch.reserve(3);
        for (Id i = 0; i < 3; ++i)
        {
            UpsertRequest req;
            req.id = i;
            req.vec.data = {static_cast<float>(i), 0.0f, 0.0f, 0.0f};
            req.metadata.namespace_id = (i < 2) ? 1 : 2;
            if (i == 0)
                req.metadata.tag_ids = {1, 2};
            else if (i == 1)
                req.metadata.tag_ids = {3};
            else
                req.metadata.tag_ids = {1};
            std::sort(req.metadata.tag_ids.begin(), req.metadata.tag_ids.end());
            batch.push_back(std::move(req));
        }
        db.UpsertBatch(std::move(batch), true).get();

        SearchRequest req;
        req.query.data = {0.0f, 0.0f, 0.0f, 0.0f};
        req.topk = 3;
        req.metric = Metric::L2;
        req.candidate_k = 0;

        auto filter = std::make_shared<Filter>();
        filter->namespace_id = 1;
        req.filter = filter;
        auto resp = db.Search(req);
        for (const auto &item : resp.items)
        {
            if (item.id >= 2)
            {
                std::cerr << "Filter namespace failed\n";
                failures++;
                break;
            }
        }

        auto tag_filter = std::make_shared<Filter>();
        tag_filter->require_all_tags = {1, 2};
        std::sort(tag_filter->require_all_tags.begin(), tag_filter->require_all_tags.end());
        req.filter = tag_filter;
        auto resp2 = db.Search(req);
        if (resp2.items.empty() || resp2.items.front().id != 0)
        {
            std::cerr << "Filter require_all_tags failed\n";
            failures++;
        }

        auto any_filter = std::make_shared<Filter>();
        any_filter->require_any_tags = {1, 3};
        any_filter->exclude_tags = {2};
        std::sort(any_filter->require_any_tags.begin(), any_filter->require_any_tags.end());
        std::sort(any_filter->exclude_tags.begin(), any_filter->exclude_tags.end());
        req.filter = any_filter;
        auto resp3 = db.Search(req);
        for (const auto &item : resp3.items)
        {
            if (item.id == 0)
            {
                std::cerr << "Filter exclude_tags failed\n";
                failures++;
                break;
            }
        }

        db.Stop();
        RemoveDir(opt.wal_dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Filter correctness test error: " << e.what() << "\n";
        failures++;
    }

    try
    {
        const std::size_t dim = 8;
        const std::size_t n = 128;
        const std::size_t topk = 10;
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<std::vector<float>> vectors(n, std::vector<float>(dim));
        std::vector<Metadata> meta(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t d = 0; d < dim; ++d)
                vectors[i][d] = dist(rng);
            meta[i].namespace_id = static_cast<std::uint32_t>(i % 3);
            meta[i].tag_ids = {static_cast<TagId>(i % 7), static_cast<TagId>(i % 11)};
            std::sort(meta[i].tag_ids.begin(), meta[i].tag_ids.end());
        }

        Seed seed(dim);
        for (std::size_t i = 0; i < n; ++i)
        {
            UpsertRequest req;
            req.id = static_cast<Id>(i);
            req.vec.data = vectors[i];
            req.metadata = meta[i];
            seed.ApplyUpserts({req});
        }

        auto snap = seed.MakeSnapshot();
        SearchRequest req;
        req.topk = topk;
        req.candidate_k = n;
        req.max_rerank_k = n;
        req.metric = Metric::L2;
        req.query.data = vectors[7];

        auto filter = std::make_shared<Filter>();
        filter->namespace_id = 1;
        filter->require_any_tags = {3, 4};
        std::sort(filter->require_any_tags.begin(), filter->require_any_tags.end());
        req.filter = filter;
        req.filtered_candidate_k = n;

        auto resp = Seed::SearchSnapshot(snap, req);
        auto gt = BruteForceFiltered(vectors, meta, *filter, req.query.data, topk);
        if (resp.items.size() != gt.size())
        {
            std::cerr << "Filtered recall test size mismatch\n";
            failures++;
        }
        else
        {
            for (std::size_t i = 0; i < gt.size(); ++i)
            {
                if (resp.items[i].id != gt[i])
                {
                    std::cerr << "Filtered recall mismatch at " << i << "\n";
                    failures++;
                    break;
                }
            }
        }

        SearchRequest rerank_req = req;
        rerank_req.filter = nullptr;
        rerank_req.topk = n;
        rerank_req.candidate_k = n;
        rerank_req.max_rerank_k = n;
        auto rerank_resp = Seed::SearchSnapshot(snap, rerank_req);
        std::vector<Id> rerank_filtered;
        rerank_filtered.reserve(topk);
        for (const auto &item : rerank_resp.items)
        {
            const auto &m = meta.at(static_cast<std::size_t>(item.id));
            if (filter->namespace_id && m.namespace_id != *filter->namespace_id)
                continue;
            bool match = true;
            if (!filter->exclude_tags.empty())
            {
                for (TagId t : m.tag_ids)
                {
                    if (std::binary_search(filter->exclude_tags.begin(), filter->exclude_tags.end(), t))
                    {
                        match = false;
                        break;
                    }
                }
            }
            if (!match)
                continue;
            if (!filter->require_all_tags.empty())
            {
                for (TagId t : filter->require_all_tags)
                {
                    if (!std::binary_search(m.tag_ids.begin(), m.tag_ids.end(), t))
                    {
                        match = false;
                        break;
                    }
                }
            }
            if (!match)
                continue;
            if (!filter->require_any_tags.empty())
            {
                bool any = false;
                for (TagId t : filter->require_any_tags)
                {
                    if (std::binary_search(m.tag_ids.begin(), m.tag_ids.end(), t))
                    {
                        any = true;
                        break;
                    }
                }
                if (!any)
                    continue;
            }
            rerank_filtered.push_back(item.id);
            if (rerank_filtered.size() >= topk)
                break;
        }
        if (resp.items.size() != rerank_filtered.size())
        {
            std::cerr << "Filtered rerank recall test size mismatch\n";
            failures++;
        }
        else
        {
            for (std::size_t i = 0; i < rerank_filtered.size(); ++i)
            {
                if (resp.items[i].id != rerank_filtered[i])
                {
                    std::cerr << "Filtered rerank recall mismatch at " << i << "\n";
                    failures++;
                    break;
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Filtered recall test error: " << e.what() << "\n";
        failures++;
    }

    try
    {
        DbOptions opt;
        opt.dim = 16;
        opt.shards = 1;
        opt.wal_dir = MakeTempDir();
        opt.centroids_load_mode = MembraneRouter::CentroidsLoadMode::None;
        opt.search_timeout_ms = 1000;

        PomaiDB db(opt);
        db.Start();

        std::atomic<bool> stop{false};
        std::atomic<int> errors{0};

        std::thread ingest([&]()
                           {
                               std::mt19937_64 rng(42);
                               std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                               Id next_id = 0;
                               while (!stop.load())
                               {
                                   std::vector<UpsertRequest> batch;
                                   batch.reserve(32);
                                   for (int i = 0; i < 32; ++i)
                                   {
                                       UpsertRequest req;
                                       req.id = next_id++;
                                       req.vec.data.resize(opt.dim);
                                       for (std::size_t d = 0; d < opt.dim; ++d)
                                           req.vec.data[d] = dist(rng);
                                       req.metadata.namespace_id = (req.id % 2 == 0) ? 1 : 2;
                                       req.metadata.tag_ids = {static_cast<TagId>(req.id % 7)};
                                       std::sort(req.metadata.tag_ids.begin(), req.metadata.tag_ids.end());
                                       batch.push_back(std::move(req));
                                   }
                                   try
                                   {
                                       db.UpsertBatch(std::move(batch), false).get();
                                   }
                                   catch (...)
                                   {
                                       errors.fetch_add(1, std::memory_order_relaxed);
                                   }
                               }
                           });

        std::thread searcher([&]()
                             {
                                 std::mt19937_64 rng(7);
                                 std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                                 SearchRequest req;
                                 req.topk = 5;
                                 req.metric = Metric::L2;
                                 req.query.data.resize(opt.dim);
                                 auto filter = std::make_shared<Filter>();
                                 filter->namespace_id = 1;
                                 filter->require_any_tags = {1, 2};
                                 std::sort(filter->require_any_tags.begin(), filter->require_any_tags.end());
                                 req.filter = filter;
                                 while (!stop.load())
                                 {
                                     for (std::size_t d = 0; d < opt.dim; ++d)
                                         req.query.data[d] = dist(rng);
                                     try
                                     {
                                         db.Search(req);
                                     }
                                     catch (...)
                                     {
                                         errors.fetch_add(1, std::memory_order_relaxed);
                                     }
                                 }
                             });

        std::this_thread::sleep_for(std::chrono::seconds(2));
        stop.store(true);
        ingest.join();
        searcher.join();

        if (errors.load() > 0)
        {
            std::cerr << "Concurrency test errors\n";
            failures++;
        }

        db.Stop();
        RemoveDir(opt.wal_dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Concurrency test error: " << e.what() << "\n";
        failures++;
    }

    if (failures == 0)
        std::cout << "Metadata filter tests passed.\n";
    else
        std::cout << "Metadata filter tests failed: " << failures << "\n";
    return failures == 0 ? 0 : 1;
}
