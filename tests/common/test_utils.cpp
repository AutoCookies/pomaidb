#include "common/test_utils.h"

#include <pomai/util/search_utils.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <thread>

namespace pomai::test
{
    namespace
    {
        std::atomic<std::uint64_t> g_temp_counter{0};
    }

    TempDir::TempDir()
    {
        auto base = std::filesystem::temp_directory_path();
        std::ostringstream ss;
        ss << "pomai_test_" << std::this_thread::get_id() << "_" << g_temp_counter.fetch_add(1);
        path_ = base / ss.str();
        std::filesystem::create_directories(path_);
    }

    TempDir::~TempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    std::mt19937 &Rng()
    {
        static std::mt19937 rng(1337);
        return rng;
    }

    Vector MakeVector(std::size_t dim, float base)
    {
        Vector v;
        v.data.resize(dim);
        for (std::size_t i = 0; i < dim; ++i)
            v.data[i] = base + static_cast<float>(i) * 0.01f;
        return v;
    }

    UpsertRequest MakeUpsert(Id id,
                             std::size_t dim,
                             float base,
                             std::uint32_t ns,
                             std::vector<TagId> tags)
    {
        UpsertRequest req;
        req.id = id;
        req.vec = MakeVector(dim, base);
        req.metadata.namespace_id = ns;
        req.metadata.tag_ids = std::move(tags);
        return req;
    }

    std::vector<UpsertRequest> MakeBatch(std::size_t count,
                                         std::size_t dim,
                                         float start,
                                         std::uint32_t ns)
    {
        std::vector<UpsertRequest> batch;
        batch.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
            batch.push_back(MakeUpsert(static_cast<Id>(i + 1), dim, start + static_cast<float>(i), ns));
        return batch;
    }

    DbOptions DefaultDbOptions(const std::string &dir, std::size_t dim, std::size_t shards)
    {
        DbOptions opts;
        opts.dim = dim;
        opts.shards = shards;
        opts.metric = Metric::L2;
        opts.wal_dir = dir;
        opts.search_pool_workers = 1;
        opts.shard_queue_capacity = 256;
        opts.search_timeout_ms = 1000;
        return opts;
    }

    SearchRequest MakeSearchRequest(const Vector &query, std::size_t topk)
    {
        SearchRequest req;
        req.query = query;
        req.topk = topk;
        req.metric = Metric::L2;
        return req;
    }

    std::vector<SearchResultItem> BruteForceL2(const std::vector<UpsertRequest> &rows,
                                               const Vector &query,
                                               std::size_t topk)
    {
        std::vector<SearchResultItem> items;
        items.reserve(rows.size());
        for (const auto &row : rows)
        {
            float dist = 0.0f;
            for (std::size_t d = 0; d < row.vec.data.size(); ++d)
            {
                float diff = row.vec.data[d] - query.data[d];
                dist += diff * diff;
            }
            items.push_back({row.id, -dist});
        }
        SortAndDedupeResults(items, topk);
        return items;
    }

    std::vector<Id> ScanAll(PomaiDB &db, ScanRequest req)
    {
        std::vector<Id> ids;
        std::string cursor = req.cursor;
        while (true)
        {
            req.cursor = cursor;
            auto resp_res = db.Scan(req);
            if (!resp_res.ok())
                break;
            auto resp = resp_res.move_value();
            if (resp.status != ScanStatus::Ok)
                break;
            for (const auto &item : resp.items)
                ids.push_back(item.id);
            if (resp.next_cursor.empty())
                break;
            cursor = resp.next_cursor;
        }
        return ids;
    }

    bool ContainsId(const std::vector<SearchResultItem> &items, Id id)
    {
        return std::any_of(items.begin(), items.end(), [id](const SearchResultItem &item)
                           { return item.id == id; });
    }
}
