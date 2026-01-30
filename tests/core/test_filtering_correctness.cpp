#include <catch2/catch.hpp>

#include <pomai/api/pomai_db.h>
#include <pomai/core/seed.h>

#include "common/test_utils.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace
{
    using namespace pomai;
    using namespace pomai::test;

    std::size_t RecallAtK(const std::vector<SearchResultItem> &truth,
                          const std::vector<SearchResultItem> &got)
    {
        std::unordered_set<Id> truth_ids;
        truth_ids.reserve(truth.size());
        for (const auto &item : truth)
            truth_ids.insert(item.id);
        std::size_t hits = 0;
        for (const auto &item : got)
            if (truth_ids.count(item.id))
                ++hits;
        return hits;
    }

    bool MatchesTagsAny(Id id, const std::unordered_map<Id, std::vector<TagId>> &tag_map, const std::vector<TagId> &want)
    {
        auto it = tag_map.find(id);
        if (it == tag_map.end())
            return false;
        for (TagId tag : want)
        {
            if (std::find(it->second.begin(), it->second.end(), tag) != it->second.end())
                return true;
        }
        return false;
    }

    bool MatchesTagsAll(Id id, const std::unordered_map<Id, std::vector<TagId>> &tag_map, const std::vector<TagId> &want)
    {
        auto it = tag_map.find(id);
        if (it == tag_map.end())
            return false;
        for (TagId tag : want)
        {
            if (std::find(it->second.begin(), it->second.end(), tag) == it->second.end())
                return false;
        }
        return true;
    }
}

TEST_CASE("Namespace filter correctness recall", "[core][filter][recall]")
{
    TempDir dir;
    const std::size_t dim = 16;
    auto opts = DefaultDbOptions(dir.str(), dim, 1);
    PomaiDB db(opts);
    db.Start();

    std::vector<UpsertRequest> batch;
    batch.reserve(200);
    std::unordered_map<Id, std::uint32_t> id_to_ns;
    for (std::size_t i = 0; i < 200; ++i)
    {
        std::uint32_t ns = (i < 100) ? 1 : 2;
        batch.push_back(MakeUpsert(static_cast<Id>(i), dim, 0.01f * static_cast<float>(i), ns));
        id_to_ns.emplace(static_cast<Id>(i), ns);
    }
    db.UpsertBatch(batch, true).get();

    Filter filter;
    filter.namespace_id = 1;
    SearchRequest req = MakeSearchRequest(batch[42].vec, 10);
    req.search_mode = SearchMode::Quality;
    req.filter = std::make_shared<Filter>(filter);

    auto resp = db.Search(req);
    db.Stop();

    REQUIRE(resp.status == SearchStatus::Ok);
    for (const auto &item : resp.items)
        REQUIRE(id_to_ns[item.id] == 1);

    std::vector<UpsertRequest> filtered_rows;
    filtered_rows.reserve(100);
    for (const auto &row : batch)
        if (row.metadata.namespace_id == 1)
            filtered_rows.push_back(row);

    auto truth = BruteForceL2(filtered_rows, req.query, req.topk);
    const std::size_t hits = RecallAtK(truth, resp.items);
    const std::size_t needed = static_cast<std::size_t>(std::ceil(req.topk * 0.9));
    REQUIRE(hits >= needed);
}

TEST_CASE("Tags any/all correctness recall", "[core][filter][recall]")
{
    TempDir dir;
    const std::size_t dim = 16;
    auto opts = DefaultDbOptions(dir.str(), dim, 1);
    PomaiDB db(opts);
    db.Start();

    std::vector<UpsertRequest> batch;
    batch.reserve(200);
    std::unordered_map<Id, std::vector<TagId>> tag_map;
    for (std::size_t i = 0; i < 200; ++i)
    {
        std::vector<TagId> tags;
        if (i % 2 == 0)
            tags.push_back(2);
        if (i % 3 == 0)
            tags.push_back(3);
        batch.push_back(MakeUpsert(static_cast<Id>(i), dim, 0.02f * static_cast<float>(i), 0, tags));
        tag_map.emplace(static_cast<Id>(i), tags);
    }
    db.UpsertBatch(batch, true).get();

    Filter any_filter;
    any_filter.require_any_tags = {2, 3};
    SearchRequest any_req = MakeSearchRequest(batch[0].vec, 10);
    any_req.search_mode = SearchMode::Quality;
    any_req.filter = std::make_shared<Filter>(any_filter);
    auto any_resp = db.Search(any_req);

    Filter all_filter;
    all_filter.require_all_tags = {2, 3};
    SearchRequest all_req = MakeSearchRequest(batch[0].vec, 10);
    all_req.search_mode = SearchMode::Quality;
    all_req.filter = std::make_shared<Filter>(all_filter);
    auto all_resp = db.Search(all_req);

    db.Stop();

    for (const auto &item : any_resp.items)
        REQUIRE(MatchesTagsAny(item.id, tag_map, any_filter.require_any_tags));
    for (const auto &item : all_resp.items)
        REQUIRE(MatchesTagsAll(item.id, tag_map, all_filter.require_all_tags));

    std::vector<UpsertRequest> any_rows;
    std::vector<UpsertRequest> all_rows;
    for (const auto &row : batch)
    {
        if (MatchesTagsAny(row.id, tag_map, any_filter.require_any_tags))
            any_rows.push_back(row);
        if (MatchesTagsAll(row.id, tag_map, all_filter.require_all_tags))
            all_rows.push_back(row);
    }

    auto any_truth = BruteForceL2(any_rows, any_req.query, any_req.topk);
    auto all_truth = BruteForceL2(all_rows, all_req.query, all_req.topk);
    const std::size_t any_hits = RecallAtK(any_truth, any_resp.items);
    const std::size_t all_hits = RecallAtK(all_truth, all_resp.items);
    const std::size_t needed = static_cast<std::size_t>(std::ceil(any_req.topk * 0.9));
    REQUIRE(any_hits >= needed);
    REQUIRE(all_hits >= needed);
}

TEST_CASE("Candidate metadata/vector mapping invariants", "[core][filter][invariants]")
{
    const std::size_t dim = 8;
    Seed seed(dim);
    std::vector<UpsertRequest> batch;
    batch.reserve(64);
    for (std::size_t i = 0; i < 64; ++i)
    {
        std::uint32_t ns = (i < 32) ? 7 : 9;
        std::vector<TagId> tags;
        if (i % 2 == 0)
            tags.push_back(2);
        batch.push_back(MakeUpsert(static_cast<Id>(i), dim, 0.05f * static_cast<float>(i), ns, tags));
    }
    seed.ApplyUpserts(batch);
    auto snap = seed.MakeSnapshot();

    Filter filter;
    filter.namespace_id = 7;
    SearchRequest req = MakeSearchRequest(batch[10].vec, 10);
    req.filter = std::make_shared<Filter>(filter);

    auto resp = Seed::SearchSnapshot(snap, req);
    REQUIRE_FALSE(resp.items.empty());

    float prev_dist = -1.0f;
    std::vector<float> buf(dim);
    for (const auto &item : resp.items)
    {
        auto it = std::find(snap->ids.begin(), snap->ids.end(), item.id);
        REQUIRE(it != snap->ids.end());
        const std::size_t row = static_cast<std::size_t>(std::distance(snap->ids.begin(), it));
        REQUIRE(row < snap->namespace_ids.size());
        REQUIRE(row + 1 < snap->tag_offsets.size());
        REQUIRE(row * dim + dim <= snap->qdata.size());
        REQUIRE(snap->namespace_ids[row] == 7);
        Seed::DequantizeRow(snap, row, buf.data());
        float dist = 0.0f;
        for (std::size_t d = 0; d < dim; ++d)
        {
            float diff = buf[d] - req.query.data[d];
            dist += diff * diff;
        }
        REQUIRE(std::fabs(item.score + dist) <= 1e-3f);
        if (prev_dist >= 0.0f)
            REQUIRE(dist >= prev_dist - 1e-4f);
        prev_dist = dist;
    }
}
