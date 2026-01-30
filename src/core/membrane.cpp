#include <pomai/core/membrane.h>
#include <pomai/util/search_utils.h>
#include <pomai/core/spatial_router.h>
#include <pomai/util/search_fanout.h>
#include <pomai/util/memory_manager.h>
#include <pomai/util/fixed_topk.h>
#include <pomai/storage/snapshot.h>
#include <pomai/storage/verify.h>

#include <stdexcept>
#include <algorithm>
#include <future>
#include <thread>
#include <chrono>
#include <array>
#include <vector>
#include <unordered_set>
#include <numeric>
#include <random>
#include <iostream>
#include <filesystem>
#include <cstring>
#include <cerrno>
#include <string_view>
#include <limits>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

namespace pomai
{
    namespace
    {
        constexpr std::size_t kCentroidsHeaderSize = 8 + 4 + 2 + 8;
        constexpr std::uint32_t kCentroidsVersion = 1;
        constexpr char kCentroidsMagic[8] = {'P', 'O', 'M', 'C', 'E', 'N', '0', '7'};
        constexpr std::size_t kShardCandidateMultiplier = 3;

        static std::size_t ChooseSearchPoolWorkers(std::size_t requested, std::size_t shard_count)
        {
            const std::size_t min_workers = 1;
            const std::size_t max_workers = 8;
            unsigned hc = std::thread::hardware_concurrency();
            if (hc == 0)
                hc = 1;
            std::size_t w = 0;
            if (requested == 0)
                w = std::min<std::size_t>(static_cast<std::size_t>(hc), shard_count);
            else
                w = requested;
            if (w < min_workers)
                w = min_workers;
            if (w > max_workers)
                w = max_workers;
            return w;
        }

        bool WriteFull(int fd, const void *buf, std::size_t len)
        {
            const char *p = static_cast<const char *>(buf);
            std::size_t remaining = len;
            while (remaining > 0)
            {
                ssize_t w = ::write(fd, p, remaining);
                if (w < 0)
                {
                    if (errno == EINTR)
                        continue;
                    return false;
                }
                if (w == 0)
                    return false;
                p += static_cast<std::size_t>(w);
                remaining -= static_cast<std::size_t>(w);
            }
            return true;
        }

        bool ReadFull(int fd, void *buf, std::size_t len)
        {
            char *p = static_cast<char *>(buf);
            std::size_t remaining = len;
            while (remaining > 0)
            {
                ssize_t r = ::read(fd, p, remaining);
                if (r < 0)
                {
                    if (errno == EINTR)
                        continue;
                    return false;
                }
                if (r == 0)
                    return false;
                p += static_cast<std::size_t>(r);
                remaining -= static_cast<std::size_t>(r);
            }
            return true;
        }

        std::string ParentDir(const std::string &path)
        {
            std::filesystem::path p(path);
            if (p.has_parent_path())
                return p.parent_path().string();
            return ".";
        }

        bool FsyncDir(const std::string &path)
        {
            std::string dir = ParentDir(path);
            int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
            if (dfd < 0)
                return false;
            int rc = ::fsync(dfd);
            ::close(dfd);
            return rc == 0;
        }

        std::uint16_t Le16ToHost(std::uint16_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap16(v);
#else
            return v;
#endif
        }

        std::uint32_t Le32ToHost(std::uint32_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap32(v);
#else
            return v;
#endif
        }

        std::uint64_t Le64ToHost(std::uint64_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap64(v);
#else
            return v;
#endif
        }

        std::uint16_t HostToLe16(std::uint16_t v) { return Le16ToHost(v); }
        std::uint32_t HostToLe32(std::uint32_t v) { return Le32ToHost(v); }
        std::uint64_t HostToLe64(std::uint64_t v) { return Le64ToHost(v); }

        float LeFloatToHost(float v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            std::uint32_t u;
            std::memcpy(&u, &v, sizeof(u));
            u = __builtin_bswap32(u);
            std::memcpy(&v, &u, sizeof(u));
#endif
            return v;
        }

        float HostToLeFloat(float v) { return LeFloatToHost(v); }

        bool FileExists(const std::string &path)
        {
            struct stat st;
            return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
        }
    }

    MembraneRouter::FilterConfig MembraneRouter::FilterConfig::Default()
    {
        return {};
    }

    MembraneRouter::MembraneRouter(std::vector<std::unique_ptr<Shard>> shards,
                                   pomai::WhisperConfig w_cfg,
                                   std::size_t dim,
                                   Metric metric,
                                   std::size_t search_pool_workers,
                                   std::size_t search_timeout_ms,
                                   std::size_t scan_batch_cap,
                                   std::size_t scan_id_order_max_rows,
                                   FilterConfig filter_config,
                                   std::function<void()> on_rejected_upsert)
        : shards_(std::move(shards)),
          brain_(w_cfg),
          probe_P_(2),
          dim_(dim),
          metric_(metric),
          search_timeout_ms_(search_timeout_ms),
          scan_batch_cap_(scan_batch_cap),
          scan_id_order_max_rows_(scan_id_order_max_rows),
          filter_config_(filter_config),
          search_pool_(ChooseSearchPoolWorkers(search_pool_workers, shards_.size())),
          on_rejected_upsert_(std::move(on_rejected_upsert))
    {
        if (shards_.empty())
            throw std::runtime_error("must have at least 1 shard");
    }

    void MembraneRouter::Start()
    {
        completion_.Start();
        if (centroids_load_mode_ != CentroidsLoadMode::None && centroids_load_mode_ != CentroidsLoadMode::Async)
        {
            if (!centroids_path_.empty() && FileExists(centroids_path_))
            {
                LoadCentroidsFromFile(centroids_path_);
            }
        }

        std::vector<std::future<void>> futures;
        futures.reserve(shards_.size());
        for (auto &s : shards_)
        {
            futures.push_back(std::async(std::launch::async, [&s]()
                                         { s->Start(); }));
        }
        for (auto &f : futures)
            f.get();
    }

    void MembraneRouter::Stop()
    {
        for (auto &s : shards_)
            s->Stop();
        search_pool_.Stop();
        completion_.Stop();
    }

    std::size_t MembraneRouter::PickShardById(Id id) const
    {
        return static_cast<std::size_t>(id % shards_.size());
    }

    std::size_t MembraneRouter::PickShard(Id id, const Vector *vec_opt) const
    {
        if (vec_opt)
        {
            try
            {
                std::size_t c_idx = router_.PickShardForInsert(*vec_opt);
                if (!centroid_to_shard_.empty())
                    return centroid_to_shard_[c_idx % centroid_to_shard_.size()];
                return c_idx % shards_.size();
            }
            catch (...)
            {
            }
        }
        return PickShardById(id);
    }

    Metadata MembraneRouter::NormalizeMetadata(const Metadata &meta)
    {
        Metadata out = meta;
        std::vector<TagId> tag_ids = meta.tag_ids;
        std::sort(tag_ids.begin(), tag_ids.end());
        tag_ids.erase(std::unique(tag_ids.begin(), tag_ids.end()), tag_ids.end());
        if (tag_ids.size() > filter_config_.max_tags_per_vector)
            throw std::runtime_error("max_tags_per_vector exceeded");
        out.tag_ids = std::move(tag_ids);
        return out;
    }

    std::shared_ptr<const Filter> MembraneRouter::NormalizeFilter(const SearchRequest &req) const
    {
        if (!req.filter)
            return nullptr;
        Filter f = *req.filter;
        auto normalize_vec = [](std::vector<TagId> &tags)
        {
            std::sort(tags.begin(), tags.end());
            tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
        };
        normalize_vec(f.require_all_tags);
        normalize_vec(f.require_any_tags);
        normalize_vec(f.exclude_tags);
        const std::size_t total_tags = f.require_all_tags.size() + f.require_any_tags.size() + f.exclude_tags.size();
        if (total_tags > filter_config_.max_filter_tags)
            throw std::runtime_error("max_filter_tags exceeded");
        return std::make_shared<Filter>(std::move(f));
    }

    std::future<Lsn> MembraneRouter::Upsert(Id id, Vector vec, bool wait_durable)
    {
        UpsertRequest r;
        r.id = id;
        r.vec = std::move(vec);
        std::vector<UpsertRequest> batch;
        batch.push_back(std::move(r));
        return UpsertBatch(std::move(batch), wait_durable);
    }

    std::future<Lsn> MembraneRouter::UpsertBatch(std::vector<UpsertRequest> batch, bool wait_durable)
    {
        if (batch.empty())
        {
            std::promise<Lsn> p;
            p.set_value(0);
            return p.get_future();
        }

        std::size_t est_bytes = 0;
        for (auto &r : batch)
        {
            if (r.vec.data.size() == dim_)
                est_bytes += sizeof(Id) + dim_ * sizeof(float);
            r.metadata = NormalizeMetadata(r.metadata);
        }

        if (!MemoryManager::Instance().CanAllocate(est_bytes))
        {
            if (on_rejected_upsert_)
                on_rejected_upsert_();
            std::promise<Lsn> p;
            p.set_exception(std::make_exception_ptr(std::runtime_error("UpsertBatch rejected: memory pressure")));
            return p.get_future();
        }

        std::vector<std::vector<UpsertRequest>> parts(shards_.size());
        for (auto &r : batch)
        {
            parts[PickShard(r.id, &r.vec)].push_back(std::move(r));
        }

        std::vector<std::future<Lsn>> futs;
        for (std::size_t i = 0; i < parts.size(); ++i)
        {
            if (!parts[i].empty())
                futs.push_back(shards_[i]->EnqueueUpserts(std::move(parts[i]), wait_durable));
        }

        auto done = std::make_shared<std::promise<Lsn>>();
        auto out = done->get_future();
        auto shared_futs = std::make_shared<std::vector<std::future<Lsn>>>(std::move(futs));
        auto task = [shared_futs, done]() mutable
        {
            Lsn max_lsn = 0;
            try
            {
                for (auto &f : *shared_futs)
                {
                    Lsn l = f.get();
                    if (l > max_lsn)
                        max_lsn = l;
                }
                done->set_value(max_lsn);
            }
            catch (...)
            {
                done->set_exception(std::current_exception());
            }
        };
        if (!completion_.Enqueue(std::move(task)))
            task();

        return out;
    }

    std::size_t MembraneRouter::TotalApproxCountUnsafe() const
    {
        std::size_t sum = 0;
        for (const auto &s : shards_)
            sum += s->ApproxCountUnsafe();
        return sum;
    }

    SearchResponse MembraneRouter::Search(const SearchRequest &req) const
    {
        auto start = std::chrono::steady_clock::now();
        auto deadline = start + std::chrono::milliseconds(search_timeout_ms_);
        auto budget = brain_.compute_budget(false);
        auto health = brain_.health();
        std::size_t adaptive_probe = probe_P_;
        SearchRequest normalized = req;
        normalized.filter = NormalizeFilter(req);
        const bool has_filter = normalized.filter && !normalized.filter->empty();
        if (normalized.filter && normalized.filter->match_none)
            return {};
        if (has_filter)
        {
            std::size_t base = std::max<std::size_t>(normalized.topk * 50, 500);
            std::size_t cap = filter_config_.filtered_candidate_k == 0 ? base : filter_config_.filtered_candidate_k;
            normalized.filtered_candidate_k = normalized.filtered_candidate_k == 0 ? std::min(base, cap) : std::max(normalized.filtered_candidate_k, normalized.topk);
            normalized.filtered_candidate_k = std::max(normalized.filtered_candidate_k, normalized.topk);
            std::uint32_t expand = normalized.filter_expand_factor == 0 ? filter_config_.filter_expand_factor : normalized.filter_expand_factor;
            expand = std::clamp<std::uint32_t>(expand, 1, 16);
            normalized.filter_expand_factor = expand;
            std::uint32_t max_visits = normalized.filter_max_visits == 0 ? filter_config_.filter_max_visits : normalized.filter_max_visits;
            normalized.filter_max_visits = max_visits;
            std::uint64_t time_budget_us = normalized.filter_time_budget_us == 0 ? filter_config_.filter_time_budget_us : normalized.filter_time_budget_us;
            normalized.filter_time_budget_us = time_budget_us;
        }

        if (health == pomai::ai::WhisperGrain::BudgetHealth::Tight)
            adaptive_probe = std::max<std::size_t>(1, probe_P_ / 2);
        else
            adaptive_probe = std::min<std::size_t>(probe_P_ + 1, shards_.size());

        std::vector<std::size_t> target_ids;
        try
        {
            auto c_idxs = router_.CandidateShardsForQuery(normalized.query, adaptive_probe);
            if (c_idxs.empty())
            {
                target_ids.resize(shards_.size());
                std::iota(target_ids.begin(), target_ids.end(), 0);
            }
            else
            {
                for (auto cidx : c_idxs)
                {
                    target_ids.push_back((!centroid_to_shard_.empty()) ? centroid_to_shard_[cidx % centroid_to_shard_.size()] : (cidx % shards_.size()));
                }
                std::vector<std::size_t> uniq;
                std::unordered_set<std::size_t> seen;
                for (auto s : target_ids)
                    if (seen.insert(s).second)
                        uniq.push_back(s);
                target_ids.swap(uniq);
            }
        }
        catch (...)
        {
            target_ids.resize(shards_.size());
            std::iota(target_ids.begin(), target_ids.end(), 0);
        }

        const std::size_t shard_topk = std::max<std::size_t>(normalized.topk, normalized.topk * kShardCandidateMultiplier);
        SearchRequest shard_req = normalized;
        shard_req.candidate_k = NormalizeCandidateK(shard_req);
        shard_req.max_rerank_k = NormalizeMaxRerankK(shard_req);
        shard_req.graph_ef = NormalizeGraphEf(shard_req, shard_req.candidate_k);
        shard_req.topk = shard_topk;
        if (shard_req.graph_ef > 0)
            budget.ops_budget = shard_req.graph_ef;
        auto req_ptr = std::make_shared<SearchRequest>(std::move(shard_req));

        std::vector<std::future<SearchResponse>> futs;
        std::vector<SearchResponse> inline_responses;
        for (auto sid : target_ids)
        {
            Shard *sh_ptr = shards_[sid].get();
            try
            {
                futs.emplace_back(search_pool_.Submit([req_ptr, budget, sh_ptr]()
                                                      { return sh_ptr->Search(*req_ptr, budget); }));
            }
            catch (...)
            {
                search_overload_.fetch_add(1, std::memory_order_relaxed);
                if (std::chrono::steady_clock::now() < deadline)
                {
                    inline_responses.push_back(sh_ptr->Search(*req_ptr, budget));
                    search_inline_.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }

        thread_local std::unique_ptr<FixedTopK> merge_topk;
        if (!merge_topk)
            merge_topk = std::make_unique<FixedTopK>(req.topk);
        merge_topk->Reset(req.topk);

        std::size_t completed = 0;
        bool filtered_partial = false;
        bool filtered_time_budget_hit = false;
        bool filtered_visit_budget_hit = false;
        bool filtered_budget_exhausted = false;
        for (auto &f : futs)
        {
            if (f.wait_until(deadline) == std::future_status::ready)
            {
                try
                {
                    auto r = f.get();
                    for (const auto &item : r.items)
                        merge_topk->Push(item.score, item.id);
                    filtered_partial = filtered_partial || r.stats.filtered_partial;
                    filtered_time_budget_hit = filtered_time_budget_hit || r.stats.filtered_time_budget_hit;
                    filtered_visit_budget_hit = filtered_visit_budget_hit || r.stats.filtered_visit_budget_hit;
                    filtered_budget_exhausted = filtered_budget_exhausted || r.stats.filtered_budget_exhausted;
                    ++completed;
                }
                catch (...)
                {
                }
            }
        }
        for (const auto &r : inline_responses)
        {
            for (const auto &item : r.items)
                merge_topk->Push(item.score, item.id);
            filtered_partial = filtered_partial || r.stats.filtered_partial;
            filtered_time_budget_hit = filtered_time_budget_hit || r.stats.filtered_time_budget_hit;
            filtered_visit_budget_hit = filtered_visit_budget_hit || r.stats.filtered_visit_budget_hit;
            filtered_budget_exhausted = filtered_budget_exhausted || r.stats.filtered_budget_exhausted;
            ++completed;
        }

        SearchResponse out;
        merge_topk->FillSorted(out.items);
        SortAndDedupeResults(out.items, req.topk);
        const std::size_t total_targets = futs.size() + inline_responses.size();
        if (completed < total_targets)
        {
            out.partial = true;
            search_partial_.fetch_add(1, std::memory_order_relaxed);
        }
        out.stats.partial = out.partial;
        out.stats.filtered_partial = filtered_partial;
        out.stats.filtered_time_budget_hit = filtered_time_budget_hit;
        out.stats.filtered_visit_budget_hit = filtered_visit_budget_hit;
        out.stats.filtered_budget_exhausted = filtered_budget_exhausted;
        if (filtered_time_budget_hit)
            search_budget_time_hit_.fetch_add(1, std::memory_order_relaxed);
        if (filtered_visit_budget_hit)
            search_budget_visit_hit_.fetch_add(1, std::memory_order_relaxed);
        if (filtered_budget_exhausted)
            search_budget_exhausted_.fetch_add(1, std::memory_order_relaxed);
        float lat = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
        brain_.observe_latency(lat);

        if (lat > 50.0f || brain_.latency_ema() > 50.0f)
        {
            for (auto &s : shards_)
                s->RequestEmergencyFreeze();
        }

        if ((search_count_.fetch_add(1, std::memory_order_relaxed) % 128) == 0)
        {
            auto hs = router_.DetectHotspot();
            std::lock_guard<std::mutex> lk(hotspot_mu_);
            if (hs)
                hotspot_ = HotspotInfo{(!centroid_to_shard_.empty()) ? centroid_to_shard_[hs->centroid_idx % centroid_to_shard_.size()] : (hs->centroid_idx % shards_.size()), hs->centroid_idx, hs->ratio};
            else
                hotspot_.reset();
        }

        return out;
    }

    std::shared_ptr<const MembraneRouter::ScanView> MembraneRouter::FindView(std::uint64_t epoch) const
    {
        std::lock_guard<std::mutex> lk(scan_views_mu_);
        for (const auto &view : scan_views_)
        {
            if (view && view->epoch == epoch)
                return view;
        }
        return nullptr;
    }

    std::shared_ptr<const MembraneRouter::ScanView> MembraneRouter::GetOrCreateView(ScanOrder order, ScanStatus &status) const
    {
        auto view = std::make_shared<ScanView>();
        view->epoch = scan_epoch_.fetch_add(1, std::memory_order_relaxed);
        view->total_rows = 0;
        for (std::size_t i = 0; i < shards_.size(); ++i)
        {
            auto state = shards_[i]->SnapshotState();
            if (!state)
                continue;
            for (const auto &seg : state->segments)
            {
                if (!seg.snap)
                    continue;
                view->grains.push_back({seg.snap, i});
                view->grain_row_counts.push_back(seg.snap->ids.size());
                view->total_rows += seg.snap->ids.size();
            }
            if (state->live_snap)
            {
                view->grains.push_back({state->live_snap, i});
                view->grain_row_counts.push_back(state->live_snap->ids.size());
                view->total_rows += state->live_snap->ids.size();
            }
        }
        if (order == ScanOrder::IdAsc)
        {
            if (view->total_rows > scan_id_order_max_rows_)
            {
                status = ScanStatus::UnsupportedOrder;
                return nullptr;
            }
            view->id_index.reserve(view->total_rows);
            for (std::size_t g = 0; g < view->grains.size(); ++g)
            {
                auto snap = view->grains[g].snap;
                if (!snap)
                    continue;
                for (std::size_t row = 0; row < snap->ids.size(); ++row)
                    view->id_index.emplace_back(g, row);
            }
            std::sort(view->id_index.begin(), view->id_index.end(),
                      [view](const auto &a, const auto &b)
                      {
                          const auto &sa = view->grains[a.first].snap;
                          const auto &sb = view->grains[b.first].snap;
                          Id ia = sa ? sa->ids[a.second] : 0;
                          Id ib = sb ? sb->ids[b.second] : 0;
                          return ia < ib;
                      });
        }
        {
            std::lock_guard<std::mutex> lk(scan_views_mu_);
            scan_views_.push_back(view);
            while (scan_views_.size() > 8)
                scan_views_.pop_front();
        }
        return view;
    }

    std::string MembraneRouter::EncodeCursor(std::uint64_t epoch, std::size_t grain, std::size_t row) const
    {
        return std::to_string(epoch) + ":" + std::to_string(grain) + ":" + std::to_string(row);
    }

    bool MembraneRouter::DecodeCursor(const std::string &cursor, std::uint64_t &epoch, std::size_t &grain, std::size_t &row) const
    {
        if (cursor.empty())
            return false;
        std::size_t p1 = cursor.find(':');
        if (p1 == std::string::npos)
            return false;
        std::size_t p2 = cursor.find(':', p1 + 1);
        if (p2 == std::string::npos)
            return false;
        try
        {
            epoch = std::stoull(cursor.substr(0, p1));
            grain = static_cast<std::size_t>(std::stoull(cursor.substr(p1 + 1, p2 - p1 - 1)));
            row = static_cast<std::size_t>(std::stoull(cursor.substr(p2 + 1)));
        }
        catch (...)
        {
            return false;
        }
        return true;
    }

    ScanResponse MembraneRouter::Scan(const ScanRequest &req) const
    {
        ScanResponse resp;
        const std::size_t batch_cap = scan_batch_cap_ == 0 ? 1024 : scan_batch_cap_;
        const std::size_t batch_size = std::min(req.batch_size == 0 ? std::size_t(1024) : req.batch_size, batch_cap);
        resp.items.reserve(batch_size);
        if (req.include_vectors)
            resp.vectors.reserve(batch_size * dim_);
        if (req.include_metadata)
            resp.tags.reserve(batch_size * 8);
        Filter normalized_filter = req.filter;
        auto normalize_vec = [](std::vector<TagId> &tags)
        {
            std::sort(tags.begin(), tags.end());
            tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
        };
        normalize_vec(normalized_filter.require_all_tags);
        normalize_vec(normalized_filter.require_any_tags);
        normalize_vec(normalized_filter.exclude_tags);
        const std::size_t total_tags = normalized_filter.require_all_tags.size() +
                                       normalized_filter.require_any_tags.size() +
                                       normalized_filter.exclude_tags.size();
        if (total_tags > filter_config_.max_filter_tags)
        {
            resp.status = ScanStatus::InvalidRequest;
            resp.error = "scan filter tags exceeded";
            return resp;
        }
        std::uint64_t epoch = 0;
        std::size_t grain_idx = 0;
        std::size_t row_idx = 0;
        std::shared_ptr<const ScanView> view;
        if (!req.cursor.empty())
        {
            if (!DecodeCursor(req.cursor, epoch, grain_idx, row_idx))
            {
                resp.status = ScanStatus::InvalidCursor;
                resp.error = "invalid cursor format";
                return resp;
            }
            view = FindView(epoch);
            if (!view)
            {
                resp.status = ScanStatus::InvalidCursor;
                resp.error = "cursor expired";
                return resp;
            }
            if (req.order == ScanOrder::IdAsc)
            {
                if (grain_idx > view->id_index.size())
                {
                    resp.status = ScanStatus::InvalidCursor;
                    resp.error = "cursor out of range";
                    return resp;
                }
            }
            else if (grain_idx > view->grains.size())
            {
                resp.status = ScanStatus::InvalidCursor;
                resp.error = "cursor out of range";
                return resp;
            }
            if (req.order == ScanOrder::Natural && grain_idx < view->grain_row_counts.size() && row_idx > view->grain_row_counts[grain_idx])
            {
                resp.status = ScanStatus::InvalidCursor;
                resp.error = "cursor row out of range";
                return resp;
            }
        }
        else
        {
            ScanStatus status = ScanStatus::Ok;
            view = GetOrCreateView(req.order, status);
            if (!view)
            {
                resp.status = status;
                resp.error = "scan order unsupported for snapshot size";
                return resp;
            }
            epoch = view->epoch;
            grain_idx = 0;
            row_idx = 0;
        }
        if (!view)
        {
            resp.status = ScanStatus::InvalidCursor;
            resp.error = "no view";
            return resp;
        }

        std::vector<float> scratch(dim_);
        bool filter_active = !normalized_filter.empty();
        while (resp.items.size() < batch_size)
        {
            if (req.order == ScanOrder::IdAsc)
            {
                if (grain_idx >= view->id_index.size())
                    break;
                const auto [g, r] = view->id_index[grain_idx];
                auto snap = view->grains[g].snap;
                if (!snap || r >= snap->ids.size())
                {
                    ++grain_idx;
                    continue;
                }
                resp.stats.scanned++;
                if (filter_active && !snap->MatchFilter(r, normalized_filter))
                {
                    ++grain_idx;
                    continue;
                }
                ScanItem item;
                item.id = snap->ids[r];
                if (req.include_metadata)
                {
                    if (r < snap->namespace_ids.size())
                        item.namespace_id = snap->namespace_ids[r];
                    std::uint32_t start = (r < snap->tag_offsets.size()) ? snap->tag_offsets[r] : 0;
                    std::uint32_t end = (r + 1 < snap->tag_offsets.size()) ? snap->tag_offsets[r + 1] : start;
                    if (start <= end && end <= snap->tag_ids.size())
                    {
                        item.tag_offset = resp.tags.size();
                        item.tag_count = end - start;
                        resp.tags.insert(resp.tags.end(), snap->tag_ids.begin() + start, snap->tag_ids.begin() + end);
                    }
                }
                if (req.include_vectors)
                {
                    item.vector_offset = resp.vectors.size();
                    Seed::DequantizeRow(snap, r, scratch.data());
                    resp.vectors.insert(resp.vectors.end(), scratch.begin(), scratch.end());
                }
                resp.items.push_back(item);
                resp.stats.returned++;
                ++grain_idx;
                continue;
            }

            if (grain_idx >= view->grains.size())
                break;
            auto snap = view->grains[grain_idx].snap;
            if (!snap)
            {
                ++grain_idx;
                row_idx = 0;
                continue;
            }
            if (row_idx >= snap->ids.size())
            {
                ++grain_idx;
                row_idx = 0;
                continue;
            }
            resp.stats.scanned++;
            if (filter_active && !snap->MatchFilter(row_idx, normalized_filter))
            {
                ++row_idx;
                continue;
            }
            ScanItem item;
            item.id = snap->ids[row_idx];
            if (req.include_metadata)
            {
                if (row_idx < snap->namespace_ids.size())
                    item.namespace_id = snap->namespace_ids[row_idx];
                std::uint32_t start = (row_idx < snap->tag_offsets.size()) ? snap->tag_offsets[row_idx] : 0;
                std::uint32_t end = (row_idx + 1 < snap->tag_offsets.size()) ? snap->tag_offsets[row_idx + 1] : start;
                if (start <= end && end <= snap->tag_ids.size())
                {
                    item.tag_offset = resp.tags.size();
                    item.tag_count = end - start;
                    resp.tags.insert(resp.tags.end(), snap->tag_ids.begin() + start, snap->tag_ids.begin() + end);
                }
            }
            if (req.include_vectors)
            {
                item.vector_offset = resp.vectors.size();
                Seed::DequantizeRow(snap, row_idx, scratch.data());
                resp.vectors.insert(resp.vectors.end(), scratch.begin(), scratch.end());
            }
            resp.items.push_back(item);
            resp.stats.returned++;
            ++row_idx;
        }

        if (req.order == ScanOrder::IdAsc)
        {
            if (grain_idx < view->id_index.size())
            {
                resp.next_cursor = EncodeCursor(epoch, grain_idx, 0);
                resp.stats.partial = true;
            }
        }
        else
        {
            if (grain_idx < view->grains.size())
            {
                resp.next_cursor = EncodeCursor(epoch, grain_idx, row_idx);
                if (row_idx < view->grain_row_counts[grain_idx])
                    resp.stats.partial = true;
            }
        }

        if (resp.stats.returned >= batch_size && !resp.next_cursor.empty())
            resp.stats.partial = true;
        if (resp.stats.returned > 0)
        {
            auto now = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> lk(scan_stats_mu_);
            if (scan_last_time_.time_since_epoch().count() == 0)
            {
                scan_last_time_ = now;
                scan_last_count_ = resp.stats.returned;
            }
            else
            {
                scan_last_count_ += resp.stats.returned;
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - scan_last_time_);
                if (elapsed.count() > 0)
                {
                    scan_items_per_sec_.store(scan_last_count_ / static_cast<std::uint64_t>(elapsed.count()), std::memory_order_relaxed);
                    scan_last_time_ = now;
                    scan_last_count_ = 0;
                }
            }
        }

        return resp;
    }

    std::future<bool> MembraneRouter::RequestCheckpoint()
    {
        std::vector<std::future<ShardCheckpointState>> futs;
        futs.reserve(shards_.size());
        for (auto &s : shards_)
            futs.push_back(s->RequestCheckpointState());
        auto done = std::make_shared<std::promise<bool>>();
        auto out = done->get_future();
        auto shared_futs = std::make_shared<std::vector<std::future<ShardCheckpointState>>>(std::move(futs));
        auto task = [this, shared_futs, done]() mutable
        {
            try
            {
                storage::SnapshotData snapshot;
                snapshot.schema.dim = static_cast<std::uint32_t>(dim_);
                snapshot.schema.metric = static_cast<std::uint32_t>(metric_);
                snapshot.schema.shards = static_cast<std::uint32_t>(shards_.size());
                snapshot.schema.index_kind = 0;
                snapshot.shards.clear();
                snapshot.shard_lsns.clear();
                snapshot.shards.reserve(shared_futs->size());
                snapshot.shard_lsns.reserve(shared_futs->size());
                std::size_t shard_id = 0;
                for (auto &f : *shared_futs)
                {
                    auto state = f.get();
                    state.shard_id = static_cast<std::uint32_t>(shard_id);
                    storage::ShardSnapshot shard;
                    shard.shard_id = state.shard_id;
                    shard.segments = std::move(state.segments);
                    shard.live = std::move(state.live);
                    snapshot.shards.push_back(std::move(shard));
                    snapshot.shard_lsns.push_back(state.durable_lsn);
                    ++shard_id;
                }

                storage::CommitResult res;
                std::string err;
                if (db_dir_.empty())
                {
                    done->set_value(false);
                    return;
                }
                if (!storage::CommitCheckpointAtomically(db_dir_, snapshot, {}, &res, &err))
                {
                    done->set_value(false);
                    return;
                }
                last_checkpoint_lsn_.store(res.manifest.checkpoint_lsn, std::memory_order_relaxed);
                done->set_value(true);
            }
            catch (...)
            {
                done->set_exception(std::current_exception());
            }
        };
        if (!completion_.Enqueue(std::move(task)))
            task();
        return out;
    }

    bool MembraneRouter::RecoverFromStorage(const std::string &db_dir, std::string *err)
    {
        storage::SnapshotData snapshot;
        storage::Manifest manifest;
        if (!storage::RecoverLatestCheckpoint(db_dir, snapshot, manifest, err))
            return false;
        last_checkpoint_lsn_.store(manifest.checkpoint_lsn, std::memory_order_relaxed);
        if (snapshot.schema.dim != dim_ || snapshot.schema.shards != shards_.size())
        {
            if (err)
                *err = "snapshot schema mismatch";
            return false;
        }
        std::vector<ShardCheckpointState> states;
        states.reserve(snapshot.shards.size());
        for (const auto &shard : snapshot.shards)
        {
            ShardCheckpointState state;
            state.shard_id = shard.shard_id;
            state.live = shard.live;
            state.segments = shard.segments;
            state.durable_lsn = 0;
            if (shard.shard_id < manifest.shard_lsns.size())
                state.durable_lsn = manifest.shard_lsns[shard.shard_id];
            else
                state.durable_lsn = manifest.checkpoint_lsn;
            states.push_back(std::move(state));
        }
        for (std::size_t i = 0; i < shards_.size() && i < states.size(); ++i)
            shards_[i]->LoadFromCheckpoint(states[i], states[i].durable_lsn);
        return true;
    }

    void MembraneRouter::ConfigureCentroids(const std::vector<Vector> &centroids)
    {
        router_.ReplaceCentroids(centroids);
        centroid_to_shard_.clear();
        for (std::size_t i = 0; i < centroids.size(); ++i)
            centroid_to_shard_.push_back(i % shards_.size());
    }

    void MembraneRouter::SetProbeCount(std::size_t p) { probe_P_ = (p == 0 ? 1 : p); }
    double MembraneRouter::SearchQueueAvgLatencyMs() const { return search_pool_.QueueWaitEmaMs(); }
    std::size_t MembraneRouter::CompactionBacklog() const
    {
        std::size_t total = 0;
        for (const auto &s : shards_)
            total += s->CompactionBacklog();
        return total;
    }

    std::uint64_t MembraneRouter::LastCompactionDurationMs() const
    {
        std::uint64_t last = 0;
        for (const auto &s : shards_)
            last = std::max(last, s->LastCompactionDurationMs());
        return last;
    }

    std::vector<std::uint64_t> MembraneRouter::WalLagLsns() const
    {
        std::vector<std::uint64_t> out;
        out.reserve(shards_.size());
        for (const auto &s : shards_)
        {
            auto written = s->WrittenLsn();
            auto durable = s->DurableLsn();
            out.push_back(written > durable ? written - durable : 0);
        }
        return out;
    }
    std::optional<MembraneRouter::HotspotInfo> MembraneRouter::CurrentHotspot() const
    {
        std::lock_guard<std::mutex> lk(hotspot_mu_);
        return hotspot_;
    }
    std::vector<Vector> MembraneRouter::SnapshotCentroids() const { return router_.SnapshotCentroids(); }
    bool MembraneRouter::HasCentroids() const { return !router_.SnapshotCentroids().empty(); }
    bool MembraneRouter::ScheduleCompletion(std::function<void()> fn, std::chrono::steady_clock::duration delay)
    {
        return completion_.Enqueue(std::move(fn), delay);
    }

    bool MembraneRouter::ComputeAndConfigureCentroids(std::size_t k, std::size_t total_samples)
    {
        if (shards_.empty())
            return false;
        const std::size_t S = shards_.size();
        std::size_t sample_size = std::clamp<std::size_t>(total_samples, 50000, 200000);
        auto &mm = MemoryManager::Instance();
        const std::size_t total = mm.TotalUsage();
        const std::size_t hard = mm.HardWatermarkBytes();
        const std::size_t avail = (hard > total) ? (hard - total) : 0;
        const std::size_t max_samples_by_mem = (dim_ > 0) ? (avail / (dim_ * sizeof(float))) : 0;
        if (max_samples_by_mem == 0)
            return false;
        sample_size = std::min(sample_size, max_samples_by_mem);
        std::vector<std::future<std::vector<Vector>>> futs;
        for (std::size_t i = 0; i < S; ++i)
        {
            Shard *sh = shards_[i].get();
            futs.push_back(std::async(std::launch::async, [sh, sample_size, S]()
                                      { return sh->SampleVectors(sample_size / S); }));
        }
        std::vector<Vector> aggregate;
        for (auto &f : futs)
        {
            try
            {
                auto p = f.get();
                aggregate.insert(aggregate.end(), std::make_move_iterator(p.begin()), std::make_move_iterator(p.end()));
            }
            catch (...)
            {
            }
        }
        if (aggregate.empty())
            return false;
        try
        {
            auto c = SpatialRouter::BuildKMeans(aggregate, k == 0 ? S * 32 : k, 10);
            ConfigureCentroids(c);
            if (!centroids_path_.empty())
                SaveCentroidsToFile(centroids_path_);
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    bool MembraneRouter::LoadCentroidsFromFile(const std::string &path)
    {
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0)
            return false;
        struct stat st;
        if (::fstat(fd, &st) != 0)
        {
            ::close(fd);
            return false;
        }
        std::array<char, 8> magic;
        std::uint32_t ver_le;
        std::uint16_t dim_le;
        std::uint64_t count_le;
        if (!ReadFull(fd, magic.data(), 8) || !ReadFull(fd, &ver_le, 4) || !ReadFull(fd, &dim_le, 2) || !ReadFull(fd, &count_le, 8))
        {
            ::close(fd);
            return false;
        }
        std::size_t count = Le64ToHost(count_le);
        std::size_t dim = Le16ToHost(dim_le);
        std::vector<float> flat(count * dim);
        if (!ReadFull(fd, flat.data(), count * dim * 4))
        {
            ::close(fd);
            return false;
        }
        std::uint64_t m_count_le;
        ReadFull(fd, &m_count_le, 8);
        std::vector<std::uint32_t> m(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            std::uint32_t v;
            ReadFull(fd, &v, 4);
            m[i] = Le32ToHost(v);
        }
        ::close(fd);
        std::vector<Vector> c(count);
        for (std::size_t i = 0; i < count; ++i)
            c[i].data.assign(flat.begin() + i * dim, flat.begin() + (i + 1) * dim);
        ConfigureCentroids(c);
        centroid_to_shard_.clear();
        for (auto val : m)
            centroid_to_shard_.push_back(val % shards_.size());
        return true;
    }

    bool MembraneRouter::SaveCentroidsToFile(const std::string &path) const
    {
        auto c = router_.SnapshotCentroids();
        if (c.empty())
            return false;
        std::string tmp = path + ".tmp";
        int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0)
            return false;
        WriteFull(fd, kCentroidsMagic, 8);
        std::uint32_t v_le = HostToLe32(kCentroidsVersion);
        std::uint16_t d_le = HostToLe16(static_cast<std::uint16_t>(dim_));
        std::uint64_t c_le = HostToLe64(c.size());
        WriteFull(fd, &v_le, 4);
        WriteFull(fd, &d_le, 2);
        WriteFull(fd, &c_le, 8);
        for (const auto &v : c)
            WriteFull(fd, v.data.data(), dim_ * 4);
        std::uint64_t mc_le = HostToLe64(centroid_to_shard_.size());
        WriteFull(fd, &mc_le, 8);
        for (auto sidx : centroid_to_shard_)
        {
            std::uint32_t m_le = HostToLe32(static_cast<std::uint32_t>(sidx));
            WriteFull(fd, &m_le, 4);
        }
        ::fdatasync(fd);
        ::close(fd);
        ::rename(tmp.c_str(), path.c_str());
        FsyncDir(path);
        return true;
    }

    void MembraneRouter::SetCentroidsFilePath(const std::string &path) { centroids_path_ = path; }
    void MembraneRouter::SetCentroidsLoadMode(CentroidsLoadMode mode) { centroids_load_mode_ = mode; }
}
