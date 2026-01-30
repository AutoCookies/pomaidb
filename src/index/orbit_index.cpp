#include <pomai/index/orbit_index.h>
#include <pomai/util/cpu_kernels.h>
#include <pomai/util/memory_manager.h>
#include <pomai/util/search_utils.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>

namespace pomai::core
{

    namespace
    {

        // Thread-local scratch to avoid allocations in Search / neighbor expansion.
        struct ThreadScratch
        {
            std::vector<std::uint32_t> visited; // token per node
            std::uint32_t token{1};

            void Reset(std::size_t n)
            {
                if (visited.size() < n)
                    visited.resize(n, 0);
                ++token;
                if (token == 0)
                { // overflow (extremely rare)
                    std::fill(visited.begin(), visited.end(), 0);
                    token = 1;
                }
            }

            bool IsVisited(std::uint32_t i) const { return visited[i] == token; }
            void Mark(std::uint32_t i) { visited[i] = token; }
        };

        thread_local ThreadScratch tls;

        // Min-heap for candidates by dist (smallest first).
        struct MinCmp
        {
            bool operator()(const std::pair<float, std::uint32_t> &a,
                            const std::pair<float, std::uint32_t> &b) const
            {
                return a.first > b.first;
            }
        };

        // Max-heap for current best (largest dist on top).
        struct MaxCmp
        {
            bool operator()(const std::pair<float, std::uint32_t> &a,
                            const std::pair<float, std::uint32_t> &b) const
            {
                return a.first < b.first;
            }
        };

    } // namespace

    OrbitIndex::OrbitIndex(std::size_t dim, std::size_t M, std::size_t ef_construction)
        : dim_(dim), M_(M), ef_construction_(ef_construction)
    {
        if (dim_ == 0)
            throw std::runtime_error("OrbitIndex dim must be > 0");
        if (M_ == 0)
            throw std::runtime_error("OrbitIndex M must be > 0");
        if (ef_construction_ == 0)
            ef_construction_ = 64;
    }

    OrbitIndex::~OrbitIndex()
    {
        if (accounted_bytes_ > 0)
        {
            MemoryManager::Instance().ReleaseUsage(MemoryManager::Pool::Indexing, accounted_bytes_);
        }
    }

    void OrbitIndex::Build(const std::vector<float> &flat_data, const std::vector<Id> &flat_ids)
    {
        if (built_.load(std::memory_order_acquire))
        {
            throw std::runtime_error("OrbitIndex::Build called twice");
        }
        if (flat_ids.empty())
        {
            // Empty index is valid
            data_.clear();
            ids_.clear();
            graph_.clear();
            total_vectors_.store(0, std::memory_order_release);
            built_.store(true, std::memory_order_release);
            return;
        }
        if (flat_data.size() != flat_ids.size() * dim_)
        {
            throw std::runtime_error("OrbitIndex::Build flat_data size mismatch");
        }

        const std::size_t N = flat_ids.size();

        // Copy data/ids once. This index owns its storage (immutable afterward).
        data_ = flat_data;
        ids_ = flat_ids;
        graph_.clear();
        graph_.resize(N);
        for (auto &g : graph_)
            g.reserve(M_ + 1);

        total_vectors_.store(N, std::memory_order_release);

        // Deterministic-ish RNG for adding a couple of long edges ("wormholes")
        std::mt19937 rng(123);

        // Build graph incrementally: for each node i, find neighbors among [0..i-1] and connect.
        // This is not full HNSW, but a pragmatic navigable small-world graph.
        for (std::uint32_t i = 0; i < (std::uint32_t)N; ++i)
        {
            if (i == 0)
                continue;

            const float *vec = data_.data() + (std::size_t)i * dim_;

            // Find neighbors using current graph (among previous nodes only)
            std::vector<std::uint32_t> neigh = FindNeighborsBuild(vec, i, ef_construction_);

            // Add 2 random long edges for global navigation when enough nodes exist
            if (i > 32)
            {
                std::uniform_int_distribution<std::uint32_t> dist(0, i - 1);
                for (int r = 0; r < 2; ++r)
                {
                    std::uint32_t rnd = dist(rng);
                    if (rnd != i)
                        neigh.push_back(rnd);
                }
            }

            // Connect to up to M_ neighbors
            if (!neigh.empty())
            {
                // Sort by distance to prefer closer connections first
                std::sort(neigh.begin(), neigh.end());
                neigh.erase(std::unique(neigh.begin(), neigh.end()), neigh.end());

                // Choose best-by-distance among candidates
                std::vector<std::pair<float, std::uint32_t>> scored;
                scored.reserve(neigh.size());
                for (auto nb : neigh)
                {
                    const float *v2 = data_.data() + (std::size_t)nb * dim_;
                    float d = pomai::kernels::L2Sqr(vec, v2, dim_);
                    scored.push_back({d, nb});
                }
                std::sort(scored.begin(), scored.end(),
                          [](const auto &a, const auto &b)
                          { return a.first < b.first; });

                const std::size_t take = std::min<std::size_t>(M_, scored.size());
                for (std::size_t k = 0; k < take; ++k)
                {
                    Connect(i, scored[k].second);
                }
            }
        }

        built_.store(true, std::memory_order_release);

        const std::size_t graph_bytes = [&]()
        {
            std::size_t bytes = 0;
            for (const auto &g : graph_)
                bytes += g.size() * sizeof(std::uint32_t);
            return bytes;
        }();
        accounted_bytes_ =
            data_.size() * sizeof(float) + ids_.size() * sizeof(Id) + graph_bytes;
        MemoryManager::Instance().AddUsage(MemoryManager::Pool::Indexing, accounted_bytes_);
    }

    void OrbitIndex::BuildFromMove(std::vector<float> &&flat_data, std::vector<Id> &&flat_ids)
    {
        if (built_.load(std::memory_order_acquire))
        {
            throw std::runtime_error("OrbitIndex::Build called twice");
        }
        if (flat_ids.empty())
        {
            data_.clear();
            ids_.clear();
            graph_.clear();
            total_vectors_.store(0, std::memory_order_release);
            built_.store(true, std::memory_order_release);
            return;
        }
        if (flat_data.size() != flat_ids.size() * dim_)
        {
            throw std::runtime_error("OrbitIndex::Build flat_data size mismatch");
        }

        const std::size_t N = flat_ids.size();

        data_ = std::move(flat_data);
        ids_ = std::move(flat_ids);

        graph_.clear();
        graph_.resize(N);
        for (auto &g : graph_)
            g.reserve(M_ + 1);

        total_vectors_.store(N, std::memory_order_release);

        std::mt19937 rng(123);

        for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(N); ++i)
        {
            if (i == 0)
                continue;

            const float *vec = data_.data() + static_cast<std::size_t>(i) * dim_;

            std::vector<std::uint32_t> neigh = FindNeighborsBuild(vec, i, ef_construction_);

            if (i > 32)
            {
                std::uniform_int_distribution<std::uint32_t> dist(0, i - 1);
                for (int r = 0; r < 2; ++r)
                {
                    std::uint32_t rnd = dist(rng);
                    if (rnd != i)
                        neigh.push_back(rnd);
                }
            }

            if (!neigh.empty())
            {
                std::sort(neigh.begin(), neigh.end());
                neigh.erase(std::unique(neigh.begin(), neigh.end()), neigh.end());

                std::vector<std::pair<float, std::uint32_t>> scored;
                scored.reserve(neigh.size());
                for (auto nb : neigh)
                {
                    const float *v2 = data_.data() + static_cast<std::size_t>(nb) * dim_;
                    float d = pomai::kernels::L2Sqr(vec, v2, dim_);
                    scored.push_back({d, nb});
                }
                std::sort(scored.begin(), scored.end(),
                          [](const auto &a, const auto &b)
                          { return a.first < b.first; });

                const std::size_t take = std::min<std::size_t>(M_, scored.size());
                for (std::size_t k = 0; k < take; ++k)
                {
                    Connect(i, scored[k].second);
                }
            }
        }

        built_.store(true, std::memory_order_release);

        const std::size_t graph_bytes = [&]()
        {
            std::size_t bytes = 0;
            for (const auto &g : graph_)
                bytes += g.size() * sizeof(std::uint32_t);
            return bytes;
        }();
        accounted_bytes_ =
            data_.size() * sizeof(float) + ids_.size() * sizeof(Id) + graph_bytes;
        MemoryManager::Instance().AddUsage(MemoryManager::Pool::Indexing, accounted_bytes_);
    }

    std::vector<std::uint32_t> OrbitIndex::FindNeighborsBuild(const float *vec, std::uint32_t curr_idx, std::size_t ef) const
    {
        // Search among [0..curr_idx-1]
        const std::uint32_t maxN = curr_idx;
        if (maxN == 0)
            return {};

        tls.Reset(maxN);

        std::priority_queue<std::pair<float, std::uint32_t>,
                            std::vector<std::pair<float, std::uint32_t>>,
                            MinCmp>
            candidates;

        std::priority_queue<std::pair<float, std::uint32_t>,
                            std::vector<std::pair<float, std::uint32_t>>,
                            MaxCmp>
            best;

        // Entry points: 0, mid, last
        std::uint32_t eps[3] = {0u, maxN / 2u, maxN - 1u};
        for (std::uint32_t ep : eps)
        {
            if (ep >= maxN)
                continue;
            if (tls.IsVisited(ep))
                continue;
            tls.Mark(ep);

            const float *v2 = data_.data() + (std::size_t)ep * dim_;
            float d = pomai::kernels::L2Sqr(vec, v2, dim_);
            candidates.push({d, ep});
            best.push({d, ep});
            if (best.size() > ef)
                best.pop();
        }

        while (!candidates.empty())
        {
            auto cur = candidates.top();
            candidates.pop();

            // If current is worse than the worst in best and best is full, stop.
            if (best.size() >= ef && cur.first > best.top().first)
                break;

            for (std::uint32_t nb : graph_[cur.second])
            {
                if (nb >= maxN)
                    continue;
                if (tls.IsVisited(nb))
                    continue;
                tls.Mark(nb);

                const float *v2 = data_.data() + (std::size_t)nb * dim_;
                float d = pomai::kernels::L2Sqr(vec, v2, dim_);

                candidates.push({d, nb});
                best.push({d, nb});
                if (best.size() > ef)
                    best.pop();
            }
        }

        std::vector<std::uint32_t> out;
        out.reserve(best.size());
        while (!best.empty())
        {
            out.push_back(best.top().second);
            best.pop();
        }
        return out;
    }

    void OrbitIndex::Connect(std::uint32_t a, std::uint32_t b)
    {
        if (a == b)
            return;

        auto &ga = graph_[a];
        if (std::find(ga.begin(), ga.end(), b) == ga.end())
        {
            ga.push_back(b);
            if (ga.size() > (std::size_t)(M_ * 3 / 2))
                Prune(a);
        }

        auto &gb = graph_[b];
        if (std::find(gb.begin(), gb.end(), a) == gb.end())
        {
            gb.push_back(a);
            if (gb.size() > (std::size_t)(M_ * 3 / 2))
                Prune(b);
        }
    }

    void OrbitIndex::Prune(std::uint32_t node)
    {
        auto &links = graph_[node];
        if (links.size() <= M_)
            return;

        const float *vec = data_.data() + (std::size_t)node * dim_;

        // Sort by distance to node
        std::sort(links.begin(), links.end(), [&](std::uint32_t x, std::uint32_t y)
                  {
        const float* vx = data_.data() + (std::size_t)x * dim_;
        const float* vy = data_.data() + (std::size_t)y * dim_;
        float dx = pomai::kernels::L2Sqr(vec, vx, dim_);
        float dy = pomai::kernels::L2Sqr(vec, vy, dim_);
        return dx < dy; });

        // Keep half closest, half randomized for navigation diversity
        const std::size_t keep_closest = M_ / 2;
        if (keep_closest < links.size())
        {
            static thread_local std::minstd_rand rrng(std::random_device{}());
            std::shuffle(links.begin() + keep_closest, links.end(), rrng);
        }
        links.resize(M_);
    }

    std::vector<std::uint32_t> OrbitIndex::FindNeighborsSearch(const float *q, std::size_t ef, std::size_t candidate_k) const
    {
        const std::size_t N = total_vectors_.load(std::memory_order_acquire);
        if (N == 0)
            return {};

        tls.Reset(N);

        std::priority_queue<std::pair<float, std::uint32_t>,
                            std::vector<std::pair<float, std::uint32_t>>,
                            MinCmp>
            candidates;

        std::priority_queue<std::pair<float, std::uint32_t>,
                            std::vector<std::pair<float, std::uint32_t>>,
                            MaxCmp>
            best;

        // Entry points
        std::uint32_t ep0 = 0;
        std::uint32_t ep1 = (N > 1) ? (std::uint32_t)(N / 2) : 0;
        std::uint32_t ep2 = (N > 2) ? (std::uint32_t)(N - 1) : 0;

        auto push_ep = [&](std::uint32_t ep)
        {
            if (ep >= N)
                return;
            if (tls.IsVisited(ep))
                return;
            tls.Mark(ep);
            const float *v = data_.data() + (std::size_t)ep * dim_;
            float d = pomai::kernels::L2Sqr(q, v, dim_);
            candidates.push({d, ep});
            best.push({d, ep});
        };

        push_ep(ep0);
        push_ep(ep1);
        push_ep(ep2);

        // ops_budget limits expansions (not exact dist calls, but close enough)
        std::size_t expansions = 0;
        ef = std::max<std::size_t>(64, std::min<std::size_t>(2048, ef));
        std::size_t best_cap = std::max<std::size_t>(1, candidate_k);

        while (!candidates.empty() && expansions < ef)
        {
            auto cur = candidates.top();
            candidates.pop();
            ++expansions;

            if (best.size() >= best_cap && cur.first > best.top().first)
                break;

            for (std::uint32_t nb : graph_[cur.second])
            {
                if (nb >= N)
                    continue;
                if (tls.IsVisited(nb))
                    continue;
                tls.Mark(nb);

                const float *v = data_.data() + (std::size_t)nb * dim_;
                float d = pomai::kernels::L2Sqr(q, v, dim_);

                candidates.push({d, nb});
                best.push({d, nb});
                if (best.size() > best_cap)
                    best.pop();
            }
        }

        std::vector<std::uint32_t> out;
        out.reserve(best.size());
        while (!best.empty())
        {
            out.push_back(best.top().second);
            best.pop();
        }
        return out;
    }

    std::vector<std::uint32_t> OrbitIndex::FindNeighborsSearchFiltered(const float *q,
                                                                       std::size_t ef,
                                                                       std::size_t candidate_k,
                                                                       std::size_t max_visits,
                                                                       std::uint64_t time_budget_us,
                                                                       std::size_t expand_factor,
                                                                       const Filter &filter,
                                                                       const Seed::Store &meta,
                                                                       bool &partial,
                                                                       bool &time_budget_hit,
                                                                       bool &visit_budget_hit) const
    {
        const std::size_t N = total_vectors_.load(std::memory_order_acquire);
        if (N == 0)
            return {};

        tls.Reset(N);
        partial = false;
        time_budget_hit = false;
        visit_budget_hit = false;

        std::priority_queue<std::pair<float, std::uint32_t>,
                            std::vector<std::pair<float, std::uint32_t>>,
                            MinCmp>
            candidates;

        std::priority_queue<std::pair<float, std::uint32_t>,
                            std::vector<std::pair<float, std::uint32_t>>,
                            MaxCmp>
            best;

        std::uint32_t ep0 = 0;
        std::uint32_t ep1 = (N > 1) ? (std::uint32_t)(N / 2) : 0;
        std::uint32_t ep2 = (N > 2) ? (std::uint32_t)(N - 1) : 0;

        auto push_ep = [&](std::uint32_t ep)
        {
            if (ep >= N)
                return;
            if (tls.IsVisited(ep))
                return;
            tls.Mark(ep);
            const float *v = data_.data() + (std::size_t)ep * dim_;
            float d = pomai::kernels::L2Sqr(q, v, dim_);
            candidates.push({d, ep});
            if (meta.MatchFilter(ep, filter))
                best.push({d, ep});
        };

        push_ep(ep0);
        push_ep(ep1);
        push_ep(ep2);

        std::size_t expansions = 0;
        const std::size_t target = std::max<std::size_t>(1, candidate_k);
        const std::size_t ef_target = std::max<std::size_t>(ef, target * std::max<std::size_t>(1, expand_factor));
        const std::size_t visit_limit = std::min<std::size_t>(ef_target, max_visits);
        const auto start_time = std::chrono::steady_clock::now();

        while (!candidates.empty() && expansions < visit_limit)
        {
            auto cur = candidates.top();
            candidates.pop();
            ++expansions;

            if (best.size() >= target && cur.first > best.top().first)
                break;

            for (std::uint32_t nb : graph_[cur.second])
            {
                if (nb >= N)
                    continue;
                if (tls.IsVisited(nb))
                    continue;
                tls.Mark(nb);

                const float *v = data_.data() + (std::size_t)nb * dim_;
                float d = pomai::kernels::L2Sqr(q, v, dim_);

                candidates.push({d, nb});
                if (meta.MatchFilter(nb, filter))
                {
                    best.push({d, nb});
                    if (best.size() > target)
                        best.pop();
                }
            }

            if (time_budget_us > 0 && (expansions % 64 == 0))
            {
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - start_time)
                                   .count();
                if (elapsed >= static_cast<std::int64_t>(time_budget_us))
                {
                    time_budget_hit = true;
                    break;
                }
            }
        }

        if (expansions >= visit_limit && best.size() < target)
            visit_budget_hit = true;

        if (best.size() < target)
            partial = true;

        std::vector<std::uint32_t> out;
        out.reserve(best.size());
        while (!best.empty())
        {
            out.push_back(best.top().second);
            best.pop();
        }
        return out;
    }

    SearchResponse OrbitIndex::Search(const SearchRequest &req, const pomai::ai::Budget &budget) const
    {
        SearchResponse resp;

        if (!built_.load(std::memory_order_acquire))
            return resp;
        if (req.query.data.size() != dim_)
            return resp;
        if (req.metric != Metric::L2)
            return resp;

        const std::size_t N = total_vectors_.load(std::memory_order_acquire);
        if (N == 0)
            return resp;

        const std::size_t candidate_k = NormalizeCandidateK(req);
        const std::uint32_t ef = NormalizeGraphEf(req, candidate_k);
        const std::size_t ops_budget = (budget.ops_budget == 0) ? ef : std::max<std::size_t>(budget.ops_budget, ef);
        const std::size_t search_ef = std::max<std::size_t>(ops_budget, candidate_k);

        // Get candidate nodes
        const float *q = req.query.data.data();
        std::vector<std::uint32_t> cand = FindNeighborsSearch(q, search_ef, candidate_k);

        // Score candidates exactly and return top results
        std::vector<std::pair<float, std::uint32_t>> scored;
        scored.reserve(cand.size());
        for (auto idx : cand)
        {
            const float *v = data_.data() + (std::size_t)idx * dim_;
            float d = pomai::kernels::L2Sqr(v, q, dim_);
            scored.push_back({d, idx});
        }

        std::sort(scored.begin(), scored.end(),
                  [](const auto &a, const auto &b)
                  {
                      if (a.first == b.first)
                          return a.second < b.second;
                      return a.first < b.first;
                  });

        const std::size_t limit = std::min<std::size_t>(req.topk, scored.size());
        resp.items.resize(limit);

        for (std::size_t i = 0; i < limit; ++i)
        {
            const auto idx = scored[i].second;
            resp.items[i] = {ids_[idx], -scored[i].first};
        }

        SortAndDedupeResults(resp.items, req.topk);
        resp.stats.filtered_candidates = scored.size();
        resp.stats.partial = resp.partial;

        return resp;
    }

    SearchResponse OrbitIndex::SearchFiltered(const SearchRequest &req,
                                              const pomai::ai::Budget &budget,
                                              const Filter &filter,
                                              const Seed::Store &meta) const
    {
        SearchResponse resp;

        if (!built_.load(std::memory_order_acquire))
            return resp;
        if (req.query.data.size() != dim_)
            return resp;
        if (req.metric != Metric::L2)
            return resp;
        if (filter.empty())
            return Search(req, budget);

        const std::size_t N = total_vectors_.load(std::memory_order_acquire);
        if (N == 0)
            return resp;

        const std::size_t candidate_k = (req.filtered_candidate_k > 0) ? std::max(req.filtered_candidate_k, req.topk)
                                                                        : NormalizeCandidateK(req);
        const std::uint32_t ef = NormalizeGraphEf(req, candidate_k);
        const std::size_t ops_budget = (budget.ops_budget == 0) ? ef : std::max<std::size_t>(budget.ops_budget, ef);
        const std::size_t search_ef = std::max<std::size_t>(ops_budget, candidate_k);
        const std::size_t max_visits = (req.filter_max_visits == 0) ? search_ef : std::max<std::size_t>(req.filter_max_visits, search_ef);
        const std::size_t expand_factor = (req.filter_expand_factor == 0) ? 1 : req.filter_expand_factor;
        const std::uint64_t time_budget_us = req.filter_time_budget_us;

        bool partial = false;
        bool time_budget_hit = false;
        bool visit_budget_hit = false;
        const float *q = req.query.data.data();
        std::vector<std::uint32_t> cand = FindNeighborsSearchFiltered(q,
                                                                      search_ef,
                                                                      candidate_k,
                                                                      max_visits,
                                                                      time_budget_us,
                                                                      expand_factor,
                                                                      filter,
                                                                      meta,
                                                                      partial,
                                                                      time_budget_hit,
                                                                      visit_budget_hit);

        std::vector<std::pair<float, std::uint32_t>> scored;
        scored.reserve(cand.size());
        for (auto idx : cand)
        {
            const float *v = data_.data() + (std::size_t)idx * dim_;
            float d = pomai::kernels::L2Sqr(v, q, dim_);
            scored.push_back({d, idx});
        }

        std::sort(scored.begin(), scored.end(),
                  [](const auto &a, const auto &b)
                  {
                      if (a.first == b.first)
                          return a.second < b.second;
                      return a.first < b.first;
                  });

        const std::size_t limit = std::min<std::size_t>(req.topk, scored.size());
        resp.items.resize(limit);

        for (std::size_t i = 0; i < limit; ++i)
        {
            const auto idx = scored[i].second;
            resp.items[i] = {ids_[idx], -scored[i].first};
        }

        SortAndDedupeResults(resp.items, req.topk);
        resp.stats.filtered_candidates = scored.size();
        resp.stats.filtered_partial = partial;
        resp.stats.filtered_time_budget_hit = time_budget_hit;
        resp.stats.filtered_visit_budget_hit = visit_budget_hit;
        resp.stats.filtered_budget_exhausted = (time_budget_hit || visit_budget_hit) && partial;
        if (resp.stats.filtered_budget_exhausted && req.search_mode == pomai::SearchMode::Quality)
            throw std::runtime_error("filtered search budget exhausted");
        resp.partial = resp.partial || partial;
        resp.stats.partial = resp.partial;

        return resp;
    }

} // namespace pomai::core
