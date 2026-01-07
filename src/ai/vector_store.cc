#include "src/ai/vector_store.h"

#include <cstring>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <random>
#include <future>

using namespace pomai::memory;

namespace pomai::ai
{

    VectorStore::~VectorStore()
    {
        stop_pppq_demoter();
    }

    bool VectorStore::init(size_t dim, size_t max_elements, size_t M, size_t ef_construction, PomaiArena *arena)
    {
        if (dim == 0 || max_elements == 0)
            return false;

        dim_ = dim;
        arena_ = arena;

        // create underlying L2 space for HNSW
        l2space_.reset(new hnswlib::L2Space(static_cast<int>(dim_)));

        try
        {
            pphnsw_.reset(new PPHNSW<float>(l2space_.get(), max_elements, M, ef_construction));
            // set a reasonable search ef
            size_t ef_for_search = std::max<size_t>(ef_construction, 64);
            pphnsw_->setEf(ef_for_search);
        }
        catch (const std::exception &e)
        {
            std::cerr << "VectorStore: failed to create PPHNSW: " << e.what() << "\n";
            return false;
        }

        if (arena_)
            pphnsw_->setPomaiArena(arena_);

        //
        // PPPQ: create, train quickly on random samples and attach to PPHNSW.
        //
        try
        {
            size_t pq_m = 8;                 // number of subquantizers
            size_t pq_k = 256;               // codebook size per sub
            size_t max_elems = max_elements; // must be >= max labels used

            auto ppq = std::make_unique<pomai::ai::PPPQ>(dim_, pq_m, pq_k, max_elems, "pppq_codes.mmap");

            // Quick synthetic training (replace with dataset samples for production)
            size_t n_train = std::min<size_t>(20000, max_elems);
            std::vector<float> samples;
            samples.resize(n_train * dim_);
            std::mt19937_64 rng(123456);
            std::uniform_real_distribution<float> ud(0.0f, 1.0f);
            for (size_t i = 0; i < n_train * dim_; ++i)
                samples[i] = ud(rng);

            ppq->train(samples.data(), n_train, 10);

            // Attach PPPQ into PPHNSW
            pphnsw_->setPPPQ(std::move(ppq));

            // Start background demoter that periodically calls PPPQ::purgeCold
            start_pppq_demoter();
        }
        catch (const std::exception &e)
        {
            std::cerr << "VectorStore: PPPQ init/train failed: " << e.what() << "\n";
            // non-fatal: continue without PPPQ
        }

        return true;
    }

    void VectorStore::attach_map(PomaiMap *map)
    {
        map_ = map;
    }

    bool VectorStore::enable_ivf(size_t num_clusters, size_t m_sub, size_t nbits, uint64_t seed)
    {
        if (dim_ == 0)
            return false;
        ivf_.reset(new PPIVF(dim_, num_clusters, m_sub, nbits));
        if (!ivf_->init_random_seed(seed))
            return false;
        ivf_enabled_ = true;
        return true;
    }

    std::unique_ptr<char[]> VectorStore::build_seed_buffer(const float *vec) const
    {
        size_t payload_size = dim_ * sizeof(float);
        std::unique_ptr<char[]> buf(new char[payload_size]);
        std::memcpy(buf.get(), reinterpret_cast<const char *>(vec), payload_size);
        return buf;
    }

    bool VectorStore::store_label_in_map(uint64_t label, const char *key, size_t klen)
    {
        if (!map_)
            return false;
        uint64_t le = label;
        const char *vptr = reinterpret_cast<const char *>(&le);
        bool ok = map_->put(key, static_cast<uint32_t>(klen), vptr, static_cast<uint32_t>(sizeof(le)));
        if (!ok)
            return false;
        Seed *s = map_->find_seed(key, static_cast<uint32_t>(klen));
        if (s)
            s->type = Seed::OBJ_VECTOR;
        return true;
    }

    uint64_t VectorStore::read_label_from_map(const char *key, size_t klen) const
    {
        if (!map_)
            return 0;
        uint32_t outlen = 0;
        const char *val = map_->get(key, static_cast<uint32_t>(klen), &outlen);
        if (!val || outlen != sizeof(uint64_t))
            return 0;
        uint64_t label = 0;
        std::memcpy(&label, val, sizeof(label));
        return label;
    }

    // Upsert: exclusive lock
    bool VectorStore::upsert(const char *key, size_t klen, const float *vec)
    {
        std::unique_lock<std::shared_mutex> write_lock(rw_mu_);

        if (!pphnsw_)
            return false;
        if (!key || klen == 0 || !vec)
            return false;

        const int quant_bits = 8;

        uint64_t label = 0;
        {
            std::lock_guard<std::mutex> lk(label_map_mu_);
            std::string skey(key, key + klen);
            auto it = key_to_label_.find(skey);
            if (it != key_to_label_.end())
                label = it->second;
        }
        if (label == 0 && map_)
        {
            label = read_label_from_map(key, klen);
            if (label != 0)
            {
                std::lock_guard<std::mutex> lk(label_map_mu_);
                std::string skey(key, key + klen);
                key_to_label_[skey] = label;
                label_to_key_[label] = skey;
            }
        }

        try
        {
            if (label != 0)
            {
                pphnsw_->addQuantizedPoint(vec, dim_, quant_bits, static_cast<hnswlib::labeltype>(label), /*replace_deleted=*/true);
            }
            else
            {
                uint64_t new_label = next_label_.fetch_add(1, std::memory_order_relaxed);
                pphnsw_->addQuantizedPoint(vec, dim_, quant_bits, static_cast<hnswlib::labeltype>(new_label), /*replace_deleted=*/false);

                if (map_)
                {
                    bool ok = store_label_in_map(new_label, key, klen);
                    if (!ok)
                    {
                        try
                        {
                            pphnsw_->markDelete(static_cast<hnswlib::labeltype>(new_label));
                        }
                        catch (...)
                        {
                        }
                        return false;
                    }
                }

                {
                    std::lock_guard<std::mutex> lk(label_map_mu_);
                    std::string skey(key, key + klen);
                    key_to_label_[skey] = new_label;
                    label_to_key_[new_label] = skey;
                }
                label = new_label;
            }

            if (ivf_enabled_ && ivf_)
            {
                int cl = ivf_->assign_cluster(vec);
                const uint8_t *code = ivf_->encode_pq(vec);
                ivf_->add_label(label, cl, code);
            }

            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "VectorStore::upsert exception: " << e.what() << "\n";
            return false;
        }
    }

    // Remove: exclusive lock
    bool VectorStore::remove(const char *key, size_t klen)
    {
        std::unique_lock<std::shared_mutex> write_lock(rw_mu_);

        if (!map_)
            return false;
        if (!key || klen == 0)
            return false;

        uint64_t label = 0;
        {
            std::lock_guard<std::mutex> lk(label_map_mu_);
            std::string skey(key, key + klen);
            auto it = key_to_label_.find(skey);
            if (it != key_to_label_.end())
            {
                label = it->second;
                key_to_label_.erase(it);
                label_to_key_.erase(label);
            }
        }
        if (label == 0)
        {
            label = read_label_from_map(key, klen);
        }

        bool map_erased = map_->erase(std::string(key, key + klen).c_str());

        if (label != 0 && pphnsw_)
        {
            try
            {
                pphnsw_->markDelete(static_cast<hnswlib::labeltype>(label));
            }
            catch (...)
            {
            }
        }

        return map_erased;
    }

    // Search: shared lock allows concurrent readers
    std::vector<std::pair<std::string, float>> VectorStore::search(const float *query, size_t dim, size_t topk)
    {
        std::shared_lock<std::shared_mutex> read_lock(rw_mu_);

        std::vector<std::pair<std::string, float>> out;
        if (!pphnsw_)
            return out;
        if (!query || dim != dim_ || topk == 0)
            return out;

        if (ivf_enabled_ && ivf_)
        {
            size_t clusters = ivf_->num_clusters();
            size_t probe_k = std::min<size_t>(clusters, std::max<size_t>(16, clusters / 100));
            auto probes = ivf_->probe_clusters(query, probe_k);
            std::unordered_set<int> probe_set(probes.begin(), probes.end());

            using Item = std::pair<double, const Seed *>;
            struct Cmp { bool operator()(Item const &a, Item const &b) const { return a.first < b.first; } };
            std::priority_queue<Item, std::vector<Item>, Cmp> heap;

            map_->scan_all([&](Seed *s)
                           {
                               if (!s) return;
                               if (s->type != Seed::OBJ_VECTOR) return;

                               uint16_t klen = s->get_klen();
                               const char *keyptr = s->payload;
                               uint64_t label = 0;
                               label = read_label_from_map(keyptr, klen);
                               if (label == 0) return;

                               int cl = ivf_->get_cluster_for_label(label);
                               if (cl < 0) return;
                               if (probe_set.find(cl) == probe_set.end()) return;

                               const char *vec_bytes = nullptr;
                               uint32_t blen = 0;
                               if ((s->flags & Seed::FLAG_INDIRECT) == 0)
                               {
                                   blen = s->get_vlen();
                                   if (blen == 0) return;
                                   vec_bytes = s->payload + klen;
                               }
                               else
                               {
                                   uint64_t offset = 0;
                                   std::memcpy(&offset, s->payload + klen, pomai::config::MAP_PTR_BYTES);
                                   const char *blob_hdr = arena_ ? arena_->blob_ptr_from_offset_for_map(offset) : nullptr;
                                   if (!blob_hdr) return;
                                   blen = *reinterpret_cast<const uint32_t *>(blob_hdr);
                                   if (blen == 0) return;
                                   vec_bytes = blob_hdr + sizeof(uint32_t);
                               }

                               if (!vec_bytes) return;
                               if (blen % sizeof(float) != 0) return;
                               size_t vec_len = blen / sizeof(float);
                               if (vec_len != dim_) return;

                               const float *vec = reinterpret_cast<const float *>(vec_bytes);
                               double sum = 0.0;
                               for (size_t i = 0; i < dim_; ++i)
                               {
                                   double d = static_cast<double>(query[i]) - static_cast<double>(vec[i]);
                                   sum += d * d;
                               }

                               if (heap.size() < topk)
                                   heap.emplace(sum, s);
                               else if (sum < heap.top().first)
                               {
                                   heap.pop();
                                   heap.emplace(sum, s);
                               }
                           });

            std::vector<Item> results;
            while (!heap.empty())
            {
                results.push_back(heap.top());
                heap.pop();
            }
            std::reverse(results.begin(), results.end());
            out.reserve(results.size());
            for (auto &it : results)
            {
                const Seed *s = it.second;
                uint16_t klen = s->get_klen();
                std::string key(s->payload, s->payload + klen);
                out.emplace_back(std::move(key), static_cast<float>(it.first));
            }
            return out;
        }

        // Fallback: HNSW adaptive search
        try
        {
            auto buf = build_seed_buffer(query);
            auto pq = pphnsw_->searchKnnAdaptive(buf.get(), topk, 0.0f);
            std::vector<std::pair<std::string, float>> tmp;
            tmp.reserve(pq.size());
            while (!pq.empty())
            {
                auto pr = pq.top();
                pq.pop();
                float dist = pr.first;
                uint64_t label = static_cast<uint64_t>(pr.second);

                std::string key;
                {
                    std::lock_guard<std::mutex> lk(label_map_mu_);
                    auto it = label_to_key_.find(label);
                    if (it != label_to_key_.end())
                        key = it->second;
                }
                if (key.empty() && map_)
                    key = std::to_string(label);
                tmp.emplace_back(std::move(key), dist);
            }
            std::reverse(tmp.begin(), tmp.end());
            out = std::move(tmp);
        }
        catch (const std::exception &e)
        {
            std::cerr << "VectorStore::search exception: " << e.what() << "\n";
        }

        return out;
    }

    size_t VectorStore::size() const
    {
        std::lock_guard<std::mutex> lk(label_map_mu_);
        return label_to_key_.size();
    }

    void VectorStore::start_pppq_demoter()
    {
        if (!pphnsw_)
            return;
        auto *ppq = pphnsw_->getPPPQ();
        if (!ppq)
            return;
        if (ppq_demote_running_.load(std::memory_order_acquire))
            return;

        ppq_demote_running_.store(true, std::memory_order_release);
        ppq_demote_thread_ = std::thread([this, ppq]()
                                         {
                                            while (ppq_demote_running_.load(std::memory_order_acquire))
                                            {
                                                try
                                                {
                                                    ppq->purgeCold(ppq_demote_cold_thresh_ms_);
                                                }
                                                catch (const std::exception &e)
                                                {
                                                    std::cerr << "PPPQ demoter exception: " << e.what() << "\n";
                                                }
                                                std::this_thread::sleep_for(std::chrono::milliseconds(ppq_demote_interval_ms_));
                                            } });
    }

    void VectorStore::stop_pppq_demoter()
    {
        ppq_demote_running_.store(false, std::memory_order_release);
        if (ppq_demote_thread_.joinable())
            ppq_demote_thread_.join();
    }

} // namespace pomai::ai