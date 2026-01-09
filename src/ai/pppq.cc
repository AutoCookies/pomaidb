#include "src/ai/pppq.h"

#include <random>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <future>
#include <chrono>

#include "src/core/config.h"

namespace pomai::ai
{

    static inline float l2sq_distance(const float *a, const float *b, size_t len)
    {
        float s = 0.0f;
        for (size_t i = 0; i < len; ++i)
        {
            float d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }

    PPPQ::PPPQ(size_t d, size_t m, size_t k, size_t max_elems, const std::string &mmap_file)
        : dim_(d), k_(k), max_elems_(max_elems), mmap_filename_(mmap_file)
    {
        // Auto-correct 'm' if dimension is too small or not divisible
        if (d < m)
        {
            m = d;
            // std::cerr << "[PPPQ] Reduced m to " << m << " to match dim\n";
        }

        // Find largest divisor <= requested m
        while (m > 0 && d % m != 0)
        {
            m--;
        }
        if (m == 0)
            m = 1; // Fallback

        m_ = m;
        subdim_ = d / m_; // Safe now

        codebooks_.assign(m_ * k_ * subdim_, 0.0f);
        precomp_dists_.resize(m_);
        codes8_.assign(max_elems_ * m_, 0);
        code_nbits_.assign(max_elems_, 8);
        in_mmap_.assign(max_elems_, 0);

        // PPEPredictor contains a std::mutex (non-movable/non-copyable).
        // Avoid vector::resize which may attempt moves/copies; use reserve + emplace_back.
        ppe_vecs_.reserve(max_elems_);
        for (size_t i = 0; i < max_elems_; ++i)
            ppe_vecs_.emplace_back();

        pending_demotes_.store(0);
        demote_tasks_completed_.store(0);
        demote_bytes_written_.store(0);
        demote_total_latency_ns_.store(0);

        // create/clear mmap file
        {
            std::ofstream f(mmap_filename_, std::ios::binary | std::ios::trunc);
            // Reserve space for max_elems_ * packed4BytesPerVec()
            size_t bytes = max_elems_ * packed4BytesPerVec();
            if (bytes)
            {
                // grow file to requested size (simple portable way)
                f.seekp(static_cast<std::streamoff>(bytes - 1));
                f.write("", 1);
            }
        }
    }

    PPPQ::~PPPQ()
    {
        // Wait for outstanding async demote tasks to complete
        std::lock_guard<std::mutex> lk(demote_futures_mu_);
        for (auto &fut : demote_futures_)
        {
            try
            {
                if (fut.valid())
                    fut.get();
            }
            catch (...)
            {
                // ignore exceptions from demote tasks in destructor
            }
        }
        demote_futures_.clear();
    }

    int PPPQ::findNearestCode(const float *subvec, float *codebook) const
    {
        int best = 0;
        float bestd = l2sq_distance(subvec, codebook + 0 * subdim_, subdim_);
        for (size_t c = 1; c < k_; ++c)
        {
            float d = l2sq_distance(subvec, codebook + c * subdim_, subdim_);
            if (d < bestd)
            {
                bestd = d;
                best = (int)c;
            }
        }
        return best;
    }

    void PPPQ::train(const float *samples, size_t n_samples, size_t max_iters)
    {
        // Simple k-means per-subquantizer
        std::mt19937_64 rng(12345);
        for (size_t sub = 0; sub < m_; ++sub)
        {
            // prepare sub-samples array view
            std::vector<const float *> subs;
            subs.reserve(n_samples);
            for (size_t i = 0; i < n_samples; ++i)
            {
                subs.push_back(samples + i * dim_ + sub * subdim_);
            }
            // init centroids by random unique samples
            std::vector<float> centroids(k_ * subdim_);
            std::uniform_int_distribution<size_t> ud(0, n_samples - 1);
            for (size_t c = 0; c < k_; ++c)
            {
                size_t idx = ud(rng);
                std::memcpy(&centroids[c * subdim_], subs[idx], sizeof(float) * subdim_);
            }
            std::vector<int> assignments(n_samples, 0);
            for (size_t iter = 0; iter < max_iters; ++iter)
            {
                bool changed = false;
                // assign
                for (size_t i = 0; i < n_samples; ++i)
                {
                    float bestd = l2sq_distance(subs[i], &centroids[0], subdim_);
                    int best = 0;
                    for (size_t c = 1; c < k_; ++c)
                    {
                        float d = l2sq_distance(subs[i], &centroids[c * subdim_], subdim_);
                        if (d < bestd)
                        {
                            bestd = d;
                            best = (int)c;
                        }
                    }
                    if (assignments[i] != best)
                    {
                        assignments[i] = best;
                        changed = true;
                    }
                }
                // update centroids
                std::vector<size_t> counts(k_, 0);
                std::vector<float> sums(k_ * subdim_, 0.0f);
                for (size_t i = 0; i < n_samples; ++i)
                {
                    int c = assignments[i];
                    counts[c]++;
                    for (size_t d = 0; d < subdim_; ++d)
                        sums[c * subdim_ + d] += subs[i][d];
                }
                for (size_t c = 0; c < k_; ++c)
                {
                    if (counts[c] == 0)
                    {
                        // reinit from random sample
                        size_t idx = ud(rng);
                        std::memcpy(&centroids[c * subdim_], subs[idx], sizeof(float) * subdim_);
                    }
                    else
                    {
                        for (size_t d = 0; d < subdim_; ++d)
                            centroids[c * subdim_ + d] = sums[c * subdim_ + d] / (float)counts[c];
                    }
                }
                if (!changed)
                    break;
            }
            // copy centroids to codebooks_
            std::memcpy(&codebooks_[sub * k_ * subdim_], &centroids[0], sizeof(float) * k_ * subdim_);
        }
        computePrecompDists();
    }

    void PPPQ::computePrecompDists()
    {
        precomp_dists_.clear();
        precomp_dists_.resize(m_);
        for (size_t sub = 0; sub < m_; ++sub)
        {
            auto &mat = precomp_dists_[sub];
            mat.assign(k_ * k_, 0.0f);
            float *cb = &codebooks_[sub * k_ * subdim_];
            for (size_t i = 0; i < k_; ++i)
            {
                for (size_t j = 0; j < k_; ++j)
                {
                    mat[i * k_ + j] = l2sq_distance(cb + i * subdim_, cb + j * subdim_, subdim_);
                }
            }
        }
    }

    void PPPQ::addVec(const float *vec, size_t id)
    {
        if (id >= max_elems_)
            throw std::out_of_range("id beyond max_elems");

        // Decide bits via PPE using configurable demote threshold from config runtime
        uint64_t demote_thresh_ms = pomai::config::runtime.demote_threshold_ms;
        uint8_t desired_bits = ppe_vecs_[id].predictBits(8, demote_thresh_ms);

        // compute codes (always compute 8-bit full codes first)
        for (size_t sub = 0; sub < m_; ++sub)
        {
            const float *subvec = vec + sub * subdim_;
            float *cb = &codebooks_[sub * k_ * subdim_];
            int code = findNearestCode(subvec, cb);
            codes8_[id * m_ + sub] = (uint8_t)code;
        }

        // If desired_bits==4 we try to demote. We attempt async demote when configured; otherwise fallback to sync.
        if (desired_bits == 4)
        {
            // prepare nibble buffer
            std::vector<uint8_t> nibble_buf(packed4BytesPerVec());
            pack4From8(&codes8_[id * m_], nibble_buf.data());

            // Read async demote configuration from runtime
            uint64_t max_pending = pomai::config::runtime.demote_async_max_pending;
            bool do_async = (max_pending > 0);

            if (do_async)
            {
                size_t cur = pending_demotes_.load(std::memory_order_acquire);
                if (cur >= static_cast<size_t>(max_pending))
                {
                    // Backpressure: if sync fallback enabled, perform synchronous write; otherwise skip demote now.
                    if (pomai::config::runtime.demote_sync_fallback)
                    {
                        writePacked4ToFile(id, nibble_buf.data(), nibble_buf.size());
                        code_nbits_[id] = 4;
                        in_mmap_[id] = 1;
                        // update metrics
                        demote_bytes_written_.fetch_add(static_cast<uint64_t>(nibble_buf.size()), std::memory_order_relaxed);
                        demote_tasks_completed_.fetch_add(1, std::memory_order_relaxed);
                        // no latency measurement for sync path here
                    }
                    else
                    {
                        // Skip demotion due to backpressure; keep 8-bit in RAM for now.
                        code_nbits_[id] = 8;
                        in_mmap_[id] = 0;
                    }
                }
                else
                {
                    schedule_async_demote(id, nibble_buf);
                    // mark as "in-mmap" tentatively to avoid races (actual write updates in worker)
                    code_nbits_[id] = 4;
                    in_mmap_[id] = 1;
                }
            }
            else
            {
                // No async support configured: synchronous demote (legacy)
                writePacked4ToFile(id, nibble_buf.data(), nibble_buf.size());
                code_nbits_[id] = 4;
                in_mmap_[id] = 1;
                demote_bytes_written_.fetch_add(static_cast<uint64_t>(nibble_buf.size()), std::memory_order_relaxed);
                demote_tasks_completed_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        else
        {
            code_nbits_[id] = 8;
            in_mmap_[id] = 0;
        }
        ppe_vecs_[id].touch();
    }

    void PPPQ::pack4From8(const uint8_t *src8, uint8_t *dst_nibbles)
    {
        // Pack m_ codes (each assumed <16) into nibbles
        size_t bi = 0;
        for (size_t i = 0; i < m_; i += 2)
        {
            uint8_t lo = src8[i] & 0x0F;
            uint8_t hi = 0;
            if (i + 1 < m_)
                hi = src8[i + 1] & 0x0F;
            dst_nibbles[bi++] = (uint8_t)((hi << 4) | lo);
        }
        // zero pad rest if any
        for (; bi < packed4BytesPerVec(); ++bi)
            dst_nibbles[bi] = 0;
    }

    void PPPQ::unpack4To8(const uint8_t *src_nibbles, uint8_t *dst8)
    {
        size_t bi = 0;
        for (size_t i = 0; i < m_; i += 2)
        {
            uint8_t v = src_nibbles[bi++];
            dst8[i] = v & 0x0F;
            if (i + 1 < m_)
                dst8[i + 1] = (v >> 4) & 0x0F;
        }
    }

    void PPPQ::writePacked4ToFile(size_t id, const uint8_t *nibble_bytes, size_t nibble_bytes_len)
    {
        std::lock_guard<std::mutex> g(mmap_file_mtx_);
        std::fstream fs(mmap_filename_, std::ios::in | std::ios::out | std::ios::binary);
        if (!fs)
        {
            // try create
            std::ofstream f(mmap_filename_, std::ios::binary | std::ios::app);
            f.close();
            fs.open(mmap_filename_, std::ios::in | std::ios::out | std::ios::binary);
        }
        size_t offset = id * packed4BytesPerVec();
        fs.seekp(static_cast<std::streamoff>(offset));
        fs.write(reinterpret_cast<const char *>(nibble_bytes), static_cast<std::streamsize>(nibble_bytes_len));
        fs.flush();
    }

    void PPPQ::readPacked4FromFile(size_t id, uint8_t *nibble_bytes, size_t nibble_bytes_len)
    {
        std::lock_guard<std::mutex> g(mmap_file_mtx_);
        std::ifstream fs(mmap_filename_, std::ios::binary);
        size_t offset = id * packed4BytesPerVec();
        fs.seekg(static_cast<std::streamoff>(offset));
        fs.read(reinterpret_cast<char *>(nibble_bytes), static_cast<std::streamsize>(nibble_bytes_len));
    }

    float PPPQ::approxDist(size_t id_a, size_t id_b)
    {
        if (id_a >= max_elems_ || id_b >= max_elems_)
            return std::numeric_limits<float>::infinity();
        // Ensure we have local 8-bit arrays for both (if stored 4-bit in file, unpack on demand)
        uint8_t codes_a[256];
        uint8_t codes_b[256];
        if (m_ > 256)
            throw std::runtime_error("m too large for stack buffers in prototype");

        // load A
        if (in_mmap_[id_a])
        {
            uint8_t buf[64];
            readPacked4FromFile(id_a, buf, packed4BytesPerVec());
            unpack4To8(buf, codes_a);
        }
        else
        {
            for (size_t s = 0; s < m_; ++s)
                codes_a[s] = codes8_[id_a * m_ + s];
        }
        // load B
        if (in_mmap_[id_b])
        {
            uint8_t buf[64];
            readPacked4FromFile(id_b, buf, packed4BytesPerVec());
            unpack4To8(buf, codes_b);
        }
        else
        {
            for (size_t s = 0; s < m_; ++s)
                codes_b[s] = codes8_[id_b * m_ + s];
        }

        // sum per-sub precomputed distances
        float sum = 0.0f;
        for (size_t sub = 0; sub < m_; ++sub)
        {
            int a = codes_a[sub];
            int b = codes_b[sub];
            sum += precomp_dists_[sub][a * k_ + b];
        }
        return sum;
    }

    void PPPQ::purgeCold(uint64_t cold_thresh_ms)
    {
        // If caller passed 0, use runtime default
        if (cold_thresh_ms == 0)
            cold_thresh_ms = pomai::config::runtime.demote_threshold_ms;

        uint64_t now = PPEPredictor::now_ms();

        // Collect candidate ids first
        std::vector<size_t> candidates;
        candidates.reserve(256);

        for (size_t id = 0; id < max_elems_; ++id)
        {
            if (in_mmap_[id])
                continue; // already demoted
            uint64_t pred = ppe_vecs_[id].predictNext();
            if (pred < now + cold_thresh_ms)
            {
                candidates.push_back(id);
            }
        }

        // Sort candidates by predicted next access descending (hot first publish)
        std::sort(candidates.begin(), candidates.end(), [this](size_t a, size_t b)
                  { return ppe_vecs_[a].predictNext() > ppe_vecs_[b].predictNext(); });

        // Process demotion in the sorted order
        for (size_t id : candidates)
        {
            // double-check flags (another thread might have changed)
            if (in_mmap_[id])
                continue;

            // demote: pack 4-bit and either schedule async demote (preferred) or perform sync demote
            std::vector<uint8_t> nibble_buf(packed4BytesPerVec());
            pack4From8(&codes8_[id * m_], nibble_buf.data());

            // decide async vs sync based on runtime config
            uint64_t max_pending = pomai::config::runtime.demote_async_max_pending;
            bool async_configured = (max_pending > 0);

            if (async_configured)
            {
                size_t cur = pending_demotes_.load(std::memory_order_acquire);

                if (cur >= static_cast<size_t>(max_pending))
                {
                    // backpressure reached
                    if (pomai::config::runtime.demote_sync_fallback)
                    {
                        // Do synchronous demote
                        writePacked4ToFile(id, nibble_buf.data(), nibble_buf.size());
                        in_mmap_[id] = 1;
                        code_nbits_[id] = 4;
                        // optionally zero RAM codes
                        for (size_t s = 0; s < m_; ++s)
                            codes8_[id * m_ + s] = 0;

                        demote_bytes_written_.fetch_add(static_cast<uint64_t>(nibble_buf.size()), std::memory_order_relaxed);
                        demote_tasks_completed_.fetch_add(1, std::memory_order_relaxed);
                    }
                    else
                    {
                        // Skip demotion due to backpressure
                        continue;
                    }
                }
                else
                {
                    // schedule asynchronous demote
                    schedule_async_demote(id, nibble_buf);
                    // mark as demoted logically; worker will perform actual write
                    in_mmap_[id] = 1;
                    code_nbits_[id] = 4;
                }
            }
            else
            {
                // synchronous fallback (legacy)
                writePacked4ToFile(id, nibble_buf.data(), nibble_buf.size());
                in_mmap_[id] = 1;
                code_nbits_[id] = 4;
                for (size_t s = 0; s < m_; ++s)
                    codes8_[id * m_ + s] = 0;

                demote_bytes_written_.fetch_add(static_cast<uint64_t>(nibble_buf.size()), std::memory_order_relaxed);
                demote_tasks_completed_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    void PPPQ::schedule_async_demote(size_t id, const std::vector<uint8_t> &nibble_buf)
    {
        // Increment pending counter (represents tasks in-flight or scheduled)
        pending_demotes_.fetch_add(1, std::memory_order_acq_rel);

        // Create async task and store future for join on destructor
        auto fut = std::async(std::launch::async, [this, id, nibble_buf]()
                              {
            const auto t0 = std::chrono::steady_clock::now();
            // Perform the file write under mmap_file_mtx_
            bool success = false;
            {
                std::lock_guard<std::mutex> g(mmap_file_mtx_);
                std::fstream fs(mmap_filename_, std::ios::in | std::ios::out | std::ios::binary);
                if (!fs)
                {
                    std::ofstream f(mmap_filename_, std::ios::binary | std::ios::app);
                    f.close();
                    fs.open(mmap_filename_, std::ios::in | std::ios::out | std::ios::binary);
                }
                if (fs)
                {
                    size_t offset = id * packed4BytesPerVec();
                    fs.seekp(static_cast<std::streamoff>(offset));
                    fs.write(reinterpret_cast<const char *>(nibble_buf.data()), static_cast<std::streamsize>(nibble_buf.size()));
                    fs.flush();
                    success = true;
                }
            }

            if (success)
            {
                // After successful write, zero RAM codes to free memory
                for (size_t s = 0; s < m_; ++s)
                    codes8_[id * m_ + s] = 0;

                // flags in_mmap_ and code_nbits_ are set by caller prior to scheduling,
                // but to be safe ensure in_mmap_ remains set.
                in_mmap_[id] = 1;
                code_nbits_[id] = 4;

                const auto t1 = std::chrono::steady_clock::now();
                uint64_t ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
                demote_total_latency_ns_.fetch_add(ns, std::memory_order_relaxed);
                demote_bytes_written_.fetch_add(static_cast<uint64_t>(nibble_buf.size()), std::memory_order_relaxed);
                demote_tasks_completed_.fetch_add(1, std::memory_order_relaxed);
            }
            else
            {
                // write failed: revert logical flags so caller may retry later
                in_mmap_[id] = 0;
                code_nbits_[id] = 8;
            }

            // Decrement pending counter
            pending_demotes_.fetch_sub(1, std::memory_order_acq_rel); });

        // store future
        {
            std::lock_guard<std::mutex> lk(demote_futures_mu_);
            demote_futures_.push_back(std::move(fut));
            // Optionally prune completed futures to avoid unbounded growth
            if (demote_futures_.size() > 1024)
            {
                std::vector<std::future<void>> alive;
                alive.reserve(demote_futures_.size());
                for (auto &f : demote_futures_)
                {
                    if (f.valid())
                    {
                        // try wait_for 0 to see if ready
                        if (f.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
                        {
                            try
                            {
                                f.get();
                            }
                            catch (...)
                            {
                            }
                            // drop
                        }
                        else
                        {
                            alive.push_back(std::move(f));
                        }
                    }
                }
                demote_futures_.swap(alive);
            }
        }
    }

} // namespace pomai::ai