/*
 * src/ai/pppq.cc
 *
 * Implementation of PPPQ prototype. This file contains:
 *  - constructor/destructor
 *  - kmeans training per-subquantizer
 *  - encode/addVec handling (computes 8-bit codes and optionally demotes to 4-bit)
 *  - approximate distance lookup (loads packed codes on demand)
 *  - demote scheduling (sync/async) with careful publish ordering using a seq protocol
 *
 * The publish protocol:
 *  - writer: seq.fetch_add(1) -> write payload -> atomic_store(code_nbits) -> atomic_store(in_mmap) -> seq.fetch_add(1)
 *  - reader: sample seq before/after reading metadata & payload and accept only when sequence unchanged and even.
 *
 * NOTE: stable_snapshot_read was adjusted to retry until a consistent snapshot is observed
 * (with a small backoff), rather than returning a possibly inconsistent "last-seen" value.
 * This prevents readers from accepting inconsistent in_mmap/code_nbits states under heavy
 * demote concurrency which was producing spurious test failures.
 */

#include "src/ai/pppq.h"

#include <random>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <future>
#include <chrono>
#include <atomic>
#include <thread>   // <--- ensure this is included so std::this_thread::sleep_for is available

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

    // ---------------- PPPQ ctor/dtor ----------------

    PPPQ::PPPQ(size_t d, size_t m, size_t k, size_t max_elems, const std::string &mmap_file)
        : dim_(d), k_(k), max_elems_(max_elems), mmap_filename_(mmap_file)
    {
        // sanitize m
        if (d < m)
            m = d;
        while (m > 0 && d % m != 0)
            --m;
        if (m == 0)
            m = 1;

        m_ = m;
        subdim_ = d / m_;

        codebooks_.assign(m_ * k_ * subdim_, 0.0f);
        precomp_dists_.resize(m_);
        codes8_.assign(max_elems_ * m_, 0);

        // allocate atomics
        code_nbits_.reset(new std::atomic<uint8_t>[max_elems_]);
        in_mmap_.reset(new std::atomic<uint8_t>[max_elems_]);
        seq_.reset(new std::atomic<uint32_t>[max_elems_]);

        for (size_t i = 0; i < max_elems_; ++i)
        {
            code_nbits_[i].store(static_cast<uint8_t>(8), std::memory_order_seq_cst);
            in_mmap_[i].store(static_cast<uint8_t>(0), std::memory_order_seq_cst);
            seq_[i].store(static_cast<uint32_t>(0), std::memory_order_seq_cst);
        }

        ppe_vecs_.reserve(max_elems_);
        for (size_t i = 0; i < max_elems_; ++i)
            ppe_vecs_.emplace_back();

        pending_demotes_.store(0);
        demote_tasks_completed_.store(0);
        demote_bytes_written_.store(0);
        demote_total_latency_ns_.store(0);

        // create/truncate mmap file and reserve space
        {
            std::ofstream f(mmap_filename_, std::ios::binary | std::ios::trunc);
            size_t bytes = max_elems_ * packed4BytesPerVec();
            if (bytes)
            {
                f.seekp(static_cast<std::streamoff>(bytes - 1));
                f.write("", 1);
            }
        }
    }

    PPPQ::~PPPQ()
    {
        // wait for outstanding demote futures
        std::lock_guard<std::mutex> lk(demote_futures_mu_);
        for (auto &f : demote_futures_)
        {
            try
            {
                if (f.valid())
                    f.get();
            }
            catch (...)
            {
            }
        }
        demote_futures_.clear();
    }

    double PPPQ::get_demote_avg_latency_ms() const noexcept
    {
        uint64_t done = demote_tasks_completed_.load(std::memory_order_acquire);
        if (done == 0)
            return 0.0;
        uint64_t tot_ns = demote_total_latency_ns_.load(std::memory_order_acquire);
        return static_cast<double>(tot_ns) / static_cast<double>(done) / 1e6;
    }

    void PPPQ::reset_demote_metrics()
    {
        pending_demotes_.store(0, std::memory_order_release);
        demote_tasks_completed_.store(0, std::memory_order_release);
        demote_bytes_written_.store(0, std::memory_order_release);
        demote_total_latency_ns_.store(0, std::memory_order_release);
    }

    // ---------------- training ----------------

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
                best = static_cast<int>(c);
            }
        }
        return best;
    }

    void PPPQ::train(const float *samples, size_t n_samples, size_t max_iters)
    {
        std::mt19937_64 rng(12345);
        for (size_t sub = 0; sub < m_; ++sub)
        {
            std::vector<const float *> subs;
            subs.reserve(n_samples);
            for (size_t i = 0; i < n_samples; ++i)
                subs.push_back(samples + i * dim_ + sub * subdim_);

            std::vector<float> centroids(k_ * subdim_);
            std::uniform_int_distribution<size_t> ud(0, n_samples - 1);
            for (size_t c = 0; c < k_; ++c)
                std::memcpy(&centroids[c * subdim_], subs[ud(rng)], sizeof(float) * subdim_);

            std::vector<int> assignments(n_samples, 0);

            for (size_t iter = 0; iter < max_iters; ++iter)
            {
                bool changed = false;
                // assign
                for (size_t i = 0; i < n_samples; ++i)
                {
                    int best = 0;
                    float bestd = l2sq_distance(subs[i], &centroids[0], subdim_);
                    for (size_t c = 1; c < k_; ++c)
                    {
                        float d = l2sq_distance(subs[i], &centroids[c * subdim_], subdim_);
                        if (d < bestd)
                        {
                            bestd = d;
                            best = static_cast<int>(c);
                        }
                    }
                    if (assignments[i] != best)
                    {
                        assignments[i] = best;
                        changed = true;
                    }
                }

                if (!changed)
                    break;

                // update centroids
                std::vector<size_t> counts(k_, 0);
                std::vector<double> sums(k_ * subdim_, 0.0);
                for (size_t i = 0; i < n_samples; ++i)
                {
                    int c = assignments[i];
                    counts[c]++;
                    for (size_t d = 0; d < subdim_; ++d)
                        sums[c * subdim_ + d] += static_cast<double>(subs[i][d]);
                }

                for (size_t c = 0; c < k_; ++c)
                {
                    if (counts[c] == 0)
                    {
                        std::memcpy(&centroids[c * subdim_], subs[ud(rng)], sizeof(float) * subdim_);
                    }
                    else
                    {
                        for (size_t d = 0; d < subdim_; ++d)
                            centroids[c * subdim_ + d] = static_cast<float>(sums[c * subdim_ + d] / static_cast<double>(counts[c]));
                    }
                }
            }

            // copy into codebooks_
            std::memcpy(&codebooks_[sub * k_ * subdim_], centroids.data(), sizeof(float) * k_ * subdim_);
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

    // ---------------- encoding / addVec ----------------

    void PPPQ::addVec(const float *vec, size_t id)
    {
        if (id >= max_elems_)
            throw std::out_of_range("id beyond max_elems");

        uint64_t demote_thresh_ms = pomai::config::runtime.demote_threshold_ms;
        uint8_t desired_bits = ppe_vecs_[id].predictBits(8, demote_thresh_ms);

        // compute 8-bit codes
        for (size_t sub = 0; sub < m_; ++sub)
        {
            const float *subvec = vec + sub * subdim_;
            float *cb = &codebooks_[sub * k_ * subdim_];
            int code = findNearestCode(subvec, cb);
            codes8_[id * m_ + sub] = static_cast<uint8_t>(code);
        }

        if (desired_bits == 4)
        {
            std::vector<uint8_t> nibble_buf(packed4BytesPerVec());
            pack4From8(&codes8_[id * m_], nibble_buf.data());

            uint64_t max_pending = pomai::config::runtime.demote_async_max_pending;
            bool do_async = (max_pending > 0);

            if (do_async)
            {
                size_t cur = pending_demotes_.load(std::memory_order_acquire);
                if (cur >= static_cast<size_t>(max_pending))
                {
                    if (pomai::config::runtime.demote_sync_fallback)
                    {
                        // synchronous demote under seq protocol
                        seq_[id].fetch_add(1, std::memory_order_seq_cst); // odd
                        writePacked4ToFile(id, nibble_buf.data(), nibble_buf.size());
                        code_nbits_[id].store(4, std::memory_order_seq_cst);
                        in_mmap_[id].store(1, std::memory_order_seq_cst);
                        for (size_t s = 0; s < m_; ++s)
                            codes8_[id * m_ + s] = 0;
                        seq_[id].fetch_add(1, std::memory_order_seq_cst); // even

                        demote_bytes_written_.fetch_add(static_cast<uint64_t>(nibble_buf.size()), std::memory_order_relaxed);
                        demote_tasks_completed_.fetch_add(1, std::memory_order_relaxed);
                    }
                    else
                    {
                        // skip demotion, keep as 8-bit
                    }
                }
                else
                {
                    // schedule async worker; it will handle seq and publishing
                    schedule_async_demote(id, nibble_buf);
                }
            }
            else
            {
                // synchronous path
                seq_[id].fetch_add(1, std::memory_order_seq_cst);
                writePacked4ToFile(id, nibble_buf.data(), nibble_buf.size());
                code_nbits_[id].store(4, std::memory_order_seq_cst);
                in_mmap_[id].store(1, std::memory_order_seq_cst);
                for (size_t s = 0; s < m_; ++s)
                    codes8_[id * m_ + s] = 0;
                seq_[id].fetch_add(1, std::memory_order_seq_cst);

                demote_bytes_written_.fetch_add(static_cast<uint64_t>(nibble_buf.size()), std::memory_order_relaxed);
                demote_tasks_completed_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        else
        {
            // publish metadata under seq protocol
            seq_[id].fetch_add(1, std::memory_order_seq_cst);
            code_nbits_[id].store(8, std::memory_order_seq_cst);
            in_mmap_[id].store(0, std::memory_order_seq_cst);
            seq_[id].fetch_add(1, std::memory_order_seq_cst);
        }

        ppe_vecs_[id].touch();
    }

    // ---------------- pack/unpack and file IO ----------------

    void PPPQ::pack4From8(const uint8_t *src8, uint8_t *dst_nibbles)
    {
        size_t bi = 0;
        for (size_t i = 0; i < m_; i += 2)
        {
            uint8_t lo = src8[i] & 0x0F;
            uint8_t hi = 0;
            if (i + 1 < m_)
                hi = src8[i + 1] & 0x0F;
            dst_nibbles[bi++] = static_cast<uint8_t>((hi << 4) | lo);
        }
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
            std::ofstream f(mmap_filename_, std::ios::binary | std::ios::app);
            f.close();
            fs.open(mmap_filename_, std::ios::in | std::ios::out | std::ios::binary);
        }
        if (!fs)
            return;
        size_t offset = id * packed4BytesPerVec();
        fs.seekp(static_cast<std::streamoff>(offset));
        fs.write(reinterpret_cast<const char *>(nibble_bytes), static_cast<std::streamsize>(nibble_bytes_len));
        fs.flush();
    }

    void PPPQ::readPacked4FromFile(size_t id, uint8_t *nibble_bytes, size_t nibble_bytes_len)
    {
        std::lock_guard<std::mutex> g(mmap_file_mtx_);
        std::ifstream fs(mmap_filename_, std::ios::binary);
        if (!fs)
            return;
        size_t offset = id * packed4BytesPerVec();
        fs.seekg(static_cast<std::streamoff>(offset));
        fs.read(reinterpret_cast<char *>(nibble_bytes), static_cast<std::streamsize>(nibble_bytes_len));
    }

    // ---------------- stable snapshot reader (member) ----------------

    bool PPPQ::stable_snapshot_read(size_t id, uint8_t &out_in_mmap, uint8_t &out_nbits, std::vector<uint8_t> &out_buf, size_t /*max_attempts*/)
    {
        // Use an unbounded retry loop with a small backoff. In practice writers are
        // short-lived (file write + a couple of atomic stores) so this will usually
        // succeed quickly; the backoff prevents excessive CPU usage under contention.
        while (true)
        {
            uint32_t s1 = seq_[id].load(std::memory_order_seq_cst);
            if (s1 & 1u)
            {
                // writer in progress; yield and retry
                std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                continue;
            }

            uint8_t inmap = in_mmap_[id].load(std::memory_order_seq_cst);
            uint8_t nbits = code_nbits_[id].load(std::memory_order_seq_cst);

            if (inmap)
            {
                // Read packed bytes under file mutex to ensure stable file contents.
                out_buf.resize(packed4BytesPerVec());
                readPacked4FromFile(id, out_buf.data(), packed4BytesPerVec());
            }
            else
            {
                out_buf.clear();
            }

            uint32_t s2 = seq_[id].load(std::memory_order_seq_cst);
            if (s1 == s2 && ((s1 & 1u) == 0u))
            {
                out_in_mmap = inmap;
                out_nbits = nbits;
                return true;
            }

            // Retry with small backoff to avoid busy-waiting under contention.
            std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        // unreachable
        return false;
    }

    // ---------------- approxDist ----------------

    float PPPQ::approxDist(size_t id_a, size_t id_b)
    {
        if (id_a >= max_elems_ || id_b >= max_elems_)
            return std::numeric_limits<float>::infinity();

        uint8_t inmap_a = 0, inmap_b = 0;
        uint8_t nbits_a = 8, nbits_b = 8;
        std::vector<uint8_t> packed_a, packed_b;

        // Wait until we can observe a consistent snapshot for each id.
        (void)stable_snapshot_read(id_a, inmap_a, nbits_a, packed_a);
        (void)stable_snapshot_read(id_b, inmap_b, nbits_b, packed_b);

        std::vector<uint8_t> unpack_a;
        std::vector<uint8_t> unpack_b;
        const uint8_t *codes_a = nullptr;
        const uint8_t *codes_b = nullptr;

        if (inmap_a)
        {
            unpack_a.resize(m_);
            unpack4To8(packed_a.data(), unpack_a.data());
            codes_a = unpack_a.data();
        }
        else
        {
            codes_a = &codes8_[id_a * m_];
        }

        if (inmap_b)
        {
            unpack_b.resize(m_);
            unpack4To8(packed_b.data(), unpack_b.data());
            codes_b = unpack_b.data();
        }
        else
        {
            codes_b = &codes8_[id_b * m_];
        }

        float sum = 0.0f;
        for (size_t sub = 0; sub < m_; ++sub)
        {
            int a = static_cast<int>(codes_a[sub]);
            int b = static_cast<int>(codes_b[sub]);
            sum += precomp_dists_[sub][a * k_ + b];
        }

        return sum;
    }

    // ---------------- purgeCold / schedule_async_demote ----------------

    void PPPQ::purgeCold(uint64_t cold_thresh_ms)
    {
        if (cold_thresh_ms == 0)
            cold_thresh_ms = pomai::config::runtime.demote_threshold_ms;

        uint64_t now = PPEPredictor::now_ms();

        std::vector<size_t> candidates;
        candidates.reserve(256);

        for (size_t id = 0; id < max_elems_; ++id)
        {
            if (in_mmap_[id].load(std::memory_order_seq_cst))
                continue;
            uint64_t pred = ppe_vecs_[id].predictNext();
            if (pred < now + cold_thresh_ms)
                candidates.push_back(id);
        }

        std::sort(candidates.begin(), candidates.end(), [this](size_t a, size_t b)
                  { return ppe_vecs_[a].predictNext() > ppe_vecs_[b].predictNext(); });

        for (size_t id : candidates)
        {
            if (in_mmap_[id].load(std::memory_order_seq_cst))
                continue;

            std::vector<uint8_t> nibble_buf(packed4BytesPerVec());
            pack4From8(&codes8_[id * m_], nibble_buf.data());

            uint64_t max_pending = pomai::config::runtime.demote_async_max_pending;
            bool async_configured = (max_pending > 0);

            if (async_configured)
            {
                size_t cur = pending_demotes_.load(std::memory_order_acquire);
                if (cur >= static_cast<size_t>(max_pending))
                {
                    if (pomai::config::runtime.demote_sync_fallback)
                    {
                        seq_[id].fetch_add(1, std::memory_order_seq_cst);
                        writePacked4ToFile(id, nibble_buf.data(), nibble_buf.size());
                        code_nbits_[id].store(4, std::memory_order_seq_cst);
                        in_mmap_[id].store(1, std::memory_order_seq_cst);
                        for (size_t s = 0; s < m_; ++s)
                            codes8_[id * m_ + s] = 0;
                        seq_[id].fetch_add(1, std::memory_order_seq_cst);

                        demote_bytes_written_.fetch_add(static_cast<uint64_t>(nibble_buf.size()), std::memory_order_relaxed);
                        demote_tasks_completed_.fetch_add(1, std::memory_order_relaxed);
                    }
                    else
                    {
                        continue;
                    }
                }
                else
                {
                    schedule_async_demote(id, nibble_buf);
                }
            }
            else
            {
                seq_[id].fetch_add(1, std::memory_order_seq_cst);
                writePacked4ToFile(id, nibble_buf.data(), nibble_buf.size());
                code_nbits_[id].store(4, std::memory_order_seq_cst);
                in_mmap_[id].store(1, std::memory_order_seq_cst);
                for (size_t s = 0; s < m_; ++s)
                    codes8_[id * m_ + s] = 0;
                seq_[id].fetch_add(1, std::memory_order_seq_cst);

                demote_bytes_written_.fetch_add(static_cast<uint64_t>(nibble_buf.size()), std::memory_order_relaxed);
                demote_tasks_completed_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    void PPPQ::schedule_async_demote(size_t id, const std::vector<uint8_t> &nibble_buf)
    {
        pending_demotes_.fetch_add(1, std::memory_order_acq_rel);

        auto fut = std::async(std::launch::async, [this, id, nibble_buf]()
        {
            const auto t0 = std::chrono::steady_clock::now();

            // mark writer-in-progress
            seq_[id].fetch_add(1, std::memory_order_seq_cst);

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
                code_nbits_[id].store(4, std::memory_order_seq_cst);
                in_mmap_[id].store(1, std::memory_order_seq_cst);
                for (size_t s = 0; s < m_; ++s)
                    codes8_[id * m_ + s] = 0;

                const auto t1 = std::chrono::steady_clock::now();
                uint64_t ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
                demote_total_latency_ns_.fetch_add(ns, std::memory_order_relaxed);
                demote_bytes_written_.fetch_add(static_cast<uint64_t>(nibble_buf.size()), std::memory_order_relaxed);
                demote_tasks_completed_.fetch_add(1, std::memory_order_relaxed);
            }
            else
            {
                in_mmap_[id].store(0, std::memory_order_seq_cst);
                code_nbits_[id].store(8, std::memory_order_seq_cst);
            }

            // mark done
            seq_[id].fetch_add(1, std::memory_order_seq_cst);
            pending_demotes_.fetch_sub(1, std::memory_order_acq_rel);
        });

        {
            std::lock_guard<std::mutex> lk(demote_futures_mu_);
            demote_futures_.push_back(std::move(fut));
            if (demote_futures_.size() > 1024)
            {
                std::vector<std::future<void>> alive;
                alive.reserve(demote_futures_.size());
                for (auto &f : demote_futures_)
                {
                    if (f.valid())
                    {
                        if (f.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
                        {
                            try { f.get(); } catch (...) {}
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