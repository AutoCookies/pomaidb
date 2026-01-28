#include "membrane.h"
#include "spatial_router.h"

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

        std::uint16_t HostToLe16(std::uint16_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap16(v);
#else
            return v;
#endif
        }

        std::uint32_t HostToLe32(std::uint32_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap32(v);
#else
            return v;
#endif
        }

        std::uint64_t HostToLe64(std::uint64_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap64(v);
#else
            return v;
#endif
        }

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

        float HostToLeFloat(float v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            std::uint32_t u;
            std::memcpy(&u, &v, sizeof(u));
            u = __builtin_bswap32(u);
            std::memcpy(&v, &u, sizeof(u));
#endif
            return v;
        }

        bool FileExists(const std::string &path)
        {
            struct stat st;
            return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
        }
    } // namespace

    MembraneRouter::MembraneRouter(std::vector<std::unique_ptr<Shard>> shards,
                                   pomai::server::WhisperConfig w_cfg,
                                   std::size_t dim)
        : shards_(std::move(shards)), brain_(w_cfg), probe_P_(2), dim_(dim)
    {
        if (shards_.empty())
            throw std::runtime_error("must have at least 1 shard");
    }

    void MembraneRouter::Start()
    {
        if (centroids_load_mode_ != CentroidsLoadMode::None && centroids_load_mode_ != CentroidsLoadMode::Async)
        {
            if (!centroids_path_.empty() && FileExists(centroids_path_))
            {
                if (LoadCentroidsFromFile(centroids_path_))
                {
                    std::cout << "[Router] Loaded centroids from " << centroids_path_ << "\n";
                }
                else
                {
                    std::cerr << "[Router] Failed to load centroids from " << centroids_path_
                              << " (will recompute in background)\n";
                }
            }
            else if (!centroids_path_.empty())
            {
                std::cout << "[Router] No centroids file at " << centroids_path_
                          << " (will recompute in background)\n";
            }
        }

        // PARALLEL BOOT: Khởi động tất cả Shard cùng lúc
        std::vector<std::future<void>> futures;
        futures.reserve(shards_.size());

        for (auto &s : shards_)
        {
            // Launch async: Mỗi shard start trên một thread riêng biệt
            futures.push_back(std::async(std::launch::async, [&s]()
                                         { s->Start(); }));
        }

        // Chờ tất cả Shard khởi động xong trước khi cho Server nhận request
        for (auto &f : futures)
        {
            f.get();
        }
    }

    void MembraneRouter::Stop()
    {
        // Stop cũng nên song song để tắt nhanh, nhưng tuần tự cho an toàn cũng được
        for (auto &s : shards_)
            s->Stop();
    }

    // Legacy id-based fallback
    std::size_t MembraneRouter::PickShardById(Id id) const
    {
        return static_cast<std::size_t>(id % shards_.size());
    }

    // New: pick shard using spatial router if centroids configured, otherwise fallback to id-mod
    std::size_t MembraneRouter::PickShard(Id id, const Vector *vec_opt) const
    {
        // Prefer routing by vector if provided and centroids exist
        if (vec_opt)
        {
            try
            {
                std::size_t centroid_idx = router_.PickShardForInsert(*vec_opt);
                if (!centroid_to_shard_.empty())
                    return centroid_to_shard_[centroid_idx % centroid_to_shard_.size()];
                // fallback mapping if centroid->shard mapping missing
                return centroid_idx % shards_.size();
            }
            catch (...)
            {
                // router not configured; fallthrough to id
            }
        }
        return PickShardById(id);
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
            auto f = p.get_future();
            p.set_value(0);
            return f;
        }

        // Route each upsert by its vector (if possible), otherwise fallback to id hashing.
        std::vector<std::vector<UpsertRequest>> parts(shards_.size());
        for (auto &r : batch)
        {
            std::size_t shard_id = PickShard(r.id, &r.vec);
            parts[shard_id].push_back(std::move(r));
        }

        std::vector<std::future<Lsn>> futs;
        futs.reserve(shards_.size());

        for (std::size_t i = 0; i < parts.size(); ++i)
        {
            if (!parts[i].empty())
            {
                futs.push_back(shards_[i]->EnqueueUpserts(std::move(parts[i]), wait_durable));
            }
        }

        std::promise<Lsn> done;
        auto out = done.get_future();

        std::thread([futs = std::move(futs), done = std::move(done)]() mutable
                    {
            Lsn max_lsn = 0;
            try {
                for (auto& f : futs) {
                    Lsn l = f.get();
                    if (l > max_lsn) max_lsn = l;
                }
                done.set_value(max_lsn);
            } catch (...) {
                done.set_exception(std::current_exception());
            } })
            .detach();

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

        auto budget = brain_.compute_budget(false);

        // If router has no centroids configured, fallback to broadcasting all shards (legacy behavior)
        std::vector<std::size_t> target_shard_ids;
        try
        {
            auto centroid_idxs = router_.CandidateShardsForQuery(req.query, probe_P_);
            if (centroid_idxs.empty())
            {
                // fallback: probe all shards
                target_shard_ids.resize(shards_.size());
                std::iota(target_shard_ids.begin(), target_shard_ids.end(), 0);
            }
            else
            {
                target_shard_ids.reserve(centroid_idxs.size());
                for (auto cidx : centroid_idxs)
                {
                    std::size_t sid = (!centroid_to_shard_.empty())
                                          ? centroid_to_shard_[cidx % centroid_to_shard_.size()]
                                          : (cidx % shards_.size());
                    target_shard_ids.push_back(sid);
                }
                // deduplicate while preserving order
                std::vector<std::size_t> uniq;
                std::unordered_set<std::size_t> seen;
                uniq.reserve(target_shard_ids.size());
                for (auto s : target_shard_ids)
                {
                    if (seen.insert(s).second)
                        uniq.push_back(s);
                }
                target_shard_ids.swap(uniq);
            }
        }
        catch (...)
        {
            // Router not configured or failed -> broadcast to all shards
            target_shard_ids.resize(shards_.size());
            std::iota(target_shard_ids.begin(), target_shard_ids.end(), 0);
        }

        std::vector<std::future<SearchResponse>> futs;
        futs.reserve(target_shard_ids.size());

        for (auto sid : target_shard_ids)
        {
            // avoid capturing member expressions in lambda capture list; take raw pointer first
            pomai::Shard *shard_ptr = shards_[sid].get();
            futs.push_back(std::async(std::launch::async, [&req, &budget, shard_ptr]()
                                      { return shard_ptr->Search(req, budget); }));
        }

        std::vector<SearchResultItem> all;
        for (auto &f : futs)
        {
            auto r = f.get();
            all.insert(all.end(), r.items.begin(), r.items.end());
        }

        std::sort(all.begin(), all.end(), [](const auto &a, const auto &b)
                  { return a.score > b.score; });
        if (all.size() > req.topk)
            all.resize(req.topk);

        SearchResponse out;
        out.items = std::move(all);

        auto end = std::chrono::steady_clock::now();
        float latency_ms = std::chrono::duration<float, std::milli>(end - start).count();
        brain_.observe_latency(latency_ms);

        return out;
    }

    std::future<bool> MembraneRouter::RequestCheckpoint()
    {
        std::vector<std::future<bool>> futs;
        futs.reserve(shards_.size());

        for (auto &s : shards_)
        {
            futs.push_back(s->RequestCheckpoint());
        }

        std::promise<bool> done;
        auto out = done.get_future();

        // Aggregate asynchronously so caller gets a future immediately.
        std::thread([futs = std::move(futs), done = std::move(done)]() mutable
                    {
            try
            {
                for (auto &f : futs)
                {
                    bool ok = f.get();
                    if (!ok)
                    {
                        done.set_value(false);
                        return;
                    }
                }
                done.set_value(true);
            }
            catch (...)
            {
                try
                {
                    done.set_exception(std::current_exception());
                }
                catch (...)
                {
                }
            } })
            .detach();

        return out;
    }

    // Admin / management helpers

    // Configure centroids (replace atomically). Also build a simple centroid->shard mapping
    // using round-robin assignment so centroids are distributed across shards.
    void MembraneRouter::ConfigureCentroids(const std::vector<Vector> &centroids)
    {
        // Replace router centroids
        router_.ReplaceCentroids(centroids);

        // Build centroid->shard mapping: round-robin assign centroids to shards for initial balance.
        centroid_to_shard_.clear();
        centroid_to_shard_.reserve(centroids.size());
        for (std::size_t i = 0; i < centroids.size(); ++i)
        {
            centroid_to_shard_.push_back(i % shards_.size());
        }
    }

    void MembraneRouter::SetProbeCount(std::size_t p)
    {
        probe_P_ = (p == 0 ? 1 : p);
    }

    std::vector<Vector> MembraneRouter::SnapshotCentroids() const
    {
        return router_.SnapshotCentroids();
    }

    bool MembraneRouter::HasCentroids() const
    {
        return !router_.SnapshotCentroids().empty();
    }

    // Compute centroids from samples across shards and install them atomically.
    bool MembraneRouter::ComputeAndConfigureCentroids(std::size_t k, std::size_t total_samples)
    {
        if (k == 0 || shards_.empty())
            return false;

        // per-shard sample budget (at least 1)
        std::size_t per_shard = std::max<std::size_t>(1, total_samples / shards_.size());

        // Collect samples concurrently from shards
        std::vector<std::future<std::vector<Vector>>> futs;
        futs.reserve(shards_.size());
        for (auto &s : shards_)
        {
            futs.push_back(std::async(std::launch::async, [s = s.get(), per_shard]()
                                      { return s->SampleVectors(per_shard); }));
        }

        std::vector<Vector> aggregate;
        for (auto &f : futs)
        {
            try
            {
                auto part = f.get();
                if (!part.empty())
                {
                    aggregate.insert(aggregate.end(), std::make_move_iterator(part.begin()), std::make_move_iterator(part.end()));
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "[Router] sample gather failed: " << e.what() << "\n";
            }
            catch (...)
            {
                std::cerr << "[Router] sample gather failed: unknown error\n";
            }
        }

        if (aggregate.empty())
        {
            std::cerr << "[Router] no samples collected for centroid build\n";
            return false;
        }

        // Downsample to reasonable kmeans input size if necessary
        const std::size_t MAX_KMEANS_INPUT = std::max<std::size_t>(4096, k * 64);
        if (aggregate.size() > MAX_KMEANS_INPUT)
        {
            std::mt19937_64 rng(std::random_device{}());
            std::shuffle(aggregate.begin(), aggregate.end(), rng);
            aggregate.resize(MAX_KMEANS_INPUT);
        }

        // Build centroids (this may be moderately CPU-heavy)
        std::vector<Vector> centroids;
        try
        {
            centroids = SpatialRouter::BuildKMeans(aggregate, k, /*iterations=*/10);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Router] BuildKMeans failed: " << e.what() << "\n";
            return false;
        }
        catch (...)
        {
            std::cerr << "[Router] BuildKMeans failed: unknown exception\n";
            return false;
        }

        // Install new centroids atomically and build simple centroid->shard mapping
        ConfigureCentroids(centroids);
        std::cout << "[Router] ConfigureCentroids: built " << centroids.size() << " centroids\n";

        if (!centroids_path_.empty() && centroids_load_mode_ != CentroidsLoadMode::None)
        {
            if (SaveCentroidsToFile(centroids_path_))
                std::cout << "[Router] Saved centroids to " << centroids_path_ << "\n";
            else
                std::cerr << "[Router] Failed to save centroids to " << centroids_path_ << "\n";
        }

        return true;
    }

    bool MembraneRouter::LoadCentroidsFromFile(const std::string &path)
    {
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0)
        {
            std::cerr << "[Router] open failed for centroids file " << path
                      << ": " << std::strerror(errno) << "\n";
            return false;
        }

        struct stat st;
        if (::fstat(fd, &st) != 0)
        {
            std::cerr << "[Router] fstat failed for centroids file " << path
                      << ": " << std::strerror(errno) << "\n";
            ::close(fd);
            return false;
        }

        if (static_cast<std::size_t>(st.st_size) < kCentroidsHeaderSize)
        {
            std::cerr << "[Router] centroids file too small: " << path << "\n";
            ::close(fd);
            return false;
        }

        std::array<char, 8> magic{};
        std::uint32_t version_le = 0;
        std::uint16_t dim_le = 0;
        std::uint64_t count_le = 0;

        if (!ReadFull(fd, magic.data(), magic.size()) ||
            !ReadFull(fd, &version_le, sizeof(version_le)) ||
            !ReadFull(fd, &dim_le, sizeof(dim_le)) ||
            !ReadFull(fd, &count_le, sizeof(count_le)))
        {
            std::cerr << "[Router] failed to read centroids header: " << path << "\n";
            ::close(fd);
            return false;
        }

        if (std::memcmp(magic.data(), kCentroidsMagic, sizeof(kCentroidsMagic)) != 0)
        {
            std::cerr << "[Router] invalid centroids magic: " << path << "\n";
            ::close(fd);
            return false;
        }

        std::uint32_t version = Le32ToHost(version_le);
        if (version != kCentroidsVersion)
        {
            std::cerr << "[Router] unsupported centroids version " << version << ": " << path << "\n";
            ::close(fd);
            return false;
        }

        std::uint16_t dim = Le16ToHost(dim_le);
        if (dim_ != 0 && dim != dim_)
        {
            std::cerr << "[Router] centroids dim mismatch: file=" << dim
                      << " expected=" << dim_ << "\n";
            ::close(fd);
            return false;
        }

        std::uint64_t count = Le64ToHost(count_le);
        if (count == 0)
        {
            std::cerr << "[Router] centroids file has zero count: " << path << "\n";
            ::close(fd);
            return false;
        }

        if (dim == 0)
        {
            std::cerr << "[Router] centroids file has zero dim: " << path << "\n";
            ::close(fd);
            return false;
        }

        if (count > (std::numeric_limits<std::size_t>::max() / dim))
        {
            std::cerr << "[Router] centroids file size overflow: " << path << "\n";
            ::close(fd);
            return false;
        }

        std::size_t total_floats = static_cast<std::size_t>(count * dim);
        std::size_t expected_size = kCentroidsHeaderSize + total_floats * sizeof(float);
        if (static_cast<std::size_t>(st.st_size) != expected_size)
        {
            std::cerr << "[Router] centroids file size mismatch: expected " << expected_size
                      << " got " << st.st_size << "\n";
            ::close(fd);
            return false;
        }

        std::vector<float> flat(total_floats);
        if (!ReadFull(fd, flat.data(), total_floats * sizeof(float)))
        {
            std::cerr << "[Router] failed to read centroids payload: " << path << "\n";
            ::close(fd);
            return false;
        }
        ::close(fd);

#if __BYTE_ORDER == __BIG_ENDIAN
        for (auto &v : flat)
            v = LeFloatToHost(v);
#endif

        std::vector<Vector> centroids;
        centroids.reserve(static_cast<std::size_t>(count));
        auto it = flat.begin();
        for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i)
        {
            Vector v;
            v.data.assign(it, it + dim);
            centroids.push_back(std::move(v));
            it += dim;
        }

        ConfigureCentroids(centroids);
        return true;
    }

    bool MembraneRouter::SaveCentroidsToFile(const std::string &path) const
    {
        auto centroids = router_.SnapshotCentroids();
        if (centroids.empty())
        {
            std::cerr << "[Router] no centroids to save\n";
            return false;
        }

        std::size_t dim = centroids.front().data.size();
        if (dim == 0)
        {
            std::cerr << "[Router] centroids have zero dim\n";
            return false;
        }
        for (const auto &c : centroids)
        {
            if (c.data.size() != dim)
            {
                std::cerr << "[Router] inconsistent centroid dims\n";
                return false;
            }
        }
        if (dim_ != 0 && dim != dim_)
        {
            std::cerr << "[Router] centroids dim mismatch: " << dim << " expected " << dim_ << "\n";
            return false;
        }

        std::string tmp_path = path + ".tmp";
        int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0)
        {
            std::cerr << "[Router] open failed for " << tmp_path << ": " << std::strerror(errno) << "\n";
            return false;
        }

        if (!WriteFull(fd, kCentroidsMagic, sizeof(kCentroidsMagic)))
        {
            std::cerr << "[Router] write magic failed\n";
            ::close(fd);
            return false;
        }

        std::uint32_t version_le = HostToLe32(kCentroidsVersion);
        std::uint16_t dim_le = HostToLe16(static_cast<std::uint16_t>(dim));
        std::uint64_t count_le = HostToLe64(static_cast<std::uint64_t>(centroids.size()));

        if (!WriteFull(fd, &version_le, sizeof(version_le)) ||
            !WriteFull(fd, &dim_le, sizeof(dim_le)) ||
            !WriteFull(fd, &count_le, sizeof(count_le)))
        {
            std::cerr << "[Router] write header failed\n";
            ::close(fd);
            return false;
        }

        for (const auto &c : centroids)
        {
            for (float f : c.data)
            {
                float out = HostToLeFloat(f);
                if (!WriteFull(fd, &out, sizeof(out)))
                {
                    std::cerr << "[Router] write centroid payload failed\n";
                    ::close(fd);
                    return false;
                }
            }
        }

        if (::fdatasync(fd) != 0)
        {
            std::cerr << "[Router] fdatasync failed for " << tmp_path << ": " << std::strerror(errno) << "\n";
            ::close(fd);
            return false;
        }
        if (::close(fd) != 0)
        {
            std::cerr << "[Router] close failed for " << tmp_path << ": " << std::strerror(errno) << "\n";
            return false;
        }

        if (::rename(tmp_path.c_str(), path.c_str()) != 0)
        {
            std::cerr << "[Router] rename failed from " << tmp_path << " to " << path
                      << ": " << std::strerror(errno) << "\n";
            return false;
        }

        if (!FsyncDir(path))
        {
            std::cerr << "[Router] fsync dir failed for " << path << ": " << std::strerror(errno) << "\n";
            return false;
        }

        return true;
    }

    void MembraneRouter::SetCentroidsFilePath(const std::string &path)
    {
        centroids_path_ = path;
    }

    void MembraneRouter::SetCentroidsLoadMode(CentroidsLoadMode mode)
    {
        centroids_load_mode_ = mode;
    }

} // namespace pomai
