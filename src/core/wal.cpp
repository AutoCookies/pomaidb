#include "wal.h"
#include "seed.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <stdexcept>

namespace pomai
{

    static void ThrowSys(const std::string &what)
    {
        throw std::runtime_error(what + ": " + std::string(std::strerror(errno)));
    }

    static void EnsureDirExists(const std::string &dir)
    {
        struct stat st;
        if (::stat(dir.c_str(), &st) == 0)
        {
            if (!S_ISDIR(st.st_mode))
                throw std::runtime_error("WAL dir not a directory: " + dir);
            return;
        }
        if (::mkdir(dir.c_str(), 0755) != 0)
            ThrowSys("mkdir failed for " + dir);
    }

    static bool FileExists(const std::string &path)
    {
        struct stat st;
        return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
    }

    static void WriteAll(int fd, const void *buf, std::size_t n)
    {
        const auto *p = reinterpret_cast<const std::uint8_t *>(buf);
        std::size_t off = 0;
        while (off < n)
        {
            ssize_t w = ::write(fd, p + off, n - off);
            if (w < 0)
            {
                if (errno == EINTR)
                    continue;
                ThrowSys("write failed");
            }
            off += (std::size_t)w;
        }
    }

    static bool ReadExact(int fd, void *buf, std::size_t n)
    {
        auto *p = reinterpret_cast<std::uint8_t *>(buf);
        std::size_t got = 0;
        while (got < n)
        {
            ssize_t r = ::read(fd, p + got, n - got);
            if (r == 0)
                return false; // EOF
            if (r < 0)
            {
                if (errno == EINTR)
                    continue;
                ThrowSys("read failed");
            }
            got += (std::size_t)r;
        }
        return true;
    }

    Wal::Wal(std::string shard_name, std::string wal_dir, std::size_t dim)
        : shard_name_(std::move(shard_name)),
          wal_dir_(std::move(wal_dir)),
          dim_(dim)
    {
        if (dim_ == 0)
            throw std::runtime_error("Wal dim must be > 0");
        wal_path_ = wal_dir_ + "/" + shard_name_ + ".wal";
    }

    Wal::~Wal() { Stop(); }

    void Wal::OpenOrCreateForAppend()
    {
        EnsureDirExists(wal_dir_);

        fd_ = ::open(wal_path_.c_str(),
                     O_CREAT | O_APPEND | O_WRONLY | O_CLOEXEC,
                     0644);
        if (fd_ < 0)
            ThrowSys("open WAL failed: " + wal_path_);
    }

    void Wal::CloseFd()
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
    }

    void Wal::Start()
    {
        if (running_.exchange(true))
            return;
        OpenOrCreateForAppend();
        fsync_th_ = std::thread(&Wal::FsyncLoop, this);
    }

    void Wal::Stop()
    {
        if (!running_.exchange(false))
            return;

        cv_.notify_all();
        if (fsync_th_.joinable())
            fsync_th_.join();

        // final flush
        if (fd_ >= 0)
            ::fdatasync(fd_);
        CloseFd();
    }

    Lsn Wal::AppendUpserts(const std::vector<UpsertRequest> &batch)
    {
        if (batch.empty())
            return 0;

        Lsn lsn = next_lsn_.fetch_add(1, std::memory_order_relaxed);

        std::lock_guard<std::mutex> lk(mu_);
        if (fd_ < 0)
            throw std::runtime_error("WAL not started");

        WriteRecordLocked(lsn, batch);
        written_lsn_ = lsn;
        cv_.notify_all();
        return lsn;
    }

    void Wal::WaitDurable(Lsn lsn)
    {
        if (lsn == 0)
            return;
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]
                 { return !running_.load() || durable_lsn_ >= lsn; });
    }

    void Wal::WriteRecordLocked(Lsn lsn, const std::vector<UpsertRequest> &batch)
    {
        for (const auto &it : batch)
        {
            if (it.vec.data.size() != dim_)
                throw std::runtime_error("WAL dim mismatch in batch");
        }

        const std::uint32_t count = (std::uint32_t)batch.size();
        const std::uint16_t dim16 = (std::uint16_t)dim_;

        // body bytes (excluding rec_bytes field)
        const std::uint32_t body_bytes =
            (std::uint32_t)(sizeof(std::uint64_t) + sizeof(std::uint32_t) + sizeof(std::uint16_t) +
                            count * (sizeof(std::uint64_t) + (std::uint32_t)dim_ * sizeof(float)));

        // header: rec_bytes + lsn + count + dim
        WriteAll(fd_, &body_bytes, sizeof(body_bytes));
        const std::uint64_t lsn64 = (std::uint64_t)lsn;
        WriteAll(fd_, &lsn64, sizeof(lsn64));
        WriteAll(fd_, &count, sizeof(count));
        WriteAll(fd_, &dim16, sizeof(dim16));

        // items
        for (const auto &it : batch)
        {
            const std::uint64_t id64 = (std::uint64_t)it.id;
            WriteAll(fd_, &id64, sizeof(id64));
            WriteAll(fd_, it.vec.data.data(), dim_ * sizeof(float));
        }
    }

    void Wal::FsyncLoop()
    {
        using namespace std::chrono_literals;
        const auto interval = 5ms;

        std::unique_lock<std::mutex> lk(mu_);
        while (running_.load())
        {
            cv_.wait_for(lk, interval, [&]
                         { return !running_.load() || durable_lsn_ < written_lsn_; });

            if (!running_.load())
                break;

            if (durable_lsn_ < written_lsn_)
            {
                const Lsn target = written_lsn_;
                lk.unlock();
                if (fd_ >= 0)
                    ::fdatasync(fd_);
                lk.lock();
                durable_lsn_ = target;
                cv_.notify_all();
            }
        }
    }

    // ✅ Replay WAL into Seed
    Lsn Wal::ReplayToSeed(Seed &seed)
    {
        if (seed.Dim() != dim_)
            throw std::runtime_error("Replay seed dim mismatch");

        if (!FileExists(wal_path_))
        {
            return 0;
        }

        int rfd = ::open(wal_path_.c_str(), O_RDONLY | O_CLOEXEC);
        if (rfd < 0)
            ThrowSys("open WAL for replay failed: " + wal_path_);

        Lsn last_lsn = 0;

        while (true)
        {
            std::uint32_t body_bytes = 0;
            if (!ReadExact(rfd, &body_bytes, sizeof(body_bytes)))
                break; // EOF clean

            std::uint64_t lsn64 = 0;
            std::uint32_t count = 0;
            std::uint16_t dim16 = 0;

            if (!ReadExact(rfd, &lsn64, sizeof(lsn64)))
                break;
            if (!ReadExact(rfd, &count, sizeof(count)))
                break;
            if (!ReadExact(rfd, &dim16, sizeof(dim16)))
                break;

            if ((std::size_t)dim16 != dim_)
            {
                ::close(rfd);
                throw std::runtime_error("WAL replay dim mismatch in file: " + wal_path_);
            }

            // read items
            std::vector<UpsertRequest> batch;
            batch.reserve(count);

            std::vector<float> tmp(dim_);
            for (std::uint32_t i = 0; i < count; ++i)
            {
                std::uint64_t id64 = 0;
                if (!ReadExact(rfd, &id64, sizeof(id64)))
                {
                    ::close(rfd);
                    throw std::runtime_error("WAL truncated");
                }
                if (!ReadExact(rfd, tmp.data(), dim_ * sizeof(float)))
                {
                    ::close(rfd);
                    throw std::runtime_error("WAL truncated");
                }

                UpsertRequest r;
                r.id = (Id)id64;
                r.vec.data = tmp; // copy dim floats
                batch.push_back(std::move(r));
            }

            seed.ApplyUpserts(batch);
            last_lsn = (Lsn)lsn64;

            // Keep next_lsn in sync so new appends don’t reuse old LSN.
            Lsn want = last_lsn + 1;
            Lsn cur = next_lsn_.load(std::memory_order_relaxed);
            if (want > cur)
                next_lsn_.store(want, std::memory_order_relaxed);
        }

        ::close(rfd);
        return last_lsn;
    }

} // namespace pomai
