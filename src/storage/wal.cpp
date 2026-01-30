#include <pomai/storage/wal.h>
#include <pomai/core/posix_compat.h>
#include <pomai/storage/crc64.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#include <filesystem>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <algorithm>
#include <endian.h>
#include <mutex>
#include <limits>
#include <memory>

namespace pomai
{

    void Wal::WaitDurable(Lsn lsn)
    {
        if (lsn == 0)
            return;
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]
                 { return wal_error_.load() || !running_.load() || durable_lsn_ >= lsn; });
        if (wal_error_.load())
            throw std::runtime_error("WAL error: " + wal_error_msg_);
    }

    void Wal::ThrowSys(const std::string &what)
    {
        throw std::runtime_error(what + ": " + std::string(std::strerror(errno)));
    }

    void Wal::EnsureDirExists(const std::string &dir)
    {
        std::error_code ec;
        if (!std::filesystem::create_directories(dir, ec) && ec)
        {
            throw std::runtime_error("failed to create directory " + dir + ": " + ec.message());
        }
    }

    void Wal::FsyncDir(const std::string &dir)
    {
        int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dfd >= 0)
        {
            if (::fsync(dfd) != 0)
            {
                ::close(dfd);
                ThrowSys("fsync(dir) failed");
            }
            ::close(dfd);
        }
        else
        {
            ThrowSys("open(wal_dir) for fsync failed");
        }
    }

    void Wal::WriteAll(int fd, const void *buf, std::size_t n)
    {
        const auto *p = reinterpret_cast<const uint8_t *>(buf);
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
            off += static_cast<std::size_t>(w);
        }
    }

    bool Wal::ReadExact(int fd, void *buf, std::size_t n)
    {
        auto *p = reinterpret_cast<uint8_t *>(buf);
        std::size_t got = 0;
        while (got < n)
        {
            ssize_t r = ::read(fd, p + got, n - got);
            if (r == 0)
                return false;
            if (r < 0)
            {
                if (errno == EINTR)
                    continue;
                ThrowSys("read failed");
            }
            got += static_cast<std::size_t>(r);
        }
        return true;
    }

    Wal::Wal(std::string shard_name, std::string wal_dir, std::size_t dim, WalOptions options)
        : shard_name_(std::move(shard_name)),
          wal_dir_(std::move(wal_dir)),
          dim_(dim),
          options_(options)
    {
        if (dim_ == 0)
            throw std::runtime_error("Wal dim must be > 0");
        wal_path_ = wal_dir_ + "/" + shard_name_ + ".wal";
    }

    Wal::~Wal()
    {
        Stop();
    }

    void Wal::OpenOrCreateForAppend()
    {
        EnsureDirExists(wal_dir_);
        struct stat st{};
        bool created = false;
        if (::stat(wal_path_.c_str(), &st) != 0)
        {
            if (errno != ENOENT)
                ThrowSys("stat WAL failed: " + wal_path_);
            created = true;
        }
        fd_ = ::open(wal_path_.c_str(),
                     O_CREAT | O_APPEND | O_WRONLY | O_CLOEXEC,
                     0644);
        if (fd_ < 0)
            ThrowSys("open WAL failed: " + wal_path_);
        if (created)
            FsyncDir(wal_dir_);
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

        crc64_init();
        OpenOrCreateForAppend();
        uring_enabled_ = false;
#ifdef __linux__
        if (options_.prefer_uring)
        {
            std::string error;
            if (uring_.Init(options_.uring_entries, options_.enable_sqpoll, &error))
                uring_enabled_ = true;
            else
                std::cerr << "[WAL] io_uring disabled: " << error << "\n";
        }
        if (uring_enabled_ && uring_.Entries() < 2)
        {
            std::cerr << "[WAL] io_uring disabled: insufficient SQ entries\n";
            uring_.Shutdown();
            uring_enabled_ = false;
        }
#endif
        writer_th_ = std::thread(&Wal::WriterLoop, this);
        completion_th_ = std::thread(&Wal::CompletionLoop, this);
    }

    void Wal::Stop()
    {
        if (!running_.load())
            return;

        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_requested_ = true;
        }
        cv_.notify_all(); // Đánh thức WriterLoop

#ifdef __linux__
        // CHIÊU THỨC BIG TECH: Gửi lệnh NOP để đánh thức CompletionLoop khỏi WaitCqe
        if (uring_enabled_)
        {
            io_uring_sqe *sqe = uring_.GetSqe();
            if (sqe)
            {
                std::memset(sqe, 0, sizeof(*sqe));
                sqe->opcode = IORING_OP_NOP;
                sqe->user_data = 0; // Tag 0 dành riêng cho Wakeup
                uring_.Submit(0);
            }
        }
#endif

        // Đợi các batch cuối cùng trở nên durable
        try
        {
            WaitDurable(next_lsn_.load() - 1);
        }
        catch (...)
        {
        }

        running_.store(false);
        cv_.notify_all();

        if (writer_th_.joinable())
            writer_th_.join();
        if (completion_th_.joinable())
            completion_th_.join();

        if (fd_ >= 0)
            ::fdatasync(fd_);
#ifdef __linux__
        uring_.Shutdown();
#endif
        CloseFd();
    }

    Lsn Wal::AppendUpserts(const std::vector<UpsertRequest> &batch, bool wait_durable)
    {
        if (batch.empty())
            return 0;

        if (batch.size() > MAX_BATCH_ROWS)
            throw std::runtime_error("WAL append rejected: batch too large; split into smaller batches");

        const uint64_t per_entry_bytes = static_cast<uint64_t>(sizeof(uint64_t) + dim_ * sizeof(float));
        if (per_entry_bytes == 0)
            throw std::runtime_error("WAL append failed: invalid per-entry size");

        const uint64_t count = static_cast<uint64_t>(batch.size());
        if (count > 0 && per_entry_bytes > 0 && count > (UINT64_MAX / per_entry_bytes))
            throw std::runtime_error("WAL append rejected: batch size would overflow payload calculations");

        const std::size_t payload_size =
            sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint16_t) +
            static_cast<std::size_t>(count * per_entry_bytes);

        if (payload_size > MAX_WAL_PAYLOAD_BYTES)
            throw std::runtime_error("WAL append rejected: payload too large; split batch");

        if (payload_size > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()))
            throw std::runtime_error("WAL append rejected: payload larger than 4GB (unsupported)");

        Lsn lsn = next_lsn_.fetch_add(1, std::memory_order_relaxed);

        const uint32_t count_u32 = static_cast<uint32_t>(batch.size());
        const uint16_t dim16 = static_cast<uint16_t>(dim_);

        auto buf = std::make_shared<std::vector<uint8_t>>();
        buf->reserve(payload_size + 24);

        auto push_le = [&](const void *ptr, std::size_t n)
        {
            const uint8_t *p = reinterpret_cast<const uint8_t *>(ptr);
            buf->insert(buf->end(), p, p + n);
        };

        uint64_t lsn_le = (uint64_t)lsn;
        uint32_t count_le = count_u32;
        uint16_t dim_le = dim16;

#if __BYTE_ORDER == __BIG_ENDIAN
        lsn_le = __builtin_bswap64(lsn_le);
        count_le = __builtin_bswap32(count_le);
        dim_le = __builtin_bswap16(dim_le);
#endif

        push_le(&lsn_le, sizeof(lsn_le));
        push_le(&count_le, sizeof(count_le));
        push_le(&dim_le, sizeof(dim_le));

        for (const auto &it : batch)
        {
            if (it.vec.data.size() != dim_)
                throw std::runtime_error("WAL dim mismatch in batch write");

            uint64_t id_le = static_cast<uint64_t>(it.id);
#if __BYTE_ORDER == __BIG_ENDIAN
            id_le = __builtin_bswap64(id_le);
#endif
            push_le(&id_le, sizeof(id_le));
            const uint8_t *vf = reinterpret_cast<const uint8_t *>(it.vec.data.data());
            push_le(vf, dim_ * sizeof(float));
        }

        uint64_t crc = crc64(0, buf->data(), buf->size());

        uint32_t payload_size_u32 = static_cast<uint32_t>(buf->size());
        uint32_t reserved = 0;
        uint64_t crc_le = crc;
        uint64_t magic_le = FOOTER_MAGIC;

#if __BYTE_ORDER == __BIG_ENDIAN
        payload_size_u32 = __builtin_bswap32(payload_size_u32);
        reserved = __builtin_bswap32(reserved);
        crc_le = __builtin_bswap64(crc_le);
        magic_le = __builtin_bswap64(magic_le);
#endif

        push_le(&payload_size_u32, sizeof(payload_size_u32));
        push_le(&reserved, sizeof(reserved));
        push_le(&crc_le, sizeof(crc_le));
        push_le(&magic_le, sizeof(magic_le));

        {
            std::unique_lock<std::mutex> lk(mu_);
            if (!running_.load())
                ThrowSys("WAL not started");
            cv_.wait(lk, [&]
                     { return !running_.load() ||
                              (!stop_requested_ &&
                               pending_writes_.size() < options_.max_pending_records &&
                               pending_bytes_ + buf->size() <= options_.max_queued_bytes); });
            if (wal_error_.load())
                throw std::runtime_error("WAL error: " + wal_error_msg_);
            if (!running_.load() || stop_requested_)
                ThrowSys("WAL not running");
            const std::size_t buf_bytes = buf->size();
            pending_bytes_ += buf_bytes;
            pending_writes_.push_back(Pending{lsn, std::move(buf), buf_bytes});
            cv_.notify_all();
        }

        if (wait_durable)
            WaitDurable(lsn);

        return lsn;
    }

    void Wal::WriterLoop()
    {
        while (running_.load())
        {
            std::deque<Pending> to_write;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&]
                         { return wal_error_.load() || !running_.load() ||
                                  !pending_writes_.empty(); });
                if (wal_error_.load())
                    break;
                if (!running_.load() && pending_writes_.empty())
                    break;
                if (options_.max_inflight_batches > 0)
                {
                    cv_.wait(lk, [&]
                             { return wal_error_.load() || inflight_batches_ < options_.max_inflight_batches ||
                                      !running_.load(); });
                    if (wal_error_.load())
                        break;
                    if (!running_.load() && pending_writes_.empty())
                        break;
                }
                std::size_t batch_bytes = 0;
                while (!pending_writes_.empty())
                {
                    Pending p = std::move(pending_writes_.front());
                    pending_writes_.pop_front();
                    pending_bytes_ -= p.bytes;
                    to_write.push_back(std::move(p));
                    batch_bytes += to_write.back().bytes;
                    if (options_.max_batch_bytes > 0 && batch_bytes >= options_.max_batch_bytes)
                        break;
                    if (options_.max_iovecs > 0 && to_write.size() >= options_.max_iovecs)
                        break;
                }
                cv_.notify_all();
            }
            if (to_write.empty())
                continue;
            if (uring_enabled_)
                SubmitBatchUring(std::move(to_write));
            else
                SubmitBatchSync(std::move(to_write));
        }
    }

    void Wal::SubmitBatchSync(std::deque<Pending> &&batch)
    {
        for (auto &p : batch)
        {
            if (fd_ < 0)
                ThrowSys("WAL writer fd invalid");
            WriteAll(fd_, p.buf->data(), p.buf->size());
            {
                std::lock_guard<std::mutex> lk(mu_);
                written_lsn_ = p.lsn;
                cv_.notify_all();
            }
        }

        if (fd_ >= 0)
        {
            if (::fdatasync(fd_) != 0)
                std::cerr << "[WAL] fdatasync failed in writer: " << std::strerror(errno) << "\n";
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            durable_lsn_ = written_lsn_;
            cv_.notify_all();
        }
    }

    void Wal::SubmitBatchUring(std::deque<Pending> &&batch)
    {
#ifdef __linux__
        if (batch.empty())
            return;

        Lsn start_lsn = batch.front().lsn;
        Lsn end_lsn = batch.back().lsn;
        auto inflight = std::make_unique<InflightBatch>();
        inflight->start_lsn = start_lsn;
        inflight->end_lsn = end_lsn;
        const std::size_t max_iovecs = options_.max_iovecs == 0 ? batch.size() : options_.max_iovecs;
        inflight->iovecs.reserve(std::min(batch.size(), max_iovecs));

        while (!batch.empty() && inflight->iovecs.size() < max_iovecs)
        {
            Pending p = std::move(batch.front());
            batch.pop_front();
            iovec io{};
            io.iov_base = p.buf->data();
            io.iov_len = p.buf->size();
            inflight->iovecs.push_back(io);
            inflight->buffers.push_back(std::move(p.buf));
        }

        io_uring_sqe *sqe = uring_.GetSqe();
        io_uring_sqe *sync_sqe = uring_.GetSqe();
        if (!sqe || !sync_sqe)
            ThrowSys("io_uring SQE exhaustion");
        WalUring::PrepWritev(sqe, fd_, inflight->iovecs.data(), static_cast<unsigned>(inflight->iovecs.size()), -1);
        sqe->flags |= IOSQE_IO_LINK;
        sqe->user_data = reinterpret_cast<__u64>(inflight.get());

        WalUring::PrepFdatasync(sync_sqe, fd_);
        sync_sqe->user_data = reinterpret_cast<__u64>(inflight.get()) | 1ULL;

        {
            std::lock_guard<std::mutex> lk(mu_);
            written_lsn_ = end_lsn;
            inflight_batches_++;
            inflight_.push_back(std::move(inflight));
            cv_.notify_all();
        }

        int ret = uring_.Submit(0);
        if (ret < 0)
        {
            std::lock_guard<std::mutex> lk(mu_);
            wal_error_ = true;
            wal_error_msg_ = std::strerror(errno);
            cv_.notify_all();
        }
#else
        SubmitBatchSync(std::move(batch));
#endif
    }

    void Wal::CompletionLoop()
    {
        while (true)
        {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!running_.load() && inflight_.empty())
                    break;
            }
#ifdef __linux__
            if (!uring_enabled_)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            io_uring_cqe *cqe = nullptr;
            if (uring_.WaitCqe(&cqe) != 0)
                continue;

            uint64_t data = cqe->user_data;
            int res = cqe->res;
            uring_.AdvanceCq(1);

            if (data == 0)
                continue; // NOP wake-up

            bool is_sync = (data & 1ULL) != 0;
            auto *batch = reinterpret_cast<InflightBatch *>(data & ~1ULL);

            {
                std::lock_guard<std::mutex> lk(mu_);
                if (res < 0 && res != -ECANCELED)
                {
                    wal_error_ = true;
                    wal_error_msg_ = std::strerror(-res);
                }
                for (auto &entry : inflight_)
                {
                    if (entry.get() == batch)
                    {
                        if (is_sync)
                            entry->sync_done = true;
                        else
                            entry->write_done = true;
                        break;
                    }
                }
                while (!inflight_.empty() && inflight_.front()->write_done)
                {
                    written_lsn_ = inflight_.front()->end_lsn;
                    if (!inflight_.front()->sync_done)
                        break;
                    durable_lsn_ = inflight_.front()->end_lsn;
                    inflight_.pop_front();
                    if (inflight_batches_ > 0)
                        inflight_batches_--;
                }
                cv_.notify_all();
            }
#else
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
        }
    }

    Lsn Wal::WrittenLsn() const
    {
        std::lock_guard<std::mutex> lk(mu_);
        return written_lsn_;
    }

    Lsn Wal::DurableLsn() const
    {
        std::lock_guard<std::mutex> lk(mu_);
        return durable_lsn_;
    }

    void Wal::TruncateToZero()
    {
        std::lock_guard<std::mutex> lk(mu_);

        if (fd_ < 0)
            ThrowSys("truncate wal: wal fd invalid");

        if (ftruncate(fd_, 0) != 0)
            ThrowSys("ftruncate failed during checkpoint");

        if (::fdatasync(fd_) != 0)
            ThrowSys("fdatasync failed after truncate");
        FsyncDir(wal_dir_);

        next_lsn_.store(1, std::memory_order_relaxed);
        written_lsn_ = 0;
        durable_lsn_ = 0;
        pending_writes_.clear();
        pending_bytes_ = 0;
        cv_.notify_all();
    }

} // namespace pomai
