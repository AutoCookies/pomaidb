#include "wal.h"
#include "seed.h"
#include "crc64.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <iostream>

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

        // O_APPEND đảm bảo atomicity ở mức OS khi ghi vào cuối file
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

        // Khởi tạo bảng CRC64 ngay khi start
        crc64_init();

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

        if (fd_ >= 0)
            ::fdatasync(fd_);
        CloseFd();
    }

    Lsn Wal::AppendUpserts(const std::vector<UpsertRequest> &batch)
    {
        if (batch.empty())
            return 0;

        // Lấy LSN tiếp theo
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

    // Ghi với Checksum CRC64
    void Wal::WriteRecordLocked(Lsn lsn, const std::vector<UpsertRequest> &batch)
    {
        // 1. Tính toán kích thước Payload
        // Payload Structure:
        // [LSN (8)] + [Count (4)] + [Dim (2)] + N * ( [ID (8)] + [VectorData (Dim*4)] )

        const std::uint32_t count = (std::uint32_t)batch.size();
        const std::uint16_t dim16 = (std::uint16_t)dim_;

        const std::size_t payload_size =
            sizeof(uint64_t) + // lsn
            sizeof(uint32_t) + // count
            sizeof(uint16_t) + // dim
            count * (sizeof(uint64_t) + dim_ * sizeof(float));

        // 2. Serialize Payload vào Buffer tạm (Memory)
        // Việc này tốn thêm 1 chút RAM nhưng đảm bảo tính toàn vẹn khi tính CRC
        std::vector<uint8_t> payload_buf;
        payload_buf.reserve(payload_size);

        // Helper để push data vào vector
        auto push_val = [&](const void *ptr, size_t size)
        {
            const uint8_t *p = static_cast<const uint8_t *>(ptr);
            payload_buf.insert(payload_buf.end(), p, p + size);
        };

        uint64_t lsn64 = (uint64_t)lsn;
        push_val(&lsn64, sizeof(lsn64));
        push_val(&count, sizeof(count));
        push_val(&dim16, sizeof(dim16));

        for (const auto &it : batch)
        {
            if (it.vec.data.size() != dim_)
                throw std::runtime_error("WAL dim mismatch in batch write");

            uint64_t id64 = (uint64_t)it.id;
            push_val(&id64, sizeof(id64));
            push_val(it.vec.data.data(), dim_ * sizeof(float));
        }

        // 3. Tính CRC64 của Payload
        uint64_t checksum = crc64(0, payload_buf.data(), payload_buf.size());

        // 4. Chuẩn bị Header và Ghi xuống đĩa
        WalRecordHeader header;
        header.payload_size = (uint32_t)payload_size;
        header.checksum = checksum;

        // Ghi Header
        WriteAll(fd_, &header, sizeof(header));
        // Ghi Payload
        WriteAll(fd_, payload_buf.data(), payload_buf.size());
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

    // Replay với tính năng Verify Checksum
    Lsn Wal::ReplayToSeed(Seed &seed)
    {
        if (seed.Dim() != dim_)
            throw std::runtime_error("Replay seed dim mismatch");

        if (!FileExists(wal_path_))
            return 0;

        int rfd = ::open(wal_path_.c_str(), O_RDONLY | O_CLOEXEC);
        if (rfd < 0)
            ThrowSys("open WAL for replay failed: " + wal_path_);

        Lsn last_lsn = 0;
        size_t total_replayed = 0;

        while (true)
        {
            // 1. Đọc Header
            WalRecordHeader header;
            if (!ReadExact(rfd, &header, sizeof(header)))
            {
                // EOF sạch sẽ (hoặc file rỗng)
                break;
            }

            // Sanity Check: Payload size không được quá vô lý (vd > 64MB)
            if (header.payload_size > 64 * 1024 * 1024)
            {
                std::cerr << "[WAL] Error: Record size too large (" << header.payload_size << "). Corruption detected. Stopping replay.\n";
                break;
            }

            // 2. Đọc Payload vào buffer
            std::vector<uint8_t> payload_buf(header.payload_size);
            if (!ReadExact(rfd, payload_buf.data(), header.payload_size))
            {
                std::cerr << "[WAL] Error: Unexpected EOF reading payload. Truncated record detected. Stopping replay.\n";
                break;
            }

            // 3. Verify Checksum
            uint64_t calculated_crc = crc64(0, payload_buf.data(), header.payload_size);
            if (calculated_crc != header.checksum)
            {
                std::cerr << "[WAL] Error: CRC Checksum Failed! Disk corrupted or partial write. Stopping replay.\n";
                // Dừng replay tại đây để bảo vệ tính toàn vẹn dữ liệu
                break;
            }

            // 4. Deserialize (Parse dữ liệu từ buffer)
            const uint8_t *ptr = payload_buf.data();

            uint64_t lsn64;
            uint32_t count;
            uint16_t dim16;

            std::memcpy(&lsn64, ptr, sizeof(lsn64));
            ptr += sizeof(lsn64);
            std::memcpy(&count, ptr, sizeof(count));
            ptr += sizeof(count);
            std::memcpy(&dim16, ptr, sizeof(dim16));
            ptr += sizeof(dim16);

            if ((size_t)dim16 != dim_)
            {
                std::cerr << "[WAL] Error: Dimension mismatch in record. Stopping replay.\n";
                break;
            }

            std::vector<UpsertRequest> batch;
            batch.reserve(count);
            std::vector<float> tmp_vec(dim_);

            for (uint32_t i = 0; i < count; ++i)
            {
                UpsertRequest r;
                uint64_t id64;

                std::memcpy(&id64, ptr, sizeof(id64));
                ptr += sizeof(id64);
                std::memcpy(tmp_vec.data(), ptr, dim_ * sizeof(float));
                ptr += dim_ * sizeof(float);

                r.id = (Id)id64;
                r.vec.data = tmp_vec;
                batch.push_back(std::move(r));
            }

            // 5. Apply vào Memory
            seed.ApplyUpserts(batch);
            last_lsn = (Lsn)lsn64;
            total_replayed++;

            // Đồng bộ bộ đếm LSN
            Lsn want = last_lsn + 1;
            Lsn cur = next_lsn_.load(std::memory_order_relaxed);
            if (want > cur)
                next_lsn_.store(want, std::memory_order_relaxed);
        }

        std::cout << "[WAL] Replay finished. Records: " << total_replayed << ", Last LSN: " << last_lsn << "\n";
        ::close(rfd);
        return last_lsn;
    }

} // namespace pomai