#pragma once

#include <cstdint>
#include <string>
#include <atomic>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>

namespace pomai::core {

/**
 * @brief WAL-Index (Shared Memory).
 * Distilled from SQLite's WAL-SHM coordination.
 * 
 * Allows multiple processes to coordinate reads and writes to the WAL
 * without a central daemon.
 */
class WALIndex {
public:
    static constexpr size_t SHM_SIZE = 32 * 1024; // 32KB index

    struct Header {
        std::atomic<uint64_t> version;
        std::atomic<uint64_t> last_committed_offset;
        std::atomic<uint32_t> n_frames;
        std::atomic<uint32_t> cksum_1;
        std::atomic<uint32_t> cksum_2;
    };

    WALIndex(const std::string& db_path) {
        shm_name_ = "/" + db_path + "-shm";
        // Sanitize name for shm_open (must start with / and have no other /)
        for (size_t i = 1; i < shm_name_.size(); ++i) {
            if (shm_name_[i] == '/') shm_name_[i] = '_';
        }

        fd_ = shm_open(shm_name_.c_str(), O_RDWR | O_CREAT, 0666);
        if (fd_ < 0) throw std::runtime_error("Failed to open SHM: " + shm_name_);

        if (ftruncate(fd_, SHM_SIZE) < 0) {
            close(fd_);
            throw std::runtime_error("Failed to truncate SHM");
        }

        ptr_ = mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (ptr_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("Failed to mmap SHM");
        }

        header_ = reinterpret_cast<Header*>(ptr_);
    }

    ~WALIndex() {
        if (ptr_ != MAP_FAILED) munmap(ptr_, SHM_SIZE);
        if (fd_ >= 0) close(fd_);
    }

    Header* header() { return header_; }

    void MarkCommitted(uint64_t offset, uint32_t n_frames) {
        header_->last_committed_offset.store(offset, std::memory_order_release);
        header_->n_frames.store(n_frames, std::memory_order_release);
    }

private:
    std::string shm_name_;
    int fd_;
    void* ptr_;
    Header* header_;
};

} // namespace pomai::core
