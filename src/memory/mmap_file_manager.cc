/*
 * src/memory/mmap_file_manager.cc
 *
 * Implementation of MmapFileManager declared in mmap_file_manager.h.
 *
 * Notes:
 *  - Uses ftruncate to size files and mmap them. This implementation favors
 *    portability and clarity.
 *  - The mapping length is page-aligned. The requested size is rounded up to
 *    the system page size for mmap.
 *  - The class holds a coarse-grained mutex protecting read/write/append
 *    operations.
 */

#include "src/memory/mmap_file_manager.h"
#include "src/ai/atomic_utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include <cstring>
#include <iostream>
#include <algorithm>
#include <system_error>

namespace pomai::memory
{

    // Helper: round up to page size
    static inline size_t round_up_to_page(size_t v)
    {
        long ps = sysconf(_SC_PAGESIZE);
        if (ps <= 0)
            ps = 4096;
        size_t page = static_cast<size_t>(ps);
        return (v + page - 1) & ~(page - 1);
    }

    MmapFileManager::MmapFileManager() noexcept
        : fd_(-1), base_addr_(nullptr), mapped_size_(0), path_(), append_offset_(0)
    {
    }

    MmapFileManager::MmapFileManager(const std::string &path, size_t size_bytes, bool create)
        : MmapFileManager()
    {
        open(path, size_bytes, create);
    }

    MmapFileManager::~MmapFileManager()
    {
        close();
    }

    // Move constructor: initialize members in same order as declared in header to avoid reorder warnings.
    MmapFileManager::MmapFileManager(MmapFileManager &&other) noexcept
        : fd_(-1), base_addr_(nullptr), mapped_size_(0)
    {
        std::lock_guard<std::mutex> lk(other.io_mu_);
        fd_ = other.fd_;
        base_addr_ = other.base_addr_;
        mapped_size_ = other.mapped_size_;
        path_ = std::move(other.path_);
        append_offset_.store(other.append_offset_.load());

        other.fd_ = -1;
        other.base_addr_ = nullptr;
        other.mapped_size_ = 0;
        other.path_.clear();
        other.append_offset_.store(0);
    }

    MmapFileManager &MmapFileManager::operator=(MmapFileManager &&other) noexcept
    {
        if (this == &other)
            return *this;
        close();
        {
            std::lock_guard<std::mutex> lk(other.io_mu_);
            fd_ = other.fd_;
            base_addr_ = other.base_addr_;
            mapped_size_ = other.mapped_size_;
            path_ = std::move(other.path_);
            append_offset_.store(other.append_offset_.load());

            other.fd_ = -1;
            other.base_addr_ = nullptr;
            other.mapped_size_ = 0;
            other.path_.clear();
            other.append_offset_.store(0);
        }
        return *this;
    }

    bool MmapFileManager::open(const std::string &path, size_t size_bytes, bool create)
    {
        std::lock_guard<std::mutex> lk(io_mu_);

        if (base_addr_ != nullptr || fd_ != -1)
        {
            // already open
            return true;
        }

        path_ = path;
        int flags = O_RDWR;
        if (create)
            flags |= O_CREAT;

        int fd = ::open(path.c_str(), flags, 0600);
        if (fd < 0)
        {
            std::cerr << "MmapFileManager::open: open(" << path << ") failed: " << strerror(errno) << "\n";
            return false;
        }

        // Inspect file size
        struct stat st;
        if (fstat(fd, &st) != 0)
        {
            std::cerr << "MmapFileManager::open: fstat failed: " << strerror(errno) << "\n";
            ::close(fd);
            return false;
        }

        size_t target_size = size_bytes;
        if (!create)
        {
            // if the file exists, prefer its existing size (unless caller explicitly passed a larger size)
            if (static_cast<size_t>(st.st_size) > 0)
                target_size = static_cast<size_t>(st.st_size);
        }

        // If target_size == 0, keep file open but do not mmap
        if (target_size == 0)
        {
            fd_ = fd;
            base_addr_ = nullptr;
            mapped_size_ = 0;
            append_offset_.store(0);
            return true;
        }

        // Round up to page size
        size_t map_size = round_up_to_page(target_size);

        // Ensure file is at least map_size bytes using ftruncate (portable)
        if (ftruncate(fd, static_cast<off_t>(map_size)) != 0)
        {
            std::cerr << "MmapFileManager::open: ftruncate failed: " << strerror(errno) << "\n";
            ::close(fd);
            return false;
        }

        // mmap the file
        void *p = mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED)
        {
            std::cerr << "MmapFileManager::open: mmap failed: " << strerror(errno) << "\n";
            ::close(fd);
            return false;
        }

        fd_ = fd;
        base_addr_ = reinterpret_cast<char *>(p);
        mapped_size_ = map_size;

        // set append offset to current logical file size (st.st_size may be smaller than mapped_size)
        size_t logical_size = static_cast<size_t>(st.st_size);
        if (logical_size > mapped_size_)
            logical_size = mapped_size_;
        append_offset_.store(logical_size);

        return true;
    }

    void MmapFileManager::close()
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        unmap_and_close();
    }

    void MmapFileManager::unmap_and_close()
    {
        if (base_addr_ && mapped_size_ > 0)
        {
            munmap(base_addr_, mapped_size_);
            base_addr_ = nullptr;
        }
        if (fd_ != -1)
        {
            ::close(fd_);
            fd_ = -1;
        }
        mapped_size_ = 0;
        append_offset_.store(0);
        path_.clear();
    }

    bool MmapFileManager::is_valid() const noexcept
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        return (fd_ != -1) && (base_addr_ != nullptr) && (mapped_size_ > 0);
    }

    int MmapFileManager::fd() const noexcept
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        return fd_;
    }

    size_t MmapFileManager::mapped_size() const noexcept
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        return mapped_size_;
    }

    const char *MmapFileManager::base_ptr() const noexcept
    {
        // return pointer without locking (pointer is stable for the life of mapping)
        return base_addr_;
    }

    char *MmapFileManager::writable_base_ptr() noexcept
    {
        return base_addr_;
    }

    bool MmapFileManager::read_at(size_t offset, void *dst, size_t len) const
    {
        if (!dst || len == 0)
            return true;
        std::lock_guard<std::mutex> lk(io_mu_);
        if (!base_addr_ || offset + len > mapped_size_)
            return false;

        // Special-case 8-byte aligned loads: use atomic load helper to avoid torn reads
        if (len == sizeof(uint64_t) && (reinterpret_cast<uintptr_t>(base_addr_ + offset) % alignof(uint64_t) == 0))
        {
            uint64_t v = 0;
            if (!atomic_load_u64_at(offset, v))
                return false;
            std::memcpy(dst, &v, sizeof(v));
            return true;
        }

        std::memcpy(dst, base_addr_ + offset, len);
        return true;
    }

    bool MmapFileManager::write_at(size_t offset, const void *src, size_t len)
    {
        if (!src || len == 0)
            return true;
        std::lock_guard<std::mutex> lk(io_mu_);
        if (!base_addr_ || offset + len > mapped_size_)
            return false;

        // Special-case 8-byte aligned stores: use atomic store helper to avoid torn writes
        if (len == sizeof(uint64_t) && (reinterpret_cast<uintptr_t>(base_addr_ + offset) % alignof(uint64_t) == 0))
        {
            uint64_t val = 0;
            std::memcpy(&val, src, sizeof(val));
            uint64_t *ptr = reinterpret_cast<uint64_t *>(base_addr_ + offset);
            pomai::ai::atomic_utils::atomic_store_u64(ptr, val);
            return true;
        }

        std::memcpy(base_addr_ + offset, src, len);
        return true;
    }

    size_t MmapFileManager::append(const void *src, size_t len)
    {
        if (!src || len == 0)
            return SIZE_MAX;
        std::lock_guard<std::mutex> lk(io_mu_);
        if (!base_addr_)
            return SIZE_MAX;

        size_t off = append_offset_.load();
        if (off + len > mapped_size_)
        {
            // no space left
            return SIZE_MAX;
        }

        // Special-case 8-byte aligned append: perform atomic store to avoid torn readers.
        if (len == sizeof(uint64_t) && (reinterpret_cast<uintptr_t>(base_addr_ + off) % alignof(uint64_t) == 0))
        {
            uint64_t val = 0;
            std::memcpy(&val, src, sizeof(val));
            uint64_t *ptr = reinterpret_cast<uint64_t *>(base_addr_ + off);
            pomai::ai::atomic_utils::atomic_store_u64(ptr, val);
            append_offset_.store(off + len);
            return off;
        }

        std::memcpy(base_addr_ + off, src, len);
        append_offset_.store(off + len);
        return off;
    }

    bool MmapFileManager::flush(size_t offset, size_t len, bool sync)
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        if (!base_addr_ || mapped_size_ == 0)
            return false;

        size_t flush_off = offset;
        size_t flush_len = len;
        if (flush_len == 0)
        {
            flush_off = 0;
            flush_len = mapped_size_;
        }
        if (flush_off + flush_len > mapped_size_)
            return false;

        int flags = sync ? MS_SYNC : MS_ASYNC;
        if (msync(base_addr_ + flush_off, flush_len, flags) != 0)
        {
            std::cerr << "MmapFileManager::flush: msync failed: " << strerror(errno) << "\n";
            return false;
        }
        return true;
    }

    bool MmapFileManager::advise_willneed(size_t offset, size_t len)
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        if (!base_addr_ || mapped_size_ == 0)
            return false;
        if (offset + len > mapped_size_)
            len = (offset >= mapped_size_) ? 0 : (mapped_size_ - offset);
        if (len == 0)
            return false;
        // MADV_WILLNEED is best-effort
        int rc = madvise(base_addr_ + offset, len, MADV_WILLNEED);
        if (rc != 0)
        {
            // Not fatal; return false for visibility
            return false;
        }
        return true;
    }

    bool MmapFileManager::advise_mode(size_t offset, size_t len, AdviseMode mode)
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        if (!base_addr_ || mapped_size_ == 0)
            return false;
        if (offset + len > mapped_size_)
            len = (offset >= mapped_size_) ? 0 : (mapped_size_ - offset);
        if (len == 0)
            return false;

        int how = MADV_NORMAL;
        switch (mode)
        {
        case AdviseMode::NORMAL:
            how = MADV_NORMAL;
            break;
        case AdviseMode::SEQUENTIAL:
            how = MADV_SEQUENTIAL;
            break;
        case AdviseMode::RANDOM:
            how = MADV_RANDOM;
            break;
        default:
            how = MADV_NORMAL;
            break;
        }

        int rc = madvise(base_addr_ + offset, len, how);
        if (rc != 0)
        {
            std::cerr << "MmapFileManager::advise_mode: madvise failed: " << strerror(errno) << "\n";
            return false;
        }
        return true;
    }

    bool MmapFileManager::mlock_range(size_t offset, size_t len)
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        if (!base_addr_ || mapped_size_ == 0 || len == 0)
            return false;
        if (offset + len > mapped_size_)
            return false;

        // Round offset down to page boundary and extend len to page alignment
        long ps = sysconf(_SC_PAGESIZE);
        if (ps <= 0)
            ps = 4096;
        size_t page = static_cast<size_t>(ps);

        size_t page_off = (offset / page) * page;
        size_t end = offset + len;
        size_t page_end = ((end + page - 1) / page) * page;
        size_t mlock_len = page_end - page_off;

        if (mlock(base_addr_ + page_off, mlock_len) != 0)
        {
            std::cerr << "MmapFileManager::mlock_range: mlock failed: " << strerror(errno) << "\n";
            return false;
        }
        return true;
    }

    bool MmapFileManager::munlock_range(size_t offset, size_t len)
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        if (!base_addr_ || mapped_size_ == 0 || len == 0)
            return false;
        if (offset + len > mapped_size_)
            return false;

        long ps = sysconf(_SC_PAGESIZE);
        if (ps <= 0)
            ps = 4096;
        size_t page = static_cast<size_t>(ps);

        size_t page_off = (offset / page) * page;
        size_t end = offset + len;
        size_t page_end = ((end + page - 1) / page) * page;
        size_t mlock_len = page_end - page_off;

        if (munlock(base_addr_ + page_off, mlock_len) != 0)
        {
            std::cerr << "MmapFileManager::munlock_range: munlock failed: " << strerror(errno) << "\n";
            return false;
        }
        return true;
    }

    bool MmapFileManager::atomic_load_u64_at(size_t offset, uint64_t &out) const
    {
        // Validate and obtain pointer under lock, then perform atomic load
        std::lock_guard<std::mutex> lk(io_mu_);
        if (!base_addr_ || mapped_size_ == 0)
            return false;
        if (offset + sizeof(uint64_t) > mapped_size_)
            return false;

        const void *addr = base_addr_ + offset;
        // If address is naturally aligned, use atomic helper (strong ordering chosen in atomic_utils).
        if (reinterpret_cast<uintptr_t>(addr) % alignof(uint64_t) == 0)
        {
            const uint64_t *ptr = reinterpret_cast<const uint64_t *>(addr);
            out = pomai::ai::atomic_utils::atomic_load_u64(ptr);
        }
        else
        {
            // Fallback: memcpy under mutex to avoid torn reads within this process.
            uint64_t tmp = 0;
            std::memcpy(&tmp, addr, sizeof(tmp));
            out = tmp;
        }
        return true;
    }

    bool MmapFileManager::atomic_store_u64_at(size_t offset, uint64_t value)
    {
        // Validate and obtain pointer under lock, then perform atomic store
        std::lock_guard<std::mutex> lk(io_mu_);
        if (!base_addr_ || mapped_size_ == 0)
            return false;
        if (offset + sizeof(uint64_t) > mapped_size_)
            return false;

        void *addr = base_addr_ + offset;
        if (reinterpret_cast<uintptr_t>(addr) % alignof(uint64_t) == 0)
        {
            uint64_t *ptr = reinterpret_cast<uint64_t *>(addr);
            pomai::ai::atomic_utils::atomic_store_u64(ptr, value);
        }
        else
        {
            // Fallback: memcpy under mutex to ensure coherent write in-process.
            std::memcpy(addr, &value, sizeof(value));
        }
        return true;
    }

} // namespace pomai::memory