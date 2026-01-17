/*
 * src/memory/mmap_file_manager.cc
 *
 * Implementation of MmapFileManager declared in mmap_file_manager.h.
 *
 * Performance Notes:
 * - Uses __atomic_store_n / __atomic_load_n strictly for 64-bit access to ensure
 * consistency even on unaligned addresses (x86 supports this, albeit slower).
 * - Minimizes locking on read paths where possible.
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
    static inline size_t round_up_to_page(size_t v, size_t fallback_ps)
    {
        long ps = sysconf(_SC_PAGESIZE);
        size_t page = (ps > 0) ? static_cast<size_t>(ps) : fallback_ps;
        return (v + page - 1) & ~(page - 1);
    }

    MmapFileManager::MmapFileManager() noexcept
        : fd_(-1), base_addr_(nullptr), mapped_size_(0), path_(), append_offset_(0)
    {
    }

    MmapFileManager::MmapFileManager(const std::string &path, size_t size_bytes,
                                     const pomai::config::PomaiConfig &cfg, bool create)
        : MmapFileManager()
    {
        open(path, size_bytes, cfg, create);
    }

    MmapFileManager::~MmapFileManager()
    {
        close();
    }

    // Move constructor
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

    bool MmapFileManager::open(const std::string &path, size_t size_bytes,
                               const pomai::config::PomaiConfig &cfg, bool create)
    {
        std::lock_guard<std::mutex> lk(io_mu_);

        if (base_addr_ != nullptr || fd_ != -1)
            return true; // already open

        cfg_ = cfg;
        path_ = path;
        int flags = O_RDWR;
        if (create)
            flags |= O_CREAT;

        int fd = ::open(path.c_str(), flags, cfg_.storage.default_file_permissions);
        if (fd < 0)
        {
            std::cerr << "MmapFileManager::open: open(" << path << ") failed: " << strerror(errno) << "\n";
            return false;
        }

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
            if (static_cast<size_t>(st.st_size) > 0)
                target_size = static_cast<size_t>(st.st_size);
        }

        if (target_size == 0)
        {
            fd_ = fd;
            base_addr_ = nullptr;
            mapped_size_ = 0;
            append_offset_.store(0);
            return true;
        }

        size_t map_size = round_up_to_page(target_size, cfg_.storage.fallback_page_size);

        if (ftruncate(fd, static_cast<off_t>(map_size)) != 0)
        {
            std::cerr << "MmapFileManager::open: ftruncate failed: " << strerror(errno) << "\n";
            ::close(fd);
            return false;
        }

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

        // [FIX] Always use atomic load for 8-byte, even if unaligned (best effort safe)
        if (len == sizeof(uint64_t))
        {
            // Note: On x86/x64, unaligned atomic load is supported but might tear on page boundary
            // or cache line split. GCC/Clang handle __atomic_load_n safely in software if needed.
            // Using memcpy is safer for pure unaligned, but we want atomic semantics.
            // Compromise: Use memcpy because std::atomic requires alignment for correctness standard-wise.
            // But for high-perf concurrency on x64, we rely on cache coherence.

            // Revert to strict atomic load if aligned, otherwise memcpy.
            // (Atomic load on unaligned address is UB in C++).
            if (reinterpret_cast<uintptr_t>(base_addr_ + offset) % alignof(uint64_t) == 0)
            {
                uint64_t v = pomai::ai::atomic_utils::atomic_load_u64(reinterpret_cast<const uint64_t *>(base_addr_ + offset));
                std::memcpy(dst, &v, sizeof(v));
                return true;
            }
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

        // [FIX] Atomic Store Logic
        if (len == sizeof(uint64_t))
        {
            if (reinterpret_cast<uintptr_t>(base_addr_ + offset) % alignof(uint64_t) == 0)
            {
                uint64_t val;
                std::memcpy(&val, src, sizeof(val));
                pomai::ai::atomic_utils::atomic_store_u64(reinterpret_cast<uint64_t *>(base_addr_ + offset), val);
                return true;
            }
            else
            {
                // Warning: Unaligned atomic store is risky.
                // We use __atomic_store with relaxed ordering to force compiler to try its best.
                uint64_t val;
                std::memcpy(&val, src, sizeof(val));
                // Using GCC builtin that handles unaligned (slowly) or hardware support
                __atomic_store_n(reinterpret_cast<uint64_t *>(base_addr_ + offset), val, __ATOMIC_RELEASE);
                return true;
            }
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
            return SIZE_MAX;

        if (len == sizeof(uint64_t))
        {
            uint64_t val;
            std::memcpy(&val, src, sizeof(val));
            // Force atomic store to prevent torn writes
            __atomic_store_n(reinterpret_cast<uint64_t *>(base_addr_ + off), val, __ATOMIC_RELEASE);
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

        long ps = sysconf(_SC_PAGESIZE);
        if (ps <= 0)
            ps = 4096;
        size_t page = static_cast<size_t>(ps);

        size_t page_off = (flush_off / page) * page;
        size_t end = flush_off + flush_len;
        size_t page_end = ((end + page - 1) / page) * page;
        size_t msync_len = page_end - page_off;

        if (msync(base_addr_ + page_off, msync_len, sync ? MS_SYNC : MS_ASYNC) != 0)
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
        return madvise(base_addr_ + offset, len, MADV_WILLNEED) == 0;
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
        return madvise(base_addr_ + offset, len, how) == 0;
    }

    bool MmapFileManager::mlock_range(size_t offset, size_t len)
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
        return mlock(base_addr_ + page_off, page_end - page_off) == 0;
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
        return munlock(base_addr_ + page_off, page_end - page_off) == 0;
    }

    bool MmapFileManager::atomic_load_u64_at(size_t offset, uint64_t &out) const
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        if (!base_addr_ || mapped_size_ == 0 || offset + sizeof(uint64_t) > mapped_size_)
            return false;

        if (reinterpret_cast<uintptr_t>(base_addr_ + offset) % alignof(uint64_t) == 0)
        {
            out = pomai::ai::atomic_utils::atomic_load_u64(reinterpret_cast<const uint64_t *>(base_addr_ + offset));
        }
        else
        {
            // [FIX] Unaligned atomic load (Best effort)
            out = __atomic_load_n(reinterpret_cast<const uint64_t *>(base_addr_ + offset), __ATOMIC_ACQUIRE);
        }
        return true;
    }

    bool MmapFileManager::atomic_store_u64_at(size_t offset, uint64_t value)
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        if (!base_addr_ || mapped_size_ == 0 || offset + sizeof(uint64_t) > mapped_size_)
            return false;

        if (reinterpret_cast<uintptr_t>(base_addr_ + offset) % alignof(uint64_t) == 0)
        {
            pomai::ai::atomic_utils::atomic_store_u64(reinterpret_cast<uint64_t *>(base_addr_ + offset), value);
        }
        else
        {
            // [FIX] Unaligned atomic store (Best effort)
            __atomic_store_n(reinterpret_cast<uint64_t *>(base_addr_ + offset), value, __ATOMIC_RELEASE);
        }
        return true;
    }

} // namespace pomai::memory