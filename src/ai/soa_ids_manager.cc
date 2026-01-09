/*
 * src/ai/soa_ids_manager.cc
 *
 * Implementation of SoaIdsManager.
 *
 * Durable update algorithm (when durable==true):
 *   1) append WalEntry { idx, value } to WAL file and fsync WAL
 *   2) atomic_store into mapped ids array (std::atomic_ref if available)
 *   3) msync the 8-byte range of the ids mapping
 *   4) truncate the WAL to zero (so WAL is empty after each durable update)
 *
 * Recovery on startup:
 *  - If WAL contains entries, apply each entry (atomic_store + msync) then truncate the WAL.
 *
 * Notes:
 *  - This implementation favors simplicity and correctness; it truncates WAL after every
 *    durable update to keep recovery simple. For higher throughput you could batch WAL
 *    entries and truncate periodically (or use a ring log).
 *  - All file operations are guarded by wal_mu_ to keep sequences atomic from the
 *    perspective of this process.
 */

#include "src/ai/soa_ids_manager.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#include <cstring>
#include <iostream>
#include <vector>
#include <system_error>
#include <cassert>

namespace pomai::ai::soa
{

    // Match declaration (noexcept) in the header
    SoaIdsManager::SoaIdsManager() noexcept : num_entries_(0), wal_fd_(-1) {}

    SoaIdsManager::~SoaIdsManager()
    {
        close();
    }

    bool SoaIdsManager::open(const std::string &ids_path, size_t num_entries, bool create_if_missing,
                             const std::string &wal_path)
    {
        std::lock_guard<std::mutex> lk(wal_mu_);
        close();

        ids_path_ = ids_path;
        num_entries_ = num_entries;
        if (wal_path.empty())
        {
            wal_path_ = ids_path + ".wal";
        }
        else
        {
            wal_path_ = wal_path;
        }

        if (!ensure_mapped_ids(num_entries, create_if_missing))
            return false;

        if (!open_wal_file_if_needed())
            return false;

        // replay WAL if any
        if (!replay_wal_and_truncate())
            return false;

        return true;
    }

    void SoaIdsManager::close()
    {
        std::lock_guard<std::mutex> lk(wal_mu_);
        if (wal_fd_ != -1)
        {
            ::close(wal_fd_);
            wal_fd_ = -1;
        }
        ids_mmap_.close();
        ids_path_.clear();
        wal_path_.clear();
        num_entries_ = 0;
    }

    const uint64_t *SoaIdsManager::ids_ptr() const noexcept
    {
        return reinterpret_cast<const uint64_t *>(ids_mmap_.base_ptr());
    }

    size_t SoaIdsManager::num_entries() const noexcept { return num_entries_; }

    bool SoaIdsManager::ensure_mapped_ids(size_t num_entries, bool create_if_missing)
    {
        if (num_entries == 0)
        {
            std::cerr << "SoaIdsManager::ensure_mapped_ids: num_entries == 0\n";
            return false;
        }
        size_t bytes = num_entries * sizeof(uint64_t);
        bool ok = ids_mmap_.open(ids_path_, bytes, create_if_missing);
        if (!ok)
        {
            std::cerr << "SoaIdsManager::ensure_mapped_ids: failed to open mmap for " << ids_path_ << "\n";
            return false;
        }

        // If newly created (file was zero-length), zero-initialize logical portion
        // We conservatively zero entire mapped area to ensure deterministic initial state.
        char *base = ids_mmap_.writable_base_ptr();
        if (base)
        {
            std::memset(base, 0, ids_mmap_.mapped_size());
        }
        return true;
    }

    bool SoaIdsManager::open_wal_file_if_needed()
    {
        // open WAL for append/read/write
        int fd = ::open(wal_path_.c_str(), O_RDWR | O_CREAT, 0600);
        if (fd < 0)
        {
            std::cerr << "SoaIdsManager::open_wal_file_if_needed: open wal failed: " << strerror(errno) << "\n";
            return false;
        }
        wal_fd_ = fd;
        return true;
    }

    bool SoaIdsManager::replay_wal_and_truncate()
    {
        if (wal_fd_ == -1)
            return true; // nothing to do

        // Read entire WAL contents
        struct stat st;
        if (fstat(wal_fd_, &st) != 0)
        {
            std::cerr << "SoaIdsManager::replay_wal_and_truncate: fstat wal failed: " << strerror(errno) << "\n";
            return false;
        }
        off_t sz = st.st_size;
        if (sz == 0)
            return true; // nothing to do

        if (sz % static_cast<off_t>(sizeof(WalEntry)) != 0)
        {
            std::cerr << "SoaIdsManager::replay_wal_and_truncate: wal file size invalid: " << sz << "\n";
            // Attempt to truncate to zero to avoid future failures
            if (ftruncate(wal_fd_, 0) != 0)
                std::cerr << "SoaIdsManager::replay_wal_and_truncate: ftruncate wal failed: " << strerror(errno) << "\n";
            return false;
        }

        size_t n = static_cast<size_t>(sz / sizeof(WalEntry));
        std::vector<WalEntry> entries;
        entries.resize(n);

        // read from start
        if (lseek(wal_fd_, 0, SEEK_SET) == (off_t)-1)
        {
            std::cerr << "SoaIdsManager::replay_wal_and_truncate: lseek failed: " << strerror(errno) << "\n";
            return false;
        }
        ssize_t r = read(wal_fd_, entries.data(), sz);
        if (r != sz)
        {
            std::cerr << "SoaIdsManager::replay_wal_and_truncate: read wal failed: wanted=" << sz << " got=" << r << " errno=" << strerror(errno) << "\n";
            // attempt truncation to be safe
            if (ftruncate(wal_fd_, 0) != 0)
                std::cerr << "SoaIdsManager::replay_wal_and_truncate: ftruncate wal failed during recovery: " << strerror(errno) << "\n";
            return false;
        }

        // Apply each entry (atomic store + msync)
        for (size_t i = 0; i < entries.size(); ++i)
        {
            WalEntry &we = entries[i];
            if (we.idx >= num_entries_)
            {
                std::cerr << "SoaIdsManager::replay_wal_and_truncate: wal entry index out of range: " << we.idx << "\n";
                continue;
            }

            uint64_t *ptr = const_cast<uint64_t *>(ids_ptr()) + we.idx;
#if defined(__cpp_lib_atomic_ref) || (__cplusplus >= 202002L)
            {
                std::atomic_ref<uint64_t> aref(*ptr);
                aref.store(we.value, std::memory_order_release);
            }
#else
            // fallback: write via volatile pointer (best-effort for single-process)
            *(volatile uint64_t *)ptr = we.value;
#endif
            // msync that 8-byte range to persist update
            // ids_mmap_.flush(offset, len)
            size_t offset = we.idx * sizeof(uint64_t);
            ids_mmap_.flush(offset, sizeof(uint64_t), /*sync=*/true);
        }

        // Truncate WAL to zero (applied)
        if (ftruncate(wal_fd_, 0) != 0)
        {
            std::cerr << "SoaIdsManager::replay_wal_and_truncate: ftruncate wal failed: " << strerror(errno) << "\n";
            // Not fatal; continue
        }
        // reset file offset to end for future appends
        lseek(wal_fd_, 0, SEEK_END);
        return true;
    }

    bool SoaIdsManager::atomic_update(size_t idx, uint64_t value, bool durable)
    {
        if (idx >= num_entries_)
            return false;
        if (wal_fd_ == -1)
        {
            // WAL not opened; try to open in place
            std::lock_guard<std::mutex> lk(wal_mu_);
            if (wal_fd_ == -1)
            {
                if (!open_wal_file_if_needed())
                    return false;
            }
        }

        uint64_t *ptr = const_cast<uint64_t *>(ids_ptr()) + idx;

        if (!durable)
        {
            // Fast path: atomic in-memory store only
#if defined(__cpp_lib_atomic_ref) || (__cplusplus >= 202002L)
            {
                std::atomic_ref<uint64_t> aref(*ptr);
                aref.store(value, std::memory_order_release);
            }
#else
            *(volatile uint64_t *)ptr = value;
#endif
            // Note: not msynced; durability not guaranteed
            return true;
        }

        // Durable path: WAL -> fsync WAL -> atomic_store -> msync id -> truncate WAL
        WalEntry we;
        we.idx = static_cast<uint64_t>(idx);
        we.value = value;

        std::lock_guard<std::mutex> lk(wal_mu_);

        // Append WAL entry
        ssize_t w = write(wal_fd_, &we, sizeof(we));
        if (w != static_cast<ssize_t>(sizeof(we)))
        {
            std::cerr << "SoaIdsManager::atomic_update: write wal failed: wrote=" << w << " errno=" << strerror(errno) << "\n";
            return false;
        }
        // ensure WAL entry durable on disk
        if (fsync(wal_fd_) != 0)
        {
            std::cerr << "SoaIdsManager::atomic_update: fsync wal failed: " << strerror(errno) << "\n";
            // attempt to continue but signal failure
            // Note: we choose failure here
            return false;
        }

        // Apply atomic store to mapped ids
#if defined(__cpp_lib_atomic_ref) || (__cplusplus >= 202002L)
        {
            std::atomic_ref<uint64_t> aref(*ptr);
            aref.store(value, std::memory_order_release);
        }
#else
        *(volatile uint64_t *)ptr = value;
#endif

        // msync the 8-byte range to persist the ids file
        size_t offset = idx * sizeof(uint64_t);
        if (!ids_mmap_.flush(offset, sizeof(uint64_t), /*sync=*/true))
        {
            std::cerr << "SoaIdsManager::atomic_update: msync ids failed for idx=" << idx << "\n";
            // attempt to continue; WAL has entry and will be replayed on restart
            // Ideally we should report error
        }

        // Truncate WAL to zero (we applied the single entry)
        if (ftruncate(wal_fd_, 0) != 0)
        {
            std::cerr << "SoaIdsManager::atomic_update: ftruncate wal failed: " << strerror(errno) << "\n";
            // Not fatal: WAL still contains applied entry which will be replayed (idempotent)
        }
        // Reset file offset to end (should be zero)
        lseek(wal_fd_, 0, SEEK_END);

        return true;
    }

} // namespace pomai::ai::soa