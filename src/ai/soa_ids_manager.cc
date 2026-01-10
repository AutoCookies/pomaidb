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
#include <memory> // <-- added for std::make_unique used in enable_wal_manager

#include "src/ai/atomic_utils.h"

namespace pomai::ai::soa
{

    // Match declaration (noexcept) in the header
    SoaIdsManager::SoaIdsManager() noexcept = default;

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

        // If a structured WAL manager was already configured, open it (enable_wal_manager may
        // have been called earlier). Otherwise open raw wal fd as legacy.
        if (wal_manager_)
        {
            if (!wal_manager_->open(wal_path_, true, pomai::memory::WalConfig{}))
            {
                std::cerr << "SoaIdsManager::open: wal_manager_->open failed\n";
                return false;
            }
        }
        else
        {
            if (!open_wal_file_if_needed())
                return false;
        }

        // replay WAL if any (WalManager path will be used if present)
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
        wal_path_.clear();

        if (wal_manager_)
        {
            wal_manager_.reset(nullptr);
        }

        ids_mmap_.close();
        ids_path_.clear();
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
        // If structured WalManager is configured, use it instead of raw fd
        if (wal_manager_)
        {
            if (!wal_manager_->open(wal_path_, true, pomai::memory::WalConfig{}))
            {
                std::cerr << "SoaIdsManager::open_wal_file_if_needed: wal_manager_->open failed\n";
                return false;
            }
            // leave wal_fd_ unchanged (legacy) but prefer wal_manager_
            return true;
        }

        // open WAL for append/read/write (legacy)
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
        // Prefer wal_manager_ if configured
        if (wal_manager_)
        {
            bool ok = wal_manager_->replay([this](uint16_t type, const void *payload, uint32_t len, uint64_t /*seq*/) -> bool
                                           {
                if (type == pomai::memory::WAL_REC_IDS_UPDATE)
                {
                    if (!payload || len != static_cast<uint32_t>(sizeof(WalEntry)))
                    {
                        std::cerr << "SoaIdsManager::replay_wal_and_truncate: wal record size mismatch\n";
                        return false; // abort replay
                    }
                    WalEntry we;
                    std::memcpy(&we, payload, sizeof(we));
                    if (we.idx >= num_entries_)
                    {
                        std::cerr << "SoaIdsManager::replay_wal_and_truncate: wal entry index out of range: " << we.idx << "\n";
                        return true; // continue
                    }
                    uint64_t *ptr = const_cast<uint64_t *>(ids_ptr()) + static_cast<size_t>(we.idx);
                    pomai::ai::atomic_utils::atomic_store_u64(ptr, we.value);

                    size_t offset = static_cast<size_t>(we.idx) * sizeof(uint64_t);
                    if (!ids_mmap_.flush(offset, sizeof(uint64_t), /*sync=*/true))
                    {
                        std::cerr << "SoaIdsManager::replay_wal_and_truncate: msync failed for idx=" << we.idx << "\n";
                        return false;
                    }
                }
                // ignore other record types
                return true; });

            if (!ok)
            {
                std::cerr << "SoaIdsManager::replay_wal_and_truncate: wal_manager_->replay failed\n";
                return false;
            }

            if (!wal_manager_->truncate_to_zero())
            {
                std::cerr << "SoaIdsManager::replay_wal_and_truncate: wal_manager_->truncate_to_zero failed\n";
                // not fatal
            }
            return true;
        }

        // Legacy path: use raw wal_fd_ and simple fixed-size WalEntry records
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

            uint64_t *ptr = const_cast<uint64_t *>(ids_ptr()) + static_cast<size_t>(we.idx);

            // Use atomic_utils to ensure correct atomic store ordering even for mmap'd memory.
            pomai::ai::atomic_utils::atomic_store_u64(ptr, we.value);

            // msync that 8-byte range to persist update
            size_t offset = static_cast<size_t>(we.idx) * sizeof(uint64_t);
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

        uint64_t *ptr = const_cast<uint64_t *>(ids_ptr()) + idx;

        if (!durable)
        {
            // Fast path: atomic in-memory store only
            pomai::ai::atomic_utils::atomic_store_u64(ptr, value);
            // Note: not msynced; durability not guaranteed
            return true;
        }

        std::lock_guard<std::mutex> lk(wal_mu_);

        // If structured WAL manager is available, use it (preferred)
        if (wal_manager_)
        {
            // prepare payload = [idx:u64][value:u64]
            uint8_t payload[sizeof(uint64_t) * 2];
            std::memcpy(payload + 0, &idx, sizeof(uint64_t));
            std::memcpy(payload + sizeof(uint64_t), &value, sizeof(uint64_t));

            auto seq = wal_manager_->append_record(pomai::memory::WAL_REC_IDS_UPDATE, payload, static_cast<uint32_t>(sizeof(payload)));
            if (!seq)
            {
                std::cerr << "SoaIdsManager::atomic_update: wal_manager_->append_record failed\n";
                return false;
            }

            // apply atomic store to mapped ids
            pomai::ai::atomic_utils::atomic_store_u64(ptr, value);

            // msync the 8-byte range to persist the ids file
            size_t offset = idx * sizeof(uint64_t);
            if (!ids_mmap_.flush(offset, sizeof(uint64_t), /*sync=*/true))
            {
                std::cerr << "SoaIdsManager::atomic_update: msync failed for idx=" << idx << "\n";
                return false;
            }

            // Truncate WAL (we've persisted data)
            if (!wal_manager_->truncate_to_zero())
            {
                std::cerr << "SoaIdsManager::atomic_update: wal_manager_->truncate_to_zero failed (warning)\n";
                // not fatal
            }

            return true;
        }

        // Legacy raw WAL fd path (existing behavior)
        if (wal_fd_ == -1)
        {
            if (!open_wal_file_if_needed())
                return false;
        }

        WalEntry we;
        we.idx = static_cast<uint64_t>(idx);
        we.value = value;

        // Append WAL entry (legacy)
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
            return false;
        }

        // Apply atomic store to mapped ids using atomic_utils
        pomai::ai::atomic_utils::atomic_store_u64(ptr, value);

        // msync the 8-byte range to persist the ids file
        size_t offset = idx * sizeof(uint64_t);
        if (!ids_mmap_.flush(offset, sizeof(uint64_t), /*sync=*/true))
        {
            std::cerr << "SoaIdsManager::atomic_update: msync ids failed for idx=" << idx << "\n";
            // attempt to continue; WAL has entry and will be replayed on restart
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

    bool SoaIdsManager::enable_wal_manager(const std::string &wal_path, const pomai::memory::WalConfig &cfg)
    {
        std::lock_guard<std::mutex> lk(wal_mu_);

        // If already configured, ensure path matches
        if (wal_manager_)
        {
            if (wal_manager_->path() == wal_path)
                return true;
            // else reset and recreate
            wal_manager_.reset(nullptr);
        }

        wal_manager_ = std::make_unique<pomai::memory::WalManager>();
        if (!wal_manager_->open(wal_path, true, cfg))
        {
            std::cerr << "SoaIdsManager::enable_wal_manager: WalManager::open failed\n";
            wal_manager_.reset(nullptr);
            return false;
        }

        // If a legacy wal_fd_ exists, close it and rely on wal_manager_ exclusively
        if (wal_fd_ != -1)
        {
            ::close(wal_fd_);
            wal_fd_ = -1;
        }

        // Replay existing WAL via wal_manager_ to ensure we are consistent
        if (!replay_wal_and_truncate())
        {
            std::cerr << "SoaIdsManager::enable_wal_manager: replay after enable failed\n";
            // keep wal_manager_ but signal failure
            return false;
        }

        // store path for informational parity
        wal_path_ = wal_path;
        return true;
    }

} // namespace pomai::ai::soa