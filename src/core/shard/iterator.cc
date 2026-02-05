#include "iterator.h"
#include "core/shard/snapshot.h"
#include "table/memtable.h"
#include "table/segment.h"

namespace pomai::core
{
    ShardIterator::ShardIterator(std::shared_ptr<ShardSnapshot> snapshot)
        : snapshot_(std::move(snapshot)),
          source_(Source::FROZEN_MEM),
          source_idx_(0),
          entry_idx_(0),
          current_id_(0)
    {
        // Initialize: position at first valid entry
        AdvanceToNextLive();
    }

    bool ShardIterator::Next()
    {
        if (!Valid()) return false;
        
        // Mark current ID as seen
        seen_.insert(current_id_);
        
        // Advance to next valid entry
        entry_idx_++;
        AdvanceToNextLive();
        
        return Valid();
    }

    VectorId ShardIterator::id() const
    {
        return current_id_;
    }

    std::span<const float> ShardIterator::vector() const
    {
        return current_vec_;
    }

    bool ShardIterator::Valid() const
    {
        return source_ != Source::DONE;
    }

    // -------------------------
    // Private Helpers
    // -------------------------

    void ShardIterator::AdvanceToNextLive()
    {
        while (true) {
            // Try to read from current source
            bool found = false;
            
            if (source_ == Source::FROZEN_MEM) {
                found = TryReadFromFrozenMem();
                
                if (!found) {
                    // Move to segments
                    source_ = Source::SEGMENT;
                    source_idx_ = 0;
                    entry_idx_ = 0;
                    continue;
                }
            } else if (source_ == Source::SEGMENT) {
                found = TryReadFromSegment();
                
                if (!found) {
                    // No more sources
                    source_ = Source::DONE;
                    return;
                }
            } else {
                // Source::DONE
                return;
            }
            
            // Check if entry is valid (not seen, not tombstone)
            if (found) {
                if (seen_.count(current_id_) == 0 && !current_vec_.empty()) {
                    // Found valid live entry (not a duplicate, not a tombstone)
                    return;
                } else {
                    // Skip duplicate or tombstone
                    entry_idx_++;
                    continue;
                }
            }
        }
    }

    bool ShardIterator::TryReadFromFrozenMem()
    {
        // Iterate through frozen memtables (newest first)
        while (source_idx_ < snapshot_->frozen_memtables.size()) {
            const auto& fmem = snapshot_->frozen_memtables[source_idx_];
            
            // MemTable doesn't have indexed access, so we use IterateWithStatus
            // We need to convert entry_idx_ to actual iteration
            // 
            // Strategy: Build vector of entries on first access to this memtable
            // (This is inefficient but simple. Production-grade would use proper iterator)
            
            size_t current_entry = 0;
            bool found = false;
            
            fmem->IterateWithStatus([&](VectorId id, std::span<const float> vec, bool is_deleted) {
                if (current_entry == entry_idx_) {
                    current_id_ = id;
                    if (!is_deleted) {
                        current_vec_.assign(vec.begin(), vec.end());
                    } else {
                        current_vec_.clear(); // Tombstone: clear vector
                    }
                    found = true;
                }
                current_entry++;
            });
            
            if (found) {
                return true; // Found entry (live or tombstone)
            }
            
            // No more entries in this memtable, move to next
            source_idx_++;
            entry_idx_ = 0;
        }
        
        return false;
    }

    bool ShardIterator::TryReadFromSegment()
    {
        // Iterate through segments (newest first)
        while (source_idx_ < snapshot_->segments.size()) {
            const auto& seg = snapshot_->segments[source_idx_];
            
            // Try to read at entry_idx_
            VectorId id;
            std::span<const float> vec;
            bool is_deleted;
            
            auto st = seg->ReadAt(static_cast<uint32_t>(entry_idx_), &id, &vec, &is_deleted);
            
            if (st.ok()) {
                current_id_ = id;
                if (!is_deleted) {
                    current_vec_.assign(vec.begin(), vec.end());
                } else {
                    current_vec_.clear(); // Tombstone: clear vector
                }
                return true; // Found entry (live or tombstone)
            }
            
            // No more entries in this segment, move to next
            source_idx_++;
            entry_idx_ = 0;
        }
        
        return false;
    }

} // namespace pomai::core
