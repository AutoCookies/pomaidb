#include "palloc_page_pool.h"
#include "core/metrics/metrics_registry.h"
#include "palloc_compat.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32) && !defined(_WIN64)
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace {

constexpr uint64_t kInvalidPageId = std::numeric_limits<uint64_t>::max();
constexpr uint64_t kInvalidSwapSlot = std::numeric_limits<uint64_t>::max();

struct PageMeta {
  uint64_t page_id = kInvalidPageId;
  uint64_t last_access_epoch = 0;
  size_t   swap_slot = kInvalidSwapSlot;
  uint32_t pin_count = 0;
  bool     dirty = false;
  bool     ref_bit = false;   // for CLOCK
  bool     in_use = false;    // frame is currently assigned to a page
};

struct SwapSlot {
  uint64_t index = kInvalidSwapSlot;
};

struct palloc_page_pool_impl {
  size_t page_size = 0;
  size_t capacity_bytes = 0;
  size_t max_resident_pages = 0;

  // Backing arena for resident pages: contiguous array of page_size * max_resident_pages bytes.
  uint8_t* arena = nullptr;

  // Swap file (POSIX only; on Windows we currently do not use swap).
#if !defined(_WIN32) && !defined(_WIN64)
  int swap_fd = -1;
#else
  int swap_fd = -1;
#endif

  std::vector<PageMeta> pages;
  std::unordered_map<uint64_t, size_t> page_index;  // page_id -> frame index

  // Mapping from page_id to swap slot for pages that are currently swapped out.
  std::unordered_map<uint64_t, uint64_t> swapped;
  std::vector<uint64_t> free_swap_slots;
  uint64_t next_swap_slot = 0;

  size_t resident_pages = 0;
  size_t dirty_pages = 0;
  size_t bytes_in_swap = 0;
  size_t evictions = 0;

  size_t clock_hand = 0;
  uint64_t epoch = 0;

  ~palloc_page_pool_impl() {
    // Flush best-effort but ignore errors during destruction.
    (void)flush_all_locked();
    if (arena) {
      palloc_free(arena);
      arena = nullptr;
    }
#if !defined(_WIN32) && !defined(_WIN64)
    if (swap_fd >= 0) {
      ::close(swap_fd);
      swap_fd = -1;
    }
#endif
  }

  uint8_t* frame_ptr(size_t frame) noexcept {
    return arena + frame * page_size;
  }

  bool ensure_swap_file(const char* path) {
#if defined(_WIN32) || defined(_WIN64)
    (void)path;
    // Swap not supported on Windows for now.
    return false;
#else
    if (!path) return false;
    if (swap_fd >= 0) return true;
    int fd = ::open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) return false;
    swap_fd = fd;
    return true;
#endif
  }

  bool write_page_to_swap(size_t frame, uint64_t page_id) {
#if defined(_WIN32) || defined(_WIN64)
    (void)frame;
    (void)page_id;
    return false;
#else
    if (swap_fd < 0) return false;
    uint64_t slot;
    if (!free_swap_slots.empty()) {
      slot = free_swap_slots.back();
      free_swap_slots.pop_back();
    } else {
      slot = next_swap_slot++;
    }
    off_t off = static_cast<off_t>(slot * page_size);
    ssize_t n = ::pwrite(swap_fd, frame_ptr(frame), page_size, off);
    if (n != static_cast<ssize_t>(page_size)) {
      // Do not reuse this slot; it may contain partial data.
      return false;
    }
    swapped[page_id] = slot;
    bytes_in_swap = std::max(bytes_in_swap, static_cast<size_t>((slot + 1) * page_size));
    pages[frame].swap_slot = slot;
    return true;
#endif
  }

  bool read_page_from_swap(size_t frame, uint64_t page_id) {
#if defined(_WIN32) || defined(_WIN64)
    (void)frame;
    (void)page_id;
    return false;
#else
    auto it = swapped.find(page_id);
    if (it == swapped.end()) {
      // Page has no backing swap; treat as zero-filled.
      std::fill_n(frame_ptr(frame), page_size, uint8_t{0});
      return true;
    }
    uint64_t slot = it->second;
    off_t off = static_cast<off_t>(slot * page_size);
    ssize_t n = ::pread(swap_fd, frame_ptr(frame), page_size, off);
    if (n != static_cast<ssize_t>(page_size)) {
      return false;
    }
    return true;
#endif
  }

  bool flush_frame_locked(size_t frame) {
    PageMeta& m = pages[frame];
    if (!m.in_use || !m.dirty) return true;
    if (swap_fd < 0) return false;
    if (!write_page_to_swap(frame, m.page_id)) {
      return false;
    }
    m.dirty = false;
    if (dirty_pages > 0) --dirty_pages;
    return true;
  }

  bool flush_all_locked() {
    bool ok = true;
    for (size_t i = 0; i < pages.size(); ++i) {
      if (!flush_frame_locked(i)) ok = false;
    }
    return ok;
  }

  // Returns frame index or pages.size() on failure.
  size_t allocate_frame_for_page(uint64_t page_id, bool for_write, bool* is_new) {
    // Try free frame first.
    for (size_t i = 0; i < pages.size(); ++i) {
      if (!pages[i].in_use) {
        PageMeta& m = pages[i];
        m.in_use = true;
        m.page_id = page_id;
        m.pin_count = 0;
        m.ref_bit = true;
        m.last_access_epoch = ++epoch;
        m.dirty = false;
        m.swap_slot = kInvalidSwapSlot;
        ++resident_pages;
        if (is_new) *is_new = (swapped.find(page_id) == swapped.end());
        if (!read_page_from_swap(i, page_id)) {
          // On read failure, treat as failure.
          m.in_use = false;
          --resident_pages;
          return pages.size();
        }
        return i;
      }
    }

    // Eviction via CLOCK over unpinned pages.
    const size_t n = pages.size();
    for (size_t pass = 0; pass < 2 * n; ++pass) {
      PageMeta& m = pages[clock_hand];
      size_t idx = clock_hand;
      clock_hand = (clock_hand + 1) % max_resident_pages; // Use max_resident_pages as per user's snippet
      if (!m.in_use || m.pin_count != 0) continue;
      if (m.ref_bit) {
        m.ref_bit = false;
        continue;
      }
      // Candidate for eviction.
      if (m.dirty) {
        if (!flush_frame_locked(idx)) {
          continue;  // try next frame
        }
      }
      // Remove from resident index.
      page_index.erase(m.page_id);
      pomai::core::metrics::MetricsRegistry::Instance().Increment("palloc_page_eviction");
      ++evictions;
      m.page_id = page_id;
      m.pin_count = 0;
      m.ref_bit = true;
      m.last_access_epoch = ++epoch;
      m.dirty = false;
      // Keep swap_slot as-is; write_page_to_swap will update if needed.
      if (is_new) *is_new = (swapped.find(page_id) == swapped.end());
      if (!read_page_from_swap(idx, page_id)) {
        // failed to bring in; mark frame unused
        m.in_use = false;
        if (resident_pages > 0) --resident_pages;
        return pages.size();
      }
      return idx;
    }
    return pages.size();
  }

};  // struct palloc_page_pool_impl

// Helper to compute defaults based on profile.
static void fill_defaults(palloc_page_pool_options* opts) {
  if (opts->page_size == 0) {
    switch (opts->device_profile) {
      case PALLOC_DEVICE_PROFILE_SMALL:
        opts->page_size = 4u * 1024u;
        break;
      case PALLOC_DEVICE_PROFILE_MEDIUM:
        opts->page_size = 8u * 1024u;
        break;
      case PALLOC_DEVICE_PROFILE_LARGE:
      default:
        opts->page_size = 16u * 1024u;
        break;
    }
  }
  if (opts->capacity_bytes == 0) {
    // Conservative defaults targeting edge devices.
    switch (opts->device_profile) {
      case PALLOC_DEVICE_PROFILE_SMALL:
        opts->capacity_bytes = 32u * 1024u * 1024u;  // 32 MiB
        break;
      case PALLOC_DEVICE_PROFILE_MEDIUM:
        opts->capacity_bytes = 64u * 1024u * 1024u;  // 64 MiB
        break;
      case PALLOC_DEVICE_PROFILE_LARGE:
      default:
        opts->capacity_bytes = 128u * 1024u * 1024u; // 128 MiB
        break;
    }
  }
}

}  // namespace

struct palloc_page_pool {
  palloc_page_pool_impl impl;
};

static palloc_page_pool* g_default_pool = nullptr;

extern "C" {

palloc_page_pool* palloc_page_pool_create(const palloc_page_pool_options* in_opts) {
  if (!in_opts) return nullptr;
  palloc_page_pool_options opts = *in_opts;
  fill_defaults(&opts);
  if (opts.page_size == 0 || opts.capacity_bytes < opts.page_size) {
    return nullptr;
  }

  auto* pool = static_cast<palloc_page_pool*>(
      palloc_malloc_aligned(sizeof(palloc_page_pool), alignof(palloc_page_pool)));
  if (!pool) return nullptr;

  // `palloc_page_pool` is a C++ type containing non-trivial members (mutex,
  // vectors, etc.). We must construct it; treating raw memory as a live
  // object is undefined behavior and breaks under ASan.
  new (pool) palloc_page_pool{};

  palloc_page_pool_impl& impl = pool->impl;
  impl.page_size = opts.page_size;
  impl.capacity_bytes = opts.capacity_bytes;
  impl.max_resident_pages = opts.capacity_bytes / opts.page_size;
  if (impl.max_resident_pages == 0) {
    pool->~palloc_page_pool();
    palloc_free(pool);
    return nullptr;
  }

  impl.arena = static_cast<uint8_t*>(
      palloc_malloc_aligned(impl.max_resident_pages * impl.page_size, opts.page_size));
  if (!impl.arena) {
    pool->~palloc_page_pool();
    palloc_free(pool);
    return nullptr;
  }

  impl.pages.resize(impl.max_resident_pages);

#if !defined(_WIN32) && !defined(_WIN64)
  if (opts.swap_file_path && !impl.ensure_swap_file(opts.swap_file_path)) {
    // We still allow creation without swap; eviction will fail once memory is full.
  }
#else
  (void)opts;
#endif

  return pool;
}

void palloc_page_pool_destroy(palloc_page_pool* pool) {
  if (!pool) return;
  pool->~palloc_page_pool();
  palloc_free(pool);
}

void* palloc_fetch_page(palloc_page_pool* pool, uint64_t page_id,
                        int for_write, int* is_new) {
  if (!pool) return nullptr;
  palloc_page_pool_impl& impl = pool->impl;

  auto it = impl.page_index.find(page_id);
  if (it != impl.page_index.end()) {
    size_t frame = it->second;
    PageMeta& m = impl.pages[frame];
    m.pin_count += 1;
    m.ref_bit = true;
    m.last_access_epoch = ++impl.epoch;
    if (is_new) *is_new = 0;
    pomai::core::metrics::MetricsRegistry::Instance().Increment("palloc_page_hit");
    (void)for_write;
    return impl.frame_ptr(frame);
  }

  pomai::core::metrics::MetricsRegistry::Instance().Increment("palloc_page_miss");
  bool local_is_new = false;
  size_t frame = impl.allocate_frame_for_page(page_id, for_write != 0, &local_is_new);
  if (frame == impl.pages.size()) {
    return nullptr;
  }

  impl.page_index[page_id] = frame;
  PageMeta& m = impl.pages[frame];
  m.pin_count = 1;
  m.ref_bit = true;
  if (for_write) {
    m.dirty = true;
    ++impl.dirty_pages;
  }
  if (is_new) *is_new = local_is_new ? 1 : 0;
  return impl.frame_ptr(frame);
}

void palloc_unpin_page(palloc_page_pool* pool, uint64_t page_id,
                       int mark_dirty_if_modified) {
  if (!pool) return;
  palloc_page_pool_impl& impl = pool->impl;
  auto it = impl.page_index.find(page_id);
  if (it == impl.page_index.end()) return;
  size_t frame = it->second;
  PageMeta& m = impl.pages[frame];
  if (m.pin_count > 0) {
    m.pin_count -= 1;
  }
  if (mark_dirty_if_modified) {
    if (!m.dirty) {
      m.dirty = true;
      ++impl.dirty_pages;
    }
  }
}

int palloc_flush_page(palloc_page_pool* pool, uint64_t page_id) {
  if (!pool) return -1;
  palloc_page_pool_impl& impl = pool->impl;
  auto it = impl.page_index.find(page_id);
  if (it == impl.page_index.end()) {
    return 0;  // nothing to do
  }
  size_t frame = it->second;
  return impl.flush_frame_locked(frame) ? 0 : -1;
}

int palloc_flush_all(palloc_page_pool* pool) {
  if (!pool) return -1;
  palloc_page_pool_impl& impl = pool->impl;
  return impl.flush_all_locked() ? 0 : -1;
}

void palloc_page_pool_get_stats(palloc_page_pool* pool,
                                palloc_page_pool_stats* out_stats) {
  if (!pool || !out_stats) return;
  palloc_page_pool_impl& impl = pool->impl;
  out_stats->page_size = impl.page_size;
  out_stats->capacity_bytes = impl.capacity_bytes;
  out_stats->resident_pages = impl.resident_pages;
  out_stats->dirty_pages = impl.dirty_pages;
  out_stats->bytes_in_swap = impl.bytes_in_swap;
  out_stats->evictions = impl.evictions;
}

palloc_page_pool* palloc_get_default_page_pool(const char* swap_file_path_hint) {
  if (g_default_pool) return g_default_pool;
  palloc_page_pool_options opts{};
  opts.page_size = 0;          // derive from profile
  opts.capacity_bytes = 0;     // derive from profile
  opts.swap_file_path = swap_file_path_hint;
  opts.device_profile = PALLOC_DEVICE_PROFILE_MEDIUM;
  g_default_pool = palloc_page_pool_create(&opts);
  return g_default_pool;
}

}  // extern "C"

