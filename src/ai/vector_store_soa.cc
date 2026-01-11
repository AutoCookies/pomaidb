#include "src/ai/vector_store_soa.h"

#include <cstring>
#include <iostream>
#include <limits>
#include <cassert>
#include <sys/stat.h>

namespace pomai::ai::soa
{

// -------------------- small helpers --------------------

static inline size_t align_up(size_t offset, size_t align) noexcept
{
    if (align == 0)
        return offset;
    size_t mask = align - 1;
    return (offset + mask) & ~mask;
}

// -------------------- static size helpers --------------------

// codebooks size: m * k * subdim * sizeof(float)
// subdim = max(1, dim / m)
size_t VectorStoreSoA::compute_codebooks_size_bytes(uint32_t dim, uint16_t pq_m, uint16_t pq_k) noexcept
{
    if (dim == 0 || pq_m == 0 || pq_k == 0)
        return 0;
    size_t m = static_cast<size_t>(pq_m);
    size_t k = static_cast<size_t>(pq_k);
    size_t subdim = (dim / pq_m);
    if (subdim == 0)
        subdim = 1;
    // check overflow: m * k * subdim * sizeof(float)
    if (subdim > (std::numeric_limits<size_t>::max() / sizeof(float)))
        return 0;
    size_t floats = m;
    if (k > (std::numeric_limits<size_t>::max() / floats)) return 0;
    floats *= k;
    if (subdim > (std::numeric_limits<size_t>::max() / floats)) return 0;
    floats *= subdim;
    if (floats > (std::numeric_limits<size_t>::max() / sizeof(float))) return 0;
    return static_cast<size_t>(floats * sizeof(float));
}

size_t VectorStoreSoA::compute_fingerprints_size_bytes(uint64_t num_vectors, uint16_t fingerprint_bits) noexcept
{
    if (fingerprint_bits == 0 || num_vectors == 0)
        return 0;
    size_t bytes_per = (fingerprint_bits + 7) / 8;
    if (bytes_per == 0)
        return 0;
    if (num_vectors > (std::numeric_limits<size_t>::max() / bytes_per))
        return 0;
    return static_cast<size_t>(num_vectors) * bytes_per;
}

// New: per-slot flags (one uint32_t per vector) used to publish fingerprints atomically.
size_t VectorStoreSoA::compute_fingerprint_flags_size_bytes(uint64_t num_vectors) noexcept
{
    if (num_vectors == 0)
        return 0;
    if (num_vectors > (std::numeric_limits<size_t>::max() / sizeof(uint32_t)))
        return 0;
    return static_cast<size_t>(num_vectors) * sizeof(uint32_t);
}

// PQ codes (8-bit per subquantizer) total size = num_vectors * pq_m
size_t VectorStoreSoA::compute_pq_codes_size_bytes(uint64_t num_vectors, uint16_t pq_m) noexcept
{
    if (pq_m == 0 || num_vectors == 0)
        return 0;
    if (static_cast<size_t>(pq_m) > (std::numeric_limits<size_t>::max() / num_vectors))
        return 0;
    return static_cast<size_t>(num_vectors) * static_cast<size_t>(pq_m);
}

size_t VectorStoreSoA::compute_pq_packed4_size_bytes(uint64_t num_vectors, uint16_t pq_m) noexcept
{
    if (pq_m == 0 || num_vectors == 0)
        return 0;
    size_t packed_per = (pq_m + 1) / 2;
    if (packed_per == 0)
        return 0;
    if (num_vectors > (std::numeric_limits<size_t>::max() / packed_per))
        return 0;
    return static_cast<size_t>(num_vectors) * packed_per;
}

size_t VectorStoreSoA::compute_ids_size_bytes(uint64_t num_vectors) noexcept
{
    if (num_vectors == 0)
        return 0;
    if (num_vectors > (std::numeric_limits<size_t>::max() / sizeof(uint64_t)))
        return 0;
    return static_cast<size_t>(num_vectors) * sizeof(uint64_t);
}

size_t VectorStoreSoA::compute_ppe_size_bytes(uint64_t num_vectors, uint32_t ppe_entry_bytes) noexcept
{
    if (ppe_entry_bytes == 0 || num_vectors == 0)
        return 0;
    if (static_cast<size_t>(ppe_entry_bytes) > (std::numeric_limits<size_t>::max() / num_vectors))
        return 0;
    return static_cast<size_t>(num_vectors) * static_cast<size_t>(ppe_entry_bytes);
}

// -------------------- create / open statics --------------------

std::unique_ptr<VectorStoreSoA> VectorStoreSoA::create_new(const std::string &path,
                                                           uint64_t num_vectors,
                                                           uint32_t dim,
                                                           uint16_t pq_m,
                                                           uint16_t pq_k,
                                                           uint16_t fingerprint_bits,
                                                           uint32_t ppe_entry_bytes,
                                                           const std::string &user_meta)
{
    if (num_vectors == 0)
        return nullptr;

    auto inst = std::unique_ptr<VectorStoreSoA>(new VectorStoreSoA());

    // compute sizes
    size_t codebooks_sz = compute_codebooks_size_bytes(dim, pq_m, pq_k);
    size_t fingerprints_sz = compute_fingerprints_size_bytes(num_vectors, fingerprint_bits);
    size_t fingerprint_flags_sz = compute_fingerprint_flags_size_bytes(num_vectors);
    size_t pq_codes_sz = compute_pq_codes_size_bytes(num_vectors, pq_m);
    size_t pq_packed4_sz = compute_pq_packed4_size_bytes(num_vectors, pq_m);
    size_t ids_sz = compute_ids_size_bytes(num_vectors);
    size_t ppe_sz = compute_ppe_size_bytes(num_vectors, ppe_entry_bytes);
    size_t user_meta_sz = user_meta.empty() ? 0 : user_meta.size();

    // Compute offsets with proper alignment (must match build_and_write_header)
    size_t offset = sizeof(SoaMmapHeader);

    // codebooks (float) -> align 4
    if (codebooks_sz > 0)
    {
        offset = align_up(offset, alignof(float));
        offset += codebooks_sz;
    }

    // fingerprints -> align 4 (byte-packed)
    if (fingerprints_sz > 0)
    {
        offset = align_up(offset, 4);
        offset += fingerprints_sz;
    }

    // fingerprint flags -> align 4 (uint32_t)
    if (fingerprint_flags_sz > 0)
    {
        offset = align_up(offset, alignof(uint32_t));
        offset += fingerprint_flags_sz;
    }

    // pq_codes (8-bit per-subquantizer) -> no strict alignment (1)
    if (pq_codes_sz > 0)
    {
        offset = align_up(offset, 1);
        offset += pq_codes_sz;
    }

    // pq_packed4 -> no special alignment
    if (pq_packed4_sz > 0)
    {
        offset = align_up(offset, 1);
        offset += pq_packed4_sz;
    }

    // ids -> align 8 (uint64_t)
    if (ids_sz > 0)
    {
        offset = align_up(offset, alignof(uint64_t));
        offset += ids_sz;
    }

    // ppe -> align 8 (PPEEntry aligned to 8)
    if (ppe_sz > 0)
    {
        offset = align_up(offset, alignof(uint64_t));
        offset += ppe_sz;
    }

    // user meta -> no strict alignment
    if (user_meta_sz > 0)
    {
        offset = align_up(offset, 1);
        offset += user_meta_sz;
    }

    size_t total = offset;

    // create/truncate mapping file
    if (!inst->mmap_.open(path, total, /*create=*/true))
    {
        std::cerr << "VectorStoreSoA::create_new: mmap open failed for " << path << "\n";
        return nullptr;
    }

    inst->base_ptr_ = inst->mmap_.base_ptr();
    if (!inst->base_ptr_)
    {
        inst->mmap_.close();
        return nullptr;
    }

    if (!inst->build_and_write_header(num_vectors, dim, pq_m, pq_k, fingerprint_bits, ppe_entry_bytes, user_meta))
    {
        inst->mmap_.close();
        return nullptr;
    }

    inst->fingerprint_bytes_ = (fingerprint_bits == 0) ? 0 : ((fingerprint_bits + 7) / 8);
    inst->pq_packed_bytes_ = (pq_m == 0) ? 0 : ((pq_m + 1) / 2);
    inst->hdr_ = reinterpret_cast<const SoaMmapHeader *>(inst->base_ptr_);
    inst->next_index_.store(0, std::memory_order_release);

    return inst;
}

std::unique_ptr<VectorStoreSoA> VectorStoreSoA::open_existing(const std::string &path)
{
    auto inst = std::unique_ptr<VectorStoreSoA>(new VectorStoreSoA());

    // open without creating; MmapFileManager will map existing file size
    if (!inst->mmap_.open(path, 0, /*create=*/false))
    {
        std::cerr << "VectorStoreSoA::open_existing: mmap open failed for " << path << "\n";
        return nullptr;
    }

    inst->base_ptr_ = inst->mmap_.base_ptr();
    if (!inst->base_ptr_)
    {
        inst->mmap_.close();
        return nullptr;
    }

    inst->hdr_ = reinterpret_cast<const SoaMmapHeader *>(inst->base_ptr_);
    if (!inst->hdr_->is_valid())
    {
        std::cerr << "VectorStoreSoA::open_existing: header invalid\n";
        inst->mmap_.close();
        return nullptr;
    }

    inst->fingerprint_bytes_ = (inst->hdr_->fingerprint_bits == 0) ? 0 : ((inst->hdr_->fingerprint_bits + 7) / 8);
    inst->pq_packed_bytes_ = (inst->hdr_->pq_m == 0) ? 0 : ((inst->hdr_->pq_m + 1) / 2);

    // scan ids for next free slot (safe for tests)
    uint64_t nv = inst->hdr_->num_vectors;
    uint64_t next = nv;
    if (inst->hdr_->ids_offset != 0 && inst->hdr_->ids_size != 0)
    {
        const uint64_t *ids_base = reinterpret_cast<const uint64_t *>(inst->base_ptr_ + static_cast<size_t>(inst->hdr_->ids_offset));
        for (uint64_t i = 0; i < nv; ++i)
        {
            uint64_t v = pomai::ai::atomic_utils::atomic_load_u64(ids_base + i);
            if (v == 0)
            {
                next = i;
                break;
            }
        }
    }
    inst->next_index_.store(next, std::memory_order_release);

    return inst;
}

// -------------------- instance open_or_create/close --------------------

bool VectorStoreSoA::open_or_create(const std::string &path, const SoaMmapHeader &hdr_template)
{
    // Try opening existing file first
    struct stat st;
    bool exists = (stat(path.c_str(), &st) == 0);

    if (exists)
    {
        // open existing
        if (!mmap_.open(path, 0, /*create=*/false))
        {
            std::cerr << "VectorStoreSoA::open_or_create: failed to open existing " << path << "\n";
            return false;
        }
        base_ptr_ = mmap_.base_ptr();
        if (!base_ptr_)
        {
            mmap_.close();
            return false;
        }
        // initialize from mapping
        if (!init_from_mapping())
        {
            mmap_.close();
            return false;
        }
        return true;
    }

    // file doesn't exist -> create new using fields from hdr_template.
    uint64_t num_vectors = hdr_template.num_vectors;
    uint32_t dim = hdr_template.dim;
    uint16_t pq_m = hdr_template.pq_m;
    uint16_t pq_k = hdr_template.pq_k;
    uint16_t fingerprint_bits = hdr_template.fingerprint_bits;

    // build total size same as create_new (use same alignment policy)
    size_t codebooks_sz = compute_codebooks_size_bytes(dim, pq_m, pq_k);
    size_t fingerprints_sz = compute_fingerprints_size_bytes(num_vectors, fingerprint_bits);
    size_t fingerprint_flags_sz = compute_fingerprint_flags_size_bytes(num_vectors);
    size_t pq_codes_sz = compute_pq_codes_size_bytes(num_vectors, pq_m);
    size_t pq_packed4_sz = compute_pq_packed4_size_bytes(num_vectors, pq_m);
    size_t ids_sz = compute_ids_size_bytes(num_vectors);
    size_t ppe_sz = compute_ppe_size_bytes(num_vectors, /*ppe_entry_bytes=*/0);
    size_t user_meta_sz = 0;

    size_t offset = sizeof(SoaMmapHeader);

    if (codebooks_sz > 0)
    {
        offset = align_up(offset, alignof(float));
        offset += codebooks_sz;
    }
    if (fingerprints_sz > 0)
    {
        offset = align_up(offset, 4);
        offset += fingerprints_sz;
    }
    if (fingerprint_flags_sz > 0)
    {
        offset = align_up(offset, alignof(uint32_t));
        offset += fingerprint_flags_sz;
    }
    if (pq_codes_sz > 0)
    {
        offset = align_up(offset, 1);
        offset += pq_codes_sz;
    }
    if (pq_packed4_sz > 0)
    {
        offset = align_up(offset, 1);
        offset += pq_packed4_sz;
    }
    if (ids_sz > 0)
    {
        offset = align_up(offset, alignof(uint64_t));
        offset += ids_sz;
    }
    if (ppe_sz > 0)
    {
        offset = align_up(offset, alignof(uint64_t));
        offset += ppe_sz;
    }
    if (user_meta_sz > 0)
    {
        offset = align_up(offset, 1);
        offset += user_meta_sz;
    }

    size_t total = offset;

    if (!mmap_.open(path, total, /*create=*/true))
    {
        std::cerr << "VectorStoreSoA::open_or_create: failed to create file " << path << "\n";
        return false;
    }

    base_ptr_ = mmap_.base_ptr();
    if (!base_ptr_)
    {
        mmap_.close();
        return false;
    }

    // build and write header
    if (!build_and_write_header(num_vectors, dim, pq_m, pq_k, fingerprint_bits, /*ppe_entry_bytes=*/0, std::string()))
    {
        mmap_.close();
        return false;
    }

    // cache and init
    hdr_ = reinterpret_cast<const SoaMmapHeader *>(base_ptr_);
    fingerprint_bytes_ = (fingerprint_bits == 0) ? 0 : ((fingerprint_bits + 7) / 8);
    pq_packed_bytes_ = (pq_m == 0) ? 0 : ((pq_m + 1) / 2);
    next_index_.store(0, std::memory_order_release);

    return true;
}

void VectorStoreSoA::close()
{
    mmap_.close();
    hdr_ = nullptr;
    base_ptr_ = nullptr;
    next_index_.store(0, std::memory_order_release);
}

// -------------------- destructor / basic ------------------------------------------------

VectorStoreSoA::~VectorStoreSoA()
{
    close();
}

bool VectorStoreSoA::is_valid() const noexcept
{
    return (hdr_ != nullptr) && hdr_->is_valid();
}

uint64_t VectorStoreSoA::num_vectors() const noexcept
{
    return hdr_ ? hdr_->num_vectors : 0;
}
uint32_t VectorStoreSoA::dim() const noexcept { return hdr_ ? hdr_->dim : 0; }
uint16_t VectorStoreSoA::pq_m() const noexcept { return hdr_ ? hdr_->pq_m : 0; }
uint16_t VectorStoreSoA::pq_k() const noexcept { return hdr_ ? hdr_->pq_k : 0; }
uint16_t VectorStoreSoA::fingerprint_bits() const noexcept { return hdr_ ? hdr_->fingerprint_bits : 0; }

// -------------------- build header ----------------------------------------------------

bool VectorStoreSoA::build_and_write_header(uint64_t num_vectors,
                                            uint32_t dim,
                                            uint16_t pq_m,
                                            uint16_t pq_k,
                                            uint16_t fingerprint_bits,
                                            uint32_t ppe_entry_bytes,
                                            const std::string &user_meta)
{
    if (!base_ptr_)
        return false;

    SoaMmapHeader h{};
    h.magic = SOA_MMAP_MAGIC;
    h.version = SOA_MMAP_HEADER_VERSION;
    h.header_size = static_cast<uint32_t>(sizeof(SoaMmapHeader));
    h.num_vectors = num_vectors;
    h.dim = dim;
    h.pq_m = pq_m;
    h.pq_k = pq_k;
    h.fingerprint_bits = fingerprint_bits;

    size_t offset = sizeof(SoaMmapHeader);

    size_t codebooks_sz = compute_codebooks_size_bytes(dim, pq_m, pq_k);
    if (codebooks_sz > 0)
    {
        offset = align_up(offset, alignof(float));
        h.codebooks_offset = static_cast<uint64_t>(offset);
        h.codebooks_size = static_cast<uint64_t>(codebooks_sz);
        offset += codebooks_sz;
    }
    else
    {
        h.codebooks_offset = 0;
        h.codebooks_size = 0;
    }

    size_t fingerprints_sz = compute_fingerprints_size_bytes(num_vectors, fingerprint_bits);
    if (fingerprints_sz > 0)
    {
        offset = align_up(offset, 4);
        h.fingerprints_offset = static_cast<uint64_t>(offset);
        h.fingerprints_size = static_cast<uint64_t>(fingerprints_sz);
        offset += fingerprints_sz;
    }
    else
    {
        h.fingerprints_offset = 0;
        h.fingerprints_size = 0;
    }

    // allocate/publish fingerprint flags block immediately after fingerprints
    size_t fingerprint_flags_sz = compute_fingerprint_flags_size_bytes(num_vectors);
    if (fingerprint_flags_sz > 0)
    {
        offset = align_up(offset, alignof(uint32_t));
        h.fingerprint_flags_offset = static_cast<uint64_t>(offset);
        h.fingerprint_flags_size = static_cast<uint64_t>(fingerprint_flags_sz);
        offset += fingerprint_flags_sz;
    }
    else
    {
        h.fingerprint_flags_offset = 0;
        h.fingerprint_flags_size = 0;
    }

    size_t pq_codes_sz = compute_pq_codes_size_bytes(num_vectors, pq_m);
    if (pq_codes_sz > 0)
    {
        offset = align_up(offset, 1);
        h.pq_codes_offset = static_cast<uint64_t>(offset);
        h.pq_codes_size = static_cast<uint64_t>(pq_codes_sz);
        offset += pq_codes_sz;
    }
    else
    {
        h.pq_codes_offset = 0;
        h.pq_codes_size = 0;
    }

    size_t pq_packed4_sz = compute_pq_packed4_size_bytes(num_vectors, pq_m);
    if (pq_packed4_sz > 0)
    {
        offset = align_up(offset, 1);
        h.pq_packed4_offset = static_cast<uint64_t>(offset);
        h.pq_packed4_size = static_cast<uint64_t>(pq_packed4_sz);
        offset += pq_packed4_sz;
    }
    else
    {
        h.pq_packed4_offset = 0;
        h.pq_packed4_size = 0;
    }

    size_t ids_sz = compute_ids_size_bytes(num_vectors);
    if (ids_sz > 0)
    {
        offset = align_up(offset, alignof(uint64_t));
        h.ids_offset = static_cast<uint64_t>(offset);
        h.ids_size = static_cast<uint64_t>(ids_sz);
        offset += ids_sz;
    }
    else
    {
        h.ids_offset = 0;
        h.ids_size = 0;
    }

    size_t ppe_sz = compute_ppe_size_bytes(num_vectors, ppe_entry_bytes);
    if (ppe_sz > 0)
    {
        offset = align_up(offset, alignof(uint64_t));
        h.ppe_offset = static_cast<uint64_t>(offset);
        h.ppe_size = static_cast<uint64_t>(ppe_sz);
        offset += ppe_sz;
    }
    else
    {
        h.ppe_offset = 0;
        h.ppe_size = 0;
    }

    size_t user_meta_sz = user_meta.empty() ? 0 : user_meta.size();
    if (user_meta_sz > 0)
    {
        offset = align_up(offset, 1);
        h.user_meta_offset = static_cast<uint64_t>(offset);
        h.user_meta_size = static_cast<uint64_t>(user_meta_sz);
        offset += user_meta_sz;
    }
    else
    {
        h.user_meta_offset = 0;
        h.user_meta_size = 0;
    }

    h.header_checksum = 0;

    // write header
    std::memcpy(const_cast<char *>(base_ptr_), &h, sizeof(SoaMmapHeader));

    // zero/initialize blocks
    if (h.codebooks_offset != 0 && h.codebooks_size != 0)
        std::memset(const_cast<char *>(base_ptr_) + static_cast<size_t>(h.codebooks_offset), 0, static_cast<size_t>(h.codebooks_size));
    if (h.fingerprints_offset != 0 && h.fingerprints_size != 0)
        std::memset(const_cast<char *>(base_ptr_) + static_cast<size_t>(h.fingerprints_offset), 0, static_cast<size_t>(h.fingerprints_size));
    if (h.fingerprint_flags_offset != 0 && h.fingerprint_flags_size != 0)
        std::memset(const_cast<char *>(base_ptr_) + static_cast<size_t>(h.fingerprint_flags_offset), 0, static_cast<size_t>(h.fingerprint_flags_size));
    if (h.pq_codes_offset != 0 && h.pq_codes_size != 0)
        std::memset(const_cast<char *>(base_ptr_) + static_cast<size_t>(h.pq_codes_offset), 0, static_cast<size_t>(h.pq_codes_size));
    if (h.pq_packed4_offset != 0 && h.pq_packed4_size != 0)
        std::memset(const_cast<char *>(base_ptr_) + static_cast<size_t>(h.pq_packed4_offset), 0, static_cast<size_t>(h.pq_packed4_size));
    if (h.ids_offset != 0 && h.ids_size != 0)
        std::memset(const_cast<char *>(base_ptr_) + static_cast<size_t>(h.ids_offset), 0, static_cast<size_t>(h.ids_size));
    if (h.ppe_offset != 0 && h.ppe_size != 0)
        std::memset(const_cast<char *>(base_ptr_) + static_cast<size_t>(h.ppe_offset), 0, static_cast<size_t>(h.ppe_size));
    if (h.user_meta_offset != 0 && h.user_meta_size != 0)
        std::memcpy(const_cast<char *>(base_ptr_) + static_cast<size_t>(h.user_meta_offset), user_meta.data(), user_meta_sz);

    // flush header region synchronously
    if (!mmap_.flush(0, sizeof(SoaMmapHeader), true))
        std::cerr << "VectorStoreSoA::build_and_write_header: msync header failed\n";

    // set hdr_ pointer so callers can use it
    hdr_ = reinterpret_cast<const SoaMmapHeader *>(base_ptr_);

    return true;
}

// -------------------- init helpers ----------------------------------------------------

bool VectorStoreSoA::init_from_mapping()
{
    if (!base_ptr_)
        return false;
    hdr_ = reinterpret_cast<const SoaMmapHeader *>(base_ptr_);
    if (!hdr_ || !hdr_->is_valid())
        return false;
    fingerprint_bytes_ = (hdr_->fingerprint_bits == 0) ? 0 : ((hdr_->fingerprint_bits + 7) / 8);
    pq_packed_bytes_ = (hdr_->pq_m == 0) ? 0 : ((hdr_->pq_m + 1) / 2);

    // compute next_index by scanning ids (safe for tests)
    uint64_t nv = hdr_->num_vectors;
    uint64_t next = nv;
    if (hdr_->ids_offset != 0 && hdr_->ids_size != 0)
    {
        const uint64_t *ids_base = reinterpret_cast<const uint64_t *>(base_ptr_ + static_cast<size_t>(hdr_->ids_offset));
        for (uint64_t i = 0; i < nv; ++i)
        {
            uint64_t v = pomai::ai::atomic_utils::atomic_load_u64(ids_base + i);
            if (v == 0)
            {
                next = i;
                break;
            }
        }
    }
    next_index_.store(next, std::memory_order_release);
    return true;
}

// -------------------- append / read ----------------------------------------------------

size_t VectorStoreSoA::append_vector(const uint8_t *fp, uint32_t fp_len,
                                     const uint8_t *pq_packed_or_codes, uint32_t pq_len,
                                     uint64_t id_entry)
{
    if (!hdr_ || !base_ptr_)
        return SIZE_MAX;

    // Validate fingerprint length
    if (fingerprint_bytes_ != fp_len)
    {
        if (fingerprint_bytes_ != 0 || fp_len != 0)
            return SIZE_MAX;
    }

    // pq_len can be either:
    //  - pq_m (raw 8-bit codes) when pq_codes block is used, or
    //  - pq_packed_bytes_ (packed4) when pq_packed4 block used, or 0 if none.
    if (pq_len != 0)
    {
        bool ok_raw = (hdr_->pq_codes_offset != 0 && pq_len == static_cast<uint32_t>(hdr_->pq_m));
        bool ok_packed = (hdr_->pq_packed4_offset != 0 && pq_packed_bytes_ > 0 && pq_len == static_cast<uint32_t>(pq_packed_bytes_));
        if (!ok_raw && !ok_packed)
            return SIZE_MAX;
    }

    std::lock_guard<std::mutex> lk(append_mu_);

    uint64_t idx = next_index_.load(std::memory_order_acquire);
    if (idx >= hdr_->num_vectors)
        return SIZE_MAX;

    // Write PQ raw codes or packed4 first (if provided).
    if (pq_len != 0)
    {
        // raw 8-bit codes
        if (hdr_->pq_codes_offset != 0 && pq_len == static_cast<uint32_t>(hdr_->pq_m))
        {
            char *dst_codes = const_cast<char *>(base_ptr_) + static_cast<size_t>(hdr_->pq_codes_offset) + static_cast<size_t>(idx) * static_cast<size_t>(hdr_->pq_m);
            std::memcpy(dst_codes, pq_packed_or_codes, pq_len);
        }
        // packed4
        else if (hdr_->pq_packed4_offset != 0 && pq_packed_bytes_ > 0 && pq_len == static_cast<uint32_t>(pq_packed_bytes_))
        {
            char *dst_pq = const_cast<char *>(base_ptr_) + static_cast<size_t>(hdr_->pq_packed4_offset) + static_cast<size_t>(idx) * pq_packed_bytes_;
            std::memcpy(dst_pq, pq_packed_or_codes, pq_len);
        }
        else
        {
            return SIZE_MAX;
        }
    }

    // Write IDs next.
    if (hdr_->ids_offset != 0 && hdr_->ids_size != 0)
    {
        uint64_t *ids_base = reinterpret_cast<uint64_t *>(const_cast<char *>(base_ptr_) + static_cast<size_t>(hdr_->ids_offset));
        pomai::ai::atomic_utils::atomic_store_u64(ids_base + idx, id_entry);
    }

    // Finally, write fingerprint bytes and atomically publish the per-slot flag.
    // Publishing the flag is done after PQ and IDs are in place to avoid readers
    // observing a published fingerprint without corresponding id/pq data.
    if (hdr_->fingerprints_offset != 0 && fingerprint_bytes_ > 0 && fp && fp_len > 0)
    {
        char *dst_fp = const_cast<char *>(base_ptr_) + static_cast<size_t>(hdr_->fingerprints_offset) + static_cast<size_t>(idx) * fingerprint_bytes_;
        std::memcpy(dst_fp, fp, fp_len);

        // publish fingerprint flag last (atomic store)
        if (hdr_->fingerprint_flags_offset != 0 && hdr_->fingerprint_flags_size != 0)
        {
            uint32_t *flags_base = reinterpret_cast<uint32_t *>(const_cast<char *>(base_ptr_) + static_cast<size_t>(hdr_->fingerprint_flags_offset));
            pomai::ai::atomic_utils::atomic_store_u32(flags_base + idx, 1u);
        }
    }

    next_index_.store(idx + 1, std::memory_order_release);
    return static_cast<size_t>(idx);
}

const uint8_t *VectorStoreSoA::pq_packed_ptr(size_t idx) const noexcept
{
    if (!hdr_ || !base_ptr_)
        return nullptr;
    if (hdr_->pq_packed4_offset == 0 || hdr_->pq_packed4_size == 0 || pq_packed_bytes_ == 0)
        return nullptr;
    if (idx >= static_cast<size_t>(hdr_->num_vectors))
        return nullptr;
    return reinterpret_cast<const uint8_t *>(base_ptr_ + static_cast<size_t>(hdr_->pq_packed4_offset) + idx * pq_packed_bytes_);
}

const uint8_t *VectorStoreSoA::pq_codes_ptr(size_t idx) const noexcept
{
    if (!hdr_ || !base_ptr_)
        return nullptr;
    if (hdr_->pq_codes_offset == 0 || hdr_->pq_codes_size == 0 || hdr_->pq_m == 0)
        return nullptr;
    if (idx >= static_cast<size_t>(hdr_->num_vectors))
        return nullptr;
    return reinterpret_cast<const uint8_t *>(base_ptr_ + static_cast<size_t>(hdr_->pq_codes_offset) + idx * static_cast<size_t>(hdr_->pq_m));
}

const float *VectorStoreSoA::codebooks_ptr() const noexcept
{
    if (!hdr_ || !base_ptr_)
        return nullptr;
    if (hdr_->codebooks_offset == 0 || hdr_->codebooks_size == 0)
        return nullptr;
    return reinterpret_cast<const float *>(base_ptr_ + static_cast<size_t>(hdr_->codebooks_offset));
}

size_t VectorStoreSoA::codebooks_size_bytes() const noexcept
{
    if (!hdr_)
        return 0;
    return static_cast<size_t>(hdr_->codebooks_size);
}

bool VectorStoreSoA::write_codebooks(const float *src, size_t float_count)
{
    if (!hdr_ || !base_ptr_ || !src)
        return false;
    if (hdr_->codebooks_offset == 0 || hdr_->codebooks_size == 0)
        return false;
    size_t expected_floats = static_cast<size_t>(hdr_->codebooks_size) / sizeof(float);
    if (float_count != expected_floats)
        return false;
    char *dst = const_cast<char *>(base_ptr_) + static_cast<size_t>(hdr_->codebooks_offset);
    std::memcpy(dst, reinterpret_cast<const char *>(src), static_cast<size_t>(hdr_->codebooks_size));
    // flush codebooks block synchronously
    return mmap_.flush(static_cast<size_t>(hdr_->codebooks_offset), static_cast<size_t>(hdr_->codebooks_size), true);
}

const uint8_t *VectorStoreSoA::fingerprint_ptr(size_t idx) const noexcept
{
    if (!hdr_ || !base_ptr_)
        return nullptr;
    if (hdr_->fingerprints_offset == 0 || hdr_->fingerprints_size == 0 || fingerprint_bytes_ == 0)
        return nullptr;
    if (idx >= static_cast<size_t>(hdr_->num_vectors))
        return nullptr;

    // If flags block present, check the per-slot published flag first (atomic).
    if (hdr_->fingerprint_flags_offset != 0 && hdr_->fingerprint_flags_size != 0)
    {
        const uint32_t *flags_base = reinterpret_cast<const uint32_t *>(base_ptr_ + static_cast<size_t>(hdr_->fingerprint_flags_offset));
        uint32_t f = pomai::ai::atomic_utils::atomic_load_u32(flags_base + idx);
        if (f == 0)
            return nullptr; // not yet published
    }

    return reinterpret_cast<const uint8_t *>(base_ptr_ + static_cast<size_t>(hdr_->fingerprints_offset) + idx * fingerprint_bytes_);
}

uint64_t VectorStoreSoA::id_entry_at(size_t idx) const noexcept
{
    if (!hdr_ || !base_ptr_)
        return 0;
    if (hdr_->ids_offset == 0 || hdr_->ids_size == 0)
        return 0;
    if (idx >= static_cast<size_t>(hdr_->num_vectors))
        return 0;
    const uint64_t *ids_base = reinterpret_cast<const uint64_t *>(base_ptr_ + static_cast<size_t>(hdr_->ids_offset));
    return pomai::ai::atomic_utils::atomic_load_u64(ids_base + idx);
}

const uint64_t *VectorStoreSoA::ids_ptr() const noexcept
{
    if (!hdr_ || !base_ptr_)
        return nullptr;
    if (hdr_->ids_offset == 0 || hdr_->ids_size == 0)
        return nullptr;
    return reinterpret_cast<const uint64_t *>(base_ptr_ + static_cast<size_t>(hdr_->ids_offset));
}

// -------------------- flush wrapper ----------------------------------------------------

bool VectorStoreSoA::flush(size_t offset, size_t len, bool sync)
{
    return mmap_.flush(offset, len, sync);
}

} // namespace pomai::ai::soa