#include "src/ai/vector_index.h"
#include <queue>
#include <cmath>
#include <cstring>
#include <arpa/inet.h>
#include "src/core/seed.h"

VectorIndex::VectorIndex(PomaiMap *map) : map_(map), arena_(map ? map->get_arena() : nullptr) {}

// compute squared L2 distance between query and candidate (both length dim)
static inline double l2sq_distance(const float *a, const float *b, size_t dim)
{
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += d * d;
    }
    return sum;
}

std::vector<char> VectorIndex::search(const float *query, size_t dim, size_t topk) const
{
    std::vector<char> out;

    if (map_ == nullptr || arena_ == nullptr || query == nullptr || dim == 0 || topk == 0)
        return out;

    // Max-heap of pairs (score, Seed*) where score is squared distance.
    using Item = std::pair<double, Seed *>;
    struct Cmp
    {
        bool operator()(Item const &a, Item const &b) const { return a.first < b.first; } // max-heap
    };
    std::priority_queue<Item, std::vector<Item>, Cmp> heap;

    // Iterate seeds and consider those marked as OBJ_VECTOR using the new scan_all API
    map_->scan_all([&](Seed *s)
                   {
        if (!s) return;
        if (s->type != Seed::OBJ_VECTOR) return;

        uint16_t klen = s->get_klen();

        const char *vec_bytes = nullptr;
        uint32_t blen = 0;

        // Handle inline vs indirect storage
        if ((s->flags & Seed::FLAG_INDIRECT) == 0)
        {
            // inline: vlen stored in header
            blen = s->get_vlen();
            if (blen == 0) return;
            vec_bytes = s->payload + klen;
        }
        else
        {
            // indirect: read offset from payload and resolve via arena
            uint64_t offset = 0;
            memcpy(&offset, s->payload + klen, pomai::config::MAP_PTR_BYTES);
            const char *blob_hdr = arena_->blob_ptr_from_offset_for_map(offset);
            if (!blob_hdr) return;
            blen = *reinterpret_cast<const uint32_t *>(blob_hdr);
            if (blen == 0) return;
            vec_bytes = blob_hdr + sizeof(uint32_t);
        }

        if (!vec_bytes) return;
        if (blen % sizeof(float) != 0) return;
        size_t vec_len = blen / sizeof(float);
        if (vec_len != dim) return; // dimension mismatch; skip

        const float *vec = reinterpret_cast<const float *>(vec_bytes);

        double dist = l2sq_distance(query, vec, dim);

        if (heap.size() < topk)
        {
            heap.emplace(dist, s);
        }
        else if (dist < heap.top().first)
        {
            heap.pop();
            heap.emplace(dist, s);
        } });

    // Extract results (heap is max-heap => extract and reverse)
    std::vector<Item> results;
    results.reserve(heap.size());
    while (!heap.empty())
    {
        results.push_back(heap.top());
        heap.pop();
    }
    std::reverse(results.begin(), results.end()); // now ascending distance

    // Build binary buffer
    size_t est_size = 0;
    for (auto &it : results)
    {
        Seed *s = it.second;
        uint16_t klen = s->get_klen();
        est_size += 4 + klen + 4;
    }
    out.reserve(est_size);

    for (auto &it : results)
    {
        Seed *s = it.second;
        uint16_t klen = s->get_klen();
        // append keylen (network order)
        uint32_t net_klen = htonl(static_cast<uint32_t>(klen));
        const char *keyptr = s->payload;
        out.insert(out.end(), reinterpret_cast<char *>(&net_klen), reinterpret_cast<char *>(&net_klen) + 4);
        out.insert(out.end(), keyptr, keyptr + klen);
        // append score as float bits (network order)
        float score_f = static_cast<float>(it.first); // squared distance as float
        uint32_t score_bits = 0;
        static_assert(sizeof(score_f) == sizeof(score_bits), "float size mismatch");
        memcpy(&score_bits, &score_f, sizeof(score_bits));
        uint32_t net_score = htonl(score_bits);
        out.insert(out.end(), reinterpret_cast<char *>(&net_score), reinterpret_cast<char *>(&net_score) + 4);
    }

    return out;
}