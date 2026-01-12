/* src/ai/holographic_scanner.h */
#pragma once

#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric> // Required for std::iota

// Fix: Include simhash.h so we can use pomai::ai::simhash::hamming_dist
#include "src/ai/simhash.h" 

#include "src/ai/vector_store_soa.h"
#include "src/ai/fingerprint.h"
#include "src/ai/pq.h"
#include "src/ai/candidate_collector.h"
#include "src/ai/prefilter.h"
#include "src/ai/pq_eval.h"
#include "src/core/config.h"

namespace pomai::ai
{

    class HolographicScanner
    {
    public:
        struct ScanResult
        {
            uint64_t id_entry; // ID/Label gốc
            float score;       // Khoảng cách xấp xỉ
        };

        // Hàm quét chính: chạy trên 1 Shard cụ thể
        static std::vector<ScanResult> scan_shard(
            const soa::VectorStoreSoA *soa,
            const FingerprintEncoder *fp_enc,
            const ProductQuantizer *pq,
            const float *query,
            size_t topk)
        {
            std::vector<ScanResult> results;

            // 1. Kiểm tra điều kiện tiên quyết
            if (!soa || !soa->is_valid() || !query)
                return results;

            size_t nv = soa->num_vectors();
            if (nv == 0)
                return results;

            // 2. Compute Query Fingerprint (cho SimHash)
            std::vector<uint8_t> qfp;
            if (fp_enc && soa->fingerprint_bits() > 0)
            {
                qfp.resize(fp_enc->bytes());
                fp_enc->compute(query, qfp.data());
            }

            // 3. Compute Query PQ Tables (cho PQ Distance)
            std::vector<float> pq_tables;
            bool use_pq = (pq != nullptr && soa->pq_m() > 0);
            if (use_pq)
            {
                pq_tables.resize(pq->m() * pq->k());
                pq->compute_distance_tables(query, pq_tables.data());
            }

            // 4. Prefilter (SimHash) - Lọc ứng viên
            std::vector<size_t> candidates;
            if (!qfp.empty())
            {
                // Lấy raw pointers từ SoA để quét cực nhanh
                // Lưu ý: api soa->fingerprint_ptr(i) trả về từng cái,
                // nhưng trong SoA block thường liền kề.
                // Để tối ưu, ta giả định block liên tục nếu offset != 0.
                
                // Cấu hình threshold từ runtime
                uint32_t thresh = pomai::config::runtime.prefilter_hamming_threshold;

                candidates.reserve(std::min<size_t>(nv, topk * 10)); // Heuristic

                for (size_t i = 0; i < nv; ++i)
                {
                    const uint8_t *fp = soa->fingerprint_ptr(i);
                    if (!fp)
                        continue; // Skip chưa publish

                    // Tính Hamming distance
                    // Requires #include "src/ai/simhash.h"
                    uint32_t ham = pomai::ai::simhash::hamming_dist(qfp.data(), fp, qfp.size());
                    if (ham <= thresh)
                    {
                        candidates.push_back(i);
                    }
                }
            }
            else
            {
                // Nếu không có fingerprint, candidates là toàn bộ
                candidates.resize(nv);
                std::iota(candidates.begin(), candidates.end(), 0);
            }

            // 5. PQ Scoring & Top-K Collection
            size_t Napprox = std::min<size_t>(candidates.size(), topk);
            CandidateCollector collector(Napprox); // Giữ top-N tốt nhất

            if (use_pq && !candidates.empty())
            {
                // Batch compute PQ distances
                // Gom packed codes
                size_t packed_bytes = ProductQuantizer::packed4BytesPerVec(pq->m());
                std::vector<uint8_t> batch_codes;
                batch_codes.reserve(candidates.size() * packed_bytes);

                for (size_t idx : candidates)
                {
                    const uint8_t *code = soa->pq_packed_ptr(idx);
                    if (code)
                    {
                        batch_codes.insert(batch_codes.end(), code, code + packed_bytes);
                    }
                    else
                    {
                        // Fallback nếu lỗi dữ liệu: chèn 0
                        batch_codes.insert(batch_codes.end(), packed_bytes, 0);
                    }
                }

                std::vector<float> dists(candidates.size());
                // Gọi hàm tính toán SIMD từ pq_eval.h
                pomai::ai::pq_approx_dist_batch_packed4(
                    pq_tables.data(), pq->m(), pq->k(),
                    batch_codes.data(), candidates.size(), dists.data());

                // Đưa vào Collector
                for (size_t i = 0; i < candidates.size(); ++i)
                {
                    collector.add(candidates[i], dists[i]);
                }
            }
            else
            {
                // Fallback nếu không có PQ: Trả về dummy score hoặc random (ít khi xảy ra nếu đã init đúng)
            }

            // 6. Format Result
            auto top_pairs = collector.topk(); // vector<pair<id, score>>
            results.reserve(top_pairs.size());
            for (const auto &p : top_pairs)
            {
                uint64_t internal_idx = p.first;
                float score = p.second;
                // Lấy ID Entry (Label) từ SoA
                uint64_t id_entry = soa->id_entry_at(internal_idx);
                results.push_back({id_entry, score});
            }

            return results;
        }
    };

} // namespace pomai::ai