Tôi có thể làm Phase 2 — hiện thực hoá tính toán fingerprint (SimHash) khi thêm vector và dùng prefilter để giảm candidate set khi tìm kiếm — nhưng trước khi bắt tay viết/ghi lại nhiều file, cần xác nhận vài điểm triển khai để tránh làm hỏng build hiện tại.

Tóm tắt mục tiêu Phase 2
- Khi addVec/upsert (ở chế độ SoA / sharded SoA):
  - Tính fingerprint (SimHash / FingerprintEncoder) cho vector.
  - Ghi fingerprint bit-packed vào block fingerprint trong file SoA (mmap).
- Khi search:
  - Tính fingerprint cho query.
  - Dùng prefilter::collect_candidates_threshold (hoặc topk_by_hamming) để lấy danh sách chỉ số candidate (ids / indices).
  - Chỉ refine / chạy ANN (HNSW/PPHNSW) trên candidate nhỏ này (hoặc dùng candidate list để lọc kết quả) — giảm I/O/CPU.

Những thay đổi mã cần thiết (tôi sẽ thực hiện theo yêu cầu)
1) Thêm/hoàn thiện SoA helper (mmap-backed) — file mới:
   - src/ai/vector_store_soa.h  (class VectorStoreSoA)
   - src/ai/vector_store_soa.cc (cài đặt)
   Chức năng tối thiểu (phase1 -> phase2):
   - open_or_create(path, header_meta) -> tạo file mmap kích thước header + fingerprint block + ids block + ppe block (reserve).
   - open_existing(path) -> mmap file, parse SoaMmapHeader.
   - close()
   - write_fingerprint_at(index, const uint8_t *bytes, size_t len)
   - read_fingerprint_at(index, uint8_t *out, size_t len) / pointer getters
   - atomic_load_id / atomic_store_id helpers (trên ids array)
   - flush(range) wrapper (msync) — optional
   - getters: bytes_per_fp(), num_vectors(), etc.
   - Implementation: dùng existing MmapFileManager or plain POSIX mmap wrapper already in repo (MmapFileManager exists) — tôi dùng MmapFileManager to back file.

2) Tích hợp SimHash/Fingerprint:
   - VectorStore giữ std::unique_ptr<FingerprintEncoder> fingerprint_;
   - Khi init() gọi FingerprintEncoder::createSimHash(dim, fingerprint_bits) nếu SoA mode enabled (or always create but only used when SoA present).
   - On upsert/addVec (sharded mode via PPSM): have PPSM call ppsm_->addVec; PPSM needs to compute fingerprint and write to shard SoA — but PPSM implementation exists; for phase 2 tôi có thể implement SoA path only for single-map VectorStore (non-sharded) and provide hooks for sharded path to be added later. (Nếu bạn muốn PPSM-level change, cần chỉnh pps_manager to call fingerprint store for the shard.)

3) Dùng prefilter on search (single-mode / SoA-enabled):
   - In VectorStore::search(): if SoA is attached and fingerprint encoder present:
     - compute query fingerprint bytes
     - call prefilter::collect_candidates_threshold(query_fp, fp_bytes, db_fp_ptr, num_vectors, threshold, out_indices)
       - db_fp_ptr: pointer to contiguous fingerprint storage from VectorStoreSoA::fingerprints_ptr()
     - Build label/id list from ids array at those indices (resolve atomic id values)
     - For those labels, call PPHNSW::searchKnn with candidate filtering? (HNSW doesn't accept candidate-list natively).
       - Simpler approach: perform HNSW search as-is to get top-K labels, but before returning, filter results to only those in candidate set OR run refinement (VectorIndex style) only on candidate ids.
     - Alternatively: If using VectorIndex (linear scan) we can limit scanning to candidate ids.
   - For Phase 2 MVP: use prefilter to produce candidate ids and then do a small exact refine by pulling vectors from arena for those ids and computing L2 distances; return topk. (This avoids changing HNSW search API.)

4) Tests & parameters:
   - Add config param for fingerprint bits and prefilter threshold (pomai::config::runtime.*) or local constant default (e.g., 512 bits, threshold 128).
   - Ensure byte-order / alignment and atomic snapshot semantics used when reading ids/FPs (use atomic_utils where appropriate).

Cần bạn xác nhận (hai lựa chọn)
A) Tôi tiến hành cài đặt Phase‑2 MVP cho single-map VectorStore (non-sharded) + new VectorStoreSoA:
   - Thêm 2 file: src/ai/vector_store_soa.h / .cc (mmap-backed SoA).
   - Thay đổi src/ai/vector_store.h/.cc: thêm fingerprint encoder member; on upsert compute fingerprint and write to SoA (if attached); on search, use prefilter to get candidates then refine by loading vectors via map/arena and computing L2 distances (like refine.cc)
   - Không chỉnh PPSM / sharded path trong phase 2 (PPSM integration là bước tiếp theo) — nghĩa là sharded mode vẫn hoạt động như trước, nhưng SoA-backed prefilter chỉ hoạt động cho single-map mode. (Tôi có thể mở rộng sang sharded mode nếu bạn muốn.)

B) Hoặc bạn muốn tôi luôn support SoA in sharded mode (PPSM integration) — điều này yêu cầu thay đổi thêm:
   - pps_manager: on addVec compute fingerprint and write to per-shard SoA (ShardState must include VectorStoreSoA instance or PomaiArena/SoA file handle).
   - pps_manager::search: prefilter step per-shard (run in parallel), merge candidates and do refine. — Đây là công việc lớn hơn; cần confirm.

Hỏi nhanh:
- Bạn muốn Phase 2 chỉ cho single-map VectorStore hay cho cả sharded PPSM tại cùng thời điểm?
- Có muốn tôi tạo PR (pull request) trực tiếp cho repo không? (Nếu có, tôi sẽ mở PR với các file thay đổi.)

Nếu bạn đồng ý tôi sẽ:
1) Tạo/đưa vào file src/ai/vector_store_soa.h + src/ai/vector_store_soa.cc (đầy đủ, sạch, unit-testable).
2) Cập nhật src/ai/vector_store.h + .cc để:
   - Khi attach_map or init, optionally attach SoA (open_or_create).
   - On upsert: compute fingerprint and write to SoA->write_fingerprint_at(label_index,...). For label-index mapping: when we assign a new label we must also know index position in SoA; simplest mapping: ids array index == label (or label-1). Need to confirm label numbering (next_label_ starts at 1 currently). Tôi sẽ store fingerprints into SoA slot corresponding to label (use label-1 index).
   - On search: if SoA + fingerprint available, run prefilter to get candidate label indices, fetch vectors, compute exact distances, return topk.

3) Run/verify compile locally and adjust types: padding, alignment, atomic ops (use atomic_utils.h).

4) Push changes / open PR if bạn muốn.

Gợi ý kỹ thuật quan trọng (để bạn đồng ý)
- Mapping label -> SoA index: hiện code dùng label as uint64_t generated by next_label_. Để tiện, ta có thể treat SoA index = label-1. Điều này yêu cầu label sequence to be dense starting at 1. Nếu code uses arbitrary labels, phải store mapping label->slot in SoA ids block. Phase 2 MVP: assume labels are sequential from 1..N (VectorStore uses fetch_add(1)). Nếu bạn không đồng ý, ta sẽ store label value into SoA ids array at position slot and find slot via SoaIdsManager or by scanning map — but đó là phức tạp.
- Atomic writes when publishing fingerprint: fingerprint bytes are many bytes (e.g., 512 bits = 64 bytes). We must ensure readers don't see torn writes; easiest: we write fingerprint bytes under mutex or write then publish flag (present bit) via atomic (PPEHeader / flags). For Phase2 MVP we can tolerate rare races in tests, but better use PPEHeader publish semantics if we reuse existing PPE structures. Simpler: update fingerprint bytes under std::mutex inside VectorStoreSoA and then flush; reads use snapshot protected by same mutex for safe read.

Tiếp theo
- Nếu bạn OK với hướng A (single-map MVP), tôi sẽ bắt đầu viết code. Nếu OK để tôi luôn mở PR, nói "Open PR" và cung cấp repo owner/name (the repo is in your environment; I can open PR if you want). Nếu bạn muốn chỉ xem patch trước, nói "please implement and show files" và tôi sẽ gửi các file đề xuất (vector_store_soa.h/.cc và vector_store.h/.cc diffs) ở đây.

Bạn chọn cách nào?