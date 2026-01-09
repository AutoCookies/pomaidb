# TODO: Atomicize PPEHeader payload & flags (audit + changes)

Mục tiêu
--------
Đảm bảo tất cả các call-sites ghi/đọc payload nằm ngay sau `PPEHeader` (offset `dst + sizeof(PPEHeader)`) và mọi cập nhật `flags` được thực hiện một cách atomic để tránh torn reads khi demote/promote xảy ra đồng thời.

Checklist (thực hiện theo thứ tự)
----------------------------------

A. Audit (tìm tất cả call sites cần sửa)
- [ ] Tìm mọi chỗ trong repo có chuỗi `+ sizeof(PPEHeader)` hoặc `dst + sizeof(PPEHeader)` (các write payload).

src/ai/ppe.h (định nghĩa PPEHeader, đã có helpers atomic_set/clear/load)

src/ai/pomai_hnsw.cc (rất nhiều chỗ: addPoint, addQuantizedPoint, restorePPEHeaders, background demoter — viết payload tại (dst + sizeof(PPEHeader)) và cập nhật flags)

src/ai/pomai_hnsw.h (declarations; API liên quan)

src/ai/pomai_space.h (đọc PPEHeader trong distance_impl)

src/ai/restore/… (không có file khác rõ ràng trong đống code bạn gửi, nhưng kiểm tra thêm nếu repo có TU khác)
(Lưu ý:) Một số file liên quan đến PPE nhưng không phải PPEHeader trực tiếp: src/ai/ppe_array.h, src/ai/ppe_predictor.h, src/ai/pppq.cc dùng predictor — không cần thay đổi atomic flags cho PPEEntry / PPEPredictor ở đây vì chúng dùng atomic riêng.

- [ ] Tìm mọi chỗ thao tác `flags` trực tiếp (ví dụ `flags |= ...`, `flags &= ~...`, hoặc đọc `flags` bằng truy cập non-atomic).
- [ ] Tập hợp danh sách file + line ranges + ngữ cảnh (copy/paste đoạn code) để review.

B. Thay thế code (implement)
- [ ] Thay mọi write payload (8 byte) tại `dst + sizeof(PPEHeader)` bằng:
      `pomai::ai::atomic_utils::atomic_store_u64(reinterpret_cast<uint64_t *>(dst + sizeof(PPEHeader)), value);`
- [ ] Thay mọi cập nhật flag non-atomic bằng:
      - `h->atomic_set_flags(PPE_FLAG_INDIRECT | PPE_FLAG_REMOTE);`  (hoặc tương ứng)
      - `h->atomic_clear_flags(PPE_FLAG_REMOTE);`
      - Đảm bảo gọi atomic_set/clear sau khi đã atomic_store payload theo thứ tự cần thiết.
- [ ] Thay mọi đọc flag non-atomic bằng `uint32_t flags = h->atomic_load_flags();`
- [ ] Thay mọi đọc payload non-atomic bằng `uint64_t payload = pomai::ai::atomic_utils::atomic_load_u64(reinterpret_cast<const uint64_t *>(dst + sizeof(PPEHeader)));`
- [ ] Khi cần thực hiện nhiều flag-bit thay đổi có thể tranh chấp: sử dụng kết hợp atomic_set_flags / atomic_clear_flags; nếu cần kết hợp read-modify-store phức tạp, dùng compare-exchange loop trên toàn 32-bit flags (tạo helper nếu cần).

C. Các file cần sửa ngay (ưu tiên)
- src/ai/pomai_hnsw.cc
  - addPoint: lưu payload -> set INDIRECT / INDICATE REMOTE
  - addQuantizedPoint: tương tự
  - restorePPEHeaders: atomic store payload + atomic_set_flags
  - background demoter worker: promotion/demotion path (atomic loads + atomic stores + atomic_set/clear)
- src/ai/pomai_space.h
  - Reader distance_impl: nếu cần đọc flags/payload (hiện đang chỉ đọc label/precision) — nếu thêm logic đọc payload, dùng atomic helpers
- So sánh & test:
  - [ ] Viết unit-test mô phỏng concurrent reader (thread A đọc flags/payload) và writer (thread B promote/demote) để check không xảy ra torn reads (đặc biệt trên x86_64).
  - [ ] Kiểm tra build trên toolchain hỗ trợ __cpp_lib_atomic_ref và không hỗ trợ (fallback volatile path).

D. Runtime & Compatibility notes
- C++20 `std::atomic_ref` tốt — helper hiện dùng `__cpp_lib_atomic_ref` detection; trên toolchains cũ helper fallback về GCC/Clang __atomic builtins hoặc volatile.
- Nếu bạn cần strong cross-process semantics (multiple processes updating same mmap), hiện tại giải pháp không đủ — cần IPC/fsync+WAL coordination. (SoaIdsManager đã có WAL flow; PPE payloads are in-index memory and intended single-process.)
- Performance: atomic stores cost rất thấp trên x86; overhead nhỏ so với IO/IO-bound demote/promote ops.

E. Testing / Validation
- [ ] Build with sanitizer (TSAN) and run demote/promote worker + concurrent reads to detect races.
- [ ] Add regression tests for demote/promote sequences.
- [ ] Run existing integration tests.

F. Deliverable patches
- For each modified file, include comment header noting change (atomicized PPE payload/flags) and link to this TODO.

Notes bổ sung
-------------
- Nếu gặp chỗ nào hiện đang dùng `memcpy(dst + sizeof(PPEHeader), &value, sizeof(value)); flags |= ...;` -> thay bằng atomic_store_u64 + atomic_set_flags theo thứ tự đề xuất.
- Nếu có code dùng `PPEHeader::flags` trực tiếp để snapshot, thay bằng atomic_load_flags().
