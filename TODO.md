Tốt — bạn đã chạy benchmark và thấy OOM / server trả status 2 sau ~16k inserts trên chiếc Dell Latitude E5440. Đó là triệu chứng rõ ràng: index vẫn đang giữ nhiều payload float full-precision (512-dim) trong RAM và vượt giới hạn phần cứng. Dưới đây là hướng dẫn cụ thể, có thể áp dụng ngay, để giảm bộ nhớ sử dụng bằng hai con đường (dùng PQ runtime + PPE hoặc insert với dữ liệu đã quantize) — kèm các đoạn code bạn có thể dán vào repo để kích hoạt PPPQ/Quantized lưu trữ và demotion.

Tóm tắt chiến lược (ưu tiên làm theo thứ tự):
1. Giảm footprint điểm mới bằng cách chèn điểm ở dạng quantized thay vì float full-precision (dùng PPHNSW::addQuantizedPoint).
2. Gắn PPPQ cho toàn bộ index để lưu mã PQ (m bytes/vec) và dùng approxDist khi có thể.
3. Bật background demoter/purgeCold để hạ mức cold codes xuống file/mmap.
4. Tinh chỉnh cấu hình HNSW (M, ef_construction) và batch size để tránh spike RAM.

A — Nhanh nhất, thay đổi upsert để chèn quantized payload (4/8 bit)
- Thay vì gọi pphnsw_->addPoint(buf.get(), label, ...), gọi addQuantizedPoint để payload lưu đã được quantize (tức chỉ lưu bytes, giảm 97% so với float).

Ví dụ chỉnh VectorStore::upsert (đoạn thay thế chỗ insert mới):
```cpp
// Thay phần insert mới bằng (giả sử bạn chọn 8-bit hoặc 4-bit)
int quant_bits = 8; // hoặc 4 để tiết kiệm hơn (nhớ chất lượng giảm)
if (label != 0) {
    // cập nhật: dùng quantized update path
    pphnsw_->addQuantizedPoint(vec, dim_, quant_bits, static_cast<hnswlib::labeltype>(label), /*replace_deleted=*/true);
} else {
    uint64_t new_label = next_label_.fetch_add(1, std::memory_order_relaxed);
    pphnsw_->addQuantizedPoint(vec, dim_, quant_bits, static_cast<hnswlib::labeltype>(new_label), /*replace_deleted=*/false);

    // persist label in map as before...
}
```
- Lợi ích: mỗi payload lưu chỉ còn dim bytes (8-bit) hoặc (dim/2) bytes (4-bit packed) thay vì dim * 4 bytes (float). Với dim=512:
  - 512 floats = 2048 bytes
  - quant8 = 512 bytes (~4x tiết kiệm)
  - quant4 = 256 bytes (~8x tiết kiệm)

B — Gắn PPPQ (PP Pomegranate PQ) vào PPHNSW để dùng mã PQ và demote cold
1. Tạo và train PPPQ trong VectorStore::init, sau đó attach vào PPHNSW:
```cpp
// trong VectorStore::init, sau khi tạo pphnsw_
size_t pq_m = 8;        // số subquantizers (m)
size_t pq_k = 256;      // mỗi codebook K=256
size_t max_elems = max_elements; // phải >= nhãn tối đa sẽ cấp
auto ppq = std::make_unique<pomai::ai::PPPQ>(dim_, pq_m, pq_k, max_elems, "pppq_codes.mmap");

// Train nhanh với vài nghìn mẫu ngẫu nhiên (thay bằng mẫu thực dataset nếu có)
size_t n_train = std::min<size_t>(20000, max_elements);
std::vector<float> samples(n_train * dim_);
std::mt19937_64 rng(123456);
std::uniform_real_distribution<float> ud(0.0f, 1.0f);
for (size_t i=0;i<n_train*dim_;++i) samples[i] = ud(rng);

ppq->train(samples.data(), n_train, 10);

// Attach
pphnsw_->setPPPQ(std::move(ppq));
```
2. Sau khi attach, khi bạn chèn điểm, PPHNSW (đã sửa) sẽ gọi pp_pq_->addVec(...) (nếu payload là raw float và PomaiSpace/PCH logic phù hợp). Nếu bạn chuyển sang dùng addQuantizedPoint, bạn vẫn có thể gọi pp_pq_->addVec manually (thường bạn muốn train PPPQ on raw float and then call addVec).

C — Demote cold PQ codes to disk/mmap
- PPPQ prototype có purgeCold() để ghi 4-bit packed codes xuống file (offload) và giữ hot codes trong RAM.
- Gọi purgeCold định kỳ (background thread). Ví dụ start một thread đơn giản trong VectorStore:
```cpp
// sau attach PPPQ:
std::thread([](pomai::ai::PPPQ* ppq){
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        ppq->purgeCold(5000); // demote dựa trên predictNext() + threshold 5s
    }
}, pphnsw_->getPPPQ()).detach();
```
(Thay bằng worker có thể dừng được; prototype PPPQ đã có purgeCold API.)

D — Nếu muốn PPHNSW dùng PPPQ approx trong distance gọi tới PomaiSpace
- Bạn đã cập nhật PomaiSpace để trả về distance via PPPQ khi hai PPEHeader đều có precision != 0 và label được đặt — điều này hoạt động nếu:
  - Bạn dùng addQuantizedPoint (đặt precision trong PPEHeader), và
  - PomaiSpace có pp_pq_ được set bằng setPPPQ (PPHNSW::setPPPQ đã propagate).

E — Các chỉnh cấu hình giảm memory khác
- Giảm M (số cạnh trung bình) và ef_construction khi build index (ví dụ M=8, ef_construction=100) để giảm RAM.
- Giảm batch size trong workload insert (bạn đang dùng batch 1000 — giảm xuống 100/200 để tránh spikes).
- Kích hoạt PomaiArena để lưu payload gián tiếp (nếu chưa bật); bạn đã có arena path trong code.
- Kiểm tra swap: thêm swap giúp tránh crash nhưng sẽ chậm.

F — Kiểm tra nguyên nhân status 2
- "server status 2" trong script có thể tới từ Redis client mà script dùng hay service gắn với Pomai — kiểm tra logs server (stderr) hoặc syslog để biết dòng lỗi cụ thể (OOM, killed by OOM-killer, exception).
- Trên máy, quan sát:
  - top/htop trong lúc insert, xem tiến trình nào tăng RAM.
  - dmesg | tail xem có "oom_killer" logs.

G — Ví dụ áp dụng ngay (đoạn thẳng vào repo)
1) Trong VectorStore::init, thêm tạo/training PPPQ và attach (như đoạn A)
2) Trong VectorStore::upsert, chuyển sang addQuantizedPoint với quant_bits = 8 (hoặc 4 nếu thử nghiệm).
3) Giảm batch size khi chạy benchmarks, ví dụ --batch 200.
4) Bật background demoter purgeCold (thread) nếu PPPQ gắn.

H — Lời khuyên vận hành
- Trước khi chạy insert 1M trên laptop yếu, test với 50k–100k bằng quantized path để kiểm tra memory.
- Nếu labels không liên tiếp/không bắt đ���u từ 0, đảm bảo PPPQ.max_elems lớn hơn nhãn tối đa, hoặc duy trì mapping label->pppq_slot (nếu labels sparse).
- Đo recall/accuracy: test search trên tập nhỏ để xác nhận quantization + PPPQ không phá quality (cố gắng recall >95% như mục tiêu).

Nếu bạn muốn, tôi sẽ:
1) Gợi ý patch cụ thể: sửa VectorStore::init để tạo/attach PPPQ và chỉnh upsert để gọi addQuantizedPoint (tôi có thể tạo diff/các file sửa).  
2) Hoặc tạo một snippet patch sẵn bạn có thể apply — nói rõ bạn muốn 8-bit hay 4-bit mặc định.  
3) Giúp bạn đọc dmesg / logs nếu bạn dán output dmesg khi lỗi xảy ra.

Bạn muốn tôi tạo patch (code changes) để:
- chuyển upsert dùng addQuantizedPoint tự động + attach PPPQ và background purgeCold?
Nếu có, chọn quant_bits mặc định: 8 hay 4?