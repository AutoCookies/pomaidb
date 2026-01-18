# Pomai VectorDB: Kế hoạch mở rộng hỗ trợ tensor đa chiều (General Tensor Support Roadmap)

## 1. Mục tiêu tổng quát

Hiện tại Pomai chỉ hỗ trợ vector 1 chiều (float32). Mục tiêu roadmap là:
- **Bước 1:** Hỗ trợ đầy đủ các loại số thực (float32, float64, ...), vector 1 chiều.
- **Bước 2:** Hỗ trợ lưu trữ & truy vấn tensor 2 chiều (ma trận), bao gồm thao tác theo chỉ số hàng/cột.
- **Bước 3:** Mở rộng kiến trúc để hỗ trợ tensor n chiều (n >= 1), lưu toàn bộ cấu trúc shape, loại dữ liệu, thao tác truy vấn linh hoạt.

---

## 2. Giai đoạn 1: Đa dạng kiểu dữ liệu số cho vector 1 chi���u (float32, float64, int...)

### 2.1. Phần giao diện API + Data Model
- [x] Cho phép `CREATE MEMBRANCE ... DATA_TYPE <float32|float64|int32|...>`, default là float32.
- [x] Lưu metadata data_type trong MembranceConfig, schema...
- [x] Khi INSERT/SEARCH, kiểm tra kiếu dữ liệu đúng type, từ chối khác schema.

### 2.2. Serialization + Storage
- [x] Thay đổi blob lưu vector: ghi đúng kiểu dữ liệu, không fix sizeof(float).
- [x] Tối ưu truy xuất, tính toán kernel theo type (float32 dùng SIMD, float64 fallback, int32 ...)

### 2.3. Core API
- [x] Kernel: viết hàm l2sq/dot/support cho float64/int32.
- [x] Thêm test cho từng loại type.
- [x] CLI và API trả về type & báo lỗi hợp lý.

---

## 3. Giai đoạn 2: Hỗ trợ rank-2 tensor (ma trận) (shape [d1, d2])

### 3.1. Schema mở rộng
- [ ] Cho phép `CREATE ... DIM <d1> [X <d2>] ...`, shape lưu là vector `[d1,d2]`.
- [ ] Metadata: Membrance lưu shape = {d1, d2} thay vì 1 số duy nhất.
- [ ] INSERT kiểm tra shape data.

### 3.2. Lưu trữ
- [ ] Blob lưu dưới dạng liên tiếp:
    - float32[d1*d2] (row-major hoặc col-major, cần chuẩn hóa format)
- [ ] Search API: khai báo/truy xuất với tensor 2 chiều.

### 3.3. Query/Operation
- [ ] Hỗ trợ basic operation: select index (row/col slice), flatten query, transpose (nếu cần).
- [ ] Tối ưu toán tử: tính toán l2sq, dot cho 2 ma trận (Nếu cần).

---

## 4. Giai đoạn 3: Hỗ trợ tensor n chiều (n >= 1) tổng quát

### 4.1. Mở rộng schema & metadata
- [ ] Cho phép `CREATE ... SHAPE (<dim1>,<dim2>,...,<dimN>)` hoặc DIM nhiều chiều trong câu lệnh tạo membrance.
- [ ] Lưu trữ shape là std::vector<size_t> hoặc array fixed/variable.
- [ ] Validate shape lúc insert/get/search.

### 4.2. Lưu trữ & API
- [ ] Blob lưu dữ liệu flatten (row-major).
- [ ] Metadata: mỗi vector phải có shape đầy đủ.
- [ ] Khi truy vấn, trả về shape đúng.
- [ ] Dataset inspector/trình debug trả về/hiển thị tensor dạng gọn (slice, shape...).

### 4.3. Toán tử & truy vấn
- [ ] Lệnh SEARCH chấp nhận dựa trên tensor rank-n (ex: slice, flatten tự động, reshape, broadcast).
- [ ] Khi query/insert, tự động flatten/unflatten đúng shape.
- [ ] Hỗ trợ optional slicing hoặc chỉ số (python-like), ex: GET ... SLICE [chỉ số].
- [ ] Nếu cần, kernel vector tự điều chỉnh rank (cảnh báo nếu phép toán không được hỗ trợ).

---

## 5. Kiểm thử & backward compatibility
- [ ] Đảm bảo config/schema/IO cũ load được (auto infer float32+shape=[d])
- [ ] Test old API các loại, auto chuyển sang schema mới nếu chỉ có dim 1.

---

## 6. CLI & Tooling
- [ ] CLI hiện shape, data_type.
- [ ] Cho phép import/export numpy/pytorch các tensor đa chiều.
- [ ] Check validate/báo lỗi giúp debug khi shape/type không khớp.

---

## 7. Tài liệu, ví dụ mẫu

- [ ] Hướng dẫn tạo bảng lưu trữ vector, matrix, tensor.
- [ ] Ví dụ: lưu ảnh RGB `[3,224,224]`, get/truy vấn shape, insert sample, search.
- [ ] Liệt kê giới hạn từng phiên bản.

---

## 8. Lưu ý kỹ thuật

- Khi flatten cần ghi rõ thứ tự shape (row-major/col-major quy chuẩn, ghi vào metadata).
- Cần thận trọng việc mismatch giữa dữ liệu lưu & type engine/kết nối client.
- Nhớ tính padding/alignment trong storage và APIs.

---

## 9. Roadmap lộ trình
- [ ] 1.0: Full vector 1D, multi-type.
- [ ] 1.1: Matrix 2D, search dạng flatten, insert/check shape.
- [ ] 2.0: General n-D tensor, API insert/query/shape.
- [ ] 2.1: Basic tensor ops/slice trực tiếp trên server.
- [ ] 3.0: Tích hợp native numpy/tensorflow/pytorch client/tool.

---

> Nếu bạn muốn ưu tiên phần nào (ví dụ: float64 trước, hay tensor 2D cho ảnh), checklist có thể phân theo milestone cụ thể.