# TODO: Kiểm tra và củng cố thực thi đúng "Memory-Mapped Architecture" của Pomai

## 1. Đảm bảo Storage Path đúng triết lý "Everything is File (as RAM)"

- [x] Blob files cho từng Shard (`shard_0.blob`, ...) được tạo dưới `data_root`
- [ ] Tạo rõ ràng các file metadata (`pomai_schema.bin`/`manifest`, label map)
- [x] WAL (`wal.log`) đang tồn tại cho durability

## 2. Quy trình lưu trữ (Insert Path)

- [ ] Khi insert, vector/data _được ghi tuần tự vào WAL trước_ (có đảm bảo fsync trước trả thành công)
- [ ] Sau khi commit WAL => cấp phát Arena cho vector bằng offset trong blob file
- [x] Arenas sử dụng `mmap`/`ftruncate` để mở rộng file vật lý, trả về pointer ánh xạ vào vùng RAM
    - [`ShardArena`](src/memory/shard_arena.h/.cc) phải dùng đúng mmap, offset logic
- [ ] Lưu pointer dạng "offset" (relative, pointer swizzling) trong mọi index trên RAM thay vì raw pointer address

## 3. Truy xuất (Access Path)

- [x] Khi truy xuất vector qua offset, code trả về đúng `ptr = base + offset`, OS tự đưa về RAM nếu thiếu (page fault)
- [ ] Test di chuyển file `.blob` từ máy này sang máy khác, data vẫn access được

## 4. Đảm bảo Arena và ShardArena không bị memcpy double/triple

- [ ] Các API trả về chỉ pointer ánh xạ trực tiếp, không tạo buffer tạm
- [ ] Khi cần force-write to disk: sử dụng đúng `msync()` hoặc rely vào OS's background flush

## 5. Tối ưu Async Flush/Demotion

- [ ] `bg_worker` thread hoặc `async_demote_range()` trigger đúng thời điểm để giải phóng RAM cho trang ít dùng (madvise(MADV_DONTNEED))
- [ ] Balance latency <-> throughput: async flush không làm block main mutator

## 6. Kiểm tra các vùng "Zero-Copy"

- [ ] Buffer trả về cho vectordata, khi đọc/ghi, là zero-copy
- [x] Xem lại logic decode/encode trong HotTier/ShardArena có dùng memcpy không cần thiết không

## 7. Đảm bảo manifest/schema cập nhật khi cấu trúc file thay đổi

- [ ] Khi thay đổi chiều hoặc định dạng (float32/float16), manifest/schema được cập nhật sync

## 8. Bảo vệ WAL/Consistency on Crash

- [ ] Boot lại server, code sẽ replay WAL và khôi phục lại được trạng thái RAM giống như trước crash

---

## 📌 Checklist thực tế trong code Pomai hiện tại:

- [x] Có ShardArena dùng mmap, offset chỉ tới blob file
- [x] WAL đã có ghi tuần tự, mở lại replay
- [x] Insert truy xuất blob qua offset
- [x] Chưa có double memcpy (nhìn code)
- [ ] Cần code tường minh msync/madvise cho trang lạnh
- [ ] TODO: Tạo battery test copy .blob sang máy khác

---

**Nếu còn mục nào trên chưa tick được → cần bổ sung! Để framework Pomai hội đủ "Memory-Mapped VectorDB" thực thụ.**
