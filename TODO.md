## Lộ trình nâng PomaiDB từ single-thread lên multi-thread “true production-safe”  
**Mục tiêu: Đảm bảo không đổi logic — chỉ tách threading chuẩn hoá, không băm bổ code.**

---

### **Nguyên tắc xương sống**
- **Không đổi logic tài nguyên, quản lý, PomaiArena, Map, …**
- **Mỗi thread own 1 shard/arena riêng** (no cross-thread mutation)
- **Không cross-thread access vào hot-path data** (giữ invariant Pomai)
- **Chỉ lock khi cần cho IO/manifest/giao tiếp (không hotpath)**
- **Không mất debugability, không race condition**

---

## **Các bước cụ thể**

---

### **Bước 1: Kiểm kê Ownership và HOTPATH**

1. **Kiểm kê các class nào chạm tới Arena, Map, Shard, Dispatcher**
   - **Arena/Shard/Map:** Hot-path phải chỉ được gọi bởi đúng 1 thread
   - **Dispatcher/Orchestrator:** Phân phối work (search/insert) tới đúng thread (không cross)

**=> Không cần sửa logic, chỉ cần tách context per-thread**

---

### **Bước 2: Refactor Orchestrator sang Worker Thread Pool**

- Tạo **worker pool** N thread (theo số shard/cpu core)
- Mỗi worker:
  - Own 1 PomaiArena, PomaiMap, hot_tier, v.v. (không share pointer hoặc mutable data qua thread)
- Dispatcher (main thread hoặc server):
  - **chỉ nhận request network, rồi push xuống đúng worker qua queue** (ring buffer, MPMC queue, hoặc lockfree queue nếu muốn sạch)
    - Ví dụ: `struct Work { Request req; }`, mã hoá: search/insert/query...
    - Gửi xuống qua `std::queue<Work>` hoặc custom lock-free nếu muốn performance
- **Không cần đổi logic insert/search của Shard/PomaiArena**; logic vẫn như cũ, chỉ chuyển gọi từ main thread sang thread worker tương ứng.

---

### **Bước 3: Giao tiếp với Worker qua Queue (Thread-Safe, No Cross-Access)**

- Mỗi worker có queue nhận request từ main thread/dispatcher.
- Khi nhận request: worker xử lý như logic cũ, trả kết quả về qua response queue hoặc callback (không block main thread).
- **Không bao giờ access data của worker khác**.

**[Hình dung]**

```
┌─────────Server(Main)─────────┐
│ ┌─────Req─────┐ ┌─────Req────┐ │
│ │   Queue 0   │ │   Queue 1  │ │ ... (N)
│ └──────┬──────┘ └──────┬─────┘ │
│        │                │      │
└────────▼────────────────▼------┘
      ┌─────Worker 0──────┐
      │ - PomaiArena0     │
      │ - PomaiMap0       │
      └───────────────────┘

      ┌─────Worker 1──────┐
      │ - PomaiArena1     │
      │ - PomaiMap1       │
      └───────────────────┘
```

---

### **Bước 4: Xử lý Insert/Search theo Hash/Shard**

- Dispatcher hash label/id → queue, hoặc roundrobin nếu không hash.
- Không đổi logic insert/search; chỉ cần chuyển gọi từ main thành worker context.

---

### **Bước 5: IO/Snapshot/Async Task — Giữ hoặc refactor nếu chia sẻ file/manifest**

- Nếu IO/snapshot manifest là tài nguyên dùng chung (manifest file), giữ mutex chỉ ở đây!
- Arena demote/snapshot của mỗi worker CHỈ xử lý blob/file/arena của chính nó.
- Nếu có logic share manifest → lock chỗ này, còn lại BỎ lock ở Arena/PomaiMap/Shard.

---

### **Bước 6: Bỏ Mutex (Hot Path) Safe**

- Khi đã chia mỗi PomaiArena/PomaiMap cho đúng 1 thread và đảm bảo không ai access, **gỡ hết mutex khỏi hotpath trong Arena/Map**.
- Mutex cho queue (MPMC), hoặc IO, hoặc manifest — VẪN GIỮ nếu còn shared resource.
- TOÀN BỘ insert/search vector trong worker = lock-free, atomic nếu cần, không mutex!

---

### **Bước 7: Test, Validate, Stress**

- Viết test không đổi logic query/search, chỉ test thread-safety (thread sanitizer, stress, chaos monkey)
- Xác nhận không race, không lost update, không use-after-free.

---

## **Lợi ích/Trade-off/RAM**

- **Không đổi logic dữ liệu, không sửa insert/search/map/arena code, chỉ wrap lại trong worker threads**
- **RAM tiêu tốn nhẹ tăng nếu dùng queue buffer (không đáng kể)**
- **Debug dễ dàng hơn vì từng thread own context và data, không có cross-access.**
- **Performance tăng cực mạnh: mỗi thread tận dụng riêng 1 core/cpu, không contention, chỉ lock nhẹ phía network/queue**

---

## **Sơ đồ triển khai**
```
 Main Thread (Network Listener)
      │
      ├─> Dispatcher (Hash/Shard routing)
      ↓            ↓             ↓
 Worker 0    Worker 1       Worker N    ...  (Thread owns PomaiArena, PomaiMap)
 Arena/Map   Arena/Map      Arena/Map

 All insert/search are only called in assigned thread/shard context.
 Manifest/snapshot/IO only uses mutex where unavoidable. Hotpath NO LOCKS.
```

---

## **Notes cực kỳ production-safe**
- Mutex chỉ cần cho IO/manifest, không cần cho data của từng worker.
- Nếu có bug cross-access, lock lại chỗ đó, không lock hotpath.
- Không thay đổi structure vector/bucket/arena/map — chỉ wrap thread context.

---

**Kết luận:**  
- Đúng Pomai philosophy.
- Đúng production-safe, không tấu hài.
- Không đổi logic, chỉ wrap threading/queue/context.
- Mutex ở đâu không cần phải bỏ, chỉ remove nếu chắc chắn single-thread ownership.

Nếu cần, tôi sẽ draft code template/refactor rung cho worker pool + thread ownership. Debug path nào có thể crash nếu thiết kế dở, tôi sẽ chỉ mặt gọi tên cụ thể.