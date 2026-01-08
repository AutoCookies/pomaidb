 Đảm bảo PomaiArena được tạo đủ to (config POMAI_ARENA_MB).
 Khi init VectorStore/PPSM, dùng hàm estimate_max_elements để chọn max_elements_total hợp lý.
 Giảm M (8) và ef_construction (50).
 Kích hoạt PPPQ/IVFPQ nếu cần nén lớn hơn.
 Theo dõi sh.full log; nếu vẫn xuất hiện, trigger startShardResize với new_per_shard lớn hơn (nhưng tránh resize quá nhỏ).
 Theo dõi free RAM trước/sau khi khởi tạo index.