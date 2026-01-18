#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include "src/core/pomai_db.h"
#include "src/ai/whispergrain.h"

namespace pomai::server {

// Struct nhỏ để lưu trạng thái phiên làm việc của Client
struct ClientState {
    std::string current_membrance;
};

class SqlExecutor {
public:
    SqlExecutor();

    // Hàm thực thi chính: Nhận lệnh thô và trả về chuỗi kết quả
    std::string execute(pomai::core::PomaiDB* db, 
                        pomai::ai::WhisperGrain& whisper,
                        ClientState& state, 
                        const std::string& raw_cmd);

private:
    // Bản đồ tần suất truy vấn để WhisperGrain điều tiết (Hot-query detection)
    std::unordered_map<std::string, uint32_t> query_freq_;
    std::mutex freq_mu_;
};

} // namespace pomai::server