#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <functional>
#include "src/core/pomai_db.h"
#include "src/ai/whispergrain.h"

namespace pomai::server
{

    struct ClientState
    {
        std::string current_membrance;
    };

    class SqlExecutor
    {
    public:
        SqlExecutor();

        std::string execute(pomai::core::PomaiDB *db,
                            pomai::ai::WhisperGrain &whisper,
                            ClientState &state,
                            const std::string &raw_cmd);

        [[nodiscard]] std::string execute_binary_insert(pomai::core::PomaiDB *db,
                                                        const char *raw_data,
                                                        size_t len);

        void set_insert_callback(std::function<bool(const std::string &membr, const std::vector<std::pair<uint64_t, std::vector<float>>> &batch)> cb);
        void set_search_callback(std::function<std::vector<std::pair<uint64_t, float>>(const std::string &membr, const std::vector<float> &q, size_t k)> cb);

    private:
        std::unordered_map<std::string, uint32_t> query_freq_;
        std::mutex freq_mu_;

        std::function<bool(const std::string &membr, const std::vector<std::pair<uint64_t, std::vector<float>>> &batch)> insert_cb_;
        std::function<std::vector<std::pair<uint64_t, float>>(const std::string &membr, const std::vector<float> &q, size_t k)> search_cb_;
    };

} // namespace pomai::server