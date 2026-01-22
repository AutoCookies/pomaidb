/*
 * src/ai/whispergrain.cc - Đã sửa lỗi đứng im EMA
 */

#include "src/ai/whispergrain.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <atomic>
#include <iostream>

namespace pomai::ai
{
    WhisperGrain::WhisperGrain(const pomai::config::WhisperConfig &cfg)
        : cfg_(cfg), latency_ema_(-1.0f), cpu_load_(0.0f)
    {
    }

    void WhisperGrain::observe_latency(float latency_ms)
    {
        if (latency_ms < 0.001f)
            latency_ms = 0.001f;

        float alpha = cfg_.latency_ema_alpha;
        // Lấy giá trị cũ với memory_order_acquire để đồng bộ giữa các luồng search
        float old_val = latency_ema_.load(std::memory_order_acquire);

        // [FIX]: Nếu là lần đầu hoặc EMA đang bị âm/không hợp lệ
        if (old_val <= 0.0f)
        {
            latency_ema_.store(latency_ms, std::memory_order_release);
            return;
        }

        float new_val = alpha * latency_ms + (1.0f - alpha) * old_val;
        latency_ema_.store(new_val, std::memory_order_release);
    }

    void WhisperGrain::set_cpu_load(float cpu_percent)
    {
        cpu_load_.store(cpu_percent, std::memory_order_relaxed);
    }

    Budget WhisperGrain::compute_budget(bool is_hot) const
    {
        Budget b;
        float ema = latency_ema_.load(std::memory_order_acquire);
        float cpu = cpu_load_.load(std::memory_order_relaxed);

        // 1. Base Budget (Mặc định 5000 ops)
        float base = static_cast<float>(cfg_.base_budget_ops);

        // 2. Latency Feedback Control
        // Handle uninitialized state (ema < 0 hoặc 0) -> dùng latency target làm baseline
        if (ema <= 0.0f)
            ema = cfg_.latency_target_ms;

        float scale = 1.0f;
        float target = cfg_.latency_target_ms;

        if (ema > target)
        {
            // [HARD LIMIT]: Nếu quá chậm, siết budget theo tỷ lệ bình phương
            scale = std::sqrt(target / ema);
        }
        else
        {
            // Hệ thống rảnh rỗi -> Cho phép bung sức mạnh (Headroom = 1.2x)
            scale = cfg_.budget_headroom;
        }

        // 3. CPU Safety Rail (Thermal Control)
        // Haswell/Kaby Lake trên ProBook rất nóng, cần siết mạnh khi CPU > 90%
        if (cpu >= cfg_.cpu_hard_threshold)
            scale *= 0.25f;
        else if (cpu >= cfg_.cpu_soft_threshold)
            scale *= 0.6f;

        // 4. Final OPS Calculation
        float ops = base * scale;

        // Ép sàn (Min QoS) để không bao giờ dừng hẳn tìm kiếm
        float min_ops = static_cast<float>(is_hot ? cfg_.hot_query_floor : cfg_.min_budget_ops);
        if (ops < min_ops)
            ops = min_ops;

        // Ép trần (Max Burst)
        float max_ops = base * cfg_.budget_headroom;
        if (ops > max_ops)
            ops = max_ops;

        b.ops_budget = static_cast<uint32_t>(ops);

        // 5. Spatial Budget (Số lượng centroids/hops tối đa)
        // Tỷ lệ vàng: 1 hop tốn tương đương 50-100 OPS (do SIMD scan bucket)
        // Chúng ta map budget ops sang bucket budget để EchoGraph.auto_navigate thực thi
        constexpr uint32_t OPS_PER_BUCKET = 500;
        b.bucket_budget = std::max<uint32_t>(16, b.ops_budget / OPS_PER_BUCKET);

        // 6. Refine Policy (Chỉ cho phép đọc SSD nếu máy thực sự rảnh)
        b.allow_exact_refine = (ema < (target - static_cast<float>(cfg_.refine_enable_margin_ms)) &&
                                cpu < cfg_.cpu_soft_threshold);

        return b;
    }
} // namespace pomai::ai