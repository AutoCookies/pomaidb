#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <cassert>

#include "src/core/cpu_kernels.h"
#include "src/ai/eternalecho_quantizer.h"
#include "src/ai/pomai_orbit.h"
#include "src/memory/arena.h"

// Helper: Tính Cosine Similarity
float cosine_similarity(const float* a, const float* b, size_t dim) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
}

void test_quantizer_quality() {
    std::cout << "\n=== TEST 1: ETERNAL ECHO QUANTIZER QUALITY ===\n";
    size_t dim = 512;
    
    // Config chuẩn (giống server)
    pomai::ai::EternalEchoConfig cfg;
    cfg.bits_per_layer = {96, 64, 48, 32, 16}; // 256 bits total
    cfg.quantize_scales = true;

    pomai::ai::EternalEchoQuantizer eeq(dim, cfg);

    // Tạo vector ngẫu nhiên
    std::vector<float> original(dim);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for(auto& x : original) x = dist(rng);

    // 1. Encode
    auto code = eeq.encode(original.data());
    
    // 2. Decode
    std::vector<float> reconstructed(dim);
    eeq.decode(code, reconstructed.data());

    // 3. So sánh
    float sim = cosine_similarity(original.data(), reconstructed.data(), dim);
    
    std::cout << "Original Norm: " << std::sqrt(pomai_dot_kernel(original.data(), original.data(), dim)) << "\n";
    std::cout << "Recon Norm   : " << std::sqrt(pomai_dot_kernel(reconstructed.data(), reconstructed.data(), dim)) << "\n";
    std::cout << "Cosine Sim   : " << sim << "\n";

    if (sim > 0.85) std::cout << "✅ KẾT QUẢ: Thuật toán nén TỐT.\n";
    else std::cout << "❌ KẾT QUẢ: Thuật toán nén làm HỎNG dữ liệu.\n";
}

void test_orbit_integrity() {
    std::cout << "\n=== TEST 2: ORBIT STORAGE INTEGRITY (Insert -> Get) ===\n";
    
    // Setup Orbit với Memory Arena ảo
    pomai::ai::orbit::PomaiOrbit::Config cfg;
    cfg.dim = 512;
    cfg.data_path = "./data/test_sanity";
    cfg.use_cortex = false;
    
    // Arena 64MB
    auto arena = pomai::memory::PomaiArena::FromMB(64);
    if (!arena.is_valid()) {
        std::cerr << "Alloc arena failed\n"; 
        return;
    }
    
    pomai::ai::orbit::PomaiOrbit orbit(cfg, &arena);

    // Tạo Data
    std::vector<float> vec(512);
    for(int i=0; i<512; ++i) vec[i] = (i % 2 == 0) ? 1.0f : -1.0f;

    uint64_t id = 12345;
    
    // 1. Insert
    bool ok = orbit.insert(vec.data(), id);
    if (!ok) { std::cout << "❌ Insert thất bại\n"; return; }

    // 2. Get
    std::vector<float> out_vec;
    bool found = orbit.get(id, out_vec);
    
    if (!found) {
        std::cout << "❌ Get không tìm thấy ID vừa insert!\n";
    } else {
        float sim = cosine_similarity(vec.data(), out_vec.data(), 512);
        std::cout << "Retrieved ID : " << id << "\n";
        std::cout << "Similarity   : " << sim << "\n";
        
        if (sim > 0.85) std::cout << "✅ KẾT QUẢ: Database lưu trữ OK.\n";
        else std::cout << "❌ KẾT QUẢ: Dữ liệu bị biến dạng khi lưu xuống Bucket.\n";
    }
}

int main() {
    // Init CPU kernels first!
    pomai_init_cpu_kernels();
    
    test_quantizer_quality();
    test_orbit_integrity();
    return 0;
}