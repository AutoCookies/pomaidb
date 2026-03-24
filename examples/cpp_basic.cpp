#include "pomai/pomai.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

int main() {
    // Remove any previous DB first for a clean run, e.g.: rm -rf /tmp/pomai_example_cpp
    const std::string path = "/tmp/pomai_example_cpp";
    const uint32_t dim = 8;
    const uint32_t shards = 1; // monolithic runtime; single shard is typical

    pomai::DBOptions opt;
    opt.path = path;
    opt.dim = dim;
    opt.shard_count = shards;
    opt.fsync = pomai::FsyncPolicy::kNever;

    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opt, &db);
    if (!st.ok()) {
        std::cerr << "Open failed: " << st.message() << "\n";
        return 1;
    }

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const uint32_t n = 100;
    std::vector<std::vector<float>> vectors;
    vectors.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        std::vector<float> v(dim);
        for (auto &x : v) x = dist(rng);
        vectors.push_back(v);
        st = db->Put(i, v);
        if (!st.ok()) {
            std::cerr << "Put failed: " << st.message() << "\n";
            return 1;
        }
    }
    st = db->Freeze("__default__");
    if (!st.ok()) {
        std::cerr << "Freeze failed: " << st.message() << "\n";
        return 1;
    }

    pomai::SearchResult out;
    st = db->Search(vectors[0], 5, &out);
    if (!st.ok()) {
        std::cerr << "Search failed: " << st.message() << "\n";
        return 1;
    }

    std::cout << "TopK results:\n";
    for (const auto& hit : out.hits) {
        std::cout << "  id=" << hit.id << " score=" << hit.score << "\n";
    }

    st = db->Close();
    if (!st.ok()) {
        std::cerr << "Close failed: " << st.message() << "\n";
        return 1;
    }
    return 0;
}
