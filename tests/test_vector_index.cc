#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <arpa/inet.h>

#include "src/memory/arena.h"
#include "src/core/map.h"
#include "src/core/seed.h"
#include "src/ai/vector_index.h"

// Minimal test for VectorIndex (linear scan over PomaiMap stored inline vectors)
// We insert small inline vector values using map.put and then call VectorIndex::search.

static std::vector<char> pack_floats(const std::vector<float> &v)
{
    std::vector<char> out;
    out.resize(v.size() * sizeof(float));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

// parse VectorIndex binary result (repeat [4B keylen][key bytes][4B score(net float)])
static std::vector<std::pair<std::string, float>> parse_results(const std::vector<char> &buf)
{
    std::vector<std::pair<std::string, float>> r;
    size_t pos = 0;
    while (pos + 4 <= buf.size())
    {
        uint32_t net_klen = 0;
        std::memcpy(&net_klen, buf.data() + pos, 4);
        pos += 4;
        uint32_t klen = ntohl(net_klen);
        if (pos + klen + 4 > buf.size())
            break;
        std::string key(buf.data() + pos, buf.data() + pos + klen);
        pos += klen;
        uint32_t net_score = 0;
        std::memcpy(&net_score, buf.data() + pos, 4);
        pos += 4;
        uint32_t score_bits = ntohl(net_score);
        float score = 0.0f;
        std::memcpy(&score, &score_bits, sizeof(score));
        r.emplace_back(key, score);
    }
    return r;
}

int main()
{
    std::cout << "[TEST] VectorIndex linear-scan test\n";

    pomai::memory::PomaiArena arena = pomai::memory::PomaiArena::FromMB(4);
    if (!arena.is_valid())
    {
        std::cerr << "[FAIL] Arena allocation failed\n";
        return 1;
    }

    uint64_t max_seeds = arena.get_capacity_bytes() / sizeof(Seed);
    uint64_t slots = 1;
    while (slots < max_seeds)
        slots <<= 1;
    if (slots == 0)
        slots = 1;

    PomaiMap map(&arena, slots);

    // Create three inline vectors in the map
    const size_t dim = 3;
    std::vector<float> va = {0.0f, 0.0f, 0.0f};
    std::vector<float> vb = {10.0f, 0.0f, 0.0f};
    std::vector<float> vc = {0.0f, 10.0f, 0.0f};

    auto pa = pack_floats(va);
    auto pb = pack_floats(vb);
    auto pc = pack_floats(vc);

    // map.put(key, klen, valptr, vlen) - store inline float bytes
    if (!map.put("a", 1, pa.data(), static_cast<uint32_t>(pa.size())))
    {
        std::cerr << "[FAIL] map.put a failed\n";
        return 2;
    }
    if (!map.put("b", 1, pb.data(), static_cast<uint32_t>(pb.size())))
    {
        std::cerr << "[FAIL] map.put b failed\n";
        return 3;
    }
    if (!map.put("c", 1, pc.data(), static_cast<uint32_t>(pc.size())))
    {
        std::cerr << "[FAIL] map.put c failed\n";
        return 4;
    }

    // Mark seeds as OBJ_VECTOR so VectorIndex will consider them
    Seed *sa = map.find_seed("a", 1);
    Seed *sb = map.find_seed("b", 1);
    Seed *sc = map.find_seed("c", 1);
    if (!sa || !sb || !sc)
    {
        std::cerr << "[FAIL] find_seed returned null\n";
        return 5;
    }
    sa->type = Seed::OBJ_VECTOR;
    sb->type = Seed::OBJ_VECTOR;
    sc->type = Seed::OBJ_VECTOR;

    VectorIndex vi(&map);

    std::vector<float> q1 = {0.05f, -0.01f, 0.0f};
    auto out1 = vi.search(q1.data(), dim, 1);
    auto res1 = parse_results(out1);
    if (res1.empty())
    {
        std::cerr << "[FAIL] vi.search returned empty\n";
        return 6;
    }
    if (res1[0].first != "a")
    {
        std::cerr << "[FAIL] expected nearest 'a' got '" << res1[0].first << "'\n";
        return 7;
    }

    std::vector<float> q2 = {9.5f, 0.2f, 0.0f};
    auto out2 = vi.search(q2.data(), dim, 1);
    auto res2 = parse_results(out2);
    if (res2.empty() || res2[0].first != "b")
    {
        std::cerr << "[FAIL] expected nearest 'b' got ";
        if (res2.empty())
            std::cerr << "empty\n";
        else
            std::cerr << "'" << res2[0].first << "'\n";
        return 8;
    }

    std::cout << "[PASS] VectorIndex tests OK\n";
    return 0;
}