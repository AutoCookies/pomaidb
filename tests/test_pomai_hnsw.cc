#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <cassert>
#include <cstring>
#include <algorithm> // For std::max

#include "ai/hnswlib/hnswlib.h"
#include "ai/pomai_hnsw.h"
#include "ai/ppe.h"

using namespace pomai::ai;
using namespace std::chrono;

#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"

void log_pass(const std::string &msg)
{
    std::cout << GREEN << "[PASS] " << RESET << msg << std::endl;
}

void log_info(const std::string &msg)
{
    std::cout << "[INFO] " << msg << std::endl;
}

int main()
{
    int dim = 128;
    size_t max_elements = 10000;
    size_t M = 16;
    size_t ef_construction = 200;

    std::cout << "=== POMAI HNSW INTEGRATION TEST ===\n";

    hnswlib::L2Space l2space(dim);
    PPHNSW<float> alg(&l2space, max_elements, M, ef_construction);

    // 1. Granularity Check
    size_t raw_vec_size = l2space.get_data_size();
    size_t pomai_seed_size = alg.getSeedSize(); // Accessed via protected inheritance

    if (pomai_seed_size == raw_vec_size + sizeof(PPEHeader))
    {
        log_pass("Granularity Check: Seed size includes Header + Vector correctly.");
        std::cout << "       Header: " << sizeof(PPEHeader) << "B | Vector: " << raw_vec_size << "B | Total: " << pomai_seed_size << "B\n";
    }
    else
    {
        std::cout << RED << "[FAIL] Size mismatch! Expected " << raw_vec_size + sizeof(PPEHeader)
                  << " but got " << pomai_seed_size << RESET << std::endl;
        return 1;
    }

    // 2. Insert Data
    log_info("Generating 1000 random seeds...");
    std::mt19937 rng(42);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    std::vector<float> data(dim * 1000);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = dist(rng);

    for (size_t i = 0; i < 1000; ++i)
    {
        alg.addPoint(data.data() + i * dim, i);
    }
    log_pass("Insertion: Added 1000 items without crash.");

    // 3. Test Membrane Update
    log_info("Sleeping 50ms to age seeds...");
    std::this_thread::sleep_for(milliseconds(50));

    size_t cold_before = alg.countColdSeeds(20 * 1000000); // 20ms threshold
    std::cout << "Cold seeds (>20ms): " << cold_before << "/1000\n";

    log_info("Searching for item ID: 10 (Heating it up)...");
    float *query = data.data() + 10 * dim;
    auto result = alg.searchKnnAdaptive(query, 5, 0.0);

    // 4. Verify results
    bool found = false;
    while (!result.empty())
    {
        if (result.top().second == 10)
            found = true;
        result.pop();
    }
    if (found)
        log_pass("Search: Found target ID 10.");
    else
        std::cout << RED << "[FAIL] Search did not find target ID 10." << RESET << std::endl;

    return 0;
}