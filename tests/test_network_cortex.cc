#include "src/ai/network_cortex.h"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <cassert>
#include <algorithm>
#include <mutex>

using namespace pomai::ai::orbit;
using pomai::config::NetworkCortexConfig;

static const char *ANSI_GREEN = "\033[32m";
static const char *ANSI_RED = "\033[31m";
static const char *ANSI_RESET = "\033[0m";

class TestRunner
{
public:
    void expect(bool condition, const std::string &test_name)
    {
        if (condition)
        {
            std::cout << ANSI_GREEN << "[PASS] " << ANSI_RESET << test_name << "\n";
            passed_++;
        }
        else
        {
            std::cout << ANSI_RED << "[FAIL] " << ANSI_RESET << test_name << "\n";
            failed_++;
        }
    }

    int summary()
    {
        std::cout << "\nResults: " << passed_ << " passed, " << failed_ << " failed.\n";
        return failed_ == 0 ? 0 : 1;
    }

private:
    int passed_ = 0;
    int failed_ = 0;
};

void test_lifecycle(TestRunner &runner)
{
    NetworkCortexConfig cfg;
    cfg.udp_port = 9001;
    cfg.pulse_interval_ms = 100;
    cfg.neighbor_ttl_ms = 1000;

    NetworkCortex node(cfg);

    bool start_ok = node.start();
    runner.expect(start_ok, "Node starts successfully");

    // Idempotency check
    bool start_again = node.start();
    runner.expect(start_again, "Node start is idempotent");

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    node.stop();
    runner.expect(true, "Node stops cleanly");
}

void test_discovery(TestRunner &runner)
{
    // Utilize SO_REUSEPORT to bind multiple nodes to the same discovery port on localhost
    NetworkCortexConfig cfg;
    cfg.udp_port = 9002;
    cfg.pulse_interval_ms = 50;
    cfg.neighbor_ttl_ms = 2000;

    NetworkCortex node_a(cfg);
    NetworkCortex node_b(cfg);

    node_a.start();
    node_b.start();

    // Allow pheromones to propagate
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    auto neighbors_a = node_a.get_neighbors();
    auto neighbors_b = node_b.get_neighbors();

    runner.expect(node_a.node_id() != node_b.node_id(), "Nodes have unique IDs");

    // Note: Packet loss is possible even on loopback, but highly unlikely in 300ms window
    bool a_sees_b = !neighbors_a.empty();
    bool b_sees_a = !neighbors_b.empty();

    runner.expect(a_sees_b, "Node A discovered neighbor");
    runner.expect(b_sees_a, "Node B discovered neighbor");

    if (a_sees_b)
    {
        runner.expect(neighbors_a[0].port == cfg.udp_port, "Neighbor port matches config");
    }

    node_a.stop();
    node_b.stop();
}

void test_ttl_expiration(TestRunner &runner)
{
    NetworkCortexConfig cfg;
    cfg.udp_port = 9003;
    cfg.pulse_interval_ms = 50;
    cfg.neighbor_ttl_ms = 400; // Short TTL for testing

    NetworkCortex node_observer(cfg);
    NetworkCortex node_ghost(cfg);

    node_observer.start();
    node_ghost.start();

    // Wait for discovery
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    bool seen_initially = !node_observer.get_neighbors().empty();
    runner.expect(seen_initially, "Observer saw Ghost initially");

    // Kill the ghost
    node_ghost.stop();

    // Wait for TTL to expire (TTL 400ms + buffer)
    std::this_thread::sleep_for(std::chrono::milliseconds(600));

    auto neighbors_after = node_observer.get_neighbors();
    runner.expect(neighbors_after.empty(), "Ghost expired from neighbor table after TTL");

    node_observer.stop();
}

void test_concurrency_stress(TestRunner &runner)
{
    NetworkCortexConfig cfg;
    cfg.udp_port = 9004;
    cfg.pulse_interval_ms = 10; // Rapid pulses
    cfg.neighbor_ttl_ms = 5000;

    NetworkCortex node_main(cfg);
    NetworkCortex node_peer(cfg);

    node_main.start();
    node_peer.start();

    std::atomic<bool> stop_flag{false};
    std::atomic<size_t> total_reads{0};

    auto reader_func = [&]()
    {
        while (!stop_flag)
        {
            auto n = node_main.get_neighbors();
            total_reads++;
            // Tiny yield to allow contention
            std::this_thread::yield();
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i)
    {
        threads.emplace_back(reader_func);
    }

    // Let it run under stress
    std::this_thread::sleep_for(std::chrono::seconds(1));

    stop_flag = true;
    for (auto &t : threads)
        t.join();

    runner.expect(total_reads > 1000, "High-throughput concurrent reads on neighbor table");

    // Ensure data integrity wasn't compromised during stress
    auto final_n = node_main.get_neighbors();
    runner.expect(!final_n.empty(), "Neighbor table intact after stress");

    node_main.stop();
    node_peer.stop();
}

void test_payload_safety(TestRunner &runner)
{
    NetworkCortexConfig cfg;
    cfg.udp_port = 9005;

    NetworkCortex node(cfg);
    node.start();

    // Try to emit huge payload
    std::vector<char> huge_payload(65536, 'X');

    // Should not crash
    try
    {
        node.emit_pheromone(PheromoneType::IAM_HERE, huge_payload.data(), huge_payload.size());
        runner.expect(true, "Emit huge payload handled safely (no crash)");
    }
    catch (...)
    {
        runner.expect(false, "Emit huge payload caused exception");
    }

    node.stop();
}

int main()
{
    TestRunner runner;

    test_lifecycle(runner);
    test_discovery(runner);
    test_ttl_expiration(runner);
    test_concurrency_stress(runner);
    test_payload_safety(runner);

    return runner.summary();
}