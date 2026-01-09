// tests/arena_demote_async_test.cc
//
// Simple unit test for PomaiArena::demote_blob_async lifecycle:
//  - allocate arena
//  - allocate a blob and prepare data_with_header
//  - call demote_blob_async -> returns placeholder id
//  - poll until resolve_pending_remote indicates completion (timeout fail)
//  - verify blob_ptr_from_offset_for_map returns non-null and payload matches original

#include "src/memory/arena.h"
#include "src/core/config.h"

#include <iostream>
#include <cstring>
#include <thread>
#include <chrono>
#include <vector>
#include <cassert>

int main()
{
    using namespace pomai::memory;
    using namespace std::chrono_literals;

    // create a small arena (e.g. 8 MB)
    PomaiArena arena = PomaiArena::FromMB(8);
    if (!arena.is_valid())
    {
        std::cerr << "Failed to allocate arena\n";
        return 2;
    }

    // Prepare a sample payload: 3 floats (12 bytes)
    const uint32_t payload_len = 3 * sizeof(float);
    std::vector<char> payload(sizeof(uint32_t) + payload_len + 1);
    std::memcpy(payload.data(), &payload_len, sizeof(uint32_t));
    float sample[3] = {1.0f, 2.0f, 3.0f};
    std::memcpy(payload.data() + sizeof(uint32_t), sample, payload_len);
    payload[sizeof(uint32_t) + payload_len] = '\0';

    // Call demote_blob_async
    uint64_t rid = arena.demote_blob_async(payload.data(), static_cast<uint32_t>(payload.size()));
    if (rid == 0)
    {
        std::cerr << "demote_blob_async returned 0 (failed or skipped)\n";
        return 2;
    }

    std::cout << "demote_blob_async returned remote id: " << rid << "\n";

    // Poll until remote_map updated and blob_ptr_from_offset_for_map returns non-null
    const int max_wait_ms = 5000;
    int waited = 0;
    const int step_ms = 50;
    const char *ptr = nullptr;
    while (waited < max_wait_ms)
    {
        ptr = arena.blob_ptr_from_offset_for_map(rid);
        if (ptr)
            break;
        // if still pending, resolve_pending_remote may return 0
        uint64_t resolved = arena.resolve_pending_remote(rid);
        if (resolved != 0)
        {
            ptr = arena.blob_ptr_from_offset_for_map(resolved);
            if (ptr)
                break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
        waited += step_ms;
    }

    if (!ptr)
    {
        std::cerr << "Timed out waiting for demote worker to write remote file\n";
        return 3;
    }

    uint32_t stored_len = *reinterpret_cast<const uint32_t *>(ptr);
    if (stored_len != payload_len)
    {
        std::cerr << "Stored length mismatch: got " << stored_len << " expected " << payload_len << "\n";
        return 4;
    }

    const char *payload_ptr = ptr + sizeof(uint32_t);
    if (std::memcmp(payload_ptr, payload.data() + sizeof(uint32_t), payload_len) != 0)
    {
        std::cerr << "Payload mismatch after demote\n";
        return 5;
    }

    std::cout << "demote_blob_async lifecycle test: success\n";
    return 0;
}