#include "src/ai/ids_block.h"
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <cassert>
#include <cstring>
#include <algorithm>

using namespace pomai::ai::soa;

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

void test_constants_and_masks(TestRunner &runner)
{
    bool mask_coverage = (IdEntry::TAG_MASK | IdEntry::PAYLOAD_MASK) == 0xFFFFFFFFFFFFFFFFULL;
    bool mask_disjoint = (IdEntry::TAG_MASK & IdEntry::PAYLOAD_MASK) == 0;

    runner.expect(mask_coverage, "Masks cover full 64 bits");
    runner.expect(mask_disjoint, "Masks are disjoint");
    runner.expect(IdEntry::TAG_LOCAL != IdEntry::TAG_REMOTE, "Tags are distinct (Local vs Remote)");
    runner.expect(IdEntry::TAG_LOCAL != IdEntry::TAG_LABEL, "Tags are distinct (Local vs Label)");
}

void test_payload_bounds(TestRunner &runner)
{
    uint64_t max_payload = IdEntry::PAYLOAD_MASK;
    uint64_t overflow = max_payload + 1;

    runner.expect(IdEntry::fits_payload(0), "Payload fits 0");
    runner.expect(IdEntry::fits_payload(max_payload), "Payload fits max value");
    runner.expect(!IdEntry::fits_payload(overflow), "Payload rejects overflow");
    runner.expect(!IdEntry::fits_payload(0xFFFFFFFFFFFFFFFFULL), "Payload rejects max uint64");
}

void test_packing_unpacking(TestRunner &runner)
{
    uint64_t val = 12345;

    uint64_t packed_local = IdEntry::pack_local_offset(val);
    bool local_ok = (IdEntry::tag_of(packed_local) == IdEntry::TAG_LOCAL) &&
                    (IdEntry::payload_of(packed_local) == val);
    runner.expect(local_ok, "Pack/Unpack Local Offset");

    uint64_t packed_remote = IdEntry::pack_remote_id(val);
    bool remote_ok = (IdEntry::tag_of(packed_remote) == IdEntry::TAG_REMOTE) &&
                     (IdEntry::payload_of(packed_remote) == val);
    runner.expect(remote_ok, "Pack/Unpack Remote ID");

    uint64_t packed_label = IdEntry::pack_label(val);
    bool label_ok = (IdEntry::tag_of(packed_label) == IdEntry::TAG_LABEL) &&
                    (IdEntry::payload_of(packed_label) == val);
    runner.expect(label_ok, "Pack/Unpack Label");
}

void test_try_pack_logic(TestRunner &runner)
{
    uint64_t max_val = IdEntry::PAYLOAD_MASK;
    uint64_t bad_val = max_val + 1;
    uint64_t out = 0;

    bool ok1 = IdEntry::try_pack_local_offset(max_val, out);
    bool check1 = ok1 && (IdEntry::payload_of(out) == max_val) && (IdEntry::tag_of(out) == IdEntry::TAG_LOCAL);
    runner.expect(check1, "try_pack_local_offset success boundary");

    bool ok2 = IdEntry::try_pack_local_offset(bad_val, out);
    runner.expect(!ok2, "try_pack_local_offset fail overflow");

    bool ok3 = IdEntry::try_pack_remote_id(max_val, out);
    bool check3 = ok3 && (IdEntry::payload_of(out) == max_val) && (IdEntry::tag_of(out) == IdEntry::TAG_REMOTE);
    runner.expect(check3, "try_pack_remote_id success boundary");

    bool ok4 = IdEntry::try_pack_remote_id(bad_val, out);
    runner.expect(!ok4, "try_pack_remote_id fail overflow");

    bool ok5 = IdEntry::try_pack_label(max_val, out);
    bool check5 = ok5 && (IdEntry::payload_of(out) == max_val) && (IdEntry::tag_of(out) == IdEntry::TAG_LABEL);
    runner.expect(check5, "try_pack_label success boundary");

    bool ok6 = IdEntry::try_pack_label(bad_val, out);
    runner.expect(!ok6, "try_pack_label fail overflow");
}

void test_atomic_basics(TestRunner &runner)
{
    uint64_t storage = 0;
    uint64_t val = IdEntry::pack_label(999);

    IdEntry::atomic_store(&storage, val);
    uint64_t loaded = IdEntry::atomic_load(&storage);

    runner.expect(loaded == val, "Atomic Store/Load consistency");

    uint64_t expected = val;
    uint64_t desired = IdEntry::pack_local_offset(111);
    bool cas_success = IdEntry::atomic_compare_exchange(&storage, expected, desired);

    runner.expect(cas_success, "Atomic CAS success");
    runner.expect(storage == desired, "Atomic CAS updated value");

    uint64_t wrong_expected = val;
    bool cas_fail = IdEntry::atomic_compare_exchange(&storage, wrong_expected, val);

    runner.expect(!cas_fail, "Atomic CAS fail on mismatch");
    runner.expect(wrong_expected == desired, "Atomic CAS updates expected on failure");
    runner.expect(storage == desired, "Atomic CAS preserved value on failure");
}

void test_atomic_concurrency(TestRunner &runner)
{
    uint64_t shared_counter = 0;
    const int num_threads = 8;
    const int increments_per_thread = 10000;

    auto worker = [&]()
    {
        for (int i = 0; i < increments_per_thread; ++i)
        {
            while (true)
            {
                uint64_t current = IdEntry::atomic_load(&shared_counter);
                uint64_t current_payload = IdEntry::payload_of(current);
                uint64_t next = IdEntry::pack_local_offset(current_payload + 1);

                if (IdEntry::atomic_compare_exchange(&shared_counter, current, next))
                {
                    break;
                }
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(worker);
    }

    for (auto &t : threads)
    {
        t.join();
    }

    uint64_t final_val = IdEntry::atomic_load(&shared_counter);
    uint64_t final_payload = IdEntry::payload_of(final_val);
    uint64_t expected = num_threads * increments_per_thread;

    runner.expect(IdEntry::tag_of(final_val) == IdEntry::TAG_LOCAL, "Concurrent Tag Consistency");
    runner.expect(final_payload == expected, "Concurrent CAS correctness");
}

int main()
{
    TestRunner runner;

    test_constants_and_masks(runner);
    test_payload_bounds(runner);
    test_packing_unpacking(runner);
    test_try_pack_logic(runner);
    test_atomic_basics(runner);
    test_atomic_concurrency(runner);

    return runner.summary();
}