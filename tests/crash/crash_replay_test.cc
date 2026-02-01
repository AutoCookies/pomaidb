#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include "pomai/pomai.h"

namespace fs = std::filesystem;

static void Die(const char *msg)
{
    std::cerr << msg << "\n";
    std::exit(1);
}

static void ChildWriter(const std::string &path)
{
    pomai::DBOptions opt;
    opt.path = path;
    opt.shard_count = 4;
    opt.dim = 16;
    opt.fsync = pomai::FsyncPolicy::kOnFlush; // Gate#1 boundary via Flush

    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opt, &db);
    if (!st.ok())
        Die(st.message().c_str());

    std::vector<float> v(opt.dim);
    for (std::uint64_t i = 1; i <= 20000; ++i)
    {
        for (std::size_t d = 0; d < v.size(); ++d)
            v[d] = static_cast<float>(i + d);
        st = db->Put(i, v);
        if (!st.ok())
            Die(st.message().c_str());

        if ((i % 200) == 0)
        {
            st = db->Flush();
            if (!st.ok())
                Die(st.message().c_str());
        }
    }

    db->Close();
    std::exit(0);
}

static void VerifyOpenAndSearch(const std::string &path)
{
    pomai::DBOptions opt;
    opt.path = path;
    opt.shard_count = 4;
    opt.dim = 16;
    opt.fsync = pomai::FsyncPolicy::kOnFlush;

    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opt, &db);
    if (!st.ok())
        Die(("reopen failed: " + st.message()).c_str());

    std::vector<float> q(opt.dim, 1.0f);
    pomai::SearchResult res{};
    st = db->Search(q, 10, &res);
    if (!st.ok())
        Die(("search failed: " + st.message()).c_str());

    db->Close();
}

int main()
{
    const std::string base = "./crash_db";
    fs::remove_all(base);
    fs::create_directories(base);

    std::mt19937_64 rng{12345};
    std::uniform_int_distribution<int> kill_delay_ms(5, 60);

    // Repeat many cycles
    for (int round = 0; round < 50; ++round)
    {
        const std::string path = base; // reuse same DB to exercise replay growth

        pid_t pid = fork();
        if (pid < 0)
            Die("fork failed");

        if (pid == 0)
        {
            ChildWriter(path);
        }

        // Parent: wait a bit then randomly kill -9
        std::this_thread::sleep_for(std::chrono::milliseconds(kill_delay_ms(rng)));
        kill(pid, SIGKILL);

        int status = 0;
        waitpid(pid, &status, 0);

        // Must be able to reopen regardless of truncation at tail.
        VerifyOpenAndSearch(path);
        std::cout << "round " << round << " OK\n";
    }

    std::cout << "crash replay test PASS\n";
    return 0;
}
