#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>
#include <string>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>

#include "pomai/pomai.h"

namespace fs = std::filesystem;

static void Die(const char *msg)
{
    std::cerr << "FAIL: " << msg << "\n";
    std::exit(1);
}

// Oracle file format: just the last synced ID as a text line "Synced: <id>\n"
static const char* kOracleFile = "oracle.txt";

static void UpdateOracle(const std::string& db_dir, std::uint64_t id) {
    fs::path p = fs::path(db_dir) / kOracleFile;
    // Open with O_SYNC to ensure the oracle itself is crash-safe
    int fd = open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_SYNC, 0644);
    if (fd < 0) Die("oracle open failed");
    
    std::string s = "Synced: " + std::to_string(id) + "\n";
    if (write(fd, s.data(), s.size()) != (ssize_t)s.size()) Die("oracle write failed");
    close(fd);
}

static std::uint64_t ReadOracle(const std::string& db_dir) {
    fs::path p = fs::path(db_dir) / kOracleFile;
    if (!fs::exists(p)) return 0;
    
    std::ifstream in(p);
    std::string line;
    std::uint64_t max_id = 0;
    if (std::getline(in, line)) {
        if (line.starts_with("Synced: ")) {
            max_id = std::stoull(line.substr(8));
        }
    }
    return max_id;
}

static void ChildWriter(const std::string &path)
{
    pomai::DBOptions opt;
    opt.path = path;
    opt.shard_count = 4;
    opt.dim = 16;
    opt.fsync = pomai::FsyncPolicy::kAlways; // Strict durability required for test

    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opt, &db);
    if (!st.ok())
        Die(st.message().c_str());
    
    // Create a named membrane to verify persistence logic too
    pomai::MembraneSpec mspec;
    mspec.name = "important_data";
    mspec.dim = 16;
    mspec.shard_count = 2;
    st = db->CreateMembrane(mspec);
    if (!st.ok() && st.code() != pomai::ErrorCode::kAlreadyExists) Die(st.message().c_str());
    st = db->OpenMembrane("important_data"); 
    if (!st.ok()) Die(st.message().c_str());

    std::vector<float> v(opt.dim);
    for (std::uint64_t i = 1; i <= 20000; ++i)
    {
        for (std::size_t d = 0; d < v.size(); ++d)
            v[d] = static_cast<float>(i + d);
        
        // Write to both default and named membrane
        st = db->Put(i, v);
        if (!st.ok()) Die(st.message().c_str());

        st = db->Put("important_data", i, v);
        if (!st.ok()) Die(st.message().c_str());

        // Always Flush to match FsyncPolicy::kAlways behavior on WAL (technically Put flushes WAL if kAlways, 
        // but Flush() ensures everything is pushed down if using kOnFlush. 
        // Since we use kAlways, Put is durable. But let's call Flush occasionally to exercise it.
        if (i % 100 == 0) {
             db->Flush();
        }

        // Update Oracle occasionally to mark progress
        // We claim durability every 50 items to stress test replay
        if (i % 50 == 0) {
            UpdateOracle(path, i);
        }
        
        // Trigger Freeze/Compact to mix in segment state
        if ((i % 1000) == 0)
        {
             st = db->Freeze("default"); 
             st = db->Freeze("important_data");
        }
    }

    db->Close();
    std::exit(0);
}

static void VerifyConsistency(const std::string &path)
{
    // 1. Read Oracle
    std::uint64_t synced_upto = ReadOracle(path);
    if (synced_upto == 0) {
        // Crashed before first sync, nothing to check strictly except Open
        // return; 
    }

    // 2. Open DB
    pomai::DBOptions opt;
    opt.path = path;
    opt.shard_count = 4;
    opt.dim = 16;
    opt.fsync = pomai::FsyncPolicy::kAlways;

    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opt, &db);
    if (!st.ok())
        Die(("reopen failed: " + st.message()).c_str());

    // 3. Verify named membrane restoration (Critical Fix Check)
    st = db->OpenMembrane("important_data");
    bool named_ok = st.ok();
    // If we synced at least once, naming must exist (created at start)
    if (synced_upto > 0 && !named_ok) {
         Die("Named membrane 'important_data' failed to open after restart");
    }

    // 4. Verify Data Consistency
    for (std::uint64_t i = 1; i <= synced_upto; ++i) {
        std::vector<float> out;
        
        // Check Default
        st = db->Get(i, &out);
        if (!st.ok() || out.empty()) {
            std::cerr << "Data Loss detected! ID " << i << " missing (Oracle said synced up to " << synced_upto << ")\n";
            Die("Data Consistency Fail");
        }

        // Check Named if available
        if (named_ok) {
            st = db->Get("important_data", i, &out);
            if (!st.ok() || out.empty()) {
                std::cerr << "Data Loss in Named Membrane! ID " << i << " missing\n";
                 Die("Data Consistency Fail (Named)");
            }
        }
    }

    db->Close();
}

int main()
{
    const std::string base = "./crash_db";
    fs::remove_all(base);
    fs::create_directories(base);

    std::mt19937_64 rng{12345};
    std::uniform_int_distribution<int> kill_delay_ms(10, 300); // Varied crash times

    // 20 Rounds of crash testing
    for (int round = 0; round < 20; ++round)
    {
        const std::string path = base + "/round_" + std::to_string(round);
        fs::remove_all(path);
        fs::create_directories(path);

        std::cout << "[Round " << round << "] Starting writer...\n";
        
        pid_t pid = fork();
        if (pid < 0) Die("fork failed");

        if (pid == 0)
        {
            ChildWriter(path);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(kill_delay_ms(rng)));
        kill(pid, SIGKILL);

        int status = 0;
        (void)waitpid(pid, &status, 0);

        std::cout << "[Round " << round << "] Verifying consistency...\n";
        VerifyConsistency(path);
    }

    std::cout << "SUCCESS: Crash replay tests passed with data consistency verification.\n";
    return 0;
}
