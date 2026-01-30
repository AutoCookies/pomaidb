#include <atomic>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <pomai/storage/wal.h>
#include <pomai/core/seed.h>

using namespace pomai;
namespace fs = std::filesystem;

static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_wal_crash.XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char *res = mkdtemp(buf.data());
    if (!res)
        throw std::runtime_error("mkdtemp failed");
    return std::string(res);
}

static void RemoveDir(const std::string &d)
{
    std::error_code ec;
    fs::remove_all(d, ec);
}

static void PersistDurable(const std::string &path, Lsn lsn)
{
    int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
    if (fd < 0)
        return;
    std::string payload = std::to_string(lsn) + "\n";
    ::write(fd, payload.data(), payload.size());
    ::fdatasync(fd);
    ::close(fd);
}

static Lsn ReadDurable(const std::string &path)
{
    std::ifstream in(path);
    Lsn lsn = 0;
    if (in)
        in >> lsn;
    return lsn;
}

static int RunChild(const std::string &dir, const std::string &durable_file)
{
    Wal w("shard-0", dir, 4);
    w.Start();
    for (int i = 0; i < 5000; ++i)
    {
        UpsertRequest r;
        r.id = static_cast<Id>(i);
        r.vec.data = {1, 2, 3, 4};
        std::vector<UpsertRequest> batch{r};
        bool wait = (i % 7 == 0);
        Lsn lsn = w.AppendUpserts(batch, wait);
        if (wait)
        {
            w.WaitDurable(lsn);
            PersistDurable(durable_file, lsn);
        }
    }
    w.Stop();
    return 0;
}

int main(int argc, char **argv)
{
    int iterations = 5;
    if (argc > 1)
        iterations = std::max(1, std::atoi(argv[1]));

    std::string base_dir = MakeTempDir();
    std::string durable_file = base_dir + "/durable_lsn.txt";
    int failures = 0;

    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> sleep_ms(10, 200);

    for (int i = 0; i < iterations; ++i)
    {
        RemoveDir(base_dir);
        fs::create_directories(base_dir);
        pid_t pid = ::fork();
        if (pid == 0)
        {
            int ret = RunChild(base_dir, durable_file);
            _exit(ret);
        }
        if (pid < 0)
            throw std::runtime_error("fork failed");

        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms(rng)));
        ::kill(pid, SIGKILL);
        ::waitpid(pid, nullptr, 0);

        Lsn durable = ReadDurable(durable_file);
        Wal w("shard-0", base_dir, 4);
        Seed seed(4);
        WalReplayStats stats = w.ReplayToSeed(seed);
        if (stats.last_lsn < durable)
        {
            std::cerr << "Crash test FAILED: replay lsn=" << stats.last_lsn
                      << " durable_lsn=" << durable << "\n";
            ++failures;
        }
        else
        {
            std::cout << "Crash test iteration " << i << " OK (durable=" << durable
                      << ", replay=" << stats.last_lsn << ")\n";
        }
    }

    RemoveDir(base_dir);
    return failures == 0 ? 0 : 2;
}
