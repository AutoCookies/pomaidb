#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <pomai/core/membrane.h>
#include <pomai/core/shard.h>

using namespace pomai;
namespace fs = std::filesystem;

static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_centroids.XXXXXX";
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

static void TruncateFile(const std::string &path, off_t newsize)
{
    int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
    if (fd < 0)
        throw std::runtime_error("open for truncate failed");
    if (ftruncate(fd, newsize) != 0)
    {
        ::close(fd);
        throw std::runtime_error("ftruncate failed");
    }
    if (fdatasync(fd) != 0)
    {
        ::close(fd);
        throw std::runtime_error("fdatasync failed");
    }
    ::close(fd);
}

static std::uint16_t HostToLe16(std::uint16_t v)
{
#if __BYTE_ORDER == __BIG_ENDIAN
    return __builtin_bswap16(v);
#else
    return v;
#endif
}

static std::uint32_t HostToLe32(std::uint32_t v)
{
#if __BYTE_ORDER == __BIG_ENDIAN
    return __builtin_bswap32(v);
#else
    return v;
#endif
}

static std::uint64_t HostToLe64(std::uint64_t v)
{
#if __BYTE_ORDER == __BIG_ENDIAN
    return __builtin_bswap64(v);
#else
    return v;
#endif
}

static float HostToLeFloat(float v)
{
#if __BYTE_ORDER == __BIG_ENDIAN
    std::uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    u = __builtin_bswap32(u);
    std::memcpy(&v, &u, sizeof(u));
#endif
    return v;
}

static void WriteCentroidsFile(const std::string &path,
                               const std::array<char, 8> &magic,
                               std::uint32_t version,
                               std::uint16_t dim,
                               const std::vector<float> &payload)
{
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0)
        throw std::runtime_error("open failed");
    if (::write(fd, magic.data(), magic.size()) != (ssize_t)magic.size())
        throw std::runtime_error("write magic failed");
    std::uint32_t version_le = HostToLe32(version);
    std::uint16_t dim_le = HostToLe16(dim);
    std::uint64_t count_le = HostToLe64(payload.size() / dim);
    if (::write(fd, &version_le, sizeof(version_le)) != (ssize_t)sizeof(version_le))
        throw std::runtime_error("write version failed");
    if (::write(fd, &dim_le, sizeof(dim_le)) != (ssize_t)sizeof(dim_le))
        throw std::runtime_error("write dim failed");
    if (::write(fd, &count_le, sizeof(count_le)) != (ssize_t)sizeof(count_le))
        throw std::runtime_error("write count failed");
    for (float f : payload)
    {
        float out = HostToLeFloat(f);
        if (::write(fd, &out, sizeof(out)) != (ssize_t)sizeof(out))
            throw std::runtime_error("write payload failed");
    }
    if (::fdatasync(fd) != 0)
        throw std::runtime_error("fdatasync failed");
    ::close(fd);
}

static std::unique_ptr<MembraneRouter> MakeRouter(std::size_t dim, const std::string &wal_dir)
{
    std::vector<std::unique_ptr<Shard>> shards;
    shards.push_back(std::make_unique<Shard>("shard-0", dim, 16, wal_dir));
    pomai::WhisperConfig cfg;
    return std::make_unique<MembraneRouter>(std::move(shards),
                                            cfg,
                                            dim,
                                            Metric::L2,
                                            0,
                                            500,
                                            MembraneRouter::FilterConfig::Default(),
                                            []() {});
}

int main()
{
    int failures = 0;

    try
    {
        std::string dir = MakeTempDir();
        std::string path = dir + "/centroids.bin";
        auto router = MakeRouter(4, dir);

        std::vector<Vector> centroids;
        Vector a;
        a.data = {1.0f, 2.0f, 3.0f, 4.0f};
        Vector b;
        b.data = {5.0f, 6.0f, 7.0f, 8.0f};
        centroids.push_back(a);
        centroids.push_back(b);
        router->ConfigureCentroids(centroids);

        if (!router->SaveCentroidsToFile(path))
            throw std::runtime_error("SaveCentroidsToFile failed");

        auto router2 = MakeRouter(4, dir);
        if (!router2->LoadCentroidsFromFile(path))
            throw std::runtime_error("LoadCentroidsFromFile failed");

        auto loaded = router2->SnapshotCentroids();
        if (loaded.size() != centroids.size())
        {
            std::cerr << "Roundtrip failed: size mismatch\n";
            ++failures;
        }
        else
        {
            for (std::size_t i = 0; i < centroids.size(); ++i)
            {
                if (loaded[i].data != centroids[i].data)
                {
                    std::cerr << "Roundtrip failed: centroid mismatch\n";
                    ++failures;
                    break;
                }
            }
        }

        RemoveDir(dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Roundtrip exception: " << e.what() << "\n";
        ++failures;
    }

    try
    {
        std::string dir = MakeTempDir();
        std::string path = dir + "/centroids.bin";
        auto router = MakeRouter(4, dir);
        Vector c;
        c.data = {1.0f, 2.0f, 3.0f, 4.0f};
        router->ConfigureCentroids({c});
        if (!router->SaveCentroidsToFile(path))
            throw std::runtime_error("save failed");

        auto size = static_cast<off_t>(fs::file_size(path));
        TruncateFile(path, size - 4);

        auto router2 = MakeRouter(4, dir);
        if (router2->LoadCentroidsFromFile(path))
        {
            std::cerr << "Truncate test FAILED: expected load failure\n";
            ++failures;
        }

        RemoveDir(dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Truncate exception: " << e.what() << "\n";
        ++failures;
    }

    try
    {
        std::string dir = MakeTempDir();
        std::string path = dir + "/centroids.bin";
        std::array<char, 8> magic = {'P', 'O', 'M', 'C', 'E', 'N', '0', '7'};
        std::vector<float> payload = {1.0f, 2.0f};
        WriteCentroidsFile(path, magic, /*version=*/99, /*dim=*/2, payload);

        auto router = MakeRouter(2, dir);
        if (router->LoadCentroidsFromFile(path))
        {
            std::cerr << "Version mismatch test FAILED: expected load failure\n";
            ++failures;
        }

        RemoveDir(dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Version mismatch exception: " << e.what() << "\n";
        ++failures;
    }

    if (failures == 0)
        std::cout << "Centroids persistence tests PASSED\n";
    else
        std::cerr << "Centroids persistence tests FAILED: " << failures << "\n";

    return failures == 0 ? 0 : 1;
}
