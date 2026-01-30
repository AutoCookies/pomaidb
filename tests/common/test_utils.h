#pragma once

#include <pomai/api/pomai_db.h>
#include <pomai/core/seed.h>

#include <cstddef>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace pomai::test
{
    class TempDir
    {
    public:
        TempDir();
        ~TempDir();
        TempDir(const TempDir &) = delete;
        TempDir &operator=(const TempDir &) = delete;

        const std::filesystem::path &path() const { return path_; }
        std::string str() const { return path_.string(); }

    private:
        std::filesystem::path path_;
    };

    std::mt19937 &Rng();

    Vector MakeVector(std::size_t dim, float base);

    UpsertRequest MakeUpsert(Id id,
                             std::size_t dim,
                             float base,
                             std::uint32_t ns = 0,
                             std::vector<TagId> tags = {});

    std::vector<UpsertRequest> MakeBatch(std::size_t count,
                                         std::size_t dim,
                                         float start = 0.1f,
                                         std::uint32_t ns = 0);

    DbOptions DefaultDbOptions(const std::string &dir, std::size_t dim = 8, std::size_t shards = 2);

    SearchRequest MakeSearchRequest(const Vector &query, std::size_t topk = 5);

    std::vector<SearchResultItem> BruteForceL2(const std::vector<UpsertRequest> &rows,
                                               const Vector &query,
                                               std::size_t topk);

    std::vector<Id> ScanAll(PomaiDB &db, ScanRequest req);

    bool ContainsId(const std::vector<SearchResultItem> &items, Id id);
}
