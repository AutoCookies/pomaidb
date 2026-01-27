#pragma once
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "membrane.h"
#include "types.h"
#include "server/config.h"

namespace pomai
{

    struct DbOptions
    {
        std::size_t dim{384};
        Metric metric{Metric::Cosine};
        std::size_t shards{4};
        std::size_t shard_queue_capacity{1024};
        std::string wal_dir{"./data"};
        pomai::server::WhisperConfig whisper;
    };

    class PomaiDB
    {
    public:
        explicit PomaiDB(DbOptions opt);

        void Start();
        void Stop();

        std::future<Lsn> Upsert(Id id, Vector vec, bool wait_durable = true);
        std::future<Lsn> UpsertBatch(std::vector<UpsertRequest> batch, bool wait_durable = true);

        SearchResponse Search(const SearchRequest &req) const;
        std::size_t TotalApproxCountUnsafe() const;

    private:
        DbOptions opt_;
        std::unique_ptr<MembraneRouter> membrane_;
    };

}