#include <pomai/storage/snapshot.h>
#include <pomai/storage/verify.h>
#include <pomai/core/seed.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

namespace fs = std::filesystem;

static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_corrupt.XXXXXX";
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

int main()
{
    int failures = 0;
    std::string dir = MakeTempDir();
    try
    {
        pomai::Seed seed(2);
        pomai::UpsertRequest req;
        req.id = 1;
        req.vec.data = {1.0f, 2.0f};
        seed.ApplyUpserts({req});

        pomai::storage::SnapshotData snapshot;
        snapshot.schema.dim = 2;
        snapshot.schema.metric = 0;
        snapshot.schema.shards = 1;
        snapshot.schema.index_kind = 0;
        pomai::storage::ShardSnapshot shard;
        shard.shard_id = 0;
        shard.live = seed.ExportPersistedState();
        snapshot.shards.push_back(std::move(shard));
        snapshot.shard_lsns.push_back(0);

        pomai::storage::CommitResult res;
        std::string err;
        if (!pomai::storage::CommitCheckpointAtomically(dir, snapshot, {}, &res, &err))
        {
            std::cerr << "Commit failed: " << err << "\n";
            ++failures;
        }

        pomai::storage::Manifest manifest;
        if (!pomai::storage::VerifyAll(dir, manifest, &err))
        {
            std::cerr << "Verify failed unexpectedly: " << err << "\n";
            ++failures;
        }

        std::string snapshot_path = dir + "/" + res.manifest.checkpoint_path;
        std::fstream file(snapshot_path, std::ios::in | std::ios::out | std::ios::binary);
        if (!file)
        {
            std::cerr << "Failed to open snapshot for corruption\n";
            ++failures;
        }
        else
        {
            file.seekp(0);
            char byte = 0;
            file.write(&byte, 1);
            file.close();
        }

        if (pomai::storage::VerifyAll(dir, manifest, &err))
        {
            std::cerr << "Expected corruption detection failure\n";
            ++failures;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Corruption test exception: " << e.what() << "\n";
        ++failures;
    }

    RemoveDir(dir);
    return failures == 0 ? 0 : 1;
}
