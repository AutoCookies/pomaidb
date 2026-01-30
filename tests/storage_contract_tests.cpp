#include <pomai/storage/snapshot.h>
#include <pomai/storage/verify.h>
#include <pomai/storage/file_util.h>
#include <pomai/storage/wal.h>
#include <pomai/core/seed.h>

#include <filesystem>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <unistd.h>

using namespace pomai;
namespace fs = std::filesystem;

static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_storage_test.XXXXXX";
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

static pomai::Seed::PersistedState MakeState(std::size_t dim, Id id, float value)
{
    pomai::Seed::PersistedState state;
    state.dim = static_cast<std::uint32_t>(dim);
    state.ids = {id};
    state.qmins.assign(dim, 0.0f);
    state.qmaxs.assign(dim, 1.0f);
    state.qscales.assign(dim, 1.0f / 255.0f);
    state.qdata.resize(dim);
    std::fill(state.qdata.begin(), state.qdata.end(), static_cast<std::uint8_t>(std::clamp<int>(static_cast<int>(value * 255.0f), 0, 255)));
    state.namespace_ids = {0};
    state.tag_offsets = {0, 0};
    state.tag_ids = {};
    state.is_fixed = true;
    state.total_ingested = 0;
    state.fixed_bounds_after = 0;
    return state;
}

static pomai::storage::SnapshotData MakeSnapshot()
{
    pomai::storage::SnapshotData data;
    data.schema.dim = 2;
    data.schema.metric = static_cast<std::uint32_t>(Metric::L2);
    data.schema.shards = 1;
    data.schema.index_kind = 0;
    pomai::storage::ShardSnapshot shard;
    shard.shard_id = 0;
    shard.live = MakeState(2, 1, 0.1f);
    data.shards.push_back(std::move(shard));
    data.shard_lsns = {10};
    return data;
}

static void CorruptFileByte(const std::string &path, std::size_t offset)
{
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekg(0, std::ios::end);
    std::size_t size = static_cast<std::size_t>(f.tellg());
    if (offset >= size)
        throw std::runtime_error("offset out of range");
    f.seekg(offset);
    char c;
    f.read(&c, 1);
    c ^= 0x5A;
    f.seekp(offset);
    f.write(&c, 1);
    f.flush();
}

int main()
{
    std::string dir = MakeTempDir();
    try
    {
        pomai::storage::SnapshotData data = MakeSnapshot();
        pomai::storage::CommitResult res;
        std::string err;
        bool ok = pomai::storage::CommitCheckpointAtomically(dir, data, {}, &res, &err);
        if (!ok)
        {
            std::cerr << "Commit failed: " << err << "\n";
            return 1;
        }

#ifndef NDEBUG
        setenv("POMAI_FAILPOINT", "before_manifest_rename", 1);
        pomai::storage::SnapshotData data2 = MakeSnapshot();
        data2.shard_lsns = {20};
        ok = pomai::storage::CommitCheckpointAtomically(dir, data2, {}, nullptr, &err);
        unsetenv("POMAI_FAILPOINT");
        if (ok)
        {
            std::cerr << "Failpoint did not trigger\n";
            return 1;
        }
        pomai::storage::Manifest manifest;
        if (!pomai::storage::VerifyManifest(dir, manifest, &err))
        {
            std::cerr << "Manifest read failed: " << err << "\n";
            return 1;
        }
        if (manifest.checkpoint_lsn != 10)
        {
            std::cerr << "Manifest atomicity failure\n";
            return 1;
        }
#endif

        std::string snapshot_path = dir + "/" + res.manifest.checkpoint_path;
        CorruptFileByte(snapshot_path, 32);
        if (pomai::storage::VerifySnapshotFile(snapshot_path, &err))
        {
            std::cerr << "Corruption not detected\n";
            return 1;
        }

        pomai::storage::ResetDirFsyncCounter();
        pomai::storage::CommitCheckpointAtomically(dir, MakeSnapshot(), {}, nullptr, &err);
        if (pomai::storage::DirFsyncCounter() < 3)
        {
            std::cerr << "Directory fsyncs not observed\n";
            return 1;
        }

        pomai::storage::DbPaths paths = pomai::storage::MakeDbPaths(dir);
        pomai::storage::EnsureDbDirs(paths);
        Seed seed(2);
        Wal wal("shard-0", paths.wal_dir, 2);
        wal.Start();
        std::vector<UpsertRequest> batch1;
        batch1.push_back({1, Vector{{0.1f, 0.2f}}, {}});
        Lsn lsn1 = wal.AppendUpserts(batch1, true);
        wal.WaitDurable(lsn1);
        seed.ApplyUpserts(batch1);

        pomai::storage::SnapshotData snap;
        snap.schema.dim = 2;
        snap.schema.metric = static_cast<std::uint32_t>(Metric::L2);
        snap.schema.shards = 1;
        snap.schema.index_kind = 0;
        pomai::storage::ShardSnapshot shard;
        shard.shard_id = 0;
        shard.live = seed.ExportPersistedState();
        snap.shards.push_back(std::move(shard));
        snap.shard_lsns = {lsn1};
        pomai::storage::CommitCheckpointAtomically(dir, snap, {}, nullptr, &err);

        std::vector<UpsertRequest> batch2;
        batch2.push_back({2, Vector{{0.5f, 0.6f}}, {}});
        wal.AppendUpserts(batch2, true);
        wal.Stop();

        pomai::storage::SnapshotData loaded;
        pomai::storage::Manifest manifest;
        if (!pomai::storage::RecoverLatestCheckpoint(dir, loaded, manifest, &err))
        {
            std::cerr << "RecoverLatestCheckpoint failed: " << err << "\n";
            return 1;
        }
        Seed recovered(2);
        recovered.LoadPersistedState(loaded.shards[0].live);
        WalReplayStats stats = wal.ReplayToSeed(recovered, lsn1);
        if (stats.records_applied != 1)
        {
            std::cerr << "WAL boundary replay mismatch\n";
            return 1;
        }
        if (recovered.Count() != 2)
        {
            std::cerr << "Recovered count mismatch\n";
            return 1;
        }

        RemoveDir(dir);
        std::cout << "storage_contract_tests PASSED\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
        RemoveDir(dir);
        return 1;
    }
}
