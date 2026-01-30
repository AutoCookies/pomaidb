#include <pomai/storage/verify.h>
#include <pomai/storage/snapshot.h>

#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: pomai_verify <db_dir>\n";
        return 2;
    }
    std::string db_dir = argv[1];
    pomai::storage::Manifest manifest;
    std::string err;
    if (!pomai::storage::VerifyAll(db_dir, manifest, &err))
    {
        std::cerr << "Verification failed: " << err << "\n";
        return 1;
    }
    std::string snapshot_path = manifest.checkpoint_path;
    if (!snapshot_path.empty() && snapshot_path.front() != '/')
        snapshot_path = db_dir + "/" + snapshot_path;
    pomai::storage::SnapshotData snapshot;
    if (!pomai::storage::ReadSnapshotFile(snapshot_path, snapshot, &err))
    {
        std::cerr << "Snapshot read failed: " << err << "\n";
        return 1;
    }
    std::cout << "Manifest version: " << manifest.version << "\n";
    std::cout << "Checkpoint path: " << manifest.checkpoint_path << "\n";
    std::cout << "Checkpoint epoch: " << manifest.checkpoint_epoch << "\n";
    std::cout << "Checkpoint LSN: " << manifest.checkpoint_lsn << "\n";
    std::cout << "Schema dim: " << snapshot.schema.dim << "\n";
    std::cout << "Schema metric: " << snapshot.schema.metric << "\n";
    std::cout << "Schema shards: " << snapshot.schema.shards << "\n";
    std::cout << "Schema index_kind: " << snapshot.schema.index_kind << "\n";
    return 0;
}
