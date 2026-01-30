#include <pomai/api/pomai_db.h>
#include <pomai/storage/verify.h>

#include <fstream>
#include <iostream>
#include <string>

namespace
{
    void PrintUsage()
    {
        std::cerr << "Usage: pomai_dump <db_dir> <output_path> [--batch N] [--cursor CUR] [--namespace ID] [--tag ID]\n";
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        PrintUsage();
        return 2;
    }
    std::string db_dir = argv[1];
    std::string output_path = argv[2];
    std::size_t batch = 1024;
    std::string cursor;
    pomai::Filter filter{};
    for (int i = 3; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--batch" && i + 1 < argc)
        {
            batch = static_cast<std::size_t>(std::stoull(argv[++i]));
        }
        else if (arg == "--cursor" && i + 1 < argc)
        {
            cursor = argv[++i];
        }
        else if (arg == "--namespace" && i + 1 < argc)
        {
            filter.namespace_id = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        }
        else if (arg == "--tag" && i + 1 < argc)
        {
            filter.require_any_tags.push_back(static_cast<pomai::TagId>(std::stoul(argv[++i])));
        }
    }

    pomai::DbOptions opts;
    opts.wal_dir = db_dir;
    pomai::storage::SnapshotData snapshot;
    pomai::storage::Manifest manifest;
    std::string err;
    if (pomai::storage::RecoverLatestCheckpoint(db_dir, snapshot, manifest, &err))
    {
        opts.dim = snapshot.schema.dim;
        opts.shards = snapshot.schema.shards;
        opts.metric = static_cast<pomai::Metric>(snapshot.schema.metric);
    }
    pomai::PomaiDB db(opts);
    if (!db.Start().ok())
    {
        std::cerr << "Failed to start database\n";
        return 1;
    }

    std::ofstream out(output_path);
    if (!out)
    {
        std::cerr << "Failed to open output: " << output_path << "\n";
        db.Stop();
        return 1;
    }

    pomai::ScanRequest req;
    req.batch_size = batch;
    req.include_vectors = true;
    req.include_metadata = true;
    req.filter = filter;
    req.cursor = cursor;

    std::string next = req.cursor;
    while (true)
    {
        req.cursor = next;
        auto resp_res = db.Scan(req);
        if (!resp_res.ok())
        {
            std::cerr << "Scan failed: " << resp_res.status().msg << "\n";
            db.Stop();
            return 1;
        }
        auto resp = resp_res.move_value();
        if (resp.status != pomai::ScanStatus::Ok)
        {
            std::cerr << "Scan failed: " << resp.error << "\n";
            db.Stop();
            return 1;
        }
        const std::size_t dim = opts.dim;
        for (const auto &item : resp.items)
        {
            out << item.id;
            if (req.include_vectors)
            {
                out << "\t";
                std::size_t offset = item.vector_offset;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    if (d > 0)
                        out << ",";
                    out << resp.vectors[offset + d];
                }
            }
            if (req.include_metadata)
            {
                out << "\t" << item.namespace_id;
                out << "\t";
                for (std::size_t t = 0; t < item.tag_count; ++t)
                {
                    if (t > 0)
                        out << ",";
                    out << resp.tags[item.tag_offset + t];
                }
            }
            out << "\n";
        }
        next = resp.next_cursor;
        if (next.empty())
            break;
    }

    db.Stop();
    return 0;
}
