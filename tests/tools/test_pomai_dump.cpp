#include <catch2/catch.hpp>

#include <pomai/api/pomai_db.h>

#include "tests/common/test_utils.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("pomai_dump style scan writes output", "[tools][dump]")
{
    TempDir dir;
    auto opts = DefaultDbOptions(dir.str(), 5, 1);

    PomaiDB db(opts);
    db.Start();
    auto batch = MakeBatch(8, 5, 0.1f, 1);
    db.UpsertBatch(batch, true).get();

    auto output_path = std::filesystem::path(dir.str()) / "dump.tsv";
    std::ofstream out(output_path);
    REQUIRE(out.good());

    ScanRequest req;
    req.batch_size = 4;
    req.include_vectors = true;
    req.include_metadata = true;

    std::string cursor;
    while (true)
    {
        req.cursor = cursor;
        auto resp = db.Scan(req);
        REQUIRE(resp.status == ScanStatus::Ok);
        for (const auto &item : resp.items)
        {
            out << item.id << "\t";
            for (std::size_t d = 0; d < opts.dim; ++d)
            {
                if (d > 0)
                    out << ",";
                out << resp.vectors[item.vector_offset + d];
            }
            out << "\t" << item.namespace_id;
            out << "\t";
            for (std::size_t t = 0; t < item.tag_count; ++t)
            {
                if (t > 0)
                    out << ",";
                out << resp.tags[item.tag_offset + t];
            }
            out << "\n";
        }
        if (resp.next_cursor.empty())
            break;
        cursor = resp.next_cursor;
    }

    out.close();
    db.Stop();

    std::ifstream in(output_path);
    REQUIRE(in.good());
    std::size_t lines = 0;
    std::string line;
    while (std::getline(in, line))
    {
        if (!line.empty())
            ++lines;
    }
    REQUIRE(lines == batch.size());
}
