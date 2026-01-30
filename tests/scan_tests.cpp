#include <pomai/api/pomai_db.h>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_scan.XXXXXX";
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
        pomai::DbOptions opts;
        opts.dim = 2;
        opts.shards = 1;
        opts.wal_dir = dir;
        pomai::PomaiDB db(opts);
        db.Start();

        std::vector<pomai::UpsertRequest> batch;
        for (std::size_t i = 0; i < 10; ++i)
        {
            pomai::UpsertRequest r;
            r.id = static_cast<pomai::Id>(100 + i);
            r.vec.data = {static_cast<float>(i), static_cast<float>(i + 1)};
            r.metadata.namespace_id = (i % 2 == 0) ? 1 : 2;
            r.metadata.tag_ids = {static_cast<pomai::TagId>(i)};
            batch.push_back(std::move(r));
        }
        db.UpsertBatch(batch, true).get();

        pomai::ScanRequest req;
        req.batch_size = 3;
        std::unordered_set<pomai::Id> ids;
        std::string cursor;
        std::size_t seen = 0;
        while (true)
        {
            req.cursor = cursor;
            auto resp = db.Scan(req);
            if (resp.status != pomai::ScanStatus::Ok)
            {
                std::cerr << "Scan failed: " << resp.error << "\n";
                ++failures;
                break;
            }
            for (const auto &item : resp.items)
            {
                if (!ids.insert(item.id).second)
                {
                    std::cerr << "Duplicate id in scan\n";
                    ++failures;
                }
                ++seen;
            }
            cursor = resp.next_cursor;
            if (cursor.empty())
                break;
        }
        if (seen != 10)
        {
            std::cerr << "Expected 10 rows, got " << seen << "\n";
            ++failures;
        }

        pomai::ScanRequest filter_req;
        filter_req.batch_size = 5;
        filter_req.filter.namespace_id = 1;
        std::size_t filtered = 0;
        cursor.clear();
        while (true)
        {
            filter_req.cursor = cursor;
            auto resp = db.Scan(filter_req);
            if (resp.status != pomai::ScanStatus::Ok)
            {
                std::cerr << "Filtered scan failed: " << resp.error << "\n";
                ++failures;
                break;
            }
            filtered += resp.items.size();
            cursor = resp.next_cursor;
            if (cursor.empty())
                break;
        }
        if (filtered != 5)
        {
            std::cerr << "Expected 5 filtered rows, got " << filtered << "\n";
            ++failures;
        }

        db.Stop();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Scan test exception: " << e.what() << "\n";
        ++failures;
    }

    RemoveDir(dir);
    return failures == 0 ? 0 : 1;
}
