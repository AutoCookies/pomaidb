#pragma once
#include <filesystem>
#include <random>
#include <string>

namespace pomai::test
{

    inline std::string TempDir(const std::string &prefix)
    {
        namespace fs = std::filesystem;
        auto base = fs::temp_directory_path();

        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<unsigned long long> dist;

        for (int i = 0; i < 100; ++i)
        {
            auto p = base / (prefix + "-" + std::to_string(dist(gen)));
            if (!fs::exists(p))
            {
                fs::create_directories(p);
                return p.string();
            }
        }
        // Fallback deterministic.
        auto p = base / (prefix + "-fallback");
        fs::create_directories(p);
        return p.string();
    }

} // namespace pomai::test
