#pragma once

#include <cstdint>
#include <cstddef>
#include <charconv>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <system_error>
#include <unistd.h>
#include <fcntl.h>
#include <array>
#include <iomanip>
#include <sstream>

#include "src/core/metadata_index.h"
#include "src/external/xxhash.h"

namespace pomai::server::utils
{
    static constexpr const char *WHITESPACE = " \t\r\n";

    [[nodiscard]] inline std::string_view trim_sv(std::string_view v) noexcept
    {
        size_t b = v.find_first_not_of(WHITESPACE);
        if (b == std::string_view::npos)
            return {};
        size_t e = v.find_last_not_of(WHITESPACE);
        return v.substr(b, e - b + 1);
    }

    [[nodiscard]] inline std::string trim(const std::string &s)
    {
        return std::string(trim_sv(s));
    }

    [[nodiscard]] inline std::vector<std::string_view> split_ws_sv(std::string_view s)
    {
        std::vector<std::string_view> result;
        result.reserve(8);
        size_t start = s.find_first_not_of(WHITESPACE);
        while (start != std::string_view::npos)
        {
            size_t end = s.find_first_of(WHITESPACE, start);
            if (end == std::string_view::npos)
            {
                result.emplace_back(s.substr(start));
                break;
            }
            result.emplace_back(s.substr(start, end - start));
            start = s.find_first_not_of(WHITESPACE, end);
        }
        return result;
    }

    [[nodiscard]] inline std::vector<std::string> split_ws(const std::string &s)
    {
        auto svs = split_ws_sv(s);
        std::vector<std::string> ret;
        ret.reserve(svs.size());
        for (auto sv : svs)
            ret.emplace_back(sv);
        return ret;
    }

    [[nodiscard]] inline std::string to_upper(std::string s)
    {
        for (char &c : s)
            if (c >= 'a' && c <= 'z')
                c -= 32;
        return s;
    }

    [[nodiscard]] inline bool iequals(std::string_view a, std::string_view b) noexcept
    {
        if (a.size() != b.size())
            return false;
        for (size_t i = 0; i < a.size(); ++i)
        {
            char c1 = a[i] >= 'a' && a[i] <= 'z' ? a[i] - 32 : a[i];
            char c2 = b[i] >= 'a' && b[i] <= 'z' ? b[i] - 32 : b[i];
            if (c1 != c2)
                return false;
        }
        return true;
    }

    static inline uint64_t hash_key(std::string_view k)
    {
        return XXH3_64bits(k.data(), k.size());
    }

    static inline uint64_t hash_label(std::string_view s)
    {
        return XXH3_64bits(s.data(), s.size());
    }

    static inline bool parse_vector(std::string_view s, std::vector<float> &out)
    {
        out.clear();
        if (s.empty())
            return false;

        size_t commas = std::count(s.begin(), s.end(), ',');
        out.reserve(commas + 1);

        const char *ptr = s.data();
        const char *end = ptr + s.size();

        while (ptr < end)
        {
            while (ptr < end && (*ptr <= ' ' || *ptr == ',' || *ptr == '[' || *ptr == ']'))
            {
                ptr++;
            }
            if (ptr == end)
                break;

            float val;
            auto [next_ptr, ec] = std::from_chars(ptr, end, val);

            if (ec == std::errc())
            {
                out.push_back(val);
                ptr = next_ptr;
            }
            else
            {
                ptr++;
            }
        }
        return !out.empty();
    }

    static inline std::vector<pomai::core::Tag> parse_tags_list(std::string_view s)
    {
        std::vector<pomai::core::Tag> out;
        if (s.empty())
            return out;

        const char *ptr = s.data();
        const char *end = ptr + s.size();

        while (ptr < end)
        {
            const char *token_end = ptr;
            while (token_end < end && *token_end != ',')
                token_end++;

            std::string_view token(ptr, token_end - ptr);
            token = trim_sv(token);

            size_t colon = token.find(':');
            if (colon != std::string_view::npos)
            {
                std::string_view k = trim_sv(token.substr(0, colon));
                std::string_view v = trim_sv(token.substr(colon + 1));

                if (v.size() >= 2 && ((v.front() == '\'' && v.back() == '\'') ||
                                      (v.front() == '"' && v.back() == '"')))
                {
                    v = v.substr(1, v.size() - 2);
                }

                if (!k.empty() && !v.empty())
                {
                    out.emplace_back(pomai::core::Tag{std::string(k), std::string(v)});
                }
            }

            ptr = token_end;
            if (ptr < end)
                ptr++;
        }
        return out;
    }

    static inline std::string bytes_human(uint64_t bytes)
    {
        char buf[64];
        double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        int n = std::snprintf(buf, sizeof(buf), "%llu (%.2f GB)", (unsigned long long)bytes, gb);
        return std::string(buf, n > 0 ? n : 0);
    }

    static inline bool read_proc_stat(uint64_t &idle, uint64_t &total)
    {
        int fd = open("/proc/stat", O_RDONLY);
        if (fd < 0)
            return false;

        char buffer[1024];
        ssize_t n = read(fd, buffer, sizeof(buffer) - 1);
        close(fd);

        if (n <= 0)
            return false;
        buffer[n] = '\0';

        if (strncmp(buffer, "cpu ", 4) != 0)
            return false;

        const char *p = buffer + 4;
        uint64_t sum = 0;
        idle = 0;

        for (int i = 0; i < 10; ++i)
        {
            while (*p == ' ')
                p++;
            if (*p < '0' || *p > '9')
                break;

            uint64_t val = 0;
            auto res = std::from_chars(p, buffer + n, val);
            if (res.ec != std::errc())
                break;

            sum += val;
            if (i == 3)
                idle = val;

            p = res.ptr;
        }

        total = sum;
        return true;
    }
}