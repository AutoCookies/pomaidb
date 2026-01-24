#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>
#include <fstream>
#include <iomanip>
#include "src/core/metadata_index.h" // Để dùng struct Tag

#pragma once

namespace pomai::server::utils
{

    static inline std::string trim(const std::string &s)
    {
        const char *ws = " \t\r\n";
        size_t b = s.find_first_not_of(ws);
        if (b == std::string::npos)
            return "";
        size_t e = s.find_last_not_of(ws);
        return s.substr(b, e - b + 1);
    }

    static inline std::string_view trim_sv(std::string_view v)
    {
        const char *ws = " \t\r\n";
        size_t b = 0;
        while (b < v.size() && std::strchr(ws, static_cast<unsigned char>(v[b])))
            ++b;
        size_t e = v.size();
        while (e > b && std::strchr(ws, static_cast<unsigned char>(v[e - 1])))
            --e;
        return v.substr(b, e - b);
    }

    static inline std::vector<std::string> split_ws(const std::string &s)
    {
        std::istringstream iss(s);
        std::vector<std::string> out;
        std::string t;
        while (iss >> t)
            out.push_back(t);
        return out;
    }

    static inline std::string to_upper(const std::string &s)
    {
        std::string r = s;
        for (char &c : r)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        return r;
    }

    static uint64_t fnv1a_hash(const void *data, size_t len)
    {
        const uint8_t *bytes = reinterpret_cast<const uint8_t *>(data);
        uint64_t h = 14695981039346656037ULL;
        for (size_t i = 0; i < len; ++i)
        {
            h ^= bytes[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

    static inline uint64_t hash_label(const std::string &s)
    {
        uint64_t h = 14695981039346656037ULL;
        for (char c : s)
        {
            h ^= static_cast<uint64_t>(c);
            h *= 1099511628211ULL;
        }
        return h;
    }

    static inline uint64_t hash_key(const std::string &k) { return fnv1a_hash(k.data(), k.size()); }

    static inline void append_end_marker(std::string &s)
    {
        if (s.empty() || s.back() != '\n')
            s.push_back('\n');
        s += "<END>\n";
    }

    // Fast parse helpers: use string_view and strtof into a small stack buffer.
    // Token is a substring view representing the number text (no surrounding whitespace).
    static inline float parse_float_token_sv(std::string_view tok)
    {
        // small stack buffer optimization
        constexpr size_t STACK_BUF_SZ = 128;
        if (tok.empty())
            return 0.0f;
        if (tok.size() < STACK_BUF_SZ)
        {
            char tmp[STACK_BUF_SZ];
            std::memcpy(tmp, tok.data(), tok.size());
            tmp[tok.size()] = '\0';
            char *endp = nullptr;
            float v = std::strtof(tmp, &endp);
            (void)endp;
            return v;
        }
        else
        {
            // fallback allocate
            std::string tmp(tok);
            char *endp = nullptr;
            float v = std::strtof(tmp.c_str(), &endp);
            (void)endp;
            return v;
        }
    }

    static inline bool parse_vector(std::string s, std::vector<float> &out)
    {
        // Remove brackets if any
        if (!s.empty() && s.front() == '[')
            s = s.substr(1);
        if (!s.empty() && s.back() == ']')
            s.pop_back();

        std::replace(s.begin(), s.end(), ',', ' ');
        std::istringstream iss(s);
        float v;
        while (iss >> v)
        {
            out.push_back(v);
        }
        return !out.empty();
    }

    // Parse CSV list of floats using string_view tokens and parse_float_token_sv.
    static std::vector<float> parse_float_list_sv(std::string_view csv)
    {
        std::vector<float> out;
        size_t i = 0;
        while (i < csv.size())
        {
            // find comma
            size_t j = csv.find(',', i);
            std::string_view tok = (j == std::string::npos) ? csv.substr(i) : csv.substr(i, j - i);
            tok = trim_sv(tok);
            if (!tok.empty())
            {
                try
                {
                    float v = parse_float_token_sv(tok);
                    out.push_back(v);
                }
                catch (...)
                {
                    // ignore parse error for this token
                }
            }
            if (j == std::string::npos)
                break;
            i = j + 1;
        }
        return out;
    }

    // Parse tags in form: k1:v1,k2:v2  -> returns vector<Tag>
    static std::vector<pomai::core::Tag> parse_tags_list(const std::string &s)
    {
        std::vector<pomai::core::Tag> out;
        size_t i = 0;
        while (i < s.size())
        {
            // find comma-separated token
            size_t j = s.find(',', i);
            std::string tok = (j == std::string::npos) ? s.substr(i) : s.substr(i, j - i);
            // split by ':' first occurrence
            size_t c = tok.find(':');
            if (c != std::string::npos)
            {
                std::string k = trim(tok.substr(0, c));
                std::string v = trim(tok.substr(c + 1));
                // strip optional quotes
                if (v.size() >= 2 && ((v.front() == '\'' && v.back() == '\'') || (v.front() == '\"' && v.back() == '\"')))
                    v = v.substr(1, v.size() - 2);
                if (!k.empty() && !v.empty())
                    out.push_back(pomai::core::Tag{k, v});
            }
            if (j == std::string::npos)
                break;
            i = j + 1;
        }
        return out;
    }

    // Helper: format bytes -> GB string with 2 decimals
    static std::string bytes_human(uint64_t bytes)
    {
        std::ostringstream ss;
        double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        ss << bytes << " (" << std::fixed << std::setprecision(2) << gb << " GB)";
        return ss.str();
    }

    static bool read_proc_stat(uint64_t &idle, uint64_t &total)
    {
        std::ifstream f("/proc/stat");
        if (!f.good())
            return false;
        std::string line;
        std::getline(f, line);
        f.close();
        // line format: cpu  user nice system idle iowait irq softirq steal guest guest_nice
        std::istringstream iss(line);
        std::string cpu_label;
        iss >> cpu_label;
        uint64_t value;
        uint64_t sum = 0;
        idle = 0;
        for (int i = 0; iss >> value; ++i)
        {
            sum += value;
            // idle is value at position 3 (0-based: user(0),nice(1),system(2),idle(3))
            if (i == 3)
                idle = value;
        }
        total = sum;
        return true;
    }

} // namespace pomai::server::utils