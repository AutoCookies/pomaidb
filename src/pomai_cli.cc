/*
 * src/pomai_cli.cc
 * Pomai-CLI: PomaiDB interactive shell.
 *
 * Updated: full SQL support + ITERATE handlers that query MEMBRANCE INFO first
 * and decode stored element types (float32, float64, int32, int8, float16)
 * into float32 for human-friendly preview / testing.
 */

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <memory>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#ifdef _WIN32
#include <Winsock2.h>
typedef int socklen_t;
#else
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#define DEFAULT_HOST "127.0.0.1"
#define DEFAULT_PORT 7777

// ANSI Colors
#define ANSI_RESET "\033[0m"
#define ANSI_GREEN "\033[32m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_CYAN "\033[36m"
#define ANSI_RED "\033[31m"

namespace cli_helpers {
    enum class DataType {
        FLOAT32,
        FLOAT64,
        INT32,
        INT8,
        FLOAT16,
        UNKNOWN
    };

    static inline size_t dtype_size(DataType dt) {
        switch (dt) {
            case DataType::FLOAT32: return 4;
            case DataType::FLOAT64: return 8;
            case DataType::INT32: return 4;
            case DataType::INT8: return 1;
            case DataType::FLOAT16: return 2;
            default: return 4;
        }
    }

    static inline DataType parse_dtype_str(const std::string &s) {
        std::string t = s;
        std::transform(t.begin(), t.end(), t.begin(), ::tolower);
        if (t == "float32" || t == "float" || t == "fp32") return DataType::FLOAT32;
        if (t == "float64" || t == "double" || t == "fp64") return DataType::FLOAT64;
        if (t == "int32" || t == "i32") return DataType::INT32;
        if (t == "int8"  || t == "i8") return DataType::INT8;
        if (t == "float16" || t == "fp16" || t == "half") return DataType::FLOAT16;
        return DataType::UNKNOWN;
    }

    // Simple IEEE-754 float16 -> float32 conversion
    static inline float fp16_to_fp32(uint16_t h) {
        uint32_t sign = (h & 0x8000u) << 16;
        uint32_t exp = (h & 0x7C00u) >> 10;
        uint32_t mant = (h & 0x03FFu);

        uint32_t bits;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign;
            } else {
                // normalize subnormal
                exp = 1;
                while ((mant & 0x0400u) == 0) {
                    mant <<= 1;
                    exp--;
                }
                mant &= 0x03FFu;
                uint32_t e = exp + (127 - 15);
                uint32_t m = mant << 13;
                bits = sign | (e << 23) | m;
            }
        } else if (exp == 0x1F) {
            bits = sign | 0x7F800000u | (mant ? (mant << 13) : 0);
        } else {
            uint32_t e = exp + (127 - 15);
            uint32_t m = mant << 13;
            bits = sign | (e << 23) | m;
        }
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    static inline std::string trim(const std::string &s) {
        const char *ws = " \t\r\n";
        size_t b = s.find_first_not_of(ws);
        if (b == std::string::npos) return "";
        size_t e = s.find_last_not_of(ws);
        return s.substr(b, e - b + 1);
    }
}

// -------- PomaiSocket impl --------
class PomaiSocket
{
    int sockfd_;
#ifdef _WIN32
    WSADATA wsaData_;
#endif
public:
    PomaiSocket(const std::string &host, int port) : sockfd_(-1)
    {
#ifdef _WIN32
        WSAStartup(MAKEWORD(2, 2), &wsaData_);
#endif
        sockfd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd_ < 0)
            throw std::runtime_error("Socket creation failed");

        sockaddr_in serv_addr{};
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port);
        if (::inet_pton(AF_INET, host.c_str(), &serv_addr.sin_addr) <= 0)
            throw std::runtime_error("Invalid host");

        if (::connect(sockfd_, (sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
            throw std::runtime_error("Connect failed");
    }
    ~PomaiSocket()
    {
        if (sockfd_ != -1) {
#ifdef _WIN32
            closesocket(sockfd_);
            WSACleanup();
#else
            close(sockfd_);
#endif
        }
    }

    void sendln(const std::string &msg)
    {
        std::string tosend = msg;
        if (!tosend.empty() && tosend.back() != '\n') tosend += '\n';
        const char *ptr = tosend.c_str();
        size_t to_send = tosend.size();
        while (to_send > 0)
        {
            ssize_t sent = ::send(sockfd_, ptr, static_cast<int>(to_send), 0);
            if (sent <= 0) throw std::runtime_error("Send failed");
            ptr += sent;
            to_send -= static_cast<size_t>(sent);
        }
    }

    std::string recv_until(char delimiter)
    {
        std::string out;
        char c;
        while (true)
        {
            int n = ::recv(sockfd_, &c, 1, 0);
            if (n <= 0) break;
            out += c;
            if (c == delimiter) break;
        }
        return out;
    }

    std::vector<char> recv_exact(size_t bytes)
    {
        std::vector<char> buf(bytes);
        size_t received = 0;
        while (received < bytes)
        {
            int n = ::recv(sockfd_, buf.data() + received, static_cast<int>(bytes - received), 0);
            if (n <= 0) throw std::runtime_error("Socket closed during binary read");
            received += static_cast<size_t>(n);
        }
        return buf;
    }

    std::string recv_multiline(int maxlines = 100)
    {
        std::stringstream ss;
        for (int i = 0; i < maxlines; ++i)
        {
            std::string l = recv_until('\n');
            if (l.empty()) break;
            ss << l;
            if (l.find("<END>") != std::string::npos) break;
        }
        return ss.str();
    }

    // Sends GET MEMBRANCE INFO <name>; and returns full textual response (without <END>)
    std::string get_membrance_info(const std::string &name)
    {
        std::string cmd = "GET MEMBRANCE INFO " + name + ";";
        sendln(cmd);
        return recv_multiline();
    }

    // Request binary stream and return parsed header + raw payload bytes.
    // header example: "OK BINARY <count> <dim> <bytes>\n" or "OK BINARY_PAIR <count> <dim> <bytes>\n"
    // Returns tuple: (header_line, payload bytes vector)
    std::pair<std::string, std::vector<char>> request_binary_stream(const std::string &cmd)
    {
        sendln(cmd);
        // 1. read header line
        std::string header = recv_until('\n');
        if (header.empty()) return {header, {}};

        // If header not OK BINARY, read multiline text (error) and return
        if (header.rfind("OK BINARY", 0) != 0 && header.rfind("OK BINARY_PAIR", 0) != 0 && header.rfind("OK BINARY_PAIR", 0) != 0) {
            // read until <END>
            if (header.find("<END>") == std::string::npos) {
                std::string rest = recv_multiline();
                header += rest;
            }
            return {header, {}};
        }

        // parse header tokens
        std::istringstream hs(header);
        std::string tag, subtype;
        size_t count = 0, dim = 0, total_bytes = 0;
        hs >> tag >> subtype >> count >> dim >> total_bytes;
        // read payload
        std::vector<char> payload;
        if (total_bytes > 0) {
            payload = recv_exact(total_bytes);
        }
        // consume trailing text until <END>
        recv_multiline();
        return {header, std::move(payload)};
    }

    int raw_fd() const { return sockfd_; }
};

static void print_help()
{
    std::cout << "Pomai-CLI\nCommands:\n"
              << "  CREATE MEMBRANCE <name> DIM <n> [DATA_TYPE <type>] [RAM <mb>]\n"
              << "  DROP MEMBRANCE <name>\n"
              << "  SHOW MEMBRANCES\n"
              << "  USE <name>\n"
              << "  INSERT INTO <name> VALUES (<label,[v...]>) , ...\n"
              << "  SEARCH <name> QUERY [v...] TOP <k>\n"
              << "  GET MEMBRANCE INFO <name>\n"
              << "  ITERATE <name> <mode> [split] [off] [lim]  -- modes: TRAIN/VAL/TEST/PAIR/TRIPLET\n"
              << "  EXEC SPLIT <name> <tr> <val> <te> [STRATIFIED key | CLUSTER | TEMPORAL key]\n"
              << "  DELETE <name> LABEL <label>\n"
              << "  QUIT\n";
}

// Helper: parse membrance info text to extract feature_dim and data_type
static bool parse_membrance_info(const std::string &info_text, size_t &out_dim, cli_helpers::DataType &out_dtype)
{
    out_dim = 0;
    out_dtype = cli_helpers::DataType::UNKNOWN;
    std::istringstream iss(info_text);
    std::string line;
    while (std::getline(iss, line))
    {
        std::string t = cli_helpers::trim(line);
        // look for "feature_dim:" or "data_type:"
        auto pos_dim = t.find("feature_dim:");
        if (pos_dim != std::string::npos)
        {
            std::string rest = t.substr(pos_dim + std::strlen("feature_dim:"));
            try { out_dim = static_cast<size_t>(std::stoul(cli_helpers::trim(rest))); }
            catch (...) { out_dim = 0; }
        }
        auto pos_dt = t.find("data_type:");
        if (pos_dt != std::string::npos)
        {
            std::string rest = t.substr(pos_dt + std::strlen("data_type:"));
            std::string dt = cli_helpers::trim(rest);
            out_dtype = cli_helpers::parse_dtype_str(dt);
        }
    }
    return out_dim > 0;
}

// Decode a single vector from raw slot bytes (of stored dtype) into float32 vector
static void decode_slot_to_float(const char *slot, size_t dim, cli_helpers::DataType dt, std::vector<float> &out)
{
    out.assign(dim, 0.0f);
    switch (dt)
    {
        case cli_helpers::DataType::FLOAT32:
            std::memcpy(out.data(), slot, dim * sizeof(float));
            break;
        case cli_helpers::DataType::FLOAT64:
        {
            const double *dp = reinterpret_cast<const double *>(slot);
            for (size_t i = 0; i < dim; ++i) out[i] = static_cast<float>(dp[i]);
            break;
        }
        case cli_helpers::DataType::INT32:
        {
            const int32_t *ip = reinterpret_cast<const int32_t *>(slot);
            for (size_t i = 0; i < dim; ++i) out[i] = static_cast<float>(ip[i]);
            break;
        }
        case cli_helpers::DataType::INT8:
        {
            const int8_t *ip = reinterpret_cast<const int8_t *>(slot);
            for (size_t i = 0; i < dim; ++i) out[i] = static_cast<float>(ip[i]);
            break;
        }
        case cli_helpers::DataType::FLOAT16:
        {
            const uint16_t *hp = reinterpret_cast<const uint16_t *>(slot);
            for (size_t i = 0; i < dim; ++i) out[i] = cli_helpers::fp16_to_fp32(hp[i]);
            break;
        }
        default:
            // attempt float32 memcpy
            std::memcpy(out.data(), slot, dim * sizeof(float));
            break;
    }
}

// Generic handler: drain binary stream and optionally preview first vector
static void handle_iterate_binary(PomaiSocket &sock, const std::string &name, const std::string &orig_cmd)
{
    // 1) Fetch membrance info to know dtype/dim
    size_t dim = 0;
    cli_helpers::DataType dtype = cli_helpers::DataType::FLOAT32;
    try {
        std::string info = sock.get_membrance_info(name);
        if (!parse_membrance_info(info, dim, dtype)) {
            // fallback: try to parse from header later
            dim = 0;
            dtype = cli_helpers::DataType::FLOAT32;
        }
    } catch (...) {
        // ignore: we'll rely on header
    }

    // 2) Request binary stream
    auto [header, payload] = sock.request_binary_stream(orig_cmd);
    if (header.empty()) {
        std::cout << ANSI_RED << "No response from server\n" << ANSI_RESET;
        return;
    }

    // If header is error text, print and return
    if (header.rfind("OK BINARY", 0) != 0 && header.rfind("OK BINARY_PAIR", 0) != 0)
    {
        std::cout << header << "\n";
        return;
    }

    // parse header
    std::istringstream hs(header);
    std::string tag, subtype;
    size_t count = 0, hdr_dim = 0, total_bytes = 0;
    hs >> tag >> subtype >> count >> hdr_dim >> total_bytes;
    if (dim == 0) dim = hdr_dim;

    size_t elem_bytes = sizeof(float); // server external format is float32 output; but internal storage may differ only for stored payloads. For ITERATE server always sends float32 external bytes (sql_executor does convert) — but keep decoding safe.
    // If server returns raw stored representation (unlikely for external protocol), we would need dtype & elem size; but sql_executor already converts to float32 before sending ITERATE.
    // So we treat payload as float32 vectors.
    size_t vec_bytes = dim * sizeof(float);
    size_t expected = count * vec_bytes;
    if (payload.size() < expected) {
        // partial or 0 payload; print header and exit
        std::cout << header;
        std::cout << ANSI_YELLOW << "Warning: payload smaller than expected\n" << ANSI_RESET;
    }

    std::cout << ANSI_CYAN << "ITERATE -> entries=" << count << " dim=" << dim << " bytes=" << payload.size() << ANSI_RESET << "\n";

    // Preview first vector (if present)
    if (payload.size() >= vec_bytes && count > 0) {
        std::vector<float> v(dim);
        std::memcpy(v.data(), payload.data(), vec_bytes);
        std::cout << ANSI_GREEN << "Preview first vector (first 8 values): ";
        for (size_t i = 0; i < std::min<size_t>(dim, 8); ++i) {
            std::cout << v[i] << (i+1 < std::min<size_t>(dim,8) ? ", " : "");
        }
        std::cout << ANSI_RESET << "\n";
    }

    // For test-draining we don't print all bytes (could be huge)
    std::cout << "Drained binary payload (" << payload.size() << " bytes).\n";
}

// Handler for PAIR: payload layout = [uint64_t label][vector bytes] * count
static void handle_iterate_pair(PomaiSocket &sock, const std::string &name, const std::string &orig_cmd)
{
    // Get info
    size_t dim = 0;
    cli_helpers::DataType dtype = cli_helpers::DataType::FLOAT32;
    try {
        std::string info = sock.get_membrance_info(name);
        parse_membrance_info(info, dim, dtype);
    } catch (...) {}

    auto [header, payload] = sock.request_binary_stream(orig_cmd);
    if (header.empty()) { std::cout << "No response\n"; return; }
    if (header.rfind("OK BINARY_PAIR", 0) != 0) { std::cout << header; return; }

    std::istringstream hs(header);
    std::string a,b;
    size_t count=0,hdr_dim=0,total_bytes=0;
    hs >> a >> b >> count >> hdr_dim >> total_bytes;
    if (dim == 0) dim = hdr_dim;
    size_t vec_bytes = dim * sizeof(float);
    size_t per = sizeof(uint64_t) + vec_bytes;
    if (payload.size() < per * count) {
        std::cout << ANSI_YELLOW << "Warning: payload smaller than expected\n" << ANSI_RESET;
    }

    std::cout << ANSI_CYAN << "PAIR stream: entries=" << count << " dim=" << dim << ANSI_RESET << "\n";
    // Print first few pairs
    size_t limit_preview = std::min<size_t>(count, 5);
    for (size_t i = 0; i < limit_preview; ++i) {
        size_t off = i * per;
        uint64_t id;
        std::memcpy(&id, payload.data() + off, sizeof(id));
        std::vector<float> v(dim);
        std::memcpy(v.data(), payload.data() + off + sizeof(id), vec_bytes);
        std::cout << "Label: " << id << " Vec[0..min(7,d-1)]: ";
        for (size_t j = 0; j < std::min<size_t>(dim, 8); ++j) {
            std::cout << v[j] << (j+1 < std::min<size_t>(dim,8) ? ", " : "");
        }
        std::cout << "\n";
    }
    std::cout << "Drained PAIR payload (" << payload.size() << " bytes).\n";
}

// Handler for TRIPLET: server returns OK BINARY header with limit * 3 * vec_bytes bytes
static void handle_iterate_triplet(PomaiSocket &sock, const std::string &name, const std::string &orig_cmd)
{
    size_t dim = 0;
    cli_helpers::DataType dtype = cli_helpers::DataType::FLOAT32;
    try { std::string info = sock.get_membrance_info(name); parse_membrance_info(info, dim, dtype); } catch (...) {}

    auto [header, payload] = sock.request_binary_stream(orig_cmd);
    if (header.empty()) { std::cout << "No response\n"; return; }
    if (header.rfind("OK BINARY", 0) != 0) { std::cout << header; return; }

    std::istringstream hs(header);
    std::string t1,t2;
    size_t count=0,hdr_dim=0,total_bytes=0;
    hs >> t1 >> t2 >> count >> hdr_dim >> total_bytes;
    if (dim == 0) dim = hdr_dim;
    size_t vec_bytes = dim * sizeof(float);
    size_t per = vec_bytes * 3;
    std::cout << ANSI_CYAN << "TRIPLET stream: triplets=" << count << " dim=" << dim << ANSI_RESET << "\n";
    if (!payload.empty()) {
        // preview first triplet
        if (payload.size() >= per) {
            std::vector<float> va(dim), vp(dim), vn(dim);
            std::memcpy(va.data(), payload.data(), vec_bytes);
            std::memcpy(vp.data(), payload.data() + vec_bytes, vec_bytes);
            std::memcpy(vn.data(), payload.data() + vec_bytes*2, vec_bytes);
            std::cout << ANSI_GREEN << "Preview anchor[0..5]: ";
            for (size_t i=0;i<std::min<size_t>(6,dim);++i) std::cout << va[i] << (i+1< std::min<size_t>(6,dim) ? ", " : "");
            std::cout << "\n";
        }
    }
    std::cout << "Drained TRIPLET payload (" << payload.size() << " bytes).\n";
}

// Top-level interactive loop
int main(int argc, char **argv)
{
    std::string host = DEFAULT_HOST;
    int port = DEFAULT_PORT;
    if (argc >= 2) host = argv[1];
    if (argc >= 3) port = std::atoi(argv[2]);

    try {
        PomaiSocket sock(host, port);
        std::cout << "Connected to " << host << ":" << port << "\n";
        print_help();

        std::string line;
        while (true)
        {
            std::cout << "> ";
            if (!std::getline(std::cin, line)) break;
            std::string cmd = cli_helpers::trim(line);
            if (cmd.empty()) continue;
            std::string up = cmd;
            std::transform(up.begin(), up.end(), up.begin(), ::toupper);

            if (up == "QUIT" || up == "EXIT") break;
            if (up == "HELP") { print_help(); continue; }

            // Special handling for ITERATE: determine mode and membrance name to choose handler
            std::string upcopy = up;
            if (upcopy.rfind("ITERATE ", 0) == 0)
            {
                // parse tokens
                std::istringstream iss(cmd);
                std::string tok;
                std::vector<std::string> parts;
                while (iss >> tok) parts.push_back(tok);
                if (parts.size() < 3) { std::cout << "ERR: Usage: ITERATE <name> <mode> [split] [off] [lim]\n"; continue; }
                std::string name = parts[1];
                std::string mode = parts[2];
                std::string upper_mode = mode;
                std::transform(upper_mode.begin(), upper_mode.end(), upper_mode.begin(), ::toupper);

                // For robust behavior: fetch membrance info first (so handlers can decode)
                try {
                    std::string info = sock.get_membrance_info(name);
                    // print small summary
                    //std::cout << info << "\n";
                } catch (...) {}

                // Build the command to send (the same user input)
                std::string full_cmd = cmd;
                if (full_cmd.back() != ';') full_cmd += ";";

                if (upper_mode == "PAIR") {
                    handle_iterate_pair(sock, name, full_cmd);
                } else if (upper_mode == "TRIPLET") {
                    handle_iterate_triplet(sock, name, full_cmd);
                } else {
                    handle_iterate_binary(sock, name, full_cmd);
                }
                continue;
            }

            // Default: send line and print multiline response
            // ensure newline
            sock.sendln(cmd);
            std::string resp = sock.recv_multiline();
            std::cout << resp;
        }

        std::cout << "Bye.\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Connection error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}