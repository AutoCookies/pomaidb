/*
 * src/pomai_cli.cc
 * Pomai-CLI: PomaiDB interactive shell.
 *
 * Updates:
 * - [NEW] Added ITERATE command support (Binary Protocol Handler).
 * - [NEW] Added specialized handlers for ITERATE PAIR and ITERATE TRIPLET.
 * - CLI will choose appropriate binary handler based on the ITERATE mode token.
 *
 * This file is a self-contained usable implementation of the CLI client.
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
        if (!tosend.empty() && tosend.back() != '\n')
            tosend += '\n';
        const char *ptr = tosend.c_str();
        size_t to_send = tosend.size();
        while (to_send > 0)
        {
            int sent = send(sockfd_, ptr, static_cast<int>(to_send), 0);
            if (sent <= 0)
                throw std::runtime_error("Send failed");
            ptr += sent;
            to_send -= static_cast<size_t>(sent);
        }
    }

    // Read until specific char
    std::string recv_until(char delimiter)
    {
        std::string out;
        char c;
        while (true)
        {
            int n = recv(sockfd_, &c, 1, 0);
            if (n <= 0)
                break;
            out += c;
            if (c == delimiter)
                break;
        }
        return out;
    }

    // Read exact number of bytes into buffer (throws on EOF/error)
    std::vector<char> recv_exact(size_t bytes)
    {
        std::vector<char> buf;
        buf.resize(bytes);
        size_t received = 0;
        while (received < bytes)
        {
            int n = recv(sockfd_, buf.data() + received, static_cast<int>(bytes - received), 0);
            if (n <= 0)
                throw std::runtime_error("Socket closed during binary read");
            received += static_cast<size_t>(n);
        }
        return buf;
    }

    // Standard text response reader
    std::string recv_multiline(int maxlines = 100)
    {
        std::stringstream ss;
        for (int i = 0; i < maxlines; ++i)
        {
            std::string l = recv_until('\n');
            if (l.empty())
                break;
            ss << l;
            if (l.find("<END>") != std::string::npos)
                break;
        }
        return ss.str();
    }

    // Generic binary ITERATE handler (drain-only, prints summary)
    void recv_binary_iterate_generic()
    {
        // 1. Read Header
        std::string header = recv_until('\n');
        std::cout << header; // Print header for visibility

        if (header.find("OK BINARY") != 0)
        {
            // Error or unknown format, flush rest as text
            if (header.find("<END>") == std::string::npos)
            {
                std::cout << recv_multiline();
            }
            return;
        }

        // 2. Parse Header
        std::stringstream ss(header);
        std::string tag;
        size_t count = 0, dim = 0, total_bytes = 0;
        ss >> tag >> tag >> count >> dim >> total_bytes;

        std::cout << ANSI_CYAN << ">> Draining " << total_bytes << " bytes of binary payload..." << ANSI_RESET << "\n";

        // 3. Drain binary payload (don't store)
        const size_t BUF_SZ = 64 * 1024;
        std::vector<char> tmp;
        tmp.resize(BUF_SZ);
        size_t remaining = total_bytes;
        while (remaining > 0)
        {
            size_t want = (remaining > BUF_SZ) ? BUF_SZ : remaining;
            int n = recv(sockfd_, tmp.data(), static_cast<int>(want), 0);
            if (n <= 0) throw std::runtime_error("Socket closed during binary read");
            remaining -= static_cast<size_t>(n);
        }

        std::cout << ANSI_GREEN << ">> Binary iterate complete: " << count << " entries, dim=" << dim << ANSI_RESET << "\n";
    }

    // Special handler for ITERATE PAIR: each entry = uint64_t label + float32[dim]
    void recv_binary_iterate_pair()
    {
        // Header
        std::string header = recv_until('\n');
        std::cout << header;
        if (header.find("OK BINARY") != 0)
        {
            if (header.find("<END>") == std::string::npos)
                std::cout << recv_multiline();
            return;
        }

        std::stringstream ss(header);
        std::string tmp;
        size_t count = 0, dim = 0, total_bytes = 0;
        ss >> tmp >> tmp >> count >> dim >> total_bytes;

        const size_t bytes_per_vec = dim * sizeof(float);
        const size_t pair_size = sizeof(uint64_t) + bytes_per_vec;
        // best-effort validation
        if (total_bytes != count * pair_size)
        {
            std::cerr << ANSI_YELLOW << "[WARN] size mismatch: header bytes=" << total_bytes
                      << " expected=" << (count * pair_size) << ANSI_RESET << "\n";
        }

        std::cout << ANSI_CYAN << ">> Receiving " << count << " pairs; dim=" << dim << ", pair_bytes=" << pair_size << ANSI_RESET << "\n";

        size_t got = 0;
        uint64_t first_label = 0;
        std::vector<float> first_vec;
        bool printed_first = false;

        for (size_t i = 0; i < count; ++i)
        {
            // read label
            auto lblb = recv_exact(sizeof(uint64_t));
            uint64_t label = 0;
            std::memcpy(&label, lblb.data(), sizeof(uint64_t));
            // read vector
            auto vbuf = recv_exact(bytes_per_vec);

            if (!printed_first)
            {
                first_label = label;
                first_vec.resize(dim);
                std::memcpy(first_vec.data(), vbuf.data(), bytes_per_vec);
                printed_first = true;
            }
            got++;
        }

        std::cout << ANSI_GREEN << ">> Received " << got << " pairs, dim=" << dim << ANSI_RESET << "\n";
        if (printed_first)
        {
            std::cout << "  -> First pair label(hash)=" << first_label << ", vec[:5]=";
            std::cout << "[";
            for (size_t j = 0; j < std::min<size_t>(5, first_vec.size()); ++j)
            {
                if (j) std::cout << " ";
                std::cout << std::fixed << std::setprecision(1) << first_vec[j];
            }
            std::cout << "]\n";
        }
    }

    // Special handler for ITERATE TRIPLET: each entry = [A][P][N] where each is float32[dim]
    void recv_binary_iterate_triplet()
    {
        std::string header = recv_until('\n');
        std::cout << header;
        if (header.find("OK BINARY") != 0)
        {
            if (header.find("<END>") == std::string::npos)
                std::cout << recv_multiline();
            return;
        }

        std::stringstream ss(header);
        std::string tmp;
        size_t count = 0, dim = 0, total_bytes = 0;
        ss >> tmp >> tmp >> count >> dim >> total_bytes;

        const size_t bytes_per_vec = dim * sizeof(float);
        const size_t triplet_size = 3 * bytes_per_vec;
        if (total_bytes != count * triplet_size)
        {
            std::cerr << ANSI_YELLOW << "[WARN] triplet size mismatch: header bytes=" << total_bytes
                      << " expected=" << (count * triplet_size) << ANSI_RESET << "\n";
        }

        std::cout << ANSI_CYAN << ">> Receiving " << count << " triplets; dim=" << dim << ANSI_RESET << "\n";

        bool printed_first = false;
        std::vector<float> a, p, n;

        for (size_t i = 0; i < count; ++i)
        {
            auto ab = recv_exact(bytes_per_vec);
            auto pb = recv_exact(bytes_per_vec);
            auto nb = recv_exact(bytes_per_vec);

            if (!printed_first)
            {
                a.resize(dim); p.resize(dim); n.resize(dim);
                std::memcpy(a.data(), ab.data(), bytes_per_vec);
                std::memcpy(p.data(), pb.data(), bytes_per_vec);
                std::memcpy(n.data(), nb.data(), bytes_per_vec);
                printed_first = true;
            }
        }

        std::cout << ANSI_GREEN << ">> Received " << count << " triplets (A,P,N), dim=" << dim << ANSI_RESET << "\n";
        if (printed_first)
        {
            std::cout << "  -> Anchor[:5]:   [";
            for (size_t j = 0; j < std::min<size_t>(5, a.size()); ++j)
            {
                if (j) std::cout << " ";
                std::cout << std::fixed << std::setprecision(1) << a[j];
            }
            std::cout << "]\n";

            std::cout << "  -> Positive[:5]: [";
            for (size_t j = 0; j < std::min<size_t>(5, p.size()); ++j)
            {
                if (j) std::cout << " ";
                std::cout << std::fixed << std::setprecision(1) << p[j];
            }
            std::cout << "]\n";

            std::cout << "  -> Negative[:5]: [";
            for (size_t j = 0; j < std::min<size_t>(5, n.size()); ++j)
            {
                if (j) std::cout << " ";
                std::cout << std::fixed << std::setprecision(1) << n[j];
            }
            std::cout << "]\n";
        }
    }
};

static std::string trim(const std::string &s)
{
    size_t l = s.find_first_not_of(" \t\r\n");
    size_t r = s.find_last_not_of(" \t\r\n");
    return (l == std::string::npos || r == std::string::npos) ? "" : s.substr(l, r - l + 1);
}
static std::string uc(const std::string &s)
{
    std::string t = s;
    std::transform(t.begin(), t.end(), t.begin(), ::toupper);
    return t;
}

class PomaiCLI
{
    std::string host_;
    int port_;
    std::unique_ptr<PomaiSocket> sock_;
    std::string current_membr_;

public:
    PomaiCLI(const std::string &host, int port) : host_(host), port_(port), sock_(nullptr) {}
    void run()
    {
        try
        {
            sock_ = std::make_unique<PomaiSocket>(host_, port_);
            print_welcome();
            prompt();
        }
        catch (const std::exception &e)
        {
            std::cerr << ANSI_RED << "Connect error: " << e.what() << ANSI_RESET << "\n";
        }
    }

private:
    void print_welcome()
    {
        std::cout
            << "\n============================================\n"
            << ANSI_YELLOW << "  PomaiDB CLI — AI Data Engine Mode" << ANSI_RESET << "\n"
            << "  Type 'help;' for syntax, '\\q' to exit.\n"
            << "============================================\n";
    }
    void prompt()
    {
        std::string inbuf;
        while (true)
        {
            std::cout << (current_membr_.empty() ? ANSI_CYAN "[pomai]> " ANSI_RESET : (ANSI_CYAN "[pomai/" + current_membr_ + "]> " ANSI_RESET));
            std::getline(std::cin, inbuf);
            if (std::cin.eof()) break;
            std::string cmd = trim(inbuf);
            if (cmd.empty()) continue;
            std::string up = uc(cmd);

            if (up == "EXIT;" || up == "\\Q" || up == "QUIT;") break;
            
            if (up == "HELP;") {
                print_help();
                continue;
            }

            // USE <name>
            if (up.rfind("USE ", 0) == 0 && up.back() == ';') {
                std::stringstream ss(cmd);
                std::string tmp, name;
                ss >> tmp >> name;
                if (!name.empty() && name.back() == ';') name.pop_back();
                current_membr_ = name;
                std::cout << "Switched to: " << ANSI_GREEN << current_membr_ << ANSI_RESET << "\n";
                continue;
            }

            // ITERATE: dispatch to appropriate binary handler based on mode token
            if (up.rfind("ITERATE ", 0) == 0 && up.back() == ';') {
                try {
                    // determine mode token (PAIR/TRIPLET or TRAIN/VAL/TEST)
                    std::stringstream ss(cmd);
                    std::string tok;
                    // ITERATE <name> <mode> ...
                    ss >> tok; // ITERATE
                    ss >> tok; // name
                    std::string mode;
                    if (!(ss >> mode)) mode = "";
                    std::string mode_uc = uc(mode);

                    // send command first
                    sock_->sendln(cmd);

                    if (mode_uc == "PAIR") {
                        sock_->recv_binary_iterate_pair();
                    } else if (mode_uc == "TRIPLET") {
                        sock_->recv_binary_iterate_triplet();
                    } else {
                        // generic binary drain (TRAIN/VAL/TEST will come here)
                        sock_->recv_binary_iterate_generic();
                    }
                } catch (const std::exception &e) {
                    std::cerr << ANSI_RED << "Error: " << e.what() << ANSI_RESET << "\n";
                }
                continue;
            }

            // Default Text Command Handler
            try {
                // Prepend current membrance for shortcuts
                if (!current_membr_.empty()) {
                    std::string upcmd = uc(cmd);
                    if (upcmd.rfind("INSERT VALUES", 0) == 0) cmd = "INSERT INTO " + current_membr_ + " " + cmd.substr(6);
                    else if (upcmd.rfind("SEARCH QUERY", 0) == 0) cmd = "SEARCH " + current_membr_ + " " + cmd.substr(6);
                    else if (upcmd.rfind("GET LABEL", 0) == 0) cmd = "GET " + current_membr_ + " " + cmd.substr(3);
                }

                sock_->sendln(cmd);
                std::cout << sock_->recv_multiline();
            } catch (const std::exception &e) {
                std::cerr << ANSI_RED << "Network Error: " << e.what() << ANSI_RESET << "\n";
                break; // Exit on network fail
            }
        }
        std::cout << "Bye!\n";
    }

    void print_help()
    {
        std::cout << ANSI_YELLOW << R"(
Commands:
  DDL:
    CREATE MEMBRANCE <name> DIM <n> [RAM <mb>];
    DROP MEMBRANCE <name>;
    EXEC SPLIT <name> <train> <val> <test>;  -- Split dataset indices

  DML:
    INSERT INTO <name> VALUES (<lbl>,[v...]);
    SEARCH <name> QUERY ([v...]) TOP k;
    ITERATE <name> <TRAIN|VAL|TEST> <off> <lim>; -- Stream binary data
    ITERATE <name> PAIR <off> <lim>;  -- Stream (uint64 label + vector)
    ITERATE <name> TRIPLET <off> <lim>; -- Stream triplets (A,P,N)

  Misc:
    USE <name>;
    GET MEMBRANCE INFO [<name>];
)" << ANSI_RESET << "\n";
    }
};

int main(int argc, char **argv)
{
    std::string host = DEFAULT_HOST;
    int port = DEFAULT_PORT;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-h" && i + 1 < argc) host = argv[++i];
        if (std::string(argv[i]) == "-p" && i + 1 < argc) port = std::stoi(argv[++i]);
    }
    PomaiCLI cli(host, port);
    cli.run();
    return 0;
}