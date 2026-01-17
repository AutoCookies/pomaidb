/*
 * src/pomai_cli.cc
 * Pomai-CLI: PomaiDB interactive shell.
 *
 * Updates:
 * - [NEW] Added ITERATE command support (Binary Protocol Handler).
 * Prevents CLI freezing when receiving raw binary data.
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
        if (sockfd_ != -1)
#ifdef _WIN32
            closesocket(sockfd_);
        WSACleanup();
#else
            close(sockfd_);
#endif
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
            int sent = send(sockfd_, ptr, to_send, 0);
            if (sent <= 0)
                throw std::runtime_error("Send failed");
            ptr += sent;
            to_send -= sent;
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

    // [NEW] Special handler for ITERATE command
    // Protocol: OK BINARY <count> <dim> <bytes>\n[RAW BYTES]
    void recv_binary_iterate()
    {
        // 1. Read Header
        std::string header = recv_until('\n');
        std::cout << header; // Print header

        if (header.find("OK BINARY") != 0) {
            // Error or unknown format, maybe text error
            if (header.find("<END>") == std::string::npos) {
                 std::cout << recv_multiline(); // Flush rest if error
            }
            return;
        }

        // 2. Parse Header
        std::stringstream ss(header);
        std::string tag;
        size_t count, dim, total_bytes;
        ss >> tag >> tag >> count >> dim >> total_bytes;

        std::cout << ANSI_CYAN << ">> Receiving " << total_bytes << " bytes of binary data..." << ANSI_RESET << "\n";

        // 3. Consume Binary Payload (Discard or Count)
        // CLI shouldn't print binary to TTY. We just drain the socket.
        size_t received = 0;
        char buf[4096];
        while (received < total_bytes) {
            size_t want = total_bytes - received;
            if (want > sizeof(buf)) want = sizeof(buf);
            
            int n = recv(sockfd_, buf, want, 0);
            if (n <= 0) throw std::runtime_error("Socket closed during binary read");
            received += n;
        }

        std::cout << ANSI_GREEN << ">> Successfully received " << count << " vectors (" << dim << "-dim)." << ANSI_RESET << "\n";
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
                if (name.back() == ';') name.pop_back();
                current_membr_ = name;
                std::cout << "Switched to: " << ANSI_GREEN << current_membr_ << ANSI_RESET << "\n";
                continue;
            }

            // [NEW] ITERATE Command Handler
            if (up.rfind("ITERATE ", 0) == 0 && up.back() == ';') {
                try {
                    sock_->sendln(cmd);
                    sock_->recv_binary_iterate();
                } catch (const std::exception &e) {
                    std::cerr << ANSI_RED << "Error: " << e.what() << ANSI_RESET << "\n";
                }
                continue;
            }

            // Default Text Command Handler
            try {
                // Prepend current membrance for shortcuts
                if (!current_membr_.empty()) {
                    if (up.rfind("INSERT VALUES", 0) == 0) cmd = "INSERT INTO " + current_membr_ + " " + cmd.substr(6);
                    else if (up.rfind("SEARCH QUERY", 0) == 0) cmd = "SEARCH " + current_membr_ + " " + cmd.substr(6);
                    else if (up.rfind("GET LABEL", 0) == 0) cmd = "GET " + current_membr_ + " " + cmd.substr(3);
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
    ITERATE <name> <TRAIN|TEST> <off> <lim>; -- Stream binary data

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