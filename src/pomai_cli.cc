/*
 * src/pomai_cli.cc
 * Pomai-CLI: PomaiDB interactive shell, PostgreSQL-like, user-friendly 10/10.
 *
 * Features:
 * - Kết nối tới PomaiDB server thông qua TCP (text protocol chuẩn SQL-like).
 * - Cú pháp: CREATE MEMBRANCE, USE, INSERT INTO, SEARCH, GET, SHOW, DROP, EXIT,...
 * - Tự động lưu màng đang chọn (CURRENT membrance).
 * - Hỗ trợ tất cả thao tác vector chuẩn (CLI/PomaiSQL).
 * - [NEW] Hỗ trợ Metadata: INSERT ... TAGS(...) và SEARCH ... WHERE ...
 * - Dễ mở rộng UI (support history, highlight...).
 *
 * 2024 - Lựu CLI cho Pomai vector database.
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
#include <cstring>
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

// ANSI Colors for CLI
#define ANSI_RESET "\033[0m"
#define ANSI_GREEN "\033[32m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_CYAN "\033[36m"

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

    std::string recv_until_prompt(char prompt = '\n')
    {
        std::string out;
        char c;
        while (true)
        {
            int n = recv(sockfd_, &c, 1, 0);
            if (n <= 0)
                break;
            out += c;
            if (c == prompt)
                break;
        }
        return out;
    }

    std::string recv_multiline(int maxlines = 100)
    { // for listing
        std::stringstream ss;
        for (int i = 0; i < maxlines; ++i)
        {
            std::string l = recv_until_prompt('\n');
            if (l.empty())
                break;
            ss << l;
            if (l.find("<END>") != std::string::npos)
                break;
        }
        return ss.str();
    }
};

/** Trim and uppercase utils */
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

/** CLI app */
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
            std::cerr << "Connect error: " << e.what() << "\n";
        }
    }

private:
    void print_welcome()
    {
        std::cout
            << "\n============================================\n"
            << ANSI_YELLOW << "  PomaiDB CLI — PomaiSQL Mode (Metadata V2)" << ANSI_RESET << "\n"
            << "  Type 'help;' for syntax, '\\q' to exit.\n"
            << "============================================\n"
            << "(Tip: End command with ;)\n";
    }
    void prompt()
    {
        std::string inbuf;
        while (true)
        {
            std::cout << (current_membr_.empty() ? ANSI_CYAN "[pomai]> " ANSI_RESET : (ANSI_CYAN "[pomai/" + current_membr_ + "]> " ANSI_RESET));
            std::getline(std::cin, inbuf);
            if (std::cin.eof())
                break;
            std::string cmd = trim(inbuf);
            if (cmd.empty())
                continue;
            std::string up = uc(cmd);

            if (up == "EXIT;" || up == "\\Q" || up == "QUIT;")
                break;
            if (up == "HELP;")
            {
                print_help();
                continue;
            }

            // Local expand for USE
            if (up.rfind("USE ", 0) == 0 && up.back() == ';')
            {
                std::istringstream iss(cmd);
                std::string u, name;
                iss >> u >> name;
                current_membr_ = name.substr(0, name.length() - 1); // remove ;
                std::cout << "Switched to membrance: " << ANSI_GREEN << current_membr_ << ANSI_RESET << "\n";
                continue;
            }

            // For commands affecting membrance, highlight info for user
            if (up.rfind("CREATE ", 0) == 0 && up.back() == ';')
            {
                // Forward to server
                sock_->sendln(cmd);
                std::cout << sock_->recv_multiline();
                continue;
            }
            if (up.rfind("DROP ", 0) == 0 && up.back() == ';')
            {
                sock_->sendln(cmd);
                std::cout << sock_->recv_multiline();
                continue;
            }
            if (up == "SHOW MEMBRANCES;")
            {
                sock_->sendln(cmd);
                std::cout << sock_->recv_multiline();
                continue;
            }

            // Insert: INSERT INTO <membr> VALUES (<label>, [f,f,...]);
            if (up.rfind("INSERT INTO ", 0) == 0 && up.back() == ';')
            {
                sock_->sendln(cmd);
                std::cout << sock_->recv_multiline();
                continue;
            }

            // Search: SEARCH <membr> QUERY ([f,...]) TOP k;
            if (up.rfind("SEARCH ", 0) == 0 && up.back() == ';')
            {
                sock_->sendln(cmd);
                std::cout << sock_->recv_multiline();
                continue;
            }

            // --- NEW: GET MEMBRANCE INFO support in CLI ---
            // Supported client forms:
            //   GET MEMBRANCE INFO;
            //   GET MEMBRANCE INFO <name>;
            //   GET MEMBRANCE <name> INFO;
            if (up.rfind("GET MEMBRANCE", 0) == 0 && up.back() == ';')
            {
                // If user supplied "GET MEMBRANCE INFO;" and no current membrance chosen -> prompt error locally
                // but server can also handle it; we just forward and pretty-print the response.
                try
                {
                    sock_->sendln(cmd);
                    std::string raw = sock_->recv_multiline();

                    // Try to parse key fields for nicer display
                    std::istringstream iss(raw);
                    std::string line;
                    std::string name;
                    long long dim = -1;
                    long long num_vectors = -1;
                    unsigned long long disk_bytes = 0;
                    double disk_gb = 0.0;
                    long long ram_mb = -1;

                    while (std::getline(iss, line))
                    {
                        std::string t = trim(line);
                        if (t.empty())
                            continue;
                        // Accept both "MEMBRANCE:" and "MEMBRANCE_INFO" prefixes used by different server versions
                        if (t.rfind("MEMBRANCE:", 0) == 0)
                        {
                            name = trim(t.substr(strlen("MEMBRANCE:")));
                        }
                        else if (t.rfind("MEMBRANCE_INFO", 0) == 0)
                        {
                            // some legacy test output used "MEMBRANCE_INFO <name>"
                            std::istringstream ls(t);
                            std::string hdr;
                            ls >> hdr >> name;
                        }
                        else if (t.rfind("dim:", 0) == 0)
                        {
                            std::istringstream ls(t.substr(4));
                            ls >> dim;
                        }
                        else if (t.rfind("num_vectors:", 0) == 0)
                        {
                            std::istringstream ls(t.substr(strlen("num_vectors:")));
                            ls >> num_vectors;
                        }
                        else if (t.rfind("num_vectors (approx):", 0) == 0)
                        {
                            std::istringstream ls(t.substr(strlen("num_vectors (approx):")));
                            ls >> num_vectors;
                        }
                        else if (t.rfind("disk_bytes:", 0) == 0)
                        {
                            // "disk_bytes: <n> (<n GB>)"
                            size_t pos = t.find(':');
                            if (pos != std::string::npos)
                            {
                                std::string rest = trim(t.substr(pos + 1));
                                std::istringstream ls(rest);
                                ls >> disk_bytes;
                                // optionally compute disk_gb
                                disk_gb = static_cast<double>(disk_bytes) / (1024.0 * 1024.0 * 1024.0);
                            }
                        }
                        else if (t.rfind("disk_gb:", 0) == 0)
                        {
                            std::istringstream ls(t.substr(strlen("disk_gb:")));
                            ls >> disk_gb;
                        }
                        else if (t.rfind("ram_mb_configured:", 0) == 0)
                        {
                            std::istringstream ls(t.substr(strlen("ram_mb_configured:")));
                            ls >> ram_mb;
                        }
                    }

                    // Print parsed summary if we found something recognizable, else print raw
                    if (!name.empty() || dim >= 0 || num_vectors >= 0 || disk_bytes > 0)
                    {
                        std::cout << "Membrance info:\n";
                        if (!name.empty())
                            std::cout << "  name: " << name << "\n";
                        if (dim >= 0)
                            std::cout << "  dim: " << dim << "\n";
                        if (num_vectors >= 0)
                            std::cout << "  num_vectors: " << num_vectors << "\n";
                        else
                            std::cout << "  num_vectors: unknown\n";
                        std::cout << "  disk_bytes: " << disk_bytes << " (" << std::fixed << std::setprecision(3) << disk_gb << " GB)\n";
                        if (ram_mb >= 0)
                            std::cout << "  ram_mb_configured: " << ram_mb << "\n";
                        // also keep raw for debugging
                        std::cout << "(raw):\n"
                                  << raw;
                    }
                    else
                    {
                        // fallback: print raw server response
                        std::cout << raw;
                    }
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error sending GET MEMBRANCE INFO: " << e.what() << "\n";
                }
                continue;
            }

            // Get: GET <membr> LABEL <label>;
            if (up.rfind("GET ", 0) == 0 && up.back() == ';')
            {
                sock_->sendln(cmd);
                std::cout << sock_->recv_multiline();
                continue;
            }

            // --- SHORT SYNTAX SUPPORT (Prepend current_membr_) ---

            // Support: INSERT VALUES (...) [TAGS (...)]
            if (current_membr_ != "" && up.rfind("INSERT VALUES", 0) == 0 && up.back() == ';')
            {
                std::ostringstream oss;
                // substr(6) removes "INSERT", keeps " VALUES..."
                // Result: "INSERT INTO <membr> VALUES..."
                oss << "INSERT INTO " << current_membr_ << " " << cmd.substr(6);
                sock_->sendln(oss.str());
                std::cout << sock_->recv_multiline();
                continue;
            }

            // Support: SEARCH QUERY (...) [WHERE ...]
            if (current_membr_ != "" && up.rfind("SEARCH QUERY", 0) == 0 && up.back() == ';')
            {
                std::ostringstream oss;
                // substr(6) removes "SEARCH", keeps " QUERY..."
                // Result: "SEARCH <membr> QUERY..."
                oss << "SEARCH " << current_membr_ << " " << cmd.substr(6);
                sock_->sendln(oss.str());
                std::cout << sock_->recv_multiline();
                continue;
            }

            if (current_membr_ != "" && up.rfind("GET LABEL", 0) == 0 && up.back() == ';')
            {
                std::ostringstream oss;
                oss << "GET " << current_membr_ << " " << cmd.substr(3);
                sock_->sendln(oss.str());
                std::cout << sock_->recv_multiline();
                continue;
            }

            // Forward any other command as raw text
            sock_->sendln(cmd);
            std::cout << sock_->recv_multiline();
        }
        std::cout << "Bye Pomai!\n";
    }

    void print_help()
    {
        std::cout << ANSI_YELLOW << R"(
Available PomaiSQL commands:
  DDL:
    - CREATE MEMBRANCE <name> DIM <n> [RAM <mb>];
    - DROP MEMBRANCE <name>;
    - SHOW MEMBRANCES;
    - USE <name>;

  DML:
    - INSERT INTO <name> VALUES (<label>, [...]) [TAGS (key:val, ...)];
    - SEARCH <name> QUERY ([...]) [WHERE key='val'] TOP k;
    - GET <name> LABEL <label>;
    - GET MEMBRANCE INFO [<name>];
    - DELETE <name> LABEL <label>;

  Shortcuts (when USE <name> is active):
    - INSERT VALUES (...) [TAGS (...)];
    - SEARCH QUERY ([...]) [WHERE ...] TOP k;
    - GET LABEL <label>;

  Examples:
    > CREATE MEMBRANCE products DIM 128 RAM 256;
    > USE products;
    > INSERT VALUES (shoe_1, [0.1, 0.2...]) TAGS (type:shoe, color:red);
    > SEARCH QUERY ([0.1, 0.2...]) WHERE color='red' TOP 5;
    > GET MEMBRANCE INFO tests;

  Misc:
    - Type \q or EXIT; to quit.
)" << ANSI_RESET << "\n";
    }
};

int main(int argc, char **argv)
{
    std::string host = DEFAULT_HOST;
    int port = DEFAULT_PORT;

    // Parse args - Pomai style: pomai-cli -h host -p port
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "-h" && i + 1 < argc)
            host = argv[++i];
        else if (std::string(argv[i]) == "-p" && i + 1 < argc)
            port = std::stoi(argv[++i]);
    }

    PomaiCLI cli(host, port);
    cli.run();
    return 0;
}