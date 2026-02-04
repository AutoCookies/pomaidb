#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "pomai/status.h"

namespace pomai::util
{

    class PosixFile
    {
    public:
        PosixFile() = default;
        ~PosixFile();

        PosixFile(const PosixFile &) = delete;
        PosixFile &operator=(const PosixFile &) = delete;
        PosixFile(PosixFile &&other) noexcept;
        PosixFile &operator=(PosixFile &&other) noexcept;

        static pomai::Status OpenAppend(const std::string &path, PosixFile *out);
        static pomai::Status OpenRead(const std::string &path, PosixFile *out);
        static pomai::Status CreateTrunc(const std::string &path, PosixFile *out);

        pomai::Status PWrite(std::uint64_t off, const void *data, std::size_t n);
        pomai::Status ReadAt(std::uint64_t off, void *data, std::size_t n, std::size_t *out_read);

        pomai::Status Flush();
        pomai::Status SyncData();
        pomai::Status SyncAll();
        pomai::Status Close();
        
        // Maps the entire file into memory (Read Only). 
        // If successful, data/size are valid until PosixFile is closed/destroyed.
        pomai::Status Map(const void** out_data, std::size_t* out_size);

        int fd() const noexcept { return fd_; }

    private:
        explicit PosixFile(int fd) : fd_(fd) {}
        int fd_ = -1;
        void* map_addr_ = nullptr;
        std::size_t map_size_ = 0;
    };

    pomai::Status FsyncDir(const std::string &dir_path);

} // namespace pomai::util
