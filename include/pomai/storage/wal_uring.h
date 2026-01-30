#pragma once

#ifdef __linux__

#include <linux/io_uring.h>
#include <sys/uio.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace pomai
{
    class WalUring
    {
    public:
        WalUring() = default;
        ~WalUring();

        WalUring(const WalUring &) = delete;
        WalUring &operator=(const WalUring &) = delete;

        bool Init(unsigned entries, bool sqpoll, std::string *error);
        void Shutdown();

        bool IsEnabled() const;
        unsigned Entries() const;

        io_uring_sqe *GetSqe();
        int Submit(unsigned min_complete);
        int WaitCqe(io_uring_cqe **cqe);
        void AdvanceCq(unsigned count);

        static void PrepWritev(io_uring_sqe *sqe, int fd, const iovec *iov, unsigned iovcnt, off_t offset);
        static void PrepFdatasync(io_uring_sqe *sqe, int fd);

    private:
        int fd_{-1};
        bool enabled_{false};
        unsigned sq_entries_{0};

        void *sq_ptr_{nullptr};
        void *cq_ptr_{nullptr};
        void *sqes_ptr_{nullptr};

        unsigned *sq_head_{nullptr};
        unsigned *sq_tail_{nullptr};
        unsigned *sq_mask_{nullptr};
        unsigned *sq_entries_ptr_{nullptr};
        unsigned *sq_flags_{nullptr};
        unsigned *sq_array_{nullptr};

        unsigned *cq_head_{nullptr};
        unsigned *cq_tail_{nullptr};
        unsigned *cq_mask_{nullptr};
        unsigned *cq_entries_ptr_{nullptr};
        io_uring_cqe *cqes_{nullptr};
    };
} // namespace pomai

#endif // __linux__
