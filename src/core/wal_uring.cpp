#ifdef __linux__

#include "wal_uring.h"

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace pomai
{
    namespace
    {
        int IoUringSetup(unsigned entries, io_uring_params *params)
        {
            return static_cast<int>(::syscall(__NR_io_uring_setup, entries, params));
        }

        int IoUringEnter(int fd, unsigned to_submit, unsigned min_complete, unsigned flags)
        {
            return static_cast<int>(::syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, nullptr, 0));
        }
    } // namespace

    WalUring::~WalUring()
    {
        Shutdown();
    }

    bool WalUring::Init(unsigned entries, bool sqpoll, std::string *error)
    {
        Shutdown();
        io_uring_params params{};
        if (sqpoll)
        {
            params.flags |= IORING_SETUP_SQPOLL;
            params.sq_thread_idle = 2000;
        }

        int fd = IoUringSetup(entries, &params);
        if (fd < 0)
        {
            if (error)
                *error = std::string("io_uring_setup failed: ") + std::strerror(errno);
            return false;
        }

        const size_t sq_ring_size = params.sq_off.array + params.sq_entries * sizeof(unsigned);
        const size_t cq_ring_size = params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);

        void *sq_ptr = ::mmap(nullptr, sq_ring_size, PROT_READ | PROT_WRITE,
                              MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
        if (sq_ptr == MAP_FAILED)
        {
            if (error)
                *error = std::string("mmap SQ ring failed: ") + std::strerror(errno);
            ::close(fd);
            return false;
        }

        void *cq_ptr = ::mmap(nullptr, cq_ring_size, PROT_READ | PROT_WRITE,
                              MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
        if (cq_ptr == MAP_FAILED)
        {
            if (error)
                *error = std::string("mmap CQ ring failed: ") + std::strerror(errno);
            ::munmap(sq_ptr, sq_ring_size);
            ::close(fd);
            return false;
        }

        void *sqes_ptr = ::mmap(nullptr, params.sq_entries * sizeof(io_uring_sqe),
                                PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                                fd, IORING_OFF_SQES);
        if (sqes_ptr == MAP_FAILED)
        {
            if (error)
                *error = std::string("mmap SQEs failed: ") + std::strerror(errno);
            ::munmap(cq_ptr, cq_ring_size);
            ::munmap(sq_ptr, sq_ring_size);
            ::close(fd);
            return false;
        }

        fd_ = fd;
        sq_ptr_ = sq_ptr;
        cq_ptr_ = cq_ptr;
        sqes_ptr_ = sqes_ptr;
        enabled_ = true;
        sq_entries_ = params.sq_entries;

        sq_head_ = reinterpret_cast<unsigned *>(static_cast<char *>(sq_ptr_) + params.sq_off.head);
        sq_tail_ = reinterpret_cast<unsigned *>(static_cast<char *>(sq_ptr_) + params.sq_off.tail);
        sq_mask_ = reinterpret_cast<unsigned *>(static_cast<char *>(sq_ptr_) + params.sq_off.ring_mask);
        sq_entries_ptr_ = reinterpret_cast<unsigned *>(static_cast<char *>(sq_ptr_) + params.sq_off.ring_entries);
        sq_flags_ = reinterpret_cast<unsigned *>(static_cast<char *>(sq_ptr_) + params.sq_off.flags);
        sq_array_ = reinterpret_cast<unsigned *>(static_cast<char *>(sq_ptr_) + params.sq_off.array);

        cq_head_ = reinterpret_cast<unsigned *>(static_cast<char *>(cq_ptr_) + params.cq_off.head);
        cq_tail_ = reinterpret_cast<unsigned *>(static_cast<char *>(cq_ptr_) + params.cq_off.tail);
        cq_mask_ = reinterpret_cast<unsigned *>(static_cast<char *>(cq_ptr_) + params.cq_off.ring_mask);
        cq_entries_ptr_ = reinterpret_cast<unsigned *>(static_cast<char *>(cq_ptr_) + params.cq_off.ring_entries);
        cqes_ = reinterpret_cast<io_uring_cqe *>(static_cast<char *>(cq_ptr_) + params.cq_off.cqes);

        return true;
    }

    void WalUring::Shutdown()
    {
        if (!enabled_)
            return;
        enabled_ = false;
        size_t sq_ring_size = 0;
        size_t cq_ring_size = 0;
        if (sq_entries_ptr_ && sq_array_ && sq_mask_)
        {
            sq_ring_size = reinterpret_cast<char *>(sq_array_) - reinterpret_cast<char *>(sq_ptr_);
            sq_ring_size += (*sq_entries_ptr_) * sizeof(unsigned);
        }
        if (cq_entries_ptr_)
        {
            cq_ring_size = reinterpret_cast<char *>(cqes_) - reinterpret_cast<char *>(cq_ptr_);
            cq_ring_size += (*cq_entries_ptr_) * sizeof(io_uring_cqe);
        }
        if (sqes_ptr_)
            ::munmap(sqes_ptr_, sq_entries_ * sizeof(io_uring_sqe));
        if (sq_ptr_ && sq_ring_size)
            ::munmap(sq_ptr_, sq_ring_size);
        if (cq_ptr_ && cq_ring_size)
            ::munmap(cq_ptr_, cq_ring_size);
        if (fd_ >= 0)
            ::close(fd_);

        fd_ = -1;
        sq_ptr_ = nullptr;
        cq_ptr_ = nullptr;
        sqes_ptr_ = nullptr;
        sq_head_ = nullptr;
        sq_tail_ = nullptr;
        sq_mask_ = nullptr;
        sq_entries_ptr_ = nullptr;
        sq_flags_ = nullptr;
        sq_array_ = nullptr;
        cq_head_ = nullptr;
        cq_tail_ = nullptr;
        cq_mask_ = nullptr;
        cq_entries_ptr_ = nullptr;
        cqes_ = nullptr;
        sq_entries_ = 0;
    }

    bool WalUring::IsEnabled() const
    {
        return enabled_;
    }

    unsigned WalUring::Entries() const
    {
        return sq_entries_;
    }

    unsigned WalUring::SqeAvailable() const
    {
        if (!enabled_)
            return 0;
        unsigned head = *sq_head_;
        unsigned tail = *sq_tail_;
        unsigned entries = *sq_entries_ptr_;
        return entries - (tail - head);
    }

    io_uring_sqe *WalUring::GetSqe()
    {
        if (!enabled_)
            return nullptr;
        const unsigned head = *sq_head_;
        unsigned tail = *sq_tail_;
        if (tail - head >= *sq_entries_ptr_)
            return nullptr;
        unsigned index = tail & *sq_mask_;
        io_uring_sqe *sqe = reinterpret_cast<io_uring_sqe *>(sqes_ptr_) + index;
        sq_array_[index] = index;
        *sq_tail_ = tail + 1;
        return sqe;
    }

    int WalUring::Submit(unsigned min_complete)
    {
        if (!enabled_)
            return -1;
        unsigned to_submit = *sq_tail_ - *sq_head_;
        unsigned flags = 0;
        if (min_complete > 0)
            flags |= IORING_ENTER_GETEVENTS;
        return IoUringEnter(fd_, to_submit, min_complete, flags);
    }

    int WalUring::WaitCqe(io_uring_cqe **cqe)
    {
        if (!enabled_)
            return -1;
        while (true)
        {
            unsigned head = *cq_head_;
            if (head != *cq_tail_)
            {
                *cqe = &cqes_[head & *cq_mask_];
                return 0;
            }
            int ret = IoUringEnter(fd_, 0, 1, IORING_ENTER_GETEVENTS);
            if (ret < 0 && errno == EINTR)
                continue;
            if (ret < 0)
                return -1;
        }
    }

    void WalUring::AdvanceCq(unsigned count)
    {
        if (!enabled_)
            return;
        *cq_head_ += count;
    }

    void WalUring::PrepWritev(io_uring_sqe *sqe, int fd, const iovec *iov, unsigned iovcnt, off_t offset)
    {
        std::memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = IORING_OP_WRITEV;
        sqe->fd = fd;
        sqe->addr = reinterpret_cast<__u64>(iov);
        sqe->len = iovcnt;
        sqe->off = offset;
    }

    void WalUring::PrepFdatasync(io_uring_sqe *sqe, int fd)
    {
        std::memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = IORING_OP_FDATASYNC;
        sqe->fd = fd;
    }
} // namespace pomai

#endif // __linux__
