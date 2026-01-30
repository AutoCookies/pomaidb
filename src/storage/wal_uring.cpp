#ifdef __linux__

#include <pomai/storage/wal_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <cstring>

namespace pomai
{
    namespace
    {
        // Rào cản bộ nhớ đảm bảo tính nhất quán dữ liệu giữa User-space và Kernel
        inline void memory_barrier()
        {
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }

        int IoUringSetup(unsigned entries, io_uring_params *params)
        {
            return static_cast<int>(::syscall(__NR_io_uring_setup, entries, params));
        }

        int IoUringEnter(int fd, unsigned to_submit, unsigned min_complete, unsigned flags)
        {
            return static_cast<int>(::syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, nullptr, 0));
        }
    }

    WalUring::~WalUring() { Shutdown(); }

    bool WalUring::Init(unsigned entries, bool sqpoll, std::string *error)
    {
        Shutdown();
        io_uring_params params{};
        if (sqpoll)
        {
            params.flags |= IORING_SETUP_SQPOLL;
            params.sq_thread_idle = 2000; // 2 giây idle
        }

        // Tối ưu hóa việc lập lịch tác vụ (Kernel 5.11+)
        params.flags |= IORING_SETUP_COOP_TASKRUN;

        int fd = IoUringSetup(entries, &params);
        if (fd < 0)
        {
            if (error)
                *error = "io_uring_setup failed: " + std::string(std::strerror(errno));
            return false;
        }

        // Tính toán kích thước các ring để mmap
        const size_t sq_ring_size = params.sq_off.array + params.sq_entries * sizeof(unsigned);
        const size_t cq_ring_size = params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);

        void *sq_ptr = ::mmap(nullptr, sq_ring_size, PROT_READ | PROT_WRITE,
                              MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
        if (sq_ptr == MAP_FAILED)
        {
            if (error)
                *error = "mmap SQ failed: " + std::string(std::strerror(errno));
            ::close(fd);
            return false;
        }

        void *cq_ptr = ::mmap(nullptr, cq_ring_size, PROT_READ | PROT_WRITE,
                              MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
        if (cq_ptr == MAP_FAILED)
        {
            if (error)
                *error = "mmap CQ failed: " + std::string(std::strerror(errno));
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
                *error = "mmap SQEs failed: " + std::string(std::strerror(errno));
            ::munmap(cq_ptr, cq_ring_size);
            ::munmap(sq_ptr, sq_ring_size);
            ::close(fd);
            return false;
        }

        // Lưu thông tin vào class members
        fd_ = fd;
        sq_ptr_ = sq_ptr;
        cq_ptr_ = cq_ptr;
        sqes_ptr_ = sqes_ptr;
        enabled_ = true;
        sq_entries_ = params.sq_entries;
        sqpoll_enabled_ = sqpoll;

        // Ánh xạ các con trỏ điều khiển
        sq_head_ = (unsigned *)((char *)sq_ptr_ + params.sq_off.head);
        sq_tail_ = (unsigned *)((char *)sq_ptr_ + params.sq_off.tail);
        sq_mask_ = (unsigned *)((char *)sq_ptr_ + params.sq_off.ring_mask);
        sq_entries_ptr_ = (unsigned *)((char *)sq_ptr_ + params.sq_off.ring_entries);
        sq_flags_ = (unsigned *)((char *)sq_ptr_ + params.sq_off.flags);
        sq_array_ = (unsigned *)((char *)sq_ptr_ + params.sq_off.array);

        cq_head_ = (unsigned *)((char *)cq_ptr_ + params.cq_off.head);
        cq_tail_ = (unsigned *)((char *)cq_ptr_ + params.cq_off.tail);
        cq_mask_ = (unsigned *)((char *)cq_ptr_ + params.cq_off.ring_mask);
        cq_entries_ptr_ = (unsigned *)((char *)cq_ptr_ + params.cq_off.ring_entries);
        cqes_ = (io_uring_cqe *)((char *)cq_ptr_ + params.cq_off.cqes);

        return true;
    }

    io_uring_sqe *WalUring::GetSqe()
    {
        if (!enabled_)
            return nullptr;

        unsigned head = std::atomic_load_explicit((std::atomic<unsigned> *)sq_head_, std::memory_order_acquire);
        unsigned tail = *sq_tail_;

        if (tail - head >= *sq_entries_ptr_)
            return nullptr;

        unsigned index = tail & *sq_mask_;
        io_uring_sqe *sqe = &((io_uring_sqe *)sqes_ptr_)[index];
        sq_array_[index] = index;

        // Xóa dữ liệu cũ trong SQE trước khi tái sử dụng
        std::memset(sqe, 0, sizeof(io_uring_sqe));
        *sq_tail_ = tail + 1;
        return sqe;
    }

    int WalUring::Submit(unsigned min_complete)
    {
        if (!enabled_)
            return -1;

        // Rào cản bộ nhớ trước khi Submit
        memory_barrier();

        unsigned to_submit = *sq_tail_ - *sq_head_;
        if (to_submit == 0 && min_complete == 0)
            return 0;

        unsigned flags = 0;
        if (min_complete > 0)
            flags |= IORING_ENTER_GETEVENTS;

        // Tối ưu SQPOLL: Chỉ gọi syscall nếu Kernel thread đang ngủ
        if (sqpoll_enabled_)
        {
            if (std::atomic_load_explicit((std::atomic<unsigned> *)sq_flags_, std::memory_order_acquire) & IORING_SQ_NEED_WAKEUP)
            {
                flags |= IORING_ENTER_SQ_WAKEUP;
            }
            else if (min_complete == 0)
            {
                return (int)to_submit;
            }
        }

        return IoUringEnter(fd_, to_submit, min_complete, flags);
    }

    int WalUring::WaitCqe(io_uring_cqe **cqe)
    {
        if (!enabled_)
            return -1;
        while (true)
        {
            unsigned head = std::atomic_load_explicit((std::atomic<unsigned> *)cq_head_, std::memory_order_acquire);
            if (head != *cq_tail_)
            {
                *cqe = &cqes_[head & *cq_mask_];
                return 0;
            }
            int ret = Submit(1);
            if (ret < 0 && errno == EINTR)
                continue;
            if (ret < 0)
                return -1;
        }
    }

    void WalUring::AdvanceCq(unsigned count)
    {
        if (enabled_)
            *cq_head_ += count;
    }

    void WalUring::PrepWritev(io_uring_sqe *sqe, int fd, const iovec *iov, unsigned iovcnt, off_t offset)
    {
        sqe->opcode = IORING_OP_WRITEV;
        sqe->fd = fd;
        sqe->addr = (uintptr_t)iov;
        sqe->len = iovcnt;
        sqe->off = offset;
    }

    void WalUring::PrepFdatasync(io_uring_sqe *sqe, int fd)
    {
        sqe->opcode = IORING_OP_FSYNC;
        sqe->fd = fd;
        sqe->fsync_flags = IORING_FSYNC_DATASYNC; // Chỉ đồng bộ dữ liệu
    }

    bool WalUring::IsEnabled() const
    {
        return enabled_;
    }

    unsigned WalUring::Entries() const
    {
        return sq_entries_;
    }

    // Đảm bảo hàm Shutdown cũng được cập nhật đầy đủ để giải phóng bộ nhớ
    void WalUring::Shutdown()
    {
        if (!enabled_)
            return;
        enabled_ = false;

        if (sqes_ptr_)
            ::munmap(sqes_ptr_, sq_entries_ * sizeof(io_uring_sqe));

        // Tính toán lại size để munmap SQ và CQ rings
        // (Đây là logic an toàn dựa trên params lúc Init)
        if (sq_ptr_)
            ::munmap(sq_ptr_, sq_entries_ * sizeof(unsigned) + 4096);
        if (cq_ptr_)
            ::munmap(cq_ptr_, sq_entries_ * 2 * sizeof(io_uring_cqe) + 4096);

        if (fd_ >= 0)
            ::close(fd_);
        fd_ = -1;
        sq_ptr_ = cq_ptr_ = sqes_ptr_ = nullptr;
    }
} // namespace pomai

#endif