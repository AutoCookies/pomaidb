#pragma once
// Fanout helper utilities that work with SearchThreadPool.
// - ParallelSubmit takes a vector of no-arg callables and submits them to the pool,
//   returning a vector of futures in the same order.
// - Each callable must be movable/copyable and invocable with no args.
//
// Usage:
//   std::vector<std::function<Ret()>> jobs = { ... };
//   auto futs = ParallelSubmit(pool, std::move(jobs));
//   for (auto &f : futs) result = f.get();
#include <pomai/concurrency/search_thread_pool.h>

#include <vector>
#include <type_traits>
#include <functional>

namespace pomai
{

    // Submit a list (vector) of no-arg callables to the pool and get back futures.
    // F should be a callable type invocable with no args.
    template <typename F>
    auto ParallelSubmit(SearchThreadPool &pool, std::vector<F> jobs)
        -> std::vector<std::future<typename std::invoke_result_t<F>>>
    {
        using R = typename std::invoke_result_t<F>;
        std::vector<std::future<R>> futs;
        futs.reserve(jobs.size());
        for (auto &job : jobs)
        {
            // Submit each job by moving it into the pool
            futs.emplace_back(pool.Submit(std::move(job)));
        }
        return futs;
    }

} // namespace pomai
