#pragma once

/**
 * @file dispatcher.hh
 * @brief Parallel-for abstraction used by all `*_batch` kernels.
 *
 * A @em Dispatcher is any type that satisfies the following duck-typed concept
 * (there is no base class to inherit from — pass any conforming type by ref):
 *
 * @code
 * struct MyDispatcher
 * {
 *     template <class F>
 *     void parallel_for(int begin, int end, F&& f);
 * };
 * @endcode
 *
 * Contract:
 * - `f` is invoked exactly once for every integer index `i` in `[begin, end)`.
 * - `f(int)` may be called concurrently from different threads. Implementations
 *   of `f` must therefore avoid data races on shared state.
 * - The order in which indices are visited is unspecified.
 * - `parallel_for` returns only after every invocation of `f` has completed.
 * - `parallel_for` is non-`const`: dispatchers may own a thread pool, work
 *   queues, or other mutable per-call state, so all kernels take the
 *   dispatcher by non-`const` reference.
 *
 * Implementations shipping with the library:
 * - `SinglethreadDispatcher` (this header) — always available; serial loop.
 * - `TbbDispatcher` (`dispatcher_tbb.hh`) — gated on `ANTIPODAL_HAS_TBB`;
 *   backed by `tbb::parallel_for`.
 * - `OpenMPDispatcher` (`dispatcher_openmp.hh`) — gated on `_OPENMP`; backed
 *   by `#pragma omp parallel for`.
 * - `PoolDispatcher` (`dispatcher_threadpool.hh`) — always available;
 *   self-contained fork–join thread pool.
 */

namespace antipodal
{
struct SinglethreadDispatcher
{
    template <class F>
    void parallel_for(int begin, int end, F&& f)
    {
        for (int i = begin; i < end; ++i)
            f(i);
    }
};
} // namespace antipodal
