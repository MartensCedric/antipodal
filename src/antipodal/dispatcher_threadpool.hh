#pragma once

/**
 * @file dispatcher_threadpool.hh
 * @brief Self-contained thread-pool Dispatcher (`PoolDispatcher`).
 *
 * Implements the Dispatcher concept (see @ref dispatcher.hh) on top of a
 * reusable fork–join thread pool — handy when TBB is not available or when a
 * minimal, dependency-free parallel backend is desired.
 *
 * Design:
 * - Pool threads are created once at construction and reused across every
 *   `parallel_for` call.
 * - The calling thread always participates, so a pool configured with `T`
 *   total threads spawns `T-1` workers.
 * - Work distribution is dynamic: the index range is sliced into fixed-size
 *   blocks and threads claim the next block via a single `fetch_add` on a
 *   shared atomic counter (cache-line-padded to avoid false sharing).
 * - Fork/join uses two `std::counting_semaphore`s — no mutex or condition
 *   variable on the hot path.
 *
 * Limitations:
 * - No nested parallelism (a single `PoolDispatcher` runs one job at a time).
 * - Not exception-safe; the user functor must not throw.
 * - Assumes roughly uniform per-element cost; pathological imbalance still
 *   loses some scaling at the tail.
 */

#include <atomic>
#include <cstddef>
#include <semaphore>
#include <thread>
#include <vector>

namespace antipodal
{
struct PoolDispatcher
{
    /**
     * @brief Construct a pool with `thread_count` total threads (including the
     *        caller). Spawns `thread_count - 1` worker threads.
     *
     * @param thread_count Total number of threads. Values `<= 1` skip worker
     *                     creation; `parallel_for` then runs serially on the
     *                     calling thread.
     * @param block_size   Granularity of dynamic work distribution. Smaller
     *                     values balance load better at the cost of more
     *                     atomic traffic; larger values amortize overhead but
     *                     can leave threads idle near the tail.
     */
    explicit PoolDispatcher(int thread_count, int block_size = 256)
      : m_thread_count(thread_count), m_block_size(block_size)
    {
        if (m_thread_count > 1)
        {
            m_workers.reserve(m_thread_count - 1);
            for (int i = 0; i < m_thread_count - 1; ++i)
                m_workers.emplace_back([this] { worker_thread_loop(); });
        }
    }

    ~PoolDispatcher()
    {
        if (!m_workers.empty())
        {
            m_shutdown.store(true, std::memory_order_release);
            m_start_sem.release(static_cast<std::ptrdiff_t>(m_workers.size()));
            for (auto& w : m_workers)
                w.join();
        }
    }

    PoolDispatcher(PoolDispatcher const&) = delete;
    PoolDispatcher& operator=(PoolDispatcher const&) = delete;
    PoolDispatcher(PoolDispatcher&&) = delete;
    PoolDispatcher& operator=(PoolDispatcher&&) = delete;

    /// Parallel-for over `[begin, end)`. Calls `f(i)` exactly once per index.
    /// The order of invocations is unspecified and `f` may be called from
    /// multiple threads concurrently. Returns once every invocation completes.
    template <class F>
    void parallel_for(int begin, int end, F&& f)
    {
        if (end <= begin)
            return;

        int const n = end - begin;
        int const total_blocks = (n + m_block_size - 1) / m_block_size;

        // Fast path: serial execution on the caller (no pool threads available
        // or only a single block of work).
        if (m_thread_count <= 1 || total_blocks <= 1)
        {
            for (int i = begin; i < end; ++i)
                f(i);
            return;
        }

        // Non-capturing trampoline that erases `F` to a function pointer.
        // Reads the per-job descriptor through `self`; everything it touches
        // there is set before the start_sem release below, which synchronizes
        // with the worker's start_sem acquire.
        auto trampoline = +[](PoolDispatcher* self, void* user_fn)
        {
            auto& fn = *static_cast<F*>(user_fn);
            auto& next_idx = self->m_next_idx.value;
            int const block_size = self->m_job_block_size;
            int const job_n = self->m_job_n;
            int const job_begin = self->m_job_begin;
            while (true)
            {
                int const start = next_idx.fetch_add(block_size, std::memory_order_relaxed);
                if (start >= job_n)
                    return;
                int const stop = (start + block_size < job_n) ? (start + block_size) : job_n;
                for (int i = start; i < stop; ++i)
                    fn(job_begin + i);
            }
        };

        // Set up the job descriptor. No mutex needed: the start_sem release
        // below provides the happens-before edge that publishes these writes
        // to every worker that acquires the semaphore.
        m_next_idx.value.store(0, std::memory_order_relaxed);
        m_job_n = n;
        m_job_block_size = m_block_size;
        m_job_begin = begin;
        m_job_fn = trampoline;
        m_job_user_data = &f;

        // Main handles one block, so at most `total_blocks - 1` workers are
        // useful — but never more than the pool actually has.
        int const workers_available = static_cast<int>(m_workers.size());
        int const workers_used = (total_blocks - 1 < workers_available) ? (total_blocks - 1) : workers_available;

        // FORK: wake `workers_used` workers.
        m_start_sem.release(workers_used);

        // Main thread participates.
        trampoline(this, &f);

        // JOIN: wait for every woken worker to signal completion.
        for (int i = 0; i < workers_used; ++i)
            m_done_sem.acquire();
    }

private:
    void worker_thread_loop()
    {
        while (true)
        {
            m_start_sem.acquire();
            if (m_shutdown.load(std::memory_order_acquire))
                return;

            auto const fn = m_job_fn;
            void* const user_data = m_job_user_data;

            // Skip the call entirely if every block has already been claimed
            // by faster threads. Avoids the function-call overhead in the
            // (common) case where a small job finishes before we wake.
            if (m_next_idx.value.load(std::memory_order_relaxed) < m_job_n)
                fn(this, user_data);

            m_done_sem.release();
        }
    }

    static constexpr std::size_t cacheline_size = 64;

    struct alignas(cacheline_size) padded_atomic_int
    {
        std::atomic<int> value{0};
    };

    using job_fn_ptr = void (*)(PoolDispatcher*, void*);

    // Hot, contended counter — isolated from the rest of the object so
    // `fetch_add` traffic does not invalidate neighboring fields.
    padded_atomic_int m_next_idx;

    int m_thread_count = 0;
    int m_block_size = 0;

    // Per-job descriptor (read by all threads, written only by main between
    // jobs; published via the start_sem release).
    int m_job_n = 0;
    int m_job_block_size = 0;
    int m_job_begin = 0;
    job_fn_ptr m_job_fn = nullptr;
    void* m_job_user_data = nullptr;

    std::vector<std::thread> m_workers;
    std::counting_semaphore<> m_start_sem{0};
    std::counting_semaphore<> m_done_sem{0};
    std::atomic<bool> m_shutdown{false};
};
} // namespace antipodal
