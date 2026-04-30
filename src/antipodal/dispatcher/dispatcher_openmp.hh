#pragma once

/**
 * @file dispatcher_openmp.hh
 * @brief OpenMP-backed Dispatcher (`OpenMPDispatcher`).
 *
 * Implements the Dispatcher concept (see @ref dispatcher.hh) on top of an
 * OpenMP parallel-for. The type is only defined when the compiler-defined
 * `_OPENMP` macro is set; including this header without OpenMP enabled is
 * always safe — it simply does not introduce `OpenMPDispatcher`.
 */

#if defined(_OPENMP)
#include <omp.h>

namespace antipodal
{
struct OpenMPDispatcher
{
    template <class F>
    void parallel_for(int begin, int end, F&& f)
    {
#pragma omp parallel for schedule(static)
        for (int i = begin; i < end; ++i)
            f(i);
    }
};
} // namespace antipodal
#endif
