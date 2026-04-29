#pragma once

/**
 * @file dispatcher_tbb.hh
 * @brief TBB-backed Dispatcher (`TbbDispatcher`).
 *
 * Implements the Dispatcher concept (see @ref dispatcher.hh) on top of
 * `tbb::parallel_for`. Header is gated on `ANTIPODAL_HAS_TBB`; including it in
 * a build that did not link TBB will fail at the `<tbb/parallel_for.h>`
 * include.
 */

#if defined(ANTIPODAL_HAS_TBB) && ANTIPODAL_HAS_TBB
#include <tbb/parallel_for.h>

namespace antipodal
{
struct TbbDispatcher
{
    template <class F>
    void parallel_for(int begin, int end, F&& f)
    {
        tbb::parallel_for(begin, end, [&](int i) { f(i); });
    }
};
} // namespace antipodal
#endif
